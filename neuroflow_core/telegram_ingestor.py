"""Polls Telegram channels, classifies people into segments, serves a dashboard.

Background poller pulls messages from Telegram groups/channels, feeds them
into the state-machine segmenter, and logs everything to SQLite. Also runs
a REST API + a minimal HTML dashboard so you can see what's happening.
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

import httpx

from neuroflow_core.telegram_segmentation import (
    TelegramSegmenter,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class IngestorConfig:
    """What the ingestor needs to run — bot token, API base, port, thresholds."""

    bot_token: str
    api_base: str = "https://api.telegram.org"
    poll_interval_s: int = 5
    http_port: int = 8888
    db_path: str = "/tmp/telegram_ingestor.db"
    cold_days: int = 7
    churn_days: int = 30
    allowed_chat_ids: list[int] | None = None  # None = listen to all


def config_from_env() -> IngestorConfig:
    """Read config from environment variables.

    TELEGRAM_BOT_TOKEN is mandatory. Falls back to reading /opt/data/.env
    line-by-line if the env var isn't set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    if not token:
        # Fallback: try reading from .env file
        env_path = "/opt/data/.env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1]
                        break

    return IngestorConfig(bot_token=token)


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------


class UserStore:
    """SQLite store for user profiles and events. Thread-safe-ish (SQLite handles the locking)."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the tables if they don't exist yet."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'lead',
                    username TEXT DEFAULT '',
                    first_seen REAL NOT NULL,
                    last_active REAL NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    tags TEXT DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    state_from TEXT,
                    state_to TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_user
                    ON events(user_id);
                CREATE INDEX IF NOT EXISTS idx_events_type
                    ON events(event_type);
            """)

    def upsert_user(self, user_id: int, state: str, username: str = "",
                    message_count: int = 0, tags: list[str] | None = None) -> None:
        """Insert or update a user record."""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO users (user_id, state, username, first_seen, last_active, message_count, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    state=excluded.state,
                    username=excluded.username,
                    last_active=excluded.last_active,
                    message_count=excluded.message_count,
                    tags=excluded.tags
            """, (user_id, state, username, now, now, message_count,
                  json.dumps(tags or [])))

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Look up a user by ID. Returns None if not found."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def log_event(self, user_id: int, event_type: str,
                  state_from: str = "", state_to: str = "",
                  metadata: dict[str, Any] | None = None) -> None:
        """Record an event in the event log."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO events (user_id, event_type, state_from, state_to, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, event_type, state_from, state_to,
                  json.dumps(metadata or {}), time.time()))

    def segment_counts(self) -> dict[str, int]:
        """How many users are in each segment right now."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) as cnt FROM users GROUP BY state"
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent events, newest first."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def total_users(self) -> int:
        """Total user count across all segments."""
        with sqlite3.connect(self._db_path) as conn:
            result: int | None = conn.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0]
            return result or 0


# ---------------------------------------------------------------------------
# Dashboard HTTP handler — explicit class, not closure
# ---------------------------------------------------------------------------

_SEGMENT_COLORS: dict[str, str] = {
    "lead": "#6b7280", "active": "#3b82f6", "warm": "#f59e0b",
    "hot": "#ef4444", "cold": "#8b5cf6", "churned": "#111827",
    "banned": "#000000",
}
_SEGMENT_ORDER = ("lead", "active", "warm", "hot", "cold", "churned", "banned")


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the NeuroFlow REST API and HTML dashboard.

    Bound to a ``store`` via class variable so it can serve data without
    depending on a closure.
    """

    store: UserStore | None = None

    def log_message(self, format: str, *args: Any) -> None:
        pass  # quiet

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, ensure_ascii=False).encode())

    def _html_response(self, html_str: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html_str.encode())

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)
        store = self.store
        if store is None:
            self._json_response({"error": "store not available"}, 500)
            return

        if path in ("", "/dashboard"):
            self._serve_dashboard(store)
        elif path == "/api/segments":
            self._json_response({
                "total_users": store.total_users(),
                "segments": store.segment_counts(),
            })
        elif path == "/api/events":
            limit = int(params.get("limit", [50])[0])
            self._json_response({"events": store.recent_events(limit)})
        elif path.startswith("/api/user/"):
            try:
                uid = int(path.split("/")[-1])
                user = store.get_user(uid)
                if user:
                    self._json_response({"user": user})
                else:
                    self._json_response({"error": "not found"}, 404)
            except (ValueError, IndexError):
                self._json_response({"error": "bad user_id"}, 400)
        else:
            self._json_response({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    # Dashboard HTML
    # ------------------------------------------------------------------

    def _serve_dashboard(self, store: UserStore) -> None:
        segments = store.segment_counts()
        total = store.total_users()
        events = store.recent_events(20)

        rows = "".join(
            f"<tr><td>{e.get('user_id', '?')}</td>"
            f"<td>{html.escape(e.get('event_type', ''))}</td>"
            f"<td>{html.escape(e.get('state_from', ''))}</td>"
            f"<td>{html.escape(e.get('state_to', ''))}</td></tr>"
            for e in events
        )

        seg_bars = ""
        for seg_name in _SEGMENT_ORDER:
            count = segments.get(seg_name, 0)
            pct = (count / total * 100) if total > 0 else 0
            seg_bars += (
                f"<div style='margin:4px 0;display:flex;align-items:center;gap:8px'>"
                f"<span style='width:70px;text-transform:capitalize'>{seg_name}</span>"
                f"<div style='flex:1;height:20px;background:#e5e7eb;border-radius:4px;overflow:hidden'>"
                f"<div style='width:{pct:.0f}%;height:100%;background:{_SEGMENT_COLORS.get(seg_name, '#999')};"
                f"border-radius:4px;transition:width .5s'></div></div>"
                f"<span style='width:50px;text-align:right'>{count}</span></div>"
            )

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroFlow — Segment Dashboard</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f9fafb; color: #111; }}
.container {{ max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
.subtitle {{ color: #6b7280; margin-bottom: 24px; }}
.card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1); padding: 20px; margin-bottom: 20px; }}
.card h2 {{ margin: 0 0 12px; font-size: 18px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
th {{ color: #6b7280; font-weight: 600; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; background: #e5e7eb; }}
.endpoint {{ font-family: monospace; font-size: 13px; background: #f3f4f6; padding: 6px 10px; border-radius: 4px; display: inline-block; }}
</style>
</head>
<body>
<div class="container">
<h1>Segment Dashboard</h1>
<p class="subtitle">Total users: <strong>{total}</strong></p>

<div class="card">
<h2>Segment distribution</h2>
{seg_bars}
</div>

<div class="card">
<h2>Recent events</h2>
<table><thead><tr><th>User</th><th>Event</th><th>From</th><th>To</th></tr></thead>
<tbody>{rows}</tbody></table>
</div>

<div class="card">
<h2>API</h2>
<div class="endpoint">GET /api/segments</div> — segment counts<br>
<div class="endpoint">GET /api/events?limit=N</div> — recent events<br>
<div class="endpoint">GET /api/user/&lt;id&gt;</div> — single user<br>
</div>
</div>
</body>
</html>"""
        self._html_response(html_page)


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class TelegramIngestor:
    """The main thing — polls Telegram, segments users, serves a REST API.

        config = IngestorConfig(bot_token="123:abc")
        ingestor = TelegramIngestor(config)
        ingestor.start()  # blocks forever
    """

    def __init__(self, config: IngestorConfig):
        """Wire up the ingestor: segmenter, user store, HTTP client."""
        self.config = config
        self.segmenter = TelegramSegmenter(
            cold_threshold_days=config.cold_days,
            churn_threshold_days=config.churn_days,
        )
        self.store = UserStore(config.db_path)
        self._http_client: httpx.Client | None = None
        self._running = False
        self._update_id = 0
        self._lock = threading.Lock()

    @property
    def client(self) -> httpx.Client:
        """Lazy HTTP client — one per lifetime."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self.config.api_base,
                timeout=httpx.Timeout(15.0),
            )
        return self._http_client

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll_once(self) -> int:
        """Fetch one batch of updates from Telegram. Returns number processed."""
        try:
            resp = self.client.get("/bot{}/getUpdates".format(self.config.bot_token), params={
                "offset": self._update_id + 1,
                "timeout": 10,
                "allowed_updates": json.dumps(["message", "chat_member", "callback_query"]),
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[ingestor] poll error: {exc}")
            return 0

        if not data.get("ok"):
            print(f"[ingestor] API error: {data.get('description', 'unknown')}")
            return 0

        updates = data.get("result", [])
        for upd in updates:
            self._update_id = upd["update_id"]
            self._process_update(upd)

        return len(updates)

    def _process_update(self, update: dict[str, Any]) -> None:
        """Route one update through segmentation and logging."""
        msg = update.get("message") or update.get("callback_query", {}).get("message")
        if not msg:
            return

        chat_id = msg["chat"]["id"]
        if self.config.allowed_chat_ids and chat_id not in self.config.allowed_chat_ids:
            return

        # For callback_query, 'from' is at the callback_query level, not in msg
        cb = update.get("callback_query")
        if cb and "from" in cb:
            user_from = cb["from"]
        else:
            user_from = msg.get("from", {})

        user_id = user_from["id"]
        username = user_from.get("username", "") or user_from.get("first_name", "")

        # Classify the message type
        msg_type = self._classify_message(msg)
        if not msg_type:
            return

        old_state = self.segmenter.get_segment(user_id)
        new_state = self.segmenter.process_message(user_id, msg_type, username=username)

        if new_state:
            self.store.upsert_user(
                user_id=user_id,
                state=new_state.value,
                username=username,
            )
            self.store.log_event(
                user_id=user_id,
                event_type=msg_type,
                state_from=old_state or "unknown",
                state_to=new_state.value,
                metadata={"chat_id": chat_id},
            )

    def _classify_message(self, msg: dict[str, Any]) -> str | None:
        """Guess the event type from a Telegram message object."""
        if "new_chat_members" in msg:
            return "joined"
        if "left_chat_member" in msg:
            return "left"
        if "text" in msg:
            text = msg["text"].lower()
            if "?" in text or text.startswith(("what", "how", "why", "when", "where", "who", "can", "do", "is")):
                return "question"
            if any(domain in text for domain in
                   (".com", ".ru", ".org", ".net", "t.me", "https://", "http://")):
                return "link"
            return "message"
        return None

    # ------------------------------------------------------------------
    # HTTP server for REST API + dashboard
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start polling and the HTTP server. Blocks forever."""
        self._running = True

        # Wire the store into the handler class so it doesn't need a closure
        DashboardHandler.store = self.store

        server = HTTPServer(("0.0.0.0", self.config.http_port), DashboardHandler)
        poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        poll_thread.start()

        print(f"[ingestor] HTTP server on :{self.config.http_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._running = False
            server.shutdown()

    def _poll_loop(self) -> None:
        """Poll Telegram in a loop until told to stop."""
        while self._running:
            count = self.poll_once()
            if count:
                print(f"[ingestor] processed {count} updates")
            time.sleep(self.config.poll_interval_s)

    def stop(self) -> None:
        """Shut down the polling loop gracefully."""
        self._running = False
