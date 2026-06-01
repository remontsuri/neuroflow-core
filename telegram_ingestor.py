"""
neuroflow-core :: Telegram Webhook Ingestor
Receives Telegram updates via long polling, runs through segmentation.
Exposes REST API for querying segments and user states.
Mythos build — production-grade, single-file, no framework lock-in.
"""

import json
import os
import sqlite3
import time
import hashlib
import threading
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ─── Config ───────────────────────────────────────────────────────────────────

def _read_bot_token():
    """Read Telegram bot token from .env or config."""
    try:
        with open("/opt/data/.env") as f:
            for line in f:
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip("'\"").strip()
                    if tok:
                        return tok
    except: pass
    try:
        with open(os.path.expanduser("~/.env")) as f:
            for line in f:
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip("'\"").strip()
                    if tok:
                        return tok
    except: pass
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")

BOT_TOKEN = _read_bot_token()
SEGMENTATION_PORT = int(os.environ.get("SEGMENTATION_PORT", "9122"))
DB_PATH = os.environ.get("SEGMENTATION_DB", "/opt/data/segmentation.db")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "2"))
# ─── State Machine ────────────────────────────────────────────────────────────

class UserState:
    LEAD = "lead"
    ACTIVE = "active"
    WARM = "warm"
    HOT = "hot"
    COLD = "cold"
    CHURNED = "churned"
    BANNED = "banned"

TRANSITIONS = {
    UserState.LEAD: {
        "message": UserState.ACTIVE,
        "reaction": UserState.ACTIVE,
        "click": UserState.WARM,
        "dm": UserState.HOT,
        "spam": UserState.BANNED,
    },
    UserState.ACTIVE: {
        "message": UserState.ACTIVE,
        "click": UserState.WARM,
        "question": UserState.WARM,
        "dm": UserState.HOT,
        "silent_7d": UserState.COLD,
        "spam": UserState.BANNED,
    },
    UserState.WARM: {
        "message": UserState.WARM,
        "click": UserState.WARM,
        "purchase": UserState.HOT,
        "contact": UserState.HOT,
        "dm": UserState.HOT,
        "silent_7d": UserState.COLD,
        "spam": UserState.BANNED,
    },
    UserState.HOT: {
        "message": UserState.HOT,
        "purchase": UserState.HOT,
        "silent_14d": UserState.COLD,
        "block": UserState.CHURNED,
        "spam": UserState.BANNED,
    },
    UserState.COLD: {
        "message": UserState.ACTIVE,
        "click": UserState.WARM,
        "dm": UserState.HOT,
        "silent_30d": UserState.CHURNED,
        "spam": UserState.BANNED,
    },
    UserState.CHURNED: {
        "message": UserState.ACTIVE,  # re-engagement
        "spam": UserState.BANNED,
    },
    UserState.BANNED: {},  # terminal
}

SEGMENT_LABELS = {
    UserState.LEAD: "Новый / лид",
    UserState.ACTIVE: "Активный",
    UserState.WARM: "Тёплый",
    UserState.HOT: "Горячий",
    UserState.COLD: "Холодный",
    UserState.CHURNED: "Ушедший",
    UserState.BANNED: "Забанен",
}

# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            state TEXT NOT NULL DEFAULT 'lead',
            segment TEXT NOT NULL DEFAULT 'Новый / лид',
            message_count INTEGER DEFAULT 1,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_trigger TEXT,
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event_type TEXT,
            trigger TEXT,
            content_preview TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS webhook_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT,
            path TEXT,
            body_preview TEXT,
            response_code INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def transition_user(user_id: int, trigger: str,
                    username: str = "", first_name: str = "",
                    content: str = "", force_state: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT state, message_count FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()

    if row:
        current_state, msg_count = row
    else:
        current_state = UserState.LEAD
        msg_count = 0

    new_state = force_state or TRANSITIONS.get(current_state, {}).get(trigger, current_state)
    msg_count += 1

    conn.execute("""
        INSERT OR REPLACE INTO users
        (user_id, username, first_name, state, segment, message_count, last_seen, last_trigger, metadata)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, '{}')
    """, (user_id, username[:32] if username else "",
          (first_name or "")[:32],
          new_state, SEGMENT_LABELS.get(new_state, new_state),
          msg_count, trigger))

    conn.execute("""
        INSERT INTO events (user_id, event_type, trigger, content_preview)
        VALUES (?, 'telegram_message', ?, ?)
    """, (user_id, trigger, content[:200] if content else ""))

    conn.commit()
    conn.close()
    return new_state, SEGMENT_LABELS.get(new_state, new_state)

def get_user(user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_users(limit: int = 100) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM users ORDER BY last_seen DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""
        SELECT state, COUNT(*) as count FROM users GROUP BY state
    """)
    by_state = {row[0]: row[1] for row in cur.fetchall()}
    cur = conn.execute("SELECT COUNT(*) FROM users")
    total = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM events WHERE timestamp > datetime('now', '-24 hours')")
    events_24h = cur.fetchone()[0]
    conn.close()
    return {"total_users": total, "by_state": by_state, "events_last_24h": events_24h}

# ─── Telegram Poller ──────────────────────────────────────────────────────────

last_update_id = 0

def poll_telegram():
    """Long-poll Telegram for new messages and push through segmentation."""
    global last_update_id
    api_base = f"https://api.telegram.org/bot{BOT_TOKEN}"

    while True:
        try:
            url = f"{api_base}/getUpdates?offset={last_update_id + 1}&timeout=30"
            req = Request(url)
            resp = urlopen(req, timeout=35)
            data = json.loads(resp.read())

            for update in data.get("result", []):
                update_id = update.get("update_id", 0)
                if update_id <= last_update_id:
                    continue
                last_update_id = update_id

                msg = update.get("message") or update.get("edited_message") or {}
                user = msg.get("from", {})
                user_id = user.get("id")
                if not user_id:
                    continue

                text = msg.get("text", "") or msg.get("caption", "")
                username = user.get("username", "")
                first_name = user.get("first_name", "")

                # Determine trigger
                if msg.get("entities"):
                    for ent in msg.get("entities", []):
                        if ent.get("type") == "bot_command":
                            trigger = "command"
                            break
                    else:
                        trigger = "message"
                else:
                    trigger = "message"

                # Special triggers
                if text:
                    text_lower = text.lower()
                    if any(w in text_lower for w in ["купить", "цена", "стоимость", "хочу"]):
                        trigger = "purchase"
                    elif "?" in text:
                        trigger = "question"
                    elif any(w in text_lower for w in ["спам", "реклама"]):
                        trigger = "spam"

                # Process through segmentation
                new_state, segment = transition_user(
                    user_id, trigger, username, first_name, text
                )

                # Log to stdout
                print(f"[SEGMENT] user={user_id} ({first_name}) trigger={trigger} → {segment}")

        except HTTPError as e:
            if e.code == 409:  # Conflict - another poller
                print(f"[POLL] Conflict (another bot instance), reconnecting in 5s...")
                time.sleep(5)
                continue
            print(f"[POLL] HTTP {e.code}: {e.read().decode()[:200]}")
            time.sleep(5)
        except (URLError, ConnectionError) as e:
            print(f"[POLL] Network error: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"[POLL] Error: {e}")
            time.sleep(5)

# ─── REST API ─────────────────────────────────────────────────────────────────

class SegmentationAPI(BaseHTTPRequestHandler):
    def _json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode())

    def _html(self, content, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self._html(self._render_dashboard())
        elif self.path == "/api/stats":
            self._json(get_stats())
        elif self.path.startswith("/api/users/"):
            uid = int(self.path.split("/")[-1])
            user = get_user(uid)
            if user:
                self._json(user)
            else:
                self._json({"error": "not found"}, 404)
        elif self.path == "/api/users":
            self._json(get_all_users())
        elif self.path == "/health":
            self._json({"status": "ok", "uptime": time.time() - start_time})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/transition":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            uid = body.get("user_id")
            trigger = body.get("trigger", "message")
            force = body.get("force_state")
            if not uid:
                self._json({"error": "user_id required"}, 400)
                return
            state, seg = transition_user(uid, trigger, force_state=force)
            self._json({"user_id": uid, "state": state, "segment": seg})
        else:
            self._json({"error": "not found"}, 404)

    def _render_dashboard(self):
        users = get_all_users(50)
        stats = get_stats()
        rows = ""
        for u in users:
            rows += f"""
            <tr>
                <td>{u['user_id']}</td>
                <td>{u.get('first_name','')}</td>
                <td><span class="state-{u.get('state','unknown')}">{u.get('segment','')}</span></td>
                <td>{u.get('message_count',0)}</td>
                <td>{u.get('last_trigger','')}</td>
                <td>{u.get('last_seen','')[:19]}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NeuroFlow Segmentation</title>
<style>
    body {{ font-family: system-ui, sans-serif; margin: 40px; background: #0d1117; color: #e6edf3; }}
    h1 {{ color: #58a6ff; }}
    .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
    .stat {{ background: #161b22; padding: 15px 25px; border-radius: 8px; border: 1px solid #30363d; }}
    .stat h3 {{ margin: 0; color: #8b949e; font-size: 12px; text-transform: uppercase; }}
    .stat .value {{ font-size: 28px; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ text-align: left; padding: 8px; border-bottom: 2px solid #30363d; color: #8b949e; }}
    td {{ padding: 8px; border-bottom: 1px solid #21262d; }}
    .state-lead {{ color: #58a6ff; }}
    .state-active {{ color: #3fb950; }}
    .state-warm {{ color: #d29922; }}
    .state-hot {{ color: #f85149; }}
    .state-cold {{ color: #8b949e; }}
    .state-churned {{ color: #6e7681; }}
    .state-banned {{ color: #f85149; text-decoration: line-through; }}
</style></head><body>
<h1>🔮 NeuroFlow Segmentation</h1>
<div class="stats">
    <div class="stat"><h3>Всего пользователей</h3><div class="value">{stats['total_users']}</div></div>
    {''.join(f'<div class="stat"><h3>{k}</h3><div class="value">{v}</div></div>' for k,v in stats.get('by_state',{}).items())}
    <div class="stat"><h3>Событий 24ч</h3><div class="value">{stats['events_last_24h']}</div></div>
</div>
<table><thead><tr><th>ID</th><th>Имя</th><th>Сегмент</th><th>Сообщений</th><th>Триггер</th><th>Последний раз</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""

# ─── Main ─────────────────────────────────────────────────────────────────────

start_time = time.time()

def run():
    init_db()
    print(f"[SEGMENT] DB initialized at {DB_PATH}")
    print(f"[SEGMENT] Bot token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:] if BOT_TOKEN else 'EMPTY'}")

    # Start Telegram poller in background thread
    if BOT_TOKEN:
        poller = threading.Thread(target=poll_telegram, daemon=True)
        poller.start()
        print(f"[SEGMENT] Telegram poller started (interval: {POLL_INTERVAL}s)")
    else:
        print("[SEGMENT] No TELEGRAM_BOT_TOKEN — polling disabled. API-only mode.")

    # Start REST API server
    server = HTTPServer(("0.0.0.0", SEGMENTATION_PORT), SegmentationAPI)
    print(f"[SEGMENT] API server on http://0.0.0.0:{SEGMENTATION_PORT}")
    print(f"[SEGMENT] Dashboard: http://localhost:{SEGMENTATION_PORT}/dashboard")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SEGMENT] Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    run()
