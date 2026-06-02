# NeuroFlow Core

**Telegram user segmentation and AI orchestration toolkit.**

NeuroFlow Core is a Python library that brings state-machine-based user
segmentation and DAG-driven agent orchestration to Telegram bots and
AI pipelines. Instead of static tags that never expire, each user moves
through a lifecycle of states (lead → active → warm → hot → cold →
churned → banned) based on real behaviour. The built-in workflow
engine lets you decompose complex goals into dependency-aware task
graphs, execute them through registered agents, and synthesise results
with traceable memory.

It ships as three composable modules under the `neuroflow_core` package:

| Module | Role |
|--------|------|
| `telegram_segmentation` | State machine — tracks user lifecycle through message events |
| `telegram_ingestor` | Telegram poller + REST API + web dashboard |
| `osw_engine` | DAG-based Orchestrate-Synthesize-Workflow engine |

---

## Architecture

```
                    ┌──────────────────────────┐
                    │   Telegram Bot / CLI     │
                    └──────┬───────────────────┘
                           │ events (join, reply, purchase, …)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  telegram_ingestor                            │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Poller       │───▶│  Segmenter   │───▶│  UserStore      │  │
│  │  (getUpdates) │    │  (FSM)       │    │  (SQLite)       │  │
│  └──────────────┘    └──────────────┘    └──────┬─────────┘  │
│                                                  │            │
│  ┌───────────────────────────────────────────────┴────────┐   │
│  │  REST API :8888   |   Dashboard (HTML/JS)              │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────────┐
           │             osw_engine                    │
           │                                          │
           │  Goal → DAG decompose → topo sort →      │
           │  parallel batches → agent dispatch →      │
           │  aggregate → verify → report              │
           │                                          │
           │  ┌──────────┐  ┌────────────┐            │
           │  │ DAG      │  │ GraphMemory│ ← trust    │
           │  │ (acyclic)│  │ (SQLite)   │   + decay  │
           │  └──────────┘  └────────────┘            │
           └──────────────────────────────────────────┘
```

### telegram_segmentation

A deterministic finite-state machine. Every Telegram user starts as a
**lead** and transitions between states based on triggers (messages,
reactions, DMs, purchases, silence). Unlike tag-based systems where
labels are set once and forgotten, the state machine **enforces decay** —
inactive users automatically drift to COLD or CHURNED.

**States**

```
        ┌─── LEAD ───→ ACTIVE ───→ WARM ───→ HOT ────┐
        │     │           │          │          │      │
        │     └──→ COLD ←─┘          │          │      │
        │          │                 │          │      │
        └──────────┴──── CHURNED ←───┴──────────┘      │
                          SPAM → BANNED                │
                                                       ▼
                                                   CONVERTED
                                                   (by business logic)
```

**Triggers** — 16 event types mapped from Telegram messages:

`join`, `view`, `reaction`, `reply`, `link`, `question`, `dm`, `purchase`,
`contact`, `silent_7d`, `silent_30d`, `left`, `spam`, `manual_promote`,
`manual_demote`, `banned_user`

### telegram_ingestor

Connects directly to the Telegram Bot API via long-polling. It:

1. Polls `getUpdates` every N seconds
2. Classifies each message into a trigger (question, link, reaction, join, …)
3. Feeds the trigger into `TelegramSegmenter` to transition the user's state
4. Persists users and events in SQLite
5. Serves a **REST API** and a **web dashboard** on configurable port (default 8888)

**REST endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /api/segments` | Segment counts and total users |
| `GET /api/events?limit=N` | Last N state-change events |
| `GET /api/user/<id>` | Single user profile |
| `GET /dashboard` | HTML segment distribution dashboard |

### osw_engine

A lightweight orchestration engine for multi-step AI workflows.

- **DAG** — Directed acyclic graph with topological sort (Kahn's
  algorithm), cycle detection, and parallel layer grouping.
- **StateMachine** — Agent lifecycle manager (idle → working → waiting →
  done/failed) with guarded transitions.
- **GraphMemory** — SQLite-backed key-value store with trust scoring
  and time decay. Facts lose weight over time; stale facts fall below
  retrieval threshold and are pruned.
- **AgentCard** — Lightweight agent registration (name, role, model,
  tools, retry budget).
- **OSWEngine** — End-to-end orchestrator: ingest goal → decompose into
  task DAG → execute in topological order → report results.

---

## Quick Start

### Requirements

- Python ≥ 3.10
- `httpx` (installed automatically)

### Install

From source (PyPI publication coming soon):

```bash
git clone https://github.com/remontsuri/neuroflow-core.git
cd neuroflow-core
pip install -e .
```

### Environment Variables

For the ingestor module only:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token from @BotFather |
| `TELEGRAM_API_BASE` | No | `https://api.telegram.org` | Custom API base URL |
| `INGESTOR_PORT` | No | `8888` | HTTP server port |
| `COLD_DAYS` | No | `7` | Days of inactivity before cold |
| `CHURN_DAYS` | No | `30` | Days of inactivity before churn |

---

## Usage Examples

### 1. User Segmentation (standalone)

```python
from neuroflow_core import TelegramSegmenter

seg = TelegramSegmenter(cold_threshold_days=7, churn_threshold_days=30)

# Simulate events from Telegram
seg.process_message(user_id=42, msg_type="reaction")   # LEAD → ACTIVE
seg.process_message(user_id=42, msg_type="reply")      # ACTIVE → WARM
seg.process_message(user_id=42, msg_type="purchase")   # WARM → HOT

print(seg.get_segment(42))       # 'hot'
print(seg.hot_leads())           # sorted list of convertible users
print(seg.segment_counts())      # {'hot': 1, 'lead': 0, …}

# Apply inactivity decay
seg.run_decay()                  # silent users → COLD / CHURNED

# Export all data
seg.export_json("/tmp/segments.json")
```

### 2. Run the Telegram Ingestor

```python
import os
from neuroflow_core import TelegramIngestor, IngestorConfig

config = IngestorConfig(
    bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
    poll_interval_s=5,
    http_port=8888,
)

ingestor = TelegramIngestor(config)
ingestor.start()  # blocks — polls Telegram + serves REST API + dashboard
```

Then open `http://localhost:8888/dashboard` in your browser.

### 3. Orchestrate a Multi-Agent Workflow

```python
from neuroflow_core import OSWEngine, AgentCard, DAG

engine = OSWEngine()

engine.register_agent(AgentCard(name="researcher", role="worker"))
engine.register_agent(AgentCard(name="writer", role="worker"))

# Accept a high-level goal
engine.ingest_goal("Analyse Q2 churn and write a summary")

# Decompose into a task DAG
engine.decompose([
    {"id": "fetch_data",     "prompt": "Fetch Q2 user data",    "depends_on": []},
    {"id": "analyse_churn",  "prompt": "Analyse churn patterns", "depends_on": ["fetch_data"]},
    {"id": "write_summary",  "prompt": "Write executive summary", "depends_on": ["analyse_churn"]},
])

# Execute all tasks (topological order, parallel-ready layers)
results = engine.execute()

# Get structured report
report = engine.report()
print(report["metrics"])       # tasks_total, tasks_completed, elapsed_seconds
print(report["results"])       # task_id → status / output

# Memory persists across runs
print(engine.memory.recall("last_goal"))
```

### 4. Use the DAG Directly

```python
from neuroflow_core import DAG

dag = DAG()
dag.add_node("fetch", endpoint="/users")
dag.add_node("enrich", endpoint="/enrich")
dag.add_node("export", endpoint="/csv")

dag.add_edge("fetch", "enrich")   # fetch → enrich
dag.add_edge("enrich", "export")  # enrich → export

print(dag.topological_sort())     # ['fetch', 'enrich', 'export']
print(dag.levels())               # [['fetch'], ['enrich'], ['export']]
```

### 5. GraphMemory with Decay

```python
from neuroflow_core import GraphMemory

mem = GraphMemory("/tmp/neuroflow_memory.db")
mem.remember("user:42:preference", "likes video content", source="analyser")
mem.remember("user:42:last_seen", "2026-06-01", source="poller")

print(mem.recall("user:42:preference"))    # 'likes video content'
print(mem.search("video"))                 # list of matching Facts

mem.run_decay(halflife_days=7.0)           # decay all fact trust scores
print(mem.stats())                         # {total_facts, avg_trust}
```

---

## API Reference

### `telegram_segmentation`

#### `UserState` (enum)

Values: `LEAD`, `ACTIVE`, `WARM`, `HOT`, `COLD`, `CHURNED`, `BANNED`.

- `is_convertible()` — `True` for WARM and HOT
- `priority_score()` — numeric priority (HOT=100, WARM=70, …)

#### `Trigger` (enum)

16 event types: `JOINED`, `VIEWED`, `REACTED`, `REPLIED`, `CLICKED_LINK`,
`ASKED_QUESTION`, `DM_SENT`, `PURCHASED`, `GAVE_CONTACT`, `SILENT_7D`,
`SILENT_30D`, `LEFT`, `SPAM`, `MANUAL_PROMOTE`, `MANUAL_DEMOTE`,
`BANNED_USER`.

#### `TRIGGER_MAP` (dict)

Maps string aliases (`"join"`, `"reaction"`, `"dm"`, …) to `Trigger` values.

#### `classify_trigger(msg_type: str) -> Optional[Trigger]`

Convert a string to the matching `Trigger` enum, or `None` if unrecognised.

#### `UserProfile`

Dataclass holding `user_id`, `state`, `username`, `first_seen`,
`last_active`, `message_count`, `reactions_received`, `dm_count`,
`tags`, `history`. Methods:
- `to_segment() -> str` — current state value
- `to_dict() -> dict` — full serialisable snapshot

#### `TelegramSegmenter`

| Method | Description |
|--------|-------------|
| `__init__(cold_threshold_days=7, churn_threshold_days=30)` | Create segmenter |
| `process_message(user_id, msg_type, username='', metadata=None) -> Optional[UserState]` | Classify event → transition state |
| `get_segment(user_id) -> Optional[str]` | Current segment name |
| `get_user(user_id) -> Optional[UserProfile]` | Full user profile |
| `run_decay()` | Apply inactivity transitions |
| `segment_counts() -> dict[str, int]` | Count per segment |
| `hot_leads() -> list[UserProfile]` | Convertible users, sorted |
| `export() -> dict` | Full state snapshot |
| `export_json(path) -> str` | Write JSON export to disk |

### `telegram_ingestor`

#### `IngestorConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bot_token` | str | — | Telegram bot token |
| `api_base` | str | `https://api.telegram.org` | Bot API base URL |
| `poll_interval_s` | int | `5` | Seconds between polls |
| `http_port` | int | `8888` | REST API / dashboard port |
| `db_path` | str | `/tmp/telegram_ingestor.db` | SQLite path |
| `cold_days` | int | `7` | Days before COLD |
| `churn_days` | int | `30` | Days before CHURNED |
| `allowed_chat_ids` | list[int] or None | `None` | Filter by chat ID |

#### `config_from_env() -> IngestorConfig`

Read `TELEGRAM_BOT_TOKEN` from environment (or `.env` file) and return
a config with defaults for everything else.

#### `TelegramIngestor`

| Method | Description |
|--------|-------------|
| `__init__(config)` | Create ingestor with config |
| `poll_once() -> int` | Fetch one batch of updates, return count |
| `start()` | Begin polling and HTTP server (blocks) |
| `stop()` | Graceful shutdown |

### `osw_engine`

#### `DAG`

| Method | Description |
|--------|-------------|
| `add_node(node_id, **data)` | Add a node with arbitrary payload |
| `add_edge(from_node, to_node)` | Add dependency edge |
| `parents(node_id) -> list[str]` | Reverse dependencies |
| `children(node_id) -> list[str]` | Forward dependencies |
| `ancestors(node_id) -> set[str]` | All transitive predecessors |
| `topological_sort() -> list[str]` | Kahn's algorithm (raises `CycleError` if cyclic) |
| `levels() -> list[list[str]]` | Parallel-ready layers |

#### `GraphMemory`

| Method | Description |
|--------|-------------|
| `__init__(db_path)` | Open or create SQLite store |
| `remember(key, value, source='system')` | Write fact with trust=1.0 |
| `recall(key) -> Optional[str]` | Read by key (trust ≥ 0.3) |
| `search(query, min_trust=0.3, limit=10) -> list[Fact]` | Fuzzy search |
| `forget(key)` | Delete fact |
| `run_decay(halflife_days=7.0) -> int` | Decay all trust, return pruned count |
| `clear()` | Delete all facts |
| `stats() -> dict` | `total_facts` and `avg_trust` |

#### `StateMachine` (agent lifecycle)

States: `IDLE`, `WORKING`, `WAITING`, `DONE`, `FAILED`, `CANCELLED`.

| Method | Description |
|--------|-------------|
| `transition(target)` | Move to target state (guarded) |
| `reset()` | Return to IDLE |
| `is_terminal() -> bool` | `True` for DONE or FAILED |

#### `AgentCard`

Dataclass: `name`, `role` (default `"worker"`), `model`, `tools`,
`max_retries` (3), `timeout_s` (300), `state_machine` (`StateMachine`).

- `is_available() -> bool` — `state_machine.state == IDLE`

#### `OSWEngine`

| Method | Description |
|--------|-------------|
| `__init__(memory_path=None)` | Create engine |
| `register_agent(card)` | Register an AgentCard |
| `ingest_goal(goal) -> str` | Accept a high-level goal |
| `decompose(tasks)` | Build DAG from task list |
| `execute() -> dict` | Run DAG, return task results |
| `report() -> dict` | Structured run report |
| `clear()` | Reset engine state |

---

## Development

### Setup

```bash
git clone https://github.com/remontsuri/neuroflow-core.git
cd neuroflow-core
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Tests

```bash
pytest
```

### Lint

```bash
ruff check .
mypy neuroflow_core
```

---

## License

MIT © 2026 remontsuri. See [LICENSE](./LICENSE).
