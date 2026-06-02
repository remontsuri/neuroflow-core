# Implementation Plan — neuroflow-core v0.1.0-beta

**Priority:** P1 (required for beta) → P2 (important) → P3 (nice to have)

---

## P1 — Critical (blocking v0.1.0-beta)

### 1. Implement OSWEngine.execute() — real agent dispatch

**File:** `neuroflow_core/osw_engine.py`

**Current:** `execute()` returns `f"Simulated execution: {prompt[:60]}"` — stub.

**Target:** Real dispatch that calls a Hermes profile agent per task.

**Options:**

| Option | Complexity | Pros | Cons |
|--------|-----------|------|------|
| **A. Subprocess** | Low | Simple, no deps | Blocking, no streaming |
| **B. Hermes CLI** | Low | Uses existing infra | Depends on `hermes` in PATH |
| **C. HTTP to gateway** | Medium | Non-blocking, parallel | Needs gateway endpoint |
| **D. callback abstraction** | Medium + | Pluggable, testable | Most code |

**Recommended: D** — add a `dispatch_fn` callback to `OSWEngine.__init__()`:

```python
class OSWEngine:
    def __init__(self, memory_path: str | None = None,
                 dispatch_fn: Callable[[str], str] | None = None):
        self._dispatch = dispatch_fn or self._default_dispatch

    def _default_dispatch(self, prompt: str) -> str:
        """Fallback: Hermes CLI subprocess."""
        import subprocess
        result = subprocess.run(
            ["hermes", "-p", "eni-worker", "-z", prompt],
            capture_output=True, text=True, timeout=300
        )
        return result.stdout
```

Then `execute()` calls `self._dispatch(node_data["prompt"])` per task.

**Tests:** Update `test_execute` to pass a mock `dispatch_fn` and verify it's called with correct prompts.

### 2. Fix callback_query dead branch in telegram_ingestor

**File:** `neuroflow_core/telegram_ingestor.py`

**Problem:** `_classify_message` checks `"callback_query" in msg` but `msg` is always the inner `message` object, never the update wrapper.

**Fix:** Move callback_query handling to `_process_update`:

```python
def _process_update(self, update: dict) -> int | None:
    if "callback_query" in update:
        user_id = update["callback_query"]["from"]["id"]
        # classify callback query data
        data = update["callback_query"].get("data", "")
        # ... update user state based on data
        return 1
    # existing message handling...
```

**Tests:** Add test with a callback_query fixture.

---

## P2 — Important (v0.1.0 release)

### 3. CLI entry points

**File:** `pyproject.toml` + new `neuroflow_core/__main__.py`

```toml
[project.scripts]
neuroflow-ingest = "neuroflow_core.telegram_ingestor:main"
```

Add `main()` to `telegram_ingestor.py`:

```python
def main():
    """Entry point: start the ingestor from CLI."""
    config = config_from_env()
    ingestor = TelegramIngestor(config)
    ingestor.start()
```

### 4. CI pipeline

**File:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy neuroflow_core
      - run: pytest
```

### 5. Fix hardcoded `/tmp/` paths

**File:** `neuroflow_core/telegram_ingestor.py`, `neuroflow_core/osw_engine.py`

- `GraphMemory.__init__` — accept `db_path` param (already done), document default
- `TelegramSegmenter.export_json()` — keep `path` param, just add example with meaningful path in docstring
- `TelegramIngestor` — let `IngestorConfig.db_path` be configurable (already done)

---

## P3 — Nice to have

### 6. Rename StateMachine → AgentStateMachine

**File:** `neuroflow_core/osw_engine.py`

Avoids naming collision with segmentation state machine. Pure rename, no logic change.

### 7. Replace handler closure with class

**File:** `neuroflow_core/telegram_ingestor.py`

Replace `_make_handler()` with explicit `DashboardHandler` class.

### 8. Pre-commit config

**File:** `.pre-commit-config.yaml`

Standard: ruff, mypy, trailing-whitespace, end-of-file-fixer.

---

## Effort estimate

| Task | Hours | Dependencies |
|------|-------|-------------|
| P1.1 Implement dispatch_fn | 1-2 | None |
| P1.2 Fix callback_query | 0.5 | None |
| P2.3 CLI entry point | 0.5 | None |
| P2.4 CI pipeline | 0.5 | None |
| P2.5 Fix /tmp/ paths | 0.25 | None |
| P3.6 Rename StateMachine | 0.25 | None |
| P3.7 Handler closure → class | 0.5 | None |
| P3.8 Pre-commit config | 0.25 | None |
| **Total** | **3.75** | — |

---

## Delivery order

```
Week 1          │  P1.1 ────────┐
                │  P1.2 ────────┤
                │               ├──→ v0.1.0-beta
Week 2          │  P2.3 ────────┤
                │  P2.4 ────────┤
                │  P2.5 ────────┘
                
Whenever         │  P3.6, P3.7, P3.8  ──→ v0.2.0
```
