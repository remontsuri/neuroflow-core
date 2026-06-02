# Codebase Inspection — neuroflow-core

**Date:** 2026-06-02  
**Scope:** All 3 modules in `neuroflow_core/` + tests + project config  
**Method:** Manual audit

---

## Verdict: Beta quality, core logic solid

| Metric | Result |
|--------|--------|
| Tests | 211/211 ✅ |
| mypy strict | Clean ✅ |
| Python packaging | Working ✅ |
| Thread safety | Proper (Lock per module) ✅ |
| Docstrings | Comprehensive ✅ |
| README | Full API reference ✅ |

---

## Findings

### 1. CRITICAL — `OSWEngine.execute()` is a stub

`osw_engine.py` line 388:
```python
result = f"Simulated execution: {node_data.get('prompt', '')[:60]}"
```

The orchestrator **never actually dispatches tasks** — it stores a placeholder string. The entire `execute()` → `report()` pipeline returns fake data. This is the central promise of the engine and it doesn't work.

**Fix:** Real dispatch via Hermes profile CLI, subprocess, or delegate_task.

### 2. MEDIUM — `telegram_ingestor._classify_message` callback_query dead branch

Lines 297-298:
```python
if "callback_query" in msg:
    return "reaction"
```

`msg` is extracted on line 243 as `update.get("message") or update.get("callback_query", {}).get("message")`. If the update is a callback_query without a message, `msg` is `None` and the function is guarded. If `msg` came from `callback_query.message`, then the `callback_query` key **is not in msg** — it's the inner message object. This branch is dead code.

**Fix:** Handle callback_query at the `_process_update` level, not inside message classification.

### 3. LOW — `StateMachine` naming collision

Two different `StateMachine` concepts:
- `osw_engine.StateMachine` — agent lifecycle (IDLE → WORKING → DONE)
- `telegram_segmentation.TRANSITIONS` + `run_decay()` — user segmentation FSM

One is explicit class, the other is a transition table + `process_message()`. The duplicated name pattern could confuse new readers. Not a bug, but worth renaming the engine's to `AgentStateMachine` or `LifecycleMachine`.

### 4. LOW — Hardcoded `/tmp/` paths

- `GraphMemory.__init__` defaults to `/tmp/graph_memory.db`
- `TelegramSegmenter.export_json` defaults to `/tmp/telegram_segments.json`

In Docker or multi-tenant setups these collide. Should accept env overrides or use `$XDG_DATA_HOME` / `~/.local/share/`.

### 5. LOW — No CLI entry point

`pyproject.toml` has no `[project.scripts]` section. The user must `python -c "from neuroflow_core import TelegramIngestor; ..."` to start the ingestor. Should have:
```toml
[project.scripts]
neuroflow-ingest = "neuroflow_core.telegram_ingestor:main"
```

### 6. LOW — HTTP handler closure pattern

`telegram_ingestor._make_handler()` returns a dynamically created class with `ingestor` in closure scope. Works but makes testing harder and confuses static analysis. Better: pass the reference explicitly via `__init__` override.

### 7. INFO — No pre-commit / CI config

Mypy and ruff are dev dependencies but there's no `.pre-commit-config.yaml` or `.github/workflows/ci.yml`. Easy to skip checks before push.

---

## Summary

```
┌─────────────────────────────────────────────────────┐
│  Tests    ████████████████████████████████████████ 211 │
│  mypy     ████████████████████████████████████████ clean│
│  Package  ████████████████████████████████████████ done │
│  README   ████████████████████████████████████████ full │
│  Execute  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ STUB │
│  CLI      ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ MISS │
└─────────────────────────────────────────────────────┘
```

**Ready to ship as v0.1.0-beta once OSWEngine.execute() is implemented.**
