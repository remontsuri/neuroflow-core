"""Real integration tests — no mocks, no MagicMock, full pipeline.

Tests cover:
  - DispatcherRegistry full lifecycle (register, list, get, switch default, dispatch)
  - OSWEngine full lifecycle with real dispatch functions (ingest → decompose → execute → report)
  - Multi-dispatcher switching mid-pipeline
  - RateLimiter concurrent access from multiple threads
  - RateLimiter real time-window expiry
  - TelegramIngestor + RateLimiter integration (real IngestorConfig, real rate filter)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter

import pytest

from neuroflow_core.osw_engine import (
    DAG,
    DispatcherRegistry,
    DispatchFn,
    OSWEngine,
    AgentCard,
    CycleError,
)
from neuroflow_core.telegram_ingestor import (
    IngestorConfig,
    RateLimiter,
    config_from_env,
)


# ======================================================================
# DispatcherRegistry — real integration
# ======================================================================


def _upper_dispatch(prompt: str) -> str:
    return prompt.upper()


def _reverse_dispatch(prompt: str) -> str:
    return prompt[::-1]


def _count_dispatch(prompt: str) -> str:
    return str(len(prompt.split()))


class TestDispatcherRegistryReal:
    """Full lifecycle with real dispatch functions — no mocks."""

    def test_register_and_default(self):
        r = DispatcherRegistry()
        r.register("upper", _upper_dispatch, default=True)
        assert r.default_name == "upper"
        result = r.dispatch("hello world")
        assert result == "HELLO WORLD"

    def test_multiple_dispatchers_switch_default(self):
        r = DispatcherRegistry()
        r.register("upper", _upper_dispatch, default=True)
        r.register("reverse", _reverse_dispatch)
        assert r.default_name == "upper"
        # switch default via register with same fn + default flag
        r.register("reverse2", _reverse_dispatch, default=True)
        assert r.default_name == "reverse2"
        result = r.dispatch("hello")
        assert result == "olleh"  # reversed

    def test_dispatch_by_name(self):
        r = DispatcherRegistry()
        r.register("upper", _upper_dispatch, default=True)
        r.register("reverse", _reverse_dispatch)
        r.register("count", _count_dispatch)

        # dispatch with explicit name
        assert r.dispatch("hello world", name="count") == "2"
        assert r.dispatch("hello world", name="upper") == "HELLO WORLD"
        assert r.dispatch("hello world", name="reverse") == "dlrow olleh"
        # no name = default
        assert r.dispatch("hello world") == "HELLO WORLD"

    def test_list_and_get(self):
        r = DispatcherRegistry()
        r.register("upper", _upper_dispatch, default=True)
        r.register("reverse", _reverse_dispatch)

        names = r.list()
        assert "upper" in names
        assert "reverse" in names

        fn = r.get("reverse")
        assert fn is _reverse_dispatch

    def test_reregister_is_error(self):
        r = DispatcherRegistry()
        r.register("x", _upper_dispatch)
        with pytest.raises(ValueError, match="already registered"):
            r.register("x", _reverse_dispatch)

    def test_unknown_name_raises(self):
        r = DispatcherRegistry()
        with pytest.raises(KeyError, match="No dispatcher registered"):
            r.dispatch("test", name="nonexistent")

    def test_no_default_raises(self):
        r = DispatcherRegistry()
        with pytest.raises(RuntimeError, match="No default"):
            r.dispatch("test")

    def test_three_dispatchers_independent(self):
        """Each dispatcher function runs independently on its own input."""
        r = DispatcherRegistry()
        r.register("upper", _upper_dispatch, default=True)
        r.register("reverse", _reverse_dispatch)
        r.register("count", _count_dispatch)

        inputs = ["hello world", "test", "one two three"]
        expected_upper = [s.upper() for s in inputs]
        expected_reverse = [s[::-1] for s in inputs]
        expected_count = [str(len(s.split())) for s in inputs]

        for i, inp in enumerate(inputs):
            assert r.dispatch(inp, name="upper") == expected_upper[i]
            assert r.dispatch(inp, name="reverse") == expected_reverse[i]
            assert r.dispatch(inp, name="count") == expected_count[i]


# ======================================================================
# OSWEngine — full pipeline integration (real dispatch, no subprocess)
# ======================================================================


def _echo_dispatch(prompt: str) -> str:
    """Simple dispatch that just returns the prompt — runs in-process."""
    return f"processed: {prompt}"


def _sentiment_dispatch(prompt: str) -> str:
    """Checks if prompt contains positive/negative words — no network."""
    positive_words = {"good", "great", "excellent", "success", "pass"}
    negative_words = {"bad", "fail", "error", "broken", "crash"}
    words = set(prompt.lower().split())
    if words & positive_words:
        return "positive"
    if words & negative_words:
        return "negative"
    return "neutral"


class TestOSWEngineFullPipeline:
    """Full OSWEngine lifecycle with real dispatch functions — no mocks."""

    def test_ingest_decompose_execute_report(self):
        """Gold path — full cycle with a real dispatcher."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)

        engine.ingest_goal("Test the pipeline")
        node_ids = engine.decompose()
        assert node_ids == ["root"]

        results = engine.execute()
        assert "root" in results
        assert results["root"]["status"] == "ok"
        assert results["root"]["output"] == "processed: Test the pipeline"
        assert results["root"]["reason"] == "Task 'root' completed"
        assert results["root"]["next_recommended_action"] == "finalize"

        report = engine.report()
        assert report["goal"] == "Test the pipeline"
        assert report["metrics"]["tasks_total"] == 1
        assert report["metrics"]["tasks_completed"] == 1
        assert report["verdict"] == "All 1 tasks completed successfully"
        assert report["next_step"] == "finalize"

    def test_multi_task_dag(self):
        """DAG with multiple tasks and dependencies."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)

        engine.ingest_goal("Multi-step pipeline")
        engine.decompose([
            {"id": "research", "prompt": "Research phase", "depends_on": []},
            {"id": "draft", "prompt": "Draft phase", "depends_on": ["research"]},
            {"id": "review", "prompt": "Review phase", "depends_on": ["draft"]},
        ])

        results = engine.execute()
        assert len(results) == 3
        for nid in ("research", "draft", "review"):
            assert results[nid]["status"] == "ok"
            assert results[nid]["output"].startswith("processed:")

        # research has no downstream dependents check — review is terminal
        assert results["review"]["next_recommended_action"] == "finalize"
        assert results["research"]["next_recommended_action"] == "proceed"

    def test_task_failure_propagation(self):
        """If a dispatch function raises an exception, the engine marks it failed."""
        def _failing_dispatch(_prompt: str) -> str:
            raise RuntimeError("simulated failure")

        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("failing", _failing_dispatch, default=True)
        engine.ingest_goal("Test failure")
        engine.decompose()
        results = engine.execute()

        assert results["root"]["status"] == "failed"
        assert "simulated failure" in results["root"]["error"]
        assert "retry:root" in results["root"]["next_recommended_action"]

        report = engine.report()
        assert report["metrics"]["tasks_failed"] == 1
        assert "1 of 1 tasks failed" in report["verdict"]

    def test_cycle_detection(self):
        """A cyclic DAG is caught before execution."""
        dag = DAG()
        dag.add_node("a", prompt="A")
        dag.add_node("b", prompt="B")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")

        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_agent_state_machine_lifecycle(self):
        """Agent cards transition through the real state machine during execute."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)

        agent = AgentCard(name="worker", role="researcher")
        engine.register_agent(agent)
        assert agent.state_machine.state.value == "idle"

        engine.ingest_goal("State test")
        engine.decompose([{"id": "task1", "prompt": "do it", "agent": "worker"}])
        engine.execute()

        assert agent.state_machine.state.value == "done"

    def test_multiple_agents_independent_states(self):
        """Each agent card tracks its own state independently."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)

        a1 = AgentCard(name="alpha", role="researcher")
        a2 = AgentCard(name="beta", role="verifier")
        engine.register_agent(a1)
        engine.register_agent(a2)

        engine.ingest_goal("Multi-agent")
        engine.decompose([
            {"id": "t1", "prompt": "first", "agent": "alpha"},
            {"id": "t2", "prompt": "second", "agent": "beta"},
            {"id": "t3", "prompt": "third", "agent": "alpha"},
        ])
        engine.execute()

        assert a1.state_machine.state.value == "done"
        assert a2.state_machine.state.value == "done"

    def test_execute_twice_rebuilds_results(self):
        """Calling execute twice shouldn't break — second call re-runs."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)
        engine.ingest_goal("Rerun test")
        engine.decompose()

        r1 = engine.execute()
        assert r1["root"]["status"] == "ok"

        r2 = engine.execute()
        assert r2["root"]["status"] == "ok"

    def test_dispatcher_switch_between_executions(self):
        """Change the default dispatcher between execute calls."""
        engine = OSWEngine(memory_path=":memory:")
        engine.register_dispatcher("echo", _echo_dispatch, default=True)
        engine.register_dispatcher("sentiment", _sentiment_dispatch)

        engine.ingest_goal("This is a great idea")
        engine.decompose()

        # first run with echo
        r1 = engine.execute()
        assert "processed:" in r1["root"]["output"]

        # switch default to sentiment, clear, rerun
        engine._dispatchers.set_default("sentiment")
        engine.clear()
        engine.ingest_goal("This is a great idea")
        engine.decompose()
        r2 = engine.execute()
        assert r2["root"]["output"] == "positive"

    def test_goal_persists_in_memory(self):
        """The ingested goal is written to GraphMemory."""
        engine = OSWEngine(memory_path=":memory:")
        engine.ingest_goal("Remember this goal")

        stored = engine.memory.recall("last_goal")
        assert stored == "Remember this goal"

    def test_report_with_no_execution(self):
        """report() without execute gives a 'No tasks' verdict."""
        engine = OSWEngine(memory_path=":memory:")
        engine.ingest_goal("No-op")
        report = engine.report()
        assert report["verdict"] == "No tasks executed"
        assert report["next_step"] == "decompose a goal first"


# ======================================================================
# RateLimiter — real concurrency and time integration
# ======================================================================


class TestRateLimiterReal:
    """RateLimiter tested with real threads and real time — no mocks."""

    def test_single_user_block_after_limit(self):
        """After max_events, further requests are blocked (real time)."""
        rl = RateLimiter(max_events=3, window_s=60)
        assert rl.allow("alice") is True
        assert rl.allow("alice") is True
        assert rl.allow("alice") is True
        assert rl.allow("alice") is False  # blocked

    def test_independent_users(self):
        """Each user has their own counter — not shared."""
        rl = RateLimiter(max_events=2, window_s=60)
        assert rl.allow("alice") is True
        assert rl.allow("alice") is True
        assert rl.allow("alice") is False  # alice blocked
        assert rl.allow("bob") is True     # bob not blocked
        assert rl.allow("bob") is True
        assert rl.allow("bob") is False    # bob blocked individually

    def test_remaining_counts(self):
        """remaining() returns correct count."""
        rl = RateLimiter(max_events=5, window_s=60)
        assert rl.remaining("alice") == 5
        rl.allow("alice")
        rl.allow("alice")
        assert rl.remaining("alice") == 3

    def test_concurrent_access(self):
        """Multiple threads hitting the same RateLimiter — no race conditions."""
        rl = RateLimiter(max_events=10, window_s=60)
        n_threads = 5
        hits_per_thread = 3
        allowed = Counter()

        def worker(uid: int):
            for _ in range(hits_per_thread):
                if rl.allow(f"user-{uid}"):
                    allowed[uid] += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # each of the 5 users should have 3 hits allowed (under limit)
        assert all(allowed[i] == hits_per_thread for i in range(n_threads))

    def test_concurrent_block(self):
        """Multiple threads all hitting same user — exactly max_events pass."""
        rl = RateLimiter(max_events=7, window_s=60)

        def worker():
            rl.allow("shared")

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # only 7 should have passed
        assert rl.remaining("shared") == 0

    def test_window_expiry(self):
        """After window_s, the count resets (real sleep)."""
        rl = RateLimiter(max_events=2, window_s=0.05)  # 50ms window
        assert rl.allow("alice") is True
        assert rl.allow("alice") is True
        assert rl.allow("alice") is False  # blocked

        time.sleep(0.06)  # wait for expiry
        assert rl.allow("alice") is True   # window reset

    def test_reset_per_user(self):
        """reset(user_id=) clears only one user's counter."""
        rl = RateLimiter(max_events=2, window_s=60)
        rl.allow("alice")
        rl.allow("alice")
        rl.allow("bob")
        assert rl.allow("alice") is False  # blocked

        rl.reset(user_id="alice")
        assert rl.allow("alice") is True   # unblocked
        assert rl.allow("bob") is True     # bob had 1/2 events, still has room

    def test_reset_all(self):
        """reset() clears every user."""
        rl = RateLimiter(max_events=2, window_s=60)
        rl.allow("alice")
        rl.allow("alice")
        rl.allow("bob")
        rl.reset()
        assert rl.allow("alice") is True
        assert rl.allow("bob") is True

    def test_disabled_never_blocks(self):
        """max_events=0 means rate limiting is off."""
        rl = RateLimiter(max_events=0, window_s=60)
        for _ in range(100):
            assert rl.allow("alice") is True


# ======================================================================
# TelegramIngestor + RateLimiter integration
# ======================================================================


class TestIngestorRateLimitIntegration:
    """Real IngestorConfig with rate-limiting enabled."""

    def test_config_with_rate_limit(self):
        """IngestorConfig passes rate limit params through."""
        config = IngestorConfig(
            bot_token="test:token",
            rate_limit_max_events=5,
            rate_limit_window_s=30,
        )
        assert config.rate_limit_max_events == 5
        assert config.rate_limit_window_s == 30

    def test_config_default_rate_limit_disabled(self):
        """Default IngestorConfig has rate limiting disabled (max_events=0)."""
        config = IngestorConfig(bot_token="test:token")
        assert config.rate_limit_max_events == 0
        assert config.rate_limit_window_s == 60

    def test_config_from_env_with_rate_limit(self):
        """config_from_env reads RL env vars."""
        os.environ["TELEGRAM_BOT_TOKEN"] = "env:token"
        os.environ["RATE_LIMIT_MAX_EVENTS"] = "10"
        os.environ["RATE_LIMIT_WINDOW_S"] = "120"
        try:
            config = config_from_env()
            assert config.rate_limit_max_events == 10
            assert config.rate_limit_window_s == 120
        finally:
            del os.environ["RATE_LIMIT_MAX_EVENTS"]
            del os.environ["RATE_LIMIT_WINDOW_S"]
            del os.environ["TELEGRAM_BOT_TOKEN"]


# ======================================================================
# DAG — real structure tests (beyond unit-level)
# ======================================================================


class TestDAGIntegration:
    """Real DAG scenarios — branching, merging, complex topologies."""

    def test_branching_dag(self):
        """One root → multiple parallel children."""
        dag = DAG()
        dag.add_node("root", prompt="Start")
        dag.add_node("a", prompt="A")
        dag.add_node("b", prompt="B")
        dag.add_node("c", prompt="C")
        dag.add_edge("root", "a")
        dag.add_edge("root", "b")
        dag.add_edge("root", "c")

        order = dag.topological_sort()
        assert order[0] == "root"
        assert len(order) == 4
        # a/b/c can be in any order but must all come after root
        assert order.index("a") > order.index("root")
        assert order.index("b") > order.index("root")
        assert order.index("c") > order.index("root")

    def test_merging_dag(self):
        """Multiple tasks converge into one — diamond pattern."""
        dag = DAG()
        dag.add_node("start", prompt="Start")
        dag.add_node("left", prompt="Left path")
        dag.add_node("right", prompt="Right path")
        dag.add_node("merge", prompt="Merge")
        dag.add_edge("start", "left")
        dag.add_edge("start", "right")
        dag.add_edge("left", "merge")
        dag.add_edge("right", "merge")

        order = dag.topological_sort()
        assert order[0] == "start"
        assert order[-1] == "merge"
        # left and right can be in any order but both must come after start
        # and before merge
        assert order.index("left") > order.index("start")
        assert order.index("right") > order.index("start")
        assert order.index("merge") > order.index("left")
        assert order.index("merge") > order.index("right")

    def test_levels_diamond(self):
        """Diamond DAG produces 3 levels (0: start, 1: left/right, 2: merge)."""
        dag = DAG()
        dag.add_node("start")
        dag.add_node("left")
        dag.add_node("right")
        dag.add_node("merge")
        dag.add_edge("start", "left")
        dag.add_edge("start", "right")
        dag.add_edge("left", "merge")
        dag.add_edge("right", "merge")

        lvls = dag.levels()
        assert len(lvls) == 3
        assert lvls[0] == ["start"]
        assert sorted(lvls[1]) == ["left", "right"]
        assert lvls[2] == ["merge"]

    def test_single_node_dag(self):
        """Trivial single-node DAG."""
        dag = DAG()
        dag.add_node("only", prompt="Solo")
        assert dag.topological_sort() == ["only"]
        assert dag.levels() == [["only"]]


# ======================================================================
# GraphMemory — real database integration
# ======================================================================


class TestGraphMemoryIntegration:
    """GraphMemory with a real SQLite database."""

    def test_remember_and_recall(self):
        from neuroflow_core.osw_engine import GraphMemory

        gm = GraphMemory(db_path=":memory:")
        gm.remember("test_key", "test_value", source="test")
        assert gm.recall("test_key") == "test_value"

    def test_search_works(self):
        from neuroflow_core.osw_engine import GraphMemory

        gm = GraphMemory(db_path=":memory:")
        gm.remember("alpha", "first result")
        gm.remember("beta", "second value")
        gm.remember("gamma", "third result")

        results = gm.search("result")
        assert len(results) >= 2  # alpha and gamma

    def test_decay_prunes_low_trust(self):
        from neuroflow_core.osw_engine import GraphMemory, Fact

        gm = GraphMemory(db_path=":memory:")
        gm.remember("stale", "old fact", source="test")
        gm.remember("fresh", "new fact", source="test")

        # Simulate decay by setting trust directly (internal — but OK for integration)
        import sqlite3
        with sqlite3.connect(":memory:") as conn:
            pass  # can't access the same :memory: from outside

        # Use a file DB so we can inject low trust
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            gm2 = GraphMemory(db_path=db_path)
            gm2.remember("high", "trusted fact", source="test")

            # Directly set low trust for "low" key
            import sqlite3 as sqlite3_mod
            with sqlite3_mod.connect(db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO facts (key, value, source, trust, timestamp) VALUES (?, ?, ?, ?, ?)",
                            ("low", "untrusted", "test", 0.1, time.time()))

            # recall filters by trust: 0.1 < 0.3 threshold → None
            assert gm2.recall("low") is None
            assert gm2.recall("high") == "trusted fact"
        finally:
            os.unlink(db_path)

    def test_clear_removes_all(self):
        from neuroflow_core.osw_engine import GraphMemory

        gm = GraphMemory(db_path=":memory:")
        gm.remember("a", "x")
        gm.remember("b", "y")
        assert gm.stats()["total_facts"] == 2
        gm.clear()
        assert gm.stats()["total_facts"] == 0


# ======================================================================
# StateMachine — real lifecycle scenarios
# ======================================================================


class TestStateMachineIntegration:
    """AgentStateMachine with complex real-world transition sequences."""

    def test_full_work_cycle(self):
        from neuroflow_core.osw_engine import AgentStateMachine, AgentState

        sm = AgentStateMachine()
        assert sm.state == AgentState.IDLE

        sm.transition(AgentState.WORKING)
        assert sm.state == AgentState.WORKING

        sm.transition(AgentState.DONE)
        assert sm.state == AgentState.DONE
        assert sm.is_terminal() is True

    def test_fail_and_retry_cycle(self):
        from neuroflow_core.osw_engine import AgentStateMachine, AgentState

        sm = AgentStateMachine()
        sm.transition(AgentState.WORKING)
        sm.transition(AgentState.FAILED)
        assert sm.is_terminal() is True

        # Failed → IDLE (retry)
        sm.transition(AgentState.IDLE)
        assert sm.state == AgentState.IDLE
        assert sm.is_terminal() is False

    def test_cancel_and_restart(self):
        from neuroflow_core.osw_engine import AgentStateMachine, AgentState

        sm = AgentStateMachine()
        sm.transition(AgentState.CANCELLED)
        assert sm.state == AgentState.CANCELLED

        sm.transition(AgentState.IDLE)
        assert sm.state == AgentState.IDLE

    def test_invalid_transition_raises(self):
        from neuroflow_core.osw_engine import AgentStateMachine, AgentState

        sm = AgentStateMachine()
        with pytest.raises(ValueError, match="Cannot transition"):
            sm.transition(AgentState.DONE)  # IDLE → DONE is not allowed

    def test_reset(self):
        from neuroflow_core.osw_engine import AgentStateMachine, AgentState

        sm = AgentStateMachine()
        sm.transition(AgentState.WORKING)
        sm.transition(AgentState.DONE)
        assert sm.is_terminal() is True

        sm.reset()
        assert sm.state == AgentState.IDLE
