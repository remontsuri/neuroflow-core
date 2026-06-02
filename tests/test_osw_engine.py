"""Tests for osw_engine — DAG, GraphMemory, StateMachine, OSWEngine."""

import time

import pytest

from neuroflow_core.osw_engine import (
    AgentState,
    AgentCard,
    OSWEngine,
    CycleError,
    Fact,
)


# ======================================================================
# DAG — topological sort, cycle detection, levels
# ======================================================================


class TestDAG:
    def test_add_node(self, dag):
        dag.add_node("a", prompt="Do A")
        assert "a" in dag.nodes
        assert dag.nodes["a"]["prompt"] == "Do A"
        assert dag.edges["a"] == set()

    def test_add_node_idempotent(self, dag):
        dag.add_node("a", prompt="First")
        dag.add_node("a", prompt="Second")  # shouldn't overwrite
        assert dag.nodes["a"]["prompt"] == "First"

    def test_add_edge_creates_nodes(self, dag):
        dag.add_edge("a", "b")
        assert "a" in dag.nodes
        assert "b" in dag.nodes
        assert "b" in dag.edges["a"]

    def test_parents(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        assert dag.parents("b") == ["a"]
        assert dag.parents("c") == ["a"]
        assert dag.parents("a") == []

    def test_children(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        assert set(dag.children("a")) == {"b", "c"}
        assert dag.children("b") == []

    def test_ancestors(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "d")
        assert dag.ancestors("d") == {"a", "b", "c"}
        assert dag.ancestors("a") == set()

    def test_topological_sort_simple(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        order = dag.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_topological_sort_complex(self, dag):
        #  a   b
        #  |  /|
        #  c d |
        #  |/  |
        #  e   f
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("b", "f")
        dag.add_edge("c", "e")
        dag.add_edge("d", "e")
        order = dag.topological_sort()
        for src, deps in dag.edges.items():
            for dst in deps:
                assert order.index(src) < order.index(dst)

    def test_topological_sort_disconnected(self, dag):
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        order = dag.topological_sort()
        assert set(order) == {"a", "b", "c"}

    def test_cycle_detection_direct(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_cycle_detection_indirect(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_cycle_detection_self_loop(self, dag):
        dag.add_edge("a", "b")
        dag.add_edge("b", "b")  # self-loop
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_cycle_error_message(self, dag):
        dag.add_edge("x", "y")
        dag.add_edge("y", "x")
        try:
            dag.topological_sort()
        except CycleError as e:
            assert "cycle" in str(e).lower()

    def test_levels_simple(self, dag):
        # a -> b -> c
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        levels = dag.levels()
        assert len(levels) == 3
        assert "a" in levels[0]
        assert "b" in levels[1]
        assert "c" in levels[2]

    def test_levels_diamond(self, dag):
        #   a
        #  / \
        # b   c
        #  \ /
        #   d
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        levels = dag.levels()
        assert len(levels) == 3
        assert "a" in levels[0]
        assert "b" in levels[1]
        assert "c" in levels[1]
        assert "d" in levels[2]

    def test_levels_disconnected(self, dag):
        dag.add_node("a", prompt="standalone")
        dag.add_node("b", prompt="also standalone")
        levels = dag.levels()
        assert len(levels) == 1
        assert set(levels[0]) == {"a", "b"}

    def test_levels_single_node(self, dag):
        dag.add_node("only")
        levels = dag.levels()
        assert levels == [["only"]]

    def test_levels_empty(self, dag):
        """Empty DAG returns a single empty level (depth 0 with no nodes)."""
        levels = dag.levels()
        assert levels == [[]]  # current implementation: one empty layer


# ======================================================================
# GraphMemory — remember, recall, search, decay
# ======================================================================


class TestGraphMemory:
    def test_remember_and_recall(self, memory):
        memory.remember("user:1:name", "Alice")
        assert memory.recall("user:1:name") == "Alice"

    def test_recall_missing_key(self, memory):
        assert memory.recall("nonexistent") is None

    def test_remember_overwrites(self, memory):
        memory.remember("key", "old_value")
        memory.remember("key", "new_value")
        assert memory.recall("key") == "new_value"

    def test_remember_with_source(self, memory):
        memory.remember("key", "value", source="test_source")
        results = memory.search("key")
        assert len(results) == 1
        assert results[0].source == "test_source"

    def test_search_by_key(self, memory):
        memory.remember("user:1:name", "Alice")
        memory.remember("user:2:name", "Bob")
        results = memory.search("user:1")
        assert len(results) == 1
        assert results[0].key == "user:1:name"

    def test_search_by_value(self, memory):
        memory.remember("id:1", "Alice Smith")
        memory.remember("id:2", "Bob Jones")
        results = memory.search("Smith")
        assert len(results) == 1
        assert results[0].key == "id:1"

    def test_search_with_min_trust(self, memory):
        memory.remember("high", "value1")
        memory.remember("low", "value2")
        import sqlite3
        with sqlite3.connect(memory._db_path) as conn:
            conn.execute("UPDATE facts SET trust = 0.1 WHERE key = 'low'")
        results = memory.search("value", min_trust=0.5)
        assert len(results) == 1
        assert results[0].key == "high"

    def test_search_limit(self, memory):
        for i in range(5):
            memory.remember(f"key:{i}", f"value:{i}")
        results = memory.search("key", limit=3)
        assert len(results) == 3

    def test_search_empty(self, memory):
        assert memory.search("nothing") == []

    def test_forget(self, memory):
        memory.remember("key", "value")
        assert memory.recall("key") == "value"
        memory.forget("key")
        assert memory.recall("key") is None

    def test_forget_nonexistent(self, memory):
        memory.forget("nonexistent")  # should not raise

    def test_clear(self, memory):
        memory.remember("a", "1")
        memory.remember("b", "2")
        memory.clear()
        assert memory.recall("a") is None
        assert memory.recall("b") is None
        assert memory.stats()["total_facts"] == 0

    def test_stats_empty(self, memory):
        stats = memory.stats()
        assert stats["total_facts"] == 0
        assert stats["avg_trust"] == 0.0

    def test_stats_with_data(self, memory):
        memory.remember("a", "1")
        memory.remember("b", "2")
        stats = memory.stats()
        assert stats["total_facts"] == 2
        assert stats["avg_trust"] == 1.0

    def test_run_decay_old_facts_pruned(self, memory):
        """Facts with very old timestamps should get pruned below trust=0.3."""
        memory.remember("old", "ancient")
        import sqlite3
        old_ts = time.time() - (100 * 86400)
        with sqlite3.connect(memory._db_path) as conn:
            conn.execute("UPDATE facts SET timestamp = ? WHERE key = 'old'", (old_ts,))
        pruned = memory.run_decay(halflife_days=7.0)
        assert pruned == 1
        assert memory.recall("old") is None

    def test_run_decay_fresh_facts_kept(self, memory):
        """Recently added facts should survive decay."""
        memory.remember("fresh", "new")
        pruned = memory.run_decay(halflife_days=7.0)
        assert pruned == 0
        assert memory.recall("fresh") == "new"

    def test_run_decay_reduces_trust(self, memory):
        memory.remember("aging", "data")
        old_ts = time.time() - (7 * 86400)
        import sqlite3
        with sqlite3.connect(memory._db_path) as conn:
            conn.execute("UPDATE facts SET timestamp = ? WHERE key = 'aging'", (old_ts,))
        memory.run_decay(halflife_days=7.0)
        assert memory.recall("aging") == "data"

    def test_run_decay_fact_below_threshold_not_recallable(self, memory):
        memory.remember("borderline", "test")
        import sqlite3
        old_ts = time.time() - (14 * 86400)
        with sqlite3.connect(memory._db_path) as conn:
            conn.execute("UPDATE facts SET timestamp = ? WHERE key = 'borderline'", (old_ts,))
        memory.run_decay(halflife_days=7.0)
        assert memory.recall("borderline") is None

    def test_recall_respects_trust_threshold(self, memory):
        memory.remember("low_trust", "secret")
        import sqlite3
        with sqlite3.connect(memory._db_path) as conn:
            conn.execute("UPDATE facts SET trust = 0.2 WHERE key = 'low_trust'")
        assert memory.recall("low_trust") is None


# ======================================================================
# Fact — decay and equality
# ======================================================================


class TestFact:
    def test_decay_default(self):
        f = Fact(key="test", value="val")
        old_trust = f.trust
        f.decay(halflife_days=7.0)
        assert f.trust == pytest.approx(old_trust, rel=0.01)

    def test_decay_old_fact(self):
        f = Fact(key="test", value="val", timestamp=time.time() - (7 * 86400))
        f.decay(halflife_days=7.0)
        assert f.trust == pytest.approx(0.5, rel=0.05)

    def test_decay_two_halflives(self):
        f = Fact(key="test", value="val", timestamp=time.time() - (14 * 86400))
        f.decay(halflife_days=7.0)
        assert f.trust == pytest.approx(0.25, rel=0.05)

    def test_equality_by_key(self):
        f1 = Fact(key="same", value="a")
        f2 = Fact(key="same", value="b")
        assert f1 == f2

    def test_inequality(self):
        f1 = Fact(key="a", value="x")
        f2 = Fact(key="b", value="x")
        assert f1 != f2

    def test_hash_by_key(self):
        f1 = Fact(key="k", value="v1")
        f2 = Fact(key="k", value="v2")
        assert hash(f1) == hash(f2)


# ======================================================================
# StateMachine → renamed to AgentStateMachine
# ======================================================================


class TestAgentStateMachine:
    def test_initial_state(self, state_machine):
        assert state_machine.state == AgentState.IDLE

    def test_valid_transition(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        assert state_machine.state == AgentState.WORKING

    def test_invalid_transition_raises(self, state_machine):
        with pytest.raises(ValueError, match="Cannot transition"):
            state_machine.transition(AgentState.FAILED)  # IDLE -> FAILED not allowed

    def test_invalid_transition_message(self, state_machine):
        try:
            state_machine.transition(AgentState.DONE)
        except ValueError as e:
            assert "idle" in str(e).lower()
            assert "done" in str(e).lower()

    def test_full_lifecycle(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.DONE)
        assert state_machine.state == AgentState.DONE

    def test_working_to_waiting_to_working(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.WAITING)
        assert state_machine.state == AgentState.WAITING
        state_machine.transition(AgentState.WORKING)
        assert state_machine.state == AgentState.WORKING

    def test_failed_to_idle(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.FAILED)
        assert state_machine.state == AgentState.FAILED
        state_machine.transition(AgentState.IDLE)
        assert state_machine.state == AgentState.IDLE

    def test_cancelled_to_idle(self, state_machine):
        state_machine.transition(AgentState.CANCELLED)
        assert state_machine.state == AgentState.CANCELLED
        state_machine.transition(AgentState.IDLE)
        assert state_machine.state == AgentState.IDLE

    def test_reset(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.reset()
        assert state_machine.state == AgentState.IDLE

    def test_is_terminal_idle(self, state_machine):
        assert state_machine.is_terminal() is False

    def test_is_terminal_done(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.DONE)
        assert state_machine.is_terminal() is True

    def test_is_terminal_failed(self, state_machine):
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.FAILED)
        assert state_machine.is_terminal() is True

    def test_is_terminal_cancelled(self, state_machine):
        state_machine.transition(AgentState.CANCELLED)
        assert state_machine.is_terminal() is False

    # ------------------------------------------------------------------
    # Invalid transition paths
    # ------------------------------------------------------------------

    def test_cannot_skip_working_to_failed(self, state_machine):
        """Cannot go IDLE -> FAILED directly."""
        with pytest.raises(ValueError):
            state_machine.transition(AgentState.FAILED)

    def test_cannot_skip_working_to_done(self, state_machine):
        """Cannot go IDLE -> DONE directly."""
        with pytest.raises(ValueError):
            state_machine.transition(AgentState.DONE)

    def test_done_is_absorbing(self, state_machine):
        """Cannot transition out of DONE state."""
        state_machine.transition(AgentState.WORKING)
        state_machine.transition(AgentState.DONE)
        for target in AgentState:
            if target != AgentState.DONE:
                with pytest.raises(ValueError):
                    state_machine.transition(target)

    def test_working_to_cancelled_not_allowed(self, state_machine):
        """WORKING cannot go to CANCELLED directly."""
        state_machine.transition(AgentState.WORKING)
        with pytest.raises(ValueError):
            state_machine.transition(AgentState.CANCELLED)


# ======================================================================
# AgentCard
# ======================================================================


class TestAgentCard:
    def test_defaults(self):
        card = AgentCard(name="worker")
        assert card.name == "worker"
        assert card.role == "worker"
        assert card.model == ""
        assert card.tools == []
        assert card.max_retries == 3
        assert card.timeout_s == 300
        assert card.state_machine.state == AgentState.IDLE

    def test_is_available(self):
        card = AgentCard(name="a")
        assert card.is_available() is True
        card.state_machine.transition(AgentState.WORKING)
        assert card.is_available() is False

    def test_custom_values(self):
        card = AgentCard(
            name="expert",
            role="researcher",
            model="claude-3",
            tools=["search", "code"],
            max_retries=5,
            timeout_s=600,
        )
        assert card.name == "expert"
        assert card.role == "researcher"
        assert card.model == "claude-3"
        assert card.tools == ["search", "code"]
        assert card.max_retries == 5
        assert card.timeout_s == 600


# ======================================================================
# OSWEngine — full orchestration
# ======================================================================


class TestOSWEngine:

    @staticmethod
    def _make_engine(**kwargs) -> OSWEngine:
        kwargs.setdefault(
            "dispatch_fn",
            lambda p: f"Mock result for: {p[:60]}",
        )
        return OSWEngine(**kwargs)

    def test_register_agent(self):
        engine = self._make_engine()
        card = AgentCard(name="researcher")
        engine.register_agent(card)
        assert "researcher" in engine.agents
        assert engine.agents["researcher"] is card

    def test_ingest_goal(self):
        engine = self._make_engine()
        result = engine.ingest_goal("Test goal")
        assert engine.goal == "Test goal"
        assert "Goal accepted" in result
        assert engine.memory.recall("last_goal") == "Test goal"

    def test_decompose_auto(self):
        engine = self._make_engine()
        engine.ingest_goal("Auto decompose")
        node_ids = engine.decompose()
        assert node_ids == ["root"]
        assert "root" in engine.dag.nodes
        assert engine.dag.nodes["root"]["prompt"] == "Auto decompose"

    def test_decompose_manual(self):
        engine = self._make_engine()
        tasks = [
            {"id": "research", "prompt": "Research topic", "depends_on": []},
            {"id": "write", "prompt": "Write report", "depends_on": ["research"]},
        ]
        node_ids = engine.decompose(tasks)
        assert node_ids == ["research", "write"]
        assert "research" in engine.dag.nodes
        assert "write" in engine.dag.nodes
        assert engine.dag.parents("write") == ["research"]

    def test_decompose_with_agents(self):
        engine = self._make_engine()
        engine.register_agent(AgentCard(name="researcher"))
        tasks = [
            {"id": "task1", "prompt": "Do research", "agent": "researcher",
             "priority": 5, "depends_on": []},
        ]
        node_ids = engine.decompose(tasks)
        assert node_ids == ["task1"]
        assert engine.dag.nodes["task1"]["agent"] == "researcher"
        assert engine.dag.nodes["task1"]["priority"] == 5

    def test_execute_simple(self):
        engine = self._make_engine()
        engine.ingest_goal("Do the thing")
        engine.decompose()
        results = engine.execute()
        assert "root" in results
        assert results["root"]["status"] == "ok"

    def test_execute_with_agent_state_transitions(self):
        engine = self._make_engine()
        card = AgentCard(name="worker")
        engine.register_agent(card)
        engine.ingest_goal("Work")
        engine.decompose([{"id": "task1", "prompt": "Do work", "agent": "worker",
                           "depends_on": []}])
        assert card.state_machine.state == AgentState.IDLE
        engine.execute()
        assert card.state_machine.state == AgentState.DONE

    def test_execute_maintains_order(self):
        engine = self._make_engine()
        engine.ingest_goal("Ordered tasks")
        tasks = [
            {"id": "first", "prompt": "Step 1", "depends_on": []},
            {"id": "second", "prompt": "Step 2", "depends_on": ["first"]},
            {"id": "third", "prompt": "Step 3", "depends_on": ["second"]},
        ]
        engine.decompose(tasks)
        engine.execute()
        order = engine.dag.topological_sort()
        assert order == ["first", "second", "third"]

    def test_execute_cycle_error(self):
        engine = self._make_engine()
        engine.ingest_goal("Cyclic")
        tasks = [
            {"id": "a", "prompt": "A", "depends_on": ["b"]},
            {"id": "b", "prompt": "B", "depends_on": ["a"]},
        ]
        engine.decompose(tasks)
        results = engine.execute()
        assert "error" in results
        assert "cycle" in results["error"].lower()

    def test_execute_memory_stores_results(self):
        engine = self._make_engine(memory_path="/tmp/test_osw_memory.db")
        try:
            engine.ingest_goal("Remember")
            engine.decompose()
            engine.execute()
            task_result = engine.memory.recall("task:root")
            assert task_result is not None
            assert "ok" in task_result
        finally:
            engine.memory.clear()

    def test_report_structure(self):
        engine = self._make_engine()
        engine.ingest_goal("Report test")
        engine.decompose()
        engine.execute()
        report = engine.report()
        assert report["goal"] == "Report test"
        assert "metrics" in report
        assert "dag" in report
        assert "results" in report
        assert report["metrics"]["tasks_total"] == 1
        assert report["metrics"]["tasks_completed"] == 1
        assert report["metrics"]["tasks_failed"] == 0
        assert report["dag"]["nodes"] == 1

    def test_report_before_execution(self):
        engine = self._make_engine()
        report = engine.report()
        assert report["goal"] == ""
        assert report["metrics"]["tasks_total"] == 0
        assert report["metrics"]["elapsed_seconds"] == 0

    def test_clear(self):
        engine = self._make_engine()
        engine.ingest_goal("Clear me")
        engine.decompose([{"id": "t1", "prompt": "task", "depends_on": []}])
        engine.execute()
        engine.clear()
        assert engine.goal == ""
        assert len(engine.dag.nodes) == 0
        assert engine.results == {}
        assert engine._metrics["tasks_total"] == 0
        assert engine._metrics["tasks_completed"] == 0
        assert engine._metrics["tasks_failed"] == 0

    def test_multiple_agents(self):
        engine = self._make_engine()
        engine.register_agent(AgentCard(name="researcher"))
        engine.register_agent(AgentCard(name="writer"))
        engine.ingest_goal("Multi-agent task")
        tasks = [
            {"id": "research", "prompt": "Research", "agent": "researcher",
             "depends_on": []},
            {"id": "write", "prompt": "Write", "agent": "writer",
             "depends_on": ["research"]},
        ]
        engine.decompose(tasks)
        engine.execute()
        assert engine.results["research"]["status"] == "ok"
        assert engine.results["write"]["status"] == "ok"

    def test_execute_with_unknown_agent(self):
        engine = self._make_engine()
        engine.ingest_goal("Unknown agent")
        tasks = [
            {"id": "t1", "prompt": "Task", "agent": "nonexistent", "depends_on": []},
        ]
        engine.decompose(tasks)
        results = engine.execute()
        assert results["t1"]["status"] == "ok"

    def test_memory_persistence(self):
        engine = self._make_engine()
        engine.ingest_goal("test")
        engine.decompose()
        engine.execute()
        assert engine.memory.recall("last_goal") == "test"
        assert engine.memory.recall("task:root") is not None

    def test_execute_metrics(self):
        engine = self._make_engine()
        engine.ingest_goal("Metrics test")
        tasks = [
            {"id": "a", "prompt": "A", "depends_on": []},
            {"id": "b", "prompt": "B", "depends_on": ["a"]},
        ]
        engine.decompose(tasks)
        engine.execute()
        report = engine.report()
        assert report["metrics"]["tasks_total"] == 2
        assert report["metrics"]["tasks_completed"] == 2
        assert report["metrics"]["tasks_failed"] == 0


# ======================================================================
# Edge-case tests (Eval block — dispatch failure, output contract)
# ======================================================================


class TestDispatchFailure:
    """What happens when dispatch_fn raises or returns garbage."""

    def test_dispatch_raises_exception(self):
        """Dispatch that raises should produce 'failed' status with error."""
        def failing_dispatch(p):
            raise RuntimeError("LLM quota exceeded")

        engine = OSWEngine(dispatch_fn=failing_dispatch)
        engine.ingest_goal("Failing task")
        engine.decompose()
        results = engine.execute()
        assert results["root"]["status"] == "failed"
        assert "LLM quota exceeded" in results["root"]["error"]

    def test_dispatch_raises_sets_agent_failed(self):
        """When dispatch fails, agent state should be FAILED."""
        def failing_dispatch(p):
            raise ValueError("Bad input")

        engine = OSWEngine(dispatch_fn=failing_dispatch)
        card = AgentCard(name="worker")
        engine.register_agent(card)
        engine.ingest_goal("Failing agent task")
        tasks = [{"id": "t1", "prompt": "Do it", "agent": "worker", "depends_on": []}]
        engine.decompose(tasks)
        engine.execute()
        assert engine.results["t1"]["status"] == "failed"
        assert card.state_machine.state == AgentState.FAILED

    def test_dispatch_non_string_result(self):
        """Dispatch returning non-string should still be accepted (coerced by caller)."""
        def weird_dispatch(p):
            return 42  # int, not str

        engine = OSWEngine(dispatch_fn=weird_dispatch)
        engine.ingest_goal("Weird output")
        engine.decompose()
        results = engine.execute()
        assert results["root"]["status"] == "ok"
        # str(42) == "42"
        assert isinstance(results["root"]["output"], int)

    def test_dispatch_empty_string(self):
        """Dispatch returning empty string should succeed with empty output."""
        def empty_dispatch(p):
            return ""

        engine = OSWEngine(dispatch_fn=empty_dispatch)
        engine.ingest_goal("Empty")
        engine.decompose()
        results = engine.execute()
        assert results["root"]["status"] == "ok"
        assert results["root"]["output"] == ""

    def test_partial_failure_in_chain(self):
        """Chain: A succeeds, B fails, C depends on B — C should still run (no skip)."""
        call_log = []

        def chain_dispatch(prompt):
            call_log.append(prompt[:20])
            if "fail" in prompt:
                raise RuntimeError("B failed")
            return f"done: {prompt[:20]}"

        engine = OSWEngine(dispatch_fn=chain_dispatch)
        engine.ingest_goal("Partial fail")
        tasks = [
            {"id": "a", "prompt": "success A", "depends_on": []},
            {"id": "b", "prompt": "fail B", "depends_on": ["a"]},
            {"id": "c", "prompt": "success C", "depends_on": ["b"]},
        ]
        engine.decompose(tasks)
        results = engine.execute()
        assert results["a"]["status"] == "ok"
        assert results["b"]["status"] == "failed"
        assert results["c"]["status"] == "ok"
        assert len(call_log) == 3  # all three ran

    def test_metrics_reflect_failures(self):
        """After partial failure, metrics should show both completed and failed."""
        def half_fail(p):
            if "fail" in p:
                raise RuntimeError("boom")
            return "ok"

        engine = OSWEngine(dispatch_fn=half_fail)
        engine.ingest_goal("Metrics with fails")
        tasks = [
            {"id": "ok1", "prompt": "fine", "depends_on": []},
            {"id": "fail1", "prompt": "fail myself", "depends_on": []},
        ]
        engine.decompose(tasks)
        engine.execute()
        report = engine.report()
        assert report["metrics"]["tasks_total"] == 2
        assert report["metrics"]["tasks_completed"] == 1
        assert report["metrics"]["tasks_failed"] == 1

    def test_all_fail_metrics(self):
        """When every task fails, metrics should show 0 completed, N failed."""
        def always_fail(p):
            raise RuntimeError("always fails")

        engine = OSWEngine(dispatch_fn=always_fail)
        engine.ingest_goal("All fail")
        tasks = [
            {"id": "a", "prompt": "A", "depends_on": []},
            {"id": "b", "prompt": "B", "depends_on": ["a"]},
        ]
        engine.decompose(tasks)
        engine.execute()
        report = engine.report()
        assert report["metrics"]["tasks_total"] == 2
        assert report["metrics"]["tasks_completed"] == 0
        assert report["metrics"]["tasks_failed"] == 2


# ======================================================================
# DispatcherRegistry — pluggable dispatch
# ======================================================================


class TestDispatcherRegistry:
    def test_register_and_dispatch(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        reg.register("mock", lambda p: f"result:{p}", default=True)
        assert reg.dispatch("hello") == "result:hello"

    def test_default_takes_first(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        reg.register("a", lambda p: "from-a")
        reg.register("b", lambda p: "from-b")
        assert reg.default_name == "a"

    def test_explicit_default_override(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        reg.register("a", lambda p: "from-a")
        reg.register("b", lambda p: "from-b", default=True)
        assert reg.default_name == "b"
        assert reg.dispatch("x") == "from-b"

    def test_dispatch_by_name(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        reg.register("mock", lambda p: "default", default=True)
        reg.register("alt", lambda p: "alt-mode")
        assert reg.dispatch("x", name="alt") == "alt-mode"

    def test_list_returns_names(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        reg.register("a", lambda p: "")
        reg.register("b", lambda p: "")
        assert sorted(reg.list()) == ["a", "b"]

    def test_get_returns_fn(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        fn = lambda p: "yes"
        reg.register("x", fn)
        assert reg.get("x") is fn

    def test_get_unknown_raises(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        with pytest.raises(KeyError, match="No dispatcher registered: 'nope'"):
            reg.get("nope")

    def test_dispatch_unknown_raises(self):
        from neuroflow_core.osw_engine import DispatcherRegistry

        reg = DispatcherRegistry()
        with pytest.raises(KeyError, match="No dispatcher registered"):
            reg.dispatch("x")


class TestDispatcherRegistryIntegration:
    """Registry integrated with OSWEngine."""

    def test_register_dispatcher(self):
        engine = OSWEngine()
        engine.register_dispatcher("test-dispatch", lambda p: f"custom:{p}")
        assert "test-dispatch" in engine._dispatchers.list()

    def test_register_dispatcher_sets_default(self):
        engine = OSWEngine()
        engine.register_dispatcher("primary", lambda p: "primary", default=True)
        engine.ingest_goal("test")
        engine.decompose([{"id": "t1", "prompt": "go", "depends_on": []}])
        results = engine.execute()
        assert results["t1"]["output"] == "primary"

    def test_register_dispatcher_does_not_break_existing_tests(self):
        """Existing tests pass dispatch_fn — it becomes 'custom' default."""
        engine = OSWEngine(dispatch_fn=lambda p: f"legacy:{p}")
        assert engine._dispatchers.default_name == "custom"
        engine.ingest_goal("test")
        engine.decompose([{"id": "t1", "prompt": "go", "depends_on": []}])
        results = engine.execute()
        assert results["t1"]["output"] == "legacy:go"

    def test_register_dispatcher_override_default(self):
        """register_dispatcher with default=True replaces the default."""
        engine = OSWEngine(dispatch_fn=lambda p: "old-default")
        engine.register_dispatcher("new-default", lambda p: "new-default", default=True)
        assert engine._dispatchers.default_name == "new-default"
        engine.ingest_goal("override")
        engine.decompose([{"id": "t1", "prompt": "go", "depends_on": []}])
        results = engine.execute()
        assert results["t1"]["output"] == "new-default"
