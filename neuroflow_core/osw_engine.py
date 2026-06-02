"""Takes a goal, splits it into a DAG of tasks, runs them, and gives you the results.

This engine handles orchestration for multi-step agent workflows. Give it a
high-level goal, it decomposes the work into a dependency graph, dispatches
tasks to agents, and collects the output.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

DispatchFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# DispatcherRegistry — pluggable named dispatch strategies
# ---------------------------------------------------------------------------


class DispatcherRegistry:
    """Pluggable dispatch registry — register named dispatchers, pick one.

    Usage::

        reg = DispatcherRegistry()
        reg.register("hermes-cli", my_hermes_dispatch, default=True)
        reg.register("mock", lambda p: f"mock:{p}")
        result = reg.dispatch("some prompt")           # uses default
        result = reg.dispatch("some prompt", name="mock")  # explicit
    """

    def __init__(self) -> None:
        self._dispatchers: dict[str, DispatchFn] = {}
        self._default_name: str = ""

    def register(self, name: str, fn: DispatchFn, *, default: bool = False) -> None:
        """Register a dispatcher by name. Pass ``default=True`` to make it the default."""
        self._dispatchers[name] = fn
        if default or not self._default_name:
            self._default_name = name

    @property
    def default_name(self) -> str:
        return self._default_name

    def list(self) -> list[str]:
        """Return all registered dispatcher names."""
        return list(self._dispatchers.keys())

    def get(self, name: str) -> DispatchFn:
        """Fetch a registered dispatcher by name."""
        if name not in self._dispatchers:
            raise KeyError(f"No dispatcher registered: '{name}'")
        return self._dispatchers[name]

    def dispatch(self, prompt: str, name: str | None = None) -> str:
        """Run a prompt through the named (or default) dispatcher."""
        fn_name = name or self._default_name
        if fn_name not in self._dispatchers:
            raise KeyError(
                f"No dispatcher registered: '{fn_name}'. "
                f"Available: {list(self._dispatchers.keys())}"
            )
        return self._dispatchers[fn_name](prompt)


# ---------------------------------------------------------------------------
# DAG — dependency graph for task orchestration
# ---------------------------------------------------------------------------


class CycleError(ValueError):
    """Raised when a cycle is detected in the DAG."""


@dataclass
class DAG:
    """A directed acyclic graph with topological sort.

    Each node carries whatever data you need. An edge a → b means 'a must
    finish before b starts'.
    """

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node_id: str, **data: Any) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = data
            self.edges.setdefault(node_id, set())

    def add_edge(self, from_node: str, to_node: str) -> None:
        """from_node must complete before to_node starts."""
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)
        self.edges[from_node].add(to_node)

    def parents(self, node_id: str) -> list[str]:
        """Tasks this node depends on (reverse edge lookup)."""
        return [n for n, deps in self.edges.items() if node_id in deps]

    def children(self, node_id: str) -> list[str]:
        return list(self.edges.get(node_id, set()))

    def ancestors(self, node_id: str) -> set[str]:
        """All nodes that must run before node_id."""
        result: set[str] = set()
        stack = [node_id]
        while stack:
            n = stack.pop()
            for p in self.parents(n):
                if p not in result:
                    result.add(p)
                    stack.append(p)
        return result

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm — raises CycleError if there's a loop."""
        in_degree = {n: 0 for n in self.nodes}
        for deps in self.edges.values():
            for d in deps:
                in_degree[d] = in_degree.get(d, 0) + 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        sorted_nodes: list[str] = []

        while queue:
            n = queue.popleft()
            sorted_nodes.append(n)
            for child in self.edges.get(n, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            raise CycleError(
                f"DAG has a cycle: sorted {len(sorted_nodes)}/{len(self.nodes)} nodes"
            )
        return sorted_nodes

    def levels(self) -> list[list[str]]:
        """Group nodes by dependency depth — nodes at the same level can run in parallel."""
        sorted_nodes = self.topological_sort()
        depth: dict[str, int] = {}
        for n in sorted_nodes:
            parent_depths = [depth.get(p, 0) for p in self.parents(n)]
            depth[n] = max(parent_depths) + 1 if parent_depths else 0
        max_depth = max(depth.values(), default=0)
        layers: list[list[str]] = [[] for _ in range(max_depth + 1)]
        for n, d in depth.items():
            layers[d].append(n)
        return layers


# ---------------------------------------------------------------------------
# GraphMemory — persistent state with trust scoring and decay
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    key: str
    value: str
    source: str = "system"
    trust: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def decay(self, halflife_days: float = 7.0) -> None:
        elapsed_days = (time.time() - self.timestamp) / 86400
        self.trust *= 0.5 ** (elapsed_days / halflife_days)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return self.key == other.key


class GraphMemory:
    """Persistent key-value store with trust scores that decay over time.

    Facts have a source and a trust score. Queries filter by trust — old
    or rarely-accessed facts naturally sink below the retrieval threshold
    and get pruned.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or os.environ.get(
            "NEUROFLOW_MEMORY_PATH", "/tmp/graph_memory.db"
        )
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    source TEXT DEFAULT 'system',
                    trust REAL DEFAULT 1.0,
                    timestamp REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_trust
                ON facts(trust DESC)
            """)

    def remember(self, key: str, value: str, source: str = "system") -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO facts (key, value, source, trust, timestamp)
                   VALUES (?, ?, ?, 1.0, ?)""",
                (key, value, source, time.time()),
            )

    def recall(self, key: str) -> str | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value, trust FROM facts WHERE key = ?", (key,)
            ).fetchone()
            if row and row[1] > 0.3:
                return str(row[0])
        return None

    def search(self, query: str, min_trust: float = 0.3, limit: int = 10) -> list[Fact]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT key, value, source, trust, timestamp FROM facts
                   WHERE (key LIKE ? OR value LIKE ?) AND trust >= ?
                   ORDER BY trust DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", min_trust, limit),
            ).fetchall()
            return [Fact(key=r[0], value=r[1], source=r[2], trust=r[3], timestamp=r[4])
                    for r in rows]

    def forget(self, key: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM facts WHERE key = ?", (key,))

    def run_decay(self, halflife_days: float = 7.0) -> int:
        """Lower trust for every fact. Returns how many dropped below 0.3 and were pruned."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT key, trust, timestamp FROM facts"
            ).fetchall()
            pruned = 0
            for key, trust, ts in rows:
                elapsed_days = (time.time() - ts) / 86400
                new_trust = trust * (0.5 ** (elapsed_days / halflife_days))
                if new_trust < 0.3:
                    conn.execute("DELETE FROM facts WHERE key = ?", (key,))
                    pruned += 1
                else:
                    conn.execute(
                        "UPDATE facts SET trust = ?, timestamp = ? WHERE key = ?",
                        (new_trust, time.time(), key),
                    )
            return pruned

    def clear(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM facts")

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            avg_trust = conn.execute(
                "SELECT AVG(trust) FROM facts"
            ).fetchone()[0] or 0.0
        return {"total_facts": total, "avg_trust": round(avg_trust, 3)}


# ---------------------------------------------------------------------------
# StateMachine — agent lifecycle manager
# ---------------------------------------------------------------------------


class AgentState(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStateMachine:
    """Agent lifecycle state machine with guarded transitions.

    Tries to go IDLE → FAILED directly? Nope — that's not in the allowed
    table and it'll raise.
    """

    ALLOWED: dict[AgentState, set[AgentState]] = {
        AgentState.IDLE: {AgentState.WORKING, AgentState.CANCELLED},
        AgentState.WORKING: {AgentState.WAITING, AgentState.DONE, AgentState.FAILED},
        AgentState.WAITING: {AgentState.WORKING, AgentState.CANCELLED},
        AgentState.DONE: set(),
        AgentState.FAILED: {AgentState.IDLE},
        AgentState.CANCELLED: {AgentState.IDLE},
    }

    def __init__(self, initial: AgentState = AgentState.IDLE):
        self._state = initial
        self._lock = threading.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(self, target: AgentState) -> None:
        with self._lock:
            if target not in self.ALLOWED.get(self._state, set()):
                raise ValueError(
                    f"Cannot transition from {self._state.value} to {target.value}"
                )
            self._state = target

    def reset(self) -> None:
        with self._lock:
            self._state = AgentState.IDLE

    def is_terminal(self) -> bool:
        return self._state in (AgentState.DONE, AgentState.FAILED)


# ---------------------------------------------------------------------------
# AgentCard — lightweight agent registration
# ---------------------------------------------------------------------------


@dataclass
class AgentCard:
    """Agent registration metadata — name, role, model, state."""

    name: str
    role: str = "worker"
    model: str = ""
    tools: list[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_s: int = 300
    state_machine: AgentStateMachine = field(default_factory=AgentStateMachine)

    def is_available(self) -> bool:
        return self.state_machine.state == AgentState.IDLE


# ---------------------------------------------------------------------------
# OSWEngine — main orchestrator
# ---------------------------------------------------------------------------


class OSWEngine:
    """The orchestrator — give it a goal, it builds a task DAG and runs the whole thing.

        engine = OSWEngine()
        engine.register_agent(AgentCard(name="researcher"))
        engine.ingest_goal("Analyse user churn in Q2 2026")
        engine.decompose()
        engine.execute()
        print(engine.report())
    """

    def __init__(
        self,
        memory_path: str | None = None,
        dispatch_fn: DispatchFn | None = None,
    ):
        self.memory = GraphMemory(
            db_path=(
                memory_path
                or os.environ.get("NEUROFLOW_MEMORY_PATH")
                or "/tmp/neuroflow_memory.db"
            )
        )
        self._dispatchers = DispatcherRegistry()
        self._dispatchers.register("hermes-cli", self._default_dispatch, default=True)
        if dispatch_fn is not None:
            self._dispatchers.register("custom", dispatch_fn, default=True)
        self.agents: dict[str, AgentCard] = {}
        self.goal: str = ""
        self.dag = DAG()
        self.results: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "started_at": 0.0,
            "tasks_total": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
        }

    def _default_dispatch(self, prompt: str) -> str:
        """Run a prompt through the `hermes` CLI as a subprocess dispatch."""
        cmd = [
            "hermes",
            "-p", "eni-worker",
        ]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"Dispatch failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def register_agent(self, card: AgentCard) -> None:
        self.agents[card.name] = card

    def register_dispatcher(self, name: str, fn: DispatchFn, *, default: bool = False) -> None:
        """Register a named dispatch strategy. Pass ``default=True`` to switch default."""
        self._dispatchers.register(name, fn, default=default)

    def ingest_goal(self, goal: str) -> str:
        self.goal = goal
        self._metrics["started_at"] = time.time()
        self.memory.remember("last_goal", goal, source="user")
        return f"Goal accepted: {goal}"

    def decompose(self, tasks: list[dict[str, Any]] | None = None) -> list[str]:
        """Turn the goal (or explicit tasks) into a DAG.

        If no tasks are provided, it creates a single root task from the
        goal string.
        """
        with self._lock:
            if tasks is None:
                tasks = [{"id": "root", "prompt": self.goal, "depends_on": []}]

            node_ids: list[str] = []
            for t in tasks:
                tid = t["id"]
                self.dag.add_node(tid, prompt=t.get("prompt", ""),
                                  agent=t.get("agent", ""),
                                  priority=t.get("priority", 0))
                for dep in t.get("depends_on", []):
                    self.dag.add_edge(dep, tid)
                node_ids.append(tid)

            self._metrics["tasks_total"] = len(node_ids)
            self.memory.remember("last_decomposition",
                                 json.dumps(node_ids), source="system")
            return node_ids

    def _task_result(self, node_id: str, status: str,
                     output_or_error: str, agent_name: str = "") -> dict[str, Any]:
        """Build a Monadix-compliant result with reason and next_recommended_action."""
        if status == "ok":
            reason = f"Task '{node_id}' completed"
            if agent_name:
                reason += f" via agent '{agent_name}'"
            next_action = "proceed" if not self._is_terminal_node(node_id) else "finalize"
        else:
            reason = f"Task '{node_id}' failed: {output_or_error}"
            next_action = f"retry:{node_id}"

        return {
            "status": status,
            "output": output_or_error if status == "ok" else "",
            "error": output_or_error if status != "ok" else "",
            "reason": reason,
            "next_recommended_action": next_action,
        }

    def _is_terminal_node(self, node_id: str) -> bool:
        """Check if a node has no downstream dependents (leaf / sink node)."""
        return not any(node_id in deps for deps in self.dag.edges.values())

    def execute(self) -> dict[str, Any]:
        """Walk the DAG topologically and run each task. Returns {task_id: result}."""
        try:
            order = self.dag.topological_sort()
        except CycleError as e:
            return {"error": str(e), "reason": "DAG contains a cycle",
                    "next_recommended_action": "fix_dag"}

        for node_id in order:
            node_data = self.dag.nodes.get(node_id, {})
            agent_name = node_data.get("agent", "")

            agent = self.agents.get(agent_name) if agent_name else None
            if agent:
                agent.state_machine.transition(AgentState.WORKING)

            try:
                prompt = node_data.get("prompt", "")
                output = self._dispatchers.dispatch(prompt)
                self.results[node_id] = self._task_result(
                    node_id, "ok", output, agent_name)

                with self._lock:
                    self._metrics["tasks_completed"] += 1

                if agent:
                    agent.state_machine.transition(AgentState.DONE)
            except Exception as exc:
                err = str(exc)
                self.results[node_id] = self._task_result(
                    node_id, "failed", err, agent_name)
                with self._lock:
                    self._metrics["tasks_failed"] += 1
                if agent:
                    agent.state_machine.transition(AgentState.FAILED)

            self.memory.remember(f"task:{node_id}",
                                 json.dumps(self.results[node_id]),
                                 source="osw_engine")

        return self.results

    def report(self) -> dict[str, Any]:
        """Package up everything — goal, metrics, DAG stats, per-task results, and verdict."""
        elapsed = time.time() - self._metrics["started_at"] if self._metrics["started_at"] else 0
        total = self._metrics["tasks_total"]
        failed = self._metrics["tasks_failed"]
        completed = self._metrics["tasks_completed"]

        if total == 0:
            verdict = "No tasks executed"
            next_step = "decompose a goal first"
        elif failed == 0:
            verdict = f"All {completed} tasks completed successfully"
            next_step = "finalize" if self.results else "continue"
        else:
            verdict = f"{failed} of {total} tasks failed"
            failed_ids = [nid for nid, r in self.results.items()
                          if r.get("status") == "failed"]
            next_step = f"retry:{','.join(failed_ids)}"

        return {
            "goal": self.goal,
            "verdict": verdict,
            "next_step": next_step,
            "metrics": {
                **self._metrics,
                "elapsed_seconds": round(elapsed, 1),
            },
            "dag": {
                "nodes": len(self.dag.nodes),
                "edges": sum(len(d) for d in self.dag.edges.values()),
            },
            "results": self.results,
        }

    def clear(self) -> None:
        """Reset everything — goal, DAG, results, metrics."""
        self.goal = ""
        self.dag = DAG()
        self.results = {}
        self._metrics = {
            "started_at": 0.0,
            "tasks_total": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
        }
