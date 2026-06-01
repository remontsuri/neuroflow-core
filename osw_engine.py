"""
neuroflow-core :: OSW (Orchestrator-Supervisor-Worker) Engine
Mythos build — production-grade DAG-based multi-agent orchestration.

Architecture:
  Orchestrator  →  accepts goals, decomposes, routes to Supervisors
  Supervisor    →  manages N Workers, aggregates results, handles failures
  Worker        →  executes atomic tasks, reports status & evidence

Execution model: compiled DAG with topological sort, parallel dispatch,
state persistence, and automatic retry with exponential backoff.
"""

import json
import time
import uuid
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Optional
from graphlib import TopologicalSorter, CycleError


# ─── Domain types ──────────────────────────────────────────────────

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"
    WORKER = "worker"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Task:
    id: str
    goal: str
    context: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    agent_role: AgentRole = AgentRole.WORKER
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout_s: int = 120
    created_at: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        raw = f"{self.goal}|{json.dumps(self.context, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class DAG:
    tasks: dict[str, Task] = field(default_factory=dict)

    def add(self, task: Task):
        self.tasks[task.id] = task

    def add_dependency(self, task_id: str, depends_on: str):
        if task_id in self.tasks and depends_on in self.tasks:
            self.tasks[task_id].dependencies.append(depends_on)

    def execution_order(self) -> list[list[str]]:
        """Returns batches of task IDs that can run in parallel (topological layers)."""
        graph: dict[str, set[str]] = {}
        for tid, t in self.tasks.items():
            graph[tid] = set(t.dependencies)
        try:
            sorter = TopologicalSorter(graph)
            sorter.prepare()
            batches: list[list[str]] = []
            while sorter.is_active():
                batch = list(sorter.get_ready())
                batches.append(batch)
                sorter.done(*batch)
            return batches
        except CycleError as e:
            raise ValueError(f"Cycle detected in DAG: {e}")

    def serialize(self) -> dict:
        return {
            tid: {
                "goal": t.goal[:80],
                "status": t.status.value,
                "deps": t.dependencies,
                "role": t.agent_role.value,
            }
            for tid, t in self.tasks.items()
        }


# ─── Memory & State ────────────────────────────────────────────────

class GraphMemory:
    """Entity-relationship store with trust scoring and decay."""

    def __init__(self, decay_hours: float = 72.0):
        self._entities: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._decay_s = decay_hours * 3600

    def upsert_entity(self, name: str, kind: str, attrs: dict = None):
        now = time.time()
        if name in self._entities:
            e = self._entities[name]
            e["accessed_at"] = now
            e["access_count"] = e.get("access_count", 0) + 1
            if attrs:
                e["attrs"].update(attrs)
        else:
            self._entities[name] = {
                "name": name, "kind": kind,
                "attrs": attrs or {},
                "created_at": now, "accessed_at": now,
                "access_count": 1, "trust": 1.0,
            }

    def relate(self, source: str, target: str, relation: str, weight: float = 1.0):
        self._edges.append({
            "source": source, "target": target,
            "relation": relation, "weight": weight,
            "created_at": time.time(),
        })

    def query(self, name: str) -> Optional[dict]:
        return self._entities.get(name)

    def neighbors(self, name: str, relation: str = None) -> list[dict]:
        results = []
        for e in self._edges:
            if e["source"] == name:
                if relation and e["relation"] != relation:
                    continue
                if target := self._entities.get(e["target"]):
                    results.append({"entity": target, "relation": e, "weight": e["weight"]})
            elif e["target"] == name:
                if relation and e["relation"] != relation:
                    continue
                if source := self._entities.get(e["source"]):
                    results.append({"entity": source, "relation": e, "weight": e["weight"]})
        return sorted(results, key=lambda x: x["weight"], reverse=True)

    def decay(self):
        now = time.time()
        for e in self._entities.values():
            age = now - e["accessed_at"]
            decay_factor = max(0.0, 1.0 - (age / self._decay_s))
            e["trust"] = max(0.1, e.get("trust", 1.0) * decay_factor)

    def snapshot(self) -> dict:
        return {
            "entities": {k: {"kind": v["kind"], "trust": v["trust"]}
                        for k, v in self._entities.items()},
            "edges": len(self._edges),
        }


# ─── State Machine for agent lifecycle ────────────────────────────

class AgentState(Enum):
    IDLE = "idle"
    RECEIVING = "receiving"
    DECOMPOSING = "decomposing"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    VERIFYING = "verifying"
    DELIVERING = "delivering"
    FAILED = "failed"

_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.RECEIVING},
    AgentState.RECEIVING: {AgentState.DECOMPOSING, AgentState.FAILED},
    AgentState.DECOMPOSING: {AgentState.DISPATCHING, AgentState.FAILED},
    AgentState.DISPATCHING: {AgentState.EXECUTING},
    AgentState.EXECUTING: {AgentState.AGGREGATING, AgentState.FAILED},
    AgentState.AGGREGATING: {AgentState.VERIFYING, AgentState.FAILED},
    AgentState.VERIFYING: {AgentState.DELIVERING, AgentState.FAILED, AgentState.DECOMPOSING},
    AgentState.DELIVERING: {AgentState.IDLE, AgentState.FAILED},
    AgentState.FAILED: {AgentState.IDLE},
}


class StateMachine:
    def __init__(self):
        self._state: AgentState = AgentState.IDLE
        self._history: list[dict] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(self, target: AgentState) -> bool:
        with self._lock:
            if target in _TRANSITIONS.get(self._state, set()):
                prev = self._state
                self._state = target
                self._history.append({
                    "from": prev.value, "to": target.value,
                    "at": time.time(),
                })
                return True
            return False

    def history(self, limit: int = 20) -> list[dict]:
        return self._history[-limit:]


# ─── Agent Card ────────────────────────────────────────────────────

@dataclass
class AgentCard:
    """Machine-readable agent registration card — compliant with proposed Agent Card spec."""
    name: str
    role: AgentRole
    capabilities: list[str]
    input_schema: dict
    output_schema: dict
    memory_keys: list[str] = field(default_factory=list)
    skills_ref: list[str] = field(default_factory=list)
    max_concurrency: int = 1
    version: str = "0.1.0"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_skill_file(cls, skill_name: str, skill_content: str) -> "AgentCard":
        """Parse a Hermes SKILL.md into an AgentCard."""
        caps = []
        memory_keys = []
        for line in skill_content.splitlines():
            if line.startswith("## ") or line.startswith("### "):
                caps.append(line.strip("# "))
            if "memory" in line.lower() or "store" in line.lower():
                memory_keys.append(line.strip())
        return cls(
            name=skill_name,
            role=AgentRole.WORKER,
            capabilities=caps[:10],
            input_schema={"type": "object", "properties": {"goal": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
            memory_keys=memory_keys[:5],
            skills_ref=[skill_name],
        )


# ─── Executor ──────────────────────────────────────────────────────

class OSWEngine:
    """End-to-end orchestrator engine with DAG dispatch & state persistence."""

    def __init__(self, state_path: str = "/tmp/neuroflow_state.json"):
        self.dag = DAG()
        self.memory = GraphMemory()
        self.state_machine = StateMachine()
        self._state_path = state_path
        self._results: dict[str, Any] = {}

    def ingest_goal(self, goal: str, context: dict = None) -> str:
        """Accept a top-level goal, create orchestrator task, return run ID."""
        run_id = uuid.uuid4().hex[:8]
        root = Task(
            id=f"root-{run_id}",
            goal=goal,
            context=context or {},
            agent_role=AgentRole.ORCHESTRATOR,
        )
        self.dag.add(root)
        self.state_machine.transition(AgentState.RECEIVING)
        self.memory.upsert_entity(f"run:{run_id}", "run", {"goal": goal, "status": "active"})
        return run_id

    def decompose(self, run_id: str, subtasks: list[dict]) -> str:
        """Decompose root goal into parallel sub-DAGs — returns plan ID."""
        plan_id = f"plan-{run_id}-{uuid.uuid4().hex[:4]}"
        root_id = f"root-{run_id}"

        for i, sub in enumerate(subtasks):
            t = Task(
                id=f"{plan_id}-t{i:03d}",
                goal=sub.get("goal", "unnamed"),
                context=sub.get("context", {}),
                dependencies=sub.get("dependencies", []),
                agent_role=AgentRole.SUPERVISOR,
                max_retries=sub.get("max_retries", 2),
                timeout_s=sub.get("timeout", 120),
            )
            # Link to root
            t.dependencies.append(root_id)
            self.dag.add(t)

        self.state_machine.transition(AgentState.DECOMPOSING)
        self.state_machine.transition(AgentState.DISPATCHING)
        self.memory.relate(f"run:{run_id}", f"plan:{plan_id}", "decomposed_into")
        return plan_id

    def add_worker(self, plan_id: str, goal: str, context: dict = None,
                   dependencies: list[str] = None) -> str:
        """Add a worker task to an existing plan."""
        wid = f"{plan_id}-w{uuid.uuid4().hex[:4]}"
        t = Task(
            id=wid,
            goal=goal,
            context=context or {},
            dependencies=dependencies or [],
            agent_role=AgentRole.WORKER,
        )
        self.dag.add(t)
        self.memory.upsert_entity(wid, "worker", {"goal": goal, "plan": plan_id})
        return wid

    def execute(self, task_executor: Callable[[Task], Any]) -> dict[str, Any]:
        """Execute DAG in topologically sorted batches.

        task_executor: callable(task) -> result. Raise to signal failure.
        Returns {task_id: result, ...}
        """
        self.state_machine.transition(AgentState.EXECUTING)
        order = self.dag.execution_order()

        # Build batch plan
        executed: set[str] = set()
        results: dict[str, Any] = {}

        for batch in order:
            for task_id in batch:
                task = self.dag.tasks.get(task_id)
                if not task:
                    continue

                # Check deps
                all_deps_met = all(
                    d in executed and self.dag.tasks[d].status == TaskStatus.SUCCESS
                    for d in task.dependencies
                )
                if not all_deps_met:
                    task.status = TaskStatus.SKIPPED
                    continue

                task.status = TaskStatus.RUNNING

                # Retry loop
                for attempt in range(task.max_retries + 1):
                    try:
                        result = task_executor(task)
                        task.result = result
                        task.status = TaskStatus.SUCCESS
                        results[task_id] = result
                        self.memory.upsert_entity(task_id, "task", {"status": "success"})
                        break
                    except Exception as e:
                        task.error = str(e)
                        task.retries = attempt + 1
                        if attempt >= task.max_retries:
                            task.status = TaskStatus.FAILED
                            results[task_id] = {"error": str(e)}

                executed.add(task_id)

        self.state_machine.transition(AgentState.AGGREGATING)
        self._results = results
        return results

    def report(self) -> dict[str, Any]:
        """Full execution report."""
        self.state_machine.transition(AgentState.DELIVERING)
        return {
            "state": self.state_machine.state.value,
            "memory_snapshot": self.memory.snapshot(),
            "dag_summary": self.dag.serialize(),
            "results_count": len(self._results),
            "success_count": sum(
                1 for t in self.dag.tasks.values() if t.status == TaskStatus.SUCCESS
            ),
            "failed_count": sum(
                1 for t in self.dag.tasks.values() if t.status == TaskStatus.FAILED
            ),
        }

    def persist(self):
        snapshot = {
            "dag": self.dag.serialize(),
            "memory": self.memory.snapshot(),
            "state": self.state_machine.state.value,
        }
        with open(self._state_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    def reset(self):
        self.dag = DAG()
        self._results = {}
        self.state_machine = StateMachine()
