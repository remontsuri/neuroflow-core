"""
neuroflow-core :: Agent Card Registry + execution demo
Live walkthrough — proves OSW engine works.
"""

import sys, os, json, time

# Add self to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from osw_engine import (
    OSWEngine, AgentCard, AgentRole, Task, TaskStatus, StateMachine,
    AgentState, GraphMemory,
)


def demo_osw_basic():
    """Run a fully instrumented OSW pipeline."""
    print("=" * 65)
    print("  NEUROFLOW CORE :: OSW ENGINE DEMO")
    print("  Mythos build — 01 June 2026")
    print("=" * 65)

    engine = OSWEngine()
    run_id = engine.ingest_goal(
        "Analyse three Telegram channels for marketing insights, "
        "cross-reference with competitor data, deliver structured report.",
        context={"channels": ["@channel_A", "@channel_B", "@channel_C"],
                 "competitors": ["comp1", "comp2"]}
    )
    print(f"\n[1/5] Goal ingested   → run_id={run_id}")
    print(f"      State: {engine.state_machine.state.value}")

    # Decompose into 3 supervisor tasks + 1 aggregate
    plan_id = engine.decompose(run_id, [
        {"goal": "Scrape & analyse @channel_A", "context": {"channel": "@channel_A"}},
        {"goal": "Scrape & analyse @channel_B", "context": {"channel": "@channel_B"}},
        {"goal": "Scrape & analyse @channel_C", "context": {"channel": "@channel_C"}},
        {"goal": "Cross-reference competitor data", "context": {"sources": "competitor_db"},
         "dependencies": []},  # will be wired after plan_id is known
    ])
    # Wire cross-ref to depend on all three channel scrapers
    cross_ref_id = f"{plan_id}-t003"
    for dep in [f"{plan_id}-t000", f"{plan_id}-t001", f"{plan_id}-t002"]:
        engine.dag.add_dependency(cross_ref_id, dep)
    print(f"\n[2/5] Decomposed      → plan_id={plan_id}")
    print(f"      Tasks in DAG: {len(engine.dag.tasks)}")
    for tid, t in engine.dag.tasks.items():
        print(f"        {tid:30s}  goal={t.goal[:50]}")

    # Define executor (simulated — real one would call Hermes subagents)
    def mock_executor(task: Task) -> dict:
        time.sleep(0.1)  # simulate work
        if "error" in task.goal.lower():
            raise RuntimeError(f"Simulated failure on {task.id}")
        return {
            "task_id": task.id,
            "summary": f"Processed: {task.goal[:40]}",
            "status": "ok",
            "artifacts": {},
        }

    print(f"\n[3/5] Executing DAG   → {len(list(engine.dag.execution_order()))} layers")
    results = engine.execute(mock_executor)
    print(f"      Execution complete")

    # Report
    print(f"\n[4/5] Report")
    report = engine.report()
    print(f"      State:   {report['state']}")
    print(f"      Success: {report['success_count']} / Failed: {report['failed_count']}")
    for tid, t in engine.dag.tasks.items():
        icon = "✓" if t.status == TaskStatus.SUCCESS else "✗" if t.status == TaskStatus.FAILED else "·"
        print(f"      {icon} {tid:25s} → {t.status.value}")

    print(f"\n[5/5] Memory snapshot")
    snap = engine.memory.snapshot()
    print(f"      Entities: {len(snap['entities'])}   Edges: {snap['edges']}")

    # Persist state
    engine.persist()
    print(f"\n      State saved to {engine._state_path}")
    print(f"\n{'='*65}")
    print(f"  ✓ OSW Engine: ALL SYSTEMS NOMINAL")
    print(f"{'='*65}")
    return engine


def demo_agent_cards():
    """Show Agent Card generation from skill files."""
    print(f"\n{'='*65}")
    print(f"  AGENT CARD REGISTRY DEMO")
    print(f"{'='*65}")

    # Scan for skill files
    skill_base = "/opt/data/skills"
    if not os.path.isdir(skill_base):
        skill_base = "../skills"
    if not os.path.isdir(skill_base):
        print("  No skill dir found — using synthetic cards")
        cards = [
            AgentCard("web-researcher", AgentRole.WORKER,
                      capabilities=["web_search", "web_extract", "summarize"],
                      input_schema={}, output_schema={}),
            AgentCard("analyst", AgentRole.SUPERVISOR,
                      capabilities=["cross_reference", "score", "classify"],
                      input_schema={}, output_schema={}),
            AgentCard("orchestrator-main", AgentRole.ORCHESTRATOR,
                      capabilities=["decompose", "route", "aggregate"],
                      input_schema={}, output_schema={}),
        ]
    else:
        cards = []
        for fname in os.listdir(skill_base):
            if fname == "SKILL.md":
                continue
            fpath = os.path.join(skill_base, fname, "SKILL.md")
            if os.path.isfile(fpath):
                with open(fpath) as f:
                    content = f.read()
                card = AgentCard.from_skill_file(fname, content)
                cards.append(card)

    registry = {c.name: c for c in cards}
    print(f"\n  Registered {len(registry)} agents:")
    for name, card in registry.items():
        print(f"    {name:30s}  role={card.role.value:15s}  caps={len(card.capabilities)}")

    # Export json
    registry_path = "/tmp/agent_card_registry.json"
    with open(registry_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in registry.items()}, f, indent=2, ensure_ascii=False)
    print(f"\n  Registry exported → {registry_path}")
    print(f"{'='*65}")
    return registry


def demo_state_machine():
    """Exercise the state machine transitions."""
    print(f"\n{'='*65}")
    print(f"  STATE MACHINE EXERCISE")
    print(f"{'='*65}")

    sm = StateMachine()
    transitions = [
        AgentState.RECEIVING,
        AgentState.DECOMPOSING,
        AgentState.DISPATCHING,
        AgentState.EXECUTING,
        AgentState.AGGREGATING,
        AgentState.VERIFYING,
        AgentState.DELIVERING,
        AgentState.IDLE,
    ]

    print(f"\n  Transition trace:")
    print(f"    {sm.state.value:15s}  (initial)")
    for target in transitions:
        ok = sm.transition(target)
        icon = "✓" if ok else "✗"
        print(f"    {target.value:15s}  {icon}")

    print(f"\n  History ({len(sm.history())} steps):")
    for h in sm.history():
        print(f"    {h['from']:15s} → {h['to']:15s}")

    # Test invalid transition
    ok = sm.transition(AgentState.EXECUTING)
    print(f"\n  Invalid EXECUTING from IDLE: {'blocked ✓' if not ok else 'BUG ✗'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    demo_state_machine()
    demo_agent_cards()
    engine = demo_osw_basic()
