"""Live multi-agent team test using Agent SDK.

Tests TeamManager with 2 agents against router-proxy at localhost:8016.
Run: cd agent-sdk-python-frozen && python -m tests.test_live_team

Created: 2026-05-27 23:15 CST
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_live_team")


async def main() -> None:
    """Run multi-agent team test."""
    from claude_agent_sdk.team_manager import TeamManager, AgentState
    from claude_agent_sdk.cost_tracker import CostTracker, BudgetPolicy
    from claude_agent_sdk.message_bus import MessageBus
    from claude_agent_sdk.agent_memory import AgentMemory
    from claude_agent_sdk.observability import global_history, global_metrics

    print("=" * 60)
    print("LIVE MULTI-AGENT TEAM TEST")
    print("=" * 60)

    # ---- Setup ----
    BASE_URL = "http://127.0.0.1:8016"
    API_KEY = "sk-litellm"

    # 1. CostTracker with $1.00 budget
    policy = BudgetPolicy(max_cost_per_run=1.00, auto_switch_model="qwen/qwen3-max")
    cost_tracker = CostTracker(policy=policy)
    print("[OK] CostTracker created with $1.00 per-run budget")

    # 2. MessageBus
    bus = MessageBus(team_id="live-test-team")
    print("[OK] MessageBus created")

    # 3. AgentMemory
    memory = AgentMemory(output_dir="/tmp/agent-team-test/")
    memory.team_set("test_start", {"timestamp": time.time(), "test": "live_team"})
    print("[OK] AgentMemory created at /tmp/agent-team-test/")

    # 4. TeamManager
    team = TeamManager(
        team_id="live-test-team",
        message_bus=bus,
        cost_tracker=cost_tracker,
    )
    print("[OK] TeamManager created")

    # 5. Add agents
    team.add_agent(
        name="architect",
        model="qwen/qwen3-max",
        task=(
            "Design a Python function that calculates fibonacci numbers efficiently. "
            "Describe the approach, time complexity, and provide the implementation. "
            "When done, say task complete."
        ),
        base_url=BASE_URL,
        api_key=API_KEY,
        max_turns=5,
        timeout=60.0,
    )
    print("[OK] Agent 'architect' added (qwen/qwen3-max)")

    team.add_agent(
        name="reviewer",
        model="nvidia/llama-3.3-70b",
        task=(
            "Review the following fibonacci implementation approach: use memoization "
            "with a dictionary cache for O(n) time complexity. Suggest improvements "
            "or alternative approaches. When done, say task complete."
        ),
        base_url=BASE_URL,
        api_key=API_KEY,
        max_turns=5,
        timeout=60.0,
    )
    print("[OK] Agent 'reviewer' added (nvidia/llama-3.3-70b)")

    # 6. Start all agents (timeout 120s)
    print()
    print("-" * 40)
    print("Starting all agents...")
    print("-" * 40)
    start_time = time.monotonic()

    try:
        results = await team.start_all(timeout=120.0)
    except Exception as e:
        print(f"[ERROR] team.start_all() failed: {e}")
        results = {}

    elapsed = time.monotonic() - start_time
    print(f"Team completed in {elapsed:.1f}s")

    # ---- Print Results ----
    print()
    print("=" * 60)
    print("AGENT RESULTS")
    print("=" * 60)

    for name, agent_result in results.items():
        print(f"\n--- Agent: {name} ---")
        print(f"  State: {agent_result.state.value}")
        print(f"  Elapsed: {agent_result.elapsed:.1f}s")
        if agent_result.error:
            print(f"  Error: {agent_result.error}")
        if agent_result.result:
            r = agent_result.result
            print(f"  Turns: {r.turn_count}")
            print(f"  Tool calls: {r.total_tool_calls}")
            print(f"  Stopped reason: {r.stopped_reason}")
            print(f"  Success: {r.success}")
            # Print final text (truncated)
            final = r.final_text[:800] if r.final_text else "(empty)"
            print(f"  Final output (first 800 chars):")
            print(f"    {final}")

    # ---- Cost Tracker Summary ----
    print()
    print("=" * 60)
    print("COST TRACKER SUMMARY")
    print("=" * 60)
    totals = cost_tracker.total_cost()
    print(f"  Total cost: ${totals['cost_usd']:.6f}")
    print(f"  Total turns: {totals['turn_count']}")
    print(f"  Total tokens: {totals['input_tokens']} in / {totals['output_tokens']} out")
    agent_costs = cost_tracker.cost_by_agent()
    for agent_name, info in agent_costs.items():
        print(f"  {agent_name}: ${info['cost_usd']:.6f} "
              f"({info['input_tokens']} in / {info['output_tokens']} out)")

    # ---- MessageBus Messages ----
    print()
    print("=" * 60)
    print("MESSAGE BUS MESSAGES")
    print("=" * 60)
    messages = bus.get_all()
    if messages:
        for msg in messages:
            content_preview = str(msg.content)[:200]
            print(f"  {msg.from_agent} -> {msg.to_agent}: "
                  f"{content_preview}")
    else:
        print("  (no messages exchanged)")

    # ---- Run History ----
    print()
    print("=" * 60)
    print("RUN HISTORY (from observability)")
    print("=" * 60)
    runs = global_history.list_runs(team_id="live-test-team", limit=10)
    print(f"  Total runs recorded: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"  Run {i+1}: team={run.get('team_id')}, result={run.get('result')}, "
              f"duration={run.get('duration_seconds', 0):.1f}s")
        for agent in run.get("agents", []):
            print(f"    Agent: {agent['name']} model={agent['model']} "
                  f"status={agent['status']} turns={agent['turns']}")

    # ---- Metrics (Prometheus format) ----
    print()
    print("=" * 60)
    print("METRICS (Prometheus text format)")
    print("=" * 60)
    metrics_text = global_metrics.expose_metrics()
    # Print only non-empty, non-comment lines
    for line in metrics_text.strip().split("\n"):
        if line and not line.startswith("#"):
            print(f"  {line}")

    # ---- Memory ----
    print()
    print("=" * 60)
    print("AGENT MEMORY")
    print("=" * 60)
    all_team_data = memory.team_get_all()
    print(f"  Team keys: {list(all_team_data.keys())}")
    print(f"  test_start: {memory.team_get('test_start')}")

    # Save final state
    memory.team_set("test_results", {
        "elapsed": elapsed,
        "agents": {name: ar.state.value for name, ar in results.items()},
        "cost": totals["cost_usd"],
    })

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
