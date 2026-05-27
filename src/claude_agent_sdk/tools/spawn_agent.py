"""SpawnAgent tool — allows the model to spawn sub-runners during execution."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SPAWN_AGENT_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "SpawnAgent",
        "description": (
            "Spawn a sub-agent to work on a subtask. The sub-agent runs autonomously "
            "with its own model, tools, and working directory. Returns the sub-agent's "
            "final output when complete. Use this to delegate work or parallelize tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the sub-agent to complete.",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use for the sub-agent (e.g. 'qwen/qwen3-max'). Defaults to parent's model.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the sub-agent. Defaults to parent's cwd.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the sub-agent (default: 20).",
                },
            },
            "required": ["task"],
        },
    },
}


async def execute_spawn_agent(
    params: dict[str, Any],
    *,
    parent_base_url: str = "http://127.0.0.1:8016",
    parent_api_key: str = "",
    parent_model: str = "qwen/qwen3-max",
    parent_cwd: str = ".",
    parent_checkpoint_dir: str | None = None,
) -> str:
    """Execute the SpawnAgent tool by creating and running a child AutonomousRunner.

    This import is deferred to avoid circular imports (runner imports tools, tools import runner).
    """
    task = params.get("task")
    if not task:
        return "[ERROR] SpawnAgent requires 'task' parameter"

    model = params.get("model", parent_model)
    cwd = params.get("cwd", parent_cwd)
    max_turns = int(params.get("max_turns", 20))

    # Deferred import to avoid circular dependency
    from claude_agent_sdk.runner import AutonomousRunner

    checkpoint_sub = None
    if parent_checkpoint_dir:
        checkpoint_sub = f"{parent_checkpoint_dir}/sub"

    logger.info(
        "Spawning sub-agent: model=%s, max_turns=%d, task=%.100s",
        model,
        max_turns,
        task,
    )

    runner = AutonomousRunner(
        base_url=parent_base_url,
        api_key=parent_api_key,
        model=model,
        max_turns=max_turns,
        checkpoint_dir=checkpoint_sub,
    )

    try:
        result = await runner.run(task=task, cwd=cwd)

        summary = (
            f"Sub-agent completed:\n"
            f"  Success: {result.success}\n"
            f"  Turns: {getattr(result, 'turns', getattr(result, 'turn_count', 0))}\n"
            f"  Tool calls: {result.total_tool_calls}\n"
            f"  Stopped reason: {result.stopped_reason}\n"
            f"  Duration: {result.elapsed_seconds:.1f}s\n\n"
            f"Output:\n{result.final_text}"
        )
        return summary

    except Exception as exc:
        logger.warning("Sub-agent failed: %s", exc, exc_info=True)
        return f"[ERROR] Sub-agent failed: {exc}"
