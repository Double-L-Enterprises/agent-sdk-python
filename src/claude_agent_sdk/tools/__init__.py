"""Built-in tool registry and dispatcher for AutonomousRunner."""

from __future__ import annotations

import logging
from typing import Any

from .bash import BASH_TOOL_DEF, execute_bash
from .file_ops import FILE_OPS_TOOL_DEFS, execute_file_op
from .search import SEARCH_TOOL_DEF, execute_search
from .spawn_agent import SPAWN_AGENT_TOOL_DEF, execute_spawn_agent

logger = logging.getLogger(__name__)

# All built-in tool definitions in OpenAI function-calling format
_ALL_TOOL_DEFS: list[dict[str, Any]] = [
    *FILE_OPS_TOOL_DEFS,
    BASH_TOOL_DEF,
    SEARCH_TOOL_DEF,
    SPAWN_AGENT_TOOL_DEF,
]

# Map from tool name -> async executor
_EXECUTORS: dict[str, Any] = {
    "Read": execute_file_op,
    "Write": execute_file_op,
    "Edit": execute_file_op,
    "Glob": execute_file_op,
    "Bash": execute_bash,
    "Grep": execute_search,
    "SpawnAgent": execute_spawn_agent,
}


def default_tools() -> list[dict[str, Any]]:
    """Return the full list of built-in tool definitions (OpenAI format)."""
    return list(_ALL_TOOL_DEFS)


async def dispatch_tool(name: str, params: dict[str, Any], cwd: str | None = None, **kwargs: Any) -> str:
    """Dispatch a tool call by name and return the string result.

    Args:
        name: Tool name (must match a registered tool).
        params: Tool input parameters as parsed from the model's tool_call.
        cwd: Working directory override (passed to tools that support it).
        **kwargs: Extra kwargs for tools like SpawnAgent (parent context).

    Returns:
        Tool output as a string. On error, returns an error message prefixed
        with "[ERROR]".
    """
    executor = _EXECUTORS.get(name)
    if executor is None:
        return f"[ERROR] Unknown tool: {name!r}. Available: {sorted(_EXECUTORS)}"

    try:
        if name == "SpawnAgent":
            result = await executor(params, **kwargs)
        elif cwd is not None and name == "Bash":
            result = await executor(params, cwd=cwd)
        else:
            result = await executor(params)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool %s raised: %s", name, exc, exc_info=True)
        return f"[ERROR] Tool {name!r} failed: {exc}"


__all__ = [
    "default_tools",
    "dispatch_tool",
    "BASH_TOOL_DEF",
    "FILE_OPS_TOOL_DEFS",
    "SEARCH_TOOL_DEF",
    "SPAWN_AGENT_TOOL_DEF",
]
