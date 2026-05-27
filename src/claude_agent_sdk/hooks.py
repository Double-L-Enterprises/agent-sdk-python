"""HookRegistry — lifecycle event system for AutonomousRunner.

Hooks allow intercepting and modifying runner behavior at key points:
- pre_tool_use: modify params or block tool calls (return None to block)
- post_tool_use: inspect/modify results after tool execution
- pre_turn: inject context before each LLM call
- post_turn: inspect responses after each LLM call
- on_stall: custom handling when model pauses without tool calls
- on_stall_timeout: fires when stall diagnosis gives up (tier 4)
- on_complete: post-processing after run finishes
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

HookCallback = Callable[..., Awaitable[Any]]


class HookRegistry:
    """Registry for lifecycle event hooks.

    Usage:
        hooks = HookRegistry()

        @hooks.hook("pre_tool_use")
        async def check_tool(tool_name: str, params: dict, **kwargs):
            if tool_name == "Bash" and "rm -rf" in params.get("command", ""):
                return None  # Block dangerous commands
            return params  # Allow (optionally modified)

        @hooks.hook("on_complete")
        async def log_result(result, **kwargs):
            print(f"Done: {result.success} in {result.turns} turns")
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookCallback]] = {}

    def register(self, event: str, callback: HookCallback) -> None:
        """Register a callback for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
        logger.debug("Hook registered: %s -> %s", event, callback.__name__)

    def hook(self, event: str) -> Callable[[HookCallback], HookCallback]:
        """Decorator to register a hook callback.

        @hooks.hook("pre_tool_use")
        async def my_hook(tool_name, params, **kwargs):
            return params
        """
        def decorator(fn: HookCallback) -> HookCallback:
            self.register(event, fn)
            return fn
        return decorator

    async def fire(self, event: str, **kwargs: Any) -> Any:
        """Fire all registered hooks for an event.

        For pre_tool_use: if any hook returns None, returns None (tool blocked).
        For pre_tool_use: hooks can return modified params (passed to next hook).
        For other events: return value of the last hook, or kwargs if no hooks.

        A failing hook logs the error but does not crash the runner.
        """
        callbacks = self._hooks.get(event, [])
        if not callbacks:
            return kwargs.get("params", kwargs)

        result = kwargs.get("params", kwargs)

        for cb in callbacks:
            try:
                ret = await cb(**kwargs)

                if event == "pre_tool_use":
                    if ret is None:
                        logger.info("Hook %s blocked tool call", cb.__name__)
                        return None
                    if isinstance(ret, dict):
                        result = ret
                        kwargs["params"] = ret
                else:
                    if ret is not None:
                        result = ret

            except Exception:
                logger.warning(
                    "Hook %s raised on event %s — continuing",
                    cb.__name__, event, exc_info=True,
                )

        return result

    def clear(self, event: str | None = None) -> None:
        """Remove all hooks, or hooks for a specific event."""
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)

    @property
    def events(self) -> list[str]:
        """List all events with registered hooks."""
        return list(self._hooks.keys())

    def __repr__(self) -> str:
        counts = {e: len(cbs) for e, cbs in self._hooks.items()}
        return f"HookRegistry({counts})"


__all__ = ["HookRegistry", "HookCallback"]
