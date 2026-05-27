"""AutonomousRunner — agentic loop with LiteLLMHTTPTransport.

The core insight: don't stop when the model outputs without tool calls.
Instead, detect whether it actually finished or just paused, and keep going
if it paused. This solves the "agent stops at 50%" problem.

Design: Works directly with LiteLLMHTTPTransport's native API
(add_user_message, add_tool_result, do_turn) rather than the SDK subprocess
message stream. The transport manages all conversation history internally.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._internal.transport.litellm_http import LiteLLMHTTPTransport
from .deep_agent_client import DeepAgentClient
from .hooks import HookRegistry
from .prompts import build_system_prompt
from .tools import default_tools, dispatch_tool

logger = logging.getLogger(__name__)

# Completion markers that indicate the model is truly done
_COMPLETION_MARKERS = frozenset(
    {
        "done",
        "complete",
        "completed",
        "finished",
        "task complete",
        "task done",
        "task finished",
        "success",
        "all done",
        "i'm done",
        "i am done",
        "work is complete",
        "work is done",
        "successfully completed",
        "successfully finished",
        "no further",
        "nothing left",
        "that's all",
        "that is all",
        "implementation complete",
        "changes complete",
    }
)


@dataclass
class RunResult:
    """Result of an autonomous task run."""

    messages: list[dict[str, Any]]
    final_text: str
    turn_count: int
    total_tool_calls: int = 0
    success: bool = False
    stopped_reason: str = "unknown"
    elapsed_seconds: float = 0.0
    checkpoint_path: str | None = None
    model_history: list[dict[str, Any]] = field(default_factory=list)
    error_message: str = ""

    @property
    def turns(self) -> int:
        """Alias for turn_count."""
        return self.turn_count


class AutonomousRunner:
    """Agentic loop that runs tasks autonomously until done.

    Works directly with LiteLLMHTTPTransport — does NOT use the SDK subprocess
    transport or the query() function.

    Usage Pattern A — task passed to run() (preferred):
        runner = AutonomousRunner(
            base_url="http://127.0.0.1:8016",
            model="qwen/qwen3-max",
            max_turns=50,
            escalation_model="claude-sonnet-4-6",
        )
        result = await runner.run(
            task="Read README.md and add an Installation section",
            cwd="/path/to/project",
            tools=default_tools(),
        )

    Usage Pattern B — with hooks (Phase 4):
        hooks = HookRegistry()
        @hooks.hook("pre_tool_use")
        async def check_tool(tool_name: str, params: dict, **kwargs):
            if tool_name == "Bash" and "rm" in params.get("command", ""):
                return None  # Block dangerous commands
            return params

        runner = AutonomousRunner(hooks=hooks)
        result = await runner.run(task="...", cwd="...")
    """

    def __init__(
        self,
        task: str | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str = "qwen/qwen3-max",
        base_url: str = "http://100.102.119.55:3002",
        api_key: str = "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        max_turns: int = 50,
        timeout: float = 120.0,
        system_prompt: str | None = None,
        checkpoint_dir: str | None = None,
        escalation_model: str | None = None,
        hooks: HookRegistry | None = None,
        pipeline_mode: str = "auto",
        deep_agent_url: str | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            task: The task/goal (optional; can be passed to run() instead).
            tools: Default tool definitions.
            model: Model name at the LiteLLM endpoint.
            base_url: LiteLLM API base URL.
            api_key: API key for LiteLLM.
            max_turns: Hard limit on loop iterations.
            timeout: Per-request HTTP timeout in seconds.
            system_prompt: Override the default system prompt.
            checkpoint_dir: If set, checkpoints are saved after each tool turn.
            escalation_model: If set, auto-switch to this model at stall tier 3.
            hooks: HookRegistry for lifecycle events.
            pipeline_mode: "auto" (keyword-detect), "deterministic" (always Deep Agent),
                or "autonomous" (always local AutonomousRunner).
            deep_agent_url: Base URL for Deep Agent API. Defaults to http://127.0.0.1:8040.
        """
        self._task = task
        self._default_tools = tools
        self._model = model
        self._escalation_model = escalation_model
        self._base_url = base_url
        self._api_key = api_key
        self._max_turns = max_turns
        self._timeout = timeout
        self._custom_system_prompt = system_prompt
        self._checkpoint_dir = checkpoint_dir
        self._current_turn = 0
        self._model_history: list[dict[str, Any]] = []
        self._hooks = hooks or HookRegistry()
        self.pipeline_mode = pipeline_mode
        self._deep_agent_url = deep_agent_url or "http://127.0.0.1:8040"

    def switch_model(self, new_model: str) -> None:
        """Switch the model mid-task. Conversation history is preserved.

        Args:
            new_model: The new model name to switch to.
        """
        old_model = self._model
        self._model = new_model
        self._model_history.append(
            {
                "from": old_model,
                "to": new_model,
                "at_turn": self._current_turn,
            }
        )
        logger.info(
            "Model switched: %s → %s at turn %d",
            old_model,
            new_model,
            self._current_turn,
        )

    def _should_use_deep_agent(self, task: str) -> bool:
        """Determine whether to route this task to Deep Agent.

        - "deterministic" → always True
        - "autonomous"    → always False
        - "auto"          → keyword-based detection via DeepAgentClient.should_use_pipeline
        """
        if self.pipeline_mode == "deterministic":
            return True
        if self.pipeline_mode == "autonomous":
            return False
        # "auto" — keyword detection
        return DeepAgentClient.should_use_pipeline(task)

    async def _delegate_to_deep_agent(
        self, task: str, project_id: str, output_dir: str
    ) -> "RunResult":
        """Delegate this task to Deep Agent's pipeline API.

        Falls back to local execution if Deep Agent is unreachable.
        """
        start_time = time.monotonic()
        client = DeepAgentClient(base_url=self._deep_agent_url)

        try:
            if not await client.health():
                logger.warning(
                    "[WARN] Deep Agent unreachable at %s — falling back to AutonomousRunner",
                    self._deep_agent_url,
                )
                return RunResult(
                    messages=[],
                    final_text=(
                        f"[WARN] Deep Agent unreachable at {self._deep_agent_url}. "
                        "Falling back to local AutonomousRunner."
                    ),
                    turn_count=0,
                    total_tool_calls=0,
                    success=False,
                    stopped_reason="deep_agent_unreachable",
                    elapsed_seconds=time.monotonic() - start_time,
                    model_history=self._model_history,
                    error_message=f"Deep Agent unreachable at {self._deep_agent_url}",
                )

            run_id = await client.start_pipeline(
                task=task,
                project_id=project_id,
                output_dir=output_dir,
                model=self._model,
            )
            logger.info("Delegated to Deep Agent: run_id=%s", run_id)

            final_status = await client.wait_for_completion(run_id)

            elapsed = time.monotonic() - start_time
            success = final_status.status == "completed"
            stopped_reason = f"deep_agent_{final_status.status}"
            summary = final_status.result_summary or f"Deep Agent pipeline {final_status.status}."
            if final_status.output_files:
                summary += f" Output files: {', '.join(final_status.output_files)}"

            logger.info(
                "Deep Agent pipeline finished: status=%s run_id=%s elapsed=%.1fs",
                final_status.status,
                run_id,
                elapsed,
            )

            return RunResult(
                messages=[{"role": "assistant", "content": summary}],
                final_text=summary,
                turn_count=0,
                total_tool_calls=0,
                success=success,
                stopped_reason=stopped_reason,
                elapsed_seconds=elapsed,
                model_history=self._model_history,
                error_message=final_status.error or "",
            )

        finally:
            await client.close()

    async def run(
        self,
        task: str | None = None,
        *,
        cwd: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> RunResult:
        """Run the agentic loop until completion or safety limit hit.

        Args:
            task: The natural-language task. If omitted, uses __init__ task.
            cwd: Working directory for Bash tool calls.
            tools: Tool definitions (OpenAI format).

        Returns:
            RunResult with full conversation history, final text, and stats.
        """
        actual_task = task or self._task
        if not actual_task:
            raise ValueError("task must be provided either to __init__ or to run()")

        # ── Pipeline routing: delegate to Deep Agent if appropriate ────────
        if self._should_use_deep_agent(actual_task):
            try:
                return await self._delegate_to_deep_agent(
                    actual_task,
                    project_id="default",
                    output_dir=cwd or os.getcwd(),
                )
            except Exception as e:
                logger.warning(
                    "Deep Agent delegation failed, falling back to local: %s", e
                )
                # Fall through to local execution

        actual_tools = tools or self._default_tools or default_tools()
        start_time = time.monotonic()

        # Build goal-anchored system prompt
        system_prompt = self._custom_system_prompt or build_system_prompt(actual_task)

        transport = LiteLLMHTTPTransport(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            timeout=self._timeout,
            system_prompt=system_prompt,
        )
        transport.set_tools(actual_tools)

        try:
            await transport.connect()
        except Exception as exc:
            logger.error("Failed to connect to LiteLLM at %s: %s", self._base_url, exc)
            error_msg = f"Could not connect to LiteLLM: {exc}"
            return RunResult(
                messages=[],
                final_text=f"[ERROR] {error_msg}",
                turn_count=0,
                total_tool_calls=0,
                success=False,
                stopped_reason="error",
                elapsed_seconds=time.monotonic() - start_time,
                model_history=self._model_history,
                error_message=error_msg,
            )

        # Seed the conversation
        transport.add_user_message(actual_task)

        turn_count = 0
        total_tool_calls = 0
        final_text = ""
        stopped_reason = "max_turns"
        local_messages: list[dict[str, Any]] = [
            {"role": "user", "content": actual_task}
        ]
        error_message = ""

        try:
            consecutive_stalls = 0
            session_id = str(uuid.uuid4())
            checkpoint_path: str | None = None

            while turn_count < self._max_turns:
                turn_count += 1
                self._current_turn = turn_count
                logger.info(
                    "Turn %d/%d (consecutive_stalls=%d)",
                    turn_count,
                    self._max_turns,
                    consecutive_stalls,
                )

                # ── Hook: pre_turn ─────────────────────────────────────
                await self._hooks.fire(
                    "pre_turn",
                    turn=turn_count,
                    messages=transport._messages,
                )

                # ── Call the model ──────────────────────────────────────
                try:
                    turn_result = await transport.do_turn()
                except Exception as exc:
                    logger.error("LiteLLM call failed on turn %d: %s", turn_count, exc)
                    error_message = f"LiteLLM call failed: {exc}"
                    final_text = f"[ERROR] {error_message}"
                    stopped_reason = "error"
                    break

                tool_calls: list[dict[str, Any]] = turn_result.get("tool_calls", [])
                content: str = turn_result.get("text", "")

                if content:
                    final_text = content

                # Track assistant response in local history
                local_messages.append(
                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                )

                logger.debug(
                    "Turn %d: text=%d chars, tool_calls=%d",
                    turn_count,
                    len(content),
                    len(tool_calls),
                )

                # ── Hook: post_turn ────────────────────────────────────
                await self._hooks.fire(
                    "post_turn",
                    turn=turn_count,
                    response=turn_result,
                )

                # ── Case 1: Tool calls — execute and continue ───────────
                if tool_calls:
                    consecutive_stalls = 0  # model is actively working
                    total_tool_calls += len(tool_calls)
                    logger.info("Executing %d tool call(s)", len(tool_calls))

                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "unknown")
                        try:
                            params: dict[str, Any] = json.loads(
                                fn.get("arguments", "{}")
                            )
                        except json.JSONDecodeError:
                            params = {}

                        logger.info("  → %s(%r)", tool_name, params)

                        # ── Hook: pre_tool_use ─────────────────────────
                        hook_result = await self._hooks.fire(
                            "pre_tool_use",
                            tool_name=tool_name,
                            params=params,
                        )

                        # None return means blocked by hook
                        if hook_result is None:
                            logger.warning("Tool %s blocked by hook", tool_name)
                            tool_result = "[BLOCKED by hook]"
                            is_error = True
                        else:
                            # Hook may have modified params
                            tool_result = await dispatch_tool(
                                tool_name,
                                hook_result or params,
                                cwd=cwd,
                                parent_base_url=self._base_url,
                                parent_api_key=self._api_key,
                                parent_model=self._model,
                                parent_cwd=cwd or ".",
                                parent_checkpoint_dir=self._checkpoint_dir,
                            )
                            is_error = tool_result.startswith("[ERROR]")

                        logger.debug(
                            "    %s: %d chars, error=%s",
                            tool_name,
                            len(tool_result),
                            is_error,
                        )

                        # ── Hook: post_tool_use ────────────────────────
                        await self._hooks.fire(
                            "post_tool_use",
                            tool_name=tool_name,
                            params=params,
                            result=tool_result,
                            is_error=is_error,
                        )

                        transport.add_tool_result(
                            tc["id"], tool_result, is_error=is_error
                        )
                        local_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result,
                            }
                        )

                    # Save checkpoint after tool execution
                    checkpoint_path = await self._save_checkpoint(
                        transport,
                        turn_count,
                        actual_task,
                        cwd,
                        total_tool_calls,
                        session_id,
                    )
                    continue  # Back to top — model sees results next turn

                # ── Case 2: No tool calls — check completion ────────────
                if self._is_complete(content):
                    logger.info(
                        "Task complete (explicit marker) after %d turns, %d tool calls",
                        turn_count,
                        total_tool_calls,
                    )
                    stopped_reason = "completed"
                    break

                # ── Case 3: Paused — smart diagnostic stall handler ──────
                consecutive_stalls += 1
                logger.info(
                    "Model paused without completion (consecutive_stalls=%d)",
                    consecutive_stalls,
                )

                # ── Hook: on_stall ─────────────────────────────────────
                await self._hooks.fire(
                    "on_stall",
                    tier=consecutive_stalls,
                    stall_count=consecutive_stalls,
                )

                # Tier 1: Gentle nudge
                if consecutive_stalls == 1:
                    cont_msg = (
                        "You responded without using any tools. To make progress on this task, "
                        "you should use your available tools (Read, Write, Edit, Bash, Grep, Glob). "
                        "What's the next action you need to take?"
                    )
                    logger.info("Stall tier 1: gentle nudge")

                # Tier 2: Diagnostic prompt
                elif consecutive_stalls == 2:
                    cont_msg = (
                        f"I notice you've responded {consecutive_stalls} times without using tools. "
                        "This usually means one of:\n"
                        "1. You're unsure what to do next — use Glob or Read to explore the project\n"
                        "2. You think you're done — if so, say 'task complete'\n"
                        "3. You're having trouble with the tool format — tools are called as function calls\n\n"
                        f"If you're not done, use a tool NOW to make progress on: {actual_task}"
                    )
                    logger.info("Stall tier 2: diagnostic prompt")

                # Tier 3: Repair prompt with examples, optionally escalate model
                elif consecutive_stalls == 3:
                    if self._escalation_model and self._model != self._escalation_model:
                        logger.info(
                            "Auto-escalating from %s to %s due to stall",
                            self._model,
                            self._escalation_model,
                        )
                        self.switch_model(self._escalation_model)
                        transport.set_model(self._escalation_model)
                        cont_msg = (
                            f"[Model upgraded to {self._escalation_model}] "
                            f"Continue working on the task. The original goal was: {actual_task}\n"
                            "Use tools to make progress."
                        )
                        consecutive_stalls = (
                            0  # reset stalls — give new model a fresh chance
                        )
                        logger.info(
                            "Stall tier 3: auto-escalation + fresh continuation"
                        )
                    else:
                        cont_msg = (
                            "You MUST use a tool on this turn. You have not used tools for 3 consecutive turns. "
                            "Here's what to do:\n\n"
                            "1. If you need to see what files exist: call Glob with pattern='**/*'\n"
                            "2. If you need to read a file: call Read with file_path='<path>'\n"
                            "3. If you need to run a command: call Bash with command='ls -la'\n\n"
                            f"The task is: {actual_task}\n"
                            "Take one concrete action right now."
                        )
                        logger.info("Stall tier 3: repair prompt with examples")

                # Tier 4: Bail with diagnosis
                elif consecutive_stalls >= 4:
                    logger.warning(
                        "Agent stalled after %d turns (%d tool calls made before stall). "
                        "The model responded 4+ times without making tool calls. Stopping.",
                        turn_count,
                        total_tool_calls,
                    )
                    final_text = (
                        "[STALL] Agent reached maximum diagnostic attempts (4 turns without tool use). "
                        "The model may not support function calling well on this task. "
                        "Checkpoint saved for resumption with a different model."
                    )
                    stopped_reason = "stall_diagnosed"
                    error_message = f"Model stalled: {consecutive_stalls} consecutive turns without tool calls"

                    # ── Hook: on_stall_timeout ────────────────────────
                    await self._hooks.fire(
                        "on_stall_timeout",
                        turn=turn_count,
                        stall_count=consecutive_stalls,
                    )

                    # Save final checkpoint before bailing
                    checkpoint_path = await self._save_checkpoint(
                        transport,
                        turn_count,
                        actual_task,
                        cwd,
                        total_tool_calls,
                        session_id,
                    )
                    break
                else:
                    # Fallback
                    cont_msg = "Continue working."

                transport.add_user_message(cont_msg)
                local_messages.append({"role": "user", "content": cont_msg})
                continue

            else:
                # for-loop exhausted without break
                logger.warning(
                    "Reached max_turns=%d without completion", self._max_turns
                )
                stopped_reason = "max_turns"

        finally:
            try:
                await transport.close()
            except Exception:
                pass

        elapsed = time.monotonic() - start_time
        logger.info(
            "Run done: stopped_reason=%s, turns=%d, tool_calls=%d, elapsed=%.1fs, checkpoint=%s",
            stopped_reason,
            turn_count,
            total_tool_calls,
            elapsed,
            checkpoint_path,
        )

        # ── Hook: on_complete ──────────────────────────────────────────
        result = RunResult(
            messages=local_messages,
            final_text=final_text,
            turn_count=turn_count,
            total_tool_calls=total_tool_calls,
            success=(stopped_reason == "completed"),
            stopped_reason=stopped_reason,
            elapsed_seconds=elapsed,
            checkpoint_path=checkpoint_path,
            model_history=self._model_history,
            error_message=error_message,
        )

        await self._hooks.fire("on_complete", result=result)

        return result

    async def _save_checkpoint(
        self,
        transport: LiteLLMHTTPTransport,
        turn_count: int,
        task: str,
        cwd: str | None,
        total_tool_calls: int,
        session_id: str,
    ) -> str | None:
        """Save conversation state to disk after each tool execution turn.

        Args:
            transport: The LiteLLM transport with message history.
            turn_count: Current turn number.
            task: The original task description.
            cwd: Working directory.
            total_tool_calls: Total tool calls executed so far.
            session_id: Unique session identifier.

        Returns:
            Path to the saved checkpoint file, or None if checkpoint_dir is not set.
        """
        if not self._checkpoint_dir:
            return None

        checkpoint_dir = Path(self._checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        checkpoint_file = checkpoint_dir / f"{session_id}_turn{turn_count}.json"
        latest_file = checkpoint_dir / "latest.json"

        checkpoint_data = {
            "session_id": session_id,
            "task": task,
            "cwd": cwd,
            "model": self._model,
            "base_url": self._base_url,
            "turn_count": turn_count,
            "total_tool_calls": total_tool_calls,
            "system_prompt": self._custom_system_prompt or "",
            "messages": transport._messages,  # Full conversation history
            "timestamp": timestamp,
            "tools": self._default_tools or default_tools(),
            "model_history": self._model_history,
        }

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug("Checkpoint saved: %s", checkpoint_file)

            # Also save as latest.json for convenient resumption
            with open(latest_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            return str(checkpoint_file)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
            return None

    @classmethod
    async def resume(
        cls,
        checkpoint_path: str,
        **kwargs: Any,
    ) -> RunResult:
        """Resume a failed/interrupted run from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
            **kwargs: Override parameters (max_turns, model, base_url, etc.)

        Returns:
            RunResult from the resumed run.
        """
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s", exc)
            return RunResult(
                messages=[],
                final_text=f"[ERROR] Could not load checkpoint: {exc}",
                turn_count=0,
                total_tool_calls=0,
                success=False,
                stopped_reason="error",
                elapsed_seconds=0.0,
                checkpoint_path=checkpoint_path,
                error_message=f"Could not load checkpoint: {exc}",
            )

        # Extract checkpoint data
        task = checkpoint.get("task", "")
        cwd = checkpoint.get("cwd")
        model = kwargs.get("model", checkpoint.get("model", "qwen/qwen3-max"))
        base_url = kwargs.get(
            "base_url", checkpoint.get("base_url", "http://100.102.119.55:3002")
        )
        max_turns = kwargs.get("max_turns", checkpoint.get("turn_count", 50) + 50)
        api_key = kwargs.get("api_key", "sk-litellm")
        timeout = kwargs.get("timeout", 120.0)
        checkpoint_dir = kwargs.get("checkpoint_dir")
        tools = kwargs.get("tools", checkpoint.get("tools"))
        escalation_model = kwargs.get("escalation_model")

        # Create a new runner instance
        runner = cls(
            task=task,
            tools=tools,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_turns=max_turns,
            timeout=timeout,
            checkpoint_dir=checkpoint_dir,
            escalation_model=escalation_model,
        )

        # Resume the run
        logger.info(
            "Resuming from checkpoint: turn %d, tool_calls %d",
            checkpoint.get("turn_count"),
            checkpoint.get("total_tool_calls"),
        )

        return await runner.run(task=task, cwd=cwd, tools=tools)

    @staticmethod
    def _is_complete(text: str) -> bool:
        """Return True if text contains an explicit task-completion marker."""
        if not text:
            return False
        lower = text.lower()
        return any(marker in lower for marker in _COMPLETION_MARKERS)


__all__ = ["AutonomousRunner", "RunResult"]
