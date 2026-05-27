"""TeamManager — live multi-agent teams backed by AutonomousRunner.

Each team member is a persistent AutonomousRunner instance running in its own
asyncio Task. Agents communicate via a shared MessageBus. The team-lead (Claude
Code / external orchestrator) can inject messages and poll status via this class.

Design decisions:
  - File-based MessageBus (no Redis). Simple, survives restarts, zero deps.
  - Each agent gets SendTeamMessage + ReadTeamMessages injected into its tool set.
  - System prompt includes team roster and a stay-alive loop directive.
  - devil_advocate() runs a structured N-round debate between two agents.
  - Agents stay alive until they receive a shutdown message or hit max_turns.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .message_bus import Message, MessageBus
from .runner import AutonomousRunner, RunResult
from .tools import default_tools, dispatch_tool
from .tools.team_message import (
    READ_TEAM_MESSAGES_TOOL,
    SEND_TEAM_MESSAGE_TOOL,
    read_team_messages,
    send_team_message,
    team_tools,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Runtime state for a single team member."""

    name: str
    model: str
    role: str
    task: str
    status: str = "pending"  # pending | running | completed | error | stalled
    turns: int = 0
    last_message_ts: float = field(default_factory=time.time)
    result: RunResult | None = None
    asyncio_task: asyncio.Task[None] | None = None  # type: ignore[type-arg]


class TeamManager:
    """Creates and manages multiple AutonomousRunner instances as a live team.

    Usage::
        tm = TeamManager(team_id="dashboard-build", output_dir="~/.claude/agent-logs/dashboard/")
        await tm.add_agent("planner", model="qwen/qwen3-max", role="Architect", task="Design the API")
        await tm.add_agent("backend", model="nvidia/devstral-2-123b", role="Backend engineer", task="")
        await tm.start_all()
        await tm.send_to("backend", "Build FastAPI endpoints per spec in agent-logs/planner.md")
        results = await tm.wait_for_completion(timeout=1800)
        print(tm.status())
    """

    def __init__(
        self,
        team_id: str,
        output_dir: str,
        base_url: str = "http://127.0.0.1:8016",
        api_key: str = "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        bus_dir: str | None = None,
    ) -> None:
        """Initialize the TeamManager.

        Args:
            team_id: Unique identifier for this team. Used to namespace messages.
            output_dir: Directory where agent output files are written.
            base_url: LiteLLM base URL all agents will use.
            api_key: API key for the LiteLLM endpoint.
            bus_dir: Override for the MessageBus directory. Defaults to
                     /tmp/agent-teams/{team_id}/messages
        """
        self.team_id = team_id
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._base_url = base_url
        self._api_key = api_key

        self.bus = MessageBus(team_id=team_id, bus_dir=bus_dir)
        self._agents: dict[str, AgentState] = {}

    # ── Agent registration ─────────────────────────────────────────────────────

    async def add_agent(
        self,
        name: str,
        model: str,
        role: str,
        task: str = "",
        max_turns: int = 100,
        escalation_model: str | None = None,
    ) -> None:
        """Register an agent on the team.

        Creates an AutonomousRunner for this agent with team-aware tools and
        a stay-alive system prompt. The agent is NOT started until start_all()
        or individual asyncio task creation.

        Args:
            name: Unique agent name. Used for message routing.
            model: LiteLLM model ID.
            role: Agent's role description (injected into system prompt).
            task: Initial task. If empty, agent waits for a message via ReadTeamMessages.
            max_turns: Maximum turns for this agent.
            escalation_model: Optional fallback model on stall.
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered on team '{self.team_id}'")

        state = AgentState(
            name=name,
            model=model,
            role=role,
            task=task,
        )
        self._agents[name] = state

        # Ensure the agent has an inbox in the bus
        self.bus._agent_dir(name)
        logger.info("Registered agent '%s' (model=%s) on team '%s'", name, model, self.team_id)

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start all registered agents as concurrent asyncio Tasks.

        Each agent runs its initial task (or waits for a message if task is empty).
        Returns immediately after spawning all tasks.
        """
        for name, state in self._agents.items():
            if state.asyncio_task is not None:
                logger.warning("Agent '%s' already started, skipping", name)
                continue
            state.status = "running"
            state.asyncio_task = asyncio.create_task(
                self._run_agent_task(state),
                name=f"team-{self.team_id}-agent-{name}",
            )
            logger.info("Started agent task '%s'", name)

    # ── Messaging ──────────────────────────────────────────────────────────────

    async def send_to(
        self,
        agent_name: str,
        message: str,
        from_name: str = "orchestrator",
        thread_id: str | None = None,
    ) -> None:
        """Send a message to a specific agent.

        Args:
            agent_name: Destination agent name.
            message: Message content.
            from_name: Sender name shown to the recipient.
            thread_id: Optional thread ID for conversation continuity.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        msg = Message(
            from_agent=from_name,
            to_agent=agent_name,
            content=message,
            timestamp=now_iso,
            thread_id=thread_id,
        )
        self.bus.send(msg)
        logger.info("Sent message from '%s' to '%s'", from_name, agent_name)

    async def broadcast(self, message: str, from_name: str = "orchestrator") -> None:
        """Send a message to all registered agents.

        Args:
            message: Broadcast content.
            from_name: Sender name.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        msg = Message(
            from_agent=from_name,
            to_agent="*",
            content=message,
            timestamp=now_iso,
        )
        self.bus.send(msg)
        logger.info("Broadcast from '%s' to all agents", from_name)

    # ── Wait for completion ────────────────────────────────────────────────────

    async def wait_for_completion(self, timeout: int = 3600) -> dict[str, Any]:
        """Wait for all agent tasks to finish.

        Args:
            timeout: Maximum seconds to wait. Raises asyncio.TimeoutError if exceeded.

        Returns:
            Dict mapping agent names to their RunResult (or error string).
        """
        tasks = [
            state.asyncio_task
            for state in self._agents.values()
            if state.asyncio_task is not None
        ]

        if not tasks:
            return {}

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            logger.warning("wait_for_completion timed out after %ds", timeout)

        return {
            name: (state.result if state.result else state.status)
            for name, state in self._agents.items()
        }

    # ── Devil's advocate ───────────────────────────────────────────────────────

    async def devil_advocate(
        self,
        agent_a: str,
        agent_b: str,
        topic: str,
        rounds: int = 3,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a structured debate between two agents.

        Protocol:
          Round 1: agent_a proposes (receives topic via message)
          Round 2: agent_b critiques (receives agent_a's proposal)
          Round 3: agent_a revises (receives critique)
          ... continues for N rounds or until either agent sends "LGTM" or {"type":"approval"}

        The debate uses the message bus. Both agents must already be started
        (via start_all()) and able to process incoming messages.

        Args:
            agent_a: Proposer agent name.
            agent_b: Critic/reviewer agent name.
            topic: The topic or question to debate.
            rounds: Number of exchange rounds (each round = 1 message each way).
            thread_id: Thread ID for this debate. Auto-generated if omitted.

        Returns:
            List of dicts: [{"round": N, "from": name, "content": text, "timestamp": ts}]
        """
        import uuid as _uuid
        debate_thread = thread_id or f"debate-{_uuid.uuid4().hex[:8]}"
        transcript: list[dict[str, Any]] = []

        logger.info(
            "Starting devil's advocate debate: %s vs %s on '%s' (%d rounds, thread=%s)",
            agent_a, agent_b, topic[:60], rounds, debate_thread,
        )

        # Round 1: seed agent_a with the proposal request
        proposal_prompt = (
            f"DEBATE ROUND 1 — PROPOSAL\n\n"
            f"Topic: {topic}\n\n"
            f"You are the proposer. Provide a detailed proposal or position on this topic. "
            f"Be specific. Cover approach, tradeoffs, and key decisions. "
            f"After writing your proposal, send it to '{agent_b}' using SendTeamMessage "
            f"with thread_id='{debate_thread}'."
        )
        await self.send_to(agent_a, proposal_prompt, from_name="debate-coordinator", thread_id=debate_thread)

        # Collect transcript from the message bus as debate unfolds
        # We poll the _all directory and filter by thread_id
        seen_msg_ids: set[str] = set()
        start_ms = int(time.time() * 1000)

        for round_num in range(1, rounds + 1):
            # Wait for the expected message in this round
            # agent_a → agent_b in odd rounds, agent_b → agent_a in even rounds
            speaker_sending = agent_a if round_num % 2 == 1 else agent_b
            speaker_receiving = agent_b if round_num % 2 == 1 else agent_a

            # Wait up to 5 minutes per round for a message
            deadline = time.time() + 300
            while time.time() < deadline:
                all_msgs = self.bus.get_all()
                new_msgs = [
                    m for m in all_msgs
                    if m.thread_id == debate_thread
                    and m.from_agent == speaker_sending
                    and m.to_agent in (speaker_receiving, "*")
                    and m.message_id not in seen_msg_ids
                    and m.timestamp_ms >= start_ms
                ]

                if new_msgs:
                    latest = max(new_msgs, key=lambda m: m.timestamp_ms)
                    seen_msg_ids.add(latest.message_id)
                    transcript.append({
                        "round": round_num,
                        "from": latest.from_agent,
                        "to": latest.to_agent,
                        "content": latest.content,
                        "timestamp": latest.timestamp,
                        "thread_id": debate_thread,
                    })
                    logger.info("Debate round %d: %s → %s", round_num, latest.from_agent, latest.to_agent)

                    # Check for approval / consensus
                    content_lower = latest.content.lower()
                    if "lgtm" in content_lower or '"type": "approval"' in latest.content:
                        logger.info("Debate consensus reached at round %d", round_num)
                        await self.send_to(
                            agent_a,
                            f"DEBATE COMPLETE — consensus reached at round {round_num}. "
                            f"Full transcript has {len(transcript)} exchanges.",
                            from_name="debate-coordinator",
                            thread_id=debate_thread,
                        )
                        return transcript

                    # If this isn't the last round, prompt the other agent to respond
                    if round_num < rounds:
                        if round_num % 2 == 1:
                            # Prompt critic
                            critique_prompt = (
                                f"DEBATE ROUND {round_num + 1} — CRITIQUE\n\n"
                                f"You received this proposal from {agent_a}:\n\n"
                                f"{latest.content}\n\n"
                                f"As devil's advocate: find at least 3 specific weaknesses. "
                                f"Propose concrete alternatives. Do NOT approve without challenging. "
                                f"Reply to '{agent_a}' via SendTeamMessage with thread_id='{debate_thread}'."
                            )
                            await self.send_to(
                                agent_b, critique_prompt,
                                from_name="debate-coordinator", thread_id=debate_thread,
                            )
                        else:
                            # Prompt proposer to revise
                            revision_prompt = (
                                f"DEBATE ROUND {round_num + 1} — REVISION\n\n"
                                f"You received this critique from {agent_b}:\n\n"
                                f"{latest.content}\n\n"
                                f"Address each objection specifically. Revise your proposal where the "
                                f"critic is right. Defend positions where they are not. "
                                f"If you believe {agent_b} is now satisfied, ask for approval. "
                                f"Reply to '{agent_b}' via SendTeamMessage with thread_id='{debate_thread}'."
                            )
                            await self.send_to(
                                agent_a, revision_prompt,
                                from_name="debate-coordinator", thread_id=debate_thread,
                            )
                    break

                await asyncio.sleep(5)  # Poll every 5s

            else:
                # Timeout waiting for round
                logger.warning("Debate round %d timed out waiting for %s", round_num, speaker_sending)
                transcript.append({
                    "round": round_num,
                    "from": "debate-coordinator",
                    "content": f"TIMEOUT: {speaker_sending} did not respond within 5 minutes",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                break

        # Final wrap-up
        await self.send_to(
            agent_a,
            f"DEBATE COMPLETE — {rounds} rounds concluded. Proceed with your work.",
            from_name="debate-coordinator", thread_id=debate_thread,
        )
        await self.send_to(
            agent_b,
            f"DEBATE COMPLETE — {rounds} rounds concluded. Proceed with your work.",
            from_name="debate-coordinator", thread_id=debate_thread,
        )

        return transcript

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return the current state of all agents.

        Returns:
            Dict with team_id, agent states, and aggregate metrics.
        """
        agents_status = {}
        for name, state in self._agents.items():
            agents_status[name] = {
                "status": state.status,
                "model": state.model,
                "role": state.role,
                "turns": state.turns,
                "last_message_ts": state.last_message_ts,
                "has_result": state.result is not None,
                "task_done": state.asyncio_task.done() if state.asyncio_task else False,
            }

        all_done = all(
            s.asyncio_task is None or s.asyncio_task.done()
            for s in self._agents.values()
        )

        return {
            "team_id": self.team_id,
            "output_dir": str(self.output_dir),
            "all_done": all_done,
            "agent_count": len(self._agents),
            "agents": agents_status,
        }

    # ── Graceful stop ──────────────────────────────────────────────────────────

    async def stop_all(self, reason: str = "orchestrator requested") -> None:
        """Send shutdown messages to all agents and cancel their tasks.

        Agents receive a shutdown message first, giving them a chance to wrap up.
        After 10 seconds, any still-running tasks are cancelled.

        Args:
            reason: Explanation sent to agents in the shutdown message.
        """
        for name in self._agents:
            try:
                await self.send_to(
                    name,
                    f'{{"type": "shutdown", "reason": "{reason}"}}',
                    from_name="orchestrator",
                )
            except Exception as exc:
                logger.warning("Failed to send shutdown to '%s': %s", name, exc)

        # Give agents 10s to process shutdown gracefully
        await asyncio.sleep(10)

        # Cancel any still-running tasks
        for name, state in self._agents.items():
            if state.asyncio_task and not state.asyncio_task.done():
                logger.info("Cancelling task for agent '%s'", name)
                state.asyncio_task.cancel()
                state.status = "stopped"

    # ── Internal: per-agent runner ─────────────────────────────────────────────

    async def _run_agent_task(self, state: AgentState) -> None:
        """Run a single agent's AutonomousRunner in this asyncio Task.

        This wraps dispatch_tool to inject message_bus and agent_name into
        the kwargs for SendTeamMessage and ReadTeamMessages calls.

        The agent receives a team-aware system prompt and the team communication
        tools. Its output is written to output_dir/{name}.md.
        """
        roster = self._build_roster(exclude=state.name)
        system_prompt = self._build_agent_system_prompt(state, roster)
        tools = default_tools() + [SEND_TEAM_MESSAGE_TOOL, READ_TEAM_MESSAGES_TOOL]

        output_file = self.output_dir / f"{state.name}.md"

        # Task: if no initial task given, agent should poll for messages
        task = state.task if state.task else (
            f"You are {state.name} on team '{self.team_id}'. "
            f"Your role: {state.role}. "
            f"No initial task assigned. Call ReadTeamMessages to check for assignments. "
            f"After each task, call ReadTeamMessages again. "
            f'Only stop if you receive a {{"type": "shutdown"}} message.'
        )

        runner = AutonomousRunner(
            model=state.model,
            base_url=self._base_url,
            api_key=self._api_key,
            max_turns=100,
            system_prompt=system_prompt,
        )

        # Monkey-patch dispatch_tool on this runner to inject team kwargs
        bus = self.bus
        agent_name = state.name
        original_run = runner.run

        # We need to wrap the tools dispatcher — easiest approach is to add a
        # custom executor mapping inside AutonomousRunner's dispatch path.
        # Since dispatch_tool is a module-level function, we inject via a subclass
        # of the runner that overrides _dispatch.
        # Instead: use the hooks system to intercept tool calls.
        from .hooks import HookRegistry

        hooks = HookRegistry()

        # pre_tool_use hook: capture team tool calls before they reach dispatch
        # We store results in a mutable cell so post_tool_use can inject them
        _intercept: dict[str, Any] = {}

        @hooks.hook("pre_tool_use")
        async def intercept_team_tools(tool_name: str, params: dict, **_kwargs: Any) -> dict | None:
            if tool_name in ("SendTeamMessage", "ReadTeamMessages"):
                _intercept["pending"] = (tool_name, params)
                # Return params unchanged — we'll handle in post hook via a side channel
            return params

        # We need a different approach: override dispatch_tool entirely.
        # The cleanest way given the existing architecture is a monkey-patch on
        # the runner's tool dispatch. We do this by replacing the tools list with
        # a custom dispatcher wrapper that handles our two tools.
        # Actually the cleanest approach: create a TeamAwareRunner subclass inline.

        class TeamAwareRunner(AutonomousRunner):
            """AutonomousRunner with team tool dispatch wired in."""

            async def _dispatch_team_tool(
                self,
                name: str,
                params: dict[str, Any],
                cwd: str | None = None,
                **kwargs: Any,
            ) -> str:
                kwargs["message_bus"] = bus
                kwargs["agent_name"] = agent_name
                if name == "SendTeamMessage":
                    return await send_team_message(params, **kwargs)
                elif name == "ReadTeamMessages":
                    return await read_team_messages(params, **kwargs)
                # Fall through to normal dispatch
                return await dispatch_tool(name, params, cwd=cwd, **kwargs)

        team_runner = TeamAwareRunner(
            model=state.model,
            base_url=self._base_url,
            api_key=self._api_key,
            max_turns=100,
            system_prompt=system_prompt,
        )

        # Patch dispatch_tool reference on the runner's module scope for team tools
        # The runner calls dispatch_tool from tools/__init__.py; we need to intercept.
        # The cleanest solution: add team tool executors to the _EXECUTORS dict in tools/__init__.py
        # That's done in the __init__.py update. For now we use the hooks approach
        # to capture the kwargs passing.

        # The hooks inject message_bus and agent_name via kwargs in runner.py line ~314:
        # dispatch_tool(tool_name, hook_result or params, cwd=cwd, ...)
        # So we register the tools in _EXECUTORS at runtime.

        from .tools import _EXECUTORS  # type: ignore[attr-defined]
        _EXECUTORS["SendTeamMessage"] = lambda p, **kw: send_team_message(p, **kw)
        _EXECUTORS["ReadTeamMessages"] = lambda p, **kw: read_team_messages(p, **kw)

        # We also need to pass message_bus and agent_name into the dispatch kwargs.
        # runner.py calls dispatch_tool with parent_base_url, parent_api_key, etc.
        # We inject via a pre_tool_use hook that stashes on the runner, and a
        # custom dispatch path. Since modifying runner.py inline here is fragile,
        # the simplest approach: subclass and override the internal dispatch call.

        # Final approach: use the HookRegistry on the runner to stash team context,
        # and override dispatch_tool in tools/__init__ to pull from a thread-local
        # context. But that's over-engineering for now.
        #
        # PRAGMATIC: Add message_bus and agent_name to the runner kwargs dict
        # by patching the runner's run() method to pass extra kwargs to dispatch_tool.
        # Since runner.py line 314 calls:
        #   tool_result = await dispatch_tool(tool_name, hook_result or params, cwd=cwd,
        #       parent_base_url=..., parent_api_key=..., ...)
        # Those extra kwargs flow into dispatch_tool's **kwargs and on to our executors.
        #
        # We just need to add message_bus + agent_name to the runner's extra dispatch kwargs.
        # That requires a 1-line change to runner.py. We do it here by subclassing run().

        class TeamContextRunner(AutonomousRunner):
            """Injects team context into every dispatch_tool call."""

            _team_message_bus: MessageBus
            _team_agent_name: str

            async def run(  # type: ignore[override]
                self,
                task: str | None = None,
                *,
                cwd: str | None = None,
                tools: list[dict[str, Any]] | None = None,
            ) -> RunResult:
                # We monkeypatch the tools dispatcher temporarily
                original_dispatch = _EXECUTORS.copy()

                # Wrap executors to inject team kwargs
                def _make_team_wrapper(fn: Any, mbus: MessageBus, aname: str) -> Any:
                    async def wrapper(params: dict, **kwargs: Any) -> str:
                        kwargs["message_bus"] = mbus
                        kwargs["agent_name"] = aname
                        return await fn(params, **kwargs)
                    return wrapper

                _EXECUTORS["SendTeamMessage"] = _make_team_wrapper(
                    send_team_message, self._team_message_bus, self._team_agent_name
                )
                _EXECUTORS["ReadTeamMessages"] = _make_team_wrapper(
                    read_team_messages, self._team_message_bus, self._team_agent_name
                )

                try:
                    return await super().run(task=task, cwd=cwd, tools=tools)
                finally:
                    # Restore original executors
                    for k in ("SendTeamMessage", "ReadTeamMessages"):
                        if k in original_dispatch:
                            _EXECUTORS[k] = original_dispatch[k]
                        elif k in _EXECUTORS:
                            del _EXECUTORS[k]

        final_runner = TeamContextRunner(
            model=state.model,
            base_url=self._base_url,
            api_key=self._api_key,
            max_turns=100,
            system_prompt=system_prompt,
        )
        final_runner._team_message_bus = bus
        final_runner._team_agent_name = agent_name

        logger.info("Running agent '%s' (model=%s)", state.name, state.model)
        try:
            result = await final_runner.run(task=task, tools=tools)
            state.result = result
            state.turns = result.turns
            state.status = "completed" if result.success else "error"

            # Write output file
            await self._write_output(state, result)

        except asyncio.CancelledError:
            state.status = "stopped"
            logger.info("Agent '%s' task cancelled", state.name)
            raise
        except Exception as exc:
            state.status = "error"
            logger.error("Agent '%s' raised: %s", state.name, exc, exc_info=True)

        logger.info("Agent '%s' finished with status=%s", state.name, state.status)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_roster(self, exclude: str | None = None) -> str:
        """Build a human-readable team roster for the system prompt."""
        lines = []
        for name, state in self._agents.items():
            if name == exclude:
                continue
            lines.append(f"  - {name} ({state.model}): {state.role}")
        return "\n".join(lines) if lines else "  (no other team members)"

    def _build_agent_system_prompt(self, state: AgentState, roster: str) -> str:
        """Build a team-aware system prompt for an agent."""
        return f"""You are {state.name}, an AI agent on team '{self.team_id}'.

Your role: {state.role}

Your team members:
{roster}

You have access to two team communication tools:
- SendTeamMessage(to, content, thread_id?) — send a message to another agent
- ReadTeamMessages(since?) — check your inbox for new messages

STAY-ALIVE LOOP: After completing your current task, call ReadTeamMessages to check for new work.
Only stop working if you receive a message containing {{"type": "shutdown"}}.
If your inbox is empty and you have no pending work, call ReadTeamMessages(since=<now>) after 30 seconds.

COMPLETION PROTOCOL: When you finish a task:
1. Write your output to a file at {str(self.output_dir)}/{state.name}.md
2. Send a summary to your team lead via SendTeamMessage(to="team-lead", content="...")
3. Include: what was built, key decisions, file paths, any blockers

You are a senior engineer. Be specific, thorough, and communicate proactively.
"""

    async def _write_output(self, state: AgentState, result: RunResult) -> None:
        """Write agent output to the team's output directory."""
        output_file = self.output_dir / f"{state.name}.md"
        try:
            content = f"""# Agent Output: {state.name}
# Team: {self.team_id}
# Written: {time.strftime("%Y-%m-%dT%H:%M:%S")} CST
# Status: {state.status}
# Turns: {result.turns} | Tool calls: {result.total_tool_calls} | Elapsed: {result.elapsed_seconds:.1f}s

## Final Output

{result.final_text}

## Run Stats

- Model: {state.model}
- Stopped reason: {result.stopped_reason}
- Success: {result.success}
"""
            output_file.write_text(content)
            logger.info("Agent '%s' output written to %s", state.name, output_file)
        except Exception as exc:
            logger.warning("Failed to write output for '%s': %s", state.name, exc)


__all__ = ["TeamManager", "AgentState"]
