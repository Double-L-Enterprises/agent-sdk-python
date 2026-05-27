"""Multi-agent orchestration — run multiple AutonomousRunners in parallel or sequence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .runner import AutonomousRunner, RunResult

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a single agent in the orchestrator."""

    name: str
    task: str
    model: str = "qwen/qwen3-max"
    cwd: str = "."
    max_turns: int = 50
    escalation_model: str | None = None
    system_prompt: str | None = None
    depends_on: list[str] = field(default_factory=list)  # agent names this depends on


@dataclass
class MultiAgentResult:
    """Aggregated result from all agents."""

    results: dict[str, RunResult]  # name -> result
    total_turns: int
    total_tool_calls: int
    total_elapsed_seconds: float
    all_succeeded: bool
    failed_agents: list[str]

    def summary(self) -> str:
        """Generate a human-readable summary of all agent results."""
        lines = []
        for name, r in self.results.items():
            status = "PASS" if r.success else f"FAIL ({r.stopped_reason})"
            lines.append(
                f"  {name}: {status} — {r.turns} turns, {r.total_tool_calls} tools, {r.elapsed_seconds:.1f}s"
            )
        return "\n".join(lines)


class MultiAgentOrchestrator:
    """Orchestrate multiple AutonomousRunners in parallel or sequential execution."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8016",
        api_key: str = "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        checkpoint_dir: str | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            base_url: LiteLLM base URL for all agents.
            api_key: API key for LiteLLM.
            checkpoint_dir: Base directory for agent checkpoints (each agent gets a subdir).
        """
        self._base_url = base_url
        self._api_key = api_key
        self._checkpoint_dir = checkpoint_dir
        self._agents: dict[str, AgentConfig] = {}

    def add_agent(
        self,
        name: str,
        task: str,
        model: str = "qwen/qwen3-max",
        cwd: str = ".",
        max_turns: int = 50,
        escalation_model: str | None = None,
        system_prompt: str | None = None,
        depends_on: list[str] | None = None,
    ) -> None:
        """Register an agent.

        Args:
            name: Unique agent identifier.
            task: Task description.
            model: Model name to use.
            cwd: Working directory for Bash calls.
            max_turns: Maximum turns for this agent.
            escalation_model: Optional model to switch to on stall.
            system_prompt: Optional custom system prompt.
            depends_on: List of agent names this agent depends on (waits for completion).
        """
        self._agents[name] = AgentConfig(
            name=name,
            task=task,
            model=model,
            cwd=cwd,
            max_turns=max_turns,
            escalation_model=escalation_model,
            system_prompt=system_prompt,
            depends_on=depends_on or [],
        )

    async def run_parallel(self) -> MultiAgentResult:
        """Run all agents with dependency respect.

        Agents with no dependencies run immediately in parallel.
        Agents with dependencies wait for their dependencies to complete first.
        Returns immediately when all agents finish.

        Returns:
            MultiAgentResult with all results and aggregate stats.
        """
        start = time.monotonic()
        results: dict[str, RunResult] = {}
        completed: set[str] = set()

        # Separate into waves based on dependencies
        remaining = dict(self._agents)

        while remaining:
            # Find agents whose dependencies are all completed
            ready = {
                name: cfg
                for name, cfg in remaining.items()
                if all(dep in completed for dep in cfg.depends_on)
            }

            if not ready:
                # Deadlock — circular dependency
                for name in remaining:
                    results[name] = RunResult(
                        messages=[],
                        final_text=f"Deadlock: dependencies {remaining[name].depends_on} never completed",
                        turn_count=0,
                        total_tool_calls=0,
                        elapsed_seconds=0,
                        success=False,
                        stopped_reason="error",
                        model_history=[],
                    )
                break

            # Run ready agents in parallel
            tasks = {name: self._run_agent(cfg) for name, cfg in ready.items()}
            agent_results = await asyncio.gather(
                *tasks.values(), return_exceptions=True
            )

            for name, result in zip(tasks.keys(), agent_results):
                if isinstance(result, Exception):
                    results[name] = RunResult(
                        messages=[],
                        final_text=str(result),
                        turn_count=0,
                        total_tool_calls=0,
                        elapsed_seconds=0,
                        success=False,
                        stopped_reason="error",
                        model_history=[],
                    )
                else:
                    results[name] = result
                completed.add(name)
                del remaining[name]

        elapsed = time.monotonic() - start
        failed = [n for n, r in results.items() if not r.success]

        return MultiAgentResult(
            results=results,
            total_turns=sum(r.turns for r in results.values()),
            total_tool_calls=sum(r.total_tool_calls for r in results.values()),
            total_elapsed_seconds=elapsed,
            all_succeeded=len(failed) == 0,
            failed_agents=failed,
        )

    async def run_sequential(self) -> MultiAgentResult:
        """Run agents one at a time in registration order.

        Each agent can see context from what previous agents completed (shared cwd).
        The task of each agent is prefixed with previous results.

        Returns:
            MultiAgentResult with all results and aggregate stats.
        """
        start = time.monotonic()
        results: dict[str, RunResult] = {}

        for name, cfg in self._agents.items():
            # Build context from previous agent results
            context = ""
            if results:
                prev_summaries = []
                for prev_name, prev_result in results.items():
                    prev_summaries.append(
                        f"Agent '{prev_name}' completed: {prev_result.final_text[:200]}"
                    )
                context = (
                    "\n\nPrevious agents completed:\n"
                    + "\n".join(prev_summaries)
                    + "\n\n"
                )

            task_with_context = context + cfg.task if context else cfg.task
            modified_cfg = AgentConfig(
                name=cfg.name,
                task=task_with_context,
                model=cfg.model,
                cwd=cfg.cwd,
                max_turns=cfg.max_turns,
                escalation_model=cfg.escalation_model,
                system_prompt=cfg.system_prompt,
                depends_on=cfg.depends_on,
            )
            results[name] = await self._run_agent(modified_cfg)

        elapsed = time.monotonic() - start
        failed = [n for n, r in results.items() if not r.success]

        return MultiAgentResult(
            results=results,
            total_turns=sum(r.turns for r in results.values()),
            total_tool_calls=sum(r.total_tool_calls for r in results.values()),
            total_elapsed_seconds=elapsed,
            all_succeeded=len(failed) == 0,
            failed_agents=failed,
        )

    async def _run_agent(self, cfg: AgentConfig) -> RunResult:
        """Run a single agent.

        Args:
            cfg: Agent configuration.

        Returns:
            RunResult from this agent's autonomous run.
        """
        checkpoint_sub = None
        if self._checkpoint_dir:
            checkpoint_sub = f"{self._checkpoint_dir}/{cfg.name}"

        runner = AutonomousRunner(
            base_url=self._base_url,
            api_key=self._api_key,
            model=cfg.model,
            max_turns=cfg.max_turns,
            escalation_model=cfg.escalation_model,
            checkpoint_dir=checkpoint_sub,
        )

        logger.info(
            "Starting agent '%s' (model=%s, cwd=%s)", cfg.name, cfg.model, cfg.cwd
        )
        result = await runner.run(
            task=cfg.task,
            cwd=cfg.cwd,
            # Note: system_prompt passed to runner but not in run() signature currently
        )
        logger.info(
            "Agent '%s' finished: success=%s, turns=%d, tools=%d",
            cfg.name,
            result.success,
            result.turns,
            result.total_tool_calls,
        )
        return result
