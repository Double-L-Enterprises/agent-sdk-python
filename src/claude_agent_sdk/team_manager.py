"""TeamManager - orchestrate multiple AutonomousRunner agents.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from .cost_tracker import CostTracker
from .message_bus import Message, MessageBus
from .observability import global_history, global_metrics
from .runner import AutonomousRunner, RunResult

logger = logging.getLogger(__name__)

class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AgentSpec:
    name: str
    model: str
    task: str
    base_url: str = "http://127.0.0.1:8016"
    api_key: str = "sk-litellm"
    max_turns: int = 50
    timeout: float = 120.0
    system_prompt: str | None = None

@dataclass
class AgentResult:
    name: str
    state: AgentState
    result: RunResult | None = None
    error: str = ""
    elapsed: float = 0.0

class TeamManager:
    def __init__(self, team_id: str = "default-team", message_bus: MessageBus | None = None,
                 cost_tracker: CostTracker | None = None) -> None:
        self.team_id = team_id
        self.bus = message_bus or MessageBus()
        self.cost_tracker = cost_tracker
        self._agents: dict[str, AgentSpec] = {}
        self._results: dict[str, AgentResult] = {}

    def add_agent(self, name: str, model: str, task: str, **kwargs: Any) -> None:
        self._agents[name] = AgentSpec(name=name, model=model, task=task, **kwargs)

    async def start_all(self, timeout: float = 120.0) -> dict[str, AgentResult]:
        atasks = {}
        for name, spec in self._agents.items():
            global_metrics.set_agent_state(agent_name=name, state="running")
            atasks[name] = asyncio.create_task(self._run_agent(spec), name="agent-" + name)
        done, pending = await asyncio.wait(atasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED)
        for name, task in atasks.items():
            if task in pending:
                task.cancel()
                self._results[name] = AgentResult(name=name, state=AgentState.TIMEOUT, error="Timed out")
                global_metrics.set_agent_state(agent_name=name, state="done")
        for name, task in atasks.items():
            if task in done:
                try:
                    task.result()
                except Exception as e:
                    if name not in self._results:
                        self._results[name] = AgentResult(name=name, state=AgentState.FAILED, error=str(e))
        return self._results

    async def _run_agent(self, spec: AgentSpec) -> None:
        start = time.monotonic()
        runner = AutonomousRunner(
            task=spec.task, model=spec.model, base_url=spec.base_url, api_key=spec.api_key,
            max_turns=spec.max_turns, timeout=spec.timeout, system_prompt=spec.system_prompt,
            cost_tracker=self.cost_tracker, agent_name=spec.name, team_id=self.team_id,
            pipeline_mode="autonomous",
        )
        try:
            result = await runner.run()
            elapsed = time.monotonic() - start
            state = AgentState.DONE if result.success else AgentState.FAILED
            self._results[spec.name] = AgentResult(name=spec.name, state=state, result=result, elapsed=elapsed)
            msg = Message(
                from_agent=spec.name, to_agent="team-lead",
                content=result.final_text[:500],
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            self.bus.send(msg)
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Agent %s failed: %s", spec.name, e)
            self._results[spec.name] = AgentResult(name=spec.name, state=AgentState.FAILED, error=str(e), elapsed=elapsed)
        finally:
            global_metrics.set_agent_state(agent_name=spec.name, state="done")

    def get_results(self) -> dict[str, AgentResult]:
        return dict(self._results)
