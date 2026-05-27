"""CostTracker — per-agent, per-team, per-run token and cost accounting.

Model pricing table (per 1M tokens, USD):
  - qwen/* and nvidia/* models: $0.00 (self-hosted, free)
  - claude-haiku-*:   $0.25 input / $1.25 output
  - claude-sonnet-*:  $3.00 input / $15.00 output
  - claude-opus-*:    $15.00 input / $75.00 output

BudgetPolicy controls per-agent, per-team, and per-run spending limits.
When a limit is exceeded, the agent is paused and the TeamManager is notified.
The optional auto_switch_model field lets the tracker suggest a cheaper fallback.

Integration points:
  - AutonomousRunner: call cost_tracker.record_turn() after each do_turn() call
  - TeamManager: wire cost_tracker into _run_agent_task(); check budget before each turn

Created: 2026-05-27 CST
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Model pricing table ───────────────────────────────────────────────────────
# Per-million-token prices in USD.  Key patterns are matched with str.startswith.
# Free models (self-hosted) resolve to $0.  Unknown models fall back to sonnet rates.

_PRICING: list[tuple[str, float, float]] = [
    # (model_prefix, input_per_M, output_per_M)
    # Free / self-hosted
    ("qwen/", 0.0, 0.0),
    ("nvidia/", 0.0, 0.0),
    ("openai/", 0.0, 0.0),           # usually routed through local LiteLLM
    ("mistral/", 0.0, 0.0),
    ("meta-llama/", 0.0, 0.0),
    ("deepseek/", 0.0, 0.0),
    ("minimax/", 0.0, 0.0),
    # Anthropic — matched by prefix so model version suffixes don't matter
    ("claude-haiku", 0.25, 1.25),
    ("claude-3-haiku", 0.25, 1.25),
    ("claude-sonnet", 3.0, 15.0),
    ("claude-3-5-sonnet", 3.0, 15.0),
    ("claude-opus", 15.0, 75.0),
    ("claude-3-opus", 15.0, 75.0),
]

# Default fallback if no prefix matches (treat as sonnet-level)
_DEFAULT_INPUT_PER_M = 3.0
_DEFAULT_OUTPUT_PER_M = 15.0


def _price_for_model(model: str) -> tuple[float, float]:
    """Return (input_per_M, output_per_M) for a model string."""
    model_lower = model.lower()
    for prefix, inp, out in _PRICING:
        if model_lower.startswith(prefix):
            return inp, out
    return _DEFAULT_INPUT_PER_M, _DEFAULT_OUTPUT_PER_M


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single LLM call."""
    inp_rate, out_rate = _price_for_model(model)
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


# ── Turn record ───────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """One LLM call within an agent's run."""

    agent_name: str
    team_id: str | None
    model: str
    turn_number: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "team_id": self.team_id,
            "model": self.model,
            "turn_number": self.turn_number,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "timestamp": self.timestamp,
        }

    def to_log_line(self) -> str:
        """Structured single-line log entry (key=value format)."""
        return (
            f"cost_record"
            f" agent={self.agent_name}"
            f" team={self.team_id or '-'}"
            f" model={self.model}"
            f" turn={self.turn_number}"
            f" input_tok={self.input_tokens}"
            f" output_tok={self.output_tokens}"
            f" cost_usd={self.cost_usd:.8f}"
            f" ts={self.timestamp}"
        )


# ── Budget policy ─────────────────────────────────────────────────────────────

@dataclass
class BudgetPolicy:
    """Spending limits for budget enforcement.

    Set any limit to None (or math.inf) to disable it.
    auto_switch_model: if set, CostTracker will suggest this model when the
    per-agent budget is exceeded instead of hard-pausing.
    """

    max_cost_per_agent: float | None = None   # USD; applies per agent instance
    max_cost_per_team: float | None = None    # USD; total across all team members
    max_cost_per_run: float | None = None     # USD; entire tracker lifetime
    auto_switch_model: str | None = None      # e.g. "qwen/qwen3-max" (free fallback)


# ── Budget exceeded event ─────────────────────────────────────────────────────

@dataclass
class BudgetExceededEvent:
    """Raised when a budget limit is hit."""

    limit_type: str        # "agent" | "team" | "run"
    agent_name: str
    team_id: str | None
    limit_usd: float
    actual_usd: float
    suggested_model: str | None   # auto_switch_model from BudgetPolicy, or None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "budget_exceeded",
            "limit_type": self.limit_type,
            "agent_name": self.agent_name,
            "team_id": self.team_id,
            "limit_usd": self.limit_usd,
            "actual_usd": round(self.actual_usd, 8),
            "suggested_model": self.suggested_model,
            "timestamp": self.timestamp,
        }


# ── CostTracker ───────────────────────────────────────────────────────────────

class CostTracker:
    """Tracks token usage and cost across agents, teams, and runs.

    Usage::
        tracker = CostTracker(
            policy=BudgetPolicy(max_cost_per_agent=0.10, auto_switch_model="qwen/qwen3-max"),
            custom_pricing={"my-custom/model": (0.5, 2.0)},  # optional overrides
        )

        # In AutonomousRunner after do_turn():
        event = tracker.record_turn(
            agent_name="backend",
            team_id="build-team",
            model="claude-sonnet-4-6",
            turn_number=turn_count,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )
        if event:
            # Budget exceeded — pause, notify, optionally switch model
            handle_budget_event(event)

        # Summaries
        tracker.cost_by_agent()
        tracker.cost_by_model()
        tracker.cost_by_team()
        tracker.total_cost()
        tracker.to_json()
    """

    def __init__(
        self,
        policy: BudgetPolicy | None = None,
        custom_pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            policy: Budget limits.  Defaults to no limits (tracking only).
            custom_pricing: Extra model_prefix → (input_per_M, output_per_M)
                            entries prepended to the global pricing table.
        """
        self._policy = policy or BudgetPolicy()
        self._custom_pricing: list[tuple[str, float, float]] = []
        if custom_pricing:
            for prefix, (inp, out) in custom_pricing.items():
                self._custom_pricing.append((prefix.lower(), inp, out))

        self._records: list[TurnRecord] = []

        # Accumulated totals (agent_name → values)
        self._agent_tokens_in: dict[str, int] = {}
        self._agent_tokens_out: dict[str, int] = {}
        self._agent_cost: dict[str, float] = {}

        # Team totals (team_id → values)
        self._team_tokens_in: dict[str, int] = {}
        self._team_tokens_out: dict[str, int] = {}
        self._team_cost: dict[str, float] = {}

        # Model totals (model → values)
        self._model_tokens_in: dict[str, int] = {}
        self._model_tokens_out: dict[str, int] = {}
        self._model_cost: dict[str, float] = {}

        # Run-level totals
        self._run_tokens_in = 0
        self._run_tokens_out = 0
        self._run_cost = 0.0

        # Paused agents (set of agent_name)
        self._paused_agents: set[str] = set()

    # ── Custom pricing helpers ────────────────────────────────────────────────

    def _price_for_model(self, model: str) -> tuple[float, float]:
        """Resolve price for model, checking custom overrides first."""
        model_lower = model.lower()
        for prefix, inp, out in self._custom_pricing:
            if model_lower.startswith(prefix):
                return inp, out
        return _price_for_model(model)

    # ── Record a turn ─────────────────────────────────────────────────────────

    def record_turn(
        self,
        agent_name: str,
        model: str,
        turn_number: int,
        input_tokens: int,
        output_tokens: int,
        team_id: str | None = None,
    ) -> BudgetExceededEvent | None:
        """Record token usage for one LLM call.

        Args:
            agent_name: Name of the agent that made the call.
            model: Model string used for the call.
            turn_number: Turn number within the agent's run.
            input_tokens: Prompt tokens consumed.
            output_tokens: Completion tokens generated.
            team_id: Team identifier, if any.

        Returns:
            BudgetExceededEvent if a budget limit was crossed, else None.
            The caller should pause the agent and/or switch models on receipt.
        """
        if agent_name in self._paused_agents:
            logger.warning(
                "record_turn called for paused agent '%s' — still recording but budget already exceeded",
                agent_name,
            )

        inp, out = self._price_for_model(model)
        cost = (input_tokens * inp + output_tokens * out) / 1_000_000

        record = TurnRecord(
            agent_name=agent_name,
            team_id=team_id,
            model=model,
            turn_number=turn_number,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._records.append(record)
        logger.debug(record.to_log_line())

        # Accumulate agent totals
        self._agent_tokens_in[agent_name] = self._agent_tokens_in.get(agent_name, 0) + input_tokens
        self._agent_tokens_out[agent_name] = self._agent_tokens_out.get(agent_name, 0) + output_tokens
        self._agent_cost[agent_name] = self._agent_cost.get(agent_name, 0.0) + cost

        # Accumulate team totals
        if team_id:
            self._team_tokens_in[team_id] = self._team_tokens_in.get(team_id, 0) + input_tokens
            self._team_tokens_out[team_id] = self._team_tokens_out.get(team_id, 0) + output_tokens
            self._team_cost[team_id] = self._team_cost.get(team_id, 0.0) + cost

        # Accumulate model totals
        self._model_tokens_in[model] = self._model_tokens_in.get(model, 0) + input_tokens
        self._model_tokens_out[model] = self._model_tokens_out.get(model, 0) + output_tokens
        self._model_cost[model] = self._model_cost.get(model, 0.0) + cost

        # Run totals
        self._run_tokens_in += input_tokens
        self._run_tokens_out += output_tokens
        self._run_cost += cost

        # Budget enforcement
        return self._check_budget(agent_name, team_id)

    def _check_budget(
        self, agent_name: str, team_id: str | None
    ) -> BudgetExceededEvent | None:
        """Check all budget limits after a turn. Returns first exceeded limit."""
        p = self._policy

        # Per-agent limit
        if p.max_cost_per_agent is not None:
            agent_total = self._agent_cost.get(agent_name, 0.0)
            if agent_total > p.max_cost_per_agent:
                self._paused_agents.add(agent_name)
                event = BudgetExceededEvent(
                    limit_type="agent",
                    agent_name=agent_name,
                    team_id=team_id,
                    limit_usd=p.max_cost_per_agent,
                    actual_usd=agent_total,
                    suggested_model=p.auto_switch_model,
                )
                logger.warning(
                    "Budget exceeded: agent=%s limit=$%.4f actual=$%.4f%s",
                    agent_name, p.max_cost_per_agent, agent_total,
                    f" → suggest model={p.auto_switch_model}" if p.auto_switch_model else "",
                )
                return event

        # Per-team limit
        if team_id and p.max_cost_per_team is not None:
            team_total = self._team_cost.get(team_id, 0.0)
            if team_total > p.max_cost_per_team:
                self._paused_agents.add(agent_name)
                event = BudgetExceededEvent(
                    limit_type="team",
                    agent_name=agent_name,
                    team_id=team_id,
                    limit_usd=p.max_cost_per_team,
                    actual_usd=team_total,
                    suggested_model=p.auto_switch_model,
                )
                logger.warning(
                    "Budget exceeded: team=%s limit=$%.4f actual=$%.4f",
                    team_id, p.max_cost_per_team, team_total,
                )
                return event

        # Per-run limit
        if p.max_cost_per_run is not None and self._run_cost > p.max_cost_per_run:
            self._paused_agents.add(agent_name)
            event = BudgetExceededEvent(
                limit_type="run",
                agent_name=agent_name,
                team_id=team_id,
                limit_usd=p.max_cost_per_run,
                actual_usd=self._run_cost,
                suggested_model=p.auto_switch_model,
            )
            logger.warning(
                "Budget exceeded: run limit=$%.4f actual=$%.4f",
                p.max_cost_per_run, self._run_cost,
            )
            return event

        return None

    # ── State management ──────────────────────────────────────────────────────

    def is_paused(self, agent_name: str) -> bool:
        """Return True if the agent has been paused due to a budget limit."""
        return agent_name in self._paused_agents

    def resume_agent(self, agent_name: str) -> None:
        """Clear the paused state for an agent (e.g. after switching to a cheaper model)."""
        self._paused_agents.discard(agent_name)
        logger.info("Agent '%s' budget-pause cleared", agent_name)

    # ── Summary methods ───────────────────────────────────────────────────────

    def cost_by_agent(self) -> dict[str, dict[str, Any]]:
        """Return cost and token breakdown per agent.

        Returns:
            {agent_name: {input_tokens, output_tokens, cost_usd, paused}}
        """
        result = {}
        for name in self._agent_cost:
            result[name] = {
                "input_tokens": self._agent_tokens_in.get(name, 0),
                "output_tokens": self._agent_tokens_out.get(name, 0),
                "cost_usd": round(self._agent_cost[name], 8),
                "paused": name in self._paused_agents,
            }
        return result

    def cost_by_model(self) -> dict[str, dict[str, Any]]:
        """Return cost and token breakdown per model.

        Returns:
            {model: {input_tokens, output_tokens, cost_usd}}
        """
        result = {}
        for model in self._model_cost:
            result[model] = {
                "input_tokens": self._model_tokens_in.get(model, 0),
                "output_tokens": self._model_tokens_out.get(model, 0),
                "cost_usd": round(self._model_cost[model], 8),
            }
        return result

    def cost_by_team(self) -> dict[str, dict[str, Any]]:
        """Return cost and token breakdown per team.

        Returns:
            {team_id: {input_tokens, output_tokens, cost_usd}}
        """
        result = {}
        for team_id in self._team_cost:
            result[team_id] = {
                "input_tokens": self._team_tokens_in.get(team_id, 0),
                "output_tokens": self._team_tokens_out.get(team_id, 0),
                "cost_usd": round(self._team_cost[team_id], 8),
            }
        return result

    def total_cost(self) -> dict[str, Any]:
        """Return run-level totals.

        Returns:
            {input_tokens, output_tokens, cost_usd, turn_count}
        """
        return {
            "input_tokens": self._run_tokens_in,
            "output_tokens": self._run_tokens_out,
            "cost_usd": round(self._run_cost, 8),
            "turn_count": len(self._records),
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize full tracker state to a JSON string."""
        payload = {
            "summary": {
                "total": self.total_cost(),
                "by_agent": self.cost_by_agent(),
                "by_model": self.cost_by_model(),
                "by_team": self.cost_by_team(),
            },
            "policy": {
                "max_cost_per_agent": self._policy.max_cost_per_agent,
                "max_cost_per_team": self._policy.max_cost_per_team,
                "max_cost_per_run": self._policy.max_cost_per_run,
                "auto_switch_model": self._policy.auto_switch_model,
            },
            "paused_agents": sorted(self._paused_agents),
            "records": [r.to_dict() for r in self._records],
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return json.dumps(payload, indent=2)

    def save(self, path: str | Path) -> None:
        """Write tracker state to a JSON file.

        Args:
            path: File path to write.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        logger.info("CostTracker state saved to %s", p)

    def log_summary(self) -> None:
        """Emit a structured summary to the logger at INFO level."""
        totals = self.total_cost()
        logger.info(
            "CostTracker summary: turns=%d input_tok=%d output_tok=%d total_usd=%.6f",
            totals["turn_count"],
            totals["input_tokens"],
            totals["output_tokens"],
            totals["cost_usd"],
        )
        for agent, stats in self.cost_by_agent().items():
            logger.info(
                "  agent=%-20s in_tok=%6d out_tok=%6d cost=$%.6f%s",
                agent,
                stats["input_tokens"],
                stats["output_tokens"],
                stats["cost_usd"],
                " [PAUSED]" if stats["paused"] else "",
            )


__all__ = [
    "CostTracker",
    "BudgetPolicy",
    "BudgetExceededEvent",
    "TurnRecord",
    "estimate_cost",
]
