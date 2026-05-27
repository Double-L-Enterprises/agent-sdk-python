"""Observability — Prometheus-compatible metrics and run history for multi-agent teams.

TeamMetrics exposes agent/team telemetry in Prometheus text format.
RunHistory persists completed run data as JSON files for post-hoc analysis.

Integration points:
  - bridge_sdk.py: GET /metrics → metrics.expose_metrics()
  - TeamManager: call record_* methods at lifecycle points
  - LiteLLMHTTPTransport: call record_turn/record_tool after each API call

Created: 2026-05-27 CST
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

logger = logging.getLogger(__name__)

# ─── Agent state codes (gauge values) ─────────────────────────────────────────
AGENT_STATE_IDLE = 0
AGENT_STATE_RUNNING = 1
AGENT_STATE_STALLED = 2
AGENT_STATE_DONE = 3

_STATE_LABELS = {
    "idle": AGENT_STATE_IDLE,
    "running": AGENT_STATE_RUNNING,
    "stalled": AGENT_STATE_STALLED,
    "done": AGENT_STATE_DONE,
    "completed": AGENT_STATE_DONE,
    "error": AGENT_STATE_DONE,
    "stopped": AGENT_STATE_DONE,
    "pending": AGENT_STATE_IDLE,
}


# ─── Counter / Gauge / Histogram storage ──────────────────────────────────────


@dataclass
class _Counter:
    """Simple monotonically increasing counter."""

    name: str
    help_text: str
    labels: dict[str, float] = field(default_factory=dict)  # label_str → value

    def inc(self, label_str: str, amount: float = 1.0) -> None:
        self.labels[label_str] = self.labels.get(label_str, 0.0) + amount

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        for lbl, val in sorted(self.labels.items()):
            lines.append(f"{self.name}{{{lbl}}} {val}")
        return "\n".join(lines)


@dataclass
class _Gauge:
    """Current-value gauge."""

    name: str
    help_text: str
    labels: dict[str, float] = field(default_factory=dict)  # label_str → value

    def set(self, label_str: str, value: float) -> None:
        self.labels[label_str] = value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        for lbl, val in sorted(self.labels.items()):
            lines.append(f"{self.name}{{{lbl}}} {val}")
        return "\n".join(lines)


@dataclass
class _HistogramBucket:
    count: int = 0
    total: float = 0.0
    buckets: dict[float, int] = field(default_factory=dict)  # le → cumulative count


@dataclass
class _Histogram:
    """Duration histogram."""

    name: str
    help_text: str
    bucket_bounds: list[float] = field(
        default_factory=lambda: [
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
            float("inf"),
        ]
    )
    _data: dict[str, _HistogramBucket] = field(default_factory=dict)

    def observe(self, label_str: str, value: float) -> None:
        if label_str not in self._data:
            self._data[label_str] = _HistogramBucket(
                buckets=dict.fromkeys(self.bucket_bounds, 0)
            )
        d = self._data[label_str]
        d.count += 1
        d.total += value
        for bound in self.bucket_bounds:
            if value <= bound:
                d.buckets[bound] += 1

    def render(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        for lbl, d in sorted(self._data.items()):
            for bound, cnt in sorted(d.buckets.items(), key=lambda x: x[0]):
                le = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'{self.name}_bucket{{{lbl},le="{le}"}} {cnt}')
            lines.append(f"{self.name}_sum{{{lbl}}} {d.total}")
            lines.append(f"{self.name}_count{{{lbl}}} {d.count}")
        return "\n".join(lines)


def _fmt_labels(**kwargs: str) -> str:
    """Format keyword args as Prometheus label string: key="val",key2="val2"."""
    return ",".join(f'{k}="{v}"' for k, v in sorted(kwargs.items()))


# ─── TeamMetrics ───────────────────────────────────────────────────────────────


class TeamMetrics:
    """Collect and expose Prometheus-compatible metrics for agent teams.

    All metrics are in-memory. Designed for a single-process deployment.
    Thread-safe only for single-threaded asyncio usage (no locks needed there).

    Usage::
        metrics = TeamMetrics()
        metrics.record_turn(agent_name="planner", model="qwen/qwen3-max", team_id="t1")
        metrics.record_tool_call(agent_name="planner", tool_name="Bash")
        metrics.set_agent_state(agent_name="planner", state="running")
        print(metrics.expose_metrics())
    """

    def __init__(self) -> None:
        self._turns = _Counter(
            "agent_turns_total",
            "Total turns executed by each agent",
        )
        self._tool_calls = _Counter(
            "agent_tool_calls_total",
            "Total tool calls made by each agent",
        )
        self._cost = _Gauge(
            "agent_cost_dollars",
            "Estimated cost in USD for each agent",
        )
        self._team_messages = _Counter(
            "team_messages_total",
            "Total messages exchanged between agents",
        )
        self._agent_state = _Gauge(
            "agent_state",
            "Current agent state: 0=idle 1=running 2=stalled 3=done",
        )
        self._team_duration = _Histogram(
            "team_duration_seconds",
            "Total wall-clock time for completed team runs",
        )

        # Track team start times for duration calculation
        self._team_start: dict[str, float] = {}

    # ── Record methods ─────────────────────────────────────────────────────────

    def record_turn(
        self,
        agent_name: str,
        model: str,
        team_id: str,
        cost_delta: float = 0.0,
    ) -> None:
        """Record one completed agent turn.

        Args:
            agent_name: Agent identifier.
            model: Model used for this turn.
            team_id: Team this agent belongs to.
            cost_delta: Incremental cost in USD for this turn (0 if unknown).
        """
        lbl = _fmt_labels(agent_name=agent_name, model=model, team_id=team_id)
        self._turns.inc(lbl)
        if cost_delta > 0:
            cost_lbl = _fmt_labels(agent_name=agent_name, model=model)
            current = self._cost.labels.get(cost_lbl, 0.0)
            self._cost.set(cost_lbl, current + cost_delta)

    def record_tool_call(self, agent_name: str, tool_name: str) -> None:
        """Record one tool call execution.

        Args:
            agent_name: Agent that made the call.
            tool_name: Name of the tool invoked.
        """
        lbl = _fmt_labels(agent_name=agent_name, tool_name=tool_name)
        self._tool_calls.inc(lbl)

    def record_message(self, from_agent: str, to_agent: str) -> None:
        """Record one inter-agent message.

        Args:
            from_agent: Sender agent name.
            to_agent: Recipient agent name.
        """
        lbl = _fmt_labels(from_agent=from_agent, to_agent=to_agent)
        self._team_messages.inc(lbl)

    def set_agent_state(self, agent_name: str, state: str) -> None:
        """Update the current state of an agent.

        Args:
            agent_name: Agent identifier.
            state: One of: idle, running, stalled, done, completed, error, stopped, pending.
        """
        numeric = _STATE_LABELS.get(state, AGENT_STATE_IDLE)
        lbl = _fmt_labels(agent_name=agent_name)
        self._agent_state.set(lbl, float(numeric))

    def record_team_start(self, team_id: str) -> None:
        """Record the start time for a team run (used for duration calculation).

        Args:
            team_id: Team identifier.
        """
        self._team_start[team_id] = time.monotonic()

    def record_team_complete(self, team_id: str, template: str = "default") -> None:
        """Record the end of a team run and observe the duration histogram.

        Args:
            team_id: Team identifier.
            template: Template or workflow type used (for labeling).
        """
        start = self._team_start.pop(team_id, None)
        if start is not None:
            duration = time.monotonic() - start
            lbl = _fmt_labels(team_id=team_id, template=template)
            self._team_duration.observe(lbl, duration)

    def set_agent_cost(self, agent_name: str, model: str, total_cost: float) -> None:
        """Set the absolute total cost for an agent (use instead of record_turn's delta).

        Args:
            agent_name: Agent identifier.
            model: Model used.
            total_cost: Total cost in USD so far.
        """
        lbl = _fmt_labels(agent_name=agent_name, model=model)
        self._cost.set(lbl, total_cost)

    # ── Expose ─────────────────────────────────────────────────────────────────

    def expose_metrics(self) -> str:
        """Return all metrics in Prometheus text exposition format.

        Returns:
            Multi-line string ready to serve at /metrics.
        """
        sections = [
            self._turns.render(),
            self._tool_calls.render(),
            self._cost.render(),
            self._team_messages.render(),
            self._agent_state.render(),
            self._team_duration.render(),
        ]
        return "\n\n".join(s for s in sections if s.strip()) + "\n"

    def reset(self) -> None:
        """Clear all metrics. Useful for testing."""
        self.__init__()  # type: ignore[misc]


# ─── RunHistory ────────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    """Stored data for a single completed team run."""

    run_id: str
    team_id: str
    timestamp: str
    duration_seconds: float
    agents: list[dict[str, Any]]  # [{name, model, status, turns, cost}]
    messages: list[dict[str, Any]]  # inter-agent messages from the bus
    costs: dict[str, float]  # agent_name → cost in USD
    result: str  # "success" | "timeout" | "error" | summary text
    agent_conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # ^ agent_name → list of {role, content} messages


class RunHistory:
    """Persist and retrieve completed agent run data.

    Storage layout::
        ~/logs/agent-runs/{team_id}/{run_id}.json

    Usage::
        history = RunHistory()
        run_id = history.save_run(
            team_id="my-team",
            agents=[{"name": "planner", "model": "qwen/qwen3-max", "status": "completed", "turns": 12, "cost": 0.002}],
            messages=[],
            costs={"planner": 0.002},
            duration=45.3,
            result="success",
        )
        runs = history.list_runs(limit=10)
        detail = history.get_run(run_id)
    """

    def __init__(self, base_dir: str = "~/logs/agent-runs") -> None:
        """Initialize RunHistory.

        Args:
            base_dir: Root directory for run JSON files. Tilde-expanded.
        """
        self._base_dir = Path(os.path.expanduser(base_dir))
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_run(
        self,
        team_id: str,
        agents: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        costs: dict[str, float],
        duration: float,
        result: str,
        agent_conversations: dict[str, list[dict[str, Any]]] | None = None,
    ) -> str:
        """Persist a completed run to disk.

        Args:
            team_id: Team identifier.
            agents: List of agent summary dicts (name, model, status, turns, cost, etc.).
            messages: Inter-agent messages exchanged during the run.
            costs: Dict of agent_name → USD cost.
            duration: Wall-clock run duration in seconds.
            result: High-level result string ("success", "timeout", "error", or text summary).
            agent_conversations: Optional per-agent conversation history.

        Returns:
            The generated run_id (UUID hex).
        """
        run_id = uuid.uuid4().hex
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        record = RunRecord(
            run_id=run_id,
            team_id=team_id,
            timestamp=timestamp,
            duration_seconds=duration,
            agents=agents,
            messages=messages,
            costs=costs,
            result=result,
            agent_conversations=agent_conversations or {},
        )

        team_dir = self._base_dir / team_id
        team_dir.mkdir(parents=True, exist_ok=True)
        run_file = team_dir / f"{run_id}.json"

        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(self._record_to_dict(record), f, indent=2)
            logger.info("Run saved: %s/%s", team_id, run_id)
        except Exception as exc:
            logger.error("Failed to save run %s: %s", run_id, exc)

        return run_id

    # ── Read ───────────────────────────────────────────────────────────────────

    def list_runs(
        self,
        since: float | None = None,
        limit: int = 50,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent completed runs (summary only, no conversations).

        Args:
            since: Unix timestamp. If set, only runs after this time are returned.
            limit: Maximum number of runs to return (newest first).
            team_id: If set, filter to runs for this team only.

        Returns:
            List of run summary dicts, newest first.
        """
        results: list[dict[str, Any]] = []

        search_dirs = []
        if team_id:
            d = self._base_dir / team_id
            if d.is_dir():
                search_dirs.append(d)
        else:
            search_dirs = [d for d in self._base_dir.iterdir() if d.is_dir()]

        for team_dir in search_dirs:
            for run_file in sorted(
                team_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                try:
                    with open(run_file, encoding="utf-8") as f:
                        data = json.load(f)
                    # Check since filter
                    if since is not None:
                        ts_epoch = self._iso_to_epoch(data.get("timestamp", ""))
                        if ts_epoch < since:
                            continue
                    # Return summary (no agent_conversations)
                    summary = {
                        k: v for k, v in data.items() if k != "agent_conversations"
                    }
                    results.append(summary)
                    if len(results) >= limit:
                        break
                except Exception as exc:
                    logger.warning("Skipping corrupt run file %s: %s", run_file, exc)

        # Sort across all teams by timestamp (newest first), then cap at limit
        results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return results[:limit]

    def get_run(self, run_id: str, team_id: str | None = None) -> dict[str, Any] | None:
        """Retrieve full run detail including agent conversations.

        Args:
            run_id: The run UUID hex string.
            team_id: Optional team ID to narrow the search directory.

        Returns:
            Full run dict including agent_conversations, or None if not found.
        """
        search_dirs = []
        if team_id:
            d = self._base_dir / team_id
            if d.is_dir():
                search_dirs.append(d)
        else:
            search_dirs = [d for d in self._base_dir.iterdir() if d.is_dir()]

        for team_dir in search_dirs:
            run_file = team_dir / f"{run_id}.json"
            if run_file.exists():
                try:
                    with open(run_file, encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:
                    logger.error("Failed to read run %s: %s", run_id, exc)
                    return None

        logger.warning("Run not found: %s", run_id)
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(r: RunRecord) -> dict[str, Any]:
        return {
            "run_id": r.run_id,
            "team_id": r.team_id,
            "timestamp": r.timestamp,
            "duration_seconds": r.duration_seconds,
            "agents": r.agents,
            "messages": r.messages,
            "costs": r.costs,
            "result": r.result,
            "agent_conversations": r.agent_conversations,
        }

    @staticmethod
    def _iso_to_epoch(iso: str) -> float:
        """Convert ISO 8601 string to Unix timestamp. Returns 0.0 on parse error."""
        try:
            import datetime

            dt = datetime.datetime.fromisoformat(iso)
            return dt.timestamp()
        except Exception:
            return 0.0


# ─── Module-level singletons (opt-in) ─────────────────────────────────────────

#: Global metrics instance. Import and use directly, or create your own.
global_metrics = TeamMetrics()

#: Global run history instance.
global_history = RunHistory()


__all__ = [
    "TeamMetrics",
    "RunHistory",
    "RunRecord",
    "AGENT_STATE_IDLE",
    "AGENT_STATE_RUNNING",
    "AGENT_STATE_STALLED",
    "AGENT_STATE_DONE",
    "global_metrics",
    "global_history",
]
