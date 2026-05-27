"""AgentMemory — persistent conversation state and shared knowledge base.

Provides two memory scopes:
  - Per-agent private memory: conversation checkpoints + private notes only
    that specific agent can read.
  - Team shared memory: a common knowledge base all agents on a team can
    query and contribute to.

Storage layout under {output_dir}/.agent-memory/:
    .agent-memory/
        _team/                  ← shared knowledge base
            knowledge.json      ← key-value store
            notes.md            ← freeform append-only notes
        {agent_name}/
            conversation.json   ← full message history + metadata
            private_notes.json  ← per-agent key-value notes

Integration with TeamManager:
  - auto_save=True (default): TeamManager calls memory.save_state() after each turn.
  - On agent restart: TeamManager calls memory.load_state() and passes the restored
    conversation to a new AutonomousRunner via its initial messages list.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _read_json(path: Path, default: Any) -> Any:
    """Read a JSON file, returning default on any error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("AgentMemory: failed to read %s: %s", path, exc)
    return default


def _write_json(path: Path, data: Any) -> None:
    """Atomically write data as JSON (write to .tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.error("AgentMemory: failed to write %s: %s", path, exc)
        raise


# ── AgentMemory ───────────────────────────────────────────────────────────────

class AgentMemory:
    """Persists agent conversation state and shared team knowledge to disk.

    Usage::
        memory = AgentMemory(output_dir="/path/to/team-output")

        # Save agent state after each turn
        memory.save_state(
            agent_name="backend",
            conversation=[{"role": "user", ...}, ...],
            metadata={"model": "qwen/qwen3-max", "turn": 5},
        )

        # Restore on restart
        state = memory.load_state("backend")
        if state:
            conversation = state["conversation"]
            metadata = state["metadata"]

        # Shared team knowledge
        memory.team_set("api_spec_path", "/tmp/api-spec.md")
        spec_path = memory.team_get("api_spec_path")

        memory.team_append_note("backend", "Discovered: auth uses JWT, not sessions.")

        # Per-agent private notes
        memory.agent_set_note("backend", "retry_count", 3)
        count = memory.agent_get_note("backend", "retry_count", default=0)
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize AgentMemory.

        Args:
            output_dir: Root directory for this team's output.  Memory files
                        are stored under {output_dir}/.agent-memory/.
        """
        self._root = Path(output_dir).expanduser().resolve() / ".agent-memory"
        self._team_dir = self._root / "_team"
        self._team_dir.mkdir(parents=True, exist_ok=True)

        # Initialize team knowledge file if absent
        self._team_knowledge_path = self._team_dir / "knowledge.json"
        if not self._team_knowledge_path.exists():
            _write_json(self._team_knowledge_path, {})

        # Team notes file (append-only markdown)
        self._team_notes_path = self._team_dir / "notes.md"
        if not self._team_notes_path.exists():
            self._team_notes_path.write_text(
                f"# Team Shared Notes\n# Created: {_now()}\n\n", encoding="utf-8"
            )

    # ── Agent directory ───────────────────────────────────────────────────────

    def _agent_dir(self, agent_name: str) -> Path:
        """Return (and create) the per-agent memory directory."""
        d = self._root / agent_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Conversation state ────────────────────────────────────────────────────

    def save_state(
        self,
        agent_name: str,
        conversation: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an agent's conversation and metadata to disk.

        Args:
            agent_name: Unique name of the agent.
            conversation: Full message list (OpenAI format: role/content dicts).
            metadata: Arbitrary key-value metadata (model, turn, task, etc.).
        """
        path = self._agent_dir(agent_name) / "conversation.json"
        payload: dict[str, Any] = {
            "agent_name": agent_name,
            "conversation": conversation,
            "metadata": metadata or {},
            "saved_at": _now(),
            "message_count": len(conversation),
        }
        _write_json(path, payload)
        logger.debug(
            "AgentMemory: saved state for '%s' (%d messages)", agent_name, len(conversation)
        )

    def load_state(self, agent_name: str) -> dict[str, Any] | None:
        """Restore an agent's conversation and metadata from disk.

        Returns:
            Dict with keys 'conversation', 'metadata', 'saved_at', 'message_count',
            or None if no checkpoint exists.
        """
        path = self._agent_dir(agent_name) / "conversation.json"
        if not path.exists():
            logger.debug("AgentMemory: no checkpoint for '%s'", agent_name)
            return None
        data = _read_json(path, None)
        if data is None:
            return None
        logger.info(
            "AgentMemory: loaded state for '%s' (%d messages, saved %s)",
            agent_name,
            data.get("message_count", 0),
            data.get("saved_at", "unknown"),
        )
        return data

    def has_state(self, agent_name: str) -> bool:
        """Return True if a persisted checkpoint exists for this agent."""
        return (self._agent_dir(agent_name) / "conversation.json").exists()

    def delete_state(self, agent_name: str) -> None:
        """Remove the persisted conversation for an agent (does not delete notes)."""
        path = self._agent_dir(agent_name) / "conversation.json"
        if path.exists():
            path.unlink()
            logger.info("AgentMemory: deleted state for '%s'", agent_name)

    # ── Team shared knowledge ─────────────────────────────────────────────────

    def team_set(self, key: str, value: Any) -> None:
        """Write a key-value entry to the team's shared knowledge base.

        Thread-safety: uses atomic rename for safe concurrent writes by
        multiple agents running in separate asyncio tasks in the same process.
        Cross-process safety: best-effort; same-second concurrent writes may
        result in last-writer-wins.

        Args:
            key: String key.
            value: JSON-serializable value.
        """
        data = _read_json(self._team_knowledge_path, {})
        data[key] = {
            "value": value,
            "updated_at": _now(),
        }
        _write_json(self._team_knowledge_path, data)

    def team_get(self, key: str, default: Any = None) -> Any:
        """Read a key from the team's shared knowledge base.

        Args:
            key: The key to look up.
            default: Value to return if key is absent.

        Returns:
            The stored value, or default.
        """
        data = _read_json(self._team_knowledge_path, {})
        entry = data.get(key)
        if entry is None:
            return default
        # Support both wrapped ({"value": ..., "updated_at": ...}) and raw values
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return entry

    def team_get_all(self) -> dict[str, Any]:
        """Return the full shared knowledge base (keys → values, unwrapped).

        Returns:
            Dict of {key: value} without metadata wrappers.
        """
        raw = _read_json(self._team_knowledge_path, {})
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "value" in v:
                result[k] = v["value"]
            else:
                result[k] = v
        return result

    def team_append_note(self, from_agent: str, note: str) -> None:
        """Append a freeform note to the team's shared notes file.

        Notes are append-only markdown and survive restarts.

        Args:
            from_agent: Name of the contributing agent.
            note: The note text to append.
        """
        entry = f"\n## [{_now()}] {from_agent}\n\n{note}\n"
        try:
            with self._team_notes_path.open("a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as exc:
            logger.error("AgentMemory: failed to append team note: %s", exc)

    def team_read_notes(self) -> str:
        """Return the full contents of the team's shared notes file."""
        try:
            return self._team_notes_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    # ── Per-agent private notes ───────────────────────────────────────────────

    def agent_set_note(self, agent_name: str, key: str, value: Any) -> None:
        """Write a private note for a specific agent.

        Only that agent should call this; other agents cannot read it.

        Args:
            agent_name: The owning agent.
            key: Note key.
            value: JSON-serializable value.
        """
        path = self._agent_dir(agent_name) / "private_notes.json"
        data = _read_json(path, {})
        data[key] = {
            "value": value,
            "updated_at": _now(),
        }
        _write_json(path, data)

    def agent_get_note(self, agent_name: str, key: str, default: Any = None) -> Any:
        """Read a private note for a specific agent.

        Args:
            agent_name: The owning agent.
            key: Note key.
            default: Value to return if key is absent.

        Returns:
            The stored value, or default.
        """
        path = self._agent_dir(agent_name) / "private_notes.json"
        data = _read_json(path, {})
        entry = data.get(key)
        if entry is None:
            return default
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return entry

    def agent_get_all_notes(self, agent_name: str) -> dict[str, Any]:
        """Return all private notes for an agent (keys → values, unwrapped)."""
        path = self._agent_dir(agent_name) / "private_notes.json"
        raw = _read_json(path, {})
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "value" in v:
                result[k] = v["value"]
            else:
                result[k] = v
        return result

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_agents(self) -> list[str]:
        """Return the names of all agents that have memory files."""
        if not self._root.exists():
            return []
        return [
            d.name
            for d in self._root.iterdir()
            if d.is_dir() and d.name != "_team"
        ]

    def status(self) -> dict[str, Any]:
        """Return a summary of memory state across all agents and the team.

        Returns:
            Dict with agents list, team knowledge key count, and team notes line count.
        """
        agents = []
        for name in self.list_agents():
            state = self.load_state(name)
            agents.append({
                "name": name,
                "has_conversation": state is not None,
                "message_count": state["message_count"] if state else 0,
                "saved_at": state["saved_at"] if state else None,
            })

        team_knowledge = _read_json(self._team_knowledge_path, {})
        team_notes = self.team_read_notes()

        return {
            "memory_root": str(self._root),
            "agents": agents,
            "team_knowledge_keys": len(team_knowledge),
            "team_notes_lines": len(team_notes.splitlines()),
        }

    def clear_all(self) -> None:
        """Delete all memory files (conversation + notes + team knowledge).

        Use with caution: irreversible.  Does not remove directories.
        """
        import shutil
        if self._root.exists():
            shutil.rmtree(self._root)
            self._root.mkdir(parents=True, exist_ok=True)
            self._team_dir.mkdir(parents=True, exist_ok=True)
            _write_json(self._team_knowledge_path, {})
            self._team_notes_path.write_text(
                f"# Team Shared Notes\n# Cleared: {_now()}\n\n", encoding="utf-8"
            )
        logger.warning("AgentMemory: all memory cleared")


__all__ = ["AgentMemory"]
