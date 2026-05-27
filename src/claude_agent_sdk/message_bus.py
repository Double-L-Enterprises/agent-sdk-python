"""File-based message bus for multi-agent team communication.

Each agent's inbox is a directory:
  /tmp/agent-teams/{team_id}/messages/{agent_name}/

Messages are JSON files named: {timestamp_ms}_{message_id}.json
Read messages are moved to a .read/ subdirectory for audit trail.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """A single message exchanged between team agents."""

    from_agent: str
    to_agent: str  # agent name, or "*" for broadcast
    content: str
    timestamp: str  # ISO-8601
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            content=data["content"],
            timestamp=data["timestamp"],
            message_id=data.get("message_id", str(uuid.uuid4())),
            thread_id=data.get("thread_id"),
            timestamp_ms=data.get("timestamp_ms", 0),
        )


class MessageBus:
    """File-based message bus for agent team communication.

    Simple, no-dependency design: messages are JSON files in directories.
    Works in tmux, survives restarts, no Redis or network required.

    Directory layout:
      {bus_dir}/{agent_name}/          — unread inbox
      {bus_dir}/{agent_name}/.read/    — read messages (archive)
      {bus_dir}/_all/                  — broadcast copies for monitoring

    Usage::
        bus = MessageBus(team_id="my-team")
        msg = Message(
            from_agent="planner",
            to_agent="backend-dev",
            content="Start building the API",
            timestamp="2026-05-27T10:00:00",
        )
        bus.send(msg)
        messages = bus.receive("backend-dev")
    """

    def __init__(self, team_id: str, bus_dir: str | None = None) -> None:
        """Initialize the message bus.

        Args:
            team_id: Unique identifier for the team. Used to namespace directories.
            bus_dir: Base directory for message storage. Defaults to
                     /tmp/agent-teams/{team_id}/messages
        """
        self.team_id = team_id
        self.bus_dir = Path(bus_dir or f"/tmp/agent-teams/{team_id}/messages")
        self.bus_dir.mkdir(parents=True, exist_ok=True)

        # Global _all directory for monitoring all messages
        self._all_dir = self.bus_dir / "_all"
        self._all_dir.mkdir(exist_ok=True)

    def _agent_dir(self, agent_name: str) -> Path:
        """Get or create the inbox directory for an agent."""
        d = self.bus_dir / agent_name
        d.mkdir(parents=True, exist_ok=True)
        read_dir = d / ".read"
        read_dir.mkdir(exist_ok=True)
        return d

    def _msg_filename(self, msg: Message) -> str:
        """Generate a sortable filename for a message."""
        return f"{msg.timestamp_ms:016d}_{msg.message_id}.json"

    def send(self, msg: Message) -> None:
        """Write a message to the recipient's inbox.

        For broadcast messages (to_agent="*"), writes to all known agent inboxes.
        Also writes a copy to _all/ for monitoring.

        Args:
            msg: The message to send.
        """
        data = json.dumps(msg.to_dict(), indent=2)
        filename = self._msg_filename(msg)

        if msg.to_agent == "*":
            # Broadcast: write to all existing agent inboxes
            for agent_dir in self.bus_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith("_"):
                    (agent_dir / filename).write_text(data)
        else:
            # Direct message
            dest = self._agent_dir(msg.to_agent)
            (dest / filename).write_text(data)

        # Write to global _all for monitoring
        (self._all_dir / filename).write_text(data)

    def receive(self, agent_name: str, since: str | None = None) -> list[Message]:
        """Read all unread messages for an agent. Marks them as read.

        Messages are moved to .read/ after being returned. This is the
        "destructive read" model — each message is delivered exactly once.

        Args:
            agent_name: The agent whose inbox to read.
            since: ISO-8601 timestamp. If provided, only returns messages
                   with timestamp >= since. Compares against timestamp_ms
                   for reliability.

        Returns:
            List of messages in chronological order (oldest first).
        """
        inbox = self._agent_dir(agent_name)
        read_dir = inbox / ".read"

        since_ms: int = 0
        if since:
            # Parse ISO timestamp to ms — approximate, good enough for filtering
            try:
                import datetime

                dt = datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))
                since_ms = int(dt.timestamp() * 1000)
            except Exception:
                since_ms = 0

        messages: list[Message] = []

        # Read all .json files in inbox (not .read/)
        msg_files = sorted(inbox.glob("*.json"))  # sorted = chronological by filename
        for msg_file in msg_files:
            try:
                data = json.loads(msg_file.read_text())
                msg = Message.from_dict(data)

                if since_ms and msg.timestamp_ms < since_ms:
                    # Move to read even if filtered (don't re-deliver old messages)
                    msg_file.rename(read_dir / msg_file.name)
                    continue

                messages.append(msg)
                # Mark as read by moving to .read/
                msg_file.rename(read_dir / msg_file.name)

            except (json.JSONDecodeError, KeyError, OSError):
                # Corrupt or deleted file — skip
                continue

        return messages

    def get_thread(self, thread_id: str) -> list[Message]:
        """Get all messages in a conversation thread.

        Searches both unread inbox and .read/ archives across all agents.

        Args:
            thread_id: Thread identifier to filter by.

        Returns:
            All messages in the thread, sorted by timestamp_ms.
        """
        messages: list[Message] = []

        # Search all agent directories
        for agent_dir in self.bus_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            # Search unread and read messages
            for search_dir in [agent_dir, agent_dir / ".read"]:
                if not search_dir.exists():
                    continue
                for msg_file in search_dir.glob("*.json"):
                    try:
                        data = json.loads(msg_file.read_text())
                        if data.get("thread_id") == thread_id:
                            messages.append(Message.from_dict(data))
                    except (json.JSONDecodeError, KeyError, OSError):
                        continue

        # Deduplicate by message_id
        seen: set[str] = set()
        unique: list[Message] = []
        for msg in messages:
            if msg.message_id not in seen:
                seen.add(msg.message_id)
                unique.append(msg)

        return sorted(unique, key=lambda m: m.timestamp_ms)

    def get_all(self, since: str | None = None) -> list[Message]:
        """Get all messages from the _all monitoring directory.

        Used by the orchestrator to watch all inter-agent communication.

        Args:
            since: ISO-8601 timestamp. Only return messages at or after this time.

        Returns:
            All messages in chronological order.
        """
        since_ms: int = 0
        if since:
            try:
                import datetime

                dt = datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))
                since_ms = int(dt.timestamp() * 1000)
            except Exception:
                since_ms = 0

        messages: list[Message] = []
        for msg_file in sorted(self._all_dir.glob("*.json")):
            try:
                data = json.loads(msg_file.read_text())
                msg = Message.from_dict(data)
                if not since_ms or msg.timestamp_ms >= since_ms:
                    messages.append(msg)
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return messages

    def agent_names(self) -> list[str]:
        """Return the names of all agents with known inboxes."""
        names = []
        for d in self.bus_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                names.append(d.name)
        return sorted(names)

    def clear_agent(self, agent_name: str) -> None:
        """Remove all unread messages for an agent. Used for cleanup/testing."""
        inbox = self._agent_dir(agent_name)
        for msg_file in inbox.glob("*.json"):
            try:
                msg_file.unlink()
            except OSError:
                pass


__all__ = ["Message", "MessageBus"]
