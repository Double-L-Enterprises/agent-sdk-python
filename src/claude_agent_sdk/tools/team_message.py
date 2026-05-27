"""Team communication tools for multi-agent teams.

These tools are injected into an agent's tool set when it is part of a team.
They allow agents to send messages to each other and read their inbox.

Tools:
  - SendTeamMessage: Send a message to a named agent (or broadcast with "*")
  - ReadTeamMessages: Check for new messages from other team members

Usage (injected by TeamManager)::
    tools = default_tools() + team_tools()
    runner = AutonomousRunner(tools=tools, ...)
    result = await runner.run(task="...", message_bus=bus, agent_name="planner")

Created: 2026-05-27 CST
"""

from __future__ import annotations

import time
from typing import Any

from ..message_bus import Message, MessageBus


# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

SEND_TEAM_MESSAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "SendTeamMessage",
        "description": (
            "Send a message to another agent on your team. "
            "Use for discussion, code review requests, questions, sharing decisions, "
            "or reporting task completion. "
            "Use to='*' to broadcast to all team members."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Agent name to send to, or '*' for broadcast to all team members",
                },
                "content": {
                    "type": "string",
                    "description": "Message content. Be specific. Include file paths, decisions, and next steps.",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Optional thread ID for continuing an existing conversation thread",
                },
            },
            "required": ["to", "content"],
        },
    },
}

READ_TEAM_MESSAGES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ReadTeamMessages",
        "description": (
            "Check for new messages from other team members addressed to you. "
            "Call this periodically between tasks to receive assignments and updates. "
            "Returns 'No new messages' if inbox is empty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "ISO-8601 timestamp to get messages since. Omit to get all unread messages.",
                },
            },
        },
    },
}


def team_tools() -> list[dict[str, Any]]:
    """Return team communication tool definitions.

    Add to an agent's tool list when it is part of a team::

        tools = default_tools() + team_tools()

    Returns:
        List of two tool definitions: SendTeamMessage and ReadTeamMessages.
    """
    return [SEND_TEAM_MESSAGE_TOOL, READ_TEAM_MESSAGES_TOOL]


# ── Tool executors ─────────────────────────────────────────────────────────────

async def send_team_message(params: dict[str, Any], **kwargs: Any) -> str:
    """Execute the SendTeamMessage tool.

    Args:
        params: Tool input — must contain 'to' and 'content'; optionally 'thread_id'.
        **kwargs: Must contain 'message_bus' (MessageBus) and 'agent_name' (str).

    Returns:
        Confirmation string with delivery status.
    """
    bus: MessageBus | None = kwargs.get("message_bus")
    agent_name: str = kwargs.get("agent_name", "unknown")

    if bus is None:
        return "[ERROR] SendTeamMessage: no message_bus in context. Agent is not part of a team."

    to = params.get("to", "")
    content = params.get("content", "")
    thread_id = params.get("thread_id")

    if not to:
        return "[ERROR] SendTeamMessage: 'to' is required"
    if not content:
        return "[ERROR] SendTeamMessage: 'content' is required"

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    msg = Message(
        from_agent=agent_name,
        to_agent=to,
        content=content,
        timestamp=now_iso,
        thread_id=thread_id,
    )

    try:
        bus.send(msg)
    except Exception as exc:
        return f"[ERROR] SendTeamMessage: failed to send — {exc}"

    if to == "*":
        recipients = ", ".join(bus.agent_names())
        return (
            f"Message broadcast to all team members ({recipients}). "
            f"message_id={msg.message_id}"
        )
    return f"Message sent to {to}. message_id={msg.message_id}"


async def read_team_messages(params: dict[str, Any], **kwargs: Any) -> str:
    """Execute the ReadTeamMessages tool.

    Reads all unread messages from the agent's inbox. Messages are consumed
    (marked as read) on retrieval — each message is delivered exactly once.

    Args:
        params: Tool input — optionally contains 'since' (ISO-8601 timestamp).
        **kwargs: Must contain 'message_bus' (MessageBus) and 'agent_name' (str).

    Returns:
        Formatted string of messages, or "No new messages."
    """
    bus: MessageBus | None = kwargs.get("message_bus")
    agent_name: str = kwargs.get("agent_name", "unknown")

    if bus is None:
        return "[ERROR] ReadTeamMessages: no message_bus in context. Agent is not part of a team."

    since = params.get("since")

    try:
        messages = bus.receive(agent_name, since=since)
    except Exception as exc:
        return f"[ERROR] ReadTeamMessages: failed to read inbox — {exc}"

    if not messages:
        return "No new messages."

    lines = [f"You have {len(messages)} new message(s):\n"]
    for i, msg in enumerate(messages, 1):
        thread_info = f" [thread: {msg.thread_id}]" if msg.thread_id else ""
        lines.append(f"--- Message {i} ---")
        lines.append(f"From: {msg.from_agent}{thread_info}")
        lines.append(f"Sent: {msg.timestamp}")
        lines.append(f"ID:   {msg.message_id}")
        lines.append(f"\n{msg.content}\n")

    return "\n".join(lines)


__all__ = [
    "SEND_TEAM_MESSAGE_TOOL",
    "READ_TEAM_MESSAGES_TOOL",
    "team_tools",
    "send_team_message",
    "read_team_messages",
]
