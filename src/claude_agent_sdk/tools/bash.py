"""Bash tool — runs shell commands via asyncio.subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT_BYTES = 100_000  # 100 KB per stream

BASH_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": (
            "Run a shell command and return its stdout/stderr. "
            "Commands run in a subprocess with a configurable timeout. "
            "Use for: installing packages, running scripts, git operations, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command.",
                },
            },
            "required": ["command"],
        },
    },
}


async def execute_bash(params: dict[str, Any], cwd: str | None = None) -> str:
    """Execute a shell command and return combined stdout+stderr output.

    Args:
        params: Tool parameters dict (command, timeout, cwd).
        cwd: Default working directory; overridden by params['cwd'] if present.

    Returns:
        Combined output string. On timeout or error, includes a descriptive
        error prefix.
    """
    command = params.get("command")
    if not command:
        return "[ERROR] Bash requires 'command' parameter"

    timeout = int(params.get("timeout") or _DEFAULT_TIMEOUT)
    working_dir = params.get("cwd") or cwd

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            env=os.environ.copy(),
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"[ERROR] Command timed out after {timeout}s\nCommand: {command}"

        stdout = _truncate(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = _truncate(stderr_bytes.decode("utf-8", errors="replace"))
        rc = proc.returncode

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if rc != 0:
            parts.append(f"[exit code: {rc}]")

        return "\n".join(parts) if parts else "(no output)"

    except FileNotFoundError as exc:
        return f"[ERROR] Shell not found: {exc}"
    except OSError as exc:
        return f"[ERROR] Could not execute command: {exc}"


def _truncate(text: str) -> str:
    if len(text.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        truncated = text.encode("utf-8")[:_MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="ignore"
        )
        return truncated + "\n[...output truncated]"
    return text
