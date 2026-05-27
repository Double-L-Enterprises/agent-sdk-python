"""Tool definitions and dispatchers for the autonomous agent."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def default_tools() -> list[dict[str, Any]]:
    """Return the default set of tools for autonomous agents."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands on the local system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the content of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to read"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit a file by replacing specific content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to edit"},
                        "old_string": {"type": "string", "description": "Text to replace"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for text in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search"},
                        "path": {"type": "string", "description": "Path to search in"},
                    },
                    "required": ["pattern", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern to match"},
                        "path": {"type": "string", "description": "Base path to search"},
                    },
                    "required": ["pattern"],
                },
            },
        },
    ]


async def dispatch_tool(name: str, params: dict[str, Any], cwd: str | None = None) -> str:
    """Dispatch a tool call to the appropriate handler."""
    try:
        if name == "bash":
            return await _bash_tool(params, cwd=cwd)
        elif name == "read":
            return await _read_tool(params)
        elif name == "write":
            return await _write_tool(params)
        elif name == "edit":
            return await _edit_tool(params)
        elif name == "grep":
            return await _grep_tool(params)
        elif name == "glob":
            return await _glob_tool(params)
        else:
            return f"[ERROR] Unknown tool: {name}"
    except Exception as exc:
        logger.error("Tool dispatch error for %s: %s", name, exc)
        return f"[ERROR] {name} failed: {exc}"


async def _bash_tool(params: dict[str, Any], cwd: str | None = None) -> str:
    """Execute a bash command."""
    import subprocess

    command = params.get("command", "")
    if not command:
        return "[ERROR] bash: command is required"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"[ERROR] Command failed with exit code {result.returncode}:\n{result.stderr}"
        return result.stdout or "(no output)"
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 30 seconds"
    except Exception as exc:
        return f"[ERROR] bash failed: {exc}"


async def _read_tool(params: dict[str, Any]) -> str:
    """Read a file's content."""
    file_path = params.get("file_path", "")
    if not file_path:
        return "[ERROR] read: file_path is required"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:50000]  # Limit to 50KB
    except FileNotFoundError:
        return f"[ERROR] File not found: {file_path}"
    except Exception as exc:
        return f"[ERROR] read failed: {exc}"


async def _write_tool(params: dict[str, Any]) -> str:
    """Write content to a file."""
    file_path = params.get("file_path", "")
    content = params.get("content", "")

    if not file_path:
        return "[ERROR] write: file_path is required"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {file_path}"
    except Exception as exc:
        return f"[ERROR] write failed: {exc}"


async def _edit_tool(params: dict[str, Any]) -> str:
    """Edit a file by replacing content."""
    file_path = params.get("file_path", "")
    old_string = params.get("old_string", "")
    new_string = params.get("new_string", "")

    if not file_path or not old_string or new_string is None:
        return "[ERROR] edit: file_path, old_string, and new_string are required"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            return f"[ERROR] String to replace not found in {file_path}"

        new_content = content.replace(old_string, new_string, 1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Edited {file_path} successfully"
    except Exception as exc:
        return f"[ERROR] edit failed: {exc}"


async def _grep_tool(params: dict[str, Any]) -> str:
    """Search for text in files."""
    import subprocess

    pattern = params.get("pattern", "")
    path = params.get("path", ".")

    if not pattern:
        return "[ERROR] grep: pattern is required"

    try:
        result = subprocess.run(
            ["grep", "-r", pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout or "(no matches)"
    except Exception as exc:
        return f"[ERROR] grep failed: {exc}"


async def _glob_tool(params: dict[str, Any]) -> str:
    """Find files matching a pattern."""
    import glob as glob_module

    pattern = params.get("pattern", "")
    path = params.get("path", ".")

    if not pattern:
        return "[ERROR] glob: pattern is required"

    try:
        import os

        if path != ".":
            full_pattern = os.path.join(path, pattern)
        else:
            full_pattern = pattern

        matches = glob_module.glob(full_pattern, recursive=True)
        return "\n".join(matches) if matches else "(no matches)"
    except Exception as exc:
        return f"[ERROR] glob failed: {exc}"
