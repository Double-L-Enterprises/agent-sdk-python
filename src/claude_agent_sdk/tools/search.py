"""Grep search tool — uses ripgrep if available, falls back to grep."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15  # seconds
_MAX_OUTPUT_BYTES = 100_000  # 100 KB

SEARCH_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "Grep",
        "description": (
            "Search for a regex pattern in files. Uses ripgrep (rg) if available, "
            "otherwise falls back to grep. Returns matching lines with file:line context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.ts').",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive matching (default: false).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context to show around matches.",
                },
            },
            "required": ["pattern"],
        },
    },
}


def _find_search_tool() -> tuple[str, str]:
    """Return (tool_name, tool_path) — prefer rg over grep."""
    rg = shutil.which("rg")
    if rg:
        return "rg", rg
    grep = shutil.which("grep")
    if grep:
        return "grep", grep
    return "grep", "grep"  # fallback; will fail if not in PATH


async def execute_search(params: dict[str, Any]) -> str:
    """Execute a grep/rg search and return matching lines.

    Args:
        params: Tool parameters dict.

    Returns:
        Matching lines as a string, or an error message.
    """
    pattern = params.get("pattern")
    if not pattern:
        return "[ERROR] Grep requires 'pattern' parameter"

    search_path = params.get("path", ".")
    glob = params.get("glob")
    case_insensitive = params.get("case_insensitive", False)
    context_lines = params.get("context_lines", 0)

    tool_name, tool_path = _find_search_tool()

    if tool_name == "rg":
        cmd = _build_rg_command(
            tool_path, pattern, search_path, glob, case_insensitive, context_lines
        )
    else:
        cmd = _build_grep_command(
            tool_path, pattern, search_path, glob, case_insensitive, context_lines
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(_DEFAULT_TIMEOUT)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"[ERROR] Search timed out after {_DEFAULT_TIMEOUT}s"

        # Exit code 1 from grep/rg means "no matches" — not an error
        rc = proc.returncode
        if rc not in (0, 1):
            err = stderr_bytes.decode("utf-8", errors="replace")[:500]
            return f"[ERROR] Search failed (exit {rc}): {err}"

        output = stdout_bytes.decode("utf-8", errors="replace")
        if not output.strip():
            return f"No matches found for pattern: {pattern!r}"

        if len(output.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            output = output.encode("utf-8")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
            output += "\n[...output truncated]"

        return output

    except FileNotFoundError:
        return f"[ERROR] Search tool not found: {tool_path}"
    except OSError as exc:
        return f"[ERROR] Search failed: {exc}"


def _build_rg_command(
    rg: str,
    pattern: str,
    path: str,
    glob: str | None,
    case_insensitive: bool,
    context_lines: int,
) -> list[str]:
    cmd = [rg, "--line-number", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if glob:
        cmd.extend(["--glob", glob])
    if context_lines:
        cmd.extend(["-C", str(context_lines)])
    cmd.extend([pattern, path])
    return cmd


def _build_grep_command(
    grep: str,
    pattern: str,
    path: str,
    glob: str | None,
    case_insensitive: bool,
    context_lines: int,
) -> list[str]:
    cmd = [grep, "-rn", "--include"]
    if glob:
        cmd.append(glob)
    else:
        cmd.append("*")
    if case_insensitive:
        cmd.append("-i")
    if context_lines:
        cmd.extend([f"-C{context_lines}"])
    cmd.extend(["-E", pattern, path])
    return cmd
