"""File operation tools: Read, Write, Edit, Glob."""

from __future__ import annotations

import ast
import fnmatch
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ------------------------------------------------------------------

FILE_OPS_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": (
                "Read the contents of a file. Returns the file's text content. "
                "Use offset and limit to read a slice of a large file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": (
                "Replace an exact string in a file with new text. "
                "The old_string must match exactly (including whitespace/indentation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern under a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern e.g. '**/*.py' or 'src/**/*.ts'.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Root directory to search under. "
                            "Defaults to current working directory."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------

_MAX_READ_BYTES = 1_000_000  # 1 MB guard


async def execute_file_op(params: dict[str, Any]) -> str:
    """Dispatch to the correct file operation based on params keys."""
    # Determine which operation to run by inspecting params
    if "old_string" in params or "new_string" in params:
        return await _edit(params)
    if "content" in params and "file_path" in params:
        return await _write(params)
    if "pattern" in params and "file_path" not in params:
        return await _glob(params)
    return await _read(params)


async def _read(params: dict[str, Any]) -> str:
    file_path = params.get("file_path")
    if not file_path:
        return "[ERROR] Read requires 'file_path' parameter"

    path = Path(file_path)
    if not path.exists():
        return f"[ERROR] File not found: {file_path}"
    if not path.is_file():
        return f"[ERROR] Not a regular file: {file_path}"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[ERROR] Could not read {file_path}: {exc}"

    lines = text.splitlines(keepends=True)
    offset = params.get("offset", 0) or 0
    limit = params.get("limit")

    if offset:
        lines = lines[offset:]
    if limit:
        lines = lines[:limit]

    result = "".join(lines)
    if len(result) > _MAX_READ_BYTES:
        result = result[:_MAX_READ_BYTES] + "\n[...truncated]"

    return result


async def _write(params: dict[str, Any]) -> str:
    file_path = params.get("file_path")
    content = params.get("content", "")
    if not file_path:
        return "[ERROR] Write requires 'file_path' parameter"

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # For new files, write directly without backup
    if not path.exists():
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"[ERROR] Could not write {file_path}: {exc}"
        return f"Successfully wrote {len(content)} characters to {file_path}"

    # For existing files, use A/B safe-write
    backup_path = path.with_suffix(path.suffix + ".bak")
    new_path = path.with_suffix(path.suffix + ".new")

    try:
        # Create backup of original file
        shutil.copy2(path, backup_path)

        # Write new content to .new file
        new_path.write_text(content, encoding="utf-8")

        # Verify the .new file
        if not new_path.exists():
            raise OSError("New file was not created")

        if new_path.stat().st_size == 0:
            raise OSError("New file is empty")

        # For .py files, verify syntax with ast.parse
        if path.suffix == ".py":
            try:
                new_content = new_path.read_text(encoding="utf-8")
                ast.parse(new_content)
            except SyntaxError as se:
                raise OSError(f"Python syntax error: {se}")
            except Exception as e:
                raise OSError(f"Could not verify Python syntax: {e}")

        # Verification passed, atomically replace original with new file
        # On most filesystems, os.replace is atomic
        import os

        os.replace(new_path, path)

        # Clean up backup after successful swap
        if backup_path.exists():
            backup_path.unlink()

        return (
            f"Successfully wrote {len(content)} characters to {file_path} (safe-write)"
        )

    except OSError as exc:
        # Verification failed or operation failed
        # Clean up .new file if it exists
        if new_path.exists():
            try:
                new_path.unlink()
            except OSError:
                pass  # Ignore cleanup errors

        # Keep backup file for recovery
        error_msg = f"[ERROR] Safe-write failed for {file_path}: {exc}"
        return error_msg

    except Exception as exc:
        # Handle any other unexpected errors
        if new_path.exists():
            try:
                new_path.unlink()
            except OSError:
                pass

        error_msg = f"[ERROR] Unexpected error during safe-write for {file_path}: {exc}"
        return error_msg


async def _edit(params: dict[str, Any]) -> str:
    file_path = params.get("file_path")
    old_string = params.get("old_string")
    new_string = params.get("new_string", "")

    if not file_path:
        return "[ERROR] Edit requires 'file_path' parameter"
    if old_string is None:
        return "[ERROR] Edit requires 'old_string' parameter"

    path = Path(file_path)
    if not path.exists():
        return f"[ERROR] File not found: {file_path}"

    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[ERROR] Could not read {file_path}: {exc}"

    count = original.count(old_string)
    if count == 0:
        # Provide a hint for debugging
        preview = original[:300].replace("\n", "\\n")
        return f"[ERROR] old_string not found in {file_path}. File preview: {preview!r}"
    if count > 1:
        return (
            f"[ERROR] old_string found {count} times in {file_path}. "
            "Provide a more specific old_string."
        )

    updated = original.replace(old_string, new_string, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"[ERROR] Could not write {file_path}: {exc}"

    return f"Successfully edited {file_path}"


async def _glob(params: dict[str, Any]) -> str:
    pattern = params.get("pattern", "**/*")
    root_str = params.get("path", ".")
    root = Path(root_str).resolve()

    if not root.exists():
        return f"[ERROR] Directory not found: {root_str}"

    try:
        # Use rglob for ** patterns, glob otherwise
        if "**" in pattern:
            # pathlib.Path.rglob strips the leading **/
            sub_pattern = pattern.lstrip("*").lstrip("/") or "*"
            matches = sorted(root.rglob(sub_pattern))
        else:
            matches = sorted(root.glob(pattern))
    except Exception as exc:
        return f"[ERROR] Glob failed: {exc}"

    # Apply fnmatch filter for patterns that pathlib doesn't handle perfectly
    filtered = [
        str(p)
        for p in matches
        if p.is_file() and fnmatch.fnmatch(str(p), f"*{pattern.lstrip('*')}")
    ]
    if not filtered:
        # Fall back to all matches (pathlib already applied the pattern)
        filtered = [str(p) for p in matches if p.is_file()]

    if not filtered:
        return f"No files matched pattern '{pattern}' under {root_str}"

    return "\n".join(filtered[:500])  # cap at 500 results
