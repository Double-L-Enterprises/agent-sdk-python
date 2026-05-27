#!/usr/bin/env python3
"""Bootstrap Phases 5-8 using MultiAgentOrchestrator.

Runs 4 agents in parallel, each building one phase:
- Phase 5: Streaming SSE output
- Phase 6: MCP Tool Bridge
- Phase 7: Session Stores
- Phase 8: Permission Model

Each agent is taught to verify file state before/after writes.

Created: 2026-05-27 14:30 CST
"""
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("bootstrap")

# Add src to path
sys.path.insert(0, "/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen/src")

from claude_agent_sdk.multi_agent import MultiAgentOrchestrator


# Stale file detection instructions — every agent gets this
STALE_FILE_INSTRUCTIONS = """
CRITICAL: VERIFY FILE STATE BEFORE AND AFTER EVERY WRITE.

Before modifying any file:
1. Use Bash to run: wc -l <file> — record the line count
2. Use Read to see the actual content — never assume what's in a file
3. Use Bash: stat <file> — check modification timestamp

After writing any file:
1. Use Bash: wc -l <file> — verify the line count matches what you wrote
2. Use Read to verify the first 10 and last 10 lines are correct
3. If line count doesn't match what you wrote, READ THE FILE AGAIN — Syncthing may have overwritten it

NEVER report a file as written without verifying. Files in this project sync via Syncthing and can be overwritten by other machines.

When editing an EXISTING file (not creating new):
1. Read the full file first
2. Use the Edit tool (find-and-replace), NOT Write (full overwrite)
3. After editing, verify the file still has all original content plus your changes
4. Check line count — it should be >= original count (unless you intentionally removed lines)
"""

PHASE_5_TASK = STALE_FILE_INSTRUCTIONS + """
BUILD Phase 5: Streaming SSE Output

Create: src/claude_agent_sdk/streaming.py

This module adds streaming support so users see model output token-by-token during a run.

Implementation:
1. StreamCallback protocol: async callable that receives text chunks
2. StreamingTransportMixin that adds streaming to LiteLLMHTTPTransport:
   - async do_turn_streaming(on_token: StreamCallback) method
   - Sets stream=True in the API call
   - Parses SSE chunks from LiteLLM's streaming response
   - Yields text deltas to the callback
   - Accumulates tool_calls from streamed chunks
   - Returns the same dict format as do_turn() when complete
3. Export from __init__.py

The SSE format from LiteLLM (OpenAI-compatible):
- Lines starting with "data: " contain JSON chunks
- Each chunk has choices[0].delta with partial content or tool_calls
- Final chunk has choices[0].finish_reason = "stop" or "tool_calls"
- Stream ends with "data: [DONE]"

Use httpx streaming: `async with client.stream("POST", url, json=payload) as response:`
Parse each line, accumulate deltas, call on_token for text.

After creating streaming.py:
- Add import to __init__.py (use Edit tool, not Write!)
- Verify: python3 -c "from claude_agent_sdk.streaming import StreamingTransportMixin; print('OK')"

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen
"""

PHASE_6_TASK = STALE_FILE_INSTRUCTIONS + """
BUILD Phase 6: MCP Tool Bridge

Create: src/claude_agent_sdk/tools/mcp_bridge.py

This module lets the runner use MCP (Model Context Protocol) servers as tool providers.

Implementation:
1. MCPBridge class:
   - __init__(servers: dict[str, MCPServerConfig]) where MCPServerConfig has type, command/url, args, env
   - async connect() — start MCP server subprocesses or connect to SSE endpoints
   - async discover_tools() -> list[dict] — get tool definitions from all servers, convert to OpenAI function format
   - async call_tool(server_name: str, tool_name: str, params: dict) -> str — dispatch a tool call to the right server
   - async close() — shut down connections

2. MCPServerConfig dataclass:
   - type: "stdio" | "sse"
   - command: str (for stdio — e.g., "npx", "python3")
   - args: list[str] (for stdio — e.g., ["-m", "mcp_server"])
   - url: str (for sse — e.g., "http://localhost:8030/sse")
   - env: dict[str, str] (extra env vars)

3. For stdio servers: use asyncio.create_subprocess_exec, communicate via JSON-RPC over stdin/stdout
   - Send: {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
   - Receive: {"jsonrpc": "2.0", "result": {"tools": [...]}, "id": 1}
   - For tool calls: {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "...", "arguments": {...}}, "id": 2}

4. For SSE servers: use httpx to POST to the SSE endpoint
   - Similar JSON-RPC but over HTTP

5. Convert MCP tool schemas to OpenAI function calling format:
   - MCP: {"name": "...", "description": "...", "inputSchema": {...}}
   - OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

After creating mcp_bridge.py:
- Add to tools/__init__.py (use Edit tool!)
- Export MCPBridge from __init__.py
- Verify: python3 -c "from claude_agent_sdk.tools.mcp_bridge import MCPBridge; print('OK')"

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen
"""

PHASE_7_TASK = STALE_FILE_INSTRUCTIONS + """
BUILD Phase 7: Session Stores

Create: src/claude_agent_sdk/session_stores.py

This module provides persistent session storage so conversations survive restarts.

Implementation:
1. SessionStore ABC:
   - async save(session_id: str, data: dict) -> None
   - async load(session_id: str) -> dict | None
   - async delete(session_id: str) -> None
   - async list_sessions() -> list[str]

2. FileSessionStore(SessionStore):
   - __init__(directory: str)
   - Saves sessions as JSON files: {directory}/{session_id}.json
   - list_sessions scans the directory

3. RedisSessionStore(SessionStore):
   - __init__(url: str = "redis://localhost:6379", prefix: str = "agent_session:")
   - Uses redis-py async client
   - Saves as JSON strings with key prefix

4. PostgresSessionStore(SessionStore):
   - __init__(dsn: str, table: str = "agent_sessions")
   - Uses asyncpg
   - Auto-creates table if not exists: (session_id TEXT PRIMARY KEY, data JSONB, updated_at TIMESTAMP)

Note: RedisSessionStore and PostgresSessionStore should handle missing dependencies gracefully:
  try:
      import redis.asyncio as redis
  except ImportError:
      raise ImportError("pip install redis for RedisSessionStore")

After creating session_stores.py:
- Export from __init__.py (use Edit!)
- Verify: python3 -c "from claude_agent_sdk.session_stores import FileSessionStore; print('OK')"

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen
"""

PHASE_8_TASK = STALE_FILE_INSTRUCTIONS + """
BUILD Phase 8: Permission Model

Create: src/claude_agent_sdk/permissions.py

This module controls what tools agents can use, with approval callbacks.

Implementation:
1. PermissionPolicy class:
   - __init__()
   - allow(*tool_names: str) -> self — whitelist these tools (chainable)
   - deny(*tool_names: str) -> self — blacklist these tools (chainable)
   - require_approval(*tool_names: str, callback: ApprovalCallback) -> self
     - ApprovalCallback = Callable[[str, dict], Awaitable[bool]]
     - Called with (tool_name, params), returns True to allow, False to block
   - sandbox(allowed_paths: list[str] | None = None, allow_network: bool = True) -> self
     - Restricts Bash and file tools to specific directories
     - If allow_network=False, blocks curl/wget/pip install in Bash

2. async check(tool_name: str, params: dict) -> tuple[bool, str]:
   - Returns (allowed: bool, reason: str)
   - Check order: deny list -> allow list -> approval callbacks -> sandbox
   - If denied: (False, "Tool X is denied by policy")
   - If needs approval: calls callback, returns result
   - If sandboxed: checks file_path/cwd against allowed_paths

3. Integration point (for hooks.py):
   - PermissionPolicy can be used as a pre_tool_use hook:
     policy = PermissionPolicy().deny("Bash").allow("Read", "Write", "Glob")
     hooks = HookRegistry()
     hooks.register("pre_tool_use", policy.as_hook())
   - as_hook() returns an async function compatible with HookRegistry

After creating permissions.py:
- Export from __init__.py (use Edit!)
- Verify: python3 -c "from claude_agent_sdk.permissions import PermissionPolicy; p = PermissionPolicy().allow('Read').deny('Bash'); print('OK')"

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen
"""


async def main():
    """Launch 4 agents to build Phases 5-8 in parallel."""
    orchestrator = MultiAgentOrchestrator(
        base_url="http://127.0.0.1:8016",
        api_key="sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        checkpoint_dir="/tmp/bootstrap-phases-5-8",
    )

    orchestrator.add_agent(
        "phase5-streaming",
        task=PHASE_5_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen",
        max_turns=25,
    )
    orchestrator.add_agent(
        "phase6-mcp",
        task=PHASE_6_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen",
        max_turns=25,
    )
    orchestrator.add_agent(
        "phase7-sessions",
        task=PHASE_7_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen",
        max_turns=25,
    )
    orchestrator.add_agent(
        "phase8-permissions",
        task=PHASE_8_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen",
        max_turns=25,
    )

    logger.info("=== Launching 4 agents in parallel ===")
    result = await orchestrator.run_parallel()

    logger.info("=== ALL AGENTS COMPLETE ===")
    logger.info("Total turns: %d", result.total_turns)
    logger.info("Total tool calls: %d", result.total_tool_calls)
    logger.info("Elapsed: %.1fs", result.total_elapsed_seconds)
    logger.info("All succeeded: %s", result.all_succeeded)

    if result.failed_agents:
        logger.warning("Failed agents: %s", result.failed_agents)

    print("\n" + result.summary())

    for name, r in result.results.items():
        print(f"\n--- {name} ---")
        print(f"  Success: {r.success}")
        print(f"  Turns: {r.turns}")
        print(f"  Tools: {r.total_tool_calls}")
        print(f"  Reason: {r.stopped_reason}")
        print(f"  Output: {r.final_text[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
