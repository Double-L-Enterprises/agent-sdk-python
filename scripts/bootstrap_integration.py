#!/usr/bin/env python3
"""Bootstrap Integration Tasks using MultiAgentOrchestrator.

Runs 4 agents to build integration features:
- Agent 1: bridge-update — Update bridge_sdk.py to support AutonomousRunner backend
- Agent 2: agent-hook — Build PreToolUse hook for Agent SDK redirection (depends on Agent 1)
- Agent 3: watchdog-redesign — Build new service health watchdog system
- Agent 4: safe-write — Add A/B safe-write to Agent SDK's file_ops.py

Agent 1 and 3 run in parallel. Agent 2 waits for Agent 1 to complete.
Agent 4 runs in parallel with 1 and 3.

Created: 2026-05-27 19:30 CST
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

BRIDGE_UPDATE_TASK = STALE_FILE_INSTRUCTIONS + """
TASK: bridge-update — Update bridge_sdk.py to support AutonomousRunner backend

Read the current bridge_sdk.py at:
/mnt/c/Users/larry/ClaudeNotes/shared/projects/valor-voice/claude-bridge/bridge_sdk.py

Currently it uses SubprocessCLITransport (spawns claude CLI). Modify it to ALSO support
AutonomousRunner as an alternative backend.

Implementation:
1. Add a route or config option that lets callers choose "cli" (existing) or "litellm" (new)
2. For the "litellm" path:
   - Import AutonomousRunner from the Claude Agent SDK
   - Use AutonomousRunner instead of the SDK client's query()
   - Read config from env vars: LITELLM_BASE_URL, LITELLM_DEFAULT_MODEL, LITELLM_API_KEY
3. Keep all existing endpoints and JSON contracts identical
4. Test that both backends work by making a test request to both paths

File location: /mnt/c/Users/larry/ClaudeNotes/shared/projects/valor-voice/claude-bridge/bridge_sdk.py

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/valor-voice/claude-bridge/
Max turns: 25, model: qwen/qwen3-max
"""

AGENT_HOOK_TASK = STALE_FILE_INSTRUCTIONS + """
TASK: agent-hook — Build PreToolUse hook for Agent SDK redirection

Create a Claude Code PreToolUse hook script at:
/mnt/c/Users/larry/ClaudeNotes/shared/claude-hooks/agent-sdk-redirect.sh

This bash hook intercepts the Agent tool when the model parameter is a free/cheap model
(qwen/*, nvidia/*) and redirects the work through bridge_sdk.py on :8020 instead of
spawning claude CLI.

Implementation:
1. Check if the tool being invoked is "Agent"
2. Parse the model parameter from the JSON input
3. If model starts with "qwen/" or "nvidia/":
   - curl the bridge_sdk.py /task endpoint with the prompt and model
   - Return the result from bridge_sdk.py
4. If model is claude-* or sonnet/opus/haiku, let it pass through normally to Claude Code
5. Include a BYPASS env var (AGENT_SDK_REDIRECT_BYPASS=1) to disable redirection

The hook should:
- Check for AGENT_SDK_REDIRECT_BYPASS=1; if set, exit 0 (pass through)
- Parse JSON from stdin to extract tool name and model
- For Agent tool with cheap models, curl http://127.0.0.1:8020/task
- Return the result or pass through to normal handler

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/claude-hooks/
Max turns: 20, model: qwen/qwen3-max

NOTE: This task depends on Agent 1 (bridge-update) completing successfully.
      Wait for Agent 1 before implementing this hook.
"""

WATCHDOG_REDESIGN_TASK = STALE_FILE_INSTRUCTIONS + """
TASK: watchdog-redesign — Build new service health watchdog system

Design and build a new watchdog system to replace the broken ai-watchdog.sh.

Current problems:
- smart-watchdog.service points to nonexistent paperclip-event-bus/watchdog.py
- ai-watchdog.sh runs via cron every 2min but misses outages and kills healthy processes

Create a new watchdog at:
/mnt/c/Users/larry/ClaudeNotes/shared/scripts/service-watchdog.sh

Implementation:
1. Check health of key services:
   - LiteLLM (:3002, requires auth header)
   - router-proxy (:8016)
   - vLLM (:18010)
   - agent-mcp (:8030)
   - bridge (:8020)

2. For each service:
   - curl with 5s timeout
   - If 401/403 = UP (auth required but alive)
   - If 000/timeout = DOWN
   - If 200 = UP

3. State tracking:
   - Only restart a service if it's been DOWN for 2 consecutive checks
   - This prevents killing healthy slow-responding services

4. Logging:
   - Log to ~/logs/watchdog.log with timestamps
   - Include service name, status (UP/DOWN), and action taken

5. Create a systemd timer unit file:
   - /etc/systemd/user/service-watchdog.service
   - /etc/systemd/user/service-watchdog.timer (runs every 2 minutes)

6. IMPORTANT: Do NOT restart services that return auth errors (401/403)
   - Those are alive. Only restart services that truly timeout or give 000.

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/scripts/
Max turns: 25, model: qwen/qwen3-max
"""

SAFE_WRITE_TASK = STALE_FILE_INSTRUCTIONS + """
TASK: safe-write — Add A/B safe-write to Agent SDK's file_ops.py

Add atomic A/B safe-write capability to the built-in Write tool in the Agent SDK.

File location:
/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen/src/claude_agent_sdk/tools/file_ops.py

Implementation:
1. Read the current file first
2. Modify the _write() function so that when overwriting an EXISTING file (path.exists() is True):
   - Copy original to {path}.bak
   - Write new content to {path}.new
   - Verify .new file:
     * Check it exists
     * Check file size > 0
     * For .py files: run ast.parse() to verify syntax
   - If verification passes: rename .new to the real path (atomic on most filesystems)
   - If verification fails: delete .new, keep .bak, return error explaining what failed
   - Clean up .bak after successful swap (or keep last N backups)

3. For NEW files (path.exists() is False):
   - Write directly as before — no backup needed

4. Use Edit tool to modify file_ops.py, NOT Write
5. Verify line count after editing

Include appropriate imports (ast, shutil, pathlib) and error handling.

Working directory: /mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen
Max turns: 20, model: qwen/qwen3-max
"""


async def main():
    """Launch 4 agents to build integration tasks."""
    orchestrator = MultiAgentOrchestrator(
        base_url="http://127.0.0.1:8016",
        api_key="sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        checkpoint_dir="/tmp/bootstrap-integration",
    )

    # Agent 1: bridge-update (independent, runs in parallel)
    orchestrator.add_agent(
        "bridge-update",
        task=BRIDGE_UPDATE_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/valor-voice/claude-bridge",
        max_turns=25,
    )

    # Agent 2: agent-hook (depends on Agent 1)
    orchestrator.add_agent(
        "agent-hook",
        task=AGENT_HOOK_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/claude-hooks",
        max_turns=20,
        depends_on=["bridge-update"],  # Wait for Agent 1 to complete
    )

    # Agent 3: watchdog-redesign (independent, runs in parallel with 1 and 4)
    orchestrator.add_agent(
        "watchdog-redesign",
        task=WATCHDOG_REDESIGN_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/scripts",
        max_turns=25,
    )

    # Agent 4: safe-write (independent, runs in parallel with 1 and 3)
    orchestrator.add_agent(
        "safe-write",
        task=SAFE_WRITE_TASK,
        model="qwen/qwen3-max",
        cwd="/mnt/c/Users/larry/ClaudeNotes/shared/projects/agent-sdk-python-frozen",
        max_turns=20,
    )

    logger.info("=== Launching 4 agents (2 parallel batches, 1 dependent) ===")
    logger.info("Batch 1: bridge-update, watchdog-redesign, safe-write (parallel)")
    logger.info("Batch 2: agent-hook (waits for bridge-update)")

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
