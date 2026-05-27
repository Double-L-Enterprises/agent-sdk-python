#!/usr/bin/env python3
"""Example: AutonomousRunner with LiteLLM backend.

Demonstrates how to use the AutonomousRunner to drive any LiteLLM-backed
model (qwen3-max, GPT-4, etc.) to complete a coding task autonomously,
using the built-in file/bash/search tools.

Run:
    python examples/litellm_autonomous.py

Or with a custom task and working directory:
    python examples/litellm_autonomous.py --task "Add type hints to utils.py" --cwd /path/to/project
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from the repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_agent_sdk.runner import AutonomousRunner
from claude_agent_sdk.tools import default_tools


async def main(task: str, cwd: str | None, model: str, max_turns: int) -> None:
    """Run the autonomous agent on a task."""
    print(f"Task : {task}")
    print(f"Model: {model}")
    print(f"CWD  : {cwd or '(current directory)'}")
    print(f"Max turns: {max_turns}")
    print("-" * 60)

    runner = AutonomousRunner(
        base_url="http://100.102.119.55:3002",
        model=model,
        api_key="sk-litellm",  # LiteLLM default key
        max_turns=max_turns,
        max_continuation_retries=3,
    )

    result = await runner.run(
        task=task,
        cwd=cwd,
        tools=default_tools(),
    )

    print("-" * 60)
    print(f"Stopped by   : {result.stopped_reason}")
    print(f"Turns used   : {result.turns}")
    print(f"Tool calls   : {result.total_tool_calls}")
    print(f"Elapsed      : {result.elapsed_seconds:.1f}s")
    print(f"Success      : {result.success}")
    print("-" * 60)
    print("Final response:")
    print(result.final_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutonomousRunner with LiteLLM")
    parser.add_argument(
        "--task",
        default="Read the README.md in this directory and summarize its contents in 3 bullet points.",
        help="Task for the agent to complete",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for file/bash tools (default: current dir)",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-max",
        help="LiteLLM model name (default: qwen/qwen3-max)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum turns before giving up (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )

    asyncio.run(main(args.task, args.cwd, args.model, args.max_turns))
