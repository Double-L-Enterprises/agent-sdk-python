"""Example: AutonomousRunner — autonomous agent that keeps working until done.

This example shows how to create an agent that runs autonomously without
stopping mid-task. The runner implements completion detection and forced
continuation to ensure the task is actually finished.

Run with:
    python examples/autonomous_runner.py
"""

import asyncio

from claude_agent_sdk.runner import AutonomousRunner
from claude_agent_sdk.tools import default_tools


async def main():
    """Run an autonomous task."""

    # Example 1: Simple task
    print("=" * 60)
    print("Example 1: Build a Python todo CLI app")
    print("=" * 60)

    runner = AutonomousRunner(
        task="Create a Python command-line todo app. It should allow users to add, list, and mark tasks as complete. Store tasks in a JSON file.",
        tools=default_tools(),
        model="qwen/qwen3-max",
        base_url="http://127.0.0.1:8016",
        api_key="sk-litellm",
        max_turns=50,
        max_continuation_retries=3,
    )

    result = await runner.run()

    print(f"\n✓ Completed in {result.turn_count} turns")
    print(f"  Status: {result.stopped_reason}")
    print(f"  Success: {result.success}")
    print(f"\nFinal response:\n{result.final_text}")
    print(f"\n[Full conversation: {len(result.messages)} messages]")


async def advanced_example():
    """More complex task with explicit control."""

    print("\n" + "=" * 60)
    print("Example 2: More complex task")
    print("=" * 60)

    task = """
    Create a Python package called 'word_counter' that:
    1. Has a module that counts word frequencies in a text file
    2. Includes a CLI that reads a file and prints the top 10 most common words
    3. Includes unit tests (pytest)
    4. Has a proper README

    Work in /tmp/word_counter/
    """

    runner = AutonomousRunner(
        task=task.strip(),
        model="qwen/qwen3-max",
        max_turns=100,  # More complex task needs more turns
    )

    result = await runner.run()

    print(f"\n✓ Completed in {result.turn_count} turns ({result.stopped_reason})")
    print(f"  Success: {result.success}")

    if result.success:
        print("\n✓ Task completed successfully!")
    else:
        print("\n⚠ Task incomplete. Final message:")
        print(result.final_text[:500])


if __name__ == "__main__":
    # Run the first example
    asyncio.run(main())

    # Uncomment to run the advanced example
    # asyncio.run(advanced_example())
