"""CLI entry point for team management in Claude Agent SDK.

Usage::
    python -m claude_agent_sdk.cli_teams list-templates
    python -m claude_agent_sdk.cli_teams list-personas
    python -m claude_agent_sdk.cli_teams run --template full_stack_team --task "Build a REST API"

Created: 2026-05-27 CST
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def _run_team(args: argparse.Namespace) -> None:
    """Create a team from template, start all agents, wait for completion."""
    from .team_templates import create_team_from_template

    if not args.template:
        print("ERROR: --template is required for 'run' command", file=sys.stderr)
        sys.exit(1)
    if not args.task:
        print("ERROR: --task is required for 'run' command", file=sys.stderr)
        sys.exit(1)

    # Resolve output directory
    output_dir = os.path.expanduser(
        args.output_dir or f"~/.claude/agent-logs/teams/{args.template}-{int(time.time())}/"
    )

    # Build overrides from CLI args
    overrides: dict[str, Any] = {}
    if args.api_base:
        overrides["base_url"] = args.api_base
    if args.api_key:
        overrides["api_key"] = args.api_key
    if args.team_id:
        overrides["team_id"] = args.team_id

    print(f"Creating team from template: {args.template}")
    print(f"Task: {args.task}")
    print(f"Output dir: {output_dir}")

    try:
        tm = create_team_from_template(
            template_name=args.template,
            task=args.task,
            output_dir=output_dir,
            **overrides,
        )
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    async def _run() -> dict[str, Any]:
        print(f"Starting {len(tm._agents)} agents...")
        await tm.start_all()

        status = tm.status()
        for name, info in status["agents"].items():
            print(f"  [{name}] model={info['model']} role={info['role']}")

        print(f"\nWaiting for completion (timeout={args.timeout}s)...")
        results = await tm.wait_for_completion(timeout=args.timeout)

        print("\n=== Results ===")
        for name, result in results.items():
            if hasattr(result, "success"):
                status_str = "OK" if result.success else "FAIL"
                print(
                    f"  [{name}] {status_str} — "
                    f"{result.turns} turns, {result.total_tool_calls} tool calls, "
                    f"{result.elapsed_seconds:.1f}s"
                )
            else:
                print(f"  [{name}] status={result}")

        await tm.stop_all(reason="CLI run complete")
        return results

    asyncio.run(_run())
    print(f"\nAgent outputs written to: {output_dir}")


def _list_templates(args: argparse.Namespace) -> None:
    """Print all available team templates."""
    from .team_templates import list_templates

    templates = list_templates()
    if args.json:
        print(json.dumps(templates, indent=2))
        return

    for tmpl in templates:
        print(f"\n{tmpl['name']} ({tmpl['agent_count']} agents, {tmpl['communication_pattern']})")
        print(f"  {tmpl['description']}")
        print(f"  Flow: {' -> '.join(tmpl['task_flow'])}")
        for agent in tmpl["agents"]:
            print(f"    - {agent['name']}: {agent['role']} [{agent['model']}]")


def _list_personas(args: argparse.Namespace) -> None:
    """Print all available agent personas."""
    from .agent_personas import list_personas

    personas = list_personas()
    if args.json:
        print(json.dumps(personas, indent=2))
        return

    for persona in personas:
        print(f"\n{persona['name']}")
        print(f"  Expertise: {persona['expertise']}")
        print(f"  Style: {persona['communication_style']}")
        if persona["review_focus"]:
            print(f"  Focus areas ({len(persona['review_focus'])}):")
            for focus in persona["review_focus"][:3]:
                print(f"    - {focus}")
            if len(persona["review_focus"]) > 3:
                print(f"    ... and {len(persona['review_focus']) - 3} more")


def main() -> None:
    """CLI entry point for team management."""
    parser = argparse.ArgumentParser(
        prog="claude_agent_sdk.cli_teams",
        description="Multi-agent team management for Claude Agent SDK",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a team from a template")
    run_parser.add_argument("--template", required=True, help="Template name")
    run_parser.add_argument("--task", required=True, help="Seed task for the team")
    run_parser.add_argument("--api-base", default=None, help="LiteLLM base URL")
    run_parser.add_argument("--api-key", default=None, help="API key")
    run_parser.add_argument("--team-id", default=None, help="Override team ID")
    run_parser.add_argument("--output-dir", default=None, help="Output directory")
    run_parser.add_argument(
        "--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)"
    )
    run_parser.set_defaults(func=_run_team)

    # list-templates command
    lt_parser = subparsers.add_parser("list-templates", help="List available templates")
    lt_parser.add_argument("--json", action="store_true", help="Output as JSON")
    lt_parser.set_defaults(func=_list_templates)

    # list-personas command
    lp_parser = subparsers.add_parser("list-personas", help="List available personas")
    lp_parser.add_argument("--json", action="store_true", help="Output as JSON")
    lp_parser.set_defaults(func=_list_personas)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.func(args)


if __name__ == "__main__":
    main()
