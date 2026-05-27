"""Pre-built team configurations for common multi-agent workflows.

Each template defines a ready-to-use team configuration with agent roles,
models, personas, and communication patterns. Use create_team_from_template()
to instantiate a TeamManager from a template name.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Agent definition within a template ─────────────────────────────────────


@dataclass
class TemplateAgent:
    """A single agent slot in a team template."""

    name: str
    model: str
    role: str
    persona: str  # Short persona label — maps to agent_personas.Persona if present
    initial_task: str = ""  # Optional seed task; empty = wait for message


# ── Team template dataclass ──────────────────────────────────────────────────


@dataclass
class TeamTemplate:
    """A complete team configuration ready for instantiation.

    Fields:
        name: Machine-readable template identifier.
        description: Human-readable description of what this team does.
        agents: Ordered list of TemplateAgent definitions.
        task_flow: Ordered agent names describing the default handoff sequence.
            E.g. ["planner", "backend-dev", "tester"] means planner goes first,
            passes output to backend-dev, who passes to tester.
        communication_pattern: One of "sequential" | "parallel" | "review_loop".
            sequential — each agent waits for the previous to finish.
            parallel   — all agents start simultaneously, coordinate via bus.
            review_loop — alternating propose/review cycles.
    """

    name: str
    description: str
    agents: list[TemplateAgent]
    task_flow: list[str]
    communication_pattern: str  # "sequential" | "parallel" | "review_loop"
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Built-in templates ───────────────────────────────────────────────────────

FULL_STACK_TEAM = TeamTemplate(
    name="full_stack_team",
    description=(
        "End-to-end feature development team. Planner architects the solution, "
        "backend-dev implements the API, frontend-dev builds the UI, tester "
        "writes and runs tests. Sequential handoff: plan → build → test."
    ),
    agents=[
        TemplateAgent(
            name="planner",
            model="qwen/qwen3-max",
            role="Solution architect and technical planner",
            persona="ARCHITECT",
            initial_task="",  # Receives task from orchestrator
        ),
        TemplateAgent(
            name="backend-dev",
            model="nvidia/devstral-2-123b",
            role="Backend engineer — API, database, business logic",
            persona="ARCHITECT",
            initial_task="",
        ),
        TemplateAgent(
            name="frontend-dev",
            model="nvidia/qwen3-coder-480b",
            role="Frontend engineer — UI components, state, routing",
            persona="UX_SPECIALIST",
            initial_task="",
        ),
        TemplateAgent(
            name="tester",
            model="qwen/qwen-plus",
            role="QA and test engineer — unit, integration, e2e",
            persona="TEST_ENGINEER",
            initial_task="",
        ),
    ],
    task_flow=["planner", "backend-dev", "frontend-dev", "tester"],
    communication_pattern="sequential",
    metadata={
        "typical_duration_minutes": 30,
        "best_for": "feature builds, greenfield modules, API + UI pairs",
    },
)


CODE_REVIEW_PAIR = TeamTemplate(
    name="code_review_pair",
    description=(
        "Author writes the code, reviewer critiques it. Ends with either LGTM "
        "or a revision cycle. Up to 3 rounds before escalating to orchestrator."
    ),
    agents=[
        TemplateAgent(
            name="author",
            model="qwen/qwen3-max",
            role="Code author — implements the feature or fix",
            persona="ARCHITECT",
            initial_task="",
        ),
        TemplateAgent(
            name="reviewer",
            model="nvidia/devstral-2-123b",
            role="Code reviewer — critiques quality, correctness, security",
            persona="DEVIL_ADVOCATE",
            initial_task="",
        ),
    ],
    task_flow=["author", "reviewer"],
    communication_pattern="review_loop",
    metadata={
        "max_review_rounds": 3,
        "best_for": "PRs, code quality gates, security review, refactors",
    },
)


DEVIL_ADVOCATE_DUO = TeamTemplate(
    name="devil_advocate_duo",
    description=(
        "Proposer presents an approach, critic challenges every assumption. "
        "Structured debate with 3 rounds. Best for architecture decisions, "
        "design trade-offs, risk analysis."
    ),
    agents=[
        TemplateAgent(
            name="proposer",
            model="qwen/qwen3-max",
            role="Proposer — presents and defends a technical approach",
            persona="ARCHITECT",
            initial_task="",
        ),
        TemplateAgent(
            name="critic",
            model="nvidia/devstral-2-123b",
            role="Devil's advocate — challenges every assumption, finds edge cases",
            persona="DEVIL_ADVOCATE",
            initial_task="",
        ),
    ],
    task_flow=["proposer", "critic"],
    communication_pattern="review_loop",
    metadata={
        "debate_rounds": 3,
        "best_for": "architecture decisions, design trade-offs, risk analysis",
    },
)


SECURITY_AUDIT_TEAM = TeamTemplate(
    name="security_audit_team",
    description=(
        "Developer builds the code, security-auditor reviews for vulnerabilities "
        "(OWASP Top 10, auth issues, injection), pen-tester probes for exploits. "
        "All three run in parallel after developer completes."
    ),
    agents=[
        TemplateAgent(
            name="developer",
            model="qwen/qwen3-max",
            role="Developer — writes production-ready code",
            persona="ARCHITECT",
            initial_task="",
        ),
        TemplateAgent(
            name="security-auditor",
            model="nvidia/devstral-2-123b",
            role="Security auditor — OWASP Top 10, auth, injection, data exposure",
            persona="SECURITY_EXPERT",
            initial_task="",
        ),
        TemplateAgent(
            name="pen-tester",
            model="qwen/qwen-plus",
            role="Penetration tester — probes for exploitable vulnerabilities",
            persona="SECURITY_EXPERT",
            initial_task="",
        ),
    ],
    task_flow=["developer", "security-auditor", "pen-tester"],
    communication_pattern="parallel",
    metadata={
        "best_for": "pre-release security review, auth systems, public APIs",
        "note": "security-auditor and pen-tester run in parallel after developer",
    },
)


RAPID_PROTOTYPE = TeamTemplate(
    name="rapid_prototype",
    description=(
        "Designer sketches the architecture and component structure, builder "
        "implements it immediately. Optimized for speed — no separate testing pass."
    ),
    agents=[
        TemplateAgent(
            name="designer",
            model="qwen/qwen3-max",
            role="Product/system designer — architecture, component sketch, data flow",
            persona="ARCHITECT",
            initial_task="",
        ),
        TemplateAgent(
            name="builder",
            model="nvidia/qwen3-coder-480b",
            role="Rapid implementer — builds the scaffold fast, ships working code",
            persona="ARCHITECT",
            initial_task="",
        ),
    ],
    task_flow=["designer", "builder"],
    communication_pattern="sequential",
    metadata={
        "best_for": "MVPs, proof-of-concepts, hackathons, quick demos",
        "typical_duration_minutes": 10,
    },
)


# ── Registry ─────────────────────────────────────────────────────────────────

_TEMPLATE_REGISTRY: dict[str, TeamTemplate] = {
    t.name: t
    for t in [
        FULL_STACK_TEAM,
        CODE_REVIEW_PAIR,
        DEVIL_ADVOCATE_DUO,
        SECURITY_AUDIT_TEAM,
        RAPID_PROTOTYPE,
    ]
}


def list_templates() -> list[dict[str, Any]]:
    """Return all available templates with descriptions.

    Returns:
        List of dicts with keys: name, description, agent_count,
        communication_pattern, task_flow, metadata.
    """
    result = []
    for tmpl in _TEMPLATE_REGISTRY.values():
        result.append(
            {
                "name": tmpl.name,
                "description": tmpl.description,
                "agent_count": len(tmpl.agents),
                "agents": [
                    {
                        "name": a.name,
                        "model": a.model,
                        "role": a.role,
                        "persona": a.persona,
                    }
                    for a in tmpl.agents
                ],
                "task_flow": tmpl.task_flow,
                "communication_pattern": tmpl.communication_pattern,
                "metadata": tmpl.metadata,
            }
        )
    return result


def create_team_from_template(
    template_name: str,
    task: str,
    output_dir: str,
    base_url: str = "http://127.0.0.1:8016",
    api_key: str = "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
    team_id: str | None = None,
    persona_overrides: dict[str, str] | None = None,
) -> TeamManager:  # type: ignore[return]  # forward ref resolved at call time
    """Create a fully configured TeamManager from a named template.

    Imports agent_personas to enrich each agent's system prompt with
    domain-specific instructions based on the template's persona assignments.

    Args:
        template_name: One of the template name strings (e.g. "full_stack_team").
            Call list_templates() to see available options.
        task: The seed task delivered to the first agent in the task_flow.
            Remaining agents wait for messages from their predecessors.
        output_dir: Directory where agent output files will be written.
        base_url: LiteLLM endpoint. Defaults to router-proxy.
        api_key: API key for the LiteLLM endpoint.
        team_id: Override the auto-generated team ID. Defaults to template name.
        persona_overrides: Dict of {agent_name: persona_name} to override
            a template's default persona assignment for specific agents.

    Returns:
        A TeamManager instance with all agents registered (not yet started).
        Call await tm.start_all() to launch.

    Raises:
        KeyError: If template_name is not in the registry.

    Example::
        tm = create_team_from_template(
            "full_stack_team",
            task="Build a REST API for user management with JWT auth",
            output_dir="~/.claude/agent-logs/user-api/",
        )
        await tm.start_all()
        await tm.wait_for_completion(timeout=3600)
    """
    import asyncio

    from .agent_personas import apply_persona_to_system_prompt, get_persona
    from .team_manager import TeamManager

    if template_name not in _TEMPLATE_REGISTRY:
        available = list(_TEMPLATE_REGISTRY.keys())
        raise KeyError(f"Template '{template_name}' not found. Available: {available}")

    tmpl = _TEMPLATE_REGISTRY[template_name]
    effective_team_id = team_id or f"{tmpl.name}-{int(__import__('time').time())}"

    tm = TeamManager(
        team_id=effective_team_id,
        output_dir=output_dir,
        base_url=base_url,
        api_key=api_key,
    )

    # Determine which agent gets the initial task (first in task_flow)
    first_agent = (
        tmpl.task_flow[0]
        if tmpl.task_flow
        else (tmpl.agents[0].name if tmpl.agents else None)
    )

    async def _register_agents() -> None:
        for agent_def in tmpl.agents:
            # Resolve persona name (allow per-agent override)
            persona_name = (persona_overrides or {}).get(
                agent_def.name, agent_def.persona
            )
            persona = get_persona(persona_name)

            # Build enriched system prompt
            base_prompt = (
                f"You are {agent_def.name}, a {agent_def.role}. "
                f"You are part of a {tmpl.communication_pattern} team "
                f"named '{effective_team_id}'."
            )
            enriched_prompt = apply_persona_to_system_prompt(base_prompt, persona)

            # Only the first agent in the flow gets the seed task
            agent_task = task if agent_def.name == first_agent else ""

            await tm.add_agent(
                name=agent_def.name,
                model=agent_def.model,
                role=agent_def.role
                + (
                    f" | Focus areas: {', '.join(persona.review_focus)}"
                    if persona and persona.review_focus
                    else ""
                ),
                task=agent_task,
            )

        logger.info(
            "Team '%s' created from template '%s' with %d agents",
            effective_team_id,
            template_name,
            len(tmpl.agents),
        )

    asyncio.get_event_loop().run_until_complete(_register_agents())
    return tm


__all__ = [
    "TeamTemplate",
    "TemplateAgent",
    "FULL_STACK_TEAM",
    "CODE_REVIEW_PAIR",
    "DEVIL_ADVOCATE_DUO",
    "SECURITY_AUDIT_TEAM",
    "RAPID_PROTOTYPE",
    "list_templates",
    "create_team_from_template",
]
