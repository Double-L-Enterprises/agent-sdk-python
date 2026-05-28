"""Team templates for common multi-agent patterns.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TemplateAgent:
    name: str
    role: str
    model: str = "qwen/qwen3-max"
    system_prompt: str | None = None

@dataclass
class TeamTemplate:
    name: str
    description: str
    agents: list[TemplateAgent] = field(default_factory=list)

CODE_REVIEW_PAIR = TeamTemplate(name="code_review_pair", description="Author + reviewer",
    agents=[TemplateAgent(name="author", role="Author"), TemplateAgent(name="reviewer", role="Reviewer")])
DEVIL_ADVOCATE_DUO = TeamTemplate(name="devil_advocate_duo", description="Builder + critic",
    agents=[TemplateAgent(name="builder", role="Builder"), TemplateAgent(name="critic", role="Critic")])
FULL_STACK_TEAM = TeamTemplate(name="full_stack_team", description="Frontend + backend + devops",
    agents=[TemplateAgent(name="frontend", role="Frontend"), TemplateAgent(name="backend", role="Backend"), TemplateAgent(name="devops", role="DevOps")])
SECURITY_AUDIT_TEAM = TeamTemplate(name="security_audit_team", description="Auditor + pentester",
    agents=[TemplateAgent(name="auditor", role="Auditor"), TemplateAgent(name="pentester", role="Pentester")])
RAPID_PROTOTYPE = TeamTemplate(name="rapid_prototype", description="Fast prototyping pair",
    agents=[TemplateAgent(name="prototyper", role="Prototyper"), TemplateAgent(name="validator", role="Validator")])

_ALL_TEMPLATES = [CODE_REVIEW_PAIR, DEVIL_ADVOCATE_DUO, FULL_STACK_TEAM, SECURITY_AUDIT_TEAM, RAPID_PROTOTYPE]

def list_templates() -> list[TeamTemplate]:
    return list(_ALL_TEMPLATES)

def create_team_from_template(template: TeamTemplate, **kwargs: Any) -> dict[str, Any]:
    return {"template": template.name, "agents": [a.name for a in template.agents]}
