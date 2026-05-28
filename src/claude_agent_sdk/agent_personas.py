"""Agent Personas - predefined personality profiles.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class Persona:
    name: str
    role: str
    system_prompt_prefix: str
    traits: list[str] = None
    def __post_init__(self):
        if self.traits is None:
            self.traits = []

_REGISTRY: dict[str, Persona] = {}

def register_persona(persona: Persona) -> None:
    _REGISTRY[persona.name] = persona

def get_persona(name: str) -> Persona | None:
    return _REGISTRY.get(name)

def list_personas() -> list[Persona]:
    return list(_REGISTRY.values())

def apply_persona_to_system_prompt(persona: Persona, prompt: str) -> str:
    return persona.system_prompt_prefix + chr(10) + chr(10) + prompt

SECURITY_EXPERT = Persona(name="security_expert", role="Security Expert", system_prompt_prefix="You are a security expert. Focus on vulnerabilities and secure coding.")
PERFORMANCE_ENGINEER = Persona(name="performance_engineer", role="Performance Engineer", system_prompt_prefix="You are a performance engineer. Focus on efficiency and optimization.")
UX_SPECIALIST = Persona(name="ux_specialist", role="UX Specialist", system_prompt_prefix="You are a UX specialist. Focus on user experience.")
ARCHITECT = Persona(name="architect", role="Architect", system_prompt_prefix="You are a software architect. Focus on design patterns and scalability.")
DEVIL_ADVOCATE = Persona(name="devil_advocate", role="Devil's Advocate", system_prompt_prefix="You are a devil's advocate. Challenge assumptions and find flaws.")
TEST_ENGINEER = Persona(name="test_engineer", role="Test Engineer", system_prompt_prefix="You are a test engineer. Focus on testability and coverage.")

for p in [SECURITY_EXPERT, PERFORMANCE_ENGINEER, UX_SPECIALIST, ARCHITECT, DEVIL_ADVOCATE, TEST_ENGINEER]:
    register_persona(p)
