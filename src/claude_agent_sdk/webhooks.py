"""Webhook manager for team events.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class WebhookFormat(str, Enum):
    JSON = "json"
    SLACK = "slack"

class WebhookEvent(str, Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    TEAM_COMPLETED = "team_completed"
    BUDGET_WARNING = "budget_warning"

@dataclass
class SMTPConfig:
    host: str = "localhost"
    port: int = 587
    username: str = ""
    password: str = ""

@dataclass
class WebhookTarget:
    url: str
    events: list[WebhookEvent] = field(default_factory=list)
    format: WebhookFormat = WebhookFormat.JSON

class WebhookManager:
    def __init__(self) -> None:
        self._targets: list[WebhookTarget] = []

    def add_target(self, target: WebhookTarget) -> None:
        self._targets.append(target)

    async def notify(self, event: WebhookEvent, data: dict[str, Any]) -> None:
        pass
