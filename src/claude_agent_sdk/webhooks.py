"""Webhook notifications for agent team lifecycle events.

WebhookManager sends HTTP POST, Slack, or email notifications when teams
and agents reach key lifecycle milestones.

Events fired by TeamManager:
  team_started, agent_joined, agent_completed, agent_failed,
  budget_exceeded, team_completed, message_sent

Integration: import WebhookManager in team_manager.py and call
  await webhook_manager.fire(event, payload) at each lifecycle point.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ─── Event types ──────────────────────────────────────────────────────────────


class WebhookEvent(str, Enum):
    TEAM_STARTED = "team_started"
    AGENT_JOINED = "agent_joined"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    TEAM_COMPLETED = "team_completed"
    MESSAGE_SENT = "message_sent"


ALL_EVENTS: frozenset[str] = frozenset(e.value for e in WebhookEvent)


# ─── Target config ─────────────────────────────────────────────────────────────


class WebhookFormat(str, Enum):
    HTTP = "http"  # Generic JSON POST
    SLACK = "slack"  # Slack Block Kit formatted message
    EMAIL = "email"  # SMTP email


@dataclass
class SMTPConfig:
    """SMTP configuration for email webhooks."""

    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)


@dataclass
class WebhookTarget:
    """A single webhook destination.

    Args:
        url: Endpoint URL (HTTP/HTTPS for http and slack formats;
             unused for email format).
        events_filter: Set of event names to deliver. Empty = all events.
        format: Output format: http | slack | email.
        smtp: Required when format=email.
        headers: Extra HTTP headers (e.g., Authorization).
        name: Human-readable label for logging.
    """

    url: str = ""
    events_filter: set[str] = field(default_factory=set)  # empty = all
    format: WebhookFormat = WebhookFormat.HTTP
    smtp: SMTPConfig | None = None
    headers: dict[str, str] = field(default_factory=dict)
    name: str = "webhook"


# ─── Payload builders ─────────────────────────────────────────────────────────


def _build_http_payload(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **data,
    }


def _build_slack_payload(event: str, data: dict[str, Any]) -> dict[str, Any]:
    """Format a Slack Block Kit message for the given event."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Choose emoji and summary by event type
    emoji_map = {
        WebhookEvent.TEAM_STARTED.value: ":rocket:",
        WebhookEvent.AGENT_JOINED.value: ":busts_in_silhouette:",
        WebhookEvent.AGENT_COMPLETED.value: ":white_check_mark:",
        WebhookEvent.AGENT_FAILED.value: ":x:",
        WebhookEvent.BUDGET_EXCEEDED.value: ":warning:",
        WebhookEvent.TEAM_COMPLETED.value: ":tada:",
        WebhookEvent.MESSAGE_SENT.value: ":speech_balloon:",
    }
    emoji = emoji_map.get(event, ":information_source:")

    team_id = data.get("team_id", "unknown-team")
    agent_name = data.get("agent_name", "")
    status = data.get("status", "")
    turns = data.get("turns", "")
    cost = data.get("cost_dollars", "")
    summary = data.get("summary", "")
    model = data.get("model", "")

    # Build header line
    header = f"{emoji} *{event.replace('_', ' ').title()}* — team `{team_id}`"
    if agent_name:
        header += f" / agent `{agent_name}`"

    fields_block = []
    if status:
        fields_block.append({"type": "mrkdwn", "text": f"*Status:* {status}"})
    if model:
        fields_block.append({"type": "mrkdwn", "text": f"*Model:* {model}"})
    if turns != "":
        fields_block.append({"type": "mrkdwn", "text": f"*Turns:* {turns}"})
    if cost != "":
        fields_block.append({"type": "mrkdwn", "text": f"*Cost:* ${cost:.4f}"})

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header, "emoji": True},
        },
    ]
    if fields_block:
        blocks.append({"type": "section", "fields": fields_block[:10]})
    if summary:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Output:* {summary[:500]}"},
            }
        )
    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_{ts}_"}],
        }
    )

    return {"blocks": blocks}


def _build_email_subject(event: str, data: dict[str, Any]) -> str:
    team_id = data.get("team_id", "unknown")
    agent_name = data.get("agent_name", "")
    label = event.replace("_", " ").title()
    if agent_name:
        return f"[Agent SDK] {label} — {team_id} / {agent_name}"
    return f"[Agent SDK] {label} — {team_id}"


def _build_email_body(event: str, data: dict[str, Any]) -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        f"Event: {event}",
        f"Timestamp: {ts}",
        "",
    ]
    for k, v in sorted(data.items()):
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


# ─── Delivery ─────────────────────────────────────────────────────────────────


async def _deliver_http(
    target: WebhookTarget,
    payload: dict[str, Any],
    attempt: int,
) -> bool:
    """POST payload as JSON to target.url. Returns True on success."""
    try:
        import httpx

        headers = {"Content-Type": "application/json", **target.headers}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(target.url, json=payload, headers=headers)
            if resp.status_code < 300:
                logger.debug(
                    "[%s] HTTP webhook delivered (attempt %d, status %d)",
                    target.name,
                    attempt,
                    resp.status_code,
                )
                return True
            logger.warning(
                "[%s] HTTP webhook returned %d (attempt %d)",
                target.name,
                resp.status_code,
                attempt,
            )
            return False
    except Exception as exc:
        logger.warning(
            "[%s] HTTP webhook attempt %d failed: %s", target.name, attempt, exc
        )
        return False


async def _deliver_slack(
    target: WebhookTarget,
    payload: dict[str, Any],
    attempt: int,
) -> bool:
    """POST Slack Block Kit payload to target.url (Slack Incoming Webhook URL)."""
    return await _deliver_http(target, payload, attempt)  # same mechanics


def _deliver_email(
    target: WebhookTarget,
    event: str,
    data: dict[str, Any],
) -> bool:
    """Send SMTP email. Runs synchronously (called from thread pool)."""
    smtp_cfg = target.smtp
    if smtp_cfg is None:
        logger.error("[%s] Email webhook has no smtp config", target.name)
        return False

    subject = _build_email_subject(event, data)
    body = _build_email_body(event, data)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_cfg.from_addr or smtp_cfg.username
    msg["To"] = ", ".join(smtp_cfg.to_addrs)
    msg.attach(MIMEText(body, "plain"))

    try:
        if smtp_cfg.use_tls:
            server = smtplib.SMTP(smtp_cfg.host, smtp_cfg.port)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_cfg.host, smtp_cfg.port)
            server.ehlo()

        if smtp_cfg.username:
            server.login(smtp_cfg.username, smtp_cfg.password)

        server.sendmail(
            smtp_cfg.from_addr or smtp_cfg.username,
            smtp_cfg.to_addrs,
            msg.as_string(),
        )
        server.quit()
        logger.info("[%s] Email webhook sent to %s", target.name, smtp_cfg.to_addrs)
        return True
    except Exception as exc:
        logger.warning("[%s] Email webhook failed: %s", target.name, exc)
        return False


# ─── WebhookManager ───────────────────────────────────────────────────────────


class WebhookManager:
    """Fire lifecycle event notifications to configured webhook targets.

    Usage::
        manager = WebhookManager([
            WebhookTarget(
                url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
                events_filter={"team_completed", "agent_failed"},
                format=WebhookFormat.SLACK,
                name="slack-prod",
            ),
            WebhookTarget(
                url="https://my-server.com/agent-events",
                format=WebhookFormat.HTTP,
                headers={"Authorization": "Bearer mytoken"},
                name="internal-events",
            ),
        ])

        await manager.fire("team_started", {"team_id": "my-team", "agent_count": 3})
    """

    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0  # seconds (exponential: 2, 4, 8)

    def __init__(self, targets: list[WebhookTarget] | None = None) -> None:
        """Initialize the WebhookManager.

        Args:
            targets: List of WebhookTarget configs. Can be added later via add_target().
        """
        self._targets: list[WebhookTarget] = list(targets or [])

    def add_target(self, target: WebhookTarget) -> None:
        """Register an additional webhook target at runtime."""
        self._targets.append(target)
        logger.info("Webhook target added: %s (%s)", target.name, target.format)

    async def fire(self, event: str, data: dict[str, Any]) -> None:
        """Fire an event to all matching targets asynchronously.

        Delivery is attempted with up to MAX_RETRIES retries per target
        using exponential backoff. Failures are logged but never raised.

        Args:
            event: Event name (use WebhookEvent values or a custom string).
            data: Event payload dict (team_id, agent_name, status, etc.).
        """
        if not self._targets:
            return

        tasks = []
        for target in self._targets:
            # Filter: skip if target has a filter and this event isn't in it
            if target.events_filter and event not in target.events_filter:
                continue
            tasks.append(
                asyncio.create_task(self._deliver_with_retry(target, event, data))
            )

        if tasks:
            # Fire-and-forget: gather without blocking the caller
            asyncio.gather(*tasks, return_exceptions=True)

    async def fire_and_wait(self, event: str, data: dict[str, Any]) -> dict[str, bool]:
        """Fire event and wait for all deliveries to complete.

        Returns:
            Dict of target.name → success bool.
        """
        if not self._targets:
            return {}

        results: dict[str, bool] = {}
        tasks = []
        for target in self._targets:
            if target.events_filter and event not in target.events_filter:
                continue
            tasks.append(
                (
                    target.name,
                    asyncio.create_task(self._deliver_with_retry(target, event, data)),
                )
            )

        for name, task in tasks:
            try:
                results[name] = await task
            except Exception:
                results[name] = False

        return results

    async def _deliver_with_retry(
        self,
        target: WebhookTarget,
        event: str,
        data: dict[str, Any],
    ) -> bool:
        """Attempt delivery with exponential backoff. Returns True on success."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            success = await self._single_deliver(target, event, data, attempt)
            if success:
                return True
            if attempt < self.MAX_RETRIES:
                wait = self.BACKOFF_BASE**attempt
                logger.info(
                    "[%s] Retry %d/%d in %.1fs",
                    target.name,
                    attempt,
                    self.MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)

        logger.error(
            "[%s] Webhook delivery failed after %d attempts for event '%s'",
            target.name,
            self.MAX_RETRIES,
            event,
        )
        return False

    async def _single_deliver(
        self,
        target: WebhookTarget,
        event: str,
        data: dict[str, Any],
        attempt: int,
    ) -> bool:
        """Single delivery attempt dispatched by format type."""
        if target.format == WebhookFormat.HTTP:
            payload = _build_http_payload(event, data)
            return await _deliver_http(target, payload, attempt)

        elif target.format == WebhookFormat.SLACK:
            payload = _build_slack_payload(event, data)
            return await _deliver_slack(target, payload, attempt)

        elif target.format == WebhookFormat.EMAIL:
            # SMTP is blocking — run in thread executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _deliver_email, target, event, data)

        else:
            logger.error("[%s] Unknown webhook format: %s", target.name, target.format)
            return False


__all__ = [
    "WebhookManager",
    "WebhookTarget",
    "WebhookEvent",
    "WebhookFormat",
    "SMTPConfig",
]
