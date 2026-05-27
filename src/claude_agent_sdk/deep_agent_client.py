"""HTTP client for Deep Agent's deterministic pipeline API.

Routes app-building tasks to Deep Agent instead of running them locally.
Deep Agent handles: stack detection, scaffolding, BOM assembly, linting, testing.
Agent SDK handles: tasks, edits, research, debugging.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Keywords that indicate an app-building task → route to Deep Agent
APP_KEYWORDS = re.compile(
    r"\b(app|dashboard|website|web\s*app|frontend|backend|scaffold|"
    r"project|landing\s*page|portal|admin\s*panel|api\s*server|"
    r"mobile\s*app|react|next\.?js|fastapi|express|django|flask)\b",
    re.IGNORECASE,
)


@dataclass
class PipelineStatus:
    run_id: str
    status: str  # started, running, completed, failed, cancelled
    current_stage: str | None = None
    stages_completed: list[str] = field(default_factory=list)
    stages_remaining: list[str] = field(default_factory=list)
    error: str | None = None
    result_summary: str | None = None
    output_files: list[str] = field(default_factory=list)


class DeepAgentClient:
    """HTTP client for Deep Agent's pipeline API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8040",
        timeout: float = 300.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def health(self) -> bool:
        """Check if Deep Agent is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def start_pipeline(
        self,
        task: str,
        project_id: str,
        output_dir: str,
        model: str = "qwen/qwen3-max",
        stages: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Start a Deep Agent pipeline run. Returns run_id."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "task": task,
            "project_id": project_id,
            "output_dir": output_dir,
            "model": model,
        }
        if stages:
            payload["stages"] = stages
        if config:
            payload["config"] = config

        resp = await client.post("/pipeline/start", json=payload)
        resp.raise_for_status()
        data = resp.json()
        run_id = data.get("run_id", data.get("id", ""))
        logger.info("Pipeline started: run_id=%s project=%s", run_id, project_id)
        return run_id

    async def get_status(self, run_id: str) -> PipelineStatus:
        """Get current pipeline status."""
        client = await self._get_client()
        resp = await client.get(f"/pipeline/{run_id}/status")
        resp.raise_for_status()
        data = resp.json()
        return PipelineStatus(
            run_id=run_id,
            status=data.get("status", "unknown"),
            current_stage=data.get("current_stage"),
            stages_completed=data.get("stages_completed", []),
            stages_remaining=data.get("stages_remaining", []),
            error=data.get("error"),
            result_summary=data.get("result_summary"),
            output_files=data.get("output_files", []),
        )

    async def get_stages(self, run_id: str) -> list[dict[str, Any]]:
        """Get stage-by-stage progress."""
        client = await self._get_client()
        resp = await client.get(f"/pipeline/{run_id}/stages")
        resp.raise_for_status()
        return resp.json()

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        client = await self._get_client()
        resp = await client.post(f"/pipeline/{run_id}/cancel")
        return resp.status_code == 200

    async def stream_events(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Stream pipeline events via SSE."""
        import json

        client = await self._get_client()
        async with client.stream(
            "GET", f"/pipeline/{run_id}/events", timeout=None
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

    async def wait_for_completion(
        self, run_id: str, poll_interval: float = 10.0, timeout: float = 3600.0
    ) -> PipelineStatus:
        """Poll until pipeline completes or times out."""
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            status = await self.get_status(run_id)
            if status.status in ("completed", "failed", "cancelled"):
                return status
            await asyncio.sleep(poll_interval)
        return await self.get_status(run_id)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def should_use_pipeline(task: str, file_count_estimate: int = 0) -> bool:
        """Determine if a task should go through Deep Agent's pipeline.

        Returns True if:
        - Task description contains app-building keywords
        - Task is estimated to create 3+ files
        """
        if APP_KEYWORDS.search(task):
            return True
        if file_count_estimate >= 3:
            return True
        return False


__all__ = ["DeepAgentClient", "PipelineStatus"]
