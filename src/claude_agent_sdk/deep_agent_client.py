"""DeepAgentClient - HTTP client for Deep Agent pipeline.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any
import httpx

logger = logging.getLogger(__name__)

@dataclass
class PipelineStatus:
    status: str
    result_summary: str | None = None
    output_files: list[str] | None = None
    error: str | None = None

class DeepAgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8040") -> None:
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def health(self) -> bool:
        try:
            client = httpx.AsyncClient(timeout=5.0)
            resp = await client.get(self._base_url + "/health")
            await client.aclose()
            return resp.status_code == 200
        except Exception:
            return False

    async def start_pipeline(self, task: str, project_id: str, output_dir: str, model: str = "qwen/qwen3-max") -> str:
        return "not-implemented"

    async def wait_for_completion(self, run_id: str) -> PipelineStatus:
        return PipelineStatus(status="not_implemented")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    @staticmethod
    def should_use_pipeline(task: str) -> bool:
        keywords = ["build app", "create app", "scaffold", "full stack", "web app"]
        lower = task.lower()
        return any(k in lower for k in keywords)
