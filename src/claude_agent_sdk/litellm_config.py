"""LiteLLM configuration helper — reads from env vars with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM-backed transport."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_turns: int = 50
    checkpoint_dir: str | None = None
    escalation_model: str | None = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> LiteLLMConfig:
        """Load config from environment variables with defaults."""
        return cls(
            base_url=os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:8016"),
            api_key=os.environ.get(
                "LITELLM_API_KEY",
                "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
            ),
            model=os.environ.get("LITELLM_DEFAULT_MODEL", "qwen/qwen3-max"),
            max_turns=int(os.environ.get("LITELLM_MAX_TURNS", "50")),
            checkpoint_dir=os.environ.get("LITELLM_CHECKPOINT_DIR"),
            escalation_model=os.environ.get("LITELLM_ESCALATION_MODEL"),
            timeout=float(os.environ.get("LITELLM_TIMEOUT", "120.0")),
        )
