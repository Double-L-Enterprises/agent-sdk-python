"""Rate limiter for API calls.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ProviderLimits:
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

class ProviderRateLimiter:
    def __init__(self, limits: ProviderLimits | None = None) -> None:
        self._limits = limits or ProviderLimits()

    async def acquire(self, model: str | None = None) -> None:
        pass

    def release(self, model: str | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

global_rate_limiter = ProviderRateLimiter()
