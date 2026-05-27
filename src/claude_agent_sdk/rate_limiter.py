from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ProviderLimits:
    max_rpm: int = 60
    max_tpm: int = 100_000
    max_concurrent: int = 5

DEFAULT_LIMITS: dict[str, ProviderLimits] = {
    "nvidia": ProviderLimits(max_rpm=60, max_tpm=100_000, max_concurrent=5),
    "qwen": ProviderLimits(max_rpm=100, max_tpm=200_000, max_concurrent=10),
    "openai": ProviderLimits(max_rpm=60, max_tpm=150_000, max_concurrent=5),
    "anthropic": ProviderLimits(max_rpm=50, max_tpm=100_000, max_concurrent=3),
    "deepseek-ai": ProviderLimits(max_rpm=60, max_tpm=100_000, max_concurrent=5),
}

@dataclass
class _ProviderState:
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    active_concurrent: int = 0
    queue_depth: int = 0
    minute_start: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class ProviderRateLimiter:
    def __init__(self, overrides: dict[str, ProviderLimits] | None = None):
        self._limits: dict[str, ProviderLimits] = {**DEFAULT_LIMITS}
        if overrides:
            self._limits.update(overrides)
        self._states: dict[str, _ProviderState] = {}

    def _get_provider(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[0]
        return model

    def _get_state(self, provider: str) -> _ProviderState:
        if provider not in self._states:
            self._states[provider] = _ProviderState()
        return self._states[provider]

    def _get_limits(self, provider: str) -> ProviderLimits:
        return self._limits.get(provider, ProviderLimits())

    def _maybe_reset_minute(self, state: _ProviderState) -> None:
        now = time.monotonic()
        if now - state.minute_start >= 60.0:
            state.requests_this_minute = 0
            state.tokens_this_minute = 0
            state.minute_start = now

    async def acquire(self, model: str, estimated_tokens: int = 0) -> None:
        provider = self._get_provider(model)
        state = self._get_state(provider)
        limits = self._get_limits(provider)

        async with state.lock:
            while True:
                self._maybe_reset_minute(state)

                rpm_ok = state.requests_this_minute < limits.max_rpm
                tpm_ok = state.tokens_this_minute + estimated_tokens <= limits.max_tpm
                concurrent_ok = state.active_concurrent < limits.max_concurrent

                if rpm_ok and tpm_ok and concurrent_ok:
                    state.requests_this_minute += 1
                    state.tokens_this_minute += estimated_tokens
                    state.active_concurrent += 1
                    return

                state.queue_depth += 1
                # Release lock and wait before retrying
                state.lock.release()
                await asyncio.sleep(1.0)
                await state.lock.acquire()
                state.queue_depth = max(0, state.queue_depth - 1)

    def release(self, model: str, actual_tokens: int = 0) -> None:
        provider = self._get_provider(model)
        state = self._get_state(provider)
        state.active_concurrent = max(0, state.active_concurrent - 1)

    def status(self) -> dict[str, dict[str, Any]]:
        result = {}
        for provider, state in self._states.items():
            limits = self._get_limits(provider)
            self._maybe_reset_minute(state)
            result[provider] = {
                "rpm": f"{state.requests_this_minute}/{limits.max_rpm}",
                "tpm": f"{state.tokens_this_minute}/{limits.max_tpm}",
                "concurrent": f"{state.active_concurrent}/{limits.max_concurrent}",
                "queue_depth": state.queue_depth,
            }
        return result

#: Global rate limiter singleton — import and use directly across all runners.
global_rate_limiter = ProviderRateLimiter()

__all__ = ["ProviderRateLimiter", "ProviderLimits", "DEFAULT_LIMITS", "global_rate_limiter"]
