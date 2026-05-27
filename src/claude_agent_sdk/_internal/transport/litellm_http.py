"""LiteLLM HTTP transport — talks to LiteLLM via OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import Transport

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0  # seconds


class LiteLLMHTTPTransport(Transport):
    """Async HTTP transport that calls LiteLLM's OpenAI-compatible endpoint.

    This transport is a drop-in replacement for SubprocessCLITransport when
    you want to route through LiteLLM (or any OpenAI-compatible proxy) instead
    of the claude CLI subprocess.

    Message format emitted from read_messages() is compatible with the SDK's
    internal message_parser.parse_message() for the ``assistant`` and ``result``
    message types.  Other SDK message types (system, user, stream_event, etc.)
    are not produced by this transport since they have no equivalent in the
    OpenAI chat completions protocol.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8016",
        api_key: str = "sk-bbc8dc18c88aed96187cb3dea585b900e79601fd9f0fcf6cc93170b0e89fcca1",
        model: str = "qwen/qwen3-max",
        timeout: float = _DEFAULT_TIMEOUT,
        system_prompt: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._system_prompt = system_prompt

        self._client: httpx.AsyncClient | None = None
        self._ready = False

        # Buffered messages waiting to be consumed by read_messages()
        # Each element is a raw dict in SDK message format.
        self._pending_messages: list[dict[str, Any]] = []

        # Conversation history sent to the API each turn.
        # Format: list of {"role": "user"|"assistant"|"tool", "content": ...}
        self._messages: list[dict[str, Any]] = []

        # Tool definitions in OpenAI function-calling format.
        self._tools: list[dict[str, Any]] = []

        # Pending user message to send on next _do_turn()
        self._pending_prompt: str | None = None

        # Pending tool results from tool calls in the last assistant response.
        # Format: list of {"role": "tool", "tool_call_id": ..., "content": ...}
        self._pending_tool_results: list[dict[str, Any]] = []

        self._session_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Transport ABC
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create httpx client and verify LiteLLM is reachable."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        # Quick health check — LiteLLM exposes /health
        try:
            resp = await self._client.get(f"{self._base_url}/health", timeout=5.0)
            resp.raise_for_status()
            logger.debug("LiteLLM health check passed: %s", resp.status_code)
        except Exception as exc:
            logger.warning("LiteLLM health check failed (continuing anyway): %s", exc)

        self._ready = True

    async def write(self, data: str) -> None:
        """Accept a JSON payload from the SDK query layer.

        For the LiteLLM transport the SDK ``query.py`` layer is not used —
        callers should use ``AutonomousRunner`` or ``LiteLLMHTTPTransport``
        directly via the helper methods below.  This stub exists so the
        transport satisfies the ABC.
        """
        try:
            msg = json.loads(data)
            if msg.get("type") == "user" and "message" in msg:
                content = msg["message"].get("content", "")
                if isinstance(content, str):
                    self._pending_prompt = content
                elif isinstance(content, list):
                    texts = [b["text"] for b in content if b.get("type") == "text"]
                    self._pending_prompt = "\n".join(texts)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield buffered SDK-format messages (populated by _do_turn)."""
        return self._read_messages_impl()

    async def _read_messages_impl(self) -> AsyncIterator[dict[str, Any]]:
        for msg in self._pending_messages:
            yield msg
        self._pending_messages.clear()

    async def close(self) -> None:
        """Close the httpx client."""
        self._ready = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        """No-op for HTTP transport — no stdin stream to close."""
        pass

    # ------------------------------------------------------------------
    # Direct-use API (used by AutonomousRunner)
    # ------------------------------------------------------------------

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        """Register OpenAI-format tool definitions for this session."""
        self._tools = tools

    def add_user_message(self, content: str) -> None:
        """Append a user turn to the conversation history."""
        self._messages.append({"role": "user", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool = False) -> None:
        """Append a tool result to the conversation history."""
        result_content = content
        if is_error:
            result_content = f"[ERROR] {content}"
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        })

    async def do_turn(self) -> dict[str, Any]:
        """Send the current conversation to LiteLLM and return the parsed response.

        Returns a dict with:
            text: str — the assistant's text response (may be empty)
            tool_calls: list[dict] — OpenAI tool_call objects if any
            stop_reason: str — "tool_calls", "stop", or "length"
            raw: dict — the full OpenAI response body
        """
        if self._client is None:
            raise RuntimeError("Transport not connected. Call connect() first.")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": False,
        }

        if self._system_prompt:
            payload["messages"] = [
                {"role": "system", "content": self._system_prompt},
                *self._messages,
            ]

        if self._tools:
            payload["tools"] = self._tools
            payload["tool_choice"] = "auto"

        logger.debug(
            "LiteLLM request: model=%s, messages=%d, tools=%d",
            self._model,
            len(payload["messages"]),
            len(self._tools),
        )

        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
        )

        if resp.status_code != 200:
            body = resp.text[:500]
            raise RuntimeError(
                f"LiteLLM returned HTTP {resp.status_code}: {body}"
            )

        raw = resp.json()
        choice = raw["choices"][0]
        message = choice["message"]
        stop_reason = choice.get("finish_reason", "stop")

        # Extract text
        text = message.get("content") or ""

        # Extract tool calls
        tool_calls = message.get("tool_calls") or []

        # Append assistant turn to history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self._messages.append(assistant_msg)

        return {
            "text": text,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
            "raw": raw,
        }

    def build_sdk_assistant_message(self, turn_result: dict[str, Any]) -> dict[str, Any]:
        """Convert a do_turn() result into SDK-format assistant message dict."""
        content_blocks: list[dict[str, Any]] = []

        if turn_result["text"]:
            content_blocks.append({"type": "text", "text": turn_result["text"]})

        for tc in turn_result["tool_calls"]:
            fn = tc.get("function", {})
            try:
                tool_input = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_input = {"raw": fn.get("arguments", "")}

            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": fn.get("name", "unknown"),
                "input": tool_input,
            })

        usage_raw = turn_result["raw"].get("usage", {})
        usage = {
            "input_tokens": usage_raw.get("prompt_tokens", 0),
            "output_tokens": usage_raw.get("completion_tokens", 0),
        }

        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": content_blocks,
                "model": self._model,
                "stop_reason": turn_result["stop_reason"],
                "id": turn_result["raw"].get("id", ""),
                "usage": usage,
            },
            "session_id": self._session_id,
            "uuid": str(uuid.uuid4()),
        }

    def build_sdk_result_message(
        self,
        *,
        is_error: bool = False,
        result_text: str = "",
        num_turns: int = 0,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """Build an SDK-format 'result' message to signal end-of-run."""
        return {
            "type": "result",
            "subtype": "success" if not is_error else "error",
            "duration_ms": duration_ms,
            "duration_api_ms": duration_ms,
            "is_error": is_error,
            "num_turns": num_turns,
            "session_id": self._session_id,
            "result": result_text,
            "usage": None,
        }

    def set_model(self, model: str) -> None:
        """Switch the model for subsequent API calls. Conversation history is preserved."""
        logger.info("Switching model from %s to %s", self._model, model)
        self._model = model
