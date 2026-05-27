"""Streaming support for LiteLLM HTTP transport."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StreamCallback(Protocol):
    """Protocol for streaming callbacks that receive text chunks."""

    async def __call__(self, text: str) -> None:
        """Called with each text chunk as it arrives."""
        ...


class StreamingTransportMixin:
    """Mixin that adds streaming support to LiteLLMHTTPTransport."""

    async def do_turn_streaming(self, on_token: StreamCallback) -> dict[str, Any]:
        """Send the current conversation to LiteLLM with streaming enabled.

        Args:
            on_token: Async callback that receives text chunks as they arrive.

        Returns:
            A dict with the same format as do_turn():
                text: str — the complete assistant's text response
                tool_calls: list[dict] — accumulated tool_call objects if any
                stop_reason: str — "tool_calls", "stop", or "length"
                raw: dict — the final response metadata (without streamed content)
        """
        if not hasattr(self, "_client") or self._client is None:
            raise RuntimeError("Transport not connected. Call connect() first.")

        if not hasattr(self, "_model"):
            raise RuntimeError("Transport not properly initialized.")

        # Build the payload similar to do_turn()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": getattr(self, "_messages", []),
            "stream": True,
        }

        system_prompt = getattr(self, "_system_prompt", None)
        if system_prompt:
            payload["messages"] = [
                {"role": "system", "content": system_prompt},
                *getattr(self, "_messages", []),
            ]

        tools = getattr(self, "_tools", [])
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Initialize accumulators
        accumulated_text = ""
        accumulated_tool_calls: list[dict[str, Any]] = []
        stop_reason = "stop"
        response_id = ""
        model = self._model

        # Make streaming request
        async with self._client.stream(
            "POST",
            f"{getattr(self, '_base_url', 'http://127.0.0.1:8016')}/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8")[:500]
                raise RuntimeError(
                    f"LiteLLM returned HTTP {response.status_code}: {body}"
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        chunk_str = line[6:]  # Remove "data: " prefix
                        chunk = json.loads(chunk_str)

                        # Extract delta from the first choice
                        if chunk.get("choices") and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            finish_reason = chunk["choices"][0].get("finish_reason")

                            # Handle text content
                            if "content" in delta and delta["content"]:
                                text_chunk = delta["content"]
                                accumulated_text += text_chunk
                                await on_token(text_chunk)

                            # Handle tool calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                for tool_call_delta in delta["tool_calls"]:
                                    index = tool_call_delta.get("index", 0)

                                    # Ensure we have enough slots in our accumulator
                                    while len(accumulated_tool_calls) <= index:
                                        accumulated_tool_calls.append(
                                            {
                                                "id": "",
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                        )

                                    # Update the tool call at this index
                                    existing_call = accumulated_tool_calls[index]
                                    if "id" in tool_call_delta:
                                        existing_call["id"] = tool_call_delta["id"]
                                    if "type" in tool_call_delta:
                                        existing_call["type"] = tool_call_delta["type"]
                                    if "function" in tool_call_delta:
                                        func_delta = tool_call_delta["function"]
                                        if "name" in func_delta:
                                            existing_call["function"]["name"] = (
                                                func_delta["name"]
                                            )
                                        if "arguments" in func_delta:
                                            existing_call["function"]["arguments"] += (
                                                func_delta["arguments"]
                                            )

                            # Handle finish reason
                            if finish_reason:
                                stop_reason = finish_reason

                            # Capture response ID and model from first chunk
                            if not response_id and "id" in chunk:
                                response_id = chunk["id"]
                            if not model and "model" in chunk:
                                model = chunk["model"]

                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Log error but continue processing other chunks
                        continue

        # Parse final tool calls arguments
        final_tool_calls = []
        for tc in accumulated_tool_calls:
            if tc["function"]["arguments"]:
                try:
                    parsed_args = json.loads(tc["function"]["arguments"])
                    tc["function"]["arguments"] = parsed_args
                except json.JSONDecodeError:
                    # Keep as string if parsing fails
                    pass
            final_tool_calls.append(tc)

        # Append assistant turn to history (similar to do_turn)
        if hasattr(self, "_messages"):
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_text,
            }
            if final_tool_calls:
                assistant_msg["tool_calls"] = final_tool_calls
            self._messages.append(assistant_msg)

        return {
            "text": accumulated_text,
            "tool_calls": final_tool_calls,
            "stop_reason": stop_reason,
            "raw": {
                "id": response_id,
                "model": model,
                "object": "chat.completion",
                "created": None,  # Would need timestamp logic if needed
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": accumulated_text,
                            "tool_calls": final_tool_calls
                            if final_tool_calls
                            else None,
                        },
                        "finish_reason": stop_reason,
                    }
                ],
                "usage": None,  # Usage would need to be tracked separately
            },
        }
