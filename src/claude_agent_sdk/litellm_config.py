"""LiteLLM config helpers.
Created: 2026-05-27 23:00 CST
"""
from dataclasses import dataclass

@dataclass
class LiteLLMConfig:
    base_url: str = "http://127.0.0.1:8016"
    api_key: str = "sk-litellm"
    model: str = "qwen/qwen3-max"
    timeout: float = 120.0
