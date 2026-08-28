"""Provider-independent analyst text generation with an optional OpenAI adapter."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Mapping

import requests

from .prompts import SYSTEM_PROMPT


class ProviderError(RuntimeError):
    """A safe provider-facing error that never includes credentials."""


class AnalystProvider(ABC):
    name = "unknown"

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], *, max_output_tokens: int = 450) -> str:
        raise NotImplementedError


class DisabledProvider(AnalystProvider):
    name = "disabled"

    def generate(self, messages: list[dict[str, str]], *, max_output_tokens: int = 450) -> str:
        del messages, max_output_tokens
        raise ProviderError("AI provider is disabled")


@dataclass
class OpenAIResponsesProvider(AnalystProvider):
    api_key: str
    model: str
    timeout: float = 15.0
    endpoint: str = "https://api.openai.com/v1/responses"
    name = "openai"

    def generate(self, messages: list[dict[str, str]], *, max_output_tokens: int = 450) -> str:
        if not self.api_key or not self.model:
            raise ProviderError("OpenAI provider configuration is incomplete")
        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "instructions": SYSTEM_PROMPT, "input": messages,
                      "max_output_tokens": max_output_tokens, "store": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError("The AI provider request failed") from exc
        output = payload.get("output", []) if isinstance(payload, dict) else []
        text = "".join(
            str(item.get("text", ""))
            for block in output if isinstance(block, dict) and block.get("type") == "message"
            for item in block.get("content", []) if isinstance(item, dict) and item.get("type") == "output_text"
        ).strip()
        if not text:
            raise ProviderError("The AI provider returned malformed output")
        return text


def provider_from_config(config: Mapping[str, Any] | None = None) -> AnalystProvider:
    """Read environment/secrets-like configuration without ever exposing a key."""
    config = config or {}
    provider_name = str(config.get("FPL_ANALYST_PROVIDER") or os.getenv("FPL_ANALYST_PROVIDER", "")).casefold()
    api_key = str(config.get("FPL_ANALYST_API_KEY") or os.getenv("FPL_ANALYST_API_KEY", ""))
    model = str(config.get("FPL_ANALYST_MODEL") or os.getenv("FPL_ANALYST_MODEL", ""))
    if not provider_name or provider_name in {"disabled", "none"} or not api_key:
        return DisabledProvider()
    if provider_name == "openai":
        return OpenAIResponsesProvider(api_key=api_key, model=model)
    return DisabledProvider()


class FakeProvider(AnalystProvider):
    """Test provider; it never makes a network call."""
    name = "fake"

    def __init__(self, response: str = "Grounded explanation", error: Exception | None = None):
        self.response, self.error = response, error

    def generate(self, messages: list[dict[str, str]], *, max_output_tokens: int = 450) -> str:
        del messages, max_output_tokens
        if self.error:
            raise self.error
        return self.response

