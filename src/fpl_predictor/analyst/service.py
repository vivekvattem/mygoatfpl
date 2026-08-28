"""Analyst orchestration: route, ground, optionally explain, and safely fall back."""

from dataclasses import dataclass
import logging
import time
from typing import Any

from .context import AnalystContext, build_analyst_context
from .deterministic import deterministic_answer
from .grounding import validate_answer
from .prompts import build_messages
from .provider import AnalystProvider, DisabledProvider, ProviderError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalystResponse:
    answer: str
    intent: str
    confidence: str
    evidence: tuple[str, ...]
    evidence_details: dict[str, Any]
    provider: str
    fallback_used: bool
    validation_passed: bool
    latency_ms: int


class AnalystService:
    def __init__(self, provider: AnalystProvider | None = None):
        self.provider = provider or DisabledProvider()

    def answer(self, question: str, bundle: Any, settings: Any,
               chip_overrides: dict[str, str] | None = None) -> AnalystResponse:
        context: AnalystContext = build_analyst_context(bundle, settings, question, chip_overrides)
        universe = set(bundle.predictions.player.astype(str)) if not bundle.predictions.empty else set()
        return self.answer_context(question, context, universe)

    def answer_context(self, question: str, context: AnalystContext,
                       universe_names: set[str]) -> AnalystResponse:
        """Explain a prebuilt/cached deterministic context; final chat text is never cached."""
        started = time.perf_counter()
        if context.clarification:
            return self._result(context.clarification, context, True, True, started)
        fallback = deterministic_answer(context.intent, context.payload, context.confidence)
        if isinstance(self.provider, DisabledProvider):
            return self._result(fallback, context, True, True, started)
        try:
            candidate = self.provider.generate(build_messages(question, context.payload, fallback))
            grounding = validate_answer(candidate, context.payload, universe_names)
            if grounding.passed:
                return self._result(candidate, context, False, True, started)
            LOGGER.warning("analyst grounding failed: intent=%s failures=%s", context.intent, grounding.failures)
            return self._result(fallback, context, True, False, started)
        except (ProviderError, TimeoutError, ValueError, TypeError):
            LOGGER.warning("analyst provider failed: intent=%s provider=%s", context.intent, self.provider.name)
            return self._result(fallback, context, True, True, started)

    def _result(self, answer: str, context: AnalystContext, fallback: bool, validation: bool,
                started: float) -> AnalystResponse:
        latency = int((time.perf_counter() - started) * 1000)
        LOGGER.info("analyst intent=%s provider=%s fallback=%s validation=%s latency_ms=%s",
                    context.intent, self.provider.name, fallback, validation, latency)
        return AnalystResponse(answer, context.intent, context.confidence, context.evidence,
                               context.evidence_details, self.provider.name, fallback, validation, latency)
