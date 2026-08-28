"""Pragmatic post-generation grounding checks for material FPL claims."""

from dataclasses import dataclass
import math
import re
from typing import Any

from .intents import normalize_text


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    failures: tuple[str, ...] = ()


def _walk(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _context_names(context: dict[str, Any]) -> set[str]:
    names = set()
    for value in _walk(context):
        if isinstance(value, str) and len(value.split()) <= 8:
            names.add(normalize_text(value))
    return names


def validate_player_mentions(answer: str, context: dict[str, Any], universe_names: set[str]) -> bool:
    supplied = _context_names(context)
    normalized_answer = normalize_text(answer)
    for name in universe_names:
        normalized = normalize_text(name)
        if normalized and re.search(rf"(?:^|\s){re.escape(normalized)}(?:$|\s)", normalized_answer):
            if normalized not in supplied:
                return False
    return True


def validate_numeric_claims(answer: str, context: dict[str, Any]) -> bool:
    """Validate numbers coupled to xPts, money, gains, scores, or minutes."""
    allowed = {round(float(value), 2) for value in _walk(context)
               if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))}
    patterns = [
        r"([+-]?\d+(?:\.\d+)?)\s*(?:xpts|projected points?|points? gain|gain)",
        r"(?:£|gbp)\s*([+-]?\d+(?:\.\d+)?)",
        r"([+-]?\d+(?:\.\d+)?)\s*(?:minutes?|mins?)",
    ]
    claimed = [float(match) for pattern in patterns for match in re.findall(pattern, answer, flags=re.I)]
    return all(any(abs(value - candidate) <= 0.011 for candidate in allowed) for value in claimed)


def validate_schedule_claims(answer: str, context: dict[str, Any]) -> bool:
    text = normalize_text(answer)
    claims_double = "double gameweek" in text or re.search(r"\bdgw\b", text)
    claims_blank = "blank gameweek" in text or re.search(r"\bbgw\b", text)
    schedule = context.get("schedule", {})
    negative_both = "no confirmed double or blank" in text or "no confirmed dgw bgw" in text
    if claims_double and "no confirmed double" not in text and not negative_both and not schedule.get("double_gameweeks"):
        return False
    if claims_blank and "no confirmed blank" not in text and not negative_both and not schedule.get("blank_gameweeks"):
        return False
    return True


def validate_chip_claims(answer: str, context: dict[str, Any]) -> bool:
    text = normalize_text(answer)
    states = context.get("chip_states", {})
    for chip, state in states.items():
        label = chip.replace("_", " ")
        if label in text and "available" in text and state != "available":
            return False
        if label in text and "used" in text and state != "used":
            return False
    return True


def validate_answer(answer: str, context: dict[str, Any], universe_names: set[str]) -> GroundingResult:
    checks = {
        "unsupported player mention": validate_player_mentions(answer, context, universe_names),
        "unsupported numeric claim": validate_numeric_claims(answer, context),
        "unsupported schedule claim": validate_schedule_claims(answer, context),
        "unsupported chip-state claim": validate_chip_claims(answer, context),
    }
    failures = tuple(label for label, passed in checks.items() if not passed)
    return GroundingResult(not failures, failures)
