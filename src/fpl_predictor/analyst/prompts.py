"""Compact grounded prompts for an optional explanation provider."""

import json
from typing import Any


SYSTEM_PROMPT = """You are an FPL analytical explanation layer.
Use only the structured evidence supplied to you.
Do not invent player prices, fixtures, injuries, xPts, ownership, chip availability,
squad state, transfer gains, or financial values. Do not override optimizer legality.
If information is unavailable, say so. Distinguish prediction from certainty and never
promise guaranteed points. Keep the answer concise and FPL-focused. Prefer supplied
numerical evidence. The confidence label is fixed by the application; do not change it."""


def build_messages(question: str, context: dict[str, Any], fallback: str) -> list[dict[str, str]]:
    compact = json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return [{
        "role": "user",
        "content": (f"Question: {question}\nStructured evidence: {compact}\n"
                    f"Deterministic recommendation to preserve: {fallback}\n"
                    "Explain the recommendation using only this evidence."),
    }]

