"""Deterministic application health classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HealthAssessment:
    level: str
    reasons: tuple[str, ...]


def assess_health(*, live_available: bool, stale: bool, schema_passed: bool,
                  artifacts_passed: bool, provider_enabled: bool = False,
                  provider_healthy: bool = True, finance_complete: bool = True) -> HealthAssessment:
    del provider_enabled
    unavailable = []
    if not live_available:
        unavailable.append("No valid live prediction data")
    if not schema_passed:
        unavailable.append("Live feature schema mismatch")
    if not artifacts_passed:
        unavailable.append("Production model artifacts unavailable")
    if unavailable:
        return HealthAssessment("UNAVAILABLE", tuple(unavailable))
    degraded = []
    if stale:
        degraded.append("Live data is stale")
    if not provider_healthy:
        degraded.append("Optional analyst provider is failing; deterministic fallback is active")
    if not finance_complete:
        degraded.append("Personalized financial state is incomplete")
    return HealthAssessment("DEGRADED" if degraded else "HEALTHY", tuple(degraded))
