"""Refresh scheduling and dependency-generation rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


VALID_INTERVALS = (5, 10, 15, 30, 60)


def validate_refresh_interval(minutes: int) -> int:
    value = int(minutes)
    if value not in VALID_INTERVALS:
        raise ValueError(f"Refresh interval must be one of {VALID_INTERVALS} minutes")
    return value


def next_refresh_at(last_check: datetime | None, interval_minutes: int,
                    now: datetime | None = None) -> datetime:
    interval = validate_refresh_interval(interval_minutes)
    base = last_check or now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base + timedelta(minutes=interval)


def refresh_due(enabled: bool, next_refresh: datetime | None,
                now: datetime | None = None, manual: bool = False) -> bool:
    if manual:
        return True
    if not enabled or next_refresh is None:
        return False
    current = now or datetime.now(timezone.utc)
    return current >= next_refresh


def invalidated_generations(category: str) -> tuple[str, ...]:
    if category == "NO_CHANGE":
        return ()
    if category == "PLAYER_DATA_CHANGED":
        return ("live_generation", "personalized_generation", "analyst_generation")
    if category == "FIXTURES_CHANGED":
        return ("fixture_generation", "live_generation", "personalized_generation", "analyst_generation")
    return ("live_generation", "fixture_generation", "personalized_generation", "analyst_generation")


def bump_generations(state: dict[str, Any], category: str) -> tuple[str, ...]:
    changed = invalidated_generations(category)
    for key in changed:
        state[key] = int(state.get(key, 0)) + 1
    return changed
