"""Session-scoped refresh orchestration and reliability telemetry for Streamlit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import streamlit as st

from fpl_predictor.api import FPLAPIClient, FPLAPIError
from fpl_predictor.config import HISTORICAL_ML_DIR, PROJECT_ROOT, RAW_DATA_DIR
from fpl_predictor.health import HealthAssessment, assess_health
from fpl_predictor.model_artifacts import ModelArtifactError, ProductionArtifacts
from fpl_predictor.monitoring import ChangeReport, change_summary, compare_live_state
from fpl_predictor.refresh import bump_generations, next_refresh_at, refresh_due
from fpl_predictor.ui.data import RefreshResult, run_pipeline_refresh
from fpl_predictor.ui.state import AppSettings


@dataclass(frozen=True)
class MonitoredRefreshResult:
    checked: bool
    success: bool
    changed: bool
    category: str
    message: str
    report: ChangeReport | None = None


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def append_event(state: dict[str, Any], level: str, category: str, message: str,
                 when: datetime | None = None) -> None:
    events = list(state.get("runtime_event_log") or [])
    events.append({"at": (when or datetime.now(timezone.utc)).isoformat(), "level": level,
                   "category": category, "message": message})
    state["runtime_event_log"] = events[-50:]


def _serialize_report(report: ChangeReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["detected_at"] = report.detected_at.isoformat()
    return payload


def run_monitored_refresh(
    settings: AppSettings,
    state: dict[str, Any],
    *,
    manual: bool = False,
    client: FPLAPIClient | None = None,
    pipeline_runner: Callable[[AppSettings], RefreshResult] = run_pipeline_refresh,
    now: datetime | None = None,
) -> MonitoredRefreshResult:
    """Check compact live state and rebuild only when meaningful inputs changed.

    Failed checks and failed rebuilds leave the last successful local artifacts
    untouched, so the UI can continue in an explicitly stale state.
    """
    checked_at = now or datetime.now(timezone.utc)
    interval = int(state.get("widget_refresh_interval_minutes", 10))
    if not refresh_due(bool(state.get("widget_auto_refresh", True)),
                       state.get("runtime_next_refresh"), checked_at, manual):
        return MonitoredRefreshResult(False, True, False, "NOT_DUE", "Refresh is not due.")

    state["runtime_last_refresh_check"] = checked_at
    state["runtime_next_refresh"] = next_refresh_at(checked_at, interval)
    api = client or FPLAPIClient()
    started = perf_counter()
    try:
        bootstrap = api.get_bootstrap_static()
        fixtures = api.get_fixtures()
    except (FPLAPIError, OSError, ValueError) as exc:
        state["runtime_refresh_latencies"] = {**(state.get("runtime_refresh_latencies") or {}),
                                               "api_seconds": perf_counter() - started}
        state["runtime_last_refresh_failure"] = {"at": checked_at.isoformat(), "message": str(exc)}
        append_event(state, "ERROR", "API_FAILURE", "Official FPL refresh failed; cached data retained.", checked_at)
        return MonitoredRefreshResult(True, False, False, "API_FAILURE",
                                      f"Official FPL API unavailable; cached data retained: {exc}")

    api_seconds = perf_counter() - started
    previous_bootstrap = _read_json(RAW_DATA_DIR / "bootstrap_static.json", None)
    previous_fixtures = _read_json(RAW_DATA_DIR / "fixtures.json", None)
    report = compare_live_state(previous_bootstrap, previous_fixtures, bootstrap, fixtures)
    state["runtime_live_fingerprint"] = report.fingerprint
    state["runtime_last_change"] = _serialize_report(report)
    state["runtime_refresh_latencies"] = {**(state.get("runtime_refresh_latencies") or {}),
                                           "api_seconds": api_seconds}
    if not report.changed:
        state["runtime_last_refresh_success"] = checked_at.isoformat()
        state["runtime_last_refresh_failure"] = None
        append_event(state, "INFO", report.category, "No meaningful live FPL changes detected.", checked_at)
        return MonitoredRefreshResult(True, True, False, report.category,
                                      "Live check complete — no meaningful changes.", report)

    pipeline_started = perf_counter()
    result = pipeline_runner(settings)
    state["runtime_refresh_latencies"] = {**state["runtime_refresh_latencies"],
                                           "pipeline_seconds": perf_counter() - pipeline_started}
    if not result.success:
        state["runtime_last_refresh_failure"] = {"at": checked_at.isoformat(), "message": result.message}
        append_event(state, "ERROR", "PIPELINE_FAILURE", "Refresh rebuild failed; cached data retained.", checked_at)
        return MonitoredRefreshResult(True, False, True, report.category,
                                      f"Live changes found, but rebuild failed; cached data retained: {result.message}", report)

    invalidated = bump_generations(state, report.category)
    state["runtime_last_refresh_success"] = result.completed_at.isoformat()
    state["runtime_last_refresh_failure"] = None
    append_event(state, "INFO", report.category,
                 f"Live outputs rebuilt; invalidated {', '.join(invalidated) or 'no'} dependent generations.", checked_at)
    for alert in report.schedule_alerts:
        append_event(state, "WARNING", "FIXTURE_ALERT", alert, checked_at)
    return MonitoredRefreshResult(True, True, True, report.category,
                                  f"{result.message} Updated: {change_summary(report)}.", report)


def ensure_refresh_schedule(state: dict[str, Any], now: datetime | None = None) -> None:
    if state.get("runtime_next_refresh") is None:
        state["runtime_next_refresh"] = next_refresh_at(
            state.get("runtime_last_refresh_check"),
            int(state.get("widget_refresh_interval_minutes", 10)),
            now or datetime.now(timezone.utc),
        )


@st.cache_resource(show_spinner=False)
def cached_artifact_health() -> dict[str, Any]:
    try:
        detail = ProductionArtifacts(
            PROJECT_ROOT, HISTORICAL_ML_DIR / "phase4_training.json"
        ).validate_all()
        return {"passed": True, "detail": detail, "error": None}
    except (ModelArtifactError, OSError, ValueError) as exc:
        return {"passed": False, "detail": {}, "error": str(exc)}


def record_analyst_response(state: dict[str, Any], response: Any) -> None:
    metrics = dict(state.get("runtime_analyst_metrics") or {})
    metrics["calls"] = int(metrics.get("calls", 0)) + 1
    provider_enabled = getattr(response, "provider", "disabled") != "disabled"
    if provider_enabled:
        metrics["provider_calls"] = int(metrics.get("provider_calls", 0)) + 1
    if getattr(response, "fallback_used", False):
        metrics["fallbacks"] = int(metrics.get("fallbacks", 0)) + 1
    elif provider_enabled:
        metrics["provider_successes"] = int(metrics.get("provider_successes", 0)) + 1
    categories = list(getattr(response, "failure_categories", ()) or ())
    failures = dict(metrics.get("failure_categories") or {})
    for category in categories:
        failures[category] = int(failures.get(category, 0)) + 1
        append_event(state, "WARNING", "ANALYST_FAILURE", f"Analyst fallback: {category}")
    metrics["failure_categories"] = failures
    latency = getattr(response, "latency_ms", None)
    if latency is not None:
        metrics["last_latency_ms"] = int(latency)
        metrics["total_latency_ms"] = int(metrics.get("total_latency_ms", 0)) + int(latency)
        metrics["average_latency_ms"] = metrics["total_latency_ms"] / metrics["calls"]
    if provider_enabled and "provider_failure" not in categories:
        metrics["grounding_validations"] = int(metrics.get("grounding_validations", 0)) + 1
        if not bool(getattr(response, "validation_passed", False)):
            metrics["grounding_failures"] = int(metrics.get("grounding_failures", 0)) + 1
    state["runtime_analyst_metrics"] = metrics


def current_health(bundle: Any, settings: AppSettings, state: dict[str, Any]) -> HealthAssessment:
    artifacts = cached_artifact_health()
    schema = bundle.live_summary.get("schema_validation", {})
    analyst = state.get("runtime_analyst_metrics") or {}
    provider_calls = int(analyst.get("provider_calls", 0))
    provider_successes = int(analyst.get("provider_successes", 0))
    provider_healthy = provider_calls == 0 or provider_successes > 0
    return assess_health(
        live_available=bundle.status.available,
        stale=bundle.status.stale or bool(state.get("runtime_last_refresh_failure")),
        schema_passed=bool(schema.get("passed", False)),
        artifacts_passed=bool(artifacts["passed"]),
        provider_enabled=provider_calls > 0,
        provider_healthy=provider_healthy,
        finance_complete=settings.bank is not None and settings.free_transfers is not None,
    )
