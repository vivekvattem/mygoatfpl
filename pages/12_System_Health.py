"""Operational visibility for active-session refresh and serving reliability."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fpl_predictor.config import LIVE_DATA_DIR
from fpl_predictor.monitoring import completed_gw_monitoring, prediction_distribution
from fpl_predictor.ui.components import (
    analyst_provider_config, configure_page, render_sidebar,
)
from fpl_predictor.analyst.provider import provider_from_config
from fpl_predictor.ui.reliability import cached_artifact_health, current_health


def _time(value) -> str:
    if value is None:
        return "Not yet"
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except (TypeError, ValueError, AttributeError):
        return "Unknown"


configure_page("System Health")
settings, bundle = render_sidebar()
assessment = current_health(bundle, settings, st.session_state)
icons = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNAVAILABLE": "🔴"}

st.title("System Health")
st.markdown(f"## {icons[assessment.level]} {assessment.level}")
if assessment.reasons:
    for reason in assessment.reasons:
        st.write(f"- {reason}")
else:
    st.success("Live data, feature schema, and production artifacts passed serving checks.")
st.caption("Monitoring is session-safe and active-session only; it is not a background scheduler.")

last_failure = st.session_state.get("runtime_last_refresh_failure") or {}
latencies = st.session_state.get("runtime_refresh_latencies") or {}
cols = st.columns(4)
cols[0].metric("Last successful check", _time(st.session_state.get("runtime_last_refresh_success")))
cols[1].metric("Next scheduled check", _time(st.session_state.get("runtime_next_refresh")))
cols[2].metric("API latency", f"{latencies.get('api_seconds', 0):.2f}s" if "api_seconds" in latencies else "—")
cols[3].metric("Pipeline latency", f"{latencies.get('pipeline_seconds', 0):.2f}s" if "pipeline_seconds" in latencies else "—")
if last_failure:
    st.warning(f"Last refresh failure ({_time(last_failure.get('at'))}): {last_failure.get('message', 'Unknown')}")
with st.expander("Latency details"):
    st.write({**bundle.latencies, **latencies,
              "analyst_provider_ms": (st.session_state.get("runtime_analyst_metrics") or {}).get("average_latency_ms")})

st.subheader("Serving contracts")
schema = bundle.live_summary.get("schema_validation", {})
artifacts = cached_artifact_health()
contract_cols = st.columns(4)
contract_cols[0].metric("Live players", len(bundle.predictions))
contract_cols[1].metric("Target GW", bundle.live_summary.get("target_gw", "—"))
contract_cols[2].metric("Feature schema", "PASS" if schema.get("passed") else "FAIL")
contract_cols[3].metric("Model artifacts", "PASS" if artifacts["passed"] else "FAIL")
with st.expander("Contract details"):
    st.write({"required_features": schema.get("required"), "available_features": schema.get("available"),
              "missing": schema.get("missing", []), "artifact_validation": artifacts["detail"]})
    if artifacts.get("error"):
        st.error(artifacts["error"])

st.subheader("Official API and live data")
api_cols = st.columns(5)
api_cols[0].metric("API status", "FAILED" if last_failure else "AVAILABLE" if bundle.status.available else "UNAVAILABLE")
api_cols[1].metric("Live freshness", "STALE" if bundle.status.stale else "CURRENT" if bundle.status.available else "UNAVAILABLE")
api_cols[2].metric("Fixture rows", len(bundle.fixtures))
api_cols[3].metric("Calendar rows", len(bundle.fixture_calendar))
api_cols[4].metric("Last failure", _time(last_failure.get("at")) if last_failure else "None")

st.subheader("Latest public FPL changes")
change = st.session_state.get("runtime_last_change") or {}
if not change:
    st.info("No live comparison has run in this browser session yet. Use Refresh Now to establish one.")
else:
    st.write(f"**Category:** {change.get('category')} · **Detected:** {_time(change.get('detected_at'))}")
    st.caption(f"Compact fingerprint: {(change.get('fingerprint') or {}).get('combined', 'unavailable')[:12]}…")
    player_changes = pd.DataFrame(change.get("player_changes", []))
    fixture_changes = pd.DataFrame(change.get("fixture_changes", []))
    if player_changes.empty:
        st.caption("No price, availability, or ownership changes detected.")
    else:
        st.dataframe(player_changes, width="stretch", hide_index=True)
    if fixture_changes.empty:
        st.caption("No fixture additions, removals, moves, kickoff, or FDR changes detected.")
    else:
        st.dataframe(fixture_changes, width="stretch", hide_index=True)
    for alert in change.get("schedule_alerts", []):
        st.warning(alert)
if bundle.fixture_calendar.empty or "schedule_label" not in bundle.fixture_calendar:
    st.caption("Confirmed DGW/BGW calendar: unavailable")
else:
    labels = bundle.fixture_calendar.schedule_label.value_counts()
    st.caption(f"Confirmed schedule rows · BGW: {int(labels.get('BGW', 0))} · "
               f"DGW: {int(labels.get('DGW', 0))} · TGW: {int(labels.get('TGW', 0))}")

st.subheader("Prediction monitoring")
distribution = prediction_distribution(bundle.predictions)
if distribution.get("status") == "UNAVAILABLE":
    st.info("Prediction distribution unavailable.")
else:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Distribution", distribution["status"])
    metric_cols[1].metric("Mean xPts", f"{distribution['mean']:.2f}")
    metric_cols[2].metric("Median xPts", f"{distribution['median']:.2f}")
    metric_cols[3].metric("95th percentile", f"{distribution['p95']:.2f}")
    metric_cols[4].metric("Maximum", f"{distribution['maximum']:.2f}")
    st.caption(f"Position means: {distribution.get('position_means', {})}")

monitoring_path = Path(LIVE_DATA_DIR) / "completed_gw_monitoring.csv"
scored = pd.read_csv(monitoring_path) if monitoring_path.exists() else pd.DataFrame()
completed = completed_gw_monitoring(scored)
st.write(f"**Completed-GW calibration:** {completed['status']}")
if completed["status"] != "INSUFFICIENT SAMPLE":
    st.json(completed)
else:
    st.caption("At least three completed Gameweeks with saved prediction/actual pairs are required. No retraining is triggered.")

st.subheader("Personalization and analyst")
analyst = st.session_state.get("runtime_analyst_metrics") or {}
provider = provider_from_config(analyst_provider_config())
personal_cols = st.columns(4)
personal_cols[0].metric("Squad", f"{len(bundle.squad)} players" if not bundle.squad.empty else "Unavailable")
personal_cols[1].metric("Finance", "Complete" if settings.bank is not None and settings.free_transfers is not None else "Incomplete")
personal_cols[2].metric("Analyst provider", provider.name.title())
personal_cols[3].metric("Fallbacks", int(analyst.get("fallbacks", 0)))
st.caption("A disabled analyst provider is a supported mode. Deterministic analysis remains available.")
st.write({"squad_source": bundle.decision_summary.get("squad_source", bundle.live_summary.get("squad_source")),
          "squad_gameweek": bundle.live_summary.get("squad_gameweek"),
          "bank": settings.bank, "free_transfers": settings.free_transfers,
          "selling_price_mode": "scenario_current_price" if settings.assume_selling_price_current else "strict"})
if analyst:
    st.write({"calls": analyst.get("calls", 0), "provider_calls": analyst.get("provider_calls", 0),
              "provider_successes": analyst.get("provider_successes", 0),
              "fallback_rate": (analyst.get("fallbacks", 0) / analyst.get("provider_calls", 1)
                                if analyst.get("provider_calls", 0) else 0),
              "average_latency_ms": analyst.get("average_latency_ms"),
              "grounding_validations": analyst.get("grounding_validations", 0),
              "grounding_failures": analyst.get("grounding_failures", 0),
              "failure_categories": analyst.get("failure_categories", {})})

st.subheader("Recent reliability events")
events = pd.DataFrame(st.session_state.get("runtime_event_log") or [])
if events.empty:
    st.info("No session events recorded yet.")
else:
    st.dataframe(events.iloc[::-1], width="stretch", hide_index=True)
