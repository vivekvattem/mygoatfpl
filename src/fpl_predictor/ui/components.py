"""Reusable Streamlit controls and read-only visual components."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from fpl_predictor.ui.data import DashboardBundle, dashboard_summary, load_dashboard_bundle
from fpl_predictor.ui.contracts import first_existing_column
from fpl_predictor.ui.formatting import money, points, scenario_mode_label
from fpl_predictor.ui.state import (
    AppSettings, SESSION_DEFAULTS, get_active_squad_file, initialize_runtime_session, project_relative_path,
)
from fpl_predictor.ui.reliability import (
    cached_artifact_health, current_health, ensure_refresh_schedule, record_analyst_response,
    run_monitored_refresh,
)


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} · FPL AI Predictor", page_icon="⚽", layout="wide")


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    initialize_runtime_session(st.session_state)
    ensure_refresh_schedule(st.session_state)


def current_settings() -> AppSettings:
    return AppSettings(
        entry_id=int(st.session_state.entry_id), squad_source=st.session_state.squad_source,
        squad_file=get_active_squad_file(st.session_state),
        bank=float(st.session_state.bank) if st.session_state.bank_known else None,
        free_transfers=int(st.session_state.free_transfers) if st.session_state.free_transfers_known else None,
        horizon=int(st.session_state.horizon), risk_profile=st.session_state.risk_profile,
        minimum_gain=float(st.session_state.minimum_gain),
        assume_selling_price_current=bool(st.session_state.assume_selling_price_current),
        refresh_ttl=int(st.session_state.refresh_ttl),
    )


@st.cache_data(ttl=600, show_spinner=False)
def cached_bundle(settings: AppSettings, live_generation: int = 0,
                  fixture_generation: int = 0,
                  personalized_generation: int = 0) -> DashboardBundle:
    del live_generation, fixture_generation, personalized_generation
    return load_dashboard_bundle(settings)


@st.fragment(run_every="60s")
def _auto_refresh_heartbeat(settings: AppSettings) -> None:
    """A one-minute active-session heartbeat; policy controls the actual interval."""
    result = run_monitored_refresh(settings, st.session_state)
    if result.checked and result.success and result.changed:
        st.session_state.last_refresh_message = result.message
        st.rerun(scope="app")


def render_sidebar() -> tuple[AppSettings, DashboardBundle]:
    initialize_session()
    # Validate once per process; subsequent pages reuse the resource result and
    # never repeatedly deserialize production models.
    cached_artifact_health()
    with st.sidebar:
        st.header("Decision controls")
        st.number_input("FPL Entry ID (demo default)", min_value=1, step=1, key="entry_id")
        st.selectbox("Squad source", ["manual_file", "public_api"], key="squad_source",
                     format_func=lambda value: "Manual pre-deadline" if value == "manual_file" else "Public post-deadline")
        st.text_input("Squad file (optional)", key="squad_file_path_input",
                      placeholder="data/live/manual_squad.json")
        st.checkbox("Bank known", key="bank_known")
        if st.session_state.bank_known:
            st.number_input("Bank (£m)", min_value=0.0, step=0.1, key="bank")
        else:
            st.caption("Bank: Unknown")
        st.checkbox("Free transfers known", key="free_transfers_known")
        if st.session_state.free_transfers_known:
            st.number_input("Free transfers", min_value=0, step=1, key="free_transfers")
        else:
            st.caption("Free transfers: Unknown")
        st.select_slider("Planning horizon", options=[1, 3, 5], key="horizon")
        st.selectbox("Risk profile", ["safe", "balanced", "aggressive"], key="risk_profile")
        st.checkbox("Scenario Mode: assume current price = selling price", key="assume_selling_price_current",
                    help="Use only for temporary transfer estimates when authoritative selling prices are unknown.")
        if st.session_state.assume_selling_price_current:
            st.warning(scenario_mode_label(True))
        st.divider()
        st.markdown("**Live reliability**")
        st.checkbox("Auto refresh", key="widget_auto_refresh",
                    help="Runs only while this browser session is active.")
        st.selectbox("Refresh interval", [5, 10, 15, 30, 60],
                     key="widget_refresh_interval_minutes",
                     format_func=lambda value: f"{value} minutes")
        scheduled = st.session_state.get("runtime_scheduled_interval")
        interval = int(st.session_state.widget_refresh_interval_minutes)
        if scheduled != interval:
            # Internal key only; the interval widget remains widget-owned.
            st.session_state.runtime_scheduled_interval = interval
            st.session_state.runtime_next_refresh = None
            ensure_refresh_schedule(st.session_state)
        refresh = st.button("Refresh Now", type="primary", width="stretch", key="widget_refresh_now")
        last_success = st.session_state.get("runtime_last_refresh_success")
        next_refresh = st.session_state.get("runtime_next_refresh")
        st.caption(f"Last successful check: {_format_runtime_time(last_success)}")
        st.caption("Next refresh: Disabled" if not st.session_state.widget_auto_refresh
                   else f"Next refresh: {_format_runtime_time(next_refresh)}")
    settings = current_settings()
    if refresh:
        with st.status("Checking official FPL state…", expanded=True) as status:
            st.write("Fetching compact player, fixture, and Gameweek state…")
            result = run_monitored_refresh(settings, st.session_state, manual=True)
            st.write("Validating response schema and comparing the live fingerprint…")
            if result.changed:
                st.write("Updating live features, projections, and signals…")
                if result.category in {"FIXTURES_CHANGED", "GAMEWEEK_CHANGED", "MULTIPLE_CHANGES"}:
                    st.write("Refreshing fixture calendar and DGW/BGW signals…")
                if settings.squad_file is not None:
                    st.write("Updating dependent personalized decisions…")
            elif result.success:
                st.write("No meaningful changes; downstream optimization remains cached.")
            if result.success:
                st.session_state.last_refresh_message = result.message
                status.update(label="Refresh complete", state="complete")
            else:
                st.session_state.last_refresh_error = result.message
                status.update(label="Refresh failed — cached data retained", state="error")
    if st.session_state.get("last_refresh_message"):
        st.sidebar.success(st.session_state.last_refresh_message)
    if st.session_state.get("last_refresh_error"):
        st.sidebar.error(st.session_state.last_refresh_error)
    _auto_refresh_heartbeat(settings)
    bundle = cached_bundle(
        settings,
        st.session_state.get("live_generation", 0),
        st.session_state.get("fixture_generation", 0),
        st.session_state.get("personalized_generation", 0),
    )
    return settings, bundle


def _format_runtime_time(value) -> str:
    if value is None:
        return "Not yet"
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        return parsed.astimezone().strftime("%H:%M:%S %Z")
    except (TypeError, ValueError, AttributeError):
        return "Unknown"


def render_data_status(bundle: DashboardBundle) -> None:
    if not bundle.status.available:
        st.error("Live FPL data is temporarily unavailable. Please retry shortly.")
        return
    refresh_failure = st.session_state.get("runtime_last_refresh_failure")
    if refresh_failure:
        cached_at = bundle.status.timestamp.astimezone().strftime("%H:%M %Z") if bundle.status.timestamp else "an unknown time"
        retry = _format_runtime_time(st.session_state.get("runtime_next_refresh"))
        st.warning(f"STALE DATA — latest refresh failed. Using data from {cached_at}. Next automatic retry: {retry}.")
    elif bundle.status.stale:
        st.warning(bundle.status.message)
    timestamp = bundle.status.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if bundle.status.timestamp else "Unknown"
    schema = bundle.live_summary.get("schema_validation", {})
    required, available = schema.get("required", "?"), schema.get("available", "?")
    passed = schema.get("passed", False)
    st.caption(f"Data: {timestamp} · Schema: {available}/{required} {'PASS' if passed else 'FAIL'} · Squad: {bundle.decision_summary.get('squad_source', 'unknown')}")


def render_reliability_status(bundle: DashboardBundle, settings: AppSettings) -> None:
    assessment = current_health(bundle, settings, st.session_state)
    icons = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNAVAILABLE": "🔴"}
    with st.container(border=True):
        st.markdown(f"**{icons[assessment.level]} System status: {assessment.level}**")
        st.caption(" · ".join(assessment.reasons) if assessment.reasons else
                   "Live data, schema, and production artifacts passed serving checks.")
        schema = bundle.live_summary.get("schema_validation", {})
        artifact_ok = cached_artifact_health()["passed"]
        status_columns = st.columns(5)
        status_columns[0].metric("Live data", "STALE" if bundle.status.stale else
                                 "HEALTHY" if bundle.status.available else "UNAVAILABLE")
        status_columns[1].metric("Schema", f"{schema.get('available', '?')}/{schema.get('required', '?')}")
        status_columns[2].metric("Models", "HEALTHY" if artifact_ok else "UNAVAILABLE")
        status_columns[3].metric("Auto refresh", "ON" if st.session_state.widget_auto_refresh else "OFF")
        status_columns[4].metric("Next refresh", _format_runtime_time(
            st.session_state.get("runtime_next_refresh")) if st.session_state.widget_auto_refresh else "Disabled")
        change = st.session_state.get("runtime_last_change") or {}
        if st.session_state.get("widget_show_reliability_alerts") and change:
            st.caption(f"Latest live check: {change.get('category', 'Unknown')} · "
                       f"{len(change.get('fixture_changes', []))} fixture changes · "
                       f"{len(change.get('player_changes', []))} player changes")
        if (st.session_state.get("widget_show_player_change_alerts") and change
                and not bundle.squad.empty and "player_id" in bundle.squad):
            owned = set(pd.to_numeric(bundle.squad.player_id, errors="coerce").dropna().astype(int))
            affected = [item for item in change.get("player_changes", [])
                        if item.get("player_id") in owned]
            if affected:
                names = sorted({item.get("player", "Unknown") for item in affected})
                st.warning(f"Owned-player live changes: {', '.join(names)}")


def render_pitch(xi: pd.DataFrame, captain: str | None, vice: str | None) -> None:
    if xi.empty:
        st.info("Run the optimizer to create the projected XI.")
        return
    for position in ("GK", "DEF", "MID", "FWD"):
        group = xi[xi.position.eq(position)]
        st.markdown(f"**{position}**")
        columns = st.columns(max(1, len(group)))
        for column, row in zip(columns, group.itertuples(index=False)):
            marker = " (C)" if row.player == captain else " (V)" if row.player == vice else ""
            availability = getattr(row, "availability", "unknown")
            with column.container(border=True):
                st.markdown(f"**{row.player}{marker}**")
                st.caption(f"{getattr(row, 'team', 'Unknown')} · {money(getattr(row, 'price', None))}")
                st.write(f"{points(getattr(row, 'availability_adjusted_xpts', None))} xPts")
                st.caption(str(availability).title())


def render_kpis(bundle: DashboardBundle) -> None:
    values = dashboard_summary(bundle)
    columns = st.columns(5)
    columns[0].metric("Projected XI", points(values["projected_xi"]))
    columns[1].metric("3-GW weighted", points(values["weighted_3gw"]))
    columns[2].metric("5-GW weighted", points(values["weighted_5gw"]))
    columns[3].metric("Captain", values["captain"] or "—")
    columns[4].metric("Transfer", values["transfer_decision"])


def render_downloads(bundle: DashboardBundle) -> None:
    files = {"Player predictions": "player_predictions.csv", "Optimized XI": "optimized_xi.csv",
             "Transfer candidates": "transfer_candidates.csv", "Replacement shortlists": "replacement_shortlists.csv",
             "Decision summary": "decision_summary.json"}
    columns = st.columns(len(files))
    for column, (label, filename) in zip(columns, files.items()):
        path = Path(project_relative_path("data/live")) / filename
        if path.exists():
            mime = "application/json" if path.suffix == ".json" else "text/csv"
            column.download_button(label, path.read_bytes(), file_name=filename, mime=mime, width="stretch")


def require_predictions(bundle: DashboardBundle) -> bool:
    artifacts = cached_artifact_health()
    if not artifacts["passed"]:
        st.error("Prediction pipeline unavailable")
        st.warning("Production model artifacts did not pass startup validation. Static pages remain available.")
        return False
    if bundle.predictions.empty:
        st.error("Prediction pipeline unavailable")
        st.info("Live FPL data is temporarily unavailable. Please retry shortly or use Refresh FPL Data.")
        return False
    schema = bundle.live_summary.get("schema_validation", {})
    if schema and not schema.get("passed", False):
        required, available = schema.get("required", "?"), schema.get("available", "?")
        st.error("Prediction pipeline unavailable")
        st.warning(f"Live schema mismatch detected. Expected: {required}. Available: {available}.")
        with st.expander("Schema diagnostics", expanded=False):
            st.write({"missing": schema.get("missing", []),
                      "dtype_mismatches": schema.get("dtype_mismatches", [])})
        return False
    return True


def render_no_squad_state() -> None:
    """Explain fresh-session capabilities without inventing personalized decisions."""
    st.info("No personalized squad loaded")
    st.write(
        "Live player rankings are available. Upload a manual squad or load public picks to unlock:\n\n"
        "- optimized XI\n- captaincy\n- transfers\n- chip personalization"
    )


SIGNAL_ICONS = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "GREY": "⚪"}


def signal_badge(signal: str, label: str | None = None) -> str:
    value = str(signal).upper() if str(signal).upper() in SIGNAL_ICONS else "GREY"
    return f"{SIGNAL_ICONS[value]} {value}" + (f" — {label}" if label else "")


def action_badge(action: str) -> str:
    signal = "GREEN" if action in {"BUY", "HOLD"} else "RED" if action == "SELL" else "YELLOW"
    return signal_badge(signal, action)


def risk_summary(players: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if players.empty or "overall_signal" not in players:
        return pd.DataFrame()
    order = pd.Categorical(players.overall_signal, categories=["RED", "YELLOW", "GREY", "GREEN"], ordered=True)
    projection = first_existing_column(players, (
        "weighted_xpts_5", "xpts_5gw", "five_gw_xpts", "adjusted_xpts", "raw_xpts", "xpts",
    ))
    result = players.assign(_order=order)
    result = (result.sort_values(["_order", projection], ascending=[True, True]) if projection is not None
              else result.sort_values("_order"))
    columns = [column for column in ["player", "overall_signal", "action", "risk_reason"] if column in result]
    return result.head(limit)[columns]


def fixture_alert(gw: int, label: str, teams: list[str], status: str = "CONFIRMED") -> None:
    signal = "RED" if label == "BGW" else "YELLOW" if label == "TGW" else "GREEN"
    with st.container(border=True):
        st.markdown(f"**{signal_badge(signal, f'GW{gw} {label}')}**")
        st.write(", ".join(teams))
        st.caption(f"Status: {status}")


def chip_card(label: str, signal: str, score: object, reason: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{label}: {signal_badge(signal)}**")
        st.metric("Score", "—" if score is None or pd.isna(score) else f"{float(score):.2f}")
        st.caption(reason)


ANALYST_SUGGESTED_QUESTIONS = (
    "What should I do this week?",
    "Should I roll my transfer?",
    "Who should I captain?",
    "Who are my biggest risks?",
    "Should I use a chip?",
    "Which player should I replace first?",
)


def analyst_suggested_questions() -> tuple[str, ...]:
    return ANALYST_SUGGESTED_QUESTIONS


def analyst_provider_config() -> dict[str, str]:
    """Read deployed Streamlit secrets safely; environment fallback lives in the provider."""
    values = {}
    try:
        secrets = st.secrets
        for key in ("FPL_ANALYST_PROVIDER", "FPL_ANALYST_API_KEY", "FPL_ANALYST_MODEL"):
            if key in secrets:
                values[key] = str(secrets[key])
    except (FileNotFoundError, RuntimeError):
        pass
    return values


@st.cache_data(ttl=600, show_spinner=False)
def cached_analyst_context(question: str, bundle: DashboardBundle, settings: AppSettings,
                           overrides: tuple[tuple[str, str], ...], analyst_generation: int = 0):
    """Cache deterministic evidence selection, never provider-generated chat text."""
    del analyst_generation
    from fpl_predictor.analyst.context import build_analyst_context
    return build_analyst_context(bundle, settings, question, dict(overrides))


def analyst_evidence_text(sources) -> str:
    from fpl_predictor.analyst.citations import evidence_badges
    return " · ".join(evidence_badges(sources)) or "No structured evidence available"


def render_analyst_result(response) -> None:
    st.markdown(response.answer)
    st.caption(f"Confidence: {response.confidence} · Provider: {response.provider} · "
               f"{'Deterministic fallback' if response.fallback_used else 'Grounded AI explanation'}")
    with st.expander("Evidence used", expanded=False):
        st.write(analyst_evidence_text(response.evidence))
        for label, value in response.evidence_details.items():
            st.write(f"**{label}:** {value if value is not None else 'Unavailable'}")


def render_explain_button(question: str, bundle: DashboardBundle, settings: AppSettings, key: str) -> None:
    """Explain existing outputs without rerunning prediction or optimization."""
    if st.button("Explain this recommendation", key=key):
        from fpl_predictor.analyst.provider import provider_from_config
        from fpl_predictor.analyst.service import AnalystService
        overrides = {chip: st.session_state.get(f"chip_{chip}", "unknown") for chip in
                     ("wildcard", "free_hit", "bench_boost", "triple_captain")}
        context = cached_analyst_context(question, bundle, settings, tuple(overrides.items()),
                                         st.session_state.get("analyst_generation", 0))
        universe = set(bundle.predictions.player.astype(str)) if not bundle.predictions.empty else set()
        response = AnalystService(
            provider_from_config(analyst_provider_config()),
            monitor=lambda value: record_analyst_response(st.session_state, value),
        ).answer_context(
            question, context, universe)
        render_analyst_result(response)
