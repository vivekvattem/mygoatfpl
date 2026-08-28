"""Reusable Streamlit controls and read-only visual components."""

from datetime import timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from fpl_predictor.ui.data import DashboardBundle, dashboard_summary, load_dashboard_bundle, run_pipeline_refresh
from fpl_predictor.ui.formatting import money, points, scenario_mode_label
from fpl_predictor.ui.state import AppSettings, SESSION_DEFAULTS, project_relative_path


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} · FPL AI Predictor", page_icon="⚽", layout="wide")


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def current_settings() -> AppSettings:
    return AppSettings(
        entry_id=int(st.session_state.entry_id), squad_source=st.session_state.squad_source,
        squad_file=project_relative_path(st.session_state.squad_file),
        bank=float(st.session_state.bank) if st.session_state.bank_known else None,
        free_transfers=int(st.session_state.free_transfers) if st.session_state.free_transfers_known else None,
        horizon=int(st.session_state.horizon), risk_profile=st.session_state.risk_profile,
        minimum_gain=float(st.session_state.minimum_gain),
        assume_selling_price_current=bool(st.session_state.assume_selling_price_current),
        refresh_ttl=int(st.session_state.refresh_ttl),
    )


@st.cache_data(ttl=600, show_spinner=False)
def cached_bundle(settings: AppSettings, refresh_generation: int = 0) -> DashboardBundle:
    del refresh_generation
    return load_dashboard_bundle(settings)


def render_sidebar() -> tuple[AppSettings, DashboardBundle]:
    initialize_session()
    with st.sidebar:
        st.header("Decision controls")
        st.number_input("FPL Entry ID (demo default)", min_value=1, step=1, key="entry_id")
        st.selectbox("Squad source", ["manual_file", "public_api"], key="squad_source",
                     format_func=lambda value: "Manual pre-deadline" if value == "manual_file" else "Public post-deadline")
        st.text_input("Squad file", key="squad_file")
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
        if st.session_state.assume_selling_price_current:
            st.warning(scenario_mode_label(True))
        refresh = st.button("Refresh FPL Data", type="primary", width="stretch")
    settings = current_settings()
    if refresh:
        with st.status("Refreshing FPL data…", expanded=True) as status:
            st.write("Fetching official bootstrap and fixtures…")
            st.write("Building live features and scoring players…")
            if settings.squad_source == "manual_file":
                st.write("Optimizing the personalized squad when a valid snapshot is available…")
            else:
                st.write("Importing the latest public post-deadline squad for inspection…")
            result = run_pipeline_refresh(settings)
            if result.success:
                st.cache_data.clear()
                st.session_state.refresh_generation = st.session_state.get("refresh_generation", 0) + 1
                st.session_state.last_refresh_message = f"{result.message} {result.completed_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
                status.update(label="Refresh complete", state="complete")
            else:
                st.session_state.last_refresh_error = result.message
                status.update(label="Refresh failed — cached data retained", state="error")
    if st.session_state.get("last_refresh_message"):
        st.sidebar.success(st.session_state.last_refresh_message)
    if st.session_state.get("last_refresh_error"):
        st.sidebar.error(st.session_state.last_refresh_error)
    bundle = cached_bundle(settings, st.session_state.get("refresh_generation", 0))
    return settings, bundle


def render_data_status(bundle: DashboardBundle) -> None:
    if not bundle.status.available:
        st.error(bundle.status.message)
        return
    if bundle.status.stale:
        st.warning(bundle.status.message)
    timestamp = bundle.status.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if bundle.status.timestamp else "Unknown"
    schema = bundle.live_summary.get("schema_validation", {})
    required, available = schema.get("required", "?"), schema.get("available", "?")
    passed = schema.get("passed", False)
    st.caption(f"Data: {timestamp} · Schema: {available}/{required} {'PASS' if passed else 'FAIL'} · Squad: {bundle.decision_summary.get('squad_source', 'unknown')}")


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
                st.caption(f"{row.team} · {money(row.price)}")
                st.write(f"{points(row.availability_adjusted_xpts)} xPts")
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
    if bundle.predictions.empty:
        st.error("No live predictions are available. Use Refresh FPL Data and check model artifacts/schema.")
        return False
    schema = bundle.live_summary.get("schema_validation", {})
    if schema and not schema.get("passed", False):
        st.error("Live/training feature schema validation failed. Prediction display has been stopped.")
        return False
    return True
