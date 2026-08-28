"""Streamlit entrypoint for the read-only FPL decision dashboard."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Streamlit Community Cloud runs from the repository root; adding the local
# src layout explicitly also keeps local launches reliable from paths with spaces.
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fpl_predictor.ui.components import (
    cached_analyst_context, configure_page, first_existing_column, render_analyst_result, render_data_status,
    render_downloads, render_kpis, render_pitch, render_sidebar, require_predictions, risk_summary, signal_badge,
)
from fpl_predictor.ui.data import dashboard_summary
from fpl_predictor.analyst.deterministic import deterministic_answer
from fpl_predictor.analyst.provider import provider_from_config
from fpl_predictor.analyst.service import AnalystService
from fpl_predictor.ui.components import analyst_provider_config

configure_page("Dashboard")
settings, bundle = render_sidebar()

st.title("⚽ FPL AI Predictor")
summary = dashboard_summary(bundle)
st.subheader(f"Gameweek {summary['target_gw'] or '—'} decision dashboard")
st.caption(f"Demo Entry {settings.entry_id} · {summary['squad_source']} · {summary['formation'] or 'No optimized formation'}")
if "last_refresh_message" in st.session_state:
    st.success(st.session_state.pop("last_refresh_message"))
if "last_refresh_error" in st.session_state:
    st.error(st.session_state.pop("last_refresh_error"))
render_data_status(bundle)

if require_predictions(bundle):
    render_kpis(bundle)
    st.subheader("Squad signals")
    if not bundle.squad.empty and "overall_signal" in bundle.squad:
        counts = bundle.squad.overall_signal.value_counts()
        signal_columns = st.columns(4)
        for column, signal in zip(signal_columns, ("GREEN", "YELLOW", "RED", "GREY")):
            column.metric(signal_badge(signal), int(counts.get(signal, 0)))
        risks, opportunities = st.columns(2)
        with risks:
            st.markdown("**Top 3 risks**")
            st.dataframe(risk_summary(bundle.squad), width="stretch", hide_index=True)
            if "weighted_xpts_5" not in bundle.squad:
                st.caption("5-GW projection unavailable; risks are ordered by signal and the best available projection.")
        with opportunities:
            st.markdown("**Top 3 opportunities**")
            owned = (bundle.predictions["owned"].fillna(False) if "owned" in bundle.predictions
                     else pd.Series(False, index=bundle.predictions.index))
            green = (bundle.predictions["overall_signal"].eq("GREEN") if "overall_signal" in bundle.predictions
                     else pd.Series(False, index=bundle.predictions.index))
            options = bundle.predictions[(~owned) & green]
            columns = [column for column in ["player", "team", "overall_signal", "action", "signal_reason"] if column in options]
            projection = first_existing_column(options, ("weighted_xpts_5", "xpts_5gw", "five_gw_xpts",
                                                         "adjusted_xpts", "raw_xpts", "xpts"))
            shown = options.nlargest(3, projection) if projection else options.head(3)
            st.dataframe(shown[columns], width="stretch", hide_index=True)
    st.subheader("Upcoming Alerts")
    if bundle.fixture_calendar.empty:
        st.info("Confirmed fixture alerts are unavailable until fixture data is refreshed.")
    else:
        alerts = bundle.fixture_calendar[bundle.fixture_calendar.schedule_label.ne("NORMAL")]
        if alerts.empty:
            st.success("No confirmed DGW/BGW/TGW alerts in the next 10 Gameweeks.")
        else:
            for (gw, label), group in alerts.groupby(["gw", "schedule_label"]):
                icon = "🔴" if label == "BGW" else "🟡" if label == "TGW" else "🟢"
                st.write(f"{icon} **GW{gw} {label} — CONFIRMED:** {', '.join(group.team.tolist())}")
    st.subheader("AI Weekly Brief")
    weekly_question = "What should I do this week?"
    chip_overrides = {chip: st.session_state.get(f"chip_{chip}", "unknown") for chip in
                      ("wildcard", "free_hit", "bench_boost", "triple_captain")}
    weekly_context = cached_analyst_context(weekly_question, bundle, settings, tuple(chip_overrides.items()),
                                            st.session_state.get("refresh_generation", 0))
    with st.container(border=True):
        st.markdown(deterministic_answer(weekly_context.intent, weekly_context.payload, weekly_context.confidence))
        st.caption("Built from current structured engines; no LLM call was made.")
        provider = provider_from_config(analyst_provider_config())
        if provider.name != "disabled" and st.button("Explain this week", key="dashboard_explain_week"):
            universe = set(bundle.predictions.player.astype(str))
            render_analyst_result(AnalystService(provider).answer_context(
                weekly_question, weekly_context, universe))
    st.divider()
    st.subheader("Projected starting XI")
    render_pitch(bundle.optimized_xi, summary["captain"], summary["vice_captain"])
    st.subheader("Optimized bench")
    if bundle.squad.empty or bundle.optimized_xi.empty:
        st.info("Squad or optimized XI data is unavailable.")
    else:
        bench = bundle.squad[~bundle.squad.player_id.isin(bundle.optimized_xi.player_id)].copy()
        gk = bench[bench.position.eq("GK")]
        outfield = bench[~bench.position.eq("GK")]
        score = f"weighted_xpts_{settings.horizon}"
        if score in outfield:
            outfield = outfield.sort_values(score, ascending=False)
        labels = ([f"GK: {gk.iloc[0].player}"] if not gk.empty else []) + [f"{i}. {name}" for i, name in enumerate(outfield.player, 1)]
        st.write(" · ".join(labels))
    with st.expander("Model and planning limitations", expanded=True):
        st.write("Current xPts estimates are optimized for average expected-points ranking. The Ridge model compresses rare high-scoring outcomes, so captaincy also uses a separate ceiling/minutes heuristic.")
        st.write("Future-GW projections hold current form, availability, and expected minutes approximately fixed. They are planning estimates, not guarantees.")
    st.subheader("Downloads")
    render_downloads(bundle)
