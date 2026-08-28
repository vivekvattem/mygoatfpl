"""Streamlit entrypoint for the read-only FPL decision dashboard."""

import streamlit as st

from fpl_predictor.ui.components import (
    configure_page, render_data_status, render_downloads, render_kpis,
    render_pitch, render_sidebar, require_predictions,
)
from fpl_predictor.ui.data import dashboard_summary

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
