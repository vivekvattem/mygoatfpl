"""Explicit, read-only Streamlit boundary for the validated Phase 6 optimizer."""

from pathlib import Path
import sys

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar
from fpl_predictor.ui.data import run_transfer_analysis, transfer_cache_key, transfer_readiness
from fpl_predictor.ui.formatting import (
    decision_status_label, prepare_one_transfer_table, prepare_replacement_table,
    prepare_two_transfer_table, transfer_status_badge,
)


def _run_or_reuse(settings, cache_key: str) -> None:
    results = st.session_state.setdefault("transfer_analysis_results", {})
    if cache_key in results:
        st.info(f"Using the successful analysis cached for these exact inputs ({results[cache_key]}).")
        return
    with st.status("Running transfer analysis…", expanded=True) as status:
        st.write("Using the existing Phase 6 optimizer with the current squad and live predictions…")
        st.write("Evaluating legal one- and two-transfer paths…")
        result = run_transfer_analysis(settings)
        if result.success:
            results[cache_key] = result.completed_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            # Only change the dashboard bundle generation. The sidebar's live-data
            # cache is not broadly invalidated by a transfer-only scenario run.
            st.session_state.refresh_generation = st.session_state.get("refresh_generation", 0) + 1
            status.update(label="Transfer analysis complete", state="complete")
            st.rerun()
        else:
            status.update(label="Transfer analysis failed", state="error")
            st.error(f"Transfer analysis could not complete: {result.message}")


configure_page("Transfers")
settings, bundle = render_sidebar()
st.title("Transfer Analysis")
render_data_status(bundle)

readiness = transfer_readiness(settings, bundle.squad)
if not readiness.ready:
    st.markdown("### ⚪ GREY — INSUFFICIENT FINANCIAL STATE" if readiness.code == "financial_unknown"
                else "### 🟥 RED — TRANSFER ANALYSIS UNAVAILABLE")
    st.subheader(readiness.heading)
    st.info(readiness.message)
    st.caption("The app never submits transfers. Financial inputs are never guessed.")
    st.stop()

if settings.assume_selling_price_current:
    st.warning("SCENARIO MODE — current price is temporarily treated as selling price. All transfer outputs below are estimates.")

cache_key = transfer_cache_key(settings)
if st.button("Run Transfer Analysis", type="primary"):
    _run_or_reuse(settings, cache_key)

summary = bundle.decision_summary
best = summary.get("best_transfer") or {}
net_column = f"net_gain_{settings.horizon}gw"
best_gain = best.get(net_column)
badge = transfer_status_badge(summary.get("transfer_decision"), best_gain, settings.minimum_gain)
st.markdown(f"### {badge}")

if not summary:
    st.info("Transfer analysis is ready. Click Run Transfer Analysis to create results for the current inputs.")
    st.stop()

decision = decision_status_label(summary.get("transfer_decision"))
metric_columns = st.columns(5)
metric_columns[0].metric("Decision", decision)
metric_columns[1].metric("Best Move", f"{best.get('out', '—')} → {best.get('in', '—')}")
metric_columns[2].metric("Expected Gain", "—" if best_gain is None else f"{best_gain:+.2f}")
metric_columns[3].metric("Hit Cost", "—" if not best else f"{best.get('hit_cost', 0):.0f}")
metric_columns[4].metric("Planning Horizon", f"{settings.horizon} GW")
st.caption(f"Configured transfer threshold: {settings.minimum_gain:.2f} xPts. "
           f"Risk profile: {settings.risk_profile.title()}.")
if summary.get("transfer_note"):
    st.warning(summary["transfer_note"])

st.subheader("Best one-transfer moves")
if bundle.one_transfers.empty:
    st.info("No legal one-transfer candidates were returned for these inputs.")
else:
    st.dataframe(prepare_one_transfer_table(bundle.one_transfers.head(10), settings.horizon, settings.minimum_gain),
                 width="stretch", hide_index=True)

st.subheader("Best two-transfer paths")
if bundle.two_transfers.empty:
    st.info("No legal two-transfer paths were returned for these inputs.")
else:
    st.dataframe(prepare_two_transfer_table(bundle.two_transfers.head(10), settings.horizon, settings.minimum_gain),
                 width="stretch", hide_index=True)

st.subheader("Replacement shortlist")
if bundle.replacements.empty:
    st.info("No replacement shortlist was returned for these inputs.")
else:
    player = st.selectbox("Replace", sorted(bundle.replacements.out.dropna().unique()))
    selected = bundle.replacements[bundle.replacements.out.eq(player)]
    st.dataframe(prepare_replacement_table(selected, bundle.predictions, settings.horizon, settings.minimum_gain),
                 width="stretch", hide_index=True)
st.caption("Signals are transparent: GREEN meets the configured threshold; YELLOW is positive but below it; RED is non-positive.")
