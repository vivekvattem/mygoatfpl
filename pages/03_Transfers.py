import streamlit as st

from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar
from fpl_predictor.ui.formatting import decision_status_label, scenario_mode_label

configure_page("Transfers")
settings, bundle = render_sidebar()
st.title("Transfer Analysis")
render_data_status(bundle)
decision = decision_status_label(bundle.decision_summary.get("transfer_decision"))
st.metric("Decision", decision)
st.info(scenario_mode_label(bool(bundle.decision_summary.get("financial_state_scenario"))))
best = bundle.decision_summary.get("best_transfer") or {}
if best:
    gain = best.get(f"net_gain_{settings.horizon}gw")
    st.write(f"Best retained legal improvement: **{gain:+.2f} xPts** over {settings.horizon} GWs")
    st.write(f"Configured threshold: **{settings.minimum_gain:.2f} xPts**")
if bundle.decision_summary.get("transfer_note"):
    st.warning(bundle.decision_summary["transfer_note"])

st.subheader("Best one-transfer moves")
if bundle.one_transfers.empty:
    st.info("Transfer candidates are unavailable. Supply bank/free transfers and valid selling prices, or explicitly enable scenario mode.")
else:
    one = bundle.one_transfers.head(10).copy()
    one["Scenario"] = bool(bundle.decision_summary.get("financial_state_scenario"))
    st.dataframe(one, width="stretch", hide_index=True)

st.subheader("Best two-transfer paths")
if bundle.two_transfers.empty:
    st.info("No retained legal two-transfer paths are available.")
else:
    st.dataframe(bundle.two_transfers.head(10), width="stretch", hide_index=True)

st.subheader("Replacement shortlist")
if bundle.replacements.empty:
    st.info("No replacement shortlist is available.")
else:
    player = st.selectbox("Replace", sorted(bundle.replacements.out.unique()))
    selected = bundle.replacements[bundle.replacements.out.eq(player)]
    st.dataframe(selected, width="stretch", hide_index=True)
st.caption("The app never submits transfers. All moves are inspection-only.")
