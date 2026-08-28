import streamlit as st

from fpl_predictor.ui.charts import fixture_run_chart, fixture_run_summary
from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar
from fpl_predictor.ui.data import dashboard_summary

configure_page("Fixtures")
settings, bundle = render_sidebar()
st.title("Fixture Outlook")
render_data_status(bundle)
target_gw = dashboard_summary(bundle)["target_gw"]
if bundle.fixtures.empty or target_gw is None:
    st.warning("Fixture data is unavailable. Refresh official FPL data.")
else:
    horizon = st.radio("Fixture window", [3, 5], horizontal=True)
    future = bundle.fixtures[bundle.fixtures.gw.between(target_gw, target_gw + horizon - 1)].copy()
    columns = ["gw", "team_name", "opponent_name", "venue", "difficulty", "model_opponent_strength"]
    st.dataframe(future[[column for column in columns if column in future]], width="stretch", hide_index=True)
    rankings = fixture_run_summary(bundle.fixtures, target_gw)
    st.plotly_chart(fixture_run_chart(rankings, horizon), width="stretch")
    metric = f"avg_fdr_{horizon}"
    left, right = st.columns(2)
    left.subheader(f"Best next {horizon} GW runs")
    left.dataframe(rankings.nsmallest(5, metric)[["team_name", metric]], hide_index=True, width="stretch")
    right.subheader(f"Worst next {horizon} GW runs")
    right.dataframe(rankings.nlargest(5, metric)[["team_name", metric]], hide_index=True, width="stretch")
    st.caption("Official FDR is always shown numerically. Model opponent strength is based only on completed fixtures known now.")
    st.subheader("Team fixture signals")
    if not bundle.team_fixture_signals.empty:
        st.dataframe(bundle.team_fixture_signals, width="stretch", hide_index=True)
