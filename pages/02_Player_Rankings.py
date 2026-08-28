import pandas as pd
import streamlit as st

from fpl_predictor.ui.charts import projection_scatter, ranking_bar
from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar, require_predictions
from fpl_predictor.ui.formatting import format_player_table

configure_page("Player Rankings")
settings, bundle = render_sidebar()
st.title("Player Rankings")
render_data_status(bundle)
if require_predictions(bundle):
    players = bundle.predictions.copy()
    controls = st.columns(4)
    positions = controls[0].multiselect("Position", sorted(players.position.dropna().unique()), default=[])
    teams = controls[1].multiselect("Team", sorted(players.team.dropna().unique()), default=[])
    ownership_filter = controls[2].selectbox("Squad ownership", ["All", "Owned", "Not owned"])
    top_n = controls[3].slider("Top N", 10, 100, 30, 5)
    price_min, price_max = float(players.price.min()), float(players.price.max())
    selected_price = st.slider("Price range", price_min, price_max, (price_min, price_max), 0.1)
    availability = st.multiselect("Availability", sorted(players.availability.dropna().unique()) if "availability" in players else [])
    min_minutes = st.slider("Minimum expected minutes", 0, 90, 0, 5)
    if positions: players = players[players.position.isin(positions)]
    if teams: players = players[players.team.isin(teams)]
    if ownership_filter != "All" and "owned" in players:
        players = players[players.owned.eq(ownership_filter == "Owned")]
    if availability and "availability" in players: players = players[players.availability.isin(availability)]
    players = players[players.price.between(*selected_price)]
    if "expected_minutes_proxy" in players:
        players = players[pd.to_numeric(players.expected_minutes_proxy, errors="coerce").fillna(0).ge(min_minutes)]
    metrics = {"Next GW xPts": "availability_adjusted_xpts", "3-GW weighted xPts": "weighted_xpts_3",
               "5-GW weighted xPts": "weighted_xpts_5", "Value": "value", "Ceiling score": "ceiling_score",
               "Risk-adjusted utility": "risk_adjusted_utility", "Ownership": "selected_by_percent"}
    available_metrics = {label: column for label, column in metrics.items() if column in players}
    label = st.selectbox("Rank by", list(available_metrics), index=min(2, len(available_metrics) - 1))
    metric = available_metrics[label]
    ranked = players.nlargest(top_n, metric)
    st.plotly_chart(ranking_bar(players, metric, min(top_n, 30)), width="stretch")
    columns = [column for column in ["player", "team", "position", "price", "availability_adjusted_xpts",
               "weighted_xpts_3", "weighted_xpts_5", "expected_minutes_proxy", "ceiling_score",
               "uncertainty_width", "selected_by_percent", "availability"] if column in ranked]
    st.dataframe(format_player_table(ranked[columns]), width="stretch", hide_index=True)
    if {"weighted_xpts_3", "weighted_xpts_5", "expected_minutes_proxy"}.issubset(players):
        st.subheader("3-GW vs 5-GW outlook")
        st.plotly_chart(projection_scatter(players), width="stretch")
