import pandas as pd
import plotly.express as px
import streamlit as st

from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar, require_predictions
from fpl_predictor.ui.formatting import format_player_table

configure_page("Player Comparison")
settings, bundle = render_sidebar()
st.title("Player Comparison")
render_data_status(bundle)
if require_predictions(bundle):
    players = bundle.predictions.sort_values("player")
    names = players.player.tolist()
    left, right = st.columns(2)
    player_a = left.selectbox("Player A", names, index=0)
    player_b = right.selectbox("Player B", names, index=min(1, len(names) - 1))
    selected = players[players.player.isin([player_a, player_b])]
    columns = [column for column in ["player", "team", "position", "price", "availability_adjusted_xpts",
               "weighted_xpts_3", "weighted_xpts_5", "xG_last_3", "xA_last_3", "xGI_last_3",
               "expected_minutes_proxy", "avg_fixture_difficulty", "ceiling_score", "risk_adjusted_utility",
               "selected_by_percent", "availability"] if column in selected]
    columns += [column for column in ["overall_signal", "action", "signal_reason", "risk_reason"]
                if column in selected and column not in columns]
    st.dataframe(format_player_table(selected[columns]), width="stretch", hide_index=True)
    metrics = [column for column in ["availability_adjusted_xpts", "weighted_xpts_3", "weighted_xpts_5",
               "xGI_last_3", "ceiling_score", "risk_adjusted_utility"] if column in selected]
    chart = selected[["player", *metrics]].melt(id_vars="player", var_name="Metric", value_name="Value")
    st.plotly_chart(px.bar(chart, x="Metric", y="Value", color="player", barmode="group"), width="stretch")
