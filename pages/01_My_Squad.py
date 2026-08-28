import pandas as pd
import streamlit as st

from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar, require_predictions
from fpl_predictor.ui.formatting import format_player_table

configure_page("My Squad")
settings, bundle = render_sidebar()
st.title("My Squad")
render_data_status(bundle)
if require_predictions(bundle):
    if bundle.squad.empty:
        st.warning("No manual squad is loaded. Configure or upload one in Settings.")
    else:
        squad = bundle.squad.copy()
        optimized_ids = set(bundle.optimized_xi.player_id) if not bundle.optimized_xi.empty else set()
        multiplier = (pd.to_numeric(squad["multiplier"], errors="coerce").fillna(0)
                      if "multiplier" in squad else pd.Series(0, index=squad.index))
        squad["Current role"] = multiplier.map(lambda value: "Starter" if value > 0 else "Bench")
        squad["Optimized role"] = squad.player_id.map(lambda value: "Starter" if value in optimized_ids else "Bench")
        squad["Changed"] = squad["Current role"].ne(squad["Optimized role"]).map({True: "Yes", False: "No"})
        captain = squad["is_captain"] if "is_captain" in squad else pd.Series(False, index=squad.index)
        vice = squad["is_vice_captain"] if "is_vice_captain" in squad else pd.Series(False, index=squad.index)
        squad["Captain"] = captain.eq(True).map({True: "Yes", False: ""})
        squad["Vice"] = vice.eq(True).map({True: "Yes", False: ""})
        columns = [column for column in ["player", "team", "position", "price", "raw_xpts",
                   "availability_adjusted_xpts", "weighted_xpts_3", "weighted_xpts_5",
                   "expected_minutes_proxy", "availability", "selected_by_percent",
                   "overall_signal", "action", "risk_reason",
                   "Captain", "Vice", "bench_position", "Current role", "Optimized role", "Changed"] if column in squad]
        st.dataframe(format_player_table(squad[columns]), width="stretch", hide_index=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Current XI")
            current = squad[pd.to_numeric(squad.get("multiplier", 0), errors="coerce").fillna(0).gt(0)]
            current_columns = [column for column in ["player", "position", "team", "availability_adjusted_xpts"]
                               if column in current]
            st.dataframe(current[current_columns], width="stretch", hide_index=True)
        with right:
            st.subheader("Optimized XI")
            columns = [column for column in ["player", "position", "team", "availability_adjusted_xpts"] if column in bundle.optimized_xi]
            st.dataframe(bundle.optimized_xi[columns], width="stretch", hide_index=True)
