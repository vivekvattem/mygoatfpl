import pandas as pd
import streamlit as st

from fpl_predictor.captaincy import rank_captains
from fpl_predictor.ui.charts import captaincy_scatter
from fpl_predictor.ui.components import configure_page, render_data_status, render_sidebar

configure_page("Captaincy")
settings, bundle = render_sidebar()
st.title("Captaincy")
render_data_status(bundle)
st.info("Captaincy uses xPts plus a separate ceiling/minutes heuristic. It is not based on Ridge xPts alone.")
if bundle.optimized_xi.empty:
    st.warning("Run the optimizer to generate captain candidates.")
else:
    columns = st.columns(3)
    all_candidates = []
    for column, profile in zip(columns, ("safe", "balanced", "aggressive")):
        result = rank_captains(bundle.optimized_xi, profile)
        all_candidates.append(result.candidates.assign(profile=profile))
        with column.container(border=True):
            st.subheader(profile.title())
            st.write(f"Captain: **{result.captain.player}**")
            st.write(f"Vice: **{result.vice_captain.player}**")
            st.metric("Captaincy score", f"{result.captain.captaincy_score:.1f}")
            st.caption(f"xPts {result.captain.availability_adjusted_xpts:.2f} · Ceiling {result.captain.ceiling_score:.1f} · Minutes {result.captain.expected_minutes_proxy:.0f}")
            if "avg_fixture_difficulty" in result.captain.index:
                st.caption(f"Fixture FDR: {result.captain.avg_fixture_difficulty:.1f}")
    candidates = pd.concat(all_candidates).drop_duplicates("player_id")
    st.plotly_chart(captaincy_scatter(candidates), width="stretch")
    show = [column for column in ["player", "team", "position", "availability_adjusted_xpts", "ceiling_score",
            "expected_minutes_proxy", "uncertainty_width", "minutes_confidence",
            "avg_fixture_difficulty"] if column in candidates]
    st.dataframe(candidates[show], width="stretch", hide_index=True)
