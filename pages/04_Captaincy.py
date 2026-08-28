import pandas as pd
import streamlit as st

from fpl_predictor.captaincy import rank_captains
from fpl_predictor.ui.charts import captaincy_scatter
from fpl_predictor.ui.components import configure_page, render_data_status, render_explain_button, render_sidebar

configure_page("Captaincy")
settings, bundle = render_sidebar()
st.title("Captaincy")
render_data_status(bundle)
st.info("Captaincy uses xPts plus a separate ceiling/minutes heuristic. It is not based on Ridge xPts alone.")
if bundle.optimized_xi.empty:
    st.warning("Run the optimizer to generate captain candidates.")
else:
    xi = bundle.optimized_xi.copy()
    required = {"availability_adjusted_xpts", "ceiling_score", "expected_minutes_proxy"}
    if not required.issubset(xi):
        st.warning("Captaincy inputs are incomplete for this session squad. Refresh personalized outputs when ready.")
        st.stop()
    if "overall_signal" in bundle.predictions:
        signal_columns = [column for column in ["player_id", "overall_signal", "minutes_signal", "availability_signal"]
                          if column in bundle.predictions]
        xi = xi.merge(bundle.predictions[signal_columns], on="player_id", how="left")
    columns = st.columns(3)
    all_candidates = []
    for column, profile in zip(columns, ("safe", "balanced", "aggressive")):
        result = rank_captains(xi, profile)
        all_candidates.append(result.candidates.assign(profile=profile))
        with column.container(border=True):
            st.subheader(profile.title())
            st.write(f"Captain: **{result.captain.player}**")
            st.write(f"Vice: **{result.vice_captain.player}**")
            st.metric("Captaincy score", f"{result.captain.captaincy_score:.1f}")
            st.caption(f"xPts {result.captain.availability_adjusted_xpts:.2f} · Ceiling {result.captain.ceiling_score:.1f} · Minutes {result.captain.expected_minutes_proxy:.0f}")
            if "avg_fixture_difficulty" in result.captain.index:
                st.caption(f"Fixture FDR: {result.captain.avg_fixture_difficulty:.1f}")
            st.caption(f"Captain risk: {result.captain.get('overall_signal', 'GREY')} · "
                       f"Minutes {result.captain.get('minutes_signal', 'GREY')} · "
                       f"Availability {result.captain.get('availability_signal', 'GREY')}")
    candidates = pd.concat(all_candidates).drop_duplicates("player_id")
    st.plotly_chart(captaincy_scatter(candidates), width="stretch")
    show = [column for column in ["player", "team", "position", "availability_adjusted_xpts", "ceiling_score",
            "expected_minutes_proxy", "uncertainty_width", "minutes_confidence",
            "avg_fixture_difficulty", "overall_signal", "minutes_signal", "availability_signal"] if column in candidates]
    st.dataframe(candidates[show], width="stretch", hide_index=True)
    render_explain_button("Who should I captain?", bundle, settings, "explain_captaincy")
