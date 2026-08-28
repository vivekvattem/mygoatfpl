import pandas as pd
import streamlit as st

from fpl_predictor.ui.charts import calibration_chart, model_comparison_chart
from fpl_predictor.ui.components import configure_page, render_sidebar

configure_page("Model Performance")
settings, bundle = render_sidebar()
st.title("Model Performance")
selected = bundle.phase4_summary.get("selected_model", {})
metrics = bundle.phase4_summary.get("test_metrics", {})
st.write(f"Production model: **Position-specific {selected.get('model_name', 'Unknown')}**")
st.write(f"Feature set: **{selected.get('feature_set', 'Unknown')}**")
cards = st.columns(7)
for card, (label, key) in zip(cards, [("MAE", "mae"), ("RMSE", "rmse"), ("Spearman", "spearman"),
                                              ("Top-10", "top_10_precision"), ("Top-25", "top_25_precision"),
                                              ("Top-50", "top_50_precision"), ("NDCG@25", "ndcg_25")]):
    value = metrics.get(key)
    card.metric(label, "—" if value is None else f"{value:.3f}")
st.subheader("Model comparison")
st.plotly_chart(model_comparison_chart(bundle.model_results), width="stretch")
st.subheader("Calibration")
st.plotly_chart(calibration_chart(bundle.calibration), width="stretch")
st.warning("The current production model performs reasonably for expected-points ranking but significantly compresses rare high-scoring outcomes.")
st.subheader("Residual diagnostics")
if bundle.residuals.empty:
    st.info("Residual analysis artifact is unavailable.")
else:
    for dimension in ("position", "price_band", "season_stage", "fixture_difficulty"):
        subset = bundle.residuals[bundle.residuals.dimension.eq(dimension)]
        if not subset.empty:
            with st.expander(dimension.replace("_", " ").title()):
                st.dataframe(subset, width="stretch", hide_index=True)
