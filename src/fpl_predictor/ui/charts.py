"""Small Plotly chart builders used across Streamlit pages."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def ranking_bar(frame: pd.DataFrame, metric: str, top_n: int = 20):
    ranked = frame.nlargest(top_n, metric).sort_values(metric)
    return px.bar(ranked, x=metric, y="player", color="position", orientation="h",
                  hover_data=[column for column in ("team", "price", "availability") if column in ranked],
                  labels={metric: metric.replace("_", " ").title(), "player": "Player"})


def captaincy_scatter(frame: pd.DataFrame):
    chart = frame.copy()
    chart["chart_minutes"] = pd.to_numeric(chart["expected_minutes_proxy"], errors="coerce").fillna(0).clip(lower=0)
    return px.scatter(chart, x="availability_adjusted_xpts", y="ceiling_score",
                      size="chart_minutes", color="position", hover_name="player",
                      hover_data=[column for column in ("team", "minutes_confidence", "uncertainty_width") if column in frame],
                      labels={"availability_adjusted_xpts": "Mean xPts", "ceiling_score": "Ceiling score",
                              "chart_minutes": "Expected minutes"})


def projection_scatter(frame: pd.DataFrame):
    chart = frame.copy()
    chart["chart_minutes"] = pd.to_numeric(chart["expected_minutes_proxy"], errors="coerce").fillna(0).clip(lower=0)
    return px.scatter(chart, x="weighted_xpts_3", y="weighted_xpts_5", color="position",
                      size="chart_minutes", hover_name="player", hover_data=["team", "price"],
                      labels={"weighted_xpts_3": "3-GW weighted xPts", "weighted_xpts_5": "5-GW weighted xPts",
                              "chart_minutes": "Expected minutes"})


def fixture_run_chart(summary: pd.DataFrame, horizon: int):
    metric = f"avg_fdr_{horizon}"
    ordered = summary.sort_values(metric, ascending=False)
    return px.bar(ordered, x=metric, y="team_name", orientation="h",
                  labels={metric: f"Average official FDR ({horizon} GW)", "team_name": "Team"},
                  color=metric, color_continuous_scale="RdYlGn_r")


def model_comparison_chart(results: pd.DataFrame):
    required = {"split", "model", "mae", "spearman", "top_25_precision", "ndcg_25"}
    if results.empty or not required.issubset(results.columns):
        return go.Figure()
    filtered = results[(results.split.eq("validation")) & results.model.isin(["Ridge", "Random Forest", "HistGradientBoosting"])]
    if filtered.empty:
        return go.Figure()
    best = filtered.sort_values("mae").groupby("model", as_index=False).first()
    melted = best.melt(id_vars="model", value_vars=["mae", "spearman", "top_25_precision", "ndcg_25"],
                       var_name="metric", value_name="value")
    return px.bar(melted, x="model", y="value", color="metric", barmode="group")


def calibration_chart(calibration: pd.DataFrame):
    required = {"model", "mean_predicted", "mean_actual", "split", "sample_count", "prediction_bin"}
    if calibration.empty or not required.issubset(calibration.columns):
        return go.Figure()
    selected = calibration[calibration.model.str.contains("Ridge", na=False)]
    figure = px.scatter(selected, x="mean_predicted", y="mean_actual", color="split",
                        size="sample_count", hover_name="prediction_bin",
                        labels={"mean_predicted": "Predicted xPts", "mean_actual": "Actual points"})
    if not selected.empty:
        low = min(selected.mean_predicted.min(), selected.mean_actual.min())
        high = max(selected.mean_predicted.max(), selected.mean_actual.max())
        figure.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"dash": "dash"})
    return figure


def fixture_run_summary(fixtures: pd.DataFrame, target_gw: int, horizons=(3, 5)) -> pd.DataFrame:
    teams = pd.DataFrame({"team_name": sorted(fixtures.team_name.dropna().unique())})
    for horizon in horizons:
        selected = fixtures[fixtures.gw.between(target_gw, target_gw + horizon - 1)].copy()
        average = selected.groupby("team_name").difficulty.mean().rename(f"avg_fdr_{horizon}")
        count = selected.groupby("team_name").size().rename(f"fixture_count_{horizon}")
        teams = teams.merge(pd.concat([average, count], axis=1), on="team_name", how="left")
    return teams
