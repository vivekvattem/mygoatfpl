"""Forward Gameweek projections using frozen known form and future fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd

from .live_features import fixture_context
from .live_inference import predict_live_players
from .loaders import load_teams

HORIZON_WEIGHTS = {
    1: (1.0,),
    3: (1.0, 0.9, 0.8),
    5: (1.0, 0.9, 0.8, 0.7, 0.6),
}


def weighted_projection(values: list[float] | pd.Series, horizon: int) -> float:
    if horizon not in HORIZON_WEIGHTS:
        raise ValueError("Supported horizons are 1, 3, and 5 Gameweeks")
    points = np.asarray(values, dtype=float)[:horizon]
    if len(points) < horizon:
        raise ValueError(f"Horizon {horizon} requires {horizon} Gameweek projections")
    return float(np.dot(points, np.asarray(HORIZON_WEIGHTS[horizon])))


def _set_future_fixture_context(features: pd.DataFrame, context: pd.DataFrame,
                                target_gw: int) -> pd.DataFrame:
    future = features.copy()
    if "is_blank" not in future:
        future["is_blank"] = 0.0
    future["is_blank"] = pd.to_numeric(future["is_blank"], errors="coerce").astype(float)
    if "is_home" not in future:
        future["is_home"] = np.nan
    future["is_home"] = future["is_home"].map(
        lambda value: np.nan if pd.isna(value) else float(bool(value))
    ).astype(float)
    future["gw"] = target_gw
    indexed = context[context.gw.eq(target_gw)].set_index("team_name") if not context.empty else pd.DataFrame()
    for index, row in future.iterrows():
        team = row["team"]
        if not indexed.empty and team in indexed.index:
            ctx = indexed.loc[team]
            future.at[index, "fixture_count"] = int(ctx.fixture_count)
            future.at[index, "home_fixture_count"] = int(ctx.home_fixture_count)
            future.at[index, "away_fixture_count"] = int(ctx.away_fixture_count)
            future.at[index, "avg_fixture_difficulty"] = ctx.avg_fixture_difficulty
            future.at[index, "min_fixture_difficulty"] = ctx.min_fixture_difficulty
            future.at[index, "max_fixture_difficulty"] = ctx.max_fixture_difficulty
            future.at[index, "opponent"] = ctx.opponent
            future.at[index, "is_blank"] = 0.0
            future.at[index, "is_home"] = float(bool(ctx.home_fixture_count)) if not ctx.away_fixture_count else np.nan
        else:
            future.at[index, "fixture_count"] = 0
            future.at[index, "home_fixture_count"] = 0
            future.at[index, "away_fixture_count"] = 0
            future.at[index, "avg_fixture_difficulty"] = np.nan
            future.at[index, "min_fixture_difficulty"] = np.nan
            future.at[index, "max_fixture_difficulty"] = np.nan
            future.at[index, "opponent"] = np.nan
            future.at[index, "is_blank"] = 1.0
            future.at[index, "is_home"] = np.nan
    return future


def build_multi_gw_projections(base_features: pd.DataFrame, bootstrap: dict,
                               fixtures: list[dict], artifacts,
                               uncertainty_summary_path: Path, target_gw: int,
                               horizon: int = 5) -> pd.DataFrame:
    """Apply the same model with fixed known form and each future fixture row."""
    if horizon < 1 or horizon > 5:
        raise ValueError("Projection horizon must be between 1 and 5")
    context = fixture_context(fixtures, load_teams(bootstrap))
    result = base_features[["player_id", "player", "team", "position", "price"]].copy()
    for offset in range(1, horizon + 1):
        future_features = _set_future_fixture_context(base_features, context, target_gw + offset - 1)
        projected, _ = predict_live_players(future_features, artifacts, uncertainty_summary_path)
        blank = projected.fixture_count.eq(0)
        projected.loc[blank, ["raw_xpts", "display_xpts", "availability_adjusted_xpts", "xpts_lower", "xpts_upper"]] = 0.0
        result[f"raw_xpts_gw{offset}"] = projected.raw_xpts.to_numpy()
        result[f"xpts_gw{offset}"] = projected.availability_adjusted_xpts.to_numpy()
        result[f"fixture_count_gw{offset}"] = projected.fixture_count.to_numpy()
    for supported in (1, 3, 5):
        if supported <= horizon:
            point_columns = [f"xpts_gw{i}" for i in range(1, supported + 1)]
            weights = np.asarray(HORIZON_WEIGHTS[supported])
            result[f"total_xpts_{supported}"] = result[point_columns].sum(axis=1)
            result[f"weighted_xpts_{supported}"] = result[point_columns].mul(weights, axis=1).sum(axis=1)
    return result
