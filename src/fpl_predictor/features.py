"""Transparent baseline player feature engineering."""

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90 metrics; players with zero minutes receive NaN values."""
    result = df.copy()
    metrics = {
        "goals_per90": "goals_scored",
        "assists_per90": "assists",
        "xG_per90": "expected_goals",
        "xA_per90": "expected_assists",
        "xGI_per90": "expected_goal_involvements",
        "threat_per90": "threat",
        "creativity_per90": "creativity",
        "points_per90": "total_points",
    }
    minutes = pd.to_numeric(result.get("minutes"), errors="coerce")
    for output, source in metrics.items():
        values = result[source] if source in result else pd.Series(np.nan, index=result.index)
        result[output] = _safe_ratio(values, minutes) * 90.0
    return result


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple output-per-price features using prices in millions."""
    result = df.copy()
    price = result["price"] if "price" in result else pd.Series(np.nan, index=result.index)
    total_points = result.get("total_points", pd.Series(np.nan, index=result.index))
    xgi = result.get("xGI_per90", pd.Series(np.nan, index=result.index))
    result["points_per_million"] = _safe_ratio(total_points, price)
    result["xGI_per_million"] = _safe_ratio(xgi, price)
    return result


def add_basic_attacking_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent baseline heuristic, not an expected-points model."""
    result = df.copy()
    required = ("xG_per90", "xA_per90", "threat_per90", "creativity_per90")
    values = {
        name: pd.to_numeric(result.get(name, pd.Series(np.nan, index=result.index)), errors="coerce")
        for name in required
    }
    result["attacking_score"] = (
        4.0 * values["xG_per90"]
        + 3.0 * values["xA_per90"]
        + 0.01 * values["threat_per90"]
        + 0.005 * values["creativity_per90"]
    )
    price = result.get("price", pd.Series(np.nan, index=result.index))
    result["attacking_value_score"] = _safe_ratio(result["attacking_score"], price)
    return result


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Phase 1 feature transformations in dependency order."""
    return add_basic_attacking_score(add_value_features(add_per90_features(df)))
