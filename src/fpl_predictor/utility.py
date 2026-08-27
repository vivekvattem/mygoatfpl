"""Transparent player utility, uncertainty, ceiling, and ownership helpers."""

import numpy as np
import pandas as pd


def risk_adjusted_utility(mean_xpts: pd.Series, lower: pd.Series, upper: pd.Series,
                          risk_lambda: float = 0.10) -> pd.Series:
    if risk_lambda < 0:
        raise ValueError("risk_lambda must be non-negative")
    width = pd.to_numeric(upper, errors="coerce") - pd.to_numeric(lower, errors="coerce")
    return pd.to_numeric(mean_xpts, errors="coerce") - risk_lambda * width.fillna(0)


def ceiling_score(frame: pd.DataFrame) -> pd.Series:
    """Heuristic upside score; it is not a calibrated haul probability."""
    def percentile(column: str, invert: bool = False) -> pd.Series:
        values = pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
        ranked = values.rank(pct=True).fillna(0.5)
        return 1 - ranked if invert else ranked
    score = (0.40 * percentile("availability_adjusted_xpts") +
             0.25 * percentile("xGI_per90_last_3") +
             0.15 * percentile("ict_index_last_3") +
             0.10 * percentile("expected_minutes_proxy") +
             0.10 * percentile("avg_fixture_difficulty", invert=True))
    return (100 * score).clip(0, 100)


def ownership_label(selected_by_percent: pd.Series, template: float = 20,
                    differential: float = 10) -> pd.Series:
    ownership = pd.to_numeric(selected_by_percent, errors="coerce")
    return pd.Series(np.select([ownership.ge(template), ownership.le(differential)],
                               ["template", "differential"], default="neutral"), index=ownership.index)


def add_player_utilities(frame: pd.DataFrame, horizon: int = 1,
                         risk_lambda: float = 0.10) -> pd.DataFrame:
    result = frame.copy()
    planning = f"weighted_xpts_{horizon}"
    if planning not in result:
        planning = "availability_adjusted_xpts"
    result["mean_xpts"] = pd.to_numeric(result["availability_adjusted_xpts"], errors="coerce")
    result["planning_xpts"] = pd.to_numeric(result[planning], errors="coerce")
    result["uncertainty_width"] = pd.to_numeric(result["xpts_upper"], errors="coerce") - pd.to_numeric(result["xpts_lower"], errors="coerce")
    result["risk_adjusted_utility"] = result["planning_xpts"] - risk_lambda * result["uncertainty_width"].fillna(0)
    result["ceiling_score"] = ceiling_score(result)
    result["ceiling_utility"] = result["planning_xpts"] + result["ceiling_score"] / 100
    result["value"] = result["planning_xpts"] / pd.to_numeric(result["price"], errors="coerce").replace(0, np.nan)
    if "selected_by_percent" in result:
        result["ownership_label"] = ownership_label(result["selected_by_percent"])
    return result
