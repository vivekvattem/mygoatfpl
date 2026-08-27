"""Transparent empirical residual uncertainty bands."""

import numpy as np
import pandas as pd


def fit_residual_bands(oof: pd.DataFrame, lower_quantile: float = .1,
                       upper_quantile: float = .9) -> pd.DataFrame:
    """Estimate position-specific residual quantiles from OOF predictions."""
    frame = oof.copy()
    frame["residual"] = frame.actual_points - frame.predicted_points
    return frame.groupby("position").residual.quantile([lower_quantile, upper_quantile]).unstack().rename(
        columns={lower_quantile: "residual_lower", upper_quantile: "residual_upper"}
    ).reset_index()


def add_uncertainty_bounds(predictions: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Add empirical ranges; these are not probabilistic guarantees."""
    result = predictions.merge(bands, on="position", how="left")
    result["lower_bound"] = (result.predicted_points + result.residual_lower).clip(lower=0)
    result["upper_bound"] = result.predicted_points + result.residual_upper
    return result
