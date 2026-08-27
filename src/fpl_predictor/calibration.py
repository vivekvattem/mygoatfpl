"""Tabular expected-points calibration diagnostics."""

import numpy as np
import pandas as pd

CALIBRATION_BINS = [-np.inf, 2, 4, 6, 8, np.inf]
CALIBRATION_LABELS = ["<2", "2-4", "4-6", "6-8", "8+"]


def calibration_table(frame: pd.DataFrame, prediction_column: str = "predicted_points") -> pd.DataFrame:
    """Compare mean prediction and outcome within fixed xPts bands."""
    paired = frame.dropna(subset=[prediction_column, "actual_points"]).copy()
    paired["prediction_bin"] = pd.cut(
        paired[prediction_column], CALIBRATION_BINS, labels=CALIBRATION_LABELS, right=False
    )
    result = paired.groupby("prediction_bin", observed=False).agg(
        mean_predicted=(prediction_column, "mean"), mean_actual=("actual_points", "mean"),
        sample_count=("actual_points", "size"),
    ).reset_index()
    result["calibration_error"] = result.mean_actual - result.mean_predicted
    return result
