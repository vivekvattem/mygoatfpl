import pandas as pd

from fpl_predictor.uncertainty import add_uncertainty_bounds, fit_residual_bands


def test_residual_quantiles_and_prediction_intervals():
    oof = pd.DataFrame({"position": ["MID"] * 5, "actual_points": [0, 2, 4, 6, 8],
                        "predicted_points": [2, 2, 2, 2, 2]})
    bands = fit_residual_bands(oof, 0, 1)
    result = add_uncertainty_bounds(pd.DataFrame({"position": ["MID"], "predicted_points": [3]}), bands)
    assert result.lower_bound.iloc[0] == 1
    assert result.upper_bound.iloc[0] == 9
