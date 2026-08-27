import pandas as pd

from fpl_predictor.calibration import calibration_table
from fpl_predictor.modeling import residual_analysis


def test_calibration_bins_and_error():
    frame = pd.DataFrame({"predicted_points": [1, 3, 5, 7, 9], "actual_points": [2, 2, 6, 6, 10]})
    table = calibration_table(frame)
    assert table.prediction_bin.astype(str).tolist() == ["<2", "2-4", "4-6", "6-8", "8+"]
    assert table.sample_count.tolist() == [1, 1, 1, 1, 1]
    assert table.iloc[0].calibration_error == 1


def test_residual_analysis_contains_requested_segments():
    frame = pd.DataFrame({"actual_points": [5, 1], "predicted_points": [3, 2],
                          "price": [10, 5], "minutes_last_5": [450, 60],
                          "is_home": [1.0, 0.0], "avg_fixture_difficulty": [2, 4],
                          "gw": [3, 12], "position": ["FWD", "DEF"]})
    result = residual_analysis(frame, "Ridge", "test")
    assert {"position", "price_band", "minutes_history_band", "home_away",
            "fixture_band", "prediction_band", "gw", "season_stage"}.issubset(result.dimension)
