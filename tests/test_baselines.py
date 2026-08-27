import numpy as np
import pandas as pd

from fpl_predictor.baselines import (
    PositionMeanBaseline, PricePositionBaseline, RidgeBenchmark,
    add_missingness_indicators, minutes_adjusted_prediction,
    previous_gw_prediction, rolling_mean_prediction,
)
from fpl_predictor.feature_registry import load_feature_registry


def test_transparent_baseline_formulas():
    frame = pd.DataFrame({"points_last_1": [4], "points_last_3": [12], "games_last_3": [2],
                          "points_last_5": [15], "games_last_5": [5],
                          "avg_minutes_last_3": [45], "points_per90_last_3": [6]})
    assert previous_gw_prediction(frame).iloc[0] == 4
    assert rolling_mean_prediction(frame, 3).iloc[0] == 6
    assert rolling_mean_prediction(frame, 5).iloc[0] == 3
    assert minutes_adjusted_prediction(frame).iloc[0] == 3


def test_missing_history_remains_missing():
    frame = pd.DataFrame({"points_last_3": [np.nan], "games_last_3": [0]})
    assert pd.isna(rolling_mean_prediction(frame, 3).iloc[0])


def test_position_mean_uses_training_targets_only():
    train = pd.DataFrame({"position": ["MID", "MID", "GK"], "target_points": [2, 4, 8]})
    evaluation = pd.DataFrame({"position": ["MID", "GK"], "target_points": [100, 100]})
    prediction = PositionMeanBaseline().fit(train).predict(evaluation)
    assert prediction.tolist() == [3.0, 8.0]


def test_price_position_and_ridge_handle_missing_values():
    train = pd.DataFrame({"price": [4.5, 5.0, 7.0, 8.0], "position": ["GK", "DEF", "MID", "FWD"],
                          "points_last_3": [1.0, np.nan, 5.0, 6.0], "target_points": [2, 3, 5, 7]})
    assert len(PricePositionBaseline().fit(train).predict(train)) == 4
    registry = load_feature_registry()
    ridge = RidgeBenchmark(["price", "position", "points_last_3"], registry).fit(train)
    assert np.isfinite(ridge.predict(train)).all()


def test_explicit_missingness_indicators():
    frame = pd.DataFrame({"xGI_per90_last_3": [np.nan, .4], "minutes_last_3": [np.nan, 90]})
    result = add_missingness_indicators(frame)
    assert result.xGI_per90_last_3_missing.tolist() == [1, 0]
    assert result.minutes_last_3_missing.tolist() == [1, 0]
