import numpy as np
import pandas as pd

from fpl_predictor.features import add_basic_attacking_score, add_per90_features, add_value_features


def test_per90_and_zero_minute_behavior():
    df = pd.DataFrame({
        "minutes": [180, 0], "goals_scored": [2, 1], "assists": [4, 1],
        "expected_goals": [1.0, 2.0], "expected_assists": [2.0, 1.0],
        "expected_goal_involvements": [3.0, 3.0], "threat": [100, 5],
        "creativity": [80, 5], "total_points": [20, 1],
    })
    result = add_per90_features(df)
    assert result.loc[0, "goals_per90"] == 1.0
    assert result.loc[0, "xGI_per90"] == 1.5
    assert np.isnan(result.loc[1, "points_per90"])
    assert not np.isinf(result.select_dtypes("number").to_numpy()).any()


def test_value_and_attacking_formula():
    df = pd.DataFrame({
        "total_points": [60], "price": [6.0], "xGI_per90": [0.5],
        "xG_per90": [0.4], "xA_per90": [0.2], "threat_per90": [30.0],
        "creativity_per90": [20.0],
    })
    valued = add_value_features(df)
    scored = add_basic_attacking_score(valued)
    assert valued.loc[0, "points_per_million"] == 10.0
    assert valued.loc[0, "xGI_per_million"] == pytest.approx(0.5 / 6.0)
    assert scored.loc[0, "attacking_score"] == pytest.approx(2.6)
    assert scored.loc[0, "attacking_value_score"] == pytest.approx(2.6 / 6.0)


import pytest
