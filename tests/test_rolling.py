import numpy as np
import pandas as pd

from fpl_predictor.rolling import (
    add_rolling_features, add_team_rolling_features, merge_team_and_opponent_features,
)


def player_history(gw4_points=99):
    rows = []
    for gw, points, minutes, xg in [(1, 2, 90, 0.1), (2, 5, 45, 0.2), (3, 10, 0, 0.3), (4, gw4_points, 90, 9.0)]:
        rows.append({
            "season": "2024-25", "gw": gw, "season_player_id": "2024-25_1",
            "price": 5.0, "minutes": minutes, "starts": int(minutes >= 60),
            "total_points": points, "goals_scored": 0, "assists": 0,
            "expected_goals": xg, "expected_assists": xg / 2,
            "expected_goal_involvements": xg * 1.5, "bonus": 0, "bps": points,
            "influence": points, "creativity": points, "threat": points,
            "ict_index": points,
        })
    return pd.DataFrame(rows)


def test_rolling_one_three_five_and_shift_before_roll():
    result = add_rolling_features(player_history())
    gw4 = result[result["gw"] == 4].iloc[0]
    assert gw4["points_last_1"] == 10
    assert gw4["points_last_3"] == 17
    assert gw4["points_last_5"] == 17
    assert gw4["minutes_last_3"] == 135
    assert gw4["xG_last_3"] == 0.6
    assert gw4["xG_per90_last_3"] == 0.4
    assert gw4["start_rate_last_3"] == 1 / 3
    assert result[result["gw"] == 1]["points_last_3"].isna().all()


def test_changing_target_gw_does_not_change_same_gw_features():
    first = add_rolling_features(player_history(20)).query("gw == 4").filter(like="last_")
    second = add_rolling_features(player_history(200)).query("gw == 4").filter(like="last_")
    pd.testing.assert_frame_equal(first, second)


def test_zero_minute_per90_is_nan_and_never_infinite():
    frame = player_history().assign(minutes=0)
    result = add_rolling_features(frame)
    assert result["xG_per90_last_3"].isna().all()
    assert not np.isinf(result.select_dtypes("number").to_numpy()).any()


def test_team_and_opponent_statistics_are_lagged():
    teams = pd.DataFrame([
        {"season": "2024-25", "gw": gw, "team": team, "goals_for": gf,
         "goals_against": ga, "team_points": pts}
        for team, values in {"A": [(1, 0, 3), (2, 1, 3), (3, 2, 1), (9, 9, 0)],
                             "B": [(0, 1, 0), (1, 2, 0), (2, 3, 1), (9, 9, 3)]}.items()
        for gw, (gf, ga, pts) in enumerate(values, 1)
    ])
    rolled = add_team_rolling_features(teams)
    players = pd.DataFrame({"season": ["2024-25"], "gw": [4], "team": ["A"], "opponent": ["B"]})
    result = merge_team_and_opponent_features(players, rolled).iloc[0]
    assert result["team_goals_for_last_3"] == 6
    assert result["team_goals_against_last_3"] == 3
    assert result["opponent_goals_for_last_3"] == 3
    assert result["opponent_goals_against_last_3"] == 6
