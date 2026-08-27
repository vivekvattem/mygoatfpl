import pandas as pd

from fpl_predictor.team_strength import calculate_team_strength, join_team_strength


def matches(gw3_a_goals=9):
    rows = []
    values = [(1, "A", "B", 1, 0), (1, "B", "A", 0, 1),
              (2, "A", "C", 3, 1), (2, "C", "A", 1, 3),
              (2, "B", "C", 2, 4), (2, "C", "B", 4, 2),
              (3, "A", "B", gw3_a_goals, 0), (3, "B", "A", 0, gw3_a_goals)]
    for fixture_id, (gw, team, opponent, gf, ga) in enumerate(values, 1):
        rows.append({"season": "2024-25", "gw": gw, "fixture_id": fixture_id,
                     "team": team, "opponent": opponent, "is_home": fixture_id % 2 == 1,
                     "goals_for": gf, "goals_against": ga})
    return pd.DataFrame(rows)


def test_team_strength_is_shifted_before_target_gw():
    first = calculate_team_strength(matches(9))
    changed = calculate_team_strength(matches(99))
    columns = ["team_attack_strength_3", "team_defense_strength_3"]
    a_gw3 = first.query("team == 'A' and gw == 3")[columns].reset_index(drop=True)
    changed_a_gw3 = changed.query("team == 'A' and gw == 3")[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_gw3, changed_a_gw3)
    assert a_gw3.iloc[0, 0] == 2.0


def test_opponent_join_aggregates_every_double_fixture():
    ratings = calculate_team_strength(matches())
    players = pd.DataFrame({"season": ["2024-25"], "gw": [3], "team": ["A"], "opponent": ["B|C"]})
    result = join_team_strength(players, ratings).iloc[0]
    b = ratings.query("team == 'B' and gw == 3").iloc[0].team_defense_strength_5
    c = ratings.query("team == 'C' and gw == 3").iloc[0].team_defense_strength_5
    assert result.opponent_defense_strength_5_mean == (b + c) / 2
    assert result.opponent_defense_strength_5_min == min(b, c)
    assert result.opponent_defense_strength_5_max == max(b, c)


def test_relative_direction_above_one_means_more_than_league_average():
    ratings = calculate_team_strength(matches())
    row = ratings.query("team == 'B' and gw == 3").iloc[0]
    assert row.team_defense_strength_3_rel > 1
