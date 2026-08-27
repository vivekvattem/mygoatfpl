from pathlib import Path

import pandas as pd

from fpl_predictor.historical import (
    VaastavHistoricalSource, add_blank_gameweeks, aggregate_player_gameweeks,
    normalize_fixture_rows,
)


def source_rows():
    return pd.DataFrame([
        {"GW": 2, "element": 7, "name": "Test_Player", "team": "Alpha",
         "position": "MID", "value": 65, "minutes": 60, "starts": 1,
         "total_points": 4, "goals_scored": 1, "assists": 0, "fixture": 20,
         "opponent_team": 2, "was_home": True, "expected_goals": 0.4},
        {"GW": 2, "element": 7, "name": "Test_Player", "team": "Alpha",
         "position": "MID", "value": 66, "minutes": 30, "starts": 0,
         "total_points": 2, "goals_scored": 0, "assists": 1, "fixture": 21,
         "opponent_team": 3, "was_home": False, "expected_goals": 0.1},
    ])


def fixture_data():
    return pd.DataFrame([
        {"id": 20, "event": 2, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"id": 21, "event": 2, "team_h": 3, "team_a": 1,
         "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])


TEAMS = pd.DataFrame({"id": [1, 2, 3], "name": ["Alpha", "Beta", "Gamma"]})


def test_normalization_and_double_gameweek_aggregation():
    rows = normalize_fixture_rows(source_rows(), fixture_data(), TEAMS, "2024-25")
    result = aggregate_player_gameweeks(rows).iloc[0]
    assert result["season_player_id"] == "2024-25_7"
    assert result["fixture_count"] == 2
    assert result["minutes"] == 90
    assert result["total_points"] == 6
    assert result["expected_goals"] == 0.5
    assert result["home_fixture_count"] == 1
    assert result["away_fixture_count"] == 1
    assert result["avg_fixture_difficulty"] == 2.5
    assert pd.isna(result["is_home"])


def test_blank_gameweek_is_distinct_from_played_zero():
    base = aggregate_player_gameweeks(normalize_fixture_rows(
        pd.concat([source_rows().iloc[[0]].assign(GW=1, fixture=10),
                   source_rows().iloc[[0]].assign(GW=3, fixture=30)]),
        pd.DataFrame([
            {"id": 10, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"id": 30, "event": 3, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 4},
        ]), TEAMS, "2024-25"
    ))
    fixtures = pd.DataFrame([
        {"event": 1, "team_h": 1, "team_a": 2},
        {"event": 3, "team_h": 1, "team_a": 3},
    ])
    result = add_blank_gameweeks(base, fixtures, TEAMS)
    blank = result[result["gw"] == 2].iloc[0]
    assert blank["fixture_count"] == 0
    assert bool(blank["did_not_play_because_team_blank"])
    assert blank["total_points"] == 0


class FakeResponse:
    content = b"a,b\n1,2\n"
    def raise_for_status(self): pass


class FakeSession:
    def get(self, url, timeout): return FakeResponse()


def test_replaceable_source_downloads_expected_files(tmp_path: Path):
    paths = VaastavHistoricalSource(session=FakeSession()).fetch_season("2099-00", tmp_path)
    assert set(paths) == {"gameweeks", "fixtures", "teams", "players"}
    assert all(path.exists() for path in paths.values())
