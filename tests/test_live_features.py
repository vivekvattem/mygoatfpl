import pandas as pd

from fpl_predictor.live_features import build_live_features, normalize_live_history


def inputs(double=False):
    bootstrap = {"elements": [{"id": 1, "first_name": "A", "second_name": "P", "web_name": "P", "team": 1,
        "element_type": 3, "now_cost": 60, "status": "a"}],
        "teams": [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}, {"id": 3, "name": "Gamma"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
        "events": [{"id": 1, "finished": True, "deadline_time": "2026-08-01T10:00:00Z"},
                   {"id": 2, "finished": False, "deadline_time": "2026-08-08T10:00:00Z"}]}
    fixtures = [{"id": 10, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4,
                 "team_h_score": 1, "team_a_score": 0, "finished": True}]
    fixtures.append({"id": 20, "event": 2, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4,
                     "team_h_score": None, "team_a_score": None, "finished": False})
    if double:
        fixtures.append({"id": 21, "event": 2, "team_h": 3, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3,
                         "team_h_score": None, "team_a_score": None, "finished": False})
    stats = {name: 0 for name in ("minutes", "starts", "total_points", "goals_scored", "assists", "expected_goals",
             "expected_assists", "expected_goal_involvements", "bonus", "bps", "influence", "creativity", "threat", "ict_index")}
    stats.update({"minutes": 90, "starts": 1, "total_points": 5, "expected_goals": .4})
    return bootstrap, fixtures, {1: {"elements": [{"id": 1, "stats": stats}]}}


def test_live_history_and_shifted_target_features():
    bootstrap, fixtures, events = inputs()
    history = normalize_live_history(bootstrap, fixtures, events)
    assert history.iloc[0].total_points == 5 and history.iloc[0].fixture_count == 1
    live = build_live_features(bootstrap, fixtures, events, 2).iloc[0]
    assert live.points_last_1 == 5 and live.minutes_last_1 == 90


def test_double_and_blank_fixture_context():
    bootstrap, fixtures, events = inputs(True)
    assert build_live_features(bootstrap, fixtures, events, 2).iloc[0].fixture_count == 2
    fixtures = [fixture for fixture in fixtures if fixture["event"] != 2]
    blank = build_live_features(bootstrap, fixtures, events, 2).iloc[0]
    assert blank.fixture_count == 0 and bool(blank.is_blank)
