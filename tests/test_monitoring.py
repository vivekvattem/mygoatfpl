from copy import deepcopy

import pandas as pd

from fpl_predictor.monitoring import (
    change_summary, classify_change, compare_live_state, completed_gw_monitoring,
    live_fingerprint, prediction_distribution,
)


def _bootstrap():
    return {
        "elements": [{"id": 1, "web_name": "Player", "now_cost": 75, "status": "a",
                      "chance_of_playing_next_round": None, "selected_by_percent": "10.0",
                      "transfers_in_event": 4, "transfers_out_event": 2, "news_added": None}],
        "teams": [{"id": 1, "name": "One"}, {"id": 2, "name": "Two"}],
        "events": [{"id": 1, "is_current": True, "is_next": False, "finished": False,
                    "data_checked": False, "deadline_time": "2026-08-30T10:00:00Z"}],
    }


def _fixtures():
    return [{"id": 10, "event": 1, "team_h": 1, "team_a": 2,
             "kickoff_time": "2026-08-30T12:00:00Z", "team_h_difficulty": 2,
             "team_a_difficulty": 4}]


def test_fingerprint_is_stable_and_meaningful_changes_are_detected():
    bootstrap, fixtures = _bootstrap(), _fixtures()
    assert live_fingerprint(bootstrap, fixtures) == live_fingerprint(deepcopy(bootstrap), deepcopy(fixtures))
    changed = deepcopy(bootstrap); changed["elements"][0]["now_cost"] = 76
    assert live_fingerprint(changed, fixtures)["players"] != live_fingerprint(bootstrap, fixtures)["players"]
    moved = deepcopy(fixtures); moved[0]["kickoff_time"] = "2026-08-31T12:00:00Z"
    assert live_fingerprint(bootstrap, moved)["fixtures"] != live_fingerprint(bootstrap, fixtures)["fixtures"]


def test_change_categories_and_player_summary():
    bootstrap, fixtures = _bootstrap(), _fixtures()
    same = compare_live_state(bootstrap, fixtures, deepcopy(bootstrap), deepcopy(fixtures))
    assert same.category == "NO_CHANGE"
    changed = deepcopy(bootstrap); changed["elements"][0]["status"] = "d"
    report = compare_live_state(bootstrap, fixtures, changed, fixtures)
    assert report.category == "PLAYER_DATA_CHANGED"
    assert "availability change" in change_summary(report)
    assert classify_change(report.fingerprint, report.fingerprint) == "NO_CHANGE"


def test_fixture_added_removed_moved_and_dgw_bgw_alerts():
    bootstrap, old = _bootstrap(), _fixtures()
    new = deepcopy(old)
    new[0]["event"] = 2
    new.append({"id": 11, "event": 2, "team_h": 1, "team_a": 2,
                "kickoff_time": "2026-09-01T12:00:00Z", "team_h_difficulty": 3,
                "team_a_difficulty": 3})
    report = compare_live_state(bootstrap, old, bootstrap, new)
    assert report.category == "FIXTURES_CHANGED"
    assert {change.kind for change in report.fixture_changes} >= {"added", "moved_gw"}
    assert any("DGW" in alert for alert in report.schedule_alerts)
    removed = compare_live_state(bootstrap, new, bootstrap, old)
    assert any(change.kind == "removed" for change in removed.fixture_changes)
    assert any("BGW" in alert or "NORMAL" in alert for alert in removed.schedule_alerts)


def test_prediction_distribution_and_completed_gw_monitoring():
    frame = pd.DataFrame({"position": ["GK", "MID", "MID"],
                          "availability_adjusted_xpts": [2.0, 4.0, 6.0]})
    result = prediction_distribution(frame)
    assert result["mean"] == 4.0 and result["median"] == 4.0
    assert result["position_means"]["MID"] == 5.0
    sparse = pd.DataFrame({"gw": [1], "actual_points": [2], "predicted_points": [3]})
    assert completed_gw_monitoring(sparse)["status"] == "INSUFFICIENT SAMPLE"
    scored = pd.DataFrame({"gw": [1, 1, 2, 2, 3, 3], "actual_points": [2, 5, 1, 7, 3, 6],
                           "predicted_points": [2.5, 4, 2, 6, 3, 5]})
    monitored = completed_gw_monitoring(scored)
    assert monitored["status"] == "NORMAL"
    assert {"mae", "rmse", "spearman", "top_10_overlap", "calibration_bins"} <= monitored.keys()
