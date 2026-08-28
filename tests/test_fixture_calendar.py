import pandas as pd

from fpl_predictor.fixture_calendar import build_fixture_calendar, fixture_matrix, team_fixture_signals


def _fixture(fid, gw, home, away, hfdr=2, afdr=4):
    return {"id": fid, "event": gw, "team_h": home, "team_a": away,
            "team_h_difficulty": hfdr, "team_a_difficulty": afdr,
            "kickoff_time": f"2026-09-{fid:02d}T12:00:00Z"}


def test_calendar_detects_normal_blank_double_and_multiple_opponents():
    teams = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    fixtures = [_fixture(1, 2, 1, 2), _fixture(2, 3, 1, 2), _fixture(3, 3, 3, 1)]
    calendar = build_fixture_calendar(fixtures, teams, 2, 2)
    assert calendar.query("team == 'A' and gw == 2").iloc[0].schedule_label == "NORMAL"
    assert bool(calendar.query("team == 'C' and gw == 2").iloc[0].is_blank)
    double = calendar.query("team == 'A' and gw == 3").iloc[0]
    assert double.fixture_count == 2 and bool(double.is_double)
    assert double.opponents == ("B", "C") and double.home_away == ("H", "A")
    assert "DGW" in fixture_matrix(calendar, 2, 2).query("team == 'A'").iloc[0]["GW+2"]


def test_calendar_detects_triple_and_congestion_signal():
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
    fixtures = [_fixture(1, 2, 1, 2), _fixture(2, 2, 3, 1), _fixture(3, 2, 1, 4)]
    calendar = build_fixture_calendar(fixtures, teams, 2, 1)
    row = calendar.query("team == 'A'").iloc[0]
    assert row.fixture_count == 3 and bool(row.is_triple) and bool(row.is_congested)
    assert team_fixture_signals(calendar, 2).query("team == 'A'").iloc[0].fixture_signal == "YELLOW"
