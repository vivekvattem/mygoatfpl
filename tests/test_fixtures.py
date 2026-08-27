import pandas as pd

from fpl_predictor.fixtures import fixture_difficulty_summary, normalize_fixtures


def fixture(gw, home=1, away=2, home_fdr=2, away_fdr=4):
    return {
        "event": gw, "team_h": home, "team_a": away,
        "team_h_difficulty": home_fdr, "team_a_difficulty": away_fdr,
        "finished": False, "kickoff_time": "2026-09-01T14:00:00Z",
    }


def test_normalization_creates_home_and_away_rows():
    result = normalize_fixtures([fixture(1)])
    assert len(result) == 2
    assert result.iloc[0][["team", "opponent", "is_home"]].tolist() == [1, 2, True]
    assert result.iloc[1][["team", "opponent", "is_home"]].tolist() == [2, 1, False]


def test_three_and_five_gameweek_horizons_and_double_gameweek():
    raw = [fixture(gw, 1, gw + 1) for gw in range(2, 7)]
    raw.append(fixture(3, 1, 9, home_fdr=3))
    normalized = normalize_fixtures(raw)
    three = fixture_difficulty_summary(normalized, next_gw=2, horizon=3)
    five = fixture_difficulty_summary(normalized, next_gw=2, horizon=5)
    team_three = three.loc[three["team"] == 1].iloc[0]
    team_five = five.loc[five["team"] == 1].iloc[0]
    assert team_three["fixtures_count"] == 4
    assert team_three["home_matches"] == 4
    assert team_three["away_matches"] == 0
    assert team_five["fixtures_count"] == 6
    assert team_three["avg_fdr"] == 2.25
    assert team_five["avg_fdr"] == pytest.approx(13 / 6)


import pytest
