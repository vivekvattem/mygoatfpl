from pathlib import Path

import pandas as pd
import pytest

import fpl_predictor.multigw as multigw


def test_multi_gw_weights():
    assert multigw.weighted_projection([2, 3, 4], 3) == pytest.approx(7.9)
    assert multigw.weighted_projection([1, 1, 1, 1, 1], 5) == pytest.approx(4.0)


def test_blank_and_double_future_gameweeks(monkeypatch):
    base = pd.DataFrame({"player_id": [1, 2], "player": ["A", "B"], "team": ["A", "B"],
                         "position": ["MID", "MID"], "price": [5, 5], "fixture_count": [1, 1]})
    bootstrap = {"teams": [{"id": 1, "name": "A", "short_name": "A", "strength": 3},
                            {"id": 2, "name": "B", "short_name": "B", "strength": 3},
                            {"id": 3, "name": "C", "short_name": "C", "strength": 3}]}
    fixtures = [{"event": 2, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
                {"event": 2, "team_h": 3, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}]
    def fake_predict(features, artifacts, path):
        out = features.copy(); out["raw_xpts"] = out.fixture_count * 2.0; out["display_xpts"] = out.raw_xpts
        out["availability_adjusted_xpts"] = out.raw_xpts; out["xpts_lower"] = 0.; out["xpts_upper"] = out.raw_xpts + 2
        return out, None
    monkeypatch.setattr(multigw, "predict_live_players", fake_predict)
    result = multigw.build_multi_gw_projections(base, bootstrap, fixtures, object(), Path("unused"), 2, 1)
    assert result.loc[result.player_id.eq(1), "xpts_gw1"].iloc[0] == 4
    assert result.loc[result.player_id.eq(2), "xpts_gw1"].iloc[0] == 2
    blank = multigw.build_multi_gw_projections(base, bootstrap, [], object(), Path("unused"), 2, 1)
    assert blank.xpts_gw1.eq(0).all()
