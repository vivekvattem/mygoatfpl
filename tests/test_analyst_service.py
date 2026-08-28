from datetime import datetime, timezone

import pandas as pd

from fpl_predictor.analyst.provider import FakeProvider
from fpl_predictor.analyst.service import AnalystService
from fpl_predictor.ui.data import DashboardBundle, DataStatus
from fpl_predictor.ui.state import AppSettings


def _minimal_bundle():
    players = pd.DataFrame({"player_id": [1], "player": ["Cole Palmer"], "web_name": ["Palmer"],
                            "team": ["Chelsea"], "position": ["MID"], "price": [10.5], "owned": [True],
                            "availability_adjusted_xpts": [5.0], "total_xpts_3": [14.0], "total_xpts_5": [22.0],
                            "expected_minutes_proxy": [85], "minutes_confidence": ["high"],
                            "fixture_signal": ["GREEN"], "overall_signal": ["GREEN"], "action": ["HOLD"],
                            "availability": ["available"], "availability_signal": ["GREEN"],
                            "ceiling_score": [70], "uncertainty_width": [4], "average_fdr_5": [2.6],
                            "fixtures_next_5": [5], "risk_reason": ["none"], "signal_reason": ["good"]})
    return DashboardBundle(players, players, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {"target_gw": 2}, {},
                           DataStatus(True, False, datetime.now(timezone.utc), "ok"))


def test_provider_timeout_falls_back_to_deterministic_answer():
    response = AnalystService(FakeProvider(error=TimeoutError())).answer(
        "Is Cole Palmer available?", _minimal_bundle(), AppSettings())
    assert response.fallback_used and "Cole Palmer" in response.answer


def test_grounding_failure_discards_provider_answer():
    response = AnalystService(FakeProvider("Erling Haaland has 99.9 xPts.")).answer(
        "Is Cole Palmer available?", _minimal_bundle(), AppSettings())
    assert response.fallback_used and not response.validation_passed
    assert "Erling Haaland" not in response.answer

