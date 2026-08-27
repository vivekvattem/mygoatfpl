import pandas as pd
import pytest

from fpl_predictor.utility import add_player_utilities, risk_adjusted_utility


def test_risk_utility_and_ceiling_score_are_transparent(legal_squad):
    risk = risk_adjusted_utility(pd.Series([5.0]), pd.Series([2.0]), pd.Series([8.0]), 0.1)
    assert risk.iloc[0] == pytest.approx(4.4)
    result = add_player_utilities(legal_squad, 3, 0.1)
    assert result.ceiling_score.between(0, 100).all()
    assert (result.risk_adjusted_utility <= result.planning_xpts).all()
