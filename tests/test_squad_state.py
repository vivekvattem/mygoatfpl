import pandas as pd
import pytest

from fpl_predictor.squad_state import SquadState, require_transfer_state, validate_squad, validate_squad_freshness


def test_legal_squad_and_club_constraints(legal_squad):
    validate_squad(legal_squad)
    illegal = legal_squad.copy(); illegal.loc[illegal.player_id.isin([1, 2, 3, 4]), "team"] = "X"
    with pytest.raises(ValueError, match="three players"):
        validate_squad(illegal)


def test_stale_squad_requires_override(legal_squad):
    state = SquadState(1, 5, "public_api", legal_squad, 0, 1, squad_gameweek=4)
    with pytest.raises(ValueError, match="stale"):
        validate_squad_freshness(state)
    validate_squad_freshness(state, allow_stale=True)


def test_unknown_financial_state_blocks_transfers(legal_squad):
    state = SquadState(1, 2, "manual_file", legal_squad, None, None)
    with pytest.raises(ValueError, match="known bank"):
        require_transfer_state(state)
    state = SquadState(1, 2, "manual_file", legal_squad.assign(selling_price=pd.NA), 0, 1)
    with pytest.raises(ValueError, match="selling prices"):
        require_transfer_state(state)
    require_transfer_state(state, assume_selling_price_current=True)
