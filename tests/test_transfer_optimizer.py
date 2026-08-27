import pandas as pd
import pytest

from fpl_predictor.squad_state import SquadState
from fpl_predictor.transfer_optimizer import (
    optimize_one_transfer, optimize_two_transfers, transfer_decision, transfer_hit_cost,
)


def test_hit_cost_supports_multiple_free_transfers():
    assert transfer_hit_cost(2, 1) == 4
    assert transfer_hit_cost(2, 2) == 0
    assert transfer_hit_cost(1, 5) == 0


def test_one_transfer_is_budget_legal_and_uses_selling_price(legal_squad, player_universe):
    legal_squad.loc[legal_squad.player_id.eq(15), "selling_price"] = 4.0
    player_universe.loc[player_universe.player_id.eq(19), "price"] = 5.0
    state = SquadState(1, 2, "manual_file", legal_squad, bank=1.0, free_transfers=1)
    moves = optimize_one_transfer(state, legal_squad, player_universe)
    move = moves[(moves.out_id.eq(15)) & moves.in_id.eq(19)].iloc[0]
    assert move.selling_price == 4 and move.new_bank == pytest.approx(0)
    assert move.gain_5gw > 0


def test_unknown_selling_price_strict_failure_and_scenario_mode(legal_squad, player_universe):
    squad = legal_squad.assign(selling_price=pd.NA)
    state = SquadState(1, 2, "manual_file", squad, bank=0, free_transfers=1)
    with pytest.raises(ValueError, match="selling prices"):
        optimize_one_transfer(state, squad, player_universe)
    assert not optimize_one_transfer(state, squad, player_universe, True).empty


def test_two_transfer_gain_and_roll_threshold(legal_squad, player_universe):
    state = SquadState(1, 2, "manual_file", legal_squad, bank=0, free_transfers=2)
    two = optimize_two_transfers(state, legal_squad, player_universe, selected_horizon=5, candidate_limit_per_position=2)
    assert not two.empty and two.iloc[0].hit_cost == 0
    assert transfer_decision(pd.DataFrame({"net_gain_5gw": [1.0]}), pd.DataFrame(), 5, 1.5) == "ROLL TRANSFER"
    assert transfer_decision(pd.DataFrame({"net_gain_5gw": [2.0]}), pd.DataFrame(), 5, 1.5) == "MAKE TRANSFER"
