from fpl_predictor.optimizer import best_15_player_squad
from fpl_predictor.squad_state import validate_squad


def test_full_squad_milp_respects_budget_positions_and_clubs(player_universe):
    result = best_15_player_squad(player_universe, 75, "weighted_xpts_5")
    validate_squad(result.optimal_15)
    assert result.budget_used <= 75
    assert result.optimal_15.player_id.nunique() == 15
