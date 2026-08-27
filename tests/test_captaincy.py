from fpl_predictor.captaincy import rank_captains
from fpl_predictor.lineup_optimizer import optimize_starting_xi
from fpl_predictor.utility import add_player_utilities


def test_captain_and_vice_are_distinct_and_profiles_work(legal_squad):
    squad = add_player_utilities(legal_squad)
    lineup = optimize_starting_xi(squad).starting_11
    for profile in ("safe", "balanced", "aggressive"):
        result = rank_captains(lineup, profile)
        assert result.captain.player_id != result.vice_captain.player_id
        assert len(result.candidates) == 5
