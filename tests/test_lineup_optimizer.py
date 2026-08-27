import pytest

from fpl_predictor.lineup_optimizer import optimize_starting_xi


def test_optimizer_selects_highest_legal_xi_and_valid_formation(legal_squad):
    result = optimize_starting_xi(legal_squad)
    counts = result.starting_11.position.value_counts()
    assert len(result.starting_11) == 11 and counts.GK == 1
    assert 3 <= counts.DEF <= 5 and 2 <= counts.MID <= 5 and 1 <= counts.FWD <= 3
    assert result.formation in {"3-5-2", "3-4-3", "4-5-1", "4-4-2", "4-3-3", "5-4-1", "5-3-2", "5-2-3"}


def test_bench_order_and_unavailable_player_handling(legal_squad):
    legal_squad.loc[legal_squad.player_id.eq(13), "availability"] = "injured"
    result = optimize_starting_xi(legal_squad)
    assert 13 not in set(result.starting_11.player_id)
    assert result.bench.availability_adjusted_xpts.is_monotonic_decreasing
    assert result.bench_gk.position == "GK"
