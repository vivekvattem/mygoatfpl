import pandas as pd

from fpl_predictor.chips import ChipState, build_chip_plan, chip_signal, resolve_chip_states


def _states(value="available"):
    return {chip: ChipState(chip, value, "test") for chip in
            ("wildcard", "free_hit", "bench_boost", "triple_captain")}


def test_chip_state_used_override_and_no_data_grey():
    states = resolve_chip_states({"chips": [{"name": "freehit"}]}, {"wildcard": "available"})
    assert states["free_hit"].state == "used" and states["wildcard"].state == "available"
    assert chip_signal(None, states["wildcard"], 10, 5) == "GREY"
    assert chip_signal(20, ChipState("x", "used", "test"), 10, 5) == "GREY"


def test_chip_plan_detects_wildcard_free_hit_bench_boost_and_tc_opportunities():
    calendar = pd.DataFrame({"team": ["X", "Y"], "gw": [2, 2], "fixture_count": [2, 0]})
    squad = pd.DataFrame({"team": ["X"] * 11 + ["Y"] * 4, "overall_signal": ["RED"] * 6 + ["YELLOW"] * 9,
                          "multiplier": [1] * 11 + [0] * 4, "expected_minutes_proxy": [80] * 15,
                          "xpts_gw1": [5] * 15, "fixture_count_gw1": [2] * 15})
    players = pd.DataFrame({"player": ["Star"], "xpts_gw1": [9.0], "fixture_count_gw1": [2],
                            "ceiling_score": [90], "expected_minutes_proxy": [90]})
    plan = build_chip_plan(calendar, squad, players, _states(), 2, 1, wildcard_gain=12, free_hit_gains={2: 13})
    row = plan.iloc[0]
    assert row.wildcard_signal == "GREEN" and row.free_hit_signal == "GREEN"
    assert row.bench_boost_signal == "GREEN" and row.triple_captain_signal == "GREEN"
