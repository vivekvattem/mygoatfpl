import pandas as pd

from fpl_predictor.signals import (
    action_label, add_player_signals, availability_signal, minutes_signal, overall_signal,
)


def test_availability_and_minutes_thresholds():
    assert availability_signal("available") == "GREEN"
    assert availability_signal("doubtful", 75) == "YELLOW"
    assert availability_signal("injured", 0) == "RED"
    assert minutes_signal(80, "high") == "GREEN"
    assert minutes_signal(60, "medium") == "YELLOW"
    assert minutes_signal(30, "high") == "RED"


def test_overall_signal_and_action_mapping_are_explainable():
    green = pd.Series({field: "GREEN" for field in ["availability_signal", "minutes_signal", "fixture_signal",
                      "form_signal", "value_signal", "outlook_signal"]})
    signal, reason = overall_signal(green)
    assert signal == "GREEN" and "+" in reason
    assert action_label(False, "GREEN") == "BUY"
    assert action_label(True, "GREEN") == "HOLD"
    assert action_label(True, "RED", True) == "SELL"
    assert action_label(True, "RED", False) == "WATCH / HOLD"


def test_player_signals_include_value_form_reasons_and_actions():
    players = pd.DataFrame({
        "player_id": [1, 2, 3], "player": ["A", "B", "C"], "team": ["X"] * 3,
        "position": ["MID"] * 3, "owned": [False, True, True], "availability": ["available", "available", "injured"],
        "chance_of_playing_next_round": [None] * 3, "expected_minutes_proxy": [90, 60, 20],
        "minutes_confidence": ["high", "medium", "high"], "xGI_last_3": [3, 1, 0],
        "points_last_3": [20, 8, 1], "start_rate_last_3": [1, .7, .1], "value": [3, 2, 1],
        "weighted_xpts_5": [25, 15, 5],
    })
    fixtures = pd.DataFrame({"team": ["X"], "fixture_signal": ["GREEN"], "fixture_reason": ["easy"],
                             "fixtures_next_5": [5], "average_fdr_5": [2.4]})
    output = add_player_signals(players, fixtures, worthwhile_out_ids=[3])
    assert {"overall_signal", "value_signal", "form_signal", "signal_reason", "risk_reason", "action"}.issubset(output)
    assert output.iloc[0].action == "BUY" and output.iloc[2].action == "SELL"
    assert output.iloc[0].value_signal == "GREEN" and output.iloc[2].value_signal == "RED"
