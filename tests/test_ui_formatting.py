import pandas as pd

from fpl_predictor.ui.components import (
    action_badge, analyst_evidence_text, analyst_suggested_questions, first_existing_column, risk_summary,
    signal_badge,
)
from fpl_predictor.analyst.citations import freshness_label

from fpl_predictor.ui.formatting import (
    decision_status_label, format_player_table, prepare_one_transfer_table, prepare_replacement_table,
    prepare_two_transfer_table, safe_public_summary, scenario_mode_label, transfer_signal, transfer_status_badge,
)


def test_decision_and_scenario_labels():
    assert decision_status_label("roll transfer") == "ROLL TRANSFER"
    assert "SCENARIO MODE" in scenario_mode_label(True)
    assert scenario_mode_label(False) == "STRICT MODE"


def test_player_table_has_readable_columns_and_rounding():
    frame = pd.DataFrame({"player": ["A"], "position": ["MID"], "price": [7.555],
                          "availability_adjusted_xpts": [4.126]})
    formatted = format_player_table(frame)
    assert formatted.columns.tolist() == ["Player", "Pos", "Price", "xPts"]
    assert formatted.Price.iloc[0] == 7.56 and formatted.xPts.iloc[0] == 4.13


def test_public_summary_removes_secret_shaped_fields():
    result = safe_public_summary({"entry_id": 1, "password": "bad", "token": "bad", "captain": "A"})
    assert result == {"entry_id": 1, "captain": "A"}


def test_transfer_signals_are_transparent_and_not_colour_only():
    assert transfer_signal(1.5, 1.5).startswith("GREEN")
    assert transfer_signal(0.2, 1.5).startswith("YELLOW")
    assert transfer_signal(0, 1.5).startswith("RED")
    assert transfer_status_badge(None, None, 1.5, blocked=True).startswith("GREY")


def test_transfer_tables_render_without_downloads():
    one = pd.DataFrame({"out": ["A"], "in": ["B"], "selling_price": [5.0], "buy_price": [5.1],
                        "new_bank": [0.2], "hit_cost": [0], "gain_1gw": [0.3], "gain_3gw": [0.8],
                        "gain_5gw": [1.6], "net_gain_5gw": [1.6], "in_id": [2]})
    two = pd.DataFrame({"out_1": ["A"], "out_2": ["C"], "in_1": ["B"], "in_2": ["D"],
                        "hit_cost": [4], "gain_3gw": [1.0], "gain_5gw": [2.0], "net_gain_5gw": [-2.0]})
    players = pd.DataFrame({"player_id": [2], "team": ["X"], "price": [5.1],
                            "availability_adjusted_xpts": [4.0], "weighted_xpts_3": [10.0],
                            "weighted_xpts_5": [15.0], "availability": ["available"]})
    assert {"OUT", "IN", "Signal"}.issubset(prepare_one_transfer_table(one, 5, 1.5))
    assert {"OUT 1", "IN 2", "Signal"}.issubset(prepare_two_transfer_table(two, 5, 1.5))
    assert {"Player", "Expected Gain", "Availability", "Signal"}.issubset(
        prepare_replacement_table(one, players, 5, 1.5)
    )


def test_signal_helpers_always_include_text_and_risk_reason():
    assert "GREEN" in signal_badge("GREEN") and "BUY" in action_badge("BUY")
    players = pd.DataFrame({"player": ["A"], "overall_signal": ["RED"], "action": ["WATCH / HOLD"],
                            "risk_reason": ["low expected minutes"], "weighted_xpts_5": [3.0]})
    summary = risk_summary(players)
    assert summary.iloc[0].risk_reason and summary.iloc[0].overall_signal == "RED"


def test_analyst_evidence_badges_and_suggestions_are_accessible_text():
    text = analyst_evidence_text(["ml_projection", "fixture_calendar"])
    assert "ML Projection" in text and "Fixture Calendar" in text
    suggestions = analyst_suggested_questions()
    assert "What should I do this week?" in suggestions and len(suggestions) >= 6
    assert freshness_label(True) == "STALE DATA" and freshness_label(False) == "LIVE DATA"


def test_risk_summary_uses_weighted_projection_when_present():
    players = pd.DataFrame({"player": ["A", "B"], "overall_signal": ["RED", "RED"],
                            "weighted_xpts_5": [5.0, 2.0]})
    assert risk_summary(players).player.tolist() == ["B", "A"]
    assert first_existing_column(players, ("missing", "weighted_xpts_5")) == "weighted_xpts_5"


def test_risk_summary_falls_back_to_xpts_then_signal_only_without_crashing():
    xpts_only = pd.DataFrame({"player": ["A", "B"], "overall_signal": ["YELLOW", "YELLOW"],
                              "xpts": [4.0, 1.0]})
    assert risk_summary(xpts_only).player.tolist() == ["B", "A"]
    no_projection = pd.DataFrame({"player": ["Green", "Risk"], "overall_signal": ["GREEN", "RED"]})
    assert risk_summary(no_projection).player.tolist() == ["Risk", "Green"]
    assert risk_summary(pd.DataFrame()).empty
