import pandas as pd

from fpl_predictor.ui.formatting import (
    decision_status_label, format_player_table, safe_public_summary, scenario_mode_label,
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
