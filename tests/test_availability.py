import pandas as pd

from fpl_predictor.availability import adjust_for_availability, classify_availability


def test_availability_adjustment_and_no_invented_probability():
    raw = pd.Series([8.0, 8.0]); chance = pd.Series([75, None])
    assert adjust_for_availability(raw, chance).tolist() == [6.0, 8.0]
    assert classify_availability("i") == "injured"
    assert classify_availability(None) == "unknown"
