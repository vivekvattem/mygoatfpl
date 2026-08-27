import numpy as np
import pandas as pd
import pytest

from fpl_predictor.validation import DatasetValidationError, audit_feature_leakage, validate_dataset


def valid_frame():
    return pd.DataFrame({
        "season": ["2024-25", "2024-25"], "season_player_id": ["2024-25_1"] * 2,
        "gw": [1, 2], "position": ["MID", "MID"], "price": [5.0, 5.1],
        "fixture_count": [1, 1], "target_points": [2, 5],
        "points_last_3": [np.nan, 2.0],
    })


def test_leakage_audit_rejects_current_outcomes_but_not_lags():
    assert audit_feature_leakage(["price", "points_last_3", "minutes"]) == ["minutes"]
    with pytest.raises(DatasetValidationError, match="Current-GW"):
        validate_dataset(valid_frame(), ["minutes"])


def test_validation_accepts_clean_data_and_rejects_duplicates_and_infinity():
    report = validate_dataset(valid_frame(), ["price", "points_last_3"])
    assert report.leakage_passed
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validate_dataset(pd.concat([valid_frame(), valid_frame().iloc[[0]]]), ["points_last_3"])
    broken = valid_frame()
    broken.loc[1, "points_last_3"] = np.inf
    with pytest.raises(DatasetValidationError, match="Infinite"):
        validate_dataset(broken, ["points_last_3"])
