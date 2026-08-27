import pandas as pd
import pytest

from fpl_predictor.eligibility import EligibilityRules, eligibility_mask
from fpl_predictor.splits import expanding_window_splits, season_holdout_split


def seasons_frame():
    return pd.DataFrame({"season": ["2022-23", "2023-24", "2024-25", "2025-26"],
                         "games_last_5": [1, 2, 3, 4], "minutes_last_5": [30, 60, 90, 120],
                         "start_rate_last_5": [0, .5, .7, 1]})


def test_holdout_never_puts_future_seasons_in_training():
    split = season_holdout_split(seasons_frame())
    assert split.train.season.tolist() == ["2022-23", "2023-24"]
    assert split.validation.season.tolist() == ["2024-25"]
    assert split.test.season.tolist() == ["2025-26"]
    with pytest.raises(ValueError, match="precede"):
        season_holdout_split(seasons_frame(), ["2024-25"], ["2023-24"], ["2025-26"])


def test_expanding_window_order():
    folds = expanding_window_splits(seasons_frame())
    assert [(fold.train_seasons, fold.validation_seasons) for fold in folds] == [
        (("2022-23",), ("2023-24",)),
        (("2022-23", "2023-24"), ("2024-25",)),
        (("2022-23", "2023-24", "2024-25"), ("2025-26",)),
    ]


def test_eligibility_masks_without_deleting_rows():
    frame = seasons_frame()
    broad = eligibility_mask(frame, EligibilityRules())
    strict = eligibility_mask(frame, EligibilityRules(3, 90, .5))
    assert broad.tolist() == [True, True, True, True]
    assert strict.tolist() == [False, False, True, True]
    assert len(frame) == 4
