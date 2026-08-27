from pathlib import Path

import numpy as np
import pandas as pd

from fpl_predictor.feature_registry import load_feature_registry
from fpl_predictor.model_registry import ModelSpec
from fpl_predictor.modeling import (
    expanding_oof_predictions, expected_minutes_proxy, load_model,
    position_specific_predict, save_model,
)
from fpl_predictor.models import build_model_pipeline


def synthetic_seasons():
    rows = []
    for season_index, season in enumerate(("2022-23", "2023-24", "2024-25")):
        for position_index, position in enumerate(("GK", "DEF", "MID", "FWD")):
            for gw in (1, 2, 3):
                rows.append({"season": season, "gw": gw,
                    "season_player_id": f"{season}_{position}_{gw}", "player_name": position,
                    "position": position, "price": 5 + position_index,
                    "points_last_3": np.nan if gw == 1 else gw + position_index,
                    "games_last_5": gw, "minutes_last_5": 90 * gw,
                    "target_points": gw + position_index})
    return pd.DataFrame(rows)


def test_expanding_oof_rows_never_come_from_training_seasons():
    frame = synthetic_seasons()
    spec = ModelSpec("Ridge", "ridge", {"alpha": 1.0})
    oof = expanding_oof_predictions(frame, [spec], {"Ridge": ["price", "position", "points_last_3"]}, load_feature_registry())
    assert (oof.trained_through_season < oof.season).all()
    assert set(oof.season) == {"2023-24", "2024-25"}


def test_expected_minutes_proxy_uses_only_named_lags():
    frame = pd.DataFrame({"avg_minutes_last_3": [60], "avg_minutes_last_5": [50],
                          "minutes_last_1": [90], "start_rate_last_3": [.5],
                          "start_rate_last_5": [.4], "target_minutes": [0]})
    first = expected_minutes_proxy(frame)
    frame.target_minutes = 90
    pd.testing.assert_series_equal(first, expected_minutes_proxy(frame))
    assert first.between(0, 90).all()


def test_position_specific_training_and_persistence(tmp_path: Path):
    frame = synthetic_seasons().query("season == '2022-23'")
    spec = ModelSpec("Ridge", "ridge", {"alpha": 1.0})
    prediction = position_specific_predict(spec, frame, frame, ["price", "position", "points_last_3"], load_feature_registry())
    assert prediction.notna().all()
    model = build_model_pipeline(spec, ["price", "position", "points_last_3"], load_feature_registry())
    model.fit(frame[["price", "position", "points_last_3"]], frame.target_points)
    path = tmp_path / "model.joblib"
    save_model(model, path, {"model_name": "Ridge"})
    assert path.exists() and path.with_suffix(".json").exists()
    assert len(load_model(path).predict(frame[["price", "position", "points_last_3"]])) == len(frame)
