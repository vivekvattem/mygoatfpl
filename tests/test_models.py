import numpy as np
import pandas as pd
import pytest

from fpl_predictor.feature_registry import FeatureRegistryError, load_feature_registry
from fpl_predictor.model_registry import MODEL_SPECS
from fpl_predictor.models import build_model_pipeline


@pytest.mark.parametrize("spec", MODEL_SPECS)
def test_model_pipelines_run_with_missing_data(spec):
    frame = pd.DataFrame({"price": [4.5, 5.5, np.nan, 8.0, 6.0, 7.0],
                          "position": ["GK", "DEF", "MID", "FWD", "MID", "DEF"],
                          "points_last_3": [1, np.nan, 4, 5, 3, 2],
                          "target_points": [2, 1, 6, 5, 3, 2]})
    model = build_model_pipeline(spec, ["price", "position", "points_last_3"], load_feature_registry())
    model.fit(frame[["price", "position", "points_last_3"]], frame.target_points)
    assert np.isfinite(model.predict(frame[["price", "position", "points_last_3"]])).all()


def test_model_pipeline_rejects_target_leakage():
    with pytest.raises(FeatureRegistryError):
        build_model_pipeline(MODEL_SPECS[0], ["target_points"], load_feature_registry())
