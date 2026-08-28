import json

import pytest

from fpl_predictor.model_artifacts import ModelArtifactError, ProductionArtifacts


def test_corrupt_manifest_and_wrong_position_fail(tmp_path):
    bad = tmp_path / "bad.json"; bad.write_text("{}")
    with pytest.raises(ModelArtifactError, match="missing metadata"):
        ProductionArtifacts(tmp_path, bad)
    manifest = {"model_name": "Ridge", "feature_set": "form", "feature_names": ["price"],
                "trained_on_seasons": ["x"], "position_model_paths": {}}
    good = tmp_path / "good.json"; good.write_text(json.dumps(manifest))
    with pytest.raises(ModelArtifactError, match="No production artifact"):
        ProductionArtifacts(tmp_path, good).load_position("MID")


def test_validate_all_requires_global_fallback(tmp_path):
    manifest = {"model_name": "Ridge", "feature_set": "form", "feature_names": ["price"],
                "trained_on_seasons": ["x"], "position_model_paths": {}, "global_model_path": None}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    with pytest.raises(ModelArtifactError, match="position GK"):
        ProductionArtifacts(tmp_path, path).validate_all()
