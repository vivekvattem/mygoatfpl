import pytest

from fpl_predictor.feature_registry import FeatureRegistryError, load_feature_registry, validate_predictor_timing


def test_registry_expands_required_provenance_fields():
    registry = load_feature_registry()
    feature = registry["points_last_3"]
    assert feature.timing == "pre_gw"
    assert feature.window == 3
    assert set(feature.to_dict()) == {"name", "category", "dtype", "source", "timing", "window", "position_applicability", "description"}


def test_target_or_unknown_cannot_be_predictor():
    registry = load_feature_registry()
    with pytest.raises(FeatureRegistryError, match="not timed pre_gw"):
        validate_predictor_timing(["target_points"], registry)
    with pytest.raises(FeatureRegistryError, match="Unregistered"):
        validate_predictor_timing(["mystery_feature"], registry)
