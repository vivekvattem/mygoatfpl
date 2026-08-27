"""Named, reviewable feature groups for linear benchmark ablations."""

BASE_FEATURES = [
    "price", "position", "fixture_count", "home_fixture_count", "away_fixture_count",
    "avg_fixture_difficulty", "is_home", "is_blank",
]
FORM_FEATURES = [
    "points_last_1", "points_last_3", "points_last_5", "xG_last_3", "xA_last_3",
    "xGI_last_3", "xGI_last_5", "points_per90_last_3", "xGI_per90_last_3",
    "bonus_last_3", "bps_last_3", "ict_index_last_3",
]
MINUTES_FEATURES = [
    "minutes_last_1", "minutes_last_3", "minutes_last_5", "avg_minutes_last_3",
    "start_rate_last_3", "start_rate_last_5", "xGI_per90_last_3_missing",
    "minutes_last_3_missing", "expected_minutes_proxy",
]
FIXTURE_FEATURES = [
    "fixture_count", "home_fixture_count", "away_fixture_count", "avg_fixture_difficulty",
    "min_fixture_difficulty", "max_fixture_difficulty", "is_home", "is_blank",
]
VALUE_FEATURES = [
    "price", "points_per_million_last_3", "points_per_million_last_5",
    "xGI_per_million_last_3", "xGI_per_million_last_5",
]
TEAM_STRENGTH_FEATURES = [
    f"{prefix}_{window}{suffix}"
    for window in (3, 5, 10)
    for prefix, suffix in (
        ("team_attack_strength", ""), ("team_defense_strength", ""),
        ("team_attack_strength", "_rel"), ("team_defense_strength", "_rel"),
        ("opponent_attack_strength", "_mean"), ("opponent_defense_strength", "_mean"),
        ("opponent_attack_strength", "_rel_mean"), ("opponent_defense_strength", "_rel_mean"),
    )
]

FEATURE_SETS = {
    "feature_set_basic": list(dict.fromkeys(BASE_FEATURES)),
    "feature_set_form": list(dict.fromkeys(BASE_FEATURES + FORM_FEATURES + MINUTES_FEATURES + VALUE_FEATURES)),
    "feature_set_full_linear": list(dict.fromkeys(
        BASE_FEATURES + FORM_FEATURES + MINUTES_FEATURES + FIXTURE_FEATURES
        + VALUE_FEATURES + TEAM_STRENGTH_FEATURES
    )),
}

full = FEATURE_SETS["feature_set_full_linear"]
FEATURE_SETS.update({
    "feature_set_without_fixture": [f for f in full if f not in FIXTURE_FEATURES],
    "feature_set_without_team_strength": [f for f in full if f not in TEAM_STRENGTH_FEATURES],
    "feature_set_without_value": [f for f in full if f not in VALUE_FEATURES],
    "feature_set_without_minutes": [f for f in full if f not in MINUTES_FEATURES],
    "feature_set_without_5gw": [f for f in full if "last_5" not in f],
    "feature_set_without_3gw": [f for f in full if "last_3" not in f and not f.endswith("_missing")],
})
