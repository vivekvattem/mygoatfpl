"""ML dataset assembly, target separation, metadata, and temporal splits."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

IDENTIFIER_COLUMNS = [
    "season", "gw", "season_player_id", "player_id", "player_name", "team", "position"
]
TARGET_COLUMNS = ["target_points", "target_minutes"]
FIXTURE_FEATURE_COLUMNS = [
    "price", "is_home", "fixture_count", "home_fixture_count", "away_fixture_count",
    "avg_fixture_difficulty", "min_fixture_difficulty", "max_fixture_difficulty",
    "is_blank", "did_not_play_because_team_blank",
]


@dataclass(frozen=True)
class DatasetColumns:
    """Explicit identifier, predictor, and target column partitions."""

    identifiers: list[str]
    features: list[str]
    targets: list[str]


def create_ml_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, DatasetColumns]:
    """Separate same-GW outcomes into targets and retain only safe predictors."""
    result = frame.copy()
    result["target_points"] = pd.to_numeric(result["total_points"], errors="coerce")
    result["target_minutes"] = pd.to_numeric(result["minutes"], errors="coerce")
    rolling_features = [column for column in result if "_last_" in column]
    features = [column for column in FIXTURE_FEATURE_COLUMNS if column in result]
    features.extend(column for column in rolling_features if column not in features)
    if "avg_points_last_3" in result:
        result["predicted_points_baseline"] = result["avg_points_last_3"]
        features.append("predicted_points_baseline")
    columns = DatasetColumns(
        identifiers=[column for column in IDENTIFIER_COLUMNS if column in result],
        features=features,
        targets=TARGET_COLUMNS,
    )
    return result[columns.identifiers + columns.features + columns.targets].copy(), columns


def chronological_split(
    frame: pd.DataFrame,
    validation_seasons: Iterable[str],
    test_seasons: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by season, never randomly, with train containing older seasons."""
    validation_set, test_set = set(validation_seasons), set(test_seasons)
    if validation_set & test_set:
        raise ValueError("Validation and test seasons must not overlap")
    validation = frame[frame["season"].isin(validation_set)].copy()
    test = frame[frame["season"].isin(test_set)].copy()
    train = frame[~frame["season"].isin(validation_set | test_set)].copy()
    return train, validation, test


def dataset_summary(frame: pd.DataFrame, columns: DatasetColumns) -> dict[str, object]:
    """Create a compact, JSON-serializable build manifest."""
    missing = frame[columns.features].isna().mean().sort_values(ascending=False)
    return {
        "seasons": sorted(frame["season"].dropna().unique().tolist()),
        "row_count": int(len(frame)),
        "player_count": int(frame["season_player_id"].nunique()),
        "gameweeks": {
            season: sorted(group["gw"].dropna().astype(int).unique().tolist())
            for season, group in frame.groupby("season")
        },
        "feature_count": len(columns.features),
        "missingness_summary": {key: round(float(value), 4) for key, value in missing.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
