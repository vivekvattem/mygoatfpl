"""Transparent expected-points baselines and an interpretable Ridge benchmark."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_registry import FeatureSpec, validate_predictor_timing

BASELINE_NAMES = (
    "Previous GW", "3-GW Mean", "5-GW Mean", "Minutes-adjusted",
    "Position Mean", "Price + Position", "Ridge",
)


def add_missingness_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicit indicators without replacing the underlying missing values."""
    result = frame.copy()
    for column in ("is_home", "is_blank", "did_not_play_because_team_blank"):
        if column in result:
            result[column] = result[column].map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            )
    for column in ("xGI_per90_last_3", "minutes_last_3"):
        result[f"{column}_missing"] = result[column].isna().astype(int)
    return result


def previous_gw_prediction(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["points_last_1"], errors="coerce")


def rolling_mean_prediction(frame: pd.DataFrame, window: int) -> pd.Series:
    points = pd.to_numeric(frame[f"points_last_{window}"], errors="coerce")
    available = pd.to_numeric(frame[f"games_last_{window}"], errors="coerce").replace(0, np.nan)
    return points / available


def minutes_adjusted_prediction(frame: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(frame["avg_minutes_last_3"], errors="coerce").clip(0, 90)
    per90 = pd.to_numeric(frame["points_per90_last_3"], errors="coerce")
    return per90 * minutes / 90.0


@dataclass
class PositionMeanBaseline:
    means: pd.Series | None = None
    fallback: float = np.nan

    def fit(self, frame: pd.DataFrame, target: str = "target_points") -> "PositionMeanBaseline":
        self.means = frame.groupby("position")[target].mean()
        self.fallback = float(pd.to_numeric(frame[target], errors="coerce").mean())
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.means is None:
            raise RuntimeError("PositionMeanBaseline must be fitted before prediction")
        return frame["position"].map(self.means).fillna(self.fallback).astype(float)


def _linear_pipeline(numeric: list[str], categorical: list[str], estimator: object) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ])),
        ("model", estimator),
    ])


class PricePositionBaseline:
    """Training-only linear price + position relationship."""

    def __init__(self) -> None:
        self.pipeline = _linear_pipeline(["price"], ["position"], LinearRegression())

    def fit(self, frame: pd.DataFrame, target: str = "target_points") -> "PricePositionBaseline":
        self.pipeline.fit(frame[["price", "position"]], frame[target])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipeline.predict(frame[["price", "position"]]), index=frame.index)


class RidgeBenchmark:
    """Fixed-alpha Ridge pipeline with governed, named features."""

    def __init__(self, features: Sequence[str], registry: dict[str, FeatureSpec], alpha: float = 10.0) -> None:
        self.features = list(features)
        validate_predictor_timing(self.features, registry)
        categorical = [name for name in self.features if name in {"position", "team"}]
        numeric = [name for name in self.features if name not in categorical]
        self.pipeline = _linear_pipeline(numeric, categorical, Ridge(alpha=alpha))

    def fit(self, frame: pd.DataFrame, target: str = "target_points") -> "RidgeBenchmark":
        self.pipeline.fit(frame[self.features], frame[target])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipeline.predict(frame[self.features]), index=frame.index)


def transparent_predictions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the four non-fitted baseline predictions."""
    return {
        "Previous GW": previous_gw_prediction(frame),
        "3-GW Mean": rolling_mean_prediction(frame, 3),
        "5-GW Mean": rolling_mean_prediction(frame, 5),
        "Minutes-adjusted": minutes_adjusted_prediction(frame),
    }
