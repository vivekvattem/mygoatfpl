"""Governed sklearn pipelines for Phase 4 benchmarks."""

import os
from typing import Sequence

import pandas as pd
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_registry import FeatureSpec, validate_predictor_timing
from .model_registry import ModelSpec


def build_model_pipeline(spec: ModelSpec, features: Sequence[str],
                         registry: dict[str, FeatureSpec]) -> Pipeline:
    """Build a missing-safe model using only governed pre-GW features."""
    feature_names = list(features)
    validate_predictor_timing(feature_names, registry)
    categorical = [name for name in feature_names if name in {"position", "team"}]
    numeric = [name for name in feature_names if name not in categorical]
    dense = spec.kind == "hist_gradient_boosting"
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if spec.kind == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]), categorical),
    ], sparse_threshold=0 if dense else .3)
    estimators = {
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
    }
    return Pipeline([("preprocessor", preprocessor), ("model", estimators[spec.kind](**spec.parameters))])


def fit_predict(pipeline: Pipeline, train: pd.DataFrame, validation: pd.DataFrame,
                features: Sequence[str]) -> pd.Series:
    pipeline.fit(train[list(features)], train.target_points)
    return pd.Series(pipeline.predict(validation[list(features)]), index=validation.index)
