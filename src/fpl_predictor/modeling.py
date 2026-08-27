"""Phase 4 data preparation, temporal prediction, persistence, and diagnostics."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .baselines import add_missingness_indicators
from .config import HISTORICAL_ML_DIR, HISTORICAL_PROCESSED_DIR, HISTORICAL_RAW_DIR, HISTORICAL_SEASONS
from .eligibility import EligibilityRules, eligibility_mask
from .evaluation import (
    high_ceiling_metrics, ndcg_metrics, regression_metrics, top_k_metrics,
    top_pick_evaluation,
)
from .feature_registry import FeatureSpec
from .models import build_model_pipeline, fit_predict
from .model_registry import ModelSpec
from .splits import expanding_window_splits
from .team_strength import calculate_team_strength, join_team_strength, load_historical_team_matches


def expected_minutes_proxy(frame: pd.DataFrame) -> pd.Series:
    """Weighted, clipped proxy using lagged minutes and start rates only."""
    values = (
        .35 * pd.to_numeric(frame.avg_minutes_last_3, errors="coerce")
        + .25 * pd.to_numeric(frame.avg_minutes_last_5, errors="coerce")
        + .20 * pd.to_numeric(frame.minutes_last_1, errors="coerce")
        + 90 * .10 * pd.to_numeric(frame.start_rate_last_3, errors="coerce")
        + 90 * .10 * pd.to_numeric(frame.start_rate_last_5, errors="coerce")
    )
    return values.clip(0, 90)


def load_governed_modeling_frame() -> pd.DataFrame:
    """Assemble the Phase 2 dataset and Phase 3 strength context."""
    dataset = pd.read_csv(HISTORICAL_ML_DIR / "player_gameweek_dataset.csv", low_memory=False)
    contexts = []
    for season in HISTORICAL_SEASONS:
        contexts.append(pd.read_csv(
            HISTORICAL_PROCESSED_DIR / f"player_gameweeks_{season}.csv",
            usecols=["season", "gw", "season_player_id", "opponent"],
        ))
    context = pd.concat(contexts, ignore_index=True)
    dataset = dataset.merge(context, on=["season", "gw", "season_player_id"], how="left", validate="one_to_one")
    matches = load_historical_team_matches(HISTORICAL_RAW_DIR, HISTORICAL_SEASONS)
    result = add_missingness_indicators(join_team_strength(dataset, calculate_team_strength(matches)))
    result["expected_minutes_proxy"] = expected_minutes_proxy(result)
    return result


def eligible(frame: pd.DataFrame, rules: EligibilityRules = EligibilityRules()) -> pd.DataFrame:
    return frame[eligibility_mask(frame, rules)].copy()


def model_metrics(frame: pd.DataFrame, predictions: pd.Series) -> dict[str, float | int]:
    scored = frame.assign(prediction=predictions)
    return {
        **regression_metrics(scored.target_points, scored.prediction),
        **top_k_metrics(scored), **ndcg_metrics(scored),
        **high_ceiling_metrics(scored), **top_pick_evaluation(scored),
    }


def position_specific_predict(spec: ModelSpec, train: pd.DataFrame, validation: pd.DataFrame,
                              features: Sequence[str], registry: dict[str, FeatureSpec]) -> pd.Series:
    """Fit independent position pipelines without borrowing validation targets."""
    predictions = pd.Series(np.nan, index=validation.index, dtype=float)
    for position in ("GK", "DEF", "MID", "FWD"):
        train_position = train[train.position.eq(position)]
        validation_position = validation[validation.position.eq(position)]
        if train_position.empty or validation_position.empty:
            continue
        pipeline = build_model_pipeline(spec, features, registry)
        predictions.loc[validation_position.index] = fit_predict(
            pipeline, train_position, validation_position, features
        )
    return predictions


def expanding_oof_predictions(frame: pd.DataFrame, specs: Sequence[ModelSpec],
                              features_by_model: dict[str, Sequence[str]],
                              registry: dict[str, FeatureSpec]) -> pd.DataFrame:
    """Generate predictions only for rows outside each fold's training window."""
    rows = []
    for fold_number, fold in enumerate(expanding_window_splits(frame), 1):
        train, validation = eligible(fold.train), eligible(fold.validation)
        for spec in specs:
            features = features_by_model[spec.name]
            pipeline = build_model_pipeline(spec, features, registry)
            predictions = fit_predict(pipeline, train, validation, features)
            output = validation[["season", "gw", "season_player_id", "player_name", "position"]].copy()
            output["player"] = output["player_name"]
            output["actual_points"] = validation.target_points
            output["predicted_points"] = predictions
            output["model_name"] = spec.name
            output["fold"] = fold_number
            output["trained_through_season"] = max(fold.train_seasons)
            rows.append(output)
    return pd.concat(rows, ignore_index=True)


def save_model(pipeline: object, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps({**metadata, "created_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


def load_model(path: Path) -> object:
    return joblib.load(path)


def residual_analysis(frame: pd.DataFrame, model_name: str, split: str) -> pd.DataFrame:
    """Summarize signed/absolute residuals across operational segments."""
    data = frame.copy()
    data["residual"] = data.actual_points - data.predicted_points
    data["absolute_error"] = data.residual.abs()
    data["price_band"] = pd.cut(data.price, [-np.inf, 5, 7.5, 10, np.inf], labels=["budget", "mid", "premium", "elite"])
    data["minutes_history_band"] = pd.cut(data.minutes_last_5, [-np.inf, 90, 270, 450, np.inf], labels=["low", "rotation", "regular", "ever_present"])
    data["home_away"] = data.is_home.map({1.0: "home", 0.0: "away"}).fillna("mixed_or_blank")
    data["fixture_band"] = pd.cut(data.avg_fixture_difficulty, [-np.inf, 2.5, 3.5, np.inf], labels=["easy", "medium", "hard"])
    data["prediction_band"] = pd.cut(data.predicted_points, [-np.inf, 2, 4, 6, 8, np.inf], labels=["<2", "2-4", "4-6", "6-8", "8+"])
    data["season_stage"] = pd.cut(data.gw, [0, 5, 10, 38], labels=["GW1-5", "GW6-10", "GW11+"])
    dimensions = ["position", "price_band", "minutes_history_band", "home_away", "fixture_band", "prediction_band", "gw", "season_stage"]
    rows = []
    for dimension in dimensions:
        for segment, group in data.groupby(dimension, observed=False, dropna=False):
            rows.append({"split": split, "model": model_name, "dimension": dimension,
                         "segment": str(segment), "sample_count": len(group),
                         "mean_actual": group.actual_points.mean(),
                         "mean_predicted": group.predicted_points.mean(),
                         "mean_residual": group.residual.mean(), "mae": group.absolute_error.mean()})
    return pd.DataFrame(rows)
