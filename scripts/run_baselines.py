#!/usr/bin/env python3
"""Run Phase 3 temporal baselines and the governed Ridge benchmark."""

import json
from datetime import datetime, timezone

import pandas as pd

from fpl_predictor.baselines import (  # noqa: E402
    PositionMeanBaseline, PricePositionBaseline, RidgeBenchmark,
    add_missingness_indicators, transparent_predictions,
)
from fpl_predictor.config import (  # noqa: E402
    HISTORICAL_ML_DIR, HISTORICAL_PROCESSED_DIR, HISTORICAL_RAW_DIR, HISTORICAL_SEASONS,
)
from fpl_predictor.eligibility import EligibilityRules, eligibility_mask  # noqa: E402
from fpl_predictor.evaluation import (  # noqa: E402
    gameweek_metrics, position_metrics, regression_metrics, top_k_metrics,
    top_pick_evaluation,
)
from fpl_predictor.feature_registry import load_feature_registry, validate_predictor_timing  # noqa: E402
from fpl_predictor.feature_sets import FEATURE_SETS  # noqa: E402
from fpl_predictor.splits import season_holdout_split  # noqa: E402
from fpl_predictor.team_strength import (  # noqa: E402
    calculate_team_strength, join_team_strength, load_historical_team_matches,
)


def load_phase3_frame() -> pd.DataFrame:
    """Load Phase 2 data and attach target fixtures plus pre-GW team ratings."""
    dataset = pd.read_csv(HISTORICAL_ML_DIR / "player_gameweek_dataset.csv", low_memory=False)
    contexts = []
    for season in HISTORICAL_SEASONS:
        path = HISTORICAL_PROCESSED_DIR / f"player_gameweeks_{season}.csv"
        context = pd.read_csv(path, usecols=["season", "gw", "season_player_id", "opponent"])
        contexts.append(context)
    context_frame = pd.concat(contexts, ignore_index=True)
    dataset = dataset.merge(context_frame, on=["season", "gw", "season_player_id"], how="left", validate="one_to_one")
    matches = load_historical_team_matches(HISTORICAL_RAW_DIR, HISTORICAL_SEASONS)
    ratings = calculate_team_strength(matches)
    return add_missingness_indicators(join_team_strength(dataset, ratings))


def predict_models(train: pd.DataFrame, evaluation: pd.DataFrame,
                   registry: dict, ridge_features: list[str]) -> dict[str, pd.Series]:
    """Fit only on the supplied historical training rows and predict evaluation rows."""
    predictions = transparent_predictions(evaluation)
    predictions["Position Mean"] = PositionMeanBaseline().fit(train).predict(evaluation)
    predictions["Price + Position"] = PricePositionBaseline().fit(train).predict(evaluation)
    predictions["Ridge"] = RidgeBenchmark(ridge_features, registry).fit(train).predict(evaluation)
    return predictions


def evaluate_predictions(frame: pd.DataFrame, predictions: dict[str, pd.Series], split_name: str
                         ) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    aggregate_rows, position_frames, gw_frames = [], [], []
    for model, prediction in predictions.items():
        scored = frame.assign(prediction=prediction)
        aggregate_rows.append({
            "split": split_name, "model": model,
            **regression_metrics(scored.target_points, scored.prediction),
            **top_k_metrics(scored), **top_pick_evaluation(scored),
        })
        by_position = position_metrics(scored)
        by_position.insert(0, "model", model)
        by_position.insert(0, "split", split_name)
        position_frames.append(by_position)
        gw_frames.append(gameweek_metrics(scored, model, split_name))
    return aggregate_rows, position_frames, gw_frames


def run() -> dict[str, object]:
    frame = load_phase3_frame()
    registry = load_feature_registry()
    ridge_features = FEATURE_SETS["feature_set_full_linear"]
    transparent_features = [
        "points_last_1", "points_last_3", "games_last_3", "points_last_5",
        "games_last_5", "avg_minutes_last_3", "points_per90_last_3", "position", "price",
    ]
    validate_predictor_timing(transparent_features, registry)
    validate_predictor_timing(ridge_features, registry)
    split = season_holdout_split(frame)
    rules = EligibilityRules()
    train = split.train[eligibility_mask(split.train, rules)].copy()
    validation = split.validation[eligibility_mask(split.validation, rules)].copy()
    test = split.test[eligibility_mask(split.test, rules)].copy()

    validation_predictions = predict_models(train, validation, registry, ridge_features)
    final_training = pd.concat([train, validation], ignore_index=True)
    test_predictions = predict_models(final_training, test, registry, ridge_features)

    aggregate_rows: list[dict[str, object]] = []
    position_frames: list[pd.DataFrame] = []
    gw_frames: list[pd.DataFrame] = []
    for target, predictions, name in ((validation, validation_predictions, "validation"),
                                      (test, test_predictions, "test")):
        aggregate, positions, gameweeks = evaluate_predictions(target, predictions, name)
        aggregate_rows.extend(aggregate)
        position_frames.extend(positions)
        gw_frames.extend(gameweeks)
    results = pd.DataFrame(aggregate_rows)
    by_position = pd.concat(position_frames, ignore_index=True)
    by_gw = pd.concat(gw_frames, ignore_index=True)
    results.to_csv(HISTORICAL_ML_DIR / "baseline_results.csv", index=False)
    by_position.to_csv(HISTORICAL_ML_DIR / "baseline_results_by_position.csv", index=False)
    by_gw.to_csv(HISTORICAL_ML_DIR / "baseline_results_by_gw.csv", index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": ["2022-23", "2023-24"],
        "validation_seasons": ["2024-25"], "test_seasons": ["2025-26"],
        "eligibility": rules.__dict__, "feature_sets": FEATURE_SETS,
        "ridge_feature_count": len(ridge_features),
        "eligible_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "results": results.to_dict(orient="records"),
    }
    (HISTORICAL_ML_DIR / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("FPL PHASE 3 BASELINE EVALUATION\n")
    print("Train seasons: 2022/23, 2023/24")
    print("Validation: 2024/25")
    print("Test: 2025/26")
    print("\nModels: " + ", ".join(validation_predictions))
    print("\nTEST RESULTS\n")
    print(results[results.split.eq("test")][["model", "mae", "rmse", "spearman"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nBY POSITION\n")
    print(by_position[by_position.split.eq("test")][["model", "position", "mae", "rmse", "spearman"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTOP-K PERFORMANCE\n")
    print(results[results.split.eq("test")][["model", "top_10_precision", "top_25_precision", "top_50_precision"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTOP-PICK REGRET\n")
    print(results[results.split.eq("test")][["model", "average_top_pick_points", "average_actual_maximum", "average_regret"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nResults saved successfully.")
    return {"results": results, "by_position": by_position, "by_gw": by_gw, "summary": summary}


if __name__ == "__main__":
    run()
