#!/usr/bin/env python3
"""Select Phase 4 models on 2024/25 only and persist frozen artifacts."""

import json
import os
from pathlib import Path
import sys

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fpl_predictor.config import HISTORICAL_ML_DIR, MODEL_DIR, ensure_data_directories  # noqa: E402
from fpl_predictor.feature_registry import load_feature_registry  # noqa: E402
from fpl_predictor.feature_sets import FEATURE_SETS  # noqa: E402
from fpl_predictor.model_registry import MODEL_SPECS  # noqa: E402
from fpl_predictor.model_selection import score_candidates, select_best_configuration  # noqa: E402
from fpl_predictor.modeling import (  # noqa: E402
    eligible, load_governed_modeling_frame, model_metrics, position_specific_predict, save_model,
)
from fpl_predictor.models import build_model_pipeline, fit_predict  # noqa: E402
from fpl_predictor.splits import season_holdout_split  # noqa: E402


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def run() -> dict[str, object]:
    ensure_data_directories()
    frame = load_governed_modeling_frame()
    split = season_holdout_split(frame)
    train, validation = eligible(split.train), eligible(split.validation)
    registry = load_feature_registry()
    rows: list[dict[str, object]] = []
    core_sets = ("feature_set_basic", "feature_set_form", "feature_set_full_linear")

    for spec in MODEL_SPECS:
        for feature_set in core_sets:
            features = FEATURE_SETS[feature_set]
            prediction = fit_predict(build_model_pipeline(spec, features, registry), train, validation, features)
            rows.append({"split": "validation", "model": spec.name, "mode": "global",
                         "feature_set": feature_set, **model_metrics(validation, prediction)})
    for feature_set in (
        "feature_set_without_fixture", "feature_set_without_team_strength",
        "feature_set_without_value", "feature_set_without_minutes",
        "feature_set_without_5gw", "feature_set_without_3gw",
    ):
        spec = MODEL_SPECS[0]
        features = FEATURE_SETS[feature_set]
        prediction = fit_predict(build_model_pipeline(spec, features, registry), train, validation, features)
        rows.append({"split": "validation", "model": spec.name, "mode": "global",
                     "feature_set": feature_set, **model_metrics(validation, prediction)})

    global_results = pd.DataFrame(rows)
    best_global = select_best_configuration(global_results)
    best_spec = next(spec for spec in MODEL_SPECS if spec.name == best_global.model)
    best_features = FEATURE_SETS[best_global.feature_set]
    position_prediction = position_specific_predict(best_spec, train, validation, best_features, registry)
    position_row = {"split": "validation", "model": best_spec.name, "mode": "position_specific",
                    "feature_set": best_global.feature_set, **model_metrics(validation, position_prediction)}
    candidates = pd.concat([global_results, pd.DataFrame([position_row])], ignore_index=True)
    scored = score_candidates(candidates)
    winner = scored.iloc[0]
    candidates.to_csv(HISTORICAL_ML_DIR / "phase4_validation_results.csv", index=False)

    best_feature_by_model = {}
    for spec in MODEL_SPECS:
        model_rows = global_results[global_results.model.eq(spec.name)]
        best_feature_by_model[spec.name] = str(select_best_configuration(model_rows).feature_set)

    final_training = pd.concat([train, validation], ignore_index=True)
    selected_spec = next(spec for spec in MODEL_SPECS if spec.name == winner.model)
    selected_features = FEATURE_SETS[str(winner.feature_set)]
    metadata = {
        "model_name": selected_spec.name, "trained_on_seasons": ["2022-23", "2023-24", "2024-25"],
        "feature_set": str(winner.feature_set), "feature_names": selected_features,
        "selection_mode": str(winner["mode"]),
        "validation_metrics": {key: float(winner[key]) for key in ("mae", "rmse", "spearman", "top_25_precision", "ndcg_25", "average_regret")},
    }
    global_pipeline = build_model_pipeline(selected_spec, selected_features, registry)
    global_pipeline.fit(final_training[selected_features], final_training.target_points)
    global_path = MODEL_DIR / f"{slug(selected_spec.name)}_global.joblib"
    save_model(global_pipeline, global_path, {**metadata, "position": "global"})
    position_paths = {}
    for position in ("GK", "DEF", "MID", "FWD"):
        subset = final_training[final_training.position.eq(position)]
        pipeline = build_model_pipeline(selected_spec, selected_features, registry)
        pipeline.fit(subset[selected_features], subset.target_points)
        path = MODEL_DIR / f"{slug(selected_spec.name)}_{position.lower()}.joblib"
        save_model(pipeline, path, {**metadata, "position": position})
        position_paths[position] = str(path.relative_to(PROJECT_ROOT))

    manifest = {
        **metadata, "global_model_path": str(global_path.relative_to(PROJECT_ROOT)),
        "position_model_paths": position_paths, "best_feature_set_by_model": best_feature_by_model,
        "selection_weights": {"mae": .30, "spearman": .25, "top_25_precision": .15, "ndcg_25": .20, "average_regret": .10},
        "test_season_used_for_selection": False,
    }
    (HISTORICAL_ML_DIR / "phase4_training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("FPL PHASE 4 MODEL TRAINING\n")
    print(candidates[["model", "mode", "feature_set", "mae", "spearman", "top_25_precision", "ndcg_25", "average_regret"]].sort_values("mae").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSelected: {winner.model} / {winner.feature_set} / {winner['mode']}")
    print("Test season consulted for selection: NO")
    print("Model artifacts saved successfully.")
    return manifest


if __name__ == "__main__":
    run()
