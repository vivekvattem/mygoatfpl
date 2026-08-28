#!/usr/bin/env python3
"""Evaluate the frozen Phase 4 selection and generate OOF diagnostics."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from fpl_predictor.calibration import calibration_table  # noqa: E402
from fpl_predictor.config import HISTORICAL_ML_DIR, MODEL_DIR  # noqa: E402
from fpl_predictor.evaluation import gameweek_metrics, position_metrics  # noqa: E402
from fpl_predictor.feature_registry import load_feature_registry  # noqa: E402
from fpl_predictor.feature_sets import FEATURE_SETS  # noqa: E402
from fpl_predictor.model_registry import MODEL_SPECS  # noqa: E402
from fpl_predictor.modeling import (  # noqa: E402
    eligible, expanding_oof_predictions, load_governed_modeling_frame, load_model,
    model_metrics, residual_analysis,
)
from fpl_predictor.splits import season_holdout_split  # noqa: E402
from fpl_predictor.uncertainty import add_uncertainty_bounds, fit_residual_bands  # noqa: E402


def predict_frozen(test: pd.DataFrame, manifest: dict[str, object]) -> pd.Series:
    features = manifest["feature_names"]
    if manifest["selection_mode"] == "global":
        model = load_model(PROJECT_ROOT / manifest["global_model_path"])
        return pd.Series(model.predict(test[features]), index=test.index)
    predictions = pd.Series(index=test.index, dtype=float)
    for position, relative_path in manifest["position_model_paths"].items():
        subset = test[test.position.eq(position)]
        model = load_model(PROJECT_ROOT / relative_path)
        predictions.loc[subset.index] = model.predict(subset[features])
    return predictions


def run() -> dict[str, object]:
    manifest = json.loads((HISTORICAL_ML_DIR / "phase4_training.json").read_text())
    frame = load_governed_modeling_frame()
    test = eligible(season_holdout_split(frame).test)
    test_prediction = predict_frozen(test, manifest)
    selected_name = f"{manifest['model_name']} ({manifest['selection_mode']})"
    test_scored = test.assign(prediction=test_prediction)
    test_metrics = {"split": "test", "model": selected_name, "mode": manifest["selection_mode"],
                    "feature_set": manifest["feature_set"], **model_metrics(test, test_prediction)}

    registry = load_feature_registry()
    features_by_model = {
        name: FEATURE_SETS[feature_set]
        for name, feature_set in manifest["best_feature_set_by_model"].items()
    }
    oof = expanding_oof_predictions(frame, MODEL_SPECS, features_by_model, registry)
    pretest_oof = oof[oof.season.ne("2025-26")]
    bands = fit_residual_bands(pretest_oof[pretest_oof.model_name.eq(manifest["model_name"])])
    test_export = test[["season", "gw", "season_player_id", "player_name", "position"]].copy()
    test_export["actual_points"] = test.target_points
    test_export["predicted_points"] = test_prediction
    test_export = add_uncertainty_bounds(test_export, bands)

    validation = pd.read_csv(HISTORICAL_ML_DIR / "phase4_validation_results.csv")
    oof_rows = []
    for model, group in oof.groupby("model_name"):
        proxy = group.rename(columns={"actual_points": "target_points", "predicted_points": "prediction"})
        oof_rows.append({"split": "oof", "model": model, "mode": "global",
                         "feature_set": manifest["best_feature_set_by_model"][model],
                         **model_metrics(proxy, proxy.prediction)})
    model_results = pd.concat([validation, pd.DataFrame(oof_rows + [test_metrics])], ignore_index=True)

    position_frames, gw_frames = [], []
    for model, group in oof.groupby("model_name"):
        scored = group.rename(columns={"actual_points": "target_points", "predicted_points": "prediction"})
        positions = position_metrics(scored)
        positions.insert(0, "model", model); positions.insert(0, "split", "oof")
        position_frames.append(positions)
        gw_frames.append(gameweek_metrics(scored, model, "oof"))
    test_positions = position_metrics(test_scored)
    test_positions.insert(0, "model", selected_name); test_positions.insert(0, "split", "test")
    position_frames.append(test_positions)
    gw_frames.append(gameweek_metrics(test_scored, selected_name, "test"))
    by_position = pd.concat(position_frames, ignore_index=True)
    by_gw = pd.concat(gw_frames, ignore_index=True)

    calibration_frames = []
    for model, group in oof.groupby("model_name"):
        table = calibration_table(group)
        table.insert(0, "model", model); table.insert(0, "split", "oof")
        calibration_frames.append(table)
    test_calibration = calibration_table(test_export)
    test_calibration.insert(0, "model", selected_name); test_calibration.insert(0, "split", "test")
    calibration_frames.append(test_calibration)
    calibration = pd.concat(calibration_frames, ignore_index=True)

    residual_frames = []
    for model, group in oof.groupby("model_name"):
        context = frame.merge(group[["season", "gw", "season_player_id", "predicted_points", "actual_points"]],
                              on=["season", "gw", "season_player_id"], how="inner")
        residual_frames.append(residual_analysis(context, model, "oof"))
    test_context = test.copy()
    test_context["actual_points"] = test.target_points; test_context["predicted_points"] = test_prediction
    residual_frames.append(residual_analysis(test_context, selected_name, "test"))
    residuals = pd.concat(residual_frames, ignore_index=True)

    oof.to_csv(HISTORICAL_ML_DIR / "oof_predictions.csv", index=False)
    test_export.to_csv(HISTORICAL_ML_DIR / "test_predictions_with_uncertainty.csv", index=False)
    model_results.to_csv(HISTORICAL_ML_DIR / "model_results.csv", index=False)
    by_position.to_csv(HISTORICAL_ML_DIR / "model_results_by_position.csv", index=False)
    by_gw.to_csv(HISTORICAL_ML_DIR / "model_results_by_gw.csv", index=False)
    calibration.to_csv(HISTORICAL_ML_DIR / "calibration_results.csv", index=False)
    residuals.to_csv(HISTORICAL_ML_DIR / "residual_analysis.csv", index=False)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "selected_model": manifest,
               "test_metrics": test_metrics, "uncertainty": {"method": "10th/90th OOF residual quantiles by position", "bands": bands.to_dict(orient="records")}}
    (HISTORICAL_ML_DIR / "phase4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("FPL PHASE 4 MODEL EVALUATION\n")
    print("OOF MODELS\n" + pd.DataFrame(oof_rows)[["model", "mae", "rmse", "spearman", "top_25_precision", "ndcg_25"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFROZEN TEST RESULT\n" + pd.DataFrame([test_metrics])[["model", "mae", "rmse", "spearman", "top_25_precision", "ndcg_25", "average_regret"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nEvaluation artifacts saved successfully.")
    return summary


if __name__ == "__main__":
    run()
