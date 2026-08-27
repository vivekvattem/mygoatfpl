#!/usr/bin/env python3
"""Refresh official current data and predict next-GW xPts for all players."""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fpl_predictor.api import FPLAPIClient, FPLAPIError  # noqa: E402
from fpl_predictor.config import (  # noqa: E402
    HISTORICAL_ML_DIR, LIVE_DATA_DIR, MODEL_DIR, RAW_ARCHIVE_DIR, RAW_DATA_DIR,
    ensure_data_directories,
)
from fpl_predictor.fixtures import get_next_gameweek  # noqa: E402
from fpl_predictor.live_features import build_live_features  # noqa: E402
from fpl_predictor.live_inference import predict_live_players  # noqa: E402
from fpl_predictor.loaders import load_events  # noqa: E402
from fpl_predictor.model_artifacts import ProductionArtifacts  # noqa: E402
from fpl_predictor.utils import save_raw_with_archive, utc_timestamp  # noqa: E402


def run_live_predictions(client: FPLAPIClient | None = None) -> dict[str, object]:
    ensure_data_directories(); LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = client or FPLAPIClient()
    bootstrap, fixtures = api.get_bootstrap_static(), api.get_fixtures()
    stamp = utc_timestamp()
    save_raw_with_archive(bootstrap, "bootstrap_static", RAW_DATA_DIR, RAW_ARCHIVE_DIR, stamp)
    save_raw_with_archive(fixtures, "fixtures", RAW_DATA_DIR, RAW_ARCHIVE_DIR, stamp)
    events = load_events(bootstrap)
    target_gw = get_next_gameweek(events)
    if target_gw is None:
        raise RuntimeError("No upcoming FPL Gameweek is available for inference")
    completed = events.loc[events.get("finished", False).fillna(False), "id"].astype(int).tolist()
    event_payloads = {gw: api.get_event_live(gw) for gw in completed if gw < target_gw}
    features = build_live_features(bootstrap, fixtures, event_payloads, target_gw)
    artifacts = ProductionArtifacts(PROJECT_ROOT, HISTORICAL_ML_DIR / "phase4_training.json")
    predictions, schema = predict_live_players(features, artifacts, HISTORICAL_ML_DIR / "phase4_summary.json")
    predictions["owned"] = False
    columns = ["player_id", "player", "team", "position", "price", "raw_xpts", "display_xpts",
               "availability_adjusted_xpts", "xpts_lower", "xpts_upper", "expected_minutes_proxy",
               "minutes_confidence", "status", "chance_of_playing_next_round", "news", "owned"]
    predictions[columns].to_csv(LIVE_DATA_DIR / "player_predictions.csv", index=False)
    summary = {"entry_id": None, "target_gw": target_gw, "model_name": artifacts.manifest["model_name"],
               "feature_set": artifacts.manifest["feature_set"],
               "schema_validation": {"passed": schema.passed, "required": len(schema.required),
                                     "available": len(schema.matching), "missing": schema.missing,
                                     "dtype_mismatches": schema.dtype_mismatches},
               "player_count": len(predictions), "squad_player_count": None,
               "squad_raw_xpts": None, "squad_adjusted_xpts": None,
               "created_at": datetime.now(timezone.utc).isoformat()}
    (LIVE_DATA_DIR / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FPL LIVE PLAYER PREDICTIONS\n")
    print(f"Target Gameweek: {target_gw}\nPlayers scored: {len(predictions)}")
    print(f"Schema parity: {'PASSED' if schema.passed else 'FAILED'} ({len(schema.matching)}/{len(schema.required)})")
    print("\nTop projected players:")
    print(predictions.nlargest(10, "display_xpts")[["player", "team", "position", "display_xpts"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nCurrent xPts estimates are optimized for average expected-points ranking\nand are not yet reliable estimates of explosive 10+ point outcomes.")
    return {"predictions": predictions, "summary": summary, "bootstrap": bootstrap, "api": api}


if __name__ == "__main__":
    try:
        run_live_predictions()
    except (FPLAPIError, RuntimeError, ValueError) as exc:
        print(f"Live prediction failed: {exc}", file=sys.stderr); raise SystemExit(1)
