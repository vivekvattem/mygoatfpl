"""Validated position-specific live expected-points inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .availability import adjust_for_availability, classify_availability, minutes_confidence
from .model_artifacts import ProductionArtifacts
from .schema_validation import require_live_schema


def predict_live_players(features: pd.DataFrame, artifacts: ProductionArtifacts,
                         phase4_summary_path: Path) -> tuple[pd.DataFrame, object]:
    required = list(artifacts.manifest["feature_names"])
    report = require_live_schema(features, required)
    raw = pd.Series(np.nan, index=features.index, dtype=float)
    for position in ("GK", "DEF", "MID", "FWD"):
        subset = features[features.position.eq(position)]
        if not subset.empty:
            raw.loc[subset.index] = artifacts.load_position(position).predict(subset[required])
    summary = json.loads(phase4_summary_path.read_text())
    bands = pd.DataFrame(summary["uncertainty"]["bands"]).set_index("position")
    output = features.copy()
    output["raw_xpts"] = raw
    output["display_xpts"] = raw.clip(lower=0)
    output["availability_adjusted_xpts"] = adjust_for_availability(output.display_xpts, output.chance_of_playing_next_round)
    output["availability"] = output.status.map(classify_availability)
    output["minutes_confidence"] = minutes_confidence(output)
    output["xpts_lower"] = [max(0, value + bands.loc[pos, "residual_lower"]) for value, pos in zip(output.display_xpts, output.position)]
    output["xpts_upper"] = [value + bands.loc[pos, "residual_upper"] for value, pos in zip(output.display_xpts, output.position)]
    return output, report
