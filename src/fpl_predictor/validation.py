"""Validation and feature-timing audits for the historical ML dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FORBIDDEN_CURRENT_OUTCOMES = {
    "total_points", "event_points", "goals_scored", "assists", "bonus", "bps",
    "minutes", "starts", "expected_goals", "expected_assists",
    "expected_goal_involvements", "influence", "creativity", "threat", "ict_index",
}
VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}


class DatasetValidationError(ValueError):
    """Raised when a historical dataset violates a correctness invariant."""


@dataclass(frozen=True)
class ValidationReport:
    duplicates_passed: bool
    infinite_values_passed: bool
    leakage_passed: bool


def audit_feature_leakage(feature_columns: list[str]) -> list[str]:
    """Return unlagged current-GW outcomes that must not be model inputs."""
    return sorted(set(feature_columns) & FORBIDDEN_CURRENT_OUTCOMES)


def validate_dataset(frame: pd.DataFrame, feature_columns: list[str]) -> ValidationReport:
    """Fail fast on duplicates, leakage, impossible values, and infinities."""
    keys = ["season", "season_player_id", "gw"]
    if frame.duplicated(keys).any():
        raise DatasetValidationError("Duplicate player-season-GW rows detected")
    leaked = audit_feature_leakage(feature_columns)
    if leaked:
        raise DatasetValidationError(f"Current-GW outcome columns found in features: {leaked}")
    numeric = frame.select_dtypes(include="number")
    numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric_values).any():
        raise DatasetValidationError("Infinite values detected")
    if not frame["gw"].dropna().between(1, 38).all():
        raise DatasetValidationError("Gameweek outside expected 1..38 range")
    if "position" in frame and not set(frame["position"].dropna().unique()).issubset(VALID_POSITIONS):
        raise DatasetValidationError("Unexpected position value detected")
    if "price" in frame and (pd.to_numeric(frame["price"], errors="coerce").dropna() <= 0).any():
        raise DatasetValidationError("Non-positive player price detected")
    if "fixture_count" in frame and (frame["fixture_count"].dropna() < 0).any():
        raise DatasetValidationError("Negative fixture count detected")
    if "target_points" in frame and not frame["target_points"].dropna().between(-10, 40).all():
        raise DatasetValidationError("Implausible target points detected")
    rolling = [column for column in feature_columns if "_last_" in column]
    gw1 = frame[frame["gw"] == 1]
    if rolling and not gw1[rolling].isna().all().all():
        raise DatasetValidationError("GW1 rolling features contain non-prior-season values")
    return ValidationReport(True, True, True)
