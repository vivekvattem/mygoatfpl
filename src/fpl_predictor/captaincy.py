"""Transparent captain and vice-captain ranking profiles."""

from dataclasses import dataclass

import pandas as pd


CAPTAIN_PROFILES = {
    "safe": {"mean": 0.65, "ceiling": 0.10, "minutes": 0.20, "risk": 0.05},
    "balanced": {"mean": 0.55, "ceiling": 0.25, "minutes": 0.15, "risk": 0.05},
    "aggressive": {"mean": 0.40, "ceiling": 0.45, "minutes": 0.10, "risk": 0.05},
}


@dataclass(frozen=True)
class CaptaincyResult:
    captain: pd.Series
    vice_captain: pd.Series
    candidates: pd.DataFrame


def rank_captains(starting_11: pd.DataFrame, profile: str = "balanced") -> CaptaincyResult:
    if profile not in CAPTAIN_PROFILES:
        raise ValueError(f"Unknown captaincy profile: {profile}")
    if len(starting_11) < 2:
        raise ValueError("Captaincy requires at least two starters")
    frame = starting_11.copy()
    weights = CAPTAIN_PROFILES[profile]
    mean = pd.to_numeric(frame["availability_adjusted_xpts"], errors="coerce").rank(pct=True).fillna(0)
    ceiling = pd.to_numeric(frame["ceiling_score"], errors="coerce").rank(pct=True).fillna(0)
    minutes = pd.to_numeric(frame["expected_minutes_proxy"], errors="coerce").div(90).clip(0, 1).fillna(0)
    uncertainty = pd.to_numeric(
        frame["uncertainty_width"] if "uncertainty_width" in frame else pd.Series(0.0, index=frame.index),
        errors="coerce",
    )
    risk = uncertainty.rank(pct=True).fillna(0)
    frame["captaincy_score"] = 100 * (weights["mean"] * mean + weights["ceiling"] * ceiling +
                                        weights["minutes"] * minutes - weights["risk"] * risk)
    candidates = frame.sort_values("captaincy_score", ascending=False)
    return CaptaincyResult(candidates.iloc[0], candidates.iloc[1], candidates.head(5))
