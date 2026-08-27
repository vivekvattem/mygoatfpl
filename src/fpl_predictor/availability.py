"""Official-FPL-only availability classification and adjustment."""

import numpy as np
import pandas as pd

STATUS_LABELS = {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended/unavailable", "u": "suspended/unavailable", "n": "unavailable"}


def classify_availability(status: object) -> str:
    return STATUS_LABELS.get(str(status).lower(), "unknown")


def adjust_for_availability(raw_xpts: pd.Series, chance: pd.Series) -> pd.Series:
    """Scale only when FPL publishes a probability; never invent one."""
    probability = pd.to_numeric(chance, errors="coerce") / 100
    return raw_xpts.where(probability.isna(), raw_xpts * probability)


def minutes_confidence(frame: pd.DataFrame) -> pd.Series:
    games = pd.to_numeric(frame.get("games_last_5"), errors="coerce")
    chance = pd.to_numeric(frame.get("chance_of_playing_next_round"), errors="coerce")
    confidence = pd.Series(np.where(games.ge(5), "high", np.where(games.ge(3), "medium", "low")), index=frame.index)
    confidence.loc[chance.notna() & chance.lt(100)] = "low"
    return confidence
