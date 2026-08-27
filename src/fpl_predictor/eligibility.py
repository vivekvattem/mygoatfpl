"""Non-destructive minimum-history eligibility masks."""

from dataclasses import dataclass

import pandas as pd

MIN_PRIOR_GWS = 1
MIN_PRIOR_MINUTES = 30


@dataclass(frozen=True)
class EligibilityRules:
    minimum_prior_gameweeks: int = MIN_PRIOR_GWS
    minimum_prior_minutes: float = MIN_PRIOR_MINUTES
    minimum_start_rate: float | None = None
    history_window: int = 5


def eligibility_mask(frame: pd.DataFrame, rules: EligibilityRules = EligibilityRules()) -> pd.Series:
    """Return eligible rows without mutating or deleting stored observations."""
    games = pd.to_numeric(frame[f"games_last_{rules.history_window}"], errors="coerce")
    minutes = pd.to_numeric(frame[f"minutes_last_{rules.history_window}"], errors="coerce")
    mask = games.ge(rules.minimum_prior_gameweeks) & minutes.ge(rules.minimum_prior_minutes)
    if rules.minimum_start_rate is not None:
        rates = pd.to_numeric(frame[f"start_rate_last_{rules.history_window}"], errors="coerce")
        mask &= rates.ge(rules.minimum_start_rate)
    return mask.fillna(False)
