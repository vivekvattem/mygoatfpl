"""Multi-metric, reviewable model-selection scoring."""

import pandas as pd

SELECTION_WEIGHTS = {
    "mae": .30,
    "spearman": .25,
    "top_25_precision": .15,
    "ndcg_25": .20,
    "average_regret": .10,
}
LOWER_IS_BETTER = {"mae", "average_regret"}


def score_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize validation metrics and apply documented weights."""
    scored = results.copy()
    scored["selection_score"] = 0.0
    for metric, weight in SELECTION_WEIGHTS.items():
        values = pd.to_numeric(scored[metric], errors="coerce")
        span = values.max() - values.min()
        normalized = pd.Series(.5, index=scored.index) if span == 0 else (values - values.min()) / span
        if metric in LOWER_IS_BETTER:
            normalized = 1 - normalized
        scored[f"selection_component_{metric}"] = normalized
        scored["selection_score"] += weight * normalized.fillna(0)
    return scored.sort_values("selection_score", ascending=False).reset_index(drop=True)


def select_best_configuration(results: pd.DataFrame) -> pd.Series:
    if results.empty:
        raise ValueError("No validation candidates supplied")
    return score_candidates(results).iloc[0]
