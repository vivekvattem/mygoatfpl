"""Regression, ranking, position, Gameweek, and top-pick evaluation."""

import math

import numpy as np
import pandas as pd


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """Calculate paired MAE, RMSE, Spearman, means, and sample count."""
    paired = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if paired.empty:
        return {"mae": np.nan, "rmse": np.nan, "spearman": np.nan,
                "mean_actual": np.nan, "mean_predicted": np.nan, "sample_count": 0}
    error = paired.actual - paired.predicted
    spearman = (
        float(paired.actual.corr(paired.predicted, method="spearman"))
        if paired.actual.nunique() > 1 and paired.predicted.nunique() > 1 else np.nan
    )
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(math.sqrt((error ** 2).mean())),
        "spearman": spearman,
        "mean_actual": float(paired.actual.mean()),
        "mean_predicted": float(paired.predicted.mean()),
        "sample_count": int(len(paired)),
    }


def position_metrics(frame: pd.DataFrame, prediction_column: str = "prediction") -> pd.DataFrame:
    rows = []
    for position in ("GK", "DEF", "MID", "FWD"):
        subset = frame[frame.position.eq(position)]
        rows.append({"position": position, **regression_metrics(subset.target_points, subset[prediction_column])})
    return pd.DataFrame(rows)


def top_k_metrics(frame: pd.DataFrame, k_values: tuple[int, ...] = (10, 25, 50),
                  prediction_column: str = "prediction") -> dict[str, float]:
    """Average Top-K overlap metrics across season-Gameweek groups."""
    scores: dict[str, list[float]] = {f"top_{k}_{metric}": [] for k in k_values for metric in ("precision", "recall")}
    for _, group in frame.dropna(subset=[prediction_column, "target_points"]).groupby(["season", "gw"]):
        for k in k_values:
            size = min(k, len(group))
            if not size:
                continue
            predicted_ids = set(group.nlargest(size, prediction_column).season_player_id)
            actual_ids = set(group.nlargest(size, "target_points").season_player_id)
            overlap = len(predicted_ids & actual_ids)
            scores[f"top_{k}_precision"].append(overlap / len(predicted_ids))
            scores[f"top_{k}_recall"].append(overlap / len(actual_ids))
    return {name: float(np.mean(values)) if values else np.nan for name, values in scores.items()}


def top_pick_evaluation(frame: pd.DataFrame, prediction_column: str = "prediction") -> dict[str, float | int]:
    """Evaluate the highest predicted eligible player in each Gameweek."""
    picks, maxima = [], []
    for _, group in frame.dropna(subset=[prediction_column, "target_points"]).groupby(["season", "gw"]):
        pick = group.loc[group[prediction_column].idxmax()]
        picks.append(float(pick.target_points))
        maxima.append(float(group.target_points.max()))
    return {
        "average_top_pick_points": float(np.mean(picks)) if picks else np.nan,
        "average_actual_maximum": float(np.mean(maxima)) if maxima else np.nan,
        "average_regret": float(np.mean(np.array(maxima) - np.array(picks))) if picks else np.nan,
        "gameweek_count": len(picks),
    }


def ndcg_metrics(frame: pd.DataFrame, k_values: tuple[int, ...] = (10, 25, 50),
                 prediction_column: str = "prediction") -> dict[str, float]:
    """Average normalized discounted cumulative gain across Gameweeks."""
    scores = {f"ndcg_{k}": [] for k in k_values}
    for _, group in frame.dropna(subset=[prediction_column, "target_points"]).groupby(["season", "gw"]):
        relevance = group.target_points.clip(lower=0).astype(float)
        for k in k_values:
            size = min(k, len(group))
            discounts = np.log2(np.arange(2, size + 2))
            predicted_order = group.assign(_rel=relevance).nlargest(size, prediction_column)._rel.to_numpy()
            ideal = np.sort(relevance.to_numpy())[::-1][:size]
            dcg = float((predicted_order / discounts).sum())
            idcg = float((ideal / discounts).sum())
            scores[f"ndcg_{k}"].append(dcg / idcg if idcg else np.nan)
    return {name: float(np.nanmean(values)) if values else np.nan for name, values in scores.items()}


def high_ceiling_metrics(frame: pd.DataFrame, thresholds: tuple[int, ...] = (8, 10, 15),
                         prediction_column: str = "prediction") -> dict[str, float | int]:
    """Treat each points threshold as a transparent high-ceiling retrieval task."""
    paired = frame.dropna(subset=[prediction_column, "target_points"])
    output: dict[str, float | int] = {}
    for threshold in thresholds:
        actual = paired.target_points.ge(threshold)
        predicted = paired[prediction_column].ge(threshold)
        true_positive = actual & predicted
        output[f"ceiling_{threshold}_precision"] = float(true_positive.sum() / predicted.sum()) if predicted.any() else np.nan
        output[f"ceiling_{threshold}_recall"] = float(true_positive.sum() / actual.sum()) if actual.any() else np.nan
        output[f"ceiling_{threshold}_avg_prediction"] = float(paired.loc[actual, prediction_column].mean()) if actual.any() else np.nan
        output[f"ceiling_{threshold}_sample_count"] = int(actual.sum())
    return output


def gameweek_metrics(frame: pd.DataFrame, model: str, split: str,
                     prediction_column: str = "prediction") -> pd.DataFrame:
    """Return per-GW regression, ranking, Top-K, and top-pick measurements."""
    rows = []
    for (season, gw), group in frame.groupby(["season", "gw"]):
        metrics = regression_metrics(group.target_points, group[prediction_column])
        topk = top_k_metrics(group, prediction_column=prediction_column)
        top_pick = top_pick_evaluation(group, prediction_column=prediction_column)
        rows.append({"split": split, "model": model, "season": season, "gw": gw,
                     **metrics, **topk, **ndcg_metrics(group, prediction_column=prediction_column),
                     **high_ceiling_metrics(group, prediction_column=prediction_column), **top_pick})
    return pd.DataFrame(rows)
