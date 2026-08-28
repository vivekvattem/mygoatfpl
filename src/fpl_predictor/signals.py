"""Deterministic, explainable player traffic-light signals."""

from typing import Iterable

import numpy as np
import pandas as pd


SIGNALS = {"GREEN", "YELLOW", "RED", "GREY"}
SIGNAL_SCORE = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0, "GREY": np.nan}


def availability_signal(availability: object, chance: object = None) -> str:
    label = str(availability).casefold()
    probability = pd.to_numeric(chance, errors="coerce")
    if label in {"injured", "unavailable", "suspended/unavailable"} or pd.notna(probability) and probability < 50:
        return "RED"
    if label == "doubtful" or pd.notna(probability) and probability < 100:
        return "YELLOW"
    if label == "available":
        return "GREEN"
    return "GREY"


def minutes_signal(minutes: object, confidence: object, green_minutes: float = 75,
                   red_minutes: float = 45) -> str:
    value = pd.to_numeric(minutes, errors="coerce")
    if pd.isna(value):
        return "GREY"
    if value < red_minutes:
        return "RED"
    if value < green_minutes or str(confidence).casefold() == "low":
        return "YELLOW"
    return "GREEN"


def percentile_signal(values: pd.Series | np.ndarray | list[object],
                      groups: pd.Series | np.ndarray | list[object] | None = None) -> pd.Series:
    """Return position-relative traffic lights for Series and array-like inputs.

    ``pd.to_numeric`` can return an ndarray for ndarray-like inputs.  Materialising a
    Series first keeps grouping, ranking, missing-value handling, and index alignment
    stable across pandas/NumPy versions.
    """
    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce")
    else:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")

    if groups is not None:
        if isinstance(groups, pd.Series):
            aligned_groups = groups.reindex(numeric.index)
        else:
            aligned_groups = pd.Series(groups, index=numeric.index)
        ranks = numeric.groupby(aligned_groups, dropna=False).rank(pct=True)
    else:
        ranks = numeric.rank(pct=True)

    signals = pd.Series(
        np.select([ranks.ge(2 / 3), ranks.le(1 / 3)], ["GREEN", "RED"], default="YELLOW"),
        index=numeric.index,
    )
    return signals.where(numeric.notna(), "GREY")


def overall_signal(row: pd.Series) -> tuple[str, str]:
    weights = {"availability_signal": 0.25, "minutes_signal": 0.20, "fixture_signal": 0.20,
               "form_signal": 0.15, "value_signal": 0.10, "outlook_signal": 0.10}
    available = [(field, SIGNAL_SCORE.get(row.get(field, "GREY")), weight) for field, weight in weights.items()]
    available = [(field, value, weight) for field, value, weight in available if not pd.isna(value)]
    if row.get("availability_signal") == "RED":
        return "RED", "Official availability is the dominant risk"
    if len(available) < 4:
        return "GREY", "Insufficient structured data for a reliable overall signal"
    score = sum(value * weight for _, value, weight in available) / sum(weight for _, _, weight in available)
    signal = "GREEN" if score >= 0.70 else "YELLOW" if score >= 0.40 else "RED"
    positives = [field.replace("_signal", "") for field, value, _ in available if value == 1][:2]
    risks = [field.replace("_signal", "") for field, value, _ in available if value == 0][:2]
    reason = "+ " + ", ".join(positives) if positives else "No strong positive component"
    if risks:
        reason += "; - " + ", ".join(risks)
    return signal, reason


def action_label(owned: bool, signal: str, worthwhile_sell: bool = False) -> str:
    if owned and signal == "GREEN":
        return "HOLD"
    if owned and signal == "RED":
        return "SELL" if worthwhile_sell else "WATCH / HOLD"
    if not owned and signal == "GREEN":
        return "BUY"
    return "WATCH"


def add_player_signals(players: pd.DataFrame, fixture_signals: pd.DataFrame | None = None,
                       worthwhile_out_ids: Iterable[int] = ()) -> pd.DataFrame:
    result = players.copy()
    if fixture_signals is not None and not fixture_signals.empty:
        result = result.merge(fixture_signals, on="team", how="left", suffixes=("", "_team"))
    result["fixture_signal"] = result.get("fixture_signal", pd.Series("GREY", index=result.index)).fillna("GREY")
    result["availability_signal"] = [availability_signal(a, c) for a, c in zip(
        result.get("availability", pd.Series(index=result.index)),
        result.get("chance_of_playing_next_round", pd.Series(index=result.index)))]
    result["minutes_signal"] = [minutes_signal(m, c) for m, c in zip(
        result.get("expected_minutes_proxy", pd.Series(index=result.index)),
        result.get("minutes_confidence", pd.Series(index=result.index)))]
    form = (0.5 * pd.to_numeric(result.get("xGI_last_3"), errors="coerce") +
            0.3 * pd.to_numeric(result.get("points_last_3"), errors="coerce") +
            0.2 * pd.to_numeric(result.get("start_rate_last_3"), errors="coerce"))
    result["form_signal"] = percentile_signal(form, result.get("position"))
    result["value_signal"] = percentile_signal(result.get("value", pd.Series(index=result.index)), result.get("position"))
    result["outlook_signal"] = percentile_signal(result.get("weighted_xpts_5", pd.Series(index=result.index)),
                                                   result.get("position"))
    combined = result.apply(overall_signal, axis=1)
    result["overall_signal"] = combined.map(lambda item: item[0])
    result["signal_reason"] = combined.map(lambda item: item[1])
    result["risk_reason"] = result.apply(_risk_reason, axis=1)
    worthwhile = {int(value) for value in worthwhile_out_ids}
    result["action"] = result.apply(
        lambda row: action_label(bool(row.get("owned", False)), row.overall_signal,
                                 int(row.player_id) in worthwhile), axis=1)
    result["transfer_signal"] = result.action.map({"BUY": "GREEN", "HOLD": "GREEN", "WATCH": "YELLOW",
                                                    "WATCH / HOLD": "YELLOW", "SELL": "RED"}).fillna("GREY")
    return result


def _risk_reason(row: pd.Series) -> str:
    risks = []
    if row.get("availability_signal") == "RED": risks.append("availability")
    if row.get("minutes_signal") == "RED": risks.append("low expected minutes")
    if row.get("fixture_signal") == "RED": risks.append("fixture outlook")
    if row.get("form_signal") == "RED": risks.append("recent form")
    return ", ".join(risks) if risks else "No major structured red flag"
