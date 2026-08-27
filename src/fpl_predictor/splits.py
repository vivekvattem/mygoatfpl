"""Chronological season holdout and expanding-window splits."""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_seasons: tuple[str, ...]
    validation_seasons: tuple[str, ...]


@dataclass(frozen=True)
class HoldoutSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _assert_order(train: Sequence[str], later: Sequence[str], label: str) -> None:
    if not train or not later or max(train) >= min(later):
        raise ValueError(f"Training seasons must precede {label} seasons")


def season_holdout_split(
    frame: pd.DataFrame,
    train_seasons: Sequence[str] = ("2022-23", "2023-24"),
    validation_seasons: Sequence[str] = ("2024-25",),
    test_seasons: Sequence[str] = ("2025-26",),
) -> HoldoutSplit:
    """Select disjoint whole seasons and reject temporal inversions."""
    sets = [set(train_seasons), set(validation_seasons), set(test_seasons)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train, validation, and test seasons must be disjoint")
    _assert_order(train_seasons, validation_seasons, "validation")
    _assert_order((*train_seasons, *validation_seasons), test_seasons, "test")
    return HoldoutSplit(
        frame[frame.season.isin(train_seasons)].copy(),
        frame[frame.season.isin(validation_seasons)].copy(),
        frame[frame.season.isin(test_seasons)].copy(),
    )


def expanding_window_splits(frame: pd.DataFrame, seasons: Sequence[str] | None = None) -> list[TemporalSplit]:
    """Return folds where each validation season strictly follows all training seasons."""
    ordered = list(seasons or sorted(frame.season.dropna().unique()))
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("Seasons must be unique and chronological")
    folds = []
    for index in range(1, len(ordered)):
        train_seasons, validation = tuple(ordered[:index]), (ordered[index],)
        folds.append(TemporalSplit(
            frame[frame.season.isin(train_seasons)].copy(),
            frame[frame.season.isin(validation)].copy(),
            train_seasons, validation,
        ))
    return folds
