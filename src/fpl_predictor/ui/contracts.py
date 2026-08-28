"""Small, explicit contracts for optional deployment/runtime data."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


PROJECTION_COLUMNS = (
    "weighted_xpts_5", "xpts_5gw", "five_gw_xpts", "weighted_xpts_3", "xpts_3gw",
    "availability_adjusted_xpts", "adjusted_xpts", "raw_xpts", "xpts",
)


def first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first available column without fabricating a substitute."""
    return next((column for column in candidates if column in frame.columns), None)


def safe_series(frame: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    """Return an index-aligned optional column."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index, name=column)


def safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return an index-aligned numeric optional column with missing values preserved."""
    return pd.to_numeric(safe_series(frame, column), errors="coerce")


def safe_value(row: Any, key: str, default: Any = None) -> Any:
    """Read an optional mapping/Series/tuple value without raising."""
    value = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    return default if value is None else value
