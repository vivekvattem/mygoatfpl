"""Fixture normalization and official fixture-difficulty summaries."""

from typing import Any

import pandas as pd

FIXTURE_COLUMNS = [
    "gw", "team", "opponent", "is_home", "difficulty", "finished", "kickoff_time"
]


def normalize_fixtures(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one row per team perspective for every real fixture."""
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        home = fixture.get("team_h")
        away = fixture.get("team_a")
        if home is None or away is None:
            continue
        common = {
            "gw": fixture.get("event"),
            "finished": fixture.get("finished", False),
            "kickoff_time": fixture.get("kickoff_time"),
        }
        rows.append({
            **common, "team": home, "opponent": away, "is_home": True,
            "difficulty": fixture.get("team_h_difficulty"),
        })
        rows.append({
            **common, "team": away, "opponent": home, "is_home": False,
            "difficulty": fixture.get("team_a_difficulty"),
        })
    result = pd.DataFrame(rows, columns=FIXTURE_COLUMNS)
    if not result.empty:
        result["gw"] = pd.to_numeric(result["gw"], errors="coerce").astype("Int64")
        result["kickoff_time"] = pd.to_datetime(
            result["kickoff_time"], errors="coerce", utc=True
        )
    return result


def get_current_gameweek(events_df: pd.DataFrame) -> int | None:
    """Return the current Gameweek, falling back to the latest finished one."""
    if events_df.empty or "id" not in events_df:
        return None
    if "is_current" in events_df:
        current = events_df.loc[events_df["is_current"].fillna(False), "id"]
        if not current.empty:
            return int(current.iloc[0])
    if "finished" in events_df:
        finished = events_df.loc[events_df["finished"].fillna(False), "id"]
        if not finished.empty:
            return int(pd.to_numeric(finished, errors="coerce").max())
    return None


def get_next_gameweek(events_df: pd.DataFrame) -> int | None:
    """Return the next Gameweek flagged by FPL, or infer it from event state."""
    if events_df.empty or "id" not in events_df:
        return None
    if "is_next" in events_df:
        upcoming = events_df.loc[events_df["is_next"].fillna(False), "id"]
        if not upcoming.empty:
            return int(upcoming.iloc[0])
    current = get_current_gameweek(events_df)
    ids = pd.to_numeric(events_df["id"], errors="coerce").dropna().astype(int)
    candidates = ids[ids > (current or 0)]
    return int(candidates.min()) if not candidates.empty else None


def fixture_difficulty_summary(
    normalized_fixtures: pd.DataFrame,
    next_gw: int | None,
    horizon: int = 5,
) -> pd.DataFrame:
    """Aggregate official FDR over a Gameweek window, preserving doubles."""
    columns = ["team", "fixtures_count", "avg_fdr", "home_matches", "away_matches"]
    if next_gw is None or normalized_fixtures.empty:
        return pd.DataFrame(columns=columns)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    end_gw = next_gw + horizon - 1
    selected = normalized_fixtures.loc[
        normalized_fixtures["gw"].between(next_gw, end_gw, inclusive="both")
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["difficulty"] = pd.to_numeric(selected["difficulty"], errors="coerce")
    selected["home_match"] = selected["is_home"].eq(True).astype(int)
    selected["away_match"] = selected["is_home"].eq(False).astype(int)
    summary = (
        selected.groupby("team", as_index=False)
        .agg(
            fixtures_count=("team", "size"),
            avg_fdr=("difficulty", "mean"),
            home_matches=("home_match", "sum"),
            away_matches=("away_match", "sum"),
        )
        .sort_values(["avg_fdr", "team"], na_position="last")
        .reset_index(drop=True)
    )
    return summary.loc[:, columns]
