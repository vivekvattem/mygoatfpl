"""Leakage-safe player, team, and opponent rolling features."""

import numpy as np
import pandas as pd

PLAYER_METRICS = {
    "minutes": "minutes",
    "starts": "starts",
    "total_points": "points",
    "goals_scored": "goals",
    "assists": "assists",
    "expected_goals": "xG",
    "expected_assists": "xA",
    "expected_goal_involvements": "xGI",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
}


def _shifted_rolling(series: pd.Series, groups: pd.Series, window: int, operation: str) -> pd.Series:
    """Apply group-local shift before every rolling calculation."""
    shifted = series.groupby(groups, sort=False).shift(1)
    rolling = shifted.groupby(groups, sort=False).rolling(window, min_periods=1)
    values = getattr(rolling, operation)().reset_index(level=0, drop=True)
    return values.reindex(series.index)


def _safe_per90(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(minutes, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator * 90.0


def add_rolling_features(
    frame: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Create player-season rolling sums and means using prior GWs only."""
    result = frame.sort_values(["season_player_id", "gw"]).copy()
    groups = result["season_player_id"]
    generated: dict[str, pd.Series] = {}
    for source, label in PLAYER_METRICS.items():
        values = pd.to_numeric(result[source], errors="coerce")
        for window in windows:
            generated[f"{label}_last_{window}"] = _shifted_rolling(values, groups, window, "sum")
            generated[f"avg_{label}_last_{window}"] = _shifted_rolling(values, groups, window, "mean")

    for window in windows:
        previous_minutes = generated[f"minutes_last_{window}"]
        generated[f"games_last_{window}"] = _shifted_rolling(
            pd.Series(1.0, index=result.index), groups, window, "sum"
        )
        generated[f"start_rate_last_{window}"] = (
            generated[f"starts_last_{window}"] / generated[f"games_last_{window}"].replace(0, np.nan)
        )
        for numerator, label in (
            ("xG", "xG_per90"), ("xA", "xA_per90"),
            ("xGI", "xGI_per90"), ("points", "points_per90"),
        ):
            generated[f"{label}_last_{window}"] = _safe_per90(
                generated[f"{numerator}_last_{window}"], previous_minutes
            )
        price = pd.to_numeric(result["price"], errors="coerce").replace(0, np.nan)
        generated[f"points_per_million_last_{window}"] = generated[f"points_last_{window}"] / price
        generated[f"xGI_per_million_last_{window}"] = generated[f"xGI_last_{window}"] / price
    return pd.concat([result, pd.DataFrame(generated, index=result.index)], axis=1).sort_index()


def build_team_gameweeks(fixtures: pd.DataFrame, teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Create one team-GW result row from official historical fixtures."""
    if fixtures.empty or not {"event", "team_h", "team_a"}.issubset(fixtures.columns):
        return pd.DataFrame()
    names = {}
    if {"id", "name"}.issubset(teams.columns):
        names = dict(zip(pd.to_numeric(teams["id"], errors="coerce"), teams["name"].astype(str)))
    rows: list[dict[str, object]] = []
    for fixture in fixtures.itertuples(index=False):
        gw = getattr(fixture, "event")
        home_score = getattr(fixture, "team_h_score", np.nan)
        away_score = getattr(fixture, "team_a_score", np.nan)
        if pd.isna(gw) or pd.isna(home_score) or pd.isna(away_score):
            continue
        home_id, away_id = int(getattr(fixture, "team_h")), int(getattr(fixture, "team_a"))
        home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
        rows.extend([
            {"season": season, "gw": int(gw), "team": names.get(home_id, str(home_id)),
             "goals_for": home_score, "goals_against": away_score, "team_points": home_points},
            {"season": season, "gw": int(gw), "team": names.get(away_id, str(away_id)),
             "goals_for": away_score, "goals_against": home_score, "team_points": away_points},
        ])
    match_rows = pd.DataFrame(rows)
    if match_rows.empty:
        return match_rows
    aggregated = match_rows.groupby(["season", "gw", "team"], as_index=False).agg(
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        team_points=("team_points", "sum"),
    )
    max_gw = int(pd.to_numeric(fixtures["event"], errors="coerce").max())
    grid = pd.MultiIndex.from_product(
        [[season], sorted(aggregated["team"].unique()), range(1, max_gw + 1)],
        names=["season", "team", "gw"],
    ).to_frame(index=False)
    return grid.merge(aggregated, on=["season", "team", "gw"], how="left").fillna(
        {"goals_for": 0.0, "goals_against": 0.0, "team_points": 0.0}
    )


def add_team_rolling_features(
    team_gameweeks: pd.DataFrame, windows: tuple[int, ...] = (3, 5)
) -> pd.DataFrame:
    """Add prior-GW team strength features without crossing season/team groups."""
    if team_gameweeks.empty:
        return team_gameweeks.copy()
    result = team_gameweeks.sort_values(["season", "team", "gw"]).copy()
    groups = result["season"].astype(str) + "|" + result["team"].astype(str)
    for window in windows:
        for source, label in (
            ("goals_for", "team_goals_for"),
            ("goals_against", "team_goals_against"),
            ("team_points", "team_points"),
        ):
            result[f"{label}_last_{window}"] = _shifted_rolling(
                pd.to_numeric(result[source], errors="coerce"), groups, window, "sum"
            )
    return result.sort_index()


def merge_team_and_opponent_features(
    players: pd.DataFrame, team_features: pd.DataFrame, windows: tuple[int, ...] = (3, 5)
) -> pd.DataFrame:
    """Attach own-team and average opponent pre-match rolling features."""
    result = players.copy()
    feature_columns = [column for column in team_features if "_last_" in column]
    if not feature_columns:
        return result
    own = team_features[["season", "gw", "team", *feature_columns]]
    result = result.merge(own, on=["season", "gw", "team"], how="left")

    lookup = team_features.set_index(["season", "gw", "team"])[feature_columns]
    opponent_values: dict[str, list[float]] = {f"opponent_{c.removeprefix('team_')}": [] for c in feature_columns}
    for row in result[["season", "gw", "opponent"]].itertuples(index=False):
        opponents = [] if pd.isna(row.opponent) else str(row.opponent).split("|")
        matches = [lookup.loc[(row.season, row.gw, opponent)] for opponent in opponents if (row.season, row.gw, opponent) in lookup.index]
        for column in feature_columns:
            output = f"opponent_{column.removeprefix('team_')}"
            opponent_values[output].append(
                float(pd.Series([match[column] for match in matches]).mean()) if matches else np.nan
            )
    for column, values in opponent_values.items():
        result[column] = values
    return result
