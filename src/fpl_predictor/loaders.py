"""Transform raw FPL payloads into clean tabular datasets."""

from typing import Any, Iterable

import numpy as np
import pandas as pd

PLAYER_COLUMNS = [
    "id", "first_name", "second_name", "web_name", "team", "team_name",
    "element_type", "position", "now_cost", "price", "total_points",
    "event_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "selected_by_percent", "transfers_in", "transfers_out",
    "transfers_in_event", "transfers_out_event", "form", "points_per_game",
    "chance_of_playing_next_round", "status", "player",
]

NUMERIC_COLUMNS = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "selected_by_percent", "form",
    "points_per_game", "influence", "creativity", "threat", "ict_index",
]


def load_players(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Build a stable player table while tolerating absent optional API fields."""
    elements = bootstrap.get("elements", [])
    teams = {row.get("id"): row.get("name") for row in bootstrap.get("teams", [])}
    positions = {
        row.get("id"): row.get("singular_name_short") or row.get("singular_name")
        for row in bootstrap.get("element_types", [])
    }
    players = pd.DataFrame(elements)
    if players.empty:
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    for column in PLAYER_COLUMNS:
        if column not in players.columns:
            players[column] = np.nan

    players["team_name"] = players["team"].map(teams)
    players["position"] = players["element_type"].map(positions)
    players["price"] = pd.to_numeric(players["now_cost"], errors="coerce") / 10.0
    for column in NUMERIC_COLUMNS:
        players[column] = pd.to_numeric(players[column], errors="coerce")

    first = players["first_name"].fillna("").astype(str).str.strip()
    second = players["second_name"].fillna("").astype(str).str.strip()
    players["player"] = (first + " " + second).str.strip()
    return players.loc[:, PLAYER_COLUMNS].copy()


def load_events(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Return Gameweek metadata from a bootstrap response."""
    return pd.DataFrame(bootstrap.get("events", []))


def load_teams(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Return team metadata from a bootstrap response."""
    return pd.DataFrame(bootstrap.get("teams", []))


def load_json_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert a generic sequence of JSON records into a DataFrame."""
    return pd.DataFrame(list(records))
