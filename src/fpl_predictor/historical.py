"""Replaceable ingestion and normalization for historical FPL data."""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests

from .config import HISTORICAL_SOURCE_URL, REQUEST_TIMEOUT

CANONICAL_COLUMNS = [
    "season", "gw", "season_player_id", "player_id", "player_name", "team",
    "position", "price", "minutes", "starts", "total_points", "goals_scored",
    "assists", "clean_sheets", "goals_conceded", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
    "selected_by_percent", "transfers_in", "transfers_out", "fixture_id",
    "opponent", "is_home", "fixture_difficulty", "fixture_count",
    "home_fixture_count", "away_fixture_count", "avg_fixture_difficulty",
    "min_fixture_difficulty", "max_fixture_difficulty", "is_blank",
    "did_not_play_because_team_blank",
]

SUM_COLUMNS = [
    "minutes", "starts", "total_points", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bonus", "bps", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",
]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class HistoricalDataError(RuntimeError):
    """Raised when a historical season cannot be retrieved or parsed."""


class HistoricalSource(Protocol):
    """Contract allowing the community source to be replaced later."""

    def fetch_season(self, season: str, destination: Path) -> dict[str, Path]: ...


class VaastavHistoricalSource:
    """Download versioned season CSVs from Vaastav Anand's FPL repository."""

    FILES = {
        "gameweeks": "gws/merged_gw.csv",
        "fixtures": "fixtures.csv",
        "teams": "teams.csv",
        "players": "players_raw.csv",
    }

    def __init__(
        self,
        base_url: str = HISTORICAL_SOURCE_URL,
        timeout: float = REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch_season(self, season: str, destination: Path) -> dict[str, Path]:
        """Download one season, reusing valid local source files."""
        season_dir = destination / season
        season_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for label, remote_path in self.FILES.items():
            local_path = season_dir / remote_path.replace("/", "_")
            paths[label] = local_path
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            url = f"{self.base_url}/{season}/{remote_path}"
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise HistoricalDataError(f"Failed to download {season} {label} from {url}: {exc}") from exc
            temporary = local_path.with_suffix(".csv.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(local_path)
        return paths


def _column(frame: pd.DataFrame, *names: str, default: object = np.nan) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, *names: str) -> pd.Series:
    return pd.to_numeric(_column(frame, *names), errors="coerce")


def _team_name_map(teams: pd.DataFrame) -> dict[int, str]:
    if not {"id", "name"}.issubset(teams.columns):
        return {}
    ids = pd.to_numeric(teams["id"], errors="coerce")
    return {int(team_id): str(name) for team_id, name in zip(ids, teams["name"]) if pd.notna(team_id)}


def normalize_fixture_rows(
    raw: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Normalize source rows while retaining every fixture in a Double Gameweek."""
    source = raw.copy()
    team_names = _team_name_map(teams)
    fixture_lookup = fixtures.set_index("id", drop=False) if "id" in fixtures else pd.DataFrame()
    fixture_id = _numeric(source, "fixture", "fixture_id")
    is_home = _column(source, "was_home", "is_home").astype("boolean")
    team = _column(source, "team", "team_name")
    numeric_team = pd.to_numeric(team, errors="coerce")
    if numeric_team.notna().sum() == team.notna().sum():
        team = numeric_team.map(team_names).fillna(numeric_team.astype("string"))
    opponent_ids = _numeric(source, "opponent_team", "opponent")

    difficulty = pd.Series(np.nan, index=source.index, dtype=float)
    if not fixture_lookup.empty:
        home_fdr = fixture_id.map(fixture_lookup["team_h_difficulty"] if "team_h_difficulty" in fixture_lookup else {})
        away_fdr = fixture_id.map(fixture_lookup["team_a_difficulty"] if "team_a_difficulty" in fixture_lookup else {})
        difficulty = home_fdr.where(is_home.fillna(False), away_fdr)

    position = _column(source, "position", "element_type")
    numeric_position = pd.to_numeric(position, errors="coerce")
    position = numeric_position.map(POSITION_MAP).fillna(position.astype(str).str.upper())
    player_id = _numeric(source, "element", "player_id", "id").astype("Int64")
    result = pd.DataFrame({
        "season": season,
        "gw": _numeric(source, "GW", "gw", "round").astype("Int64"),
        "player_id": player_id,
        "player_name": _column(source, "name", "player_name", "web_name").astype(str),
        "team": team,
        "position": position,
        "price": _numeric(source, "value", "price") / (10 if "value" in source else 1),
        "minutes": _numeric(source, "minutes"),
        "starts": _numeric(source, "starts"),
        "total_points": _numeric(source, "total_points", "event_points"),
        "goals_scored": _numeric(source, "goals_scored"),
        "assists": _numeric(source, "assists"),
        "clean_sheets": _numeric(source, "clean_sheets"),
        "goals_conceded": _numeric(source, "goals_conceded"),
        "bonus": _numeric(source, "bonus"),
        "bps": _numeric(source, "bps"),
        "expected_goals": _numeric(source, "expected_goals"),
        "expected_assists": _numeric(source, "expected_assists"),
        "expected_goal_involvements": _numeric(source, "expected_goal_involvements"),
        "expected_goals_conceded": _numeric(source, "expected_goals_conceded"),
        "influence": _numeric(source, "influence"),
        "creativity": _numeric(source, "creativity"),
        "threat": _numeric(source, "threat"),
        "ict_index": _numeric(source, "ict_index"),
        "selected_by_percent": _numeric(source, "selected_by_percent"),
        "transfers_in": _numeric(source, "transfers_in"),
        "transfers_out": _numeric(source, "transfers_out"),
        "fixture_id": fixture_id.astype("Int64"),
        "opponent": opponent_ids.map(team_names).fillna(opponent_ids.astype("string")),
        "is_home": is_home,
        "fixture_difficulty": pd.to_numeric(difficulty, errors="coerce"),
    })
    result["season_player_id"] = season + "_" + player_id.astype("string")
    return result


def _join_unique(values: pd.Series) -> object:
    unique = [str(value) for value in values.dropna().unique()]
    return "|".join(unique) if unique else np.nan


def aggregate_player_gameweeks(fixture_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixture records into exactly one row per player-season-GW."""
    if fixture_rows.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    fixture_rows = fixture_rows.loc[fixture_rows["position"].isin(POSITION_MAP.values())].copy()
    keys = ["season", "gw", "season_player_id"]
    aggregations: dict[str, object] = {
        "player_id": "first", "player_name": "first", "team": "last",
        "position": "last", "price": "last", "selected_by_percent": "last",
        "transfers_in": "last", "transfers_out": "last", "fixture_id": _join_unique,
        "opponent": _join_unique, "is_home": "all",
        "fixture_difficulty": "mean",
    }
    aggregations.update({column: lambda values: values.sum(min_count=1) for column in SUM_COLUMNS})
    grouped = fixture_rows.groupby(keys, as_index=False, dropna=False).agg(aggregations)
    counts = fixture_rows.groupby(keys, as_index=False).agg(
        fixture_count=("fixture_id", "nunique"),
        home_fixture_count=("is_home", lambda x: int(x.fillna(False).sum())),
        away_fixture_count=("is_home", lambda x: int(x.fillna(False).eq(False).sum())),
        avg_fixture_difficulty=("fixture_difficulty", "mean"),
        min_fixture_difficulty=("fixture_difficulty", "min"),
        max_fixture_difficulty=("fixture_difficulty", "max"),
    )
    grouped = grouped.drop(columns=["fixture_difficulty"]).merge(counts, on=keys, how="left")
    grouped["is_home"] = grouped["home_fixture_count"].gt(0).astype("boolean")
    mixed = grouped["home_fixture_count"].gt(0) & grouped["away_fixture_count"].gt(0)
    grouped.loc[mixed, "is_home"] = pd.NA
    grouped["is_blank"] = False
    grouped["did_not_play_because_team_blank"] = False
    return grouped.reindex(columns=CANONICAL_COLUMNS)


def add_blank_gameweeks(player_gws: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Insert missing in-season player-GW rows and mark genuine team blanks."""
    if player_gws.empty:
        return player_gws.copy()
    team_names = _team_name_map(teams)
    max_gw = int(pd.to_numeric(fixtures.get("event"), errors="coerce").max())
    schedule: dict[tuple[str, int], int] = {}
    if {"event", "team_h", "team_a"}.issubset(fixtures.columns):
        for row in fixtures.itertuples(index=False):
            gw = getattr(row, "event")
            if pd.isna(gw):
                continue
            for team_id in (getattr(row, "team_h"), getattr(row, "team_a")):
                key = (team_names.get(int(team_id), str(team_id)), int(gw))
                schedule[key] = schedule.get(key, 0) + 1

    additions: list[dict[str, object]] = []
    for _, history in player_gws.groupby("season_player_id", sort=False):
        history = history.sort_values("gw")
        first_gw = int(history["gw"].min())
        last_gw = min(max_gw, int(history["gw"].max()))
        by_gw = history.set_index("gw")
        for gw in range(first_gw, last_gw + 1):
            if gw in by_gw.index:
                continue
            previous = history.loc[history["gw"] < gw]
            reference = (previous.iloc[-1] if not previous.empty else history.iloc[0]).to_dict()
            team = str(reference["team"])
            fixture_count = schedule.get((team, gw), 0)
            row = {column: np.nan for column in CANONICAL_COLUMNS}
            for column in ("season", "season_player_id", "player_id", "player_name", "team", "position", "price"):
                row[column] = reference[column]
            row.update({
                "gw": gw, "fixture_count": fixture_count, "home_fixture_count": 0,
                "away_fixture_count": 0, "is_blank": fixture_count == 0,
                "did_not_play_because_team_blank": fixture_count == 0,
            })
            for column in SUM_COLUMNS:
                row[column] = 0.0
            additions.append(row)
    addition_frame = pd.DataFrame(additions).dropna(axis=1, how="all")
    result = pd.concat([player_gws, addition_frame], ignore_index=True)
    return result.sort_values(["season", "season_player_id", "gw"]).reset_index(drop=True)


def load_and_normalize_season(paths: dict[str, Path], season: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Read one downloaded season and return player-GWs, fixtures, and raw row count."""
    raw = pd.read_csv(paths["gameweeks"], low_memory=False)
    fixtures = pd.read_csv(paths["fixtures"], low_memory=False)
    teams = pd.read_csv(paths["teams"], low_memory=False)
    fixture_rows = normalize_fixture_rows(raw, fixtures, teams, season)
    player_gws = add_blank_gameweeks(aggregate_player_gameweeks(fixture_rows), fixtures, teams)
    return player_gws, fixtures, len(raw)
