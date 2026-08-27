"""Leakage-safe rolling team and opponent strength ratings."""

from pathlib import Path

import numpy as np
import pandas as pd


def build_team_match_rows(fixtures: pd.DataFrame, teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize completed fixtures to one team-perspective row per match."""
    names = dict(zip(pd.to_numeric(teams["id"], errors="coerce"), teams["name"].astype(str)))
    rows: list[dict[str, object]] = []
    required = {"id", "event", "team_h", "team_a", "team_h_score", "team_a_score"}
    if not required.issubset(fixtures.columns):
        return pd.DataFrame(columns=["season", "gw", "fixture_id", "team", "opponent", "is_home", "goals_for", "goals_against"])
    for fixture in fixtures.itertuples(index=False):
        if pd.isna(fixture.event) or pd.isna(fixture.team_h_score) or pd.isna(fixture.team_a_score):
            continue
        home, away = names.get(fixture.team_h, str(fixture.team_h)), names.get(fixture.team_a, str(fixture.team_a))
        common = {"season": season, "gw": int(fixture.event), "fixture_id": int(fixture.id)}
        rows.extend([
            {**common, "team": home, "opponent": away, "is_home": True,
             "goals_for": float(fixture.team_h_score), "goals_against": float(fixture.team_a_score)},
            {**common, "team": away, "opponent": home, "is_home": False,
             "goals_for": float(fixture.team_a_score), "goals_against": float(fixture.team_h_score)},
        ])
    return pd.DataFrame(rows).sort_values(["season", "gw", "fixture_id", "team"]).reset_index(drop=True)


def _mean_last(history: pd.DataFrame, column: str, window: int, home: bool | None = None) -> float:
    if home is not None:
        history = history[history["is_home"].eq(home)]
    values = pd.to_numeric(history.tail(window)[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def calculate_team_strength(
    matches: pd.DataFrame, windows: tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """Calculate ratings from fixtures strictly before each target Gameweek.

    Attack is goals scored per prior fixture. Defense is goals conceded per
    prior fixture, so values above 1.0 relative to the league indicate a weaker
    defense (more goals conceded).
    """
    if matches.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for season, season_matches in matches.groupby("season", sort=True):
        gameweeks = range(1, int(season_matches["gw"].max()) + 1)
        for gw in gameweeks:
            league_history = season_matches[season_matches["gw"] < gw]
            league_attack = pd.to_numeric(league_history["goals_for"], errors="coerce").mean()
            league_defense = pd.to_numeric(league_history["goals_against"], errors="coerce").mean()
            for team, team_matches in season_matches.groupby("team", sort=True):
                history = team_matches[team_matches["gw"] < gw].sort_values(["gw", "fixture_id"])
                row: dict[str, object] = {"season": season, "gw": gw, "team": team}
                for window in windows:
                    attack = _mean_last(history, "goals_for", window)
                    defense = _mean_last(history, "goals_against", window)
                    row.update({
                        f"team_attack_strength_{window}": attack,
                        f"team_defense_strength_{window}": defense,
                        f"team_attack_strength_{window}_rel": attack / league_attack if pd.notna(league_attack) and league_attack != 0 else np.nan,
                        f"team_defense_strength_{window}_rel": defense / league_defense if pd.notna(league_defense) and league_defense != 0 else np.nan,
                        f"team_attack_home_{window}": _mean_last(history, "goals_for", window, True),
                        f"team_attack_away_{window}": _mean_last(history, "goals_for", window, False),
                        f"team_defense_home_{window}": _mean_last(history, "goals_against", window, True),
                        f"team_defense_away_{window}": _mean_last(history, "goals_against", window, False),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def join_team_strength(
    players: pd.DataFrame,
    ratings: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """Join own ratings and aggregate every opponent in target-GW doubles."""
    own_columns = [column for column in ratings if column.startswith("team_") and column not in {"team"}]
    result = players.merge(
        ratings[["season", "gw", "team", *own_columns]],
        on=["season", "gw", "team"], how="left",
    )
    rating_labels = ("attack_strength", "defense_strength", "attack_strength_rel", "defense_strength_rel",
                     "attack_home", "attack_away", "defense_home", "defense_away")
    def names(label: str, window: int) -> tuple[str, str]:
        if label.endswith("_rel"):
            base = label.removesuffix("_rel")
            return f"team_{base}_{window}_rel", f"opponent_{base}_{window}_rel"
        return f"team_{label}_{window}", f"opponent_{label}_{window}"
    context = result[["season", "gw", "opponent"]].copy()
    context["_row_id"] = result.index
    context["opponent"] = context["opponent"].apply(
        lambda value: str(value).split("|") if pd.notna(value) else [np.nan]
    )
    exploded = context.explode("opponent", ignore_index=True)
    opponent_ratings = ratings.rename(columns={"team": "opponent"})
    exploded = exploded.merge(opponent_ratings, on=["season", "gw", "opponent"], how="left")
    source_to_output: dict[str, str] = {}
    for window in windows:
        for label in rating_labels:
            source, output_base = names(label, window)
            source_to_output[source] = output_base
    aggregated = exploded.groupby("_row_id")[list(source_to_output)].agg(["mean", "min", "max"])
    aggregated.columns = [
        f"{source_to_output[source]}_{aggregation}" for source, aggregation in aggregated.columns
    ]
    return result.join(aggregated, how="left")


def load_historical_team_matches(raw_dir: Path, seasons: tuple[str, ...]) -> pd.DataFrame:
    """Load cached Phase 2 fixture/team sources for configured seasons."""
    frames = []
    for season in seasons:
        season_dir = raw_dir / season
        fixtures = pd.read_csv(season_dir / "fixtures.csv", low_memory=False)
        teams = pd.read_csv(season_dir / "teams.csv", low_memory=False)
        frames.append(build_team_match_rows(fixtures, teams, season))
    return pd.concat(frames, ignore_index=True)
