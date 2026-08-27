"""Leakage-safe construction of current-season player and target-GW features."""

from typing import Any

import numpy as np
import pandas as pd

from .features import add_per90_features
from .baselines import add_missingness_indicators
from .fixtures import normalize_fixtures
from .loaders import load_events, load_players, load_teams
from .modeling import expected_minutes_proxy
from .rolling import PLAYER_METRICS, add_rolling_features
from .team_strength import build_team_match_rows, calculate_team_strength, join_team_strength


def current_season_label(events: pd.DataFrame) -> str:
    deadlines = pd.to_datetime(events.get("deadline_time"), errors="coerce", utc=True).dropna()
    year = int(deadlines.min().year) if not deadlines.empty else pd.Timestamp.now().year
    return f"{year}-{str(year + 1)[-2:]}"


def fixture_context(fixtures: list[dict[str, Any]], teams: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_fixtures(fixtures)
    names = teams.set_index("id")["name"].to_dict()
    normalized["team_name"] = normalized.team.map(names)
    normalized["opponent_name"] = normalized.opponent.map(names)
    return normalized.groupby(["gw", "team_name"], as_index=False).agg(
        fixture_count=("opponent", "size"), home_fixture_count=("is_home", "sum"),
        away_fixture_count=("is_home", lambda x: int(x.eq(False).sum())),
        avg_fixture_difficulty=("difficulty", "mean"), min_fixture_difficulty=("difficulty", "min"),
        max_fixture_difficulty=("difficulty", "max"), opponent=("opponent_name", lambda x: "|".join(x.dropna().astype(str).unique())),
    )


def normalize_live_history(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]],
                           event_payloads: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Create one current player × completed GW row, including blanks and doubles."""
    players, events, teams = load_players(bootstrap), load_events(bootstrap), load_teams(bootstrap)
    context = fixture_context(fixtures, teams)
    season = current_season_label(events)
    rows = []
    for gw, payload in sorted(event_payloads.items()):
        stats_by_id = {item["id"]: item.get("stats", {}) for item in payload.get("elements", [])}
        for player in players.itertuples(index=False):
            stats = stats_by_id.get(player.id, {})
            fixture = context[(context.gw.eq(gw)) & context.team_name.eq(player.team_name)]
            ctx = fixture.iloc[0].to_dict() if not fixture.empty else {}
            row = {"season": season, "gw": gw, "season_player_id": f"{season}_{player.id}",
                   "player_id": player.id, "player_name": player.player, "team": player.team_name,
                   "position": player.position, "price": player.price}
            for source in PLAYER_METRICS:
                row[source] = pd.to_numeric(stats.get(source), errors="coerce")
            row.update({"fixture_count": int(ctx.get("fixture_count", 0)),
                        "home_fixture_count": int(ctx.get("home_fixture_count", 0)),
                        "away_fixture_count": int(ctx.get("away_fixture_count", 0)),
                        "avg_fixture_difficulty": ctx.get("avg_fixture_difficulty", np.nan),
                        "min_fixture_difficulty": ctx.get("min_fixture_difficulty", np.nan),
                        "max_fixture_difficulty": ctx.get("max_fixture_difficulty", np.nan),
                        "opponent": ctx.get("opponent", np.nan), "is_blank": not bool(ctx),
                        "is_home": bool(ctx.get("home_fixture_count", 0)) if ctx and not ctx.get("away_fixture_count") else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def build_live_features(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]],
                        event_payloads: dict[int, dict[str, Any]], target_gw: int) -> pd.DataFrame:
    """Append empty target rows, then reuse Phase 2 shift-before-roll definitions."""
    players, events, teams = load_players(bootstrap), load_events(bootstrap), load_teams(bootstrap)
    history = normalize_live_history(bootstrap, fixtures, event_payloads)
    season = current_season_label(events)
    context = fixture_context(fixtures, teams)
    targets = []
    for player in players.itertuples(index=False):
        fixture = context[(context.gw.eq(target_gw)) & context.team_name.eq(player.team_name)]
        ctx = fixture.iloc[0].to_dict() if not fixture.empty else {}
        row = {"season": season, "gw": target_gw, "season_player_id": f"{season}_{player.id}",
               "player_id": player.id, "player_name": player.player, "team": player.team_name,
               "position": player.position, "price": player.price, "fixture_count": int(ctx.get("fixture_count", 0)),
               "home_fixture_count": int(ctx.get("home_fixture_count", 0)), "away_fixture_count": int(ctx.get("away_fixture_count", 0)),
               "avg_fixture_difficulty": ctx.get("avg_fixture_difficulty", np.nan), "min_fixture_difficulty": ctx.get("min_fixture_difficulty", np.nan),
               "max_fixture_difficulty": ctx.get("max_fixture_difficulty", np.nan), "opponent": ctx.get("opponent", np.nan),
               "is_blank": not bool(ctx), "is_home": bool(ctx.get("home_fixture_count", 0)) if ctx and not ctx.get("away_fixture_count") else np.nan}
        row.update({metric: np.nan for metric in PLAYER_METRICS})
        targets.append(row)
    combined = pd.concat([history, pd.DataFrame(targets)], ignore_index=True)
    featured = add_rolling_features(combined)
    target = featured[featured.gw.eq(target_gw)].copy()
    raw_fixtures = pd.DataFrame(fixtures)
    matches = build_team_match_rows(raw_fixtures, teams, season)
    if not matches.empty:
        target = join_team_strength(target, calculate_team_strength(matches, through_gw=target_gw))
    target["expected_minutes_proxy"] = expected_minutes_proxy(target)
    details = players.rename(columns={"id": "player_id"})
    live_columns = ["player_id", "player", "web_name", "status", "chance_of_playing_next_round", "news", "news_added", "selected_by_percent", "form", "points_per_game"]
    return add_missingness_indicators(target.merge(details[live_columns], on="player_id", how="left", validate="one_to_one"))
