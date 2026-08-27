"""Public-entry parsing and deliberately conservative squad resolution."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import unicodedata

import pandas as pd

from .api import FPLAPIClient, FPLAPIError


@dataclass(frozen=True)
class EntryInfo:
    entry_id: int
    manager_name: str
    team_name: str
    overall_points: int | None
    overall_rank: int | None
    current_gameweek: int | None
    bank: float | None
    squad_value: float | None
    free_transfers: None = None


@dataclass(frozen=True)
class SquadResolution:
    """A resolved squad with an explicit freshness/source contract.

    Official picks are historical, public snapshots.  They must never be
    presented as a current pre-deadline squad.  A manually supplied file is
    the only supported current-squad source before a deadline.
    """

    source: str  # public_api | manual_file | unavailable
    squad_gameweek: int | None
    squad_kind: str | None  # latest_public_squad | current_pre_deadline_squad
    payload: dict[str, Any] | None
    reason: str | None = None


@dataclass(frozen=True)
class ManualSquad:
    picks: pd.DataFrame
    bank: float | None
    free_transfers: int | None


def parse_entry(payload: dict[str, Any]) -> EntryInfo:
    """Parse only fields publicly returned by the official entry endpoint."""
    if "id" not in payload:
        raise ValueError("Invalid FPL entry response: missing id")
    manager = " ".join(filter(None, [payload.get("player_first_name"), payload.get("player_last_name")]))
    return EntryInfo(int(payload["id"]), manager, str(payload.get("name", "")),
                     payload.get("summary_overall_points"), payload.get("summary_overall_rank"),
                     payload.get("current_event"),
                     payload.get("last_deadline_bank") / 10 if payload.get("last_deadline_bank") is not None else None,
                     payload.get("last_deadline_value") / 10 if payload.get("last_deadline_value") is not None else None)


def parse_picks(payload: dict[str, Any], players: pd.DataFrame) -> pd.DataFrame:
    """Preserve purchase/selling prices and current squad ordering."""
    picks = pd.DataFrame(payload.get("picks", []))
    if picks.empty:
        raise ValueError("FPL picks response contains no squad")
    picks = picks.rename(columns={"element": "player_id"})
    details = players[["id", "player", "position", "team_name", "price"]].rename(
        columns={"id": "player_id", "team_name": "team"}
    )
    picks = picks.merge(details, on="player_id", how="left", validate="many_to_one")
    for source, output in (("purchase_price", "purchase_price"), ("selling_price", "selling_price")):
        picks[output] = pd.to_numeric(picks.get(source), errors="coerce") / 10
    picks["current_price"] = picks["price"]
    picks["bench_position"] = (pd.to_numeric(picks["position_x"] if "position_x" in picks else picks.get("position"), errors="coerce") - 11).clip(lower=0)
    if "position_y" in picks:
        picks["position"] = picks["position_y"]
    return picks


def _history_gameweeks(history: dict[str, Any], completed_gameweeks: list[int]) -> list[int]:
    """Return a small, plausible set of public-picks Gameweeks, newest first."""
    completed = {int(gw) for gw in completed_gameweeks}
    events = {
        int(item["event"])
        for item in history.get("current", [])
        if isinstance(item, dict) and item.get("event") is not None
    }
    return sorted(events & completed, reverse=True)


def resolve_entry_squad(
    client: FPLAPIClient,
    entry_id: int,
    history: dict[str, Any],
    completed_gameweeks: list[int],
) -> SquadResolution:
    """Find the newest available public squad without assuming participation.

    Late-created entries can legitimately have a history row but no picks for
    a past Gameweek.  A 404 is therefore an expected absence, not a crash.
    We make at most one request per history-backed completed Gameweek.
    """
    candidates = _history_gameweeks(history, completed_gameweeks)
    if not candidates:
        return SquadResolution("unavailable", None, None, None,
                               "No completed Gameweeks with public entry history were available.")
    not_found = 0
    for gameweek in candidates:
        try:
            payload = client.get_entry_picks(entry_id, gameweek)
        except FPLAPIError as exc:
            if getattr(exc, "status_code", None) == 404 or "404" in str(exc):
                not_found += 1
                continue
            return SquadResolution("unavailable", None, None, None,
                                   f"Public picks could not be retrieved: {exc}")
        if payload.get("picks"):
            return SquadResolution("public_api", gameweek, "latest_public_squad", payload)
    reason = "No public picks exist for this entry's completed participation Gameweek(s)."
    if not_found:
        reason += " The official API returned no picks for the available historical snapshot(s)."
    return SquadResolution("unavailable", None, None, None, reason)


def _normalise_name(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").casefold().split()
    )


def resolve_player_id(value: int | str, players: pd.DataFrame,
                      position: str | None = None, team: str | None = None) -> int:
    """Resolve an exact current player ID/full name/unique web name."""
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        player_id = int(value)
        if player_id not in set(players["id"].astype(int)):
            raise ValueError(f"Unknown current player_id: {player_id}")
        return player_id
    wanted = _normalise_name(str(value))
    pool = players.copy()
    if position is not None:
        pool = pool[pool.position.eq(position)]
    if team is not None:
        pool = pool[pool.team_name.map(_normalise_name).eq(_normalise_name(team))]
    full = pool[pool["player"].map(_normalise_name).eq(wanted)]
    matches = full if len(full) else pool[pool["web_name"].map(_normalise_name).eq(wanted)]
    if len(matches) == 1:
        return int(matches.iloc[0]["id"])
    if matches.empty:
        raise ValueError(f"Current player was not found: {value!r}")
    candidates = ", ".join(f"{row.id}: {row.player} ({row.position}, {row.team_name})"
                           for row in matches[["id", "player", "position", "team_name"]].itertuples(index=False))
    raise ValueError(f"Player name is ambiguous: {value!r}. Candidates: {candidates}")


def _manual_player_id(item: dict[str, Any], players: pd.DataFrame) -> int:
    if item.get("player_id") is not None:
        try:
            return resolve_player_id(int(item["player_id"]), players)
        except ValueError as exc:
            raise ValueError(f"Manual squad contains {str(exc).lower()}") from exc
    name = item.get("player")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Each manual squad player needs a valid player_id or player name")
    try:
        return resolve_player_id(name, players, item.get("position"), item.get("team"))
    except ValueError as exc:
        raise ValueError(f"Manual squad {exc}") from exc


def load_manual_squad(path: str | Path, players: pd.DataFrame) -> ManualSquad:
    """Load a strict current-squad declaration; no fuzzy player matching."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read manual squad file: {exc}") from exc
    entries = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Manual squad JSON must contain a 'players' list")
    if len(entries) != 15:
        raise ValueError(f"Manual squad must contain exactly 15 players (received {len(entries)})")
    rows: list[dict[str, Any]] = []
    for order, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each manual squad player must be a JSON object")
        row = dict(item)
        row["player_id"] = _manual_player_id(item, players)
        row["position_order"] = order
        rows.append(row)
    picks = pd.DataFrame(rows)
    if picks.player_id.duplicated().any():
        raise ValueError("Manual squad contains duplicate player IDs")
    details = players[["id", "player", "web_name", "position", "team_name", "price"]].rename(
        columns={"id": "player_id", "team_name": "team"}
    )
    picks = picks.merge(details, on="player_id", how="left", validate="one_to_one")
    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    actual = picks.position.value_counts().to_dict()
    if any(actual.get(position, 0) != count for position, count in required.items()):
        raise ValueError(f"Manual squad must have positions {required}; received {actual}")
    clubs = picks.team.value_counts()
    if (clubs > 3).any():
        raise ValueError("Manual squad cannot contain more than three players from one club")
    for flag in ("is_captain", "is_vice_captain"):
        if flag not in picks:
            picks[flag] = False
        picks[flag] = picks[flag].astype("boolean").fillna(False).astype(bool)
        if picks[flag].sum() > 1:
            raise ValueError(f"Manual squad can contain at most one {flag}")
    if bool((picks.is_captain & picks.is_vice_captain).any()):
        raise ValueError("Manual squad captain and vice-captain must be different players")
    for field in ("purchase_price", "selling_price"):
        picks[field] = pd.to_numeric(picks[field], errors="coerce") if field in picks else float("nan")
    picks["current_price"] = picks["price"]
    picks["multiplier"] = pd.to_numeric(picks["multiplier"], errors="coerce").fillna(1) if "multiplier" in picks else 1
    picks["bench_position"] = pd.to_numeric(picks["bench_position"], errors="coerce").fillna(0) if "bench_position" in picks else 0
    bank = pd.to_numeric(payload.get("bank"), errors="coerce")
    transfers = pd.to_numeric(payload.get("free_transfers"), errors="coerce")
    return ManualSquad(picks, None if pd.isna(bank) else float(bank), None if pd.isna(transfers) else int(transfers))
