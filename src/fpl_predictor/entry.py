"""Typed parsing of public official FPL entry and picks responses."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


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
    details = players.rename(columns={"id": "player_id", "team_name": "team"})
    picks = picks.merge(details[["player_id", "player", "position", "team", "price"]], on="player_id", how="left", validate="many_to_one")
    for source, output in (("purchase_price", "purchase_price"), ("selling_price", "selling_price")):
        picks[output] = pd.to_numeric(picks.get(source), errors="coerce") / 10
    picks["current_price"] = picks["price"]
    picks["bench_position"] = (pd.to_numeric(picks["position_x"] if "position_x" in picks else picks.get("position"), errors="coerce") - 11).clip(lower=0)
    if "position_y" in picks:
        picks["position"] = picks["position_y"]
    return picks
