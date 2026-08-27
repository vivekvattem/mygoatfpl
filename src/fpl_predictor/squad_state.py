"""Typed personalized squad state and FPL legality validation."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pandas as pd

from .entry import load_manual_squad

SQUAD_POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


@dataclass(frozen=True)
class SquadState:
    entry_id: int
    target_gw: int
    squad_source: str
    players: pd.DataFrame
    bank: float | None = None
    free_transfers: int | None = None
    squad_gameweek: int | None = None

    def with_financial_overrides(self, bank: float | None = None,
                                 free_transfers: int | None = None) -> "SquadState":
        return replace(self, bank=self.bank if bank is None else bank,
                       free_transfers=self.free_transfers if free_transfers is None else free_transfers)


def validate_squad(players: pd.DataFrame) -> None:
    if len(players) != 15:
        raise ValueError(f"A legal FPL squad requires 15 players; received {len(players)}")
    if players["player_id"].duplicated().any():
        raise ValueError("Squad contains duplicate player IDs")
    counts = players["position"].value_counts().to_dict()
    if any(counts.get(position, 0) != count for position, count in SQUAD_POSITION_COUNTS.items()):
        raise ValueError(f"Illegal squad position counts: {counts}")
    if (players["team"].value_counts() > 3).any():
        raise ValueError("Squad exceeds the maximum of three players per club")


def validate_squad_freshness(state: SquadState, allow_stale: bool = False) -> None:
    if state.squad_source == "unavailable":
        raise ValueError("Cannot optimize an unavailable squad")
    if (state.squad_source == "public_api" and
            (state.squad_gameweek is None or state.squad_gameweek < state.target_gw) and
            not allow_stale):
        raise ValueError("Public squad snapshot is stale; pass --allow-stale-squad to override explicitly")


def load_state_overrides(path: str | Path | None) -> dict[str, float | int | None]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manual state must be a JSON object")
    result = {key: payload.get(key) for key in ("bank", "free_transfers") if key in payload}
    if result.get("bank") is not None and float(result["bank"]) < 0:
        raise ValueError("Bank cannot be negative")
    if result.get("free_transfers") is not None and int(result["free_transfers"]) < 0:
        raise ValueError("Free transfers cannot be negative")
    return result


def load_manual_squad_state(entry_id: int, target_gw: int, squad_file: str | Path,
                            current_players: pd.DataFrame,
                            state_file: str | Path | None = None,
                            bank: float | None = None,
                            free_transfers: int | None = None) -> SquadState:
    manual = load_manual_squad(squad_file, current_players)
    overrides = load_state_overrides(state_file)
    resolved_bank = bank if bank is not None else overrides.get("bank", manual.bank)
    resolved_ft = free_transfers if free_transfers is not None else overrides.get("free_transfers", manual.free_transfers)
    state = SquadState(entry_id, target_gw, "manual_file", manual.picks,
                       None if resolved_bank is None else float(resolved_bank),
                       None if resolved_ft is None else int(resolved_ft))
    validate_squad(state.players)
    return state


def require_transfer_state(state: SquadState, assume_selling_price_current: bool = False) -> None:
    if state.bank is None:
        raise ValueError("Transfer optimization requires known bank; supply --bank")
    if state.free_transfers is None:
        raise ValueError("Transfer optimization requires known free transfers; supply --free-transfers")
    if not assume_selling_price_current and state.players["selling_price"].isna().any():
        raise ValueError("Transfer optimization requires selling prices; supply them or use --assume-selling-price-current scenario mode")
