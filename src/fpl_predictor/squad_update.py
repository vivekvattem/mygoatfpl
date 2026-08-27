"""Safe updates to the local pre-deadline manual squad snapshot."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from .entry import load_manual_squad, resolve_player_id


def update_manual_squad(path: str | Path, current_players: pd.DataFrame,
                        player_out: int | str, player_in: int | str,
                        captain: int | str | None = None,
                        vice_captain: int | str | None = None,
                        purchase_price: float | None = None,
                        selling_price: float | None = None,
                        bank: float | None = None,
                        create_backup: bool = True) -> tuple[Path | None, pd.DataFrame]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    old_id = resolve_player_id(player_out, current_players)
    new_id = resolve_player_id(player_in, current_players)
    entries = payload.get("players", [])
    outgoing = next((item for item in entries if int(item.get("player_id", -1)) == old_id), None)
    if outgoing is None:
        raise ValueError("Outgoing player is not in the current manual squad")
    if any(int(item.get("player_id", -1)) == new_id for item in entries):
        raise ValueError("Incoming player is already in the current squad")
    old_position = current_players.loc[current_players.id.eq(old_id), "position"].iloc[0]
    new_position = current_players.loc[current_players.id.eq(new_id), "position"].iloc[0]
    if old_position != new_position:
        raise ValueError(f"Replacement must preserve position ({old_position} required)")
    if outgoing.get("is_captain") and captain is None:
        raise ValueError("Outgoing player is captain; explicitly supply --captain")
    if outgoing.get("is_vice_captain") and vice_captain is None:
        raise ValueError("Outgoing player is vice-captain; explicitly supply --vice-captain")
    replacement = {key: value for key, value in outgoing.items()
                   if key not in {"player", "is_captain", "is_vice_captain"}}
    replacement["player_id"] = new_id
    replacement["purchase_price"] = purchase_price
    replacement["selling_price"] = selling_price
    entries[entries.index(outgoing)] = replacement
    captain_id = resolve_player_id(captain, current_players) if captain is not None else None
    vice_id = resolve_player_id(vice_captain, current_players) if vice_captain is not None else None
    if captain_id is not None and captain_id not in {int(item["player_id"]) for item in entries}:
        raise ValueError("Captain reassignment must name a player in the updated squad")
    if vice_id is not None and vice_id not in {int(item["player_id"]) for item in entries}:
        raise ValueError("Vice-captain reassignment must name a player in the updated squad")
    if captain_id is not None:
        for item in entries:
            item["is_captain"] = int(item["player_id"]) == captain_id
    if vice_id is not None:
        for item in entries:
            item["is_vice_captain"] = int(item["player_id"]) == vice_id
    if bank is not None:
        payload["bank"] = float(bank)
    temporary = source.with_suffix(".validate.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        validated = load_manual_squad(temporary, current_players).picks
    finally:
        temporary.unlink(missing_ok=True)
    backup = None
    if create_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = source.with_name(f"{source.stem}_{stamp}.json")
        shutil.copy2(source, backup)
    source.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return backup, validated
