import json

import pandas as pd
import pytest

from fpl_predictor.squad_update import update_manual_squad


def _players_and_file(tmp_path):
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["DEF", "MID", "FWD"]
    players = pd.DataFrame({"id": range(1, 19), "player": [f"P{i}" for i in range(1, 19)],
                            "web_name": [f"P{i}" for i in range(1, 19)], "position": positions,
                            "team_name": [f"T{(i - 1) // 3}" for i in range(1, 19)], "price": [5.] * 18})
    entries = [{"player_id": i, "multiplier": 1, "bench_position": 0,
                "is_captain": i == 13, "is_vice_captain": i == 9,
                "purchase_price": None, "selling_price": None} for i in range(1, 16)]
    path = tmp_path / "manual.json"; path.write_text(json.dumps({"players": entries, "bank": None, "free_transfers": None}))
    return players, path


def test_manual_squad_update_preserves_legality_and_creates_backup(tmp_path):
    players, path = _players_and_file(tmp_path)
    backup, squad = update_manual_squad(path, players, "P3", "P16")
    assert backup.exists() and 3 not in set(squad.player_id) and 16 in set(squad.player_id)
    assert len(squad) == 15 and squad.player_id.nunique() == 15


def test_captain_removal_requires_explicit_reassignment(tmp_path):
    players, path = _players_and_file(tmp_path)
    with pytest.raises(ValueError, match="captain"):
        update_manual_squad(path, players, "P13", "P18", create_backup=False)
