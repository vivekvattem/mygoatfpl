import json

import pandas as pd
import pytest

from fpl_predictor.api import FPLAPIError
from fpl_predictor.entry import load_manual_squad, parse_entry, parse_picks, resolve_entry_squad


def test_official_entry_and_squad_parsing_preserves_prices():
    entry = parse_entry({"id": 12, "player_first_name": "A", "player_last_name": "B", "name": "XI",
                         "summary_overall_points": 100, "summary_overall_rank": 50, "current_event": 3,
                         "last_deadline_bank": 15, "last_deadline_value": 1005})
    assert entry.manager_name == "A B" and entry.bank == 1.5 and entry.squad_value == 100.5
    assert entry.free_transfers is None
    players = pd.DataFrame({"id": [7], "player": ["Player"], "position": ["MID"], "team_name": ["Club"], "price": [6.5]})
    picks = parse_picks({"picks": [{"element": 7, "position": 1, "multiplier": 2, "is_captain": True,
                                     "is_vice_captain": False, "purchase_price": 60, "selling_price": 63}]}, players)
    assert picks.iloc[0].purchase_price == 6 and picks.iloc[0].selling_price == 6.3


def test_invalid_entry_payload_fails():
    with pytest.raises(ValueError, match="missing id"):
        parse_entry({})


class FakeClient:
    def __init__(self, picks):
        self.picks = picks
        self.calls = []

    def get_entry_picks(self, entry_id, gameweek):
        self.calls.append(gameweek)
        value = self.picks[gameweek]
        if isinstance(value, Exception):
            raise value
        return value


def test_late_created_entry_404_is_cleanly_unavailable():
    client = FakeClient({1: FPLAPIError("picks returned 404", status_code=404)})
    result = resolve_entry_squad(client, 8974446, {"current": [{"event": 1}]}, [1])
    assert result.source == "unavailable" and result.squad_gameweek is None
    assert client.calls == [1]


def test_latest_valid_public_picks_are_explicitly_historical():
    client = FakeClient({3: FPLAPIError("404", status_code=404), 2: {"picks": [{"element": 1}]}})
    result = resolve_entry_squad(client, 12, {"current": [{"event": 2}, {"event": 3}]}, [1, 2, 3])
    assert result.source == "public_api" and result.squad_gameweek == 2
    assert result.squad_kind == "latest_public_squad"


def _manual_players():
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return pd.DataFrame({"id": list(range(1, 16)), "player": [f"Player {i}" for i in range(1, 16)],
                         "web_name": [f"P{i}" for i in range(1, 16)], "position": positions,
                         "team_name": [f"Club{(i - 1) // 3}" for i in range(1, 16)], "price": [5.0] * 15})


def test_manual_squad_is_validated_and_preserves_unknown_prices(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps({"players": [{"player_id": i, "is_captain": i == 3, "is_vice_captain": i == 4}
                                             for i in range(1, 16)], "bank": 1.2, "free_transfers": 2}))
    squad = load_manual_squad(path, _manual_players())
    assert len(squad.picks) == 15 and squad.bank == 1.2 and squad.free_transfers == 2
    assert squad.picks.purchase_price.isna().all() and squad.picks.selling_price.isna().all()


def test_manual_squad_rejects_wrong_shape_and_ambiguous_name(tmp_path):
    players = _manual_players()
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"players": [{"player_id": i} for i in range(1, 15)]}))
    with pytest.raises(ValueError, match="exactly 15"):
        load_manual_squad(bad, players)
    players.loc[1, "web_name"] = "same"; players.loc[2, "web_name"] = "same"
    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps({"players": [{"player_id": i} for i in range(1, 15)] + [{"player": "same"}]}))
    with pytest.raises(ValueError, match="ambiguous"):
        load_manual_squad(ambiguous, players)


def test_manual_squad_rejects_max_three_from_one_club(tmp_path):
    players = _manual_players(); players["team_name"] = "One Club"
    path = tmp_path / "clubs.json"; path.write_text(json.dumps({"players": [{"player_id": i} for i in range(1, 16)]}))
    with pytest.raises(ValueError, match="more than three"):
        load_manual_squad(path, players)


def test_manual_squad_rejects_invalid_position_composition_and_unknown_id(tmp_path):
    players = _manual_players()
    players.loc[0, "position"] = "DEF"
    bad_positions = tmp_path / "positions.json"
    bad_positions.write_text(json.dumps({"players": [{"player_id": i} for i in range(1, 16)]}))
    with pytest.raises(ValueError, match="positions"):
        load_manual_squad(bad_positions, players)
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"players": [{"player_id": i} for i in range(1, 15)] + [{"player_id": 999}]}))
    with pytest.raises(ValueError, match="unknown current player_id"):
        load_manual_squad(unknown, _manual_players())
