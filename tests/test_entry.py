import pandas as pd
import pytest

from fpl_predictor.entry import parse_entry, parse_picks


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
