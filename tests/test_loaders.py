import numpy as np

from fpl_predictor.loaders import load_players


def bootstrap(element):
    return {
        "elements": [element],
        "teams": [{"id": 8, "name": "Example FC"}],
        "element_types": [{"id": 3, "singular_name_short": "MID"}],
    }


def test_price_team_position_and_numeric_conversion():
    player = load_players(bootstrap({
        "id": 1, "first_name": "Ada", "second_name": "Lovelace",
        "team": 8, "element_type": 3, "now_cost": 75,
        "expected_goals": "2.25", "form": "bad-data",
    })).iloc[0]
    assert player["price"] == 7.5
    assert player["team_name"] == "Example FC"
    assert player["position"] == "MID"
    assert player["expected_goals"] == 2.25
    assert np.isnan(player["form"])
    assert player["player"] == "Ada Lovelace"


def test_missing_optional_columns_do_not_crash():
    result = load_players(bootstrap({"id": 1, "team": 8, "element_type": 3}))
    assert len(result) == 1
    assert "expected_assists" in result.columns
    assert np.isnan(result.iloc[0]["expected_assists"])
