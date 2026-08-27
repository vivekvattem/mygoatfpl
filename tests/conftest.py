import pandas as pd
import pytest


@pytest.fixture
def legal_squad():
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    teams = ["A", "B", "A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "C", "D", "E"]
    points = [4, 2, 8, 7, 6, 5, 1, 9, 8, 7, 6, 1, 10, 9, 2]
    frame = pd.DataFrame({"player_id": range(1, 16), "player": [f"P{i}" for i in range(1, 16)],
                          "position": positions, "team": teams, "price": [5.0] * 15,
                          "current_price": [5.0] * 15, "selling_price": [5.0] * 15,
                          "availability_adjusted_xpts": points, "raw_xpts": points,
                          "xpts_lower": [max(0, x - 2) for x in points], "xpts_upper": [x + 2 for x in points],
                          "expected_minutes_proxy": [90] * 15, "availability": ["available"] * 15,
                          "xGI_per90_last_3": [0.2] * 15, "ict_index_last_3": [5] * 15,
                          "avg_fixture_difficulty": [3] * 15, "selected_by_percent": [10] * 15})
    for horizon in (1, 3, 5):
        frame[f"weighted_xpts_{horizon}"] = frame.availability_adjusted_xpts * horizon
    return frame


@pytest.fixture
def player_universe(legal_squad):
    extras = legal_squad.iloc[[0, 2, 7, 12]].copy()
    extras["player_id"] = [16, 17, 18, 19]
    extras["player"] = ["New GK", "New DEF", "New MID", "New FWD"]
    extras["team"] = "F"; extras["price"] = [5.0, 5.0, 5.0, 5.0]
    extras["availability_adjusted_xpts"] = [8, 12, 13, 14]
    for horizon in (1, 3, 5):
        extras[f"weighted_xpts_{horizon}"] = extras.availability_adjusted_xpts * horizon
    return pd.concat([legal_squad, extras], ignore_index=True)
