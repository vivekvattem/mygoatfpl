import pandas as pd

from fpl_predictor.dataset import chronological_split, create_ml_dataset


def test_target_separation_and_baseline():
    frame = pd.DataFrame({
        "season": ["2024-25"], "gw": [2], "season_player_id": ["2024-25_1"],
        "player_id": [1], "player_name": ["A"], "team": ["X"], "position": ["MID"],
        "price": [5.5], "is_home": [True], "fixture_count": [1],
        "home_fixture_count": [1], "away_fixture_count": [0],
        "avg_fixture_difficulty": [2], "min_fixture_difficulty": [2],
        "max_fixture_difficulty": [2], "is_blank": [False],
        "did_not_play_because_team_blank": [False], "points_last_3": [7.0],
        "avg_points_last_3": [3.5], "total_points": [10], "minutes": [90],
    })
    dataset, columns = create_ml_dataset(frame)
    assert dataset.loc[0, "target_points"] == 10
    assert dataset.loc[0, "target_minutes"] == 90
    assert dataset.loc[0, "predicted_points_baseline"] == 3.5
    assert "total_points" not in columns.features
    assert "minutes" not in columns.features


def test_chronological_split_by_season():
    frame = pd.DataFrame({"season": ["2022-23", "2023-24", "2024-25", "2025-26"]})
    train, validation, test = chronological_split(frame, ["2024-25"], ["2025-26"])
    assert train["season"].tolist() == ["2022-23", "2023-24"]
    assert validation["season"].tolist() == ["2024-25"]
    assert test["season"].tolist() == ["2025-26"]
