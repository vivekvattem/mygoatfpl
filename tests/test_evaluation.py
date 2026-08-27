import pandas as pd

from fpl_predictor.evaluation import (
    gameweek_metrics, high_ceiling_metrics, ndcg_metrics, position_metrics, regression_metrics, top_k_metrics,
    top_pick_evaluation,
)


def evaluation_frame():
    return pd.DataFrame({
        "season": ["2025-26"] * 6, "gw": [1, 1, 1, 2, 2, 2],
        "season_player_id": [f"p{i}" for i in range(6)],
        "position": ["GK", "DEF", "MID", "GK", "DEF", "FWD"],
        "target_points": [10, 5, 0, 1, 8, 4],
        "prediction": [9, 1, 2, 2, 3, 10],
    })


def test_regression_and_position_metrics():
    frame = evaluation_frame()
    metrics = regression_metrics(frame.target_points, frame.prediction)
    assert metrics["sample_count"] == 6
    assert metrics["mae"] == 19 / 6
    positions = position_metrics(frame)
    assert set(positions.position) == {"GK", "DEF", "MID", "FWD"}


def test_top_k_is_averaged_by_gameweek():
    metrics = top_k_metrics(evaluation_frame(), (1, 2))
    assert metrics["top_1_precision"] == .5
    assert metrics["top_1_recall"] == .5
    assert metrics["top_2_precision"] == .75


def test_top_pick_regret_and_gameweek_rows():
    result = top_pick_evaluation(evaluation_frame())
    assert result["average_top_pick_points"] == 7
    assert result["average_actual_maximum"] == 9
    assert result["average_regret"] == 2
    assert len(gameweek_metrics(evaluation_frame(), "Test", "test")) == 2


def test_ndcg_and_high_ceiling_metrics():
    perfect = evaluation_frame().assign(prediction=lambda x: x.target_points)
    ndcg = ndcg_metrics(perfect, (10,))
    assert ndcg["ndcg_10"] == 1.0
    ceiling = high_ceiling_metrics(perfect, (8, 10, 15))
    assert ceiling["ceiling_8_precision"] == 1.0
    assert ceiling["ceiling_8_recall"] == 1.0
    assert ceiling["ceiling_15_sample_count"] == 0
