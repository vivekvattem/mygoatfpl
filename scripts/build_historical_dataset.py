#!/usr/bin/env python3
"""Build the Phase 2 leakage-safe historical player-Gameweek dataset."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpl_predictor.config import (  # noqa: E402
    HISTORICAL_ML_DIR, HISTORICAL_PROCESSED_DIR, HISTORICAL_RAW_DIR,
    HISTORICAL_SEASONS, ensure_data_directories,
)
from fpl_predictor.dataset import create_ml_dataset, dataset_summary  # noqa: E402
from fpl_predictor.historical import (  # noqa: E402
    HistoricalDataError, VaastavHistoricalSource, load_and_normalize_season,
)
from fpl_predictor.rolling import (  # noqa: E402
    add_rolling_features, add_team_rolling_features, build_team_gameweeks,
    merge_team_and_opponent_features,
)
from fpl_predictor.validation import validate_dataset  # noqa: E402


def build_historical_dataset(seasons: tuple[str, ...]) -> dict[str, object]:
    """Ingest configured seasons and write normalized and ML-ready outputs."""
    ensure_data_directories()
    source = VaastavHistoricalSource(timeout=120.0)
    player_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []
    successful: list[str] = []
    unavailable: dict[str, str] = {}
    raw_rows = 0

    for season in seasons:
        try:
            paths = source.fetch_season(season, HISTORICAL_RAW_DIR)
            players, fixtures, season_rows = load_and_normalize_season(paths, season)
            teams = pd.read_csv(paths["teams"], low_memory=False)
        except (HistoricalDataError, OSError, ValueError, pd.errors.ParserError) as exc:
            unavailable[season] = str(exc)
            continue
        players.to_csv(HISTORICAL_PROCESSED_DIR / f"player_gameweeks_{season}.csv", index=False)
        player_frames.append(players)
        team_frames.append(build_team_gameweeks(fixtures, teams, season))
        successful.append(season)
        raw_rows += season_rows

    if not player_frames:
        details = "; ".join(f"{season}: {reason}" for season, reason in unavailable.items())
        raise HistoricalDataError(f"No configured historical seasons could be loaded. {details}")

    canonical = pd.concat(player_frames, ignore_index=True)
    player_features = add_rolling_features(canonical)
    team_gws = pd.concat(team_frames, ignore_index=True)
    team_features = add_team_rolling_features(team_gws)
    enriched = merge_team_and_opponent_features(player_features, team_features)
    ml_dataset, columns = create_ml_dataset(enriched)
    report = validate_dataset(ml_dataset, columns.features)
    summary = dataset_summary(ml_dataset, columns)
    summary.update({"raw_row_count": raw_rows, "unavailable_seasons": unavailable})

    dataset_path = HISTORICAL_ML_DIR / "player_gameweek_dataset.csv"
    summary_path = HISTORICAL_ML_DIR / "dataset_summary.json"
    ml_dataset.to_csv(dataset_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    blank_rows = int(ml_dataset["is_blank"].fillna(False).sum())
    double_rows = int(ml_dataset["fixture_count"].fillna(0).gt(1).sum())
    print("FPL HISTORICAL DATASET BUILD\n")
    print("Seasons:")
    for season in successful:
        print(season.replace("-", "/"))
    if unavailable:
        print("\nUnavailable/incompatible seasons:")
        for season, reason in unavailable.items():
            print(f"{season.replace('-', '/')}: {reason}")
    print(f"\nRaw rows: {raw_rows}")
    print(f"Player-GW rows: {len(ml_dataset)}")
    print(f"Unique player-seasons: {ml_dataset['season_player_id'].nunique()}")
    print(f"Features: {len(columns.features)}")
    print(f"\nBlank GW rows: {blank_rows}")
    print(f"Double GW rows: {double_rows}")
    print(f"\nLeakage validation: {'PASSED' if report.leakage_passed else 'FAILED'}")
    print(f"Infinite-value validation: {'PASSED' if report.infinite_values_passed else 'FAILED'}")
    print(f"Duplicate validation: {'PASSED' if report.duplicates_passed else 'FAILED'}")
    print(f"\nSaved:\n{dataset_path.relative_to(PROJECT_ROOT)}")
    print("\nDataset build successful.")
    return {"dataset": ml_dataset, "columns": columns, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=list(HISTORICAL_SEASONS))
    args = parser.parse_args()
    try:
        build_historical_dataset(tuple(args.seasons))
    except HistoricalDataError as exc:
        print(f"Historical dataset build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
