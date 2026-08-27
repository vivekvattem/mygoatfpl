#!/usr/bin/env python3
"""Download, transform, archive, and report on current official FPL data."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpl_predictor.api import FPLAPIClient, FPLAPIError  # noqa: E402
from fpl_predictor.config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RAW_ARCHIVE_DIR,
    RAW_DATA_DIR,
    ensure_data_directories,
)
from fpl_predictor.features import build_baseline_features  # noqa: E402
from fpl_predictor.fixtures import (  # noqa: E402
    fixture_difficulty_summary,
    get_current_gameweek,
    get_next_gameweek,
    normalize_fixtures,
)
from fpl_predictor.loaders import load_events, load_players, load_teams  # noqa: E402
from fpl_predictor.utils import save_raw_with_archive, utc_timestamp  # noqa: E402


def refresh_data(client: FPLAPIClient | None = None) -> dict[str, object]:
    """Execute the complete Phase 1 refresh and return a small result summary."""
    ensure_data_directories()
    api = client or FPLAPIClient()
    bootstrap = api.get_bootstrap_static()
    raw_fixtures = api.get_fixtures()

    timestamp = utc_timestamp()
    save_raw_with_archive(bootstrap, "bootstrap_static", RAW_DATA_DIR, RAW_ARCHIVE_DIR, timestamp)
    save_raw_with_archive(raw_fixtures, "fixtures", RAW_DATA_DIR, RAW_ARCHIVE_DIR, timestamp)

    teams = load_teams(bootstrap)
    events = load_events(bootstrap)
    players = build_baseline_features(load_players(bootstrap))
    fixtures = normalize_fixtures(raw_fixtures)
    current_gw = get_current_gameweek(events)
    next_gw = get_next_gameweek(events)
    summary_3 = fixture_difficulty_summary(fixtures, next_gw, horizon=3)
    summary_5 = fixture_difficulty_summary(fixtures, next_gw, horizon=5)

    team_names = teams.set_index("id")["name"].to_dict() if {"id", "name"}.issubset(teams) else {}
    for frame in (fixtures, summary_3, summary_5):
        if "team" in frame:
            frame.insert(frame.columns.get_loc("team") + 1, "team_name", frame["team"].map(team_names))

    players.to_csv(PROCESSED_DATA_DIR / "players.csv", index=False)
    fixtures.to_csv(PROCESSED_DATA_DIR / "fixtures.csv", index=False)
    summary_3.to_csv(PROCESSED_DATA_DIR / "fixture_summary_3gw.csv", index=False)
    summary_5.to_csv(PROCESSED_DATA_DIR / "fixture_summary_5gw.csv", index=False)

    print("FPL DATA REFRESH\n")
    print(f"Players loaded: {len(players)}")
    print(f"Teams loaded: {len(teams)}")
    print(f"Fixtures loaded: {len(raw_fixtures)}")
    print(f"Current Gameweek: {current_gw if current_gw is not None else 'Not active'}")
    print(f"Next Gameweek: {next_gw if next_gw is not None else 'Season complete'}")
    print("\nBest 5 fixture runs over next 5 GWs:")
    for rank, row in enumerate(summary_5.head(5).itertuples(index=False), start=1):
        label = getattr(row, "team_name", None) or f"Team {row.team}"
        print(f"{rank}. {label} — avg FDR {row.avg_fdr:.2f} ({row.fixtures_count} fixtures)")
    if summary_5.empty:
        print("No upcoming fixtures found.")
    print("\nData saved successfully.")
    return {"players": players, "fixtures": fixtures, "next_gw": next_gw}


def main() -> int:
    try:
        refresh_data()
    except FPLAPIError as exc:
        print(f"FPL data refresh failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
