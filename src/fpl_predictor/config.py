"""Project-wide configuration."""

from pathlib import Path

BASE_API_URL = "https://fantasy.premierleague.com/api"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_ARCHIVE_DIR = RAW_DATA_DIR / "archive"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
REQUEST_TIMEOUT = 30.0


def ensure_data_directories() -> None:
    """Create all runtime data directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        RAW_ARCHIVE_DIR,
        PROCESSED_DATA_DIR,
        HISTORICAL_DATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
