"""Project-wide configuration."""

from pathlib import Path

BASE_API_URL = "https://fantasy.premierleague.com/api"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_ARCHIVE_DIR = RAW_DATA_DIR / "archive"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
HISTORICAL_RAW_DIR = HISTORICAL_DATA_DIR / "raw"
HISTORICAL_PROCESSED_DIR = HISTORICAL_DATA_DIR / "processed"
HISTORICAL_ML_DIR = HISTORICAL_DATA_DIR / "ml"
MODEL_DIR = PROJECT_ROOT / "models"
HISTORICAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
HISTORICAL_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
REQUEST_TIMEOUT = 30.0


def ensure_data_directories() -> None:
    """Create all runtime data directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        RAW_ARCHIVE_DIR,
        PROCESSED_DATA_DIR,
        HISTORICAL_DATA_DIR,
        HISTORICAL_RAW_DIR,
        HISTORICAL_PROCESSED_DIR,
        HISTORICAL_ML_DIR,
        MODEL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
