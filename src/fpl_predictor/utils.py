"""File persistence utilities for repeatable data refreshes."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Return a sortable UTC timestamp suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(payload: Any, path: Path) -> None:
    """Write JSON atomically enough for a local single-process pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def save_raw_with_archive(payload: Any, name: str, raw_dir: Path, archive_dir: Path, timestamp: str) -> None:
    """Save a current raw snapshot and an immutable timestamped copy."""
    save_json(payload, raw_dir / f"{name}.json")
    save_json(payload, archive_dir / f"{name}_{timestamp}.json")
