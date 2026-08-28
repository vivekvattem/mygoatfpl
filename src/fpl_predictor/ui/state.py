"""Runtime-only dashboard settings and validation."""

from dataclasses import dataclass
from pathlib import Path

from fpl_predictor.config import LIVE_DATA_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class AppSettings:
    entry_id: int = 8974446
    squad_source: str = "manual_file"
    squad_file: Path = LIVE_DATA_DIR / "manual_squad.json"
    bank: float | None = None
    free_transfers: int | None = None
    horizon: int = 5
    risk_profile: str = "balanced"
    minimum_gain: float = 1.5
    assume_selling_price_current: bool = False
    refresh_ttl: int = 600


def validate_settings(settings: AppSettings) -> None:
    if settings.entry_id <= 0:
        raise ValueError("Entry ID must be a positive integer")
    if settings.squad_source not in {"manual_file", "public_api"}:
        raise ValueError("Squad source must be manual_file or public_api")
    if settings.bank is not None and settings.bank < 0:
        raise ValueError("Bank cannot be negative")
    if settings.free_transfers is not None and settings.free_transfers < 0:
        raise ValueError("Free transfers cannot be negative")
    if settings.horizon not in {1, 3, 5}:
        raise ValueError("Planning horizon must be 1, 3, or 5")
    if settings.risk_profile not in {"safe", "balanced", "aggressive"}:
        raise ValueError("Risk profile must be safe, balanced, or aggressive")
    if settings.minimum_gain < 0 or settings.refresh_ttl < 60:
        raise ValueError("Transfer threshold must be non-negative and TTL at least 60 seconds")


def project_relative_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def transfer_state_label(settings: AppSettings) -> str:
    if settings.bank is None or settings.free_transfers is None:
        return "TRANSFER STATE UNKNOWN"
    return "SCENARIO MODE" if settings.assume_selling_price_current else "STRICT FINANCIAL STATE"


SESSION_DEFAULTS = {
    "entry_id": 8974446,
    "squad_source": "manual_file",
    "squad_file": "data/live/manual_squad.json",
    "bank_known": False,
    "bank": 0.0,
    "free_transfers_known": False,
    "free_transfers": 1,
    "horizon": 5,
    "risk_profile": "balanced",
    "minimum_gain": 1.5,
    "assume_selling_price_current": False,
    "refresh_ttl": 600,
    "chip_wildcard": "unknown",
    "chip_free_hit": "unknown",
    "chip_bench_boost": "unknown",
    "chip_triple_captain": "unknown",
}
