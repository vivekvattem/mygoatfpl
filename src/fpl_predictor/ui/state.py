"""Runtime-only dashboard settings and validation."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from fpl_predictor.config import LIVE_DATA_DIR, PROJECT_ROOT
from fpl_predictor.entry import load_manual_squad


@dataclass(frozen=True)
class AppSettings:
    entry_id: int = 8974446
    squad_source: str = "manual_file"
    squad_file: Path | None = LIVE_DATA_DIR / "manual_squad.json"
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


def project_relative_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def get_active_squad_file(state: Mapping[str, Any]) -> Path | None:
    """Resolve a squad path without mutating any widget-owned session key.

    An uploaded runtime file is session-scoped and wins over an explicit path.  If
    neither is available, use the bundled default only when it actually exists.
    """
    uploaded = project_relative_path(state.get("active_squad_file"))
    if uploaded is not None and uploaded.exists():
        return uploaded
    configured = str(state.get("squad_file_path_input", "")).strip()
    if configured:
        return project_relative_path(configured)
    default = LIVE_DATA_DIR / "manual_squad.json"
    return default if default.exists() else None


def active_squad_source(state: Mapping[str, Any]) -> str:
    """Return an accessible source label without exposing host filesystem paths."""
    uploaded = project_relative_path(state.get("active_squad_file"))
    if uploaded is not None and uploaded.exists():
        return "Uploaded squad for this session"
    if str(state.get("squad_file_path_input", "")).strip():
        return "Configured squad file"
    return "Bundled manual squad" if (LIVE_DATA_DIR / "manual_squad.json").exists() else "No active manual squad"


def write_uploaded_squad_to_runtime(contents: bytes, players, runtime_path: Path | None = None) -> Path:
    """Validate an upload before atomically replacing the session runtime squad file."""
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Manual squad JSON is invalid: {exc}") from exc
    target = runtime_path or LIVE_DATA_DIR / "session_manual_squad.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f"{target.stem}.pending{target.suffix}")
    try:
        pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        load_manual_squad(pending, players)
        pending.replace(target)
    except (OSError, ValueError):
        pending.unlink(missing_ok=True)
        raise
    return target


def initialize_runtime_session(state: dict[str, Any]) -> str:
    """Give each browser session a private upload namespace."""
    token = str(state.get("runtime_session_id") or "").strip()
    if not token:
        token = uuid4().hex
        state["runtime_session_id"] = token
    return token


def runtime_squad_path(state: dict[str, Any]) -> Path:
    """Return a session-isolated path in Cloud's ephemeral runtime filesystem."""
    return LIVE_DATA_DIR / "runtime" / initialize_runtime_session(state) / "manual_squad.json"


def activate_uploaded_squad(state: dict[str, Any], runtime_path: Path) -> None:
    """Set only internal runtime state; never write a Streamlit widget key."""
    state["active_squad_file"] = str(runtime_path)


def transfer_state_label(settings: AppSettings) -> str:
    if settings.bank is None or settings.free_transfers is None:
        return "TRANSFER STATE UNKNOWN"
    return "SCENARIO MODE" if settings.assume_selling_price_current else "STRICT FINANCIAL STATE"


SESSION_DEFAULTS = {
    "entry_id": 8974446,
    "squad_source": "manual_file",
    "squad_file_path_input": "",
    "active_squad_file": None,
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
    # Widget-owned reliability controls. Runtime refresh code never assigns to
    # these keys after the widgets have been constructed.
    "widget_auto_refresh": True,
    "widget_refresh_interval_minutes": 10,
    "widget_show_reliability_alerts": True,
    "widget_show_player_change_alerts": True,
    # Internal session-only monitoring state.
    "runtime_last_refresh_check": None,
    "runtime_last_refresh_success": None,
    "runtime_last_refresh_failure": None,
    "runtime_next_refresh": None,
    "runtime_live_fingerprint": None,
    "runtime_last_change": None,
    "runtime_event_log": [],
    "runtime_refresh_latencies": {},
    "runtime_analyst_metrics": {},
    "live_generation": 0,
    "fixture_generation": 0,
    "personalized_generation": 0,
    "analyst_generation": 0,
}
