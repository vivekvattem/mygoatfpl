"""Read-only dashboard data access and explicit pipeline refresh boundary."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from fpl_predictor.config import HISTORICAL_ML_DIR, LIVE_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from fpl_predictor.fixtures import get_next_gameweek, normalize_fixtures
from fpl_predictor.fixture_calendar import build_fixture_calendar, team_fixture_signals
from fpl_predictor.live_features import current_season_label
from fpl_predictor.loaders import load_events
from fpl_predictor.team_strength import build_team_match_rows, calculate_team_strength
from fpl_predictor.signals import add_player_signals
from fpl_predictor.ui.state import AppSettings, project_relative_path, validate_settings


@dataclass(frozen=True)
class DataStatus:
    available: bool
    stale: bool
    timestamp: datetime | None
    message: str


@dataclass
class DashboardBundle:
    predictions: pd.DataFrame
    squad: pd.DataFrame
    optimized_xi: pd.DataFrame
    one_transfers: pd.DataFrame
    two_transfers: pd.DataFrame
    replacements: pd.DataFrame
    fixtures: pd.DataFrame
    model_results: pd.DataFrame
    calibration: pd.DataFrame
    residuals: pd.DataFrame
    decision_summary: dict[str, Any]
    live_summary: dict[str, Any]
    phase4_summary: dict[str, Any]
    status: DataStatus
    fixture_calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_fixture_signals: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class RefreshResult:
    success: bool
    message: str
    completed_at: datetime


@dataclass(frozen=True)
class TransferReadiness:
    """Explicit UI state before the expensive Phase 6 optimizer is invoked."""

    code: str
    heading: str
    message: str

    @property
    def ready(self) -> bool:
        return self.code == "ready"


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def data_status(path: Path, ttl_seconds: int, now: datetime | None = None) -> DataStatus:
    if not path.exists():
        return DataStatus(False, True, None, "No successful live refresh is available yet.")
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = ((now or datetime.now(timezone.utc)) - timestamp).total_seconds()
    stale = age > ttl_seconds
    return DataStatus(True, stale, timestamp,
                      "STALE DATA — showing the last successful refresh." if stale else "Live cache is current.")


def load_fixture_table() -> pd.DataFrame:
    fixture_path, bootstrap_path = RAW_DATA_DIR / "fixtures.json", RAW_DATA_DIR / "bootstrap_static.json"
    if not fixture_path.exists() or not bootstrap_path.exists():
        return pd.DataFrame()
    try:
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    frame = normalize_fixtures(fixtures)
    teams = pd.DataFrame(bootstrap.get("teams", []))
    if frame.empty or teams.empty:
        return frame
    names = teams.set_index("id")["name"].to_dict()
    frame["team_name"] = frame.team.map(names)
    frame["opponent_name"] = frame.opponent.map(names)
    frame["venue"] = frame.is_home.map({True: "H", False: "A"})
    raw_fixtures = pd.DataFrame(fixtures)
    target_gw = get_next_gameweek(load_events(bootstrap))
    matches = build_team_match_rows(raw_fixtures, teams, current_season_label(load_events(bootstrap)))
    ratings = calculate_team_strength(matches, through_gw=target_gw) if target_gw else pd.DataFrame()
    if not ratings.empty:
        current = ratings[ratings.gw.eq(target_gw)].set_index("team")
        strength = current["team_defense_strength_5_rel"].to_dict()
        frame["model_opponent_strength"] = frame.opponent_name.map(strength)
    return frame


def load_confirmed_fixture_calendar(target_gw: int | None, horizon: int = 10) -> pd.DataFrame:
    fixture_path, bootstrap_path = RAW_DATA_DIR / "fixtures.json", RAW_DATA_DIR / "bootstrap_static.json"
    if target_gw is None or not fixture_path.exists() or not bootstrap_path.exists():
        return pd.DataFrame()
    try:
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    teams = pd.DataFrame(bootstrap.get("teams", []))
    return build_fixture_calendar(fixtures, teams, int(target_gw), horizon) if not teams.empty else pd.DataFrame()


def load_manual_squad_view(path: str | Path, predictions: pd.DataFrame) -> pd.DataFrame:
    source = project_relative_path(path)
    payload = _json(source)
    entries = payload.get("players")
    if not isinstance(entries, list) or predictions.empty:
        return pd.DataFrame()
    picks = pd.DataFrame(entries)
    if picks.empty or "player_id" not in picks:
        return pd.DataFrame()
    identity = [column for column in predictions if column != "player_id"]
    return picks.merge(predictions[["player_id", *identity]], on="player_id", how="left", validate="one_to_one")


def transfer_readiness(settings: AppSettings, squad: pd.DataFrame) -> TransferReadiness:
    """Describe why transfer analysis can or cannot safely run."""
    unknown = []
    if settings.bank is None:
        unknown.append("Bank: Unknown")
    if settings.free_transfers is None:
        unknown.append("Free transfers: Unknown")
    if unknown:
        return TransferReadiness(
            "financial_unknown", "TRANSFER ANALYSIS PAUSED",
            "\n".join([*unknown, "", "Enter these values in Settings to enable legal transfer optimization."]),
        )
    if squad.empty:
        return TransferReadiness("squad_unavailable", "SQUAD UNAVAILABLE",
                                 "Load a valid manual squad and current live predictions before transfer analysis.")
    selling = pd.to_numeric(squad.get("selling_price"), errors="coerce") if "selling_price" in squad else pd.Series(dtype=float)
    if selling.empty or selling.isna().any():
        if not settings.assume_selling_price_current:
            return TransferReadiness(
                "selling_prices_unknown", "SELLING PRICES UNKNOWN",
                "Authoritative transfer advice is unavailable because current selling prices are not public for this "
                "pre-deadline manual squad.\n\nYou can either:\n1. enter selling prices manually, or\n"
                "2. enable Scenario Mode to temporarily assume current price = selling price.",
            )
    return TransferReadiness("ready", "TRANSFER ANALYSIS READY",
                             "Scenario estimates use current price = selling price." if settings.assume_selling_price_current
                             else "Authoritative financial inputs are available.")


def transfer_cache_key(settings: AppSettings) -> str:
    """Stable cache discriminator for a transfer result, including live-output freshness."""
    squad_path = project_relative_path(settings.squad_file)
    live_path = LIVE_DATA_DIR / "player_predictions.csv"
    def fingerprint(path: Path) -> str:
        if not path.exists():
            return "missing"
        return f"{path.stat().st_mtime_ns}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    payload = {
        "entry_id": settings.entry_id, "squad_source": settings.squad_source,
        "squad": fingerprint(squad_path), "live_predictions": fingerprint(live_path),
        "bank": settings.bank, "free_transfers": settings.free_transfers,
        "horizon": settings.horizon, "risk_profile": settings.risk_profile,
        "minimum_gain": settings.minimum_gain,
        "assume_selling_price_current": settings.assume_selling_price_current,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run_transfer_analysis(settings: AppSettings, timeout: int = 300) -> RefreshResult:
    """Run only the existing optimizer command, on an explicit Transfers-page action."""
    if settings.squad_source != "manual_file":
        return RefreshResult(False, "Transfer analysis requires a current manual squad; public picks are inspection-only.",
                             datetime.now(timezone.utc))
    return run_pipeline_refresh(settings, timeout=timeout)


def load_dashboard_bundle(settings: AppSettings) -> DashboardBundle:
    validate_settings(settings)
    live = _json(LIVE_DATA_DIR / "live_summary.json")
    predictions = _csv(LIVE_DATA_DIR / "player_decision_universe.csv")
    if predictions.empty:
        predictions = _csv(LIVE_DATA_DIR / "player_predictions.csv")
    squad = (load_manual_squad_view(settings.squad_file, predictions)
             if settings.squad_source == "manual_file" else _csv(LIVE_DATA_DIR / "my_squad.csv"))
    decision = _json(LIVE_DATA_DIR / "decision_summary.json")
    # A decision report belongs to one entry/source. Never present an older
    # manual optimization as though it came from a newly selected public squad.
    squad_path = project_relative_path(settings.squad_file)
    decision_path = LIVE_DATA_DIR / "decision_summary.json"
    manual_missing = settings.squad_source == "manual_file" and not squad_path.exists()
    manual_newer = (settings.squad_source == "manual_file" and squad_path.exists()
                    and decision_path.exists() and squad_path.stat().st_mtime > decision_path.stat().st_mtime)
    settings_mismatch = bool(decision) and any((
        decision.get("horizon") != settings.horizon,
        decision.get("risk_profile") != settings.risk_profile,
        decision.get("minimum_gain") != settings.minimum_gain,
        bool(decision.get("financial_state_scenario")) != settings.assume_selling_price_current,
        decision.get("bank") != settings.bank,
        decision.get("free_transfers") != settings.free_transfers,
    ))
    if (decision.get("entry_id") != settings.entry_id
            or decision.get("squad_source") != settings.squad_source
            or manual_missing or manual_newer or settings_mismatch):
        decision = {}
    if settings.squad_source == "public_api" and live.get("squad_source") != "public_api":
        squad = pd.DataFrame()
    one_transfers = _csv(LIVE_DATA_DIR / "transfer_candidates.csv")
    calendar = load_confirmed_fixture_calendar(live.get("target_gw"), 10)
    fixture_signals = (team_fixture_signals(calendar, int(live["target_gw"]), 5)
                       if not calendar.empty and live.get("target_gw") is not None else pd.DataFrame())
    worthwhile_ids: list[int] = []
    net_column = f"net_gain_{settings.horizon}gw"
    if decision and net_column in one_transfers:
        worthwhile_ids = one_transfers.loc[one_transfers[net_column].ge(settings.minimum_gain), "out_id"].tolist()
    if not predictions.empty and not squad.empty:
        predictions["owned"] = predictions.player_id.isin(squad.player_id)
    predictions = add_player_signals(predictions, fixture_signals, worthwhile_ids) if not predictions.empty else predictions
    if not squad.empty and not predictions.empty:
        signal_columns = [column for column in predictions if column.endswith("_signal") or column in
                          {"player_id", "action", "signal_reason", "risk_reason", "fixture_reason",
                           "fixtures_next_5", "average_fdr_5"}]
        existing = [column for column in signal_columns if column != "player_id" and column in squad]
        squad = squad.drop(columns=existing).merge(predictions[signal_columns], on="player_id", how="left")
    phase4 = _json(HISTORICAL_ML_DIR / "phase4_summary.json")
    status_path = LIVE_DATA_DIR / ("decision_summary.json" if decision else "live_summary.json")
    return DashboardBundle(
        predictions=predictions,
        squad=squad,
        optimized_xi=_csv(LIVE_DATA_DIR / "optimized_xi.csv"),
        one_transfers=one_transfers,
        two_transfers=_csv(LIVE_DATA_DIR / "two_transfer_candidates.csv"),
        replacements=_csv(LIVE_DATA_DIR / "replacement_shortlists.csv"),
        fixtures=load_fixture_table(),
        model_results=_csv(HISTORICAL_ML_DIR / "model_results.csv"),
        calibration=_csv(HISTORICAL_ML_DIR / "calibration_results.csv"),
        residuals=_csv(HISTORICAL_ML_DIR / "residual_analysis.csv"),
        decision_summary=decision,
        live_summary=live,
        phase4_summary=phase4,
        status=data_status(status_path, settings.refresh_ttl),
        fixture_calendar=calendar,
        team_fixture_signals=fixture_signals,
    )


def dashboard_summary(bundle: DashboardBundle) -> dict[str, Any]:
    summary = bundle.decision_summary
    xi = bundle.optimized_xi
    return {
        "target_gw": summary.get("target_gw", bundle.live_summary.get("target_gw")),
        "entry_id": summary.get("entry_id", bundle.live_summary.get("entry_id")),
        "squad_source": summary.get("squad_source", bundle.live_summary.get("squad_source", "unavailable")),
        "projected_xi": summary.get("optimized_xi_xpts"),
        "weighted_3gw": float(xi.weighted_xpts_3.sum()) if "weighted_xpts_3" in xi else None,
        "weighted_5gw": float(xi.weighted_xpts_5.sum()) if "weighted_xpts_5" in xi else None,
        "captain": summary.get("captain"),
        "vice_captain": summary.get("vice_captain"),
        "formation": summary.get("formation"),
        "transfer_decision": summary.get("transfer_decision", "UNAVAILABLE"),
        "schema": bundle.live_summary.get("schema_validation", {}),
    }


def run_pipeline_refresh(settings: AppSettings, timeout: int = 300) -> RefreshResult:
    """Invoke the existing validated CLI boundary; no optimizer logic lives here."""
    validate_settings(settings)
    squad_path = project_relative_path(settings.squad_file)
    if settings.squad_source == "public_api":
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "live_squad_report.py"),
                   "--entry-id", str(settings.entry_id)]
        success_message = "Official FPL data, predictions, and the latest public squad snapshot refreshed. Optimization remains disabled for public historical picks."
    elif not squad_path.exists():
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "live_predictions.py")]
        success_message = f"Player predictions refreshed. Personalized optimization was skipped because the manual squad file was not found: {squad_path}"
    else:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "optimize_squad.py"),
                   "--entry-id", str(settings.entry_id), "--squad-file", str(squad_path),
                   "--horizon", str(settings.horizon), "--risk-profile", settings.risk_profile,
                   "--minimum-gain", str(settings.minimum_gain)]
        if settings.bank is not None:
            command.extend(["--bank", str(settings.bank)])
        if settings.free_transfers is not None:
            command.extend(["--free-transfers", str(settings.free_transfers)])
        if settings.assume_selling_price_current:
            command.append("--assume-selling-price-current")
        success_message = "Official FPL data, predictions, and personalized decisions refreshed successfully."
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RefreshResult(False, f"Refresh could not complete: {exc}", datetime.now(timezone.utc))
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "Unknown pipeline error").strip().splitlines()[-1]
        return RefreshResult(False, detail, datetime.now(timezone.utc))
    return RefreshResult(True, success_message, datetime.now(timezone.utc))
