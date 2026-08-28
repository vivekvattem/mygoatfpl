from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from fpl_predictor.ui.data import (
    DashboardBundle, DataStatus, dashboard_summary, data_status,
    load_manual_squad_view, run_pipeline_refresh, transfer_cache_key, transfer_readiness,
)
from fpl_predictor.ui.state import AppSettings
from fpl_predictor.ui.charts import projection_scatter


def _bundle():
    xi = pd.DataFrame({"weighted_xpts_3": [10.0, 12.0], "weighted_xpts_5": [15.0, 17.0]})
    return DashboardBundle(pd.DataFrame(), pd.DataFrame(), xi, pd.DataFrame(), pd.DataFrame(),
                           pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           {"target_gw": 2, "entry_id": 1, "squad_source": "manual_file",
                            "optimized_xi_xpts": 8.5, "captain": "A", "vice_captain": "B",
                            "formation": "3-4-3", "transfer_decision": "ROLL TRANSFER"},
                           {"schema_validation": {"passed": True, "required": 33, "available": 33}}, {},
                           DataStatus(True, False, datetime.now(timezone.utc), "ok"))


def test_dashboard_summary_is_dynamic():
    summary = dashboard_summary(_bundle())
    assert summary["weighted_3gw"] == 22 and summary["weighted_5gw"] == 32
    assert summary["captain"] == "A" and summary["formation"] == "3-4-3"


def test_data_status_marks_missing_and_stale(tmp_path):
    path = tmp_path / "summary.json"
    assert not data_status(path, 300).available
    path.write_text("{}")
    now = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) + timedelta(seconds=301)
    status = data_status(path, 300, now)
    assert status.available and status.stale and "STALE DATA" in status.message


def test_manual_squad_view_uses_declared_ids(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps({"players": [{"player_id": 2, "multiplier": 1}]}))
    predictions = pd.DataFrame({"player_id": [1, 2], "player": ["A", "B"], "team": ["X", "Y"]})
    view = load_manual_squad_view(path, predictions)
    assert view.player.tolist() == ["B"] and view.multiplier.iloc[0] == 1


def test_public_refresh_uses_read_only_entry_report(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("fpl_predictor.ui.data.subprocess.run", fake_run)
    result = run_pipeline_refresh(AppSettings(entry_id=42, squad_source="public_api"))
    assert result.success
    assert observed["command"][-3].endswith("scripts/live_squad_report.py")
    assert observed["command"][-2:] == ["--entry-id", "42"]
    assert "disabled" in result.message


def test_missing_manual_squad_still_refreshes_rankings(monkeypatch, tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("fpl_predictor.ui.data.subprocess.run", fake_run)
    result = run_pipeline_refresh(AppSettings(squad_file=tmp_path / "missing.json"))
    assert result.success
    assert observed["command"][-1].endswith("scripts/live_predictions.py")
    assert "optimization was skipped" in result.message


def test_sparse_minutes_are_safe_for_projection_chart():
    players = pd.DataFrame({"player": ["A", "B"], "team": ["X", "Y"],
                            "position": ["MID", "FWD"], "price": [7.0, 8.0],
                            "weighted_xpts_3": [10.0, 11.0], "weighted_xpts_5": [15.0, 16.0],
                            "expected_minutes_proxy": [float("nan"), 80.0]})
    figure = projection_scatter(players)
    sizes = [size for trace in figure.data for size in trace.marker.size]
    assert all(pd.notna(size) and size >= 0 for size in sizes)


def test_transfer_readiness_explains_unknown_bank_and_free_transfers():
    state = transfer_readiness(AppSettings(bank=None, free_transfers=None), pd.DataFrame())
    assert state.code == "financial_unknown"
    assert "Bank: Unknown" in state.message and "Free transfers: Unknown" in state.message


def test_transfer_readiness_blocks_unknown_selling_prices_unless_scenario_enabled():
    squad = pd.DataFrame({"selling_price": [float("nan")], "player_id": [1]})
    strict = transfer_readiness(AppSettings(bank=0, free_transfers=1), squad)
    scenario = transfer_readiness(AppSettings(bank=0, free_transfers=1, assume_selling_price_current=True), squad)
    assert strict.code == "selling_prices_unknown"
    assert scenario.ready


def test_transfer_cache_key_changes_with_optimizer_inputs(tmp_path):
    squad = tmp_path / "squad.json"; squad.write_text('{"players": []}')
    first = AppSettings(squad_file=squad, bank=0, free_transfers=1, horizon=5)
    changed = AppSettings(squad_file=squad, bank=0.1, free_transfers=1, horizon=5)
    assert transfer_cache_key(first) != transfer_cache_key(changed)
