from pathlib import Path
import json

import pytest

from fpl_predictor.config import PROJECT_ROOT
from fpl_predictor.ui.state import (
    AppSettings, activate_uploaded_squad, active_squad_source, get_active_squad_file,
    project_relative_path, runtime_squad_path, transfer_state_label, validate_settings,
    write_uploaded_squad_to_runtime,
)


def test_settings_validation_preserves_unknown_financial_state():
    settings = AppSettings(bank=None, free_transfers=None)
    validate_settings(settings)
    assert transfer_state_label(settings) == "TRANSFER STATE UNKNOWN"


def test_scenario_label_requires_explicit_toggle():
    strict = AppSettings(bank=0, free_transfers=1)
    scenario = AppSettings(bank=0, free_transfers=1, assume_selling_price_current=True)
    assert transfer_state_label(strict) == "STRICT FINANCIAL STATE"
    assert transfer_state_label(scenario) == "SCENARIO MODE"


def test_invalid_settings_fail_actionably():
    with pytest.raises(ValueError, match="Entry ID"):
        validate_settings(AppSettings(entry_id=0))
    with pytest.raises(ValueError, match="Bank"):
        validate_settings(AppSettings(bank=-0.1))


def test_deployment_paths_derive_from_project_root():
    path = project_relative_path("data/live/manual_squad.json")
    assert path == PROJECT_ROOT / "data/live/manual_squad.json"
    assert "/Users/vivekvattem" not in Path("src/fpl_predictor/ui/state.py").read_text()


def _current_players(legal_squad):
    players = legal_squad.rename(columns={"player_id": "id", "team": "team_name"}).copy()
    players["web_name"] = players.player
    return players[["id", "player", "web_name", "position", "team_name", "price"]]


def _manual_payload(legal_squad):
    return {"players": [{"player_id": int(player_id), "multiplier": 1}
                        for player_id in legal_squad.player_id]}


def test_uploaded_runtime_path_uses_internal_state_not_widget_keys(tmp_path, legal_squad):
    runtime = write_uploaded_squad_to_runtime(json.dumps(_manual_payload(legal_squad)).encode(),
                                              _current_players(legal_squad), tmp_path / "runtime.json")
    state = {"squad_file_upload": "widget-owned", "squad_file_path_input": "configured.json"}
    activate_uploaded_squad(state, runtime)
    assert state["squad_file_upload"] == "widget-owned"
    assert state["squad_file_path_input"] == "configured.json"
    assert state["active_squad_file"] == str(runtime)
    assert get_active_squad_file(state) == runtime
    assert active_squad_source(state) == "Uploaded squad for this session"


def test_invalid_upload_preserves_previous_valid_runtime_squad(tmp_path, legal_squad):
    runtime = write_uploaded_squad_to_runtime(json.dumps(_manual_payload(legal_squad)).encode(),
                                              _current_players(legal_squad), tmp_path / "runtime.json")
    original = runtime.read_bytes()
    state = {"active_squad_file": str(runtime), "squad_file_path_input": ""}
    with pytest.raises(ValueError, match="invalid"):
        write_uploaded_squad_to_runtime(b"{not json", _current_players(legal_squad), runtime)
    assert runtime.read_bytes() == original
    assert get_active_squad_file(state) == runtime


def test_valid_upload_becomes_active_and_is_session_scoped(tmp_path, legal_squad):
    runtime = write_uploaded_squad_to_runtime(json.dumps(_manual_payload(legal_squad)).encode(),
                                              _current_players(legal_squad), tmp_path / "session_runtime.json")
    state = {"active_squad_file": None, "squad_file_path_input": ""}
    activate_uploaded_squad(state, runtime)
    assert get_active_squad_file(state) == runtime
    # A new Cloud session has no active runtime pointer and therefore does not
    # retain the upload through session state.
    fresh_session = {"active_squad_file": None, "squad_file_path_input": ""}
    assert get_active_squad_file(fresh_session) != runtime
    assert active_squad_source(fresh_session) != "Uploaded squad for this session"


def test_runtime_upload_paths_are_isolated_between_cloud_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr("fpl_predictor.ui.state.LIVE_DATA_DIR", tmp_path)
    first, second = {}, {}
    assert runtime_squad_path(first) != runtime_squad_path(second)
    assert runtime_squad_path(first) == runtime_squad_path(first)
