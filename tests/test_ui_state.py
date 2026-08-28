from pathlib import Path

import pytest

from fpl_predictor.config import PROJECT_ROOT
from fpl_predictor.ui.state import AppSettings, project_relative_path, transfer_state_label, validate_settings


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
