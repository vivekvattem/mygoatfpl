from datetime import datetime, timedelta, timezone
import json

import pytest

from fpl_predictor.refresh import (
    bump_generations, invalidated_generations, next_refresh_at, refresh_due, validate_refresh_interval,
)
from fpl_predictor.api import FPLAPIError
from fpl_predictor.ui.data import RefreshResult
from fpl_predictor.ui.reliability import run_monitored_refresh
from fpl_predictor.ui.state import AppSettings


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def test_refresh_enabled_disabled_and_manual_bypass():
    due = NOW - timedelta(seconds=1)
    assert refresh_due(True, due, NOW)
    assert not refresh_due(False, due, NOW)
    assert refresh_due(False, NOW + timedelta(hours=1), NOW, manual=True)


def test_interval_validation_and_next_refresh():
    assert next_refresh_at(NOW, 10) == NOW + timedelta(minutes=10)
    for value in (5, 10, 15, 30, 60):
        assert validate_refresh_interval(value) == value
    with pytest.raises(ValueError, match="one of"):
        validate_refresh_interval(1)


def test_dependency_invalidation_is_scoped():
    assert invalidated_generations("NO_CHANGE") == ()
    assert invalidated_generations("PLAYER_DATA_CHANGED") == (
        "live_generation", "personalized_generation", "analyst_generation")
    assert "fixture_generation" in invalidated_generations("FIXTURES_CHANGED")
    state = {"live_generation": 2, "fixture_generation": 4}
    bump_generations(state, "PLAYER_DATA_CHANGED")
    assert state["live_generation"] == 3
    assert state["fixture_generation"] == 4


class _Client:
    def __init__(self, bootstrap, fixtures):
        self.bootstrap, self.fixtures = bootstrap, fixtures
        self.calls = 0

    def get_bootstrap_static(self):
        self.calls += 1
        return self.bootstrap

    def get_fixtures(self):
        self.calls += 1
        return self.fixtures


def _live_payloads(price=75):
    bootstrap = {"elements": [{"id": 1, "web_name": "P", "now_cost": price, "status": "a"}],
                 "events": [{"id": 1, "is_current": True}], "teams": []}
    fixtures = [{"id": 10, "event": 1, "team_h": 1, "team_a": 2}]
    return bootstrap, fixtures


def test_monitored_refresh_skips_pipeline_when_unchanged(monkeypatch, tmp_path):
    bootstrap, fixtures = _live_payloads()
    (tmp_path / "bootstrap_static.json").write_text(json.dumps(bootstrap))
    (tmp_path / "fixtures.json").write_text(json.dumps(fixtures))
    monkeypatch.setattr("fpl_predictor.ui.reliability.RAW_DATA_DIR", tmp_path)
    client = _Client(bootstrap, fixtures)
    pipeline_calls = []
    state = {"widget_auto_refresh": False, "widget_refresh_interval_minutes": 10,
             "runtime_next_refresh": NOW + timedelta(hours=1)}
    result = run_monitored_refresh(AppSettings(), state, manual=True, client=client,
                                   pipeline_runner=lambda settings: pipeline_calls.append(settings))
    assert result.success and not result.changed and not pipeline_calls
    assert client.calls == 2


def test_monitored_refresh_rebuilds_and_bumps_only_dependencies(monkeypatch, tmp_path):
    old_bootstrap, fixtures = _live_payloads()
    new_bootstrap, _ = _live_payloads(price=76)
    (tmp_path / "bootstrap_static.json").write_text(json.dumps(old_bootstrap))
    (tmp_path / "fixtures.json").write_text(json.dumps(fixtures))
    monkeypatch.setattr("fpl_predictor.ui.reliability.RAW_DATA_DIR", tmp_path)
    calls = []
    def runner(settings):
        calls.append(settings)
        return RefreshResult(True, "rebuilt", NOW)
    state = {"widget_auto_refresh": True, "widget_refresh_interval_minutes": 10,
             "runtime_next_refresh": NOW, "fixture_generation": 3}
    result = run_monitored_refresh(AppSettings(), state, client=_Client(new_bootstrap, fixtures),
                                   pipeline_runner=runner, now=NOW)
    assert result.category == "PLAYER_DATA_CHANGED" and len(calls) == 1
    assert state["live_generation"] == 1 and state["fixture_generation"] == 3
    assert state["personalized_generation"] == 1


def test_failed_api_check_preserves_prior_outputs(monkeypatch, tmp_path):
    prior = tmp_path / "bootstrap_static.json"
    prior.write_text('{"valid": true}')
    class FailingClient:
        def get_bootstrap_static(self):
            raise FPLAPIError("offline")
        def get_fixtures(self):
            raise AssertionError("not reached")
    monkeypatch.setattr("fpl_predictor.ui.reliability.RAW_DATA_DIR", tmp_path)
    state = {"widget_auto_refresh": True, "widget_refresh_interval_minutes": 10,
             "runtime_next_refresh": NOW}
    result = run_monitored_refresh(AppSettings(), state, client=FailingClient(), now=NOW)
    assert not result.success and result.category == "API_FAILURE"
    assert prior.read_text() == '{"valid": true}'
    assert state["runtime_last_refresh_failure"]
