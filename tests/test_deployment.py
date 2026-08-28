"""Deployment contracts independent of developer-machine runtime files."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import fpl_predictor.ui.data as ui_data
import fpl_predictor.ui.state as ui_state
from fpl_predictor.config import HISTORICAL_ML_DIR, PROJECT_ROOT
from fpl_predictor.model_artifacts import ProductionArtifacts
from fpl_predictor.ui import components
from fpl_predictor.ui.contracts import first_existing_column, safe_numeric, safe_series, safe_value


def test_optional_column_contract_preserves_index_and_missingness() -> None:
    frame = pd.DataFrame({"xpts": [2.0, None]}, index=[7, 9])
    assert first_existing_column(frame, ("weighted_xpts_5", "xpts")) == "xpts"
    assert safe_series(frame, "missing").index.tolist() == [7, 9]
    assert safe_numeric(frame, "missing").isna().all()
    assert safe_value({"known": 2}, "missing", "Unknown") == "Unknown"


def test_all_package_modules_import_without_path_mutation() -> None:
    source = PROJECT_ROOT / "src" / "fpl_predictor"
    for path in source.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT / "src").with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        importlib.import_module(".".join(parts))
    application_sources = [PROJECT_ROOT / "app.py", *sorted((PROJECT_ROOT / "pages").glob("*.py"))]
    assert all("sys.path" not in path.read_text(encoding="utf-8") for path in application_sources)


def test_committed_production_artifacts_load() -> None:
    report = ProductionArtifacts(PROJECT_ROOT, HISTORICAL_ML_DIR / "phase4_training.json").validate_all()
    assert set(report["positions"]) == {"GK", "DEF", "MID", "FWD"}
    assert report["global"] and report["feature_count"] == 33


def test_every_streamlit_page_handles_fresh_no_data_state(monkeypatch, tmp_path) -> None:
    live = tmp_path / "live"
    raw = tmp_path / "raw"
    live.mkdir(); raw.mkdir()
    monkeypatch.setattr(ui_data, "LIVE_DATA_DIR", live)
    monkeypatch.setattr(ui_data, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(ui_state, "LIVE_DATA_DIR", live)
    components.cached_bundle.clear()
    scripts = [PROJECT_ROOT / "app.py", *sorted((PROJECT_ROOT / "pages").glob("*.py"))]
    for script in scripts:
        app = AppTest.from_file(str(script), default_timeout=30).run()
        assert not list(app.exception), f"{script.name}: {[item.value for item in app.exception]}"


def test_dashboard_explains_no_squad_without_fake_personalization(monkeypatch, tmp_path) -> None:
    live = tmp_path / "live"; raw = tmp_path / "raw"
    live.mkdir(); raw.mkdir()
    pd.DataFrame({
        "player_id": [1], "player": ["Available Player"], "team": ["Club"], "position": ["MID"],
        "price": [7.0], "availability_adjusted_xpts": [4.5], "owned": [False],
    }).to_csv(live / "player_predictions.csv", index=False)
    (live / "live_summary.json").write_text(
        '{"target_gw": 2, "schema_validation": {"passed": true, "required": 33, "available": 33}}'
    )
    monkeypatch.setattr(ui_data, "LIVE_DATA_DIR", live)
    monkeypatch.setattr(ui_data, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(ui_state, "LIVE_DATA_DIR", live)
    components.cached_bundle.clear()
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    assert not list(app.exception)
    assert any("No personalized squad loaded" in item.value for item in app.info)
