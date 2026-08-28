"""Regression checks for the project's installable src-layout package."""

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_src_layout_package_discovery() -> None:
    """Editable installs must discover ``fpl_predictor`` without PYTHONPATH."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    setuptools = config["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"]["where"] == ["src"]
    assert config["project"]["requires-python"] == ">=3.11,<3.15"
    assert (PROJECT_ROOT / "src" / "fpl_predictor" / "__init__.py").is_file()
