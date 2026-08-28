#!/usr/bin/env python3
"""Validate the repository's critical Streamlit serving contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from fpl_predictor.api import FPLAPIClient, FPLAPIError
from fpl_predictor.config import HISTORICAL_ML_DIR, MODEL_DIR, PROJECT_ROOT
from fpl_predictor.model_artifacts import ModelArtifactError, ProductionArtifacts


REQUIRED_CONFIG = (
    Path("pyproject.toml"), Path("requirements.txt"), Path(".streamlit/config.toml"),
    Path(".streamlit/secrets.toml.example"), Path("config/features.yaml"),
)
PACKAGES = ("streamlit", "pandas", "numpy", "scipy", "scikit-learn", "joblib", "plotly", "requests", "pydantic")


def run_check(require_network: bool = False) -> int:
    critical: list[str] = []
    warnings: list[str] = []
    print("FPL DEPLOYMENT CHECK\n")
    print(f"Python: {platform.python_version()}")
    if not ((3, 11) <= sys.version_info[:2] < (3, 15)):
        critical.append("Python must be between 3.11 and 3.14")
    print(f"Package: {Path(__import__('fpl_predictor').__file__).resolve()}")
    print(f"Project root: {PROJECT_ROOT}")
    if PROJECT_ROOT != Path(__file__).resolve().parents[1]:
        critical.append("Package project root does not match this repository")
    missing_config = [str(path) for path in REQUIRED_CONFIG if not (PROJECT_ROOT / path).is_file()]
    if missing_config:
        critical.append(f"Missing required config files: {missing_config}")

    manifest_path = HISTORICAL_ML_DIR / "phase4_training.json"
    try:
        artifact_report = ProductionArtifacts(PROJECT_ROOT, manifest_path).validate_all()
        print(f"Model artifacts: PASS ({len(artifact_report['positions'])} position models + global)")
        print(f"Feature schema: PASS ({artifact_report['feature_count']} required features)")
    except ModelArtifactError as exc:
        critical.append(f"Model artifacts: {exc}")

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            critical.append(f"Missing dependency: {package}")
    print("Packages: " + json.dumps(versions, sort_keys=True))

    try:
        payload = FPLAPIClient().get_bootstrap_static()
        if not isinstance(payload, dict) or not payload.get("elements"):
            raise FPLAPIError("bootstrap response contains no players")
        print(f"Official FPL API: REACHABLE ({len(payload['elements'])} players)")
    except (FPLAPIError, OSError) as exc:
        message = f"Official FPL API: UNAVAILABLE ({exc})"
        (critical if require_network else warnings).append(message)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if critical:
        for failure in critical:
            print(f"FAIL: {failure}")
        print("\nDeployment check failed.")
        return 1
    print("\nDeployment check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-network", action="store_true",
                        help="Treat official FPL API unavailability as a critical failure")
    return run_check(parser.parse_args().require_network)


if __name__ == "__main__":
    raise SystemExit(main())
