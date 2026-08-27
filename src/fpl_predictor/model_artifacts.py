"""Validated access to frozen production model artifacts."""

import json
from pathlib import Path

from .modeling import load_model


class ModelArtifactError(ValueError): pass


class ProductionArtifacts:
    def __init__(self, project_root: Path, manifest_path: Path) -> None:
        self.project_root = project_root
        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Cannot read production manifest: {exc}") from exc
        required = {"model_name", "feature_set", "feature_names", "trained_on_seasons", "position_model_paths"}
        missing = required - self.manifest.keys()
        if missing:
            raise ModelArtifactError(f"Production manifest missing metadata: {sorted(missing)}")

    def load_position(self, position: str):
        path_value = self.manifest["position_model_paths"].get(position)
        if not path_value:
            raise ModelArtifactError(f"No production artifact for position {position}")
        path = self.project_root / path_value
        metadata_path = path.with_suffix(".json")
        if not path.exists() or not metadata_path.exists():
            raise ModelArtifactError(f"Missing model or metadata for position {position}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("position") != position or metadata.get("feature_names") != self.manifest["feature_names"]:
            raise ModelArtifactError(f"Artifact metadata mismatch for position {position}")
        return load_model(path)
