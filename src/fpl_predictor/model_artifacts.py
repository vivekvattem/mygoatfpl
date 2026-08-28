"""Validated access to frozen production model artifacts."""

import json
from pathlib import Path
from typing import Any

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
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Cannot read metadata for position {position}: {exc}") from exc
        if metadata.get("position") != position or metadata.get("feature_names") != self.manifest["feature_names"]:
            raise ModelArtifactError(f"Artifact metadata mismatch for position {position}")
        try:
            return load_model(path)
        except Exception as exc:
            raise ModelArtifactError(f"Cannot load production artifact for position {position}: {exc}") from exc

    def validate_all(self) -> dict[str, Any]:
        """Load every serving artifact and verify the global fallback metadata."""
        positions = ("GK", "DEF", "MID", "FWD")
        loaded = {position: type(self.load_position(position)).__name__ for position in positions}
        global_value = self.manifest.get("global_model_path")
        if not global_value:
            raise ModelArtifactError("Production manifest has no global fallback artifact")
        global_path = self.project_root / str(global_value)
        metadata_path = global_path.with_suffix(".json")
        if not global_path.exists() or not metadata_path.exists():
            raise ModelArtifactError("Missing global fallback model or metadata")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("position") != "global":
                raise ModelArtifactError("Global fallback metadata position mismatch")
            if metadata.get("feature_names") != self.manifest["feature_names"]:
                raise ModelArtifactError("Global fallback feature metadata mismatch")
            load_model(global_path)
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError(f"Cannot load global fallback artifact: {exc}") from exc
        return {"positions": loaded, "global": True,
                "feature_count": len(self.manifest["feature_names"])}
