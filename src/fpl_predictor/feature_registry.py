"""Machine-readable feature timing and provenance governance."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import PROJECT_ROOT

REGISTRY_PATH = PROJECT_ROOT / "config" / "features.yaml"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    dtype: str
    source: str
    timing: str
    window: int | None
    position_applicability: str
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FeatureRegistryError(ValueError):
    """Raised when feature provenance is missing or unsafe."""


def _spec(name: str, category: str, dtype: str, source: str, timing: str,
          description: str, window: int | None = None) -> FeatureSpec:
    return FeatureSpec(name, category, dtype, source, timing, window, "all", description)


def load_feature_registry(path: Path = REGISTRY_PATH) -> dict[str, FeatureSpec]:
    """Expand compact JSON-compatible YAML rules into concrete feature records."""
    config = json.loads(path.read_text(encoding="utf-8"))
    valid = set(config["valid_timings"])
    registry: dict[str, FeatureSpec] = {}
    for name in config["identifiers"]:
        registry[name] = _spec(name, "identifier", "string", "dataset", "identifier", "Row identifier.")
    for name in config["targets"]:
        registry[name] = _spec(name, "target", "float", "historical_fpl", "target", "Current-Gameweek supervised target.")
    for name in config["excluded"]:
        registry[name] = _spec(name, "baseline_output", "float", "derived", "excluded", "Precomputed diagnostic excluded from model inputs.")
    for name, values in config["static_features"].items():
        registry[name] = _spec(name, values[0], values[1], values[2], "pre_gw", values[3])

    for label, source in config["rolling_metrics"].items():
        for window in config["rolling_windows"]:
            for prefix, statistic in (("", "sum"), ("avg_", "mean")):
                name = f"{prefix}{label}_last_{window}"
                registry[name] = _spec(name, "player_form", "float", source, "pre_gw",
                    f"Prior-{window}-GW {statistic} of {source}; shifted before rolling.", window)
    for label in config["derived_rolling"]:
        for window in config["rolling_windows"]:
            name = f"{label}_last_{window}"
            registry[name] = _spec(name, "player_form", "float", "derived", "pre_gw",
                f"Leakage-safe prior-{window}-GW {label} feature.", window)
    for label in config["legacy_team_features"]:
        for window in (3, 5):
            name = f"{label}_last_{window}"
            registry[name] = _spec(name, "team_form", "float", "historical_results", "pre_gw",
                f"Prior-{window}-GW team/opponent result statistic.", window)

    for window in config["team_strength_windows"]:
        own = [
            f"team_attack_strength_{window}", f"team_defense_strength_{window}",
            f"team_attack_strength_{window}_rel", f"team_defense_strength_{window}_rel",
            f"team_attack_home_{window}", f"team_attack_away_{window}",
            f"team_defense_home_{window}", f"team_defense_away_{window}",
        ]
        for name in own:
            registry[name] = _spec(name, "team_strength", "float", "historical_results", "pre_gw",
                f"Prior-{window}-fixture team rating; current GW excluded.", window)
        for label in ("attack_strength", "defense_strength", "attack_strength_rel", "defense_strength_rel",
                      "attack_home", "attack_away", "defense_home", "defense_away"):
            for aggregation in ("mean", "min", "max"):
                if label.endswith("_rel"):
                    base = label.removesuffix("_rel")
                    name = f"opponent_{base}_{window}_rel_{aggregation}"
                else:
                    name = f"opponent_{label}_{window}_{aggregation}"
                registry[name] = _spec(name, "opponent_strength", "float", "historical_results", "pre_gw",
                    f"{aggregation.title()} prior-{window}-fixture opponent {label} across target fixtures.", window)
    if any(spec.timing not in valid for spec in registry.values()):
        raise FeatureRegistryError("Registry contains an invalid timing value")
    return registry


def validate_predictor_timing(features: Iterable[str], registry: dict[str, FeatureSpec]) -> None:
    """Require every predictor to be registered explicitly as pre-Gameweek."""
    names = list(features)
    missing = sorted(set(names) - set(registry))
    unsafe = sorted(name for name in names if name in registry and registry[name].timing != "pre_gw")
    if missing:
        raise FeatureRegistryError(f"Unregistered predictors: {missing}")
    if unsafe:
        raise FeatureRegistryError(f"Predictors are not timed pre_gw: {unsafe}")
