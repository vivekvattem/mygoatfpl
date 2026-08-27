"""Conservative Phase 4 model configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    parameters: dict[str, object]


MODEL_SPECS = (
    ModelSpec("Ridge", "ridge", {"alpha": 10.0}),
    ModelSpec("Random Forest", "random_forest", {
        "n_estimators": 80, "max_depth": 10, "min_samples_leaf": 20,
        "max_features": .7, "n_jobs": 1, "random_state": 42,
    }),
    ModelSpec("HistGradientBoosting", "hist_gradient_boosting", {
        "learning_rate": .05, "max_leaf_nodes": 31, "max_iter": 100,
        "l2_regularization": 1.0, "random_state": 42,
    }),
)
