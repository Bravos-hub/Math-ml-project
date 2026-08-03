"""Model registry: required models plus a dummy mean baseline.

The blue-print requires OLS, Ridge, Random Forest, and XGBoost, compared
against a dummy-mean baseline for reference (blueprint section 16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression


class UnavailableEstimator:
    """Explicit placeholder used when an optional binary is unavailable."""
    def fit(self, X, y):
        raise RuntimeError("XGBoost is unavailable; install requirements.lock")

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def predict(self, X):
        raise RuntimeError("XGBoost is unavailable; install requirements.lock")


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    parameter_grid: dict[str, list[Any]]


def get_model_registry(
    random_seed: int = 42,
    load_optional: bool = False,
) -> dict[str, ModelSpec]:
    registry = {
        "dummy_mean": ModelSpec(
            estimator=DummyRegressor(strategy="mean"),
            parameter_grid={},
        ),
        "ols": ModelSpec(
            estimator=LinearRegression(),
            parameter_grid={},
        ),
        "ridge": ModelSpec(
            estimator=Ridge(),
            parameter_grid={
                "model__alpha": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ],
            },
        ),
        "pls": ModelSpec(
            estimator=PLSRegression(scale=False),
            parameter_grid={"model__n_components": [2, 4]},
        ),
        "random_forest": ModelSpec(
            estimator=RandomForestRegressor(
                random_state=random_seed,
                n_jobs=1,
            ),
            parameter_grid={
                "model__n_estimators": [300],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": ["sqrt"],
            },
        ),
    }
    xgb = None
    if load_optional:
        try:
            from xgboost import XGBRegressor as xgb
        except Exception:
            xgb = None
    if xgb is not None:
        registry["xgboost"] = ModelSpec(
            estimator=xgb(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_seed,
                n_jobs=1,
                verbosity=0,
            ),
            parameter_grid={
                "model__n_estimators": [200],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.05],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__min_child_weight": [1],
                "model__reg_lambda": [1.0],
            },
        )
    else:
        registry["xgboost"] = ModelSpec(
            estimator=UnavailableEstimator(),
            parameter_grid={},
        )
    return registry
