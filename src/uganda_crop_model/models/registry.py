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
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    parameter_grid: dict[str, list[Any]]


def get_model_registry(
    random_seed: int = 42,
) -> dict[str, ModelSpec]:
    return {
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
        "random_forest": ModelSpec(
            estimator=RandomForestRegressor(
                random_state=random_seed,
                n_jobs=1,
            ),
            parameter_grid={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [2, 4, 8],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "xgboost": ModelSpec(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_seed,
                n_jobs=1,
                verbosity=0,
            ),
            parameter_grid={
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 5],
                "model__reg_lambda": [1.0, 10.0],
            },
        ),
    }