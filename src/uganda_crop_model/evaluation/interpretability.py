"""Non-causal model interpretation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


def standardized_coefficients(model, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return standardized coefficients for a fitted linear estimator."""
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    coefficients = np.asarray(estimator.coef_).reshape(-1)
    if len(coefficients) != len(feature_names):
        raise ValueError("feature_names does not match coefficient count")
    return pd.DataFrame({"feature": feature_names, "coefficient": coefficients}).sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def coefficient_bootstrap_ci(
    X: np.ndarray, y: np.ndarray, fit_predictor, *, iterations: int = 500, random_seed: int = 42
) -> pd.DataFrame:
    """Bootstrap coefficient estimates from a supplied fitting callback."""
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        estimates.append(np.asarray(fit_predictor(X[idx], y[idx]), dtype=float).reshape(-1))
    values = np.asarray(estimates)
    return pd.DataFrame({
        "mean": values.mean(axis=0),
        "lower_95": np.percentile(values, 2.5, axis=0),
        "upper_95": np.percentile(values, 97.5, axis=0),
    })


def residual_diagnostics(y_true: Sequence[float], y_pred: Sequence[float]) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return pd.DataFrame({
        "observed": y_true,
        "predicted": y_pred,
        "residual": residual,
        "absolute_error": np.abs(residual),
    })


def variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.select_dtypes(include=np.number).dropna()
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])
    if numeric.empty or numeric.shape[0] <= numeric.shape[1]:
        return pd.DataFrame({"feature": numeric.columns, "vif": np.nan})
    values = numeric.to_numpy(dtype=float)
    rows = []
    for i, name in enumerate(numeric.columns):
        target = values[:, i]
        others = np.delete(values, i, axis=1)
        if others.shape[1] == 0:
            vif = 1.0
        else:
            r2 = LinearRegression().fit(others, target).score(others, target)
            vif = float(1.0 / max(1.0 - r2, np.finfo(float).eps))
        rows.append({"feature": name, "vif": vif})
    return pd.DataFrame(rows)


def heldout_permutation_importance(model, X_test, y_test, feature_names: Sequence[str], *, random_seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X_test, y_test, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=random_seed)
    return pd.DataFrame({
        "feature": list(feature_names),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def partial_dependence_table(model, X, features: Sequence[int | str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        result = PartialDependenceDisplay.from_estimator(model, X, [feature])
        grid = result.pd_results[0]["grid_values"][0]
        average = result.pd_results[0]["average"][0]
        rows.extend({"feature": str(feature), "grid": float(x), "partial_dependence": float(y)} for x, y in zip(grid, average))
    return pd.DataFrame(rows)


def model_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare held-out predictions across models without causal language."""
    predictions = predictions.rename(
        columns={"y_true": "observed_yield", "y_pred": "predicted_yield"}
    )
    required = {"model", "observed_yield", "predicted_yield"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions.columns))}")
    rows = []
    for model, group in predictions.groupby("model"):
        rows.append({"model": model, "mae": mean_absolute_error(group.observed_yield, group.predicted_yield), "n": len(group)})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
