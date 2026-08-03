"""Predeclared sensitivity analyses for uncertainty and season assumptions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def complete_case_columns(data: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    """Return eligible features after a complete-case sensitivity check."""
    return [c for c in feature_columns if data[c].notna().all()]


def filter_uncertain_targets(
    data: pd.DataFrame,
    *,
    cv_column: str = "cv_production_pct",
    maximum_cv_pct: float = 30.0,
) -> pd.DataFrame:
    """Apply a declared survey-uncertainty exclusion rule."""
    if cv_column not in data:
        raise ValueError(f"Missing uncertainty column: {cv_column}")
    return data.loc[pd.to_numeric(data[cv_column], errors="coerce").le(maximum_cv_pct)].copy()


def weighted_fit_metrics(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cv_column: str,
    training_weights: pd.Series,
) -> dict[str, float]:
    """Fit a model with inverse-variance-style survey weights."""
    model = clone(estimator)
    weights = 1.0 / np.square(pd.to_numeric(training_weights, errors="coerce").fillna(training_weights.median()).clip(lower=1e-6))
    if not isinstance(model, Pipeline):
        model.fit(X_train, y_train, sample_weight=weights.to_numpy())
    else:
        # Most sklearn pipelines expose the final estimator as ``model``.
        final_name = model.steps[-1][0]
        model.fit(X_train, y_train, **{f"{final_name}__sample_weight": weights.to_numpy()})
    pred = model.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
        "n_test": int(len(y_test)),
        "uncertainty_weight_column": cv_column,
    }


def season_window_sensitivity(
    daily: pd.DataFrame,
    calendars: Sequence[pd.DataFrame],
    builder,
) -> pd.DataFrame:
    """Compare feature summaries under predeclared alternative calendars."""
    rows = []
    for index, calendar in enumerate(calendars):
        features = builder(daily, calendar)
        rows.append({
            "calendar_id": index,
            "rows": len(features),
            "feature_columns": len(features.columns),
            "rain_total_mean": float(features["rain_total_mm"].mean()),
        })
    return pd.DataFrame(rows)
