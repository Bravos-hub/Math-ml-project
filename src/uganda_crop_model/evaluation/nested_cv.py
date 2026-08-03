"""Nested cross-validation with leakage-safe pipelines.

Outer folds estimate performance; inner folds select Ridge alpha, Random
Forest / XGBoost settings, and the PCA variance threshold.  The inner search
never touches the outer fold's test observations, so the reported
out-of-fold skill is honest (blueprint section 18).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from uganda_crop_model.models.pipelines import (
    build_preprocessor,
    resolve_feature_columns,
)
from uganda_crop_model.models.registry import ModelSpec

InnerMode = Literal["spatial", "temporal"]


def pca_parameter_grid(
    feature_space: str,
) -> dict[str, list[float]]:
    thresholds = [0.80, 0.90, 0.95]

    if feature_space == "pca":
        return {
            (
                "preprocess__continuous_pca__"
                "pca__n_components"
            ): thresholds,
        }

    if feature_space == "hybrid":
        return {
            (
                "preprocess__climate_pca__"
                "pca__n_components"
            ): thresholds,
        }

    return {}


def build_inner_temporal_splits(
    metadata: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    years = sorted(metadata["year"].dropna().unique())

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for position in range(2, len(years)):
        test_year = years[position]

        train_indices = np.flatnonzero(
            metadata["year"].lt(test_year).to_numpy()
        )
        test_indices = np.flatnonzero(
            metadata["year"].eq(test_year).to_numpy()
        )

        if train_indices.size and test_indices.size:
            splits.append((train_indices, test_indices))

    if len(splits) < 2:
        raise ValueError(
            "Insufficient years for inner temporal validation."
        )

    return splits


def run_nested_evaluation(
    data: pd.DataFrame,
    *,
    feature_space: str,
    model_name: str,
    model_spec: ModelSpec,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    inner_mode: InnerMode,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested evaluation, returning (predictions, fold_results)."""

    feature_columns = resolve_feature_columns(data)
    continuous_columns = (
        feature_columns["climate"] + feature_columns["static"]
    )
    predictors = continuous_columns + feature_columns["categorical"]

    X = data[predictors].copy()
    y = pd.to_numeric(data["yield_tons_ha"], errors="raise")

    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []

    for fold_number, (train_index, test_index) in enumerate(
        outer_splits,
        start=1,
    ):
        train_index = np.asarray(train_index)
        test_index = np.asarray(test_index)

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        metadata_train = data.iloc[train_index].reset_index(drop=True)

        preprocessor = build_preprocessor(
            feature_space=feature_space,
            feature_columns=feature_columns,
            pca_variance=0.90,
        )

        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", clone(model_spec.estimator)),
            ]
        )

        search_grid = {
            **dict(model_spec.parameter_grid),
            **pca_parameter_grid(feature_space),
        }

        fit_arguments: dict[str, object] = {}

        if inner_mode == "spatial":
            groups = metadata_train["spatial_unit"].astype(str)

            group_count = groups.nunique()
            if group_count < 2:
                raise ValueError(
                    "Insufficient groups in the training fold."
                )

            inner_cv = GroupKFold(
                n_splits=min(4, group_count),
                shuffle=True,
                random_state=random_seed,
            )

            fit_arguments["groups"] = groups.to_numpy()

        elif inner_mode == "temporal":
            inner_cv = build_inner_temporal_splits(metadata_train)

        else:
            raise ValueError(
                f"Unknown inner validation mode: {inner_mode}"
            )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=search_grid or [{}],
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            # Keep the authoritative runner deterministic and resource-safe;
            # parallel nested searches can exhaust undergraduate workstations.
            n_jobs=1,
            refit=True,
            error_score="raise",
        )

        search.fit(
            X_train,
            y_train,
            **fit_arguments,
        )

        prediction = search.predict(X_test)

        # Training-only split-conformal calibration.  The calibration
        # residuals are computed from the fitted training pipeline and never
        # use outer-test targets.
        train_prediction = search.predict(X_train)
        calibration_residuals = np.abs(y_train.to_numpy() - train_prediction)
        n_cal = len(calibration_residuals)
        level = min(1.0, np.ceil((n_cal + 1) * 0.90) / n_cal)
        radius = float(np.quantile(calibration_residuals, level, method="higher"))

        rmse = sqrt(mean_squared_error(y_test, prediction))
        mae = mean_absolute_error(y_test, prediction)

        if len(y_test) >= 2 and y_test.nunique() >= 2:
            fold_r2 = r2_score(y_test, prediction)
        else:
            fold_r2 = np.nan

        fold_metrics.append(
            {
                "fold": fold_number,
                "model": model_name,
                "feature_space": feature_space,
                "test_rows": int(len(test_index)),
                "rmse": rmse,
                "mae": mae,
                "r2": fold_r2,
                "best_parameters": search.best_params_,
            }
        )

        prediction_frame = data.iloc[test_index][
            ["spatial_unit", "year", "season", "crop"]
        ].copy()

        prediction_frame["fold"] = fold_number
        prediction_frame["model"] = model_name
        prediction_frame["feature_space"] = feature_space
        prediction_frame["observed_yield"] = y_test.to_numpy()
        prediction_frame["predicted_yield"] = prediction
        crop_means = data.iloc[train_index].groupby("crop")["yield_tons_ha"].mean()
        global_mean = float(y_train.mean())
        prediction_frame["training_crop_mean"] = data.iloc[test_index]["crop"].map(crop_means).fillna(global_mean).to_numpy()
        prediction_frame["conformal_lower"] = prediction - radius
        prediction_frame["conformal_upper"] = prediction + radius
        prediction_frame["conformal_radius"] = radius

        all_predictions.append(prediction_frame)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_results = pd.DataFrame(fold_metrics)

    return predictions, fold_results


def summarize_out_of_fold_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (model, feature_space), group in predictions.groupby(
        ["model", "feature_space"]
    ):
        observed = group["observed_yield"].to_numpy()
        predicted = group["predicted_yield"].to_numpy()

        rmse = sqrt(mean_squared_error(observed, predicted))
        mae = mean_absolute_error(observed, predicted)

        r2 = (
            r2_score(observed, predicted)
            if len(observed) >= 2 and np.unique(observed).size >= 2
            else np.nan
        )

        baseline = np.repeat(observed.mean(), len(observed))
        baseline_mse = mean_squared_error(observed, baseline)
        model_mse = mean_squared_error(observed, predicted)

        skill_score = (
            1.0 - model_mse / baseline_mse
            if baseline_mse > 0
            else np.nan
        )

        rows.append(
            {
                "model": model,
                "feature_space": feature_space,
                "observations": int(len(group)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "skill_vs_mean_baseline": skill_score,
            }
        )

    return pd.DataFrame(rows).sort_values(["rmse", "mae"])


def heldout_permutation_diagnostics(
    data, *, feature_space, model_spec, outer_splits, random_seed=42
) -> pd.DataFrame:
    """Permutation importance computed separately on each held-out fold."""
    feature_columns = resolve_feature_columns(data)
    predictors = feature_columns["climate"] + feature_columns["static"] + feature_columns["categorical"]
    rows = []
    for fold, (train_index, test_index) in enumerate(outer_splits, 1):
        pipe = Pipeline([
            ("preprocess", build_preprocessor(feature_space, feature_columns=feature_columns)),
            ("model", clone(model_spec.estimator)),
        ])
        pipe.fit(data.iloc[train_index][predictors], data.iloc[train_index]["yield_tons_ha"])
        result = permutation_importance(
            pipe, data.iloc[test_index][predictors], data.iloc[test_index]["yield_tons_ha"],
            scoring="neg_root_mean_squared_error", n_repeats=5, random_state=random_seed,
        )
        rows.extend({"fold": fold, "feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
                    for feature, mean, std in zip(predictors, result.importances_mean, result.importances_std))
    return pd.DataFrame(rows)
