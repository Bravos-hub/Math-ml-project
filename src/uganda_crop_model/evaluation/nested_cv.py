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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from uganda_crop_model.models.pipelines import (
    build_preprocessor,
    resolve_feature_columns,
)
from uganda_crop_model.models.registry import ModelSpec

InnerMode = Literal["spatial", "temporal"]
TargetScale = Literal["raw", "crop_centered"]


def skill_score(
    observed: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> float:
    """MSE skill relative to a prediction made from training data only."""

    model_mse = mean_squared_error(observed, predicted)
    baseline_mse = mean_squared_error(observed, baseline)
    return 1.0 - model_mse / baseline_mse if baseline_mse > 0 else np.nan


def _cluster_bootstrap_intervals(
    group: pd.DataFrame,
    *,
    iterations: int = 200,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap metrics by spatial unit to preserve within-unit dependence."""

    units = group["spatial_unit"].dropna().unique()
    names = ("rmse", "mae", "global_skill", "crop_skill")
    if len(units) < 2:
        return {
            f"{name}_ci_{bound}": np.nan
            for name in names
            for bound in ("lower", "upper")
        }

    rng = np.random.default_rng(random_seed)
    draws = {name: [] for name in names}
    for _ in range(iterations):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled = pd.concat(
            [group[group["spatial_unit"].eq(unit)] for unit in sampled_units],
            ignore_index=True,
        )
        observed = sampled.get(
            "observed_evaluation_target", sampled["observed_yield"]
        ).to_numpy()
        predicted = sampled.get(
            "predicted_evaluation_target", sampled["predicted_yield"]
        ).to_numpy()
        draws["rmse"].append(sqrt(mean_squared_error(observed, predicted)))
        draws["mae"].append(mean_absolute_error(observed, predicted))
        if "training_global_mean" in sampled:
            draws["global_skill"].append(
                skill_score(
                    sampled["observed_yield"].to_numpy(),
                    sampled["predicted_yield"].to_numpy(),
                    sampled["training_global_mean"].to_numpy(),
                )
            )
        if "training_crop_mean" in sampled:
            draws["crop_skill"].append(
                skill_score(
                    sampled["observed_yield"].to_numpy(),
                    sampled["predicted_yield"].to_numpy(),
                    sampled["training_crop_mean"].to_numpy(),
                )
            )

    output: dict[str, float] = {}
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        output[f"{name}_ci_lower"] = (
            float(np.quantile(finite, 0.025)) if finite.size else np.nan
        )
        output[f"{name}_ci_upper"] = (
            float(np.quantile(finite, 0.975)) if finite.size else np.nan
        )
    return output


def _split_proper_and_calibration(
    metadata_train: pd.DataFrame,
    *,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = metadata_train["spatial_unit"].astype(str).to_numpy()
    if np.unique(groups).size < 2:
        raise ValueError(
            "At least two outer-training groups are required for calibration."
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_seed,
    )
    return next(splitter.split(metadata_train, groups=groups))


def _training_crop_offsets(
    training: pd.DataFrame,
    target: pd.Series,
    other: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray]:
    crop_means = target.groupby(training["crop"].to_numpy()).mean()
    global_mean = float(target.mean())
    training_offsets = training["crop"].map(crop_means).fillna(global_mean)
    other_offsets = other["crop"].map(crop_means).fillna(global_mean)
    return training_offsets, other_offsets.to_numpy(dtype=float)


def pca_parameter_grid(
    feature_space: str,
) -> dict[str, list[float]]:
    thresholds = [0.80, 0.90, 0.95]

    if feature_space == "pca":
        return {
            ("preprocess__continuous_pca__pca__n_components"): thresholds,
        }

    if feature_space == "hybrid":
        return {
            ("preprocess__climate_pca__pca__n_components"): thresholds,
        }

    return {}


def build_inner_temporal_splits(
    metadata: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    years = sorted(metadata["year"].dropna().unique())

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for position in range(2, len(years)):
        test_year = years[position]

        train_indices = np.flatnonzero(metadata["year"].lt(test_year).to_numpy())
        test_indices = np.flatnonzero(metadata["year"].eq(test_year).to_numpy())

        if train_indices.size and test_indices.size:
            splits.append((train_indices, test_indices))

    if len(splits) < 2:
        raise ValueError("Insufficient years for inner temporal validation.")

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
    target_scale: TargetScale = "raw",
    conformal_alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested evaluation, returning (predictions, fold_results)."""

    feature_columns = resolve_feature_columns(data)
    continuous_columns = feature_columns["climate"] + feature_columns["static"]
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

        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index]

        metadata_train = data.iloc[train_index].reset_index(drop=True)
        proper_index, calibration_index = _split_proper_and_calibration(
            metadata_train,
            random_seed=random_seed + fold_number,
        )
        X_proper = X_train.iloc[proper_index]
        X_calibration = X_train.iloc[calibration_index]
        y_proper_raw = y_train.iloc[proper_index]
        y_calibration_raw = y_train.iloc[calibration_index]
        metadata_proper = metadata_train.iloc[proper_index].reset_index(drop=True)
        metadata_calibration = metadata_train.iloc[calibration_index]

        if target_scale == "crop_centered":
            proper_offsets, calibration_offsets = _training_crop_offsets(
                metadata_proper,
                y_proper_raw.reset_index(drop=True),
                metadata_calibration,
            )
            _, test_offsets = _training_crop_offsets(
                metadata_proper,
                y_proper_raw.reset_index(drop=True),
                data.iloc[test_index],
            )
            y_proper = y_proper_raw.reset_index(drop=True) - proper_offsets.reset_index(
                drop=True
            )
        else:
            y_proper = y_proper_raw
            calibration_offsets = np.zeros(len(calibration_index), dtype=float)
            test_offsets = np.zeros(len(test_index), dtype=float)

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
            groups = metadata_proper["spatial_unit"].astype(str)

            group_count = groups.nunique()
            if group_count < 2:
                raise ValueError("Insufficient groups in the training fold.")

            inner_cv = GroupKFold(
                n_splits=min(4, group_count),
                shuffle=True,
                random_state=random_seed,
            )

            fit_arguments["groups"] = groups.to_numpy()

        elif inner_mode == "temporal":
            inner_cv = build_inner_temporal_splits(metadata_proper)

        else:
            raise ValueError(f"Unknown inner validation mode: {inner_mode}")

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
            X_proper,
            y_proper,
            **fit_arguments,
        )

        prediction = search.predict(X_test) + test_offsets

        # Genuine group split-conformal calibration: calibration groups are
        # untouched by tuning and model fitting, and the outer test remains
        # untouched until final prediction.
        calibration_prediction = search.predict(X_calibration) + calibration_offsets
        calibration_residuals = np.abs(
            y_calibration_raw.to_numpy() - calibration_prediction
        )
        n_cal = len(calibration_residuals)
        level = min(
            1.0,
            np.ceil((n_cal + 1) * (1.0 - conformal_alpha)) / n_cal,
        )
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
                "target_scale": target_scale,
                "test_rows": len(test_index),
                "rmse": rmse,
                "mae": mae,
                "r2": fold_r2,
                "best_parameters": search.best_params_,
                "calibration_size": int(n_cal),
                "calibration_group_count": int(
                    metadata_calibration["spatial_unit"].nunique()
                ),
                "conformal_alpha": conformal_alpha,
            }
        )

        prediction_frame = data.iloc[test_index][
            ["spatial_unit", "year", "season", "crop"]
        ].copy()

        prediction_frame["fold"] = fold_number
        prediction_frame["model"] = model_name
        prediction_frame["feature_space"] = feature_space
        prediction_frame["target_scale"] = target_scale
        prediction_frame["observed_yield"] = y_test.to_numpy()
        prediction_frame["predicted_yield"] = prediction
        crop_means = data.iloc[train_index].groupby("crop")["yield_tons_ha"].mean()
        global_mean = float(y_train.mean())
        prediction_frame["training_global_mean"] = global_mean
        prediction_frame["training_crop_mean"] = (
            data.iloc[test_index]["crop"].map(crop_means).fillna(global_mean).to_numpy()
        )
        if target_scale == "crop_centered":
            prediction_frame["observed_evaluation_target"] = (
                y_test.to_numpy() - test_offsets
            )
            prediction_frame["predicted_evaluation_target"] = prediction - test_offsets
        else:
            prediction_frame["observed_evaluation_target"] = y_test.to_numpy()
            prediction_frame["predicted_evaluation_target"] = prediction
        prediction_frame["conformal_lower"] = prediction - radius
        prediction_frame["conformal_upper"] = prediction + radius
        prediction_frame["conformal_radius"] = radius
        prediction_frame["conformal_alpha"] = conformal_alpha
        prediction_frame["calibration_size"] = n_cal
        prediction_frame["calibration_group_count"] = int(
            metadata_calibration["spatial_unit"].nunique()
        )

        all_predictions.append(prediction_frame)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_results = pd.DataFrame(fold_metrics)

    return predictions, fold_results


def summarize_out_of_fold_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    group_columns = ["model", "feature_space"]
    if "target_scale" in predictions:
        group_columns.append("target_scale")

    for keys, group in predictions.groupby(group_columns):
        model, feature_space = keys[:2]
        target_scale = keys[2] if len(keys) == 3 else "raw"
        observed = group.get(
            "observed_evaluation_target", group["observed_yield"]
        ).to_numpy()
        predicted = group.get(
            "predicted_evaluation_target", group["predicted_yield"]
        ).to_numpy()

        rmse = sqrt(mean_squared_error(observed, predicted))
        mae = mean_absolute_error(observed, predicted)

        r2 = (
            r2_score(observed, predicted)
            if len(observed) >= 2 and np.unique(observed).size >= 2
            else np.nan
        )

        observed_raw = group["observed_yield"].to_numpy()
        predicted_raw = group["predicted_yield"].to_numpy()
        global_skill = (
            skill_score(
                observed_raw,
                predicted_raw,
                group["training_global_mean"].to_numpy(),
            )
            if "training_global_mean" in group
            else np.nan
        )
        crop_skill = (
            skill_score(
                observed_raw,
                predicted_raw,
                group["training_crop_mean"].to_numpy(),
            )
            if "training_crop_mean" in group
            else np.nan
        )

        row = {
            "model": model,
            "feature_space": feature_space,
            "observations": len(group),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "skill_vs_training_global_mean": global_skill,
            "skill_vs_training_crop_mean": crop_skill,
            "target_scale": target_scale,
            "result_scope": "overall",
            "registered_primary_metric": target_scale == "raw",
        }
        row.update(_cluster_bootstrap_intervals(group))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["rmse", "mae"])


def summarize_conformal_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out interval coverage overall, by crop, and by season."""

    usable = predictions.dropna(subset=["conformal_lower", "conformal_upper"]).copy()
    usable["covered"] = usable["observed_yield"].between(
        usable["conformal_lower"], usable["conformal_upper"]
    )
    usable["interval_width"] = usable["conformal_upper"] - usable["conformal_lower"]
    rows: list[dict[str, object]] = []

    def append(groups, level: str) -> None:
        for keys, group in groups:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {
                "aggregation_level": level,
                "model": keys[0],
                "feature_space": keys[1],
                "target_scale": keys[2],
                "crop": keys[3] if level == "crop" else "",
                "season": keys[3] if level == "season" else "",
                "nominal_coverage": float(1.0 - group["conformal_alpha"].iloc[0]),
                "actual_coverage": float(group["covered"].mean()),
                "mean_interval_width": float(group["interval_width"].mean()),
                "observations": len(group),
                "calibration_size_min": int(group["calibration_size"].min()),
                "calibration_group_count_min": int(
                    group["calibration_group_count"].min()
                ),
                "calibration_group_count_max": int(
                    group["calibration_group_count"].max()
                ),
            }
            rows.append(row)

    core = ["model", "feature_space", "target_scale"]
    append(usable.groupby(core), "overall")
    append(usable.groupby(core + ["crop"]), "crop")
    append(usable.groupby(core + ["season"]), "season")
    return pd.DataFrame(rows)


def heldout_permutation_diagnostics(
    data, *, feature_space, model_spec, outer_splits, random_seed=42
) -> pd.DataFrame:
    """Permutation importance computed separately on each held-out fold."""
    feature_columns = resolve_feature_columns(data)
    predictors = (
        feature_columns["climate"]
        + feature_columns["static"]
        + feature_columns["categorical"]
    )
    rows = []
    for fold, (train_index, test_index) in enumerate(outer_splits, 1):
        pipe = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(feature_space, feature_columns=feature_columns),
                ),
                ("model", clone(model_spec.estimator)),
            ]
        )
        pipe.fit(
            data.iloc[train_index][predictors], data.iloc[train_index]["yield_tons_ha"]
        )
        result = permutation_importance(
            pipe,
            data.iloc[test_index][predictors],
            data.iloc[test_index]["yield_tons_ha"],
            scoring="neg_root_mean_squared_error",
            n_repeats=5,
            random_state=random_seed,
        )
        rows.extend(
            {
                "fold": fold,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(
                predictors, result.importances_mean, result.importances_std
            )
        )
    return pd.DataFrame(rows)
