"""Training-fold-only baselines for spatial and temporal validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselinePrediction:
    """One baseline's predictions and row-level applicability metadata."""

    values: np.ndarray
    baseline_applicable: np.ndarray
    fallback_used: np.ndarray
    fallback_level: np.ndarray


def _lookup_mean(
    test: pd.DataFrame,
    table: pd.Series,
    keys: tuple[str, ...],
    *,
    crop_mean: pd.Series,
    global_mean: float,
) -> BaselinePrediction:
    values: list[float] = []
    applicable: list[bool] = []
    fallback_levels: list[str] = []

    for row in test.itertuples(index=False):
        key: object = tuple(getattr(row, name) for name in keys)
        if len(keys) == 1:
            key = key[0]
        value = table.get(key, np.nan)
        if pd.notna(value):
            values.append(float(value))
            applicable.append(True)
            fallback_levels.append("none")
            continue

        crop_value = crop_mean.get(row.crop, np.nan)
        if pd.notna(crop_value):
            values.append(float(crop_value))
            fallback_levels.append("training_crop_mean")
        else:
            values.append(global_mean)
            fallback_levels.append("training_global_mean")
        applicable.append(False)

    applicable_array = np.asarray(applicable, dtype=bool)
    return BaselinePrediction(
        values=np.asarray(values, dtype=float),
        baseline_applicable=applicable_array,
        fallback_used=~applicable_array,
        fallback_level=np.asarray(fallback_levels, dtype=object),
    )


def previous_available_wave(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str = "yield_tons_ha",
) -> BaselinePrediction:
    """Predict from the closest strictly earlier matching survey wave."""

    crop_mean = train.groupby("crop")[target].mean()
    global_mean = float(train[target].mean())
    values: list[float] = []
    applicable: list[bool] = []
    fallback_levels: list[str] = []

    for row in test.itertuples(index=False):
        candidates = train[
            train["spatial_unit"].eq(row.spatial_unit)
            & train["crop"].eq(row.crop)
            & train["season"].eq(row.season)
            & train["year"].lt(row.year)
        ].sort_values("year")
        if not candidates.empty:
            values.append(float(candidates.iloc[-1][target]))
            applicable.append(True)
            fallback_levels.append("none")
            continue

        crop_value = crop_mean.get(row.crop, np.nan)
        if pd.notna(crop_value):
            values.append(float(crop_value))
            fallback_levels.append("training_crop_mean")
        else:
            values.append(global_mean)
            fallback_levels.append("training_global_mean")
        applicable.append(False)

    applicable_array = np.asarray(applicable, dtype=bool)
    return BaselinePrediction(
        values=np.asarray(values, dtype=float),
        baseline_applicable=applicable_array,
        fallback_used=~applicable_array,
        fallback_level=np.asarray(fallback_levels, dtype=object),
    )


def predict_training_baselines(
    data: pd.DataFrame,
    train_index,
    test_index,
    *,
    validation_mode: str = "spatial",
) -> dict[str, BaselinePrediction]:
    """Return baselines that are meaningful for the validation design."""

    train = data.iloc[train_index]
    test = data.iloc[test_index]
    y = pd.to_numeric(train["yield_tons_ha"], errors="raise")
    global_mean = float(y.mean())
    crop_mean = train.groupby("crop")["yield_tons_ha"].mean()
    crop_season_mean = train.groupby(["crop", "season"])["yield_tons_ha"].mean()

    n_test = len(test)
    no_fallback = np.zeros(n_test, dtype=bool)
    results = {
        "training_global_mean": BaselinePrediction(
            values=np.repeat(global_mean, n_test),
            baseline_applicable=np.ones(n_test, dtype=bool),
            fallback_used=no_fallback,
            fallback_level=np.repeat("none", n_test),
        ),
        "training_crop_mean": _lookup_mean(
            test,
            crop_mean,
            ("crop",),
            crop_mean=crop_mean,
            global_mean=global_mean,
        ),
        "training_crop_season_mean": _lookup_mean(
            test,
            crop_season_mean,
            ("crop", "season"),
            crop_mean=crop_mean,
            global_mean=global_mean,
        ),
    }

    if validation_mode in {"temporal", "stress"}:
        subregion_crop = train.groupby(["spatial_unit", "crop"])["yield_tons_ha"].mean()
        results["historical_subregion_crop_mean"] = _lookup_mean(
            test,
            subregion_crop,
            ("spatial_unit", "crop"),
            crop_mean=crop_mean,
            global_mean=global_mean,
        )
        results["previous_available_wave"] = previous_available_wave(train, test)

    return results
