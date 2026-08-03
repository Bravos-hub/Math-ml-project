"""Contracts for the previously missing sensitivity/interpretability layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from uganda_crop_model.evaluation.interpretability import (
    model_agreement,
    residual_diagnostics,
    variance_inflation_factors,
)
from uganda_crop_model.evaluation.sensitivity import (
    complete_case_columns,
    filter_uncertain_targets,
)
from uganda_crop_model.data.extract_chirps import extract_polygon_daily_mean
from uganda_crop_model.features.remote_sensing import validate_vegetation_features


def test_complete_case_and_uncertainty_rules_are_explicit():
    data = pd.DataFrame({"x": [1.0, 2.0], "z": [1.0, np.nan], "cv": [10.0, 40.0]})
    assert complete_case_columns(data, ["x", "z"]) == ["x"]
    assert len(filter_uncertain_targets(data, cv_column="cv", maximum_cv_pct=30)) == 1


def test_interpretability_tables_have_stable_schema():
    residuals = residual_diagnostics([1, 2], [1.5, 1.5])
    assert {"observed", "predicted", "residual", "absolute_error"} <= set(residuals)
    agreement = model_agreement(pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "observed_yield": [1, 2, 1, 2],
        "predicted_yield": [1, 2, 1.5, 1.5],
    }))
    assert list(agreement.columns) == ["model", "mae", "n"]


def test_vif_is_computed_without_optional_statsmodels():
    result = variance_inflation_factors(pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]}))
    assert set(result.columns) == {"feature", "vif"}
    assert (result["vif"] >= 1).all()


def test_polygon_daily_extractor_returns_area_weighted_long_series():
    data = xr.DataArray(
        np.arange(8, dtype=float).reshape(2, 2, 2),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.date_range("2020-01-01", periods=2),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
        name="precip",
    )
    boundaries = pd.DataFrame({
        "spatial_unit": ["u"],
        "geometry": [{"type": "Polygon", "coordinates": [[[-1, -1], [2, -1], [2, 2], [-1, 2], [-1, -1]]]}],
    })
    result = extract_polygon_daily_mean(data, boundaries)
    assert len(result) == 2
    assert np.allclose(result["rain_mm"], [1.5, 5.5], atol=1e-3)


def test_remote_sensing_features_are_timed_before_harvest():
    result = validate_vegetation_features(pd.DataFrame({
        "spatial_unit": ["u", "u"], "year": [2020, 2020],
        "observation_date": ["2020-05-01", "2020-08-01"],
        "harvest_date": ["2020-07-01", "2020-07-01"], "ndvi_mean": [0.5, 0.7],
    }))
    assert result["vegetation_timing"].tolist() == ["pre_harvest", "retrospective"]
