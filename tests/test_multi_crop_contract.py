"""Scientific contract tests for the multi-crop final dataset.

The multi-crop dataset expands the analytical sample from the 42 maize rows
to ~370 rows across ten food crops that share the same
``production / harvested area`` yield definition in both AAS 2018 and
AAS 2020.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uganda_crop_model.models.pipelines import (
    resolve_feature_columns,
)
from uganda_crop_model.quality.dataset import (
    AnalysisPolicy,
    validate_final_dataset,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MULTI_CROP_DATASET = PROCESSED / "final_multi_crop_subregion_season_year.csv"
MULTI_CROP_SEASONAL_DATASET = PROCESSED / "final_multi_crop_seasonal.csv"
MULTI_CROP_ANNUAL_DATASET = PROCESSED / "final_multi_crop_annual.csv"

EXPECTED_CROPS = {
    "maize",
    "beans",
    "groundnuts",
    "sorghum",
    "millet",
    "rice",
    "soya_beans",
    "simsim",
    "irish_potatoes",
    "sweet_potatoes",
}

# The multi-crop sample satisfies the 100-row and 5-unit requirements at
# final-mode defaults; only the 4-year minimum cannot be met (AAS 2018 and
# AAS 2020 are the only published waves).
RELAXED_POLICY = AnalysisPolicy(
    minimum_rows=100,
    minimum_environmental_units=20,
    minimum_years=1,
)

KEY = ["spatial_unit", "year", "season", "crop"]


@pytest.fixture(scope="module")
def multi_crop_dataset() -> pd.DataFrame:
    if not MULTI_CROP_DATASET.exists():
        pytest.skip("multi-crop dataset not built yet (run make data-final)")
    return pd.read_csv(MULTI_CROP_DATASET)


@pytest.fixture(scope="module")
def seasonal_dataset() -> pd.DataFrame:
    if not MULTI_CROP_SEASONAL_DATASET.exists():
        pytest.skip("seasonal dataset not built yet")
    return pd.read_csv(MULTI_CROP_SEASONAL_DATASET)


def test_sample_size_exceeds_100(multi_crop_dataset):
    assert len(multi_crop_dataset) >= 100


def test_spatial_coverage_at_least_5(multi_crop_dataset):
    assert multi_crop_dataset["spatial_unit"].nunique() >= 5


def test_all_expected_crops_present(multi_crop_dataset):
    assert EXPECTED_CROPS.issubset(set(multi_crop_dataset["crop"]))


def test_no_duplicate_analytical_keys(multi_crop_dataset):
    assert not multi_crop_dataset.duplicated(KEY).any()


def test_no_proxy_or_synthetic_targets(multi_crop_dataset):
    assert not multi_crop_dataset["is_proxy"].astype(bool).any()
    assert not multi_crop_dataset["is_synthetic"].astype(bool).any()
    assert not multi_crop_dataset["is_geographically_assigned"].astype(bool).any()


def test_single_target_definition(multi_crop_dataset):
    definitions = dict(multi_crop_dataset.groupby("year")["target_definition"].unique())
    # AAS 2018 publishes the official yield (total production over the
    # second-season harvested area); AAS 2020 recomputes production over
    # harvested area from the seasonal blocks.  Each year must use exactly
    # one documented definition.
    assert len(definitions[2018]) == 1
    assert len(definitions[2020]) == 1
    assert definitions[2018][0] == "published_official_subregion_yield_over_harvested"
    assert definitions[2020][0] == "production_mt_divided_by_area_harvested_ha"


def test_yield_recalculation(multi_crop_dataset):
    for _, row in multi_crop_dataset[
        multi_crop_dataset["yield_consistency_ok"].fillna(False).astype(bool)
    ].iterrows():
        calculated = row["production_mt"] / row["area_harvested_ha"]
        assert np.isclose(
            calculated,
            row["yield_tons_ha"],
            rtol=1e-6,
            atol=1e-8,
        )


def test_years_are_2018_and_2020(multi_crop_dataset):
    assert set(multi_crop_dataset["year"]) == {2018, 2020}


def test_2018_rows_are_annual(multi_crop_dataset):
    assert (
        multi_crop_dataset.loc[multi_crop_dataset["year"] == 2018, "season"] == "annual"
    ).all()


def test_2020_rows_are_seasonal(multi_crop_dataset):
    seasons = set(multi_crop_dataset.loc[multi_crop_dataset["year"] == 2020, "season"])
    assert {"first_season", "second_season"}.issubset(seasons)


def test_gate_passes_at_100_rows(seasonal_dataset):
    resolved = resolve_feature_columns(seasonal_dataset)
    features = resolved["climate"] + resolved["categorical"]
    validate_final_dataset(seasonal_dataset, features, RELAXED_POLICY)


def test_final_policy_fails_honestly_on_environment_count(seasonal_dataset):
    resolved = resolve_feature_columns(seasonal_dataset)
    features = resolved["climate"] + resolved["categorical"]
    with pytest.raises(ValueError, match="[Ii]nsufficient|requires at least"):
        validate_final_dataset(
            seasonal_dataset,
            features,
            AnalysisPolicy(),
        )


def test_combined_gate_rejects_mixed_temporal_granularity(multi_crop_dataset):
    resolved = resolve_feature_columns(multi_crop_dataset)
    with pytest.raises(ValueError, match="mix annual and seasonal"):
        validate_final_dataset(
            multi_crop_dataset,
            resolved["climate"],
            AnalysisPolicy(
                minimum_rows=100,
                minimum_environmental_units=10,
                minimum_years=1,
            ),
        )


def test_split_datasets_are_homogeneous():
    seasonal = pd.read_csv(MULTI_CROP_SEASONAL_DATASET)
    annual = pd.read_csv(MULTI_CROP_ANNUAL_DATASET)
    assert set(seasonal["target_temporal_granularity"]) == {"seasonal"}
    assert set(annual["target_temporal_granularity"]) == {"annual"}


def test_no_all_null_features(multi_crop_dataset):
    resolved = resolve_feature_columns(multi_crop_dataset)
    for column in resolved["climate"]:
        assert multi_crop_dataset[column].notna().any()


def test_yield_positive_and_finite(multi_crop_dataset):
    target = pd.to_numeric(
        multi_crop_dataset["yield_tons_ha"],
        errors="raise",
    )
    assert np.isfinite(target).all()
    assert (target > 0).all()
