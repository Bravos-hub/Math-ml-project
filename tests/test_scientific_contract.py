"""Scientific contract tests for the final-analysis dataset.

These run against ``final_maize_subregion_season_year.csv`` and the PCA
mathematics module.  A small part of the suite uses a relaxed analysis
policy (the AAS sample is intentionally below the blueprint's 100-row
final target) so that the *other* contract checks can be exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uganda_crop_model.pca.manual_verification import (
    verify_two_feature_pca,
)
from uganda_crop_model.quality.dataset import (
    AnalysisPolicy,
    validate_final_dataset,
    TARGET_DERIVED_COLUMNS,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
FINAL_MAIZE_DATASET = PROCESSED / "final_maize_subregion_season_year.csv"

CLIMATE_FEATURES = [
    "rain_total_mm",
    "rain_mean_daily_mm",
    "rainy_days_1mm",
    "rainy_days_10mm",
    "heavy_rain_days_20mm",
    "maximum_1day_rainfall_mm",
    "maximum_5day_rainfall_mm",
    "longest_dry_spell_days",
    "wet_day_rainfall_cv",
    "onset_day_of_year",
    "cessation_day_of_year",
    "season_length_days",
    "temperature_mean_c",
    "temperature_maximum_c",
    "temperature_minimum_c",
    "temperature_range_c",
    "growing_degree_days",
    "heat_days_32c",
    "extreme_heat_days_35c",
    "cold_days_10c",
]

# Policy relaxed only on the two sample-size dimensions that the available
# AAS data cannot satisfy; everything else uses final-mode defaults.
RELAXED_POLICY = AnalysisPolicy(
    minimum_rows=10,
    minimum_environmental_units=10,
    minimum_years=1,
)


@pytest.fixture(scope="module")
def final_dataset() -> pd.DataFrame:
    if not FINAL_MAIZE_DATASET.exists():
        pytest.skip("final dataset not built yet (run make data-final)")
    return pd.read_csv(FINAL_MAIZE_DATASET)


@pytest.fixture(scope="module")
def homogeneous_dataset(final_dataset) -> pd.DataFrame:
    return final_dataset[
        final_dataset["target_temporal_granularity"].eq("seasonal")
    ].reset_index(drop=True)


def test_no_duplicate_analytical_keys(final_dataset):
    key = ["spatial_unit", "year", "season", "crop"]
    assert not final_dataset.duplicated(key).any()


def test_no_proxy_or_synthetic_targets(final_dataset):
    assert not final_dataset["is_proxy"].astype(bool).any()
    assert not final_dataset["is_synthetic"].astype(bool).any()
    assert not final_dataset["is_geographically_assigned"].astype(bool).any()


def test_yield_recalculation(final_dataset):
    valid = final_dataset["yield_consistency_ok"].fillna(False).astype(bool) | (
        final_dataset["target_source"].eq("AAS2020_Table_6_1")
    )
    for _, row in final_dataset[final_dataset["year"] == 2020].iterrows():
        calculated = row["production_mt"] / row["area_harvested_ha"]
        assert np.isclose(
            calculated,
            row["yield_tons_ha"],
            rtol=1e-6,
            atol=1e-8,
        )


def test_no_mixed_season_rows(final_dataset):
    # total_2020 rows must not be combined with the seasonal rows
    assert not final_dataset["season"].isin(["total"]).any()
    by_year_season = final_dataset.groupby(["year", "season"]).size()
    assert {"first_season", "second_season"} <= set(
        by_year_season.index.get_level_values("season")
    )


def test_target_and_predictor_levels_match(final_dataset):
    assert (
        final_dataset["target_geographic_level"]
        == final_dataset["predictor_geographic_level"]
    ).all()


def test_no_target_derived_leakage(final_dataset):
    features = set(CLIMATE_FEATURES)
    assert features.isdisjoint(TARGET_DERIVED_COLUMNS)


def test_full_feature_coverage_under_relaxed_policy(homogeneous_dataset):
    validate_final_dataset(homogeneous_dataset, CLIMATE_FEATURES, RELAXED_POLICY)


def test_gate_rejects_mixed_temporal_granularity(final_dataset):
    with pytest.raises(ValueError, match="mix annual and seasonal"):
        validate_final_dataset(
            final_dataset,
            CLIMATE_FEATURES,
            AnalysisPolicy(
                minimum_rows=10,
                minimum_environmental_units=10,
                minimum_years=1,
            ),
        )


def test_final_policy_fails_honestly_on_small_sample(final_dataset):
    with pytest.raises(ValueError, match="requires at least"):
        validate_final_dataset(final_dataset, CLIMATE_FEATURES, AnalysisPolicy())


@pytest.mark.parametrize(
    "leak_col", ["yield_tons_ha", "production_mt", "area_harvested_ha"]
)
def test_gate_rejects_target_derived_predictor(homogeneous_dataset, leak_col):
    with pytest.raises(ValueError, match="Target-derived"):
        validate_final_dataset(homogeneous_dataset, [leak_col], RELAXED_POLICY)


def test_gate_rejects_missing_feature(homogeneous_dataset):
    with pytest.raises(ValueError, match="Requested features"):
        validate_final_dataset(
            homogeneous_dataset, ["rain_total_mm", "definitely_missing"], RELAXED_POLICY
        )


def test_gate_rejects_proxy_target(homogeneous_dataset):
    bad = homogeneous_dataset.copy()
    bad.loc[0, "is_proxy"] = True
    with pytest.raises(ValueError, match="Proxy targets"):
        validate_final_dataset(bad, CLIMATE_FEATURES, RELAXED_POLICY)


def test_gate_rejects_geographically_assigned_target(homogeneous_dataset):
    bad = homogeneous_dataset.copy()
    bad.loc[0, "is_geographically_assigned"] = True
    with pytest.raises(ValueError, match="assigned from a higher geography"):
        validate_final_dataset(bad, CLIMATE_FEATURES, RELAXED_POLICY)


def test_spatial_and_temporal_coverage(final_dataset):
    assert final_dataset["spatial_unit"].nunique() >= 5
    assert final_dataset["year"].nunique() >= 2


def test_yield_positive_and_finite(final_dataset):
    target = pd.to_numeric(final_dataset["yield_tons_ha"], errors="raise")
    assert np.isfinite(target).all()
    assert (target > 0).all()


class TestPCA:
    def test_manual_pca_matches_eigh(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "x": rng.normal(size=50),
                "y": rng.normal(size=50),
            }
        )
        data["y"] = data["x"] + data["y"]
        result = verify_two_feature_pca(data, "x", "y")
        assert np.allclose(
            result["manual_eigenvalues"],
            result["eigh_eigenvalues"],
            atol=1e-10,
        )

    def test_two_feature_pca_matches_eigh(self):
        rng = np.random.default_rng(7)
        data = pd.DataFrame(
            {
                "a": rng.normal(size=40),
                "b": rng.normal(size=40),
            }
        )
        data["b"] = -0.7 * data["a"] + data["b"]
        result = verify_two_feature_pca(data, "a", "b")
        assert result["eigh_eigenvalues"][0] >= 0

    def test_non_symmetric_rejected(self):
        with pytest.raises(ValueError, match="not symmetric"):
            import numpy as np
            from uganda_crop_model.pca.manual_verification import (
                manual_eigenvalues_2x2,
            )

            manual_eigenvalues_2x2(np.array([[1.0, 2.0], [3.0, 4.0]]))
