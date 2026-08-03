"""Contract tests for the leakage-safe modeling pipeline.

These tests exercise the modeling, validation, and evaluation modules on the
authoritative final dataset (skipped if it is not built) plus synthetic data
for the split and diagnostics components that need more years than the AAS
sample provides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from uganda_crop_model.evaluation.nested_cv import (
    run_nested_evaluation,
    summarize_out_of_fold_predictions,
)
from uganda_crop_model.evaluation.metadata import (
    build_quality_band,
    missingness_report,
)
from uganda_crop_model.evaluation.outliers import (
    robust_mahalanobis_flags,
)
from uganda_crop_model.evaluation.uncertainty import (
    temporal_split_conformal_predict,
)
from uganda_crop_model.models.pipelines import (
    CATEGORICAL_FEATURES,
    CLIMATE_FEATURES,
    build_preprocessor,
    resolve_feature_columns,
)
from uganda_crop_model.models.registry import get_model_registry
from uganda_crop_model.pca.diagnostics import (
    build_pca_loading_tables,
    parallel_analysis,
)
from uganda_crop_model.pca.stability import (
    align_components,
    bootstrap_pca_loadings,
)
from uganda_crop_model.validation.splits import (
    future_unseen_location_splits,
    rolling_origin_year_splits,
    spatial_group_splits,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
FINAL_MAIZE_DATASET = PROCESSED / "final_maize_subregion_season_year.csv"


@pytest.fixture(scope="module")
def final_dataset() -> pd.DataFrame:
    if not FINAL_MAIZE_DATASET.exists():
        pytest.skip("final dataset not built yet (run make data-final)")
    return pd.read_csv(FINAL_MAIZE_DATASET)


def _synthetic_panel(
    years: int = 6,
    spatial_units: int = 10,
    seasons: tuple[str, ...] = ("first_season", "second_season"),
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2015, 2015 + years):
        for unit in range(spatial_units):
            for season in seasons:
                for crop in ("maize",):
                    rows.append(
                        {
                            "spatial_unit": f"unit_{unit:02d}",
                            "year": year,
                            "season": season,
                            "crop": crop,
                            "rain_total_mm": rng.normal(800, 200),
                            "rainy_days_1mm": rng.normal(60, 20),
                            "heavy_rain_days_20mm": rng.normal(5, 2),
                            "maximum_5day_rainfall_mm": rng.normal(90, 25),
                            "longest_dry_spell_days": rng.normal(14, 5),
                            "wet_day_rainfall_cv": rng.normal(0.8, 0.3),
                            "temperature_mean_c": rng.normal(22, 3),
                            "temperature_maximum_c": rng.normal(28, 3),
                            "growing_degree_days": rng.normal(900, 200),
                            "heat_days_32c": rng.normal(20, 15),
                            "yield_tons_ha": rng.normal(2.0, 0.5),
                        }
                    )
    return pd.DataFrame(rows)


def _synthetic_year(
    years: int = 4,
    spatial_units: int = 8,
    seasons: tuple[str, ...] = ("first_season", "second_season"),
) -> pd.DataFrame:
    return _synthetic_panel(
        years=years,
        spatial_units=spatial_units,
        seasons=seasons,
    )


class TestFeatureColumns:
    def test_resolve_uses_only_present_columns(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        assert set(resolved["climate"]) <= set(final_dataset.columns)
        assert set(resolved["climate"]) <= set(CLIMATE_FEATURES)
        assert "soil_ph" in resolved["static"]

    def test_resolve_raises_without_climate(self):
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="climate"):
            resolve_feature_columns(df)


class TestPreprocessors:
    def test_raw_preprocessor_transforms(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        pre = build_preprocessor(
            "raw",
            feature_columns=resolved,
        )
        predictors = resolved["climate"] + resolved["static"] + resolved["categorical"]
        out = pre.fit_transform(final_dataset[predictors])
        assert out.shape[0] == len(final_dataset)
        assert np.isfinite(out).all()

    def test_pca_preprocessor_shrinks_columns(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        pre = build_preprocessor(
            "pca",
            feature_columns=resolved,
            pca_variance=0.90,
        )
        predictors = resolved["climate"] + resolved["static"] + resolved["categorical"]
        out = pre.fit_transform(final_dataset[predictors])
        n_climate = len(resolved["climate"])
        assert out.shape[1] <= n_climate + len(resolved["categorical"])

    def test_hybrid_preprocessor_works_without_static(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        pre = build_preprocessor(
            "hybrid",
            feature_columns=resolved,
        )
        predictors = resolved["climate"] + resolved["static"] + resolved["categorical"]
        out = pre.fit_transform(final_dataset[predictors])
        assert out.shape[0] == len(final_dataset)

    def test_hybrid_and_pca_are_distinct_after_soil_integration(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        predictors = resolved["climate"] + resolved["static"] + resolved["categorical"]
        pca = build_preprocessor("pca", feature_columns=resolved).fit_transform(
            final_dataset[predictors]
        )
        hybrid = build_preprocessor("hybrid", feature_columns=resolved).fit_transform(
            final_dataset[predictors]
        )
        assert resolved["static"]
        assert pca.shape != hybrid.shape or not np.allclose(pca, hybrid)


class TestSplits:
    def test_spatial_folds_do_not_overlap(self):
        df = _synthetic_year()
        splits = spatial_group_splits(df, requested_splits=5)
        assert len(splits) >= 2
        for train_index, test_index in splits:
            train_groups = set(df.iloc[train_index]["spatial_unit"])
            test_groups = set(df.iloc[test_index]["spatial_unit"])
            assert train_groups.isdisjoint(test_groups)

    def test_temporal_folds_never_train_on_future(self):
        df = _synthetic_year(years=6)
        splits = list(rolling_origin_year_splits(df, minimum_training_years=3))
        assert len(splits) >= 2
        for train_index, test_index in splits:
            max_train_year = df.iloc[train_index]["year"].max()
            min_test_year = df.iloc[test_index]["year"].min()
            assert max_train_year < min_test_year

    def test_rolling_origin_requires_years(self):
        df = _synthetic_year(years=2)
        with pytest.raises(ValueError, match="[Nn]ot enough years"):
            list(
                rolling_origin_year_splits(
                    df,
                    minimum_training_years=3,
                )
            )

    def test_future_unseen_location_never_leaks(self):
        df = _synthetic_year(years=5)
        splits = list(future_unseen_location_splits(df))
        assert splits
        for train_index, test_index in splits:
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            assert (train["year"] < test["year"].min()).all()
            test_unit = str(test["spatial_unit"].iloc[0])
            assert not {str(u) for u in train["spatial_unit"]}.intersection({test_unit})


def synthetic_registry_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame(
        {
            "spatial_unit": rng.choice([f"u{i}" for i in range(12)], n),
            "year": rng.choice([2018, 2019, 2020, 2021, 2022], n),
            "season": rng.choice(["first_season", "second_season"], n),
            "crop": "maize",
            "rain_total_mm": rng.normal(800, 200, n),
            "growing_degree_days": rng.normal(900, 200),
            "yield_tons_ha": rng.normal(2.0, 0.6),
        }
    ).sort_values(["spatial_unit", "year", "season"])


def test_model_registry_has_required_models():
    registry = get_model_registry(random_seed=42)
    expected = {
        "dummy_mean",
        "ols",
        "ridge",
        "random_forest",
        "xgboost",
    }
    assert expected.issubset(set(registry))


class TestPcaDiagnostics:
    def test_parallel_analysis_returns_positive_count(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 5))
        X[:, 1] = X[:, 0] + X[:, 1]
        result = parallel_analysis(X, iterations=50, random_seed=42)
        assert result["retained_components"] >= 1
        assert len(result["observed_eigenvalues"]) == 5
        assert len(result["null_threshold"]) == 5

    def test_build_pca_loading_tables_shapes(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(80, 4))
        X[:, 1] = X[:, 0] * 0.7
        pca = PCA(n_components=3, svd_solver="full").fit(X)
        features = ["a", "b", "c", "d"]
        loadings, contributions = build_pca_loading_tables(
            pca,
            features,
        )
        assert loadings.shape == (4, 3)
        assert contributions.shape == (4, 3)
        # each column of contributions should sum to ~100
        assert np.allclose(
            contributions.sum(axis=0),
            100.0,
            atol=1e-6,
        )

    def test_bootstrap_stability_reasonable(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(120, 4))
        result = bootstrap_pca_loadings(
            X,
            n_components=2,
            iterations=50,
            random_seed=42,
        )
        assert result["mean_components"].shape == (2, 4)
        assert result["lower_95"].shape == (2, 4)
        assert result["upper_95"].shape == (2, 4)

    def test_align_components_preserves_match(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=(2, 3))
        candidate = reference * -1.0
        aligned = align_components(reference, candidate)
        assert np.allclose(aligned, reference)


class TestNestedEvaluation:
    def test_nested_evaluation_end_to_end(self):
        df = _synthetic_panel(
            years=5,
            spatial_units=6,
            seasons=("first_season", "second_season"),
        )
        registry = get_model_registry(random_seed=42)
        outer = spatial_group_splits(df, requested_splits=4, random_seed=42)

        spec = registry["ridge"]
        predictions, fold_results = run_nested_evaluation(
            df,
            feature_space="raw",
            model_name="ridge",
            model_spec=spec,
            outer_splits=outer,
            inner_mode="spatial",
            random_seed=42,
        )

        assert len(fold_results) == len(outer)
        assert "observed_yield" in predictions.columns
        assert "predicted_yield" in predictions.columns
        assert np.isfinite(predictions["predicted_yield"]).all()
        assert (predictions["calibration_size"] > 0).all()
        assert (predictions["calibration_group_count"] > 0).all()
        assert (predictions["calibration_size"] < len(df)).all()
        assert {
            "training_global_mean",
            "training_crop_mean",
            "observed_evaluation_target",
            "predicted_evaluation_target",
        } <= set(predictions.columns)

        summary = summarize_out_of_fold_predictions(predictions)
        assert not summary.empty
        assert set(["rmse", "mae", "r2"]).issubset(summary.columns)
        assert {
            "skill_vs_training_global_mean",
            "skill_vs_training_crop_mean",
            "result_scope",
            "registered_primary_metric",
        } <= set(summary.columns)

    def test_inner_temporal_split_requires_years(self):
        from uganda_crop_model.evaluation.nested_cv import (
            build_inner_temporal_splits,
        )

        df = _synthetic_panel(years=2, spatial_units=3)
        with pytest.raises(ValueError, match="Insufficient years"):
            build_inner_temporal_splits(df)


class TestUncertaintyAndDiag:
    def test_conformal_interval_contains_prediction(self):
        rng = np.random.default_rng(1)
        n = 100
        panel = pd.DataFrame(
            {
                "year": np.repeat(range(2018, 2023), n // 5),
                "x": rng.normal(size=n),
                "y": rng.normal(size=n) + np.arange(n) % 5,
            }
        )
        from sklearn.linear_model import Ridge

        panel = panel.sort_values("year")
        train = panel.iloc[: n // 2]
        test = panel.iloc[n // 2 :]
        prediction, lower, upper = temporal_split_conformal_predict(
            Ridge(),
            train[["x"]],
            train["y"],
            train["year"],
            test[["x"]],
        )
        assert np.all(lower <= prediction + 1e-9)
        assert np.all(upper + 1e-9 >= prediction)

    def test_outlier_flags_basic(self, final_dataset):
        resolved = resolve_feature_columns(final_dataset)
        cols = resolved["climate"]
        flags = robust_mahalanobis_flags(
            final_dataset,
            cols,
        )
        assert set(flags.columns) == {
            "mahalanobis_distance_squared",
            "mahalanobis_threshold",
            "is_multivariate_outlier",
        }

    def test_missingness_report_shape(self, final_dataset):
        report = missingness_report(final_dataset)
        assert "missing_count" in report.columns
        assert "missing_fraction" in report.columns
        assert len(report) == final_dataset.shape[1]

    def test_quality_band_labels(self, final_dataset):
        valid = final_dataset.dropna(
            subset=["cv_area_harvested_pct", "cv_production_pct"]
        )
        band = build_quality_band(valid)
        assert band.isin(
            [
                "higher_precision",
                "moderate_precision",
                "low_precision",
                "very_low_precision",
            ]
        ).all()
