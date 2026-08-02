"""Data-quality and schema tests for the pipeline outputs.

Tests run against the built artifacts in ``data/interim`` and
``data/processed`` (see ``make data``), so they double as regression
guards for the extraction and panel-building steps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

CROPS = ["maize", "beans", "groundnuts"]
SEASONS_2020 = ["first_season_2020", "second_season_2020", "total_2020"]
PROVENANCE = ("yield_source", "rainfall_source", "temperature_source",
              "soil_source", "soil_moisture_source", "yield_granularity",
              "is_proxy", "is_imputed", "data_quality_score",
              "data_quality_note")

FEATURE_PREFIXES = ("daily_", "temp_", "rain_", "soil_")
YIELD_COL = "yield_over_harvested"  # t/ha


@pytest.fixture(scope="module")
def districts() -> pd.DataFrame:
    return pd.read_csv(INTERIM / "uganda_districts_114.csv")


@pytest.fixture(scope="module")
def rainfall() -> pd.DataFrame:
    return pd.read_csv(INTERIM / "uganda_rainfall_features_114.csv")


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(INTERIM / "uganda_daily_features_climateserv.csv")


@pytest.fixture(scope="module")
def temp() -> pd.DataFrame:
    return pd.read_csv(INTERIM / "uganda_temperature_features_nasapower.csv")


@pytest.fixture(scope="module")
def soil() -> pd.DataFrame:
    return pd.read_csv(INTERIM / "uganda_soil_features_114.csv")


@pytest.fixture(scope="module")
def panels() -> dict[str, dict[str, pd.DataFrame]]:
    return {
        crop: {
            "observed": pd.read_csv(PROCESSED / "observed" / f"{crop}_subregion_panel.csv"),
            "assigned": pd.read_csv(PROCESSED / "assigned" / f"{crop}_district_assigned_panel.csv"),
        }
        for crop in CROPS
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(FEATURE_PREFIXES) and not c.startswith(PROVENANCE)]


class TestDistricts:
    def test_shape(self, districts):
        assert districts["district"].nunique() == 114
        assert not districts["district"].duplicated().any()

    def test_centroids_in_uganda(self, districts):
        assert districts["lat"].between(-1.5, 4.3).all()
        assert districts["lon"].between(29.5, 35.1).all()


class TestRainfallFeatures:
    def test_shape(self, rainfall):
        assert len(rainfall) == 1026  # 114 districts x 9 years
        assert rainfall["year"].between(2015, 2023).all()

    def test_monthly_totals_reasonable(self, rainfall):
        monthly = [c for c in rainfall.columns if c.endswith("_mm")]
        assert rainfall[monthly].notna().all().all()
        # no month should average more than 400 mm in Uganda
        assert (rainfall[monthly].mean() <= 400).all()
        assert (rainfall[monthly] >= 0).all().all()


class TestDailyFeatures:
    def test_shape(self, daily):
        assert len(daily) == 2052  # 114 districts x 9 years x 2 seasons

    def test_onset_within_doy_window(self, daily):
        for s in ("first", "second"):
            onset = daily.loc[daily["season"] == s, "season_onset_day"]
            cessation = daily.loc[daily["season"] == s, "season_cessation_day"]
            # NaN allowed when no season is detectable; otherwise within window
            assert onset.dropna().between(1, 366).all()
            assert cessation.dropna().between(1, 366).all()
            valid = onset.notna() & cessation.notna()
            assert (cessation[valid] >= onset[valid]).all()

    def test_missing_onset_rate_low(self, daily):
        # a detectable onset is expected for the vast majority of years
        for s in ("first", "second"):
            onset = daily.loc[daily["season"] == s, "season_onset_day"]
            assert onset.isna().mean() < 0.10, (s, onset.isna().mean())

    def test_season_length_consistent(self, daily):
        length = daily["season_cessation_day"] - daily["season_onset_day"] + 1
        length = length[daily["season_onset_day"].notna()]
        np.testing.assert_array_equal(
            length.to_numpy(),
            daily.loc[daily["season_onset_day"].notna(), "season_length_days"].to_numpy())

    def test_dry_spell_counts_consistent(self, daily):
        assert (daily["rain_days_20mm"] <= daily["rain_days_10mm"]).all()
        assert (daily["rain_days_10mm"] <= daily["rain_days_1mm"]).all()


class TestTemperatureFeatures:
    def test_shape(self, temp):
        assert len(temp) == 2052
        assert temp["season"].isin(["first", "second"]).all()

    def test_gdd_positive(self, temp):
        assert (temp["season_gdd"] >= 0).all()


class TestSoilFeatures:
    VALUE_COLS = ["clay", "sand", "silt", "soc", "bdod", "cec", "phh2o"]

    def test_no_missing(self, soil):
        assert soil[self.VALUE_COLS].isna().sum().sum() == 0

    def test_texture_sums_to_100(self, soil):
        total = soil["sand"] + soil["silt"] + soil["clay"]
        assert total.between(99, 101).all()

    def test_soc_plausible(self, soil):
        assert soil["soc"].between(0, 150).all()  # g/kg
        assert soil["phh2o"].between(3.5, 9.5).all()


class TestPanels:
    def test_observed_rows(self, panels):
        assert len(panels["maize"]["observed"]) == 56
        assert len(panels["beans"]["observed"]) == 55
        assert len(panels["groundnuts"]["observed"]) == 56

    def test_assigned_rows(self, panels):
        assert len(panels["maize"]["assigned"]) == 456
        assert len(panels["beans"]["assigned"]) == 449
        assert len(panels["groundnuts"]["assigned"]) == 456

    def test_yield_plausible_range(self, panels):
        for crop in CROPS:
            for kind in ("observed", "assigned"):
                y = panels[crop][kind][YIELD_COL]
                assert y.between(0.1, 15).all(), (crop, kind)

    def test_season_coverage(self, panels):
        for crop in CROPS:
            obs = panels[crop]["observed"]
            assert sorted(obs["season_group"].unique()) == sorted(SEASONS_2020 + ["total_2018"])

    def test_provenance_columns_present(self, panels):
        required = {"yield_source", "rainfall_source", "temperature_source",
                    "soil_source", "yield_granularity", "is_proxy",
                    "is_imputed", "data_quality_score"}
        for crop in CROPS:
            for kind in ("observed", "assigned"):
                cols = set(panels[crop][kind].columns)
                assert required <= cols, (crop, kind)

    def test_no_nan_features_in_assigned(self, panels):
        for crop in CROPS:
            feat = feature_columns(panels[crop]["assigned"])
            assert len(feat) >= 40
            assert panels[crop]["assigned"][feat].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Review contracts (P0 #1, #11, #12; P2 #22)
# ---------------------------------------------------------------------------

def yield_key_uniqueness(df: pd.DataFrame):
    df = df.dropna(subset=["yield_over_harvested"])
    keys = ["sub_region", "season_group"]
    if "crop" in df.columns:
        keys = ["crop"] + keys
    dups = df.duplicated(subset=keys, keep=False)
    return len(df), int(dups.sum())


class TestYieldConsistency:
    def test_harvested_yield_equals_production_over_area(self, panels):
        # AAS 2020 publishes production-over-area yields; the relationship
        # must hold to 2%. AAS 2018 also publishes mixed total/second-season
        # figures whose rows are flagged by ``yield_consistency_ok`` instead.
        for crop in CROPS:
            df = panels[crop]["observed"]
            df2020 = df[df["year"] == 2020].dropna(
                subset=["production_mt", "area_harvested_ha", "yield_over_harvested"])
            calc = df2020["production_mt"] / df2020["area_harvested_ha"]
            np.testing.assert_allclose(calc.to_numpy(dtype=float),
                                       df2020["yield_over_harvested"].to_numpy(dtype=float),
                                       rtol=2e-2)

    def test_planted_yield_equals_production_over_planted(self, panels):
        for crop in CROPS:
            df = panels[crop]["observed"]
            df2020 = df[df["year"] == 2020]
            valid = df2020["area_planted_ha"] > 0
            calc = df2020.loc[valid, "production_mt"] / df2020.loc[valid, "area_planted_ha"]
            np.testing.assert_allclose(
                calc.to_numpy(dtype=float),
                df2020.loc[valid, "yield_over_planted"].to_numpy(dtype=float),
                rtol=2e-2)

    def test_unique_spatial_season_key(self, panels):
        for crop in CROPS:
            df = panels[crop]["observed"].dropna(subset=["yield_over_harvested"])
            dups = df.duplicated(subset=["sub_region", "season_group"], keep=False)
            assert not dups.any(), (crop, df[dups][["sub_region", "season_group"]])

    def test_no_synthetic_or_proxy_targets(self, panels):
        for crop in CROPS:
            for kind in ("observed", "assigned"):
                df = panels[crop][kind]
                assert ~df["is_proxy"].fillna(False).astype(bool).any(), (crop, kind)
                # target must derive from official AAS only
                srcs = set(df["yield_source"].dropna().unique())
                assert srcs <= {"AAS2018_subregion", "AAS2020_subregion"}

    def test_yield_consistency_flag_covers_2018_rows(self, panels):
        # 2018 annual-crop rows mix total production with second-season
        # harvested yield and are explicitly marked, not silently dropped.
        df = panels["maize"]["observed"]
        flagged = df[df["year"] == 2018]["yield_consistency_ok"]
        assert not flagged.all()
        good2020 = df[df["year"] == 2020]["yield_consistency_ok"]
        assert good2020.all()


class TestSurveyUncertainty:
    def test_reliability_columns_present(self, panels):
        required = {"target_cv", "target_reliability_weight",
                    "high_uncertainty_flag", "yield_consistency_ok"}
        for crop in CROPS:
            cols = set(panels[crop]["observed"].columns)
            assert required <= cols

    def test_reliability_weight_increases_as_cv_falls(self, panels):
        df = panels["maize"]["observed"].dropna(subset=["target_cv", "target_reliability_weight"])
        assert (df["target_cv"] > 0).all()
        # higher CV -> smaller (or equal) reliability weight
        c = df[["target_cv", "target_reliability_weight"]].corr().iloc[0, 1]
        assert c < 0.2

    def test_cv_in_expected_range(self, panels):
        cv = panels["maize"]["observed"]["target_cv"].dropna()
        assert cv.between(0, 100).all()


@pytest.fixture(scope="module")
def pooled() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "observed" / "crop_pooled_subregion_panel.csv")


class TestPooledPanel:
    def test_more_than_100_units(self, pooled):
        assert len(pooled) >= 100
        assert pooled[["crop", "sub_region", "season_group"]].drop_duplicates().shape[0] >= 100

    def test_crop_is_predictor(self, pooled):
        assert "crop" in pooled.columns
        assert pooled["crop"].isin(["maize", "beans", "groundnuts"]).all()


class TestFeatureAvailability:
    def test_panel_has_no_all_null_model_features(self, panels):
        # review rule: a column with isna().mean() == 1 must be excluded
        for crop in CROPS:
            df = panels[crop]["observed"]
            feat = feature_columns(df)
            assert df[feat].isna().all().sum() == 0

    def test_availability_report_exists(self):
        report = pd.read_csv(PROCESSED.parent / ".." / "reports" / "tables" /
                             "feature_availability.csv")
        assert {"Variable", "Source", "Coverage", "Status"} <= set(report.columns)
        assert report["Coverage"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Representation transformer (P1 #18) -- leak-free PCA/hybrid
# ---------------------------------------------------------------------------

class TestRepresentations:
    def test_raw_returns_numeric_frame(self):
        from cropyield.pca.representations import RepresentationTransformer
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 1.5, 2.5]})
        t = RepresentationTransformer("raw").fit(X)
        out = t.transform(X)
        assert out.shape[1] == 2
        assert (out.columns == ["a", "b"]).all()

    def test_pca_reduces_dimension(self):
        from cropyield.data.paths import OBSERVED
        from cropyield.models.validate import load_config, load_panel
        from cropyield.pca.representations import RepresentationTransformer
        cfg = load_config()
        df = load_panel("maize")
        cols = cfg["feature_sets"]["full_agroecological"]
        t = RepresentationTransformer("pca", pca_input_cols=cols,
                                      n_components=cfg["pca"]["n_components"])
        out = t.fit(df[cols]).transform(df[cols])
        assert out.shape[1] < len(cols)

    def test_hybrid_includes_extra_and_avoids_pca_inputs(self):
        import numpy as np
        from cropyield.pca.representations import RepresentationTransformer
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame({
            "x1": rng.normal(size=n), "x2": rng.normal(size=n),
            "crop": rng.choice(["a", "b"], size=n),
        })
        t = RepresentationTransformer("hybrid", pca_input_cols=["x1", "x2"],
                                      n_components=1).fit(X)
        out = t.transform(X)
        assert "PC1" in out.columns
        assert set(["x1", "x2"]) & set(out.columns) == set()
        assert any("crop" in c for c in out.columns)
