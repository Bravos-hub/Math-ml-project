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
