"""Merge the AAS target with sub-region climate features into the final dataset.

The single authoritative analytical dataset is
``data/processed/final_maize_subregion_season_year.csv``:

* grain:        spatial_unit x year x season x crop
* target:       yield_tons_ha (AAS official, production / harvested area)
* predictors:   sub-region rainfall and temperature summarised over the same
                season windows

Production/area columns are retained for audit but MUST NOT be used as
predictors (the quality gate forbids target-derived leakage).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build_aas_targets import (
    build_combined_maize_targets,
    build_combined_multi_crop_targets,
)
from .paths import (
    AAS2018_MAIZE,
    AAS2020_CROP_TABLES,
    AAS2020_MAIZE,
    FINAL_MAIZE_DATASET,
    FINAL_MULTI_CROP_DATASET,
    INTERIM,
    PUBLIC,
)
from .season_calendar import make_season_calendar
from .subregion_soil import build_optional_elevation_features, build_subregion_soil_features
from ..features.rainfall import build_seasonal_rainfall_features
from ..features.temperature import build_seasonal_temperature_features


def build_final_maize_dataset(
    *,
    aas2020_file: Path = AAS2020_MAIZE,
    aas2018_file: Path = AAS2018_MAIZE,
    district_map_file: Path = INTERIM / "uganda_districts_114.csv",
    daily_rain_file: Path = INTERIM / "subregion_daily_rainfall.csv",
    daily_temp_file: Path = INTERIM / "subregion_daily_temperature.csv",
    soil_file: Path = INTERIM / "uganda_soil_features_114.csv",
    elevation_file: Path | None = INTERIM / "uganda_elevation_features_114.csv",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Assemble the final sub-region season-year maize dataset."""

    targets = build_combined_maize_targets(aas2020_file, aas2018_file)
    return _merge_climate_and_targets(
        targets,
        district_map_file=district_map_file,
        daily_rain_file=daily_rain_file,
        daily_temp_file=daily_temp_file,
        soil_file=soil_file,
        elevation_file=elevation_file,
        years=years,
    )


def build_final_multi_crop_dataset(
    *,
    aas2020_crop_tables: dict[str, Path] | None = None,
    aas2018_file: Path = AAS2018_MAIZE,
    district_map_file: Path = INTERIM / "uganda_districts_114.csv",
    daily_rain_file: Path = INTERIM / "subregion_daily_rainfall.csv",
    daily_temp_file: Path = INTERIM / "subregion_daily_temperature.csv",
    soil_file: Path = INTERIM / "uganda_soil_features_114.csv",
    elevation_file: Path | None = INTERIM / "uganda_elevation_features_114.csv",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Assemble the final sub-region season-year multi-crop dataset.

    The eligible crop set is the ten food crops that report both production
    and harvested area in AAS 2018 and AAS 2020, so every row uses the same
    ``production / harvested area`` yield definition.
    """

    crop_tables = aas2020_crop_tables or dict(AAS2020_CROP_TABLES)
    targets = build_combined_multi_crop_targets(
        crop_tables,
        aas2018_file,
    )
    return _merge_climate_and_targets(
        targets,
        district_map_file=district_map_file,
        daily_rain_file=daily_rain_file,
        daily_temp_file=daily_temp_file,
        soil_file=soil_file,
        elevation_file=elevation_file,
        years=years,
    )


def _merge_climate_and_targets(
    targets: pd.DataFrame,
    *,
    district_map_file: Path,
    daily_rain_file: Path,
    daily_temp_file: Path,
    soil_file: Path,
    elevation_file: Path | None,
    years: list[int] | None,
) -> pd.DataFrame:
    """Attach the shared sub-region climate blocks to a target frame."""

    calendar = make_season_calendar(
        district_map_file,
        years=years,
    )

    rainfall = build_seasonal_rainfall_features(
        pd.read_csv(daily_rain_file),
        calendar,
    )
    temperature = build_seasonal_temperature_features(
        pd.read_csv(daily_temp_file),
        calendar,
    )

    # The climate blocks share the same predictor_geographic_level constant;
    # drop the duplicated column before merging to avoid suffix collisions.
    rainfall = rainfall.drop(columns=["predictor_geographic_level"])
    temperature = temperature.drop(columns=["predictor_geographic_level"])

    climate = rainfall.merge(
        temperature,
        on=["spatial_unit", "year", "season"],
        how="outer",
    )

    merged = targets.merge(
        climate,
        on=["spatial_unit", "year", "season"],
        how="left",
    )

    soil = build_subregion_soil_features(soil_file, district_map_file)
    elevation = build_optional_elevation_features(elevation_file, district_map_file)
    merged = merged.merge(soil, on="spatial_unit", how="left", validate="many_to_one")
    if not elevation.empty:
        merged = merged.merge(elevation, on="spatial_unit", how="left", validate="many_to_one")
    merged["predictor_geographic_level"] = "sub_region"
    merged["elevation_source"] = merged.get("elevation_source", "not_available")

    return pin_order(merged)


def pin_order(df: pd.DataFrame, data_version: str = "aas-chirps-v1") -> pd.DataFrame:
    """Assign a stable column order and recorded data versions."""
    same_column_order = [
        "spatial_unit", "year", "season", "crop",
        "yield_tons_ha", "yield_tons_ha_planted", "harvested_fraction",
        "area_planted_ha", "area_harvested_ha", "production_mt",
        "cv_area_planted_pct", "cv_area_harvested_pct", "cv_production_pct",
        "rain_total_mm", "rain_mean_daily_mm",
        "rainy_days_1mm", "rainy_days_10mm", "heavy_rain_days_20mm",
        "maximum_1day_rainfall_mm", "maximum_5day_rainfall_mm",
        "longest_dry_spell_days", "wet_day_rainfall_cv",
        "rainfall_onset_date", "rainfall_cessation_date",
        "onset_day_of_year", "cessation_day_of_year", "season_length_days",
        "temperature_mean_c", "temperature_maximum_c",
        "temperature_minimum_c", "temperature_range_c",
        "growing_degree_days", "heat_days_32c",
        "extreme_heat_days_35c", "cold_days_10c",
        "target_source", "target_source_type",
        "target_geographic_level", "predictor_geographic_level",
        "target_definition", "target_year",
        "target_season",
        "is_proxy", "is_synthetic", "is_geographically_assigned",
        "season_definition", "yield_consistency_ok",
        "rainfall_source", "temperature_source",
        "soil_source", "soil_source_version", "soil_depth_cm",
        "soil_quality_flag", "soil_district_count",
        "soil_ph", "soil_ph_sd", "soil_organic_carbon", "soil_organic_carbon_sd",
        "clay_pct", "clay_pct_sd", "sand_pct", "sand_pct_sd", "silt_pct", "silt_pct_sd",
        "bulk_density", "bulk_density_sd", "cation_exchange_capacity", "cation_exchange_capacity_sd",
        "elevation_m", "elevation_m_sd", "elevation_source",
        "data_version", "processing_version",
    ]

    df = df.copy()
    df["data_version"] = data_version
    df["processing_version"] = "uganda_crop_model-0.2.0"

    missing = [c for c in df.columns if c not in set(same_column_order)]
    order = [c for c in same_column_order if c in df.columns] + missing

    return df[order].sort_values(
        ["spatial_unit", "year", "season", "crop"]
    ).reset_index(drop=True)


def save_final_maize_dataset(output: Path | None = None) -> pd.DataFrame:
    from uganda_crop_model.data.paths import ensure_dirs

    ensure_dirs()
    df = build_final_maize_dataset()
    output = output or FINAL_MAIZE_DATASET
    df.to_csv(output, index=False)
    return df


def save_final_multi_crop_dataset(output: Path | None = None) -> pd.DataFrame:
    from uganda_crop_model.data.paths import ensure_dirs

    ensure_dirs()
    df = build_final_multi_crop_dataset()
    output = output or FINAL_MULTI_CROP_DATASET
    df.to_csv(output, index=False)
    return df


if __name__ == "__main__":
    df = save_final_maize_dataset()
    print(f"[✓] Final maize dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df[["year", "season"]].value_counts().sort_index().to_string())
