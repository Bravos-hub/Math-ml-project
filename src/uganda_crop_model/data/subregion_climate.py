"""Aggregate district-level daily climate to the subregion analytical grain.

The AAS target is reported at the 14-UBOS-sub-region level (see
``data/interim/uganda_districts_114.csv`` for the district->sub_region
membership map).  Daily CHIRPS rainfall and NASA POWER temperature are
presently available as district-point time series
(``uganda_daily_rainfall_climateserv.csv`` and
``uganda_daily_temperature_nasapower.csv`` from the ClimateSERV / POWER
APIs), so the sub-region daily series is computed as the unweighted mean of
its member districts for each date.

This is a transparent, documented approximation: it is the district-to-
subregion aggregate of the blueprint's polygon-average design.  The
geographic level of the predictors remains ``sub_region`` so that target
and predictors match.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import DAILY_RAINFALL, DAILY_TEMPERATURE, DISTRICT_MAP, INTERIM

RAINFALL_SOURCE = "CHIRPS_v2.0_daily_via_ClimateSERV"
TEMPERATURE_SOURCE = "NASA_POWER_daily"


def load_district_map(path: Path = DISTRICT_MAP) -> pd.DataFrame:
    """Load the district -> sub_region membership map."""
    df = pd.read_csv(path)
    df = df[["district", "sub_region"]].drop_duplicates()
    if df["district"].duplicated().any():
        raise ValueError("District map contains duplicate district names.")
    return df


def aggregate_daily_rainfall_to_subregion(
    district_map: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    daily_path: Path = DAILY_RAINFALL,
) -> pd.DataFrame:
    """Aggregate daily district rainfall to a daily sub-region mean.

    Input long-format columns: ``district``, ``date``, ``rain_mm``.
    Output columns: ``spatial_unit``, ``date``, ``rain_mm``.
    """
    if daily is None:
        daily = pd.read_csv(daily_path)

    for column in ("district", "date", "rain_mm"):
        if column not in daily.columns:
            raise ValueError(f"Daily rainfall missing column: {column}")

    df = daily.copy()
    df = df.merge(district_map, on="district", how="inner")
    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={"sub_region": "spatial_unit"})

    aggregated = (
        df.groupby(["spatial_unit", "date"], as_index=False)["rain_mm"].mean()
    )

    return aggregated.sort_values(
        ["spatial_unit", "date"]
    ).reset_index(drop=True)


def aggregate_daily_temperature_to_subregion(
    district_map: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    daily_path: Path = DAILY_TEMPERATURE,
) -> pd.DataFrame:
    """Aggregate daily temperature to a sub-region daily mean.

    Input columns: ``district``, ``date``, ``tmax_c``, ``tmin_c``.
    Output columns: ``spatial_unit``, ``date``, ``tmax_c``, ``tmin_c``.
    """
    if daily is None:
        daily = pd.read_csv(daily_path)

    for column in ("district", "date", "tmax_c", "tmin_c"):
        if column not in daily.columns:
            raise ValueError(f"Daily temperature missing column: {column}")

    df = daily.copy()
    df = df.merge(district_map, on="district", how="inner")
    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={"sub_region": "spatial_unit"})

    aggregated = (
        df.groupby(["spatial_unit", "date"], as_index=False)[
            ["tmax_c", "tmin_c"]
        ].mean()
    )

    return aggregated.sort_values(
        ["spatial_unit", "date"]
    ).reset_index(drop=True)


def save_subregion_daily(
    district_map: pd.DataFrame,
    output_rain: Path | None = None,
    output_temp: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache the daily sub-region rainfall and temperature series."""
    rainfall = aggregate_daily_rainfall_to_subregion(district_map)
    temperature = aggregate_daily_temperature_to_subregion(district_map)

    output_rain = output_rain or INTERIM / "subregion_daily_rainfall.csv"
    output_temp = output_temp or INTERIM / "subregion_daily_temperature.csv"

    rainfall.to_csv(output_rain, index=False)
    temperature.to_csv(output_temp, index=False)

    return rainfall, temperature


if __name__ == "__main__":
    from uganda_crop_model.data.paths import ensure_dirs

    ensure_dirs()
    district_map = load_district_map()
    rainfall, temperature = save_subregion_daily(district_map)
    print(f"[✓] Sub-region daily rainfall: {rainfall.shape[0]} rows")
    print(f"[✓] Sub-region daily temperature: {temperature.shape[0]} rows")