#!/usr/bin/env python3
"""
Build district-year soil moisture features from the local C3S monthly NetCDF
archives.

Input directories:
  - 1f321577cb54fac5d8c4b696f95c8978/
    - C3S-SOILMOISTURE-L3S-SSMV-COMBINED-MONTHLY-*.nc
    - C3S-RZSM-L3S-RZSMV-MONTHLY-*.nc

Outputs:
  - uganda_soil_moisture_features.csv
  - eastern_uganda_soil_moisture_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DATA_DIR = Path("1f321577cb54fac5d8c4b696f95c8978")
OUTPUT_FILE = Path("uganda_soil_moisture_features.csv")
EASTERN_OUTPUT_FILE = Path("eastern_uganda_soil_moisture_features.csv")

DISTRICTS = {
    "Mbale": (1.075, 34.175),
    "Kapchorwa": (1.400, 34.450),
    "Iganga": (0.617, 33.483),
    "Jinja": (0.425, 33.204),
    "Tororo": (0.693, 34.181),
    "Soroti": (1.715, 33.611),
    "Lira": (2.250, 32.917),
    "Gulu": (2.774, 32.299),
    "Mbarara": (-0.607, 30.658),
    "Arua": (3.020, 30.911),
    "Masaka": (-0.333, 31.733),
    "Fort_Portal": (0.671, 30.275),
    "Hoima": (1.433, 31.350),
    "Kabale": (-1.250, 29.983),
    "Kasese": (0.183, 30.083),
}

EASTERN_DISTRICTS = {"Mbale", "Kapchorwa", "Iganga", "Jinja", "Tororo"}
SEARCH_RADIUS_CELLS = 2


def get_season(month):
    if month in (3, 4, 5):
        return "MAM"
    if month in (9, 10, 11):
        return "SON"
    if month in (12, 1, 2):
        return "DJF"
    return "JJA"


def nearest_valid_value(data_array, lat, lon, radius_cells=SEARCH_RADIUS_CELLS):
    point = data_array.sel(lat=lat, lon=lon, method="nearest")
    value = point.values.squeeze()
    if not np.isnan(value):
        return float(value)

    lat_idx = int(np.abs(data_array["lat"].values - lat).argmin())
    lon_idx = int(np.abs(data_array["lon"].values - lon).argmin())

    best_value = np.nan
    best_distance = None

    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            i = lat_idx + di
            j = lon_idx + dj
            if i < 0 or j < 0 or i >= data_array.sizes["lat"] or j >= data_array.sizes["lon"]:
                continue

            candidate = data_array.isel(lat=i, lon=j).values.squeeze()
            if np.isnan(candidate):
                continue

            candidate_lat = float(data_array["lat"].isel(lat=i).values)
            candidate_lon = float(data_array["lon"].isel(lon=j).values)
            distance = (candidate_lat - lat) ** 2 + (candidate_lon - lon) ** 2

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_value = float(candidate)

    return best_value


def extract_monthly_records(surface_files, rootzone_files):
    rootzone_map = {path.name.replace("C3S-RZSM-L3S-RZSMV", ""): path for path in rootzone_files}
    records = []

    for surface_path in surface_files:
        suffix = surface_path.name.replace("C3S-SOILMOISTURE-L3S-SSMV-COMBINED", "")
        rootzone_path = rootzone_map.get(suffix)
        if rootzone_path is None:
            print(f"[WARN] No matching root-zone file for {surface_path.name}")
            continue

        with xr.open_dataset(surface_path) as surface_ds, xr.open_dataset(rootzone_path) as rootzone_ds:
            timestamp = pd.to_datetime(surface_ds["time"].values[0])

            for district, (lat, lon) in DISTRICTS.items():
                records.append(
                    {
                        "district": district,
                        "year": int(timestamp.year),
                        "month": int(timestamp.month),
                        "season": get_season(int(timestamp.month)),
                        "surface_sm": nearest_valid_value(surface_ds["sm"].isel(time=0), lat, lon),
                        "rzsm_1": nearest_valid_value(rootzone_ds["rzsm_1"].isel(time=0), lat, lon),
                        "rzsm_2": nearest_valid_value(rootzone_ds["rzsm_2"].isel(time=0), lat, lon),
                        "rzsm_3": nearest_valid_value(rootzone_ds["rzsm_3"].isel(time=0), lat, lon),
                        "rzsm_1m": nearest_valid_value(rootzone_ds["rzsm_1m"].isel(time=0), lat, lon),
                        "surface_nobs": nearest_valid_value(surface_ds["nobs"].isel(time=0), lat, lon),
                        "rzsm_1m_nobs": nearest_valid_value(rootzone_ds["nobs_1m"].isel(time=0), lat, lon),
                    }
                )

    return pd.DataFrame(records)


def build_feature_table(monthly):
    seasonal = (
        monthly.groupby(["district", "year", "season"])[["surface_sm", "rzsm_1m", "rzsm_1", "rzsm_2", "rzsm_3"]]
        .mean()
        .reset_index()
    )

    seasonal = seasonal.pivot_table(
        index=["district", "year"],
        columns="season",
        values=["surface_sm", "rzsm_1m", "rzsm_1", "rzsm_2", "rzsm_3"],
    )
    seasonal.columns = [f"{season}_{var}" for var, season in seasonal.columns]
    seasonal = seasonal.reset_index()

    annual = (
        monthly.groupby(["district", "year"])
        .agg(
            annual_surface_sm=("surface_sm", "mean"),
            annual_rzsm_1m=("rzsm_1m", "mean"),
            annual_rzsm_1=("rzsm_1", "mean"),
            annual_rzsm_2=("rzsm_2", "mean"),
            annual_rzsm_3=("rzsm_3", "mean"),
            surface_sm_cv=("surface_sm", lambda x: x.std() / x.mean() if x.mean() else 0.0),
            rzsm_1m_cv=("rzsm_1m", lambda x: x.std() / x.mean() if x.mean() else 0.0),
            min_surface_sm=("surface_sm", "min"),
            max_surface_sm=("surface_sm", "max"),
            min_rzsm_1m=("rzsm_1m", "min"),
            max_rzsm_1m=("rzsm_1m", "max"),
            annual_surface_nobs=("surface_nobs", "mean"),
            annual_rzsm_1m_nobs=("rzsm_1m_nobs", "mean"),
        )
        .reset_index()
    )

    features = seasonal.merge(annual, on=["district", "year"], how="inner", validate="one_to_one")
    features["soil_moisture_index"] = features["annual_rzsm_1m"]
    return features.sort_values(["district", "year"]).reset_index(drop=True)


def main():
    print("=" * 70)
    print("  BUILD UGANDA SOIL MOISTURE FEATURES")
    print("=" * 70)

    surface_files = sorted(DATA_DIR.glob("C3S-SOILMOISTURE-L3S-SSMV-COMBINED-MONTHLY-*.nc"))
    rootzone_files = sorted(DATA_DIR.glob("C3S-RZSM-L3S-RZSMV-MONTHLY-*.nc"))

    if not surface_files or not rootzone_files:
        raise FileNotFoundError("Soil moisture NetCDF archives were not found.")

    print(f"[INFO] Surface soil moisture files: {len(surface_files)}")
    print(f"[INFO] Root-zone soil moisture files: {len(rootzone_files)}")

    monthly = extract_monthly_records(surface_files, rootzone_files)
    print(f"[✓] Extracted monthly records: {len(monthly)}")

    features = build_feature_table(monthly)
    features.to_csv(OUTPUT_FILE, index=False)
    print(f"[✓] Saved: {OUTPUT_FILE}")
    print(f"[✓] Shape: {features.shape[0]} rows x {features.shape[1]} columns")
    print(features.head(5).to_string(index=False))

    eastern = features[features["district"].isin(EASTERN_DISTRICTS)].reset_index(drop=True)
    eastern.to_csv(EASTERN_OUTPUT_FILE, index=False)
    print()
    print(f"[✓] Saved: {EASTERN_OUTPUT_FILE}")
    print(f"[✓] Shape: {eastern.shape[0]} rows x {eastern.shape[1]} columns")
    print(eastern.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
