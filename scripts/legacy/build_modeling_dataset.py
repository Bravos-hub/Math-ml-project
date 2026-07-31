#!/usr/bin/env python3
"""
Build a canonical district-year modeling dataset for the Uganda agriculture
project.

This script enforces the schema defined in DATA_DICTIONARY.md. It:
  1. Loads rainfall, temperature, soil, and yield tables when available.
  2. Normalizes district names and year types.
  3. Merges sources into a single district-year panel.
  4. Adds planned optional columns as empty placeholders when absent.
  5. Writes consistent output tables for downstream PCA and prediction code.

Outputs:
  - uganda_modeling_dataset_canonical.csv
  - eastern_uganda_maize_modeling_dataset.csv
  - eastern_uganda_maize_modeling_dataset_2020_2023.csv
"""

from pathlib import Path

import pandas as pd


OUTPUT_FILE = Path("uganda_modeling_dataset_canonical.csv")
EASTERN_OUTPUT_FILE = Path("eastern_uganda_maize_modeling_dataset.csv")
EASTERN_OVERLAP_OUTPUT_FILE = Path("eastern_uganda_maize_modeling_dataset_2020_2023.csv")

RAINFALL_FILE = Path("uganda_rainfall_features.csv")
TEMPERATURE_FILE = Path("uganda_temperature_features.csv")
SOIL_FILE = Path("uganda_soil_features.csv")
SOIL_MOISTURE_FILE = Path("uganda_soil_moisture_features.csv")
YIELD_FILES = [
    Path("ubos_maize_yield_district.csv"),
    Path("ubos_district_yield_proxy.csv"),
    Path("uganda_full_pipeline_data.csv"),
]

EASTERN_DISTRICTS = ["Mbale", "Kapchorwa", "Iganga", "Jinja", "Tororo"]
OVERLAP_YEARS = (2020, 2023)

CANONICAL_COLUMNS = [
    "district",
    "year",
    "yield_tons_ha",
    "MAM",
    "SON",
    "DJF",
    "JJA",
    "annual_rainfall",
    "rain_cv",
    "max_monthly",
    "min_monthly",
    "rainy_months",
    "MAM_tmax",
    "MAM_tmin",
    "MAM_gdd",
    "MAM_heat_stress",
    "MAM_cold_stress",
    "SON_tmax",
    "SON_tmin",
    "SON_gdd",
    "SON_heat_stress",
    "SON_cold_stress",
    "DJF_tmax",
    "DJF_tmin",
    "DJF_gdd",
    "DJF_heat_stress",
    "DJF_cold_stress",
    "JJA_tmax",
    "JJA_tmin",
    "JJA_gdd",
    "JJA_heat_stress",
    "JJA_cold_stress",
    "annual_tmax",
    "annual_tmin",
    "annual_gdd",
    "annual_heat_stress",
    "annual_cold_stress",
    "elevation_m",
    "lat",
    "lon",
    "phh2o",
    "soc",
    "clay",
    "sand",
    "silt",
    "bdod",
    "cec",
    "source_status",
    "n_households",
    "total_area_ha",
    "total_production_kg",
    "fertilizer_kg_ha",
    "planting_density",
    "ndvi_peak",
    "evi_peak",
    "soil_moisture_index",
    "planting_date",
    "solar_radiation",
]


def normalize_district(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    return cleaned.str.title()


def prepare_keys(df):
    df = df.copy()
    df["district"] = normalize_district(df["district"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def load_rainfall():
    require_file(RAINFALL_FILE)
    df = pd.read_csv(RAINFALL_FILE)
    required = {
        "district", "year", "MAM", "SON", "DJF", "JJA",
        "annual_rainfall", "rain_cv", "max_monthly", "min_monthly",
        "rainy_months",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Rainfall file missing columns: {sorted(missing)}")
    return prepare_keys(df)


def load_temperature():
    require_file(TEMPERATURE_FILE)
    df = pd.read_csv(TEMPERATURE_FILE)
    required = {
        "district", "year", "annual_gdd", "annual_tmax", "annual_tmin",
        "elevation_m",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Temperature file missing columns: {sorted(missing)}")
    return prepare_keys(df)


def load_soil():
    if not SOIL_FILE.exists():
        print(f"[WARN] Soil file not found: {SOIL_FILE}")
        return None

    df = pd.read_csv(SOIL_FILE)
    if "district" not in df.columns:
        raise ValueError("Soil file missing district column")
    return prepare_keys(df)


def load_soil_moisture():
    if not SOIL_MOISTURE_FILE.exists():
        print(f"[WARN] Soil moisture file not found: {SOIL_MOISTURE_FILE}")
        return None

    df = pd.read_csv(SOIL_MOISTURE_FILE)
    required = {"district", "year", "soil_moisture_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Soil moisture file missing columns: {sorted(missing)}")

    keep = ["district", "year", "soil_moisture_index"]
    return prepare_keys(df[keep])


def load_yield():
    for path in YIELD_FILES:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if not {"district", "year"}.issubset(df.columns):
            print(f"[WARN] Skipping {path}: missing district/year")
            continue

        if "yield_tons_ha" not in df.columns:
            if "yield_weighted" in df.columns:
                df = df.rename(columns={"yield_weighted": "yield_tons_ha"})
            else:
                print(f"[WARN] Skipping {path}: no yield column")
                continue

        keep = ["district", "year", "yield_tons_ha"]
        for col in ("n_households", "total_area_ha", "total_production_kg"):
            if col in df.columns:
                keep.append(col)

        print(f"[INFO] Using yield source: {path}")
        return prepare_keys(df[keep])

    print("[WARN] No yield file found. Yield columns will remain empty.")
    return None


def add_missing_columns(df):
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def enforce_column_order(df):
    return df[CANONICAL_COLUMNS].sort_values(["district", "year"]).reset_index(drop=True)


def build_dataset():
    rainfall = load_rainfall()
    temperature = load_temperature()
    base = rainfall.merge(
        temperature,
        on=["district", "year"],
        how="inner",
        validate="one_to_one",
    )

    soil = load_soil()
    if soil is not None:
        soil = soil.drop_duplicates(subset=["district"])
        base = base.merge(soil, on="district", how="left", validate="many_to_one")

    soil_moisture = load_soil_moisture()
    if soil_moisture is not None:
        base = base.merge(
            soil_moisture,
            on=["district", "year"],
            how="left",
            suffixes=("", "_sm"),
            validate="one_to_one",
        )
        if "soil_moisture_index_sm" in base.columns:
            base["soil_moisture_index"] = base["soil_moisture_index"].fillna(base["soil_moisture_index_sm"])
            base = base.drop(columns=["soil_moisture_index_sm"])

    yield_df = load_yield()
    if yield_df is not None:
        base = base.merge(yield_df, on=["district", "year"], how="left", validate="one_to_one")

    dataset = add_missing_columns(base)
    dataset = enforce_column_order(dataset)
    return dataset


def main():
    print("=" * 70)
    print("  BUILD CANONICAL UGANDA MODELING DATASET")
    print("=" * 70)

    dataset = build_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"[✓] Saved canonical dataset: {OUTPUT_FILE}")
    print(f"[✓] Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(dataset.head(5).to_string(index=False))

    eastern = dataset[dataset["district"].isin(EASTERN_DISTRICTS)].reset_index(drop=True)
    eastern.to_csv(EASTERN_OUTPUT_FILE, index=False)
    print()
    print(f"[✓] Saved Eastern Uganda subset: {EASTERN_OUTPUT_FILE}")
    print(f"[✓] Shape: {eastern.shape[0]} rows x {eastern.shape[1]} columns")
    print(eastern.head(5).to_string(index=False))

    eastern_overlap = eastern[
        eastern["year"].between(OVERLAP_YEARS[0], OVERLAP_YEARS[1], inclusive="both")
    ].reset_index(drop=True)
    eastern_overlap.to_csv(EASTERN_OVERLAP_OUTPUT_FILE, index=False)
    print()
    print(f"[✓] Saved Eastern overlap subset: {EASTERN_OVERLAP_OUTPUT_FILE}")
    print(f"[✓] Shape: {eastern_overlap.shape[0]} rows x {eastern_overlap.shape[1]} columns")
    print(eastern_overlap.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
