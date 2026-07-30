#!/usr/bin/env python3
"""
Fetch district-level soil features from SoilGrids and merge them with the
existing rainfall and temperature feature tables.

Outputs:
  - uganda_soil_features.csv
  - uganda_rainfall_temperature_soil_features.csv
  - uganda_full_modeling_dataset.csv (if a compatible yield file is found)
  - uganda_soil_fallback_template.csv

Notes:
  - SoilGrids is a static soil product, so soil rows are merged by district and
    repeated across years in the climate panel.
  - The SoilGrids REST API is beta and can be unavailable; this script retries
    failed requests and can fall back to a cached soil CSV if one exists.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

SOIL_PROPERTIES = ["phh2o", "soc", "clay", "sand", "silt", "bdod", "cec"]
TARGET_DEPTH_LABELS = {"0-30cm", "0-30 cm"}

SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_OUTPUT = Path("uganda_soil_features.csv")
SOIL_FALLBACK_TEMPLATE = Path("uganda_soil_fallback_template.csv")
MERGED_OUTPUT = Path("uganda_rainfall_temperature_soil_features.csv")
FULL_OUTPUT = Path("uganda_full_modeling_dataset.csv")
RAINFALL_FILE = Path("uganda_rainfall_features.csv")
TEMPERATURE_FILE = Path("uganda_temperature_features.csv")
SOIL_FALLBACK_CANDIDATES = [
    Path("uganda_soil_fallback.csv"),
    Path("uganda_soil_features_local.csv"),
    Path("uganda_soil_fallback_template.csv"),
]
YIELD_CANDIDATES = [
    Path("ubos_maize_yield_district.csv"),
    Path("ubos_district_yield_proxy.csv"),
]


def build_session():
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "uganda-pca-soil-fetcher/1.0"})
    return session


def extract_property_value(prop_payload):
    layers = prop_payload.get("layers", [])
    for layer in layers:
        depth = str(layer.get("depth", "")).strip()
        if depth in TARGET_DEPTH_LABELS:
            values = layer.get("values", {})
            for key in ("mean", "Q0.5", "median"):
                value = values.get(key)
                if value is not None:
                    return value

    for layer in layers:
        values = layer.get("values", {})
        for key in ("mean", "Q0.5", "median"):
            value = values.get(key)
            if value is not None:
                return value

    return np.nan


def fetch_soil_record(session, district, lat, lon):
    params = {
        "lon": lon,
        "lat": lat,
        "depth": "0-30cm",
        "value": "mean",
        "property": SOIL_PROPERTIES,
    }
    record = {
        "district": district,
        "lat": lat,
        "lon": lon,
        "source_status": "ok",
    }

    try:
        response = session.get(SOIL_URL, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"  [WARN] request failed for {district}: {exc}")
        record["source_status"] = "request_failed"
        for prop in SOIL_PROPERTIES:
            record[prop] = np.nan
        return record
    except ValueError as exc:
        print(f"  [WARN] invalid JSON for {district}: {exc}")
        record["source_status"] = "invalid_json"
        for prop in SOIL_PROPERTIES:
            record[prop] = np.nan
        return record

    props = payload.get("properties", {})
    if not props:
        print(f"  [WARN] no properties returned for {district}")
        record["source_status"] = "no_properties"

    for prop in SOIL_PROPERTIES:
        prop_payload = props.get(prop, {})
        record[prop] = extract_property_value(prop_payload) if prop_payload else np.nan

    if all(pd.isna(record[prop]) for prop in SOIL_PROPERTIES):
        record["source_status"] = "empty_values"

    return record


def fetch_soil_features():
    session = build_session()
    records = []

    for district, (lat, lon) in DISTRICTS.items():
        print(f"Fetching soil data for {district}...")
        records.append(fetch_soil_record(session, district, lat, lon))
        time.sleep(1.0)

    soil_df = pd.DataFrame(records)
    ordered_cols = ["district", "lat", "lon"] + SOIL_PROPERTIES + ["source_status"]
    return soil_df[ordered_cols]


def load_cached_soil_features():
    if not SOIL_OUTPUT.exists():
        return None

    cached = pd.read_csv(SOIL_OUTPUT)
    print(f"[INFO] Loaded cached soil features from {SOIL_OUTPUT}")
    return cached


def build_fallback_template():
    rows = []
    for district, (lat, lon) in DISTRICTS.items():
        row = {
            "district": district,
            "lat": lat,
            "lon": lon,
            "phh2o": np.nan,
            "soc": np.nan,
            "clay": np.nan,
            "sand": np.nan,
            "silt": np.nan,
            "bdod": np.nan,
            "cec": np.nan,
            "source_status": "local_template",
            "source_note": "Fill from a local district soil table, raster sample, or WoSIS summary.",
        }
        rows.append(row)

    template = pd.DataFrame(rows)
    template.to_csv(SOIL_FALLBACK_TEMPLATE, index=False)
    print(f"[INFO] Wrote fallback template: {SOIL_FALLBACK_TEMPLATE}")


def load_local_fallback_soil_features():
    for path in SOIL_FALLBACK_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "district" not in df.columns:
            print(f"[WARN] Skipping {path}: missing district column")
            continue

        fallback = pd.DataFrame(
            {
                "district": df["district"],
                "lat": df["lat"] if "lat" in df.columns else np.nan,
                "lon": df["lon"] if "lon" in df.columns else np.nan,
            }
        )

        for prop in SOIL_PROPERTIES:
            fallback[prop] = df[prop] if prop in df.columns else np.nan

        if fallback[SOIL_PROPERTIES].isna().all().all():
            print(f"[WARN] Skipping {path}: no populated soil property values")
            continue

        if "source_status" in df.columns:
            fallback["source_status"] = df["source_status"]
        else:
            fallback["source_status"] = "local_fallback"

        missing_lat = fallback["lat"].isna()
        missing_lon = fallback["lon"].isna()
        for district, (lat, lon) in DISTRICTS.items():
            mask = fallback["district"].astype(str).str.strip() == district
            fallback.loc[mask & missing_lat, "lat"] = lat
            fallback.loc[mask & missing_lon, "lon"] = lon

        print(f"[INFO] Loaded local soil fallback: {path}")
        ordered_cols = ["district", "lat", "lon"] + SOIL_PROPERTIES + ["source_status"]
        return fallback[ordered_cols]

    return None


def merge_feature_tables(soil_df):
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Missing rainfall features: {RAINFALL_FILE}")
    if not TEMPERATURE_FILE.exists():
        raise FileNotFoundError(f"Missing temperature features: {TEMPERATURE_FILE}")

    rainfall = pd.read_csv(RAINFALL_FILE)
    temperature = pd.read_csv(TEMPERATURE_FILE)

    climate = rainfall.merge(
        temperature,
        on=["district", "year"],
        how="inner",
        validate="one_to_one",
    )
    merged = climate.merge(
        soil_df.drop(columns=["lat", "lon"], errors="ignore"),
        on="district",
        how="left",
        validate="many_to_one",
    )

    merged = merged.sort_values(["district", "year"]).reset_index(drop=True)
    return merged


def load_yield_table():
    for path in YIELD_CANDIDATES:
        if not path.exists():
            continue

        yield_df = pd.read_csv(path)
        required = {"district", "year"}
        if not required.issubset(yield_df.columns):
            print(f"[WARN] Skipping {path}: missing district/year columns")
            continue

        if "yield_tons_ha" not in yield_df.columns:
            if "yield_weighted" in yield_df.columns:
                yield_df = yield_df.rename(columns={"yield_weighted": "yield_tons_ha"})
            else:
                print(f"[WARN] Skipping {path}: missing yield_tons_ha or yield_weighted")
                continue

        keep_cols = ["district", "year", "yield_tons_ha"]
        extras = [col for col in ("n_households", "total_area_ha", "total_production_kg") if col in yield_df.columns]
        keep_cols.extend(extras)

        print(f"[INFO] Using yield file: {path}")
        return yield_df[keep_cols]

    print("[WARN] No compatible yield file found. Skipping final yield merge.")
    return None


def merge_with_yield(features_df):
    yield_df = load_yield_table()
    if yield_df is None:
        return None

    merged = features_df.merge(
        yield_df,
        on=["district", "year"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.sort_values(["district", "year"]).reset_index(drop=True)
    return merged


def main():
    print("=" * 70)
    print("  SOILGRIDS FETCH + RAINFALL/TEMPERATURE/SOIL MERGE")
    print("=" * 70)

    soil_df = fetch_soil_features()

    success_count = (soil_df["source_status"] == "ok").sum()
    if success_count == 0:
        print("[WARN] No live SoilGrids responses succeeded.")
        local_fallback = load_local_fallback_soil_features()
        if local_fallback is not None:
            soil_df = local_fallback
        else:
            cached = load_cached_soil_features()
            if cached is not None and not cached[SOIL_PROPERTIES].isna().all().all():
                soil_df = cached
                print("[INFO] Using cached soil features with non-empty values.")
            else:
                print("[WARN] Proceeding with all-NaN soil columns.")
                build_fallback_template()

    soil_df.to_csv(SOIL_OUTPUT, index=False)
    print(f"[✓] Saved soil features: {SOIL_OUTPUT}")
    print(soil_df.to_string(index=False))

    merged = merge_feature_tables(soil_df)
    merged.to_csv(MERGED_OUTPUT, index=False)

    print()
    print(f"[✓] Saved merged dataset: {MERGED_OUTPUT}")
    print(f"[✓] Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(merged.head(5).to_string(index=False))

    full_dataset = merge_with_yield(merged)
    if full_dataset is not None:
        full_dataset.to_csv(FULL_OUTPUT, index=False)
        print()
        print(f"[✓] Saved full modeling dataset: {FULL_OUTPUT}")
        print(f"[✓] Shape: {full_dataset.shape[0]} rows × {full_dataset.shape[1]} columns")
        print(full_dataset.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
