#!/usr/bin/env python3
"""
================================================================================
CHIRTS-ERA5 TEMPERATURE DOWNLOADER + GDD COMPUTATION
For: Bugema University PCA Agriculture Project
================================================================================

CHIRTS-ERA5 provides daily Tmax and Tmin at 0.05° resolution (same as CHIRPS)
from the Climate Hazards Center (CHC) at UC Santa Barbara.

Source: https://data.chc.ucsb.edu/products/CHIRTS-ERA5/
Resolution: 0.05° (same as CHIRPS)
Period: 1980–present
Variables: Tmax (°C), Tmin (°C)

REQUIREMENTS:
    pip install requests tqdm xarray netCDF4 pandas numpy rasterio

WHAT YOU GET:
    • Daily Tmax/Tmin for 15 Ugandan districts (2015-2023)
    • Growing Degree Days (GDD) for maize: GDD = Σ(max(0, (Tmax+Tmin)/2 - 10))
    • Seasonal temperature features (MAM, SON, DJF, JJA)
    • Heat stress days (Tmax > 35°C)
    • File: uganda_temperature_features.csv (ready to merge with rainfall + yield)
"""

import os
import urllib.request
import requests
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRTS-ERA5/v1.0/daily_tifs/"
LOCAL_DIR = "chirts_era5_daily"

# Uganda bounds (same as CHIRPS)
UGANDA_BOUNDS = {
    'lat_min': -1.5, 'lat_max': 4.2,
    'lon_min': 29.5, 'lon_max': 35.0
}

# District centroids
DISTRICTS = {
    'Mbale':      (1.075, 34.175),
    'Kapchorwa':  (1.400, 34.450),
    'Iganga':     (0.617, 33.483),
    'Jinja':      (0.425, 33.204),
    'Tororo':     (0.693, 34.181),
    'Soroti':     (1.715, 33.611),
    'Lira':       (2.250, 32.917),
    'Gulu':       (2.774, 32.299),
    'Mbarara':    (-0.607, 30.658),
    'Arua':       (3.020, 30.911),
    'Masaka':     (-0.333, 31.733),
    'Fort_Portal':(0.671, 30.275),
    'Hoima':      (1.433, 31.350),
    'Kabale':     (-1.250, 29.983),
    'Kasese':     (0.183, 30.083)
}

YEARS = list(range(2015, 2024))
T_BASE = 10.0  # Base temperature for maize GDD (°C)
T_MAX_OPT = 35.0  # Upper threshold for maize (heat stress)

# =============================================================================
# STEP 1: DOWNLOAD CHIRTS-ERA5 DAILY TIFS
# =============================================================================
def download_chirts_daily():
    """
    Download CHIRTS-ERA5 daily GeoTIFFs for Tmax and Tmin.
    Files are organized as: BASE_URL/{year}/Tmax.{year}.{doy}.tif
    where doy = day of year (001-365/366)
    """
    os.makedirs(LOCAL_DIR, exist_ok=True)

    for year in YEARS:
        year_dir = os.path.join(LOCAL_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        # Check if year is leap year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap else 365

        for doy in range(1, n_days + 1):
            doy_str = f"{doy:03d}"

            for var in ['Tmax', 'Tmin']:
                filename = f"{var}.{year}.{doy_str}.tif"
                filepath = os.path.join(year_dir, filename)

                if os.path.exists(filepath):
                    continue

                url = f"{BASE_URL}{year}/{filename}"

                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        urllib.request.urlretrieve(url, filepath)
                except Exception as e:
                    print(f"[WARN] Could not download {filename}: {e}")
                    continue

        print(f"[✓] Year {year} downloaded")

# =============================================================================
# STEP 2: EXTRACT TEMPERATURE FOR DISTRICTS
# =============================================================================
def extract_temperature_features():
    """
    Extract daily Tmax/Tmin for each district centroid,
    then compute seasonal aggregates and GDD.
    """
    import rasterio
    from rasterio.sample import sample_gen

    records = []

    for year in YEARS:
        year_dir = os.path.join(LOCAL_DIR, str(year))
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap else 365

        for doy in range(1, n_days + 1):
            doy_str = f"{doy:03d}"
            date = datetime.strptime(f"{year}-{doy}", "%Y-%j")

            for var in ['Tmax', 'Tmin']:
                filename = f"{var}.{year}.{doy_str}.tif"
                filepath = os.path.join(year_dir, filename)

                if not os.path.exists(filepath):
                    continue

                try:
                    with rasterio.open(filepath) as src:
                        for district, (lat, lon) in DISTRICTS.items():
                            val = list(src.sample([(lon, lat)]))[0][0]
                            records.append({
                                'district': district,
                                'date': date,
                                'year': year,
                                'month': date.month,
                                'day_of_year': doy,
                                'variable': var,
                                'temperature_c': float(val)
                            })
                except Exception as e:
                    print(f"[WARN] Error reading {filename}: {e}")

    df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(df)} daily temperature records")
    return df

# =============================================================================
# STEP 3: COMPUTE GDD AND SEASONAL FEATURES
# =============================================================================
def compute_gdd_and_features(temp_df):
    """
    Compute Growing Degree Days and seasonal temperature features.
    """
    # Pivot to wide format
    temp_wide = temp_df.pivot_table(
        index=['district', 'date', 'year', 'month', 'day_of_year'],
        columns='variable',
        values='temperature_c'
    ).reset_index()

    # Compute daily GDD for maize
    # GDD = max(0, (Tmax + Tmin)/2 - Tbase)
    temp_wide['t_mean'] = (temp_wide['Tmax'] + temp_wide['Tmin']) / 2
    temp_wide['gdd_daily'] = np.maximum(0, temp_wide['t_mean'] - T_BASE)

    # Heat stress days (Tmax > 35°C)
    temp_wide['heat_stress'] = (temp_wide['Tmax'] > T_MAX_OPT).astype(int)

    # Cold stress days (Tmin < 5°C)
    temp_wide['cold_stress'] = (temp_wide['Tmin'] < 5).astype(int)

    # Define seasons
    def get_season(month):
        if month in [3, 4, 5]:     return 'MAM'
        elif month in [9, 10, 11]: return 'SON'
        elif month in [12, 1, 2]:  return 'DJF'
        else:                       return 'JJA'

    temp_wide['season'] = temp_wide['month'].apply(get_season)

    # Seasonal aggregates
    seasonal = temp_wide.groupby(['district', 'year', 'season']).agg(
        tmax_mean=('Tmax', 'mean'),
        tmin_mean=('Tmin', 'mean'),
        tmean_mean=('t_mean', 'mean'),
        gdd_total=('gdd_daily', 'sum'),
        heat_stress_days=('heat_stress', 'sum'),
        cold_stress_days=('cold_stress', 'sum')
    ).reset_index()

    seasonal_pivot = seasonal.pivot_table(
        index=['district', 'year'],
        columns='season',
        values=['tmax_mean', 'tmin_mean', 'tmean_mean', 'gdd_total', 'heat_stress_days', 'cold_stress_days']
    ).reset_index()

    # Flatten column names
    seasonal_pivot.columns = [
        f"{col[1]}_{col[0]}" if col[1] != '' else col[0]
        for col in seasonal_pivot.columns.values
    ]

    # Annual aggregates
    annual = temp_wide.groupby(['district', 'year']).agg(
        annual_tmax_mean=('Tmax', 'mean'),
        annual_tmin_mean=('Tmin', 'mean'),
        annual_tmean_mean=('t_mean', 'mean'),
        annual_gdd=('gdd_daily', 'sum'),
        annual_heat_stress=('heat_stress', 'sum'),
        annual_cold_stress=('cold_stress', 'sum'),
        growing_season_length=('gdd_daily', lambda x: (x > 0).sum())
    ).reset_index()

    # Merge
    features = seasonal_pivot.merge(annual, on=['district', 'year'])

    print(f"[✓] Temperature feature matrix shape: {features.shape}")
    print("\nPreview:")
    print(features.head())

    features.to_csv('uganda_temperature_features.csv', index=False)
    print("\n[✓] Saved: uganda_temperature_features.csv")

    return features

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CHIRTS-ERA5 TEMPERATURE + GDD COMPUTATION")
    print("=" * 70)

    # Step 1: Download (comment out if already downloaded)
    # download_chirts_daily()

    # Step 2: Extract
    print("\n[INFO] Extracting temperature data...")
    temp_df = extract_temperature_features()

    # Step 3: Compute GDD and features
    print("\n[INFO] Computing GDD and seasonal features...")
    features = compute_gdd_and_features(temp_df)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print("""
Next steps:
  1. Merge with rainfall data:
     rainfall = pd.read_csv('uganda_rainfall_features.csv')
     combined = rainfall.merge(features, on=['district', 'year'])
     combined.to_csv('uganda_climate_features.csv', index=False)

  2. Run PCA on combined features (rainfall + temperature)

  3. Predict yield with thermal time included
    """)

if __name__ == "__main__":
    main()
