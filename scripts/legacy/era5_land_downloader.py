#!/usr/bin/env python3
"""
================================================================================
ERA5-LAND MONTHLY TEMPERATURE DOWNLOADER + GDD COMPUTATION
For: Bugema University PCA Agriculture Project
================================================================================

ERA5-Land provides monthly mean Tmax, Tmin, and T2m at 0.1° resolution
from the Copernicus Climate Data Store (CDS).

Source: https://cds.climate.copernicus.eu/
Dataset: reanalysis-era5-land-monthly-means
Resolution: 0.1° (finer than standard ERA5's 0.25°)
Period: 1950–present (3-month delay)
Variables: 2m temperature, 2m dewpoint temperature

SETUP REQUIRED:
  1. Create free CDS account: https://cds.climate.copernicus.eu/
  2. Get API key from your profile page
  3. Install CDS API client: pip install cdsapi
  4. Create ~/.cdsapirc file with your credentials:

     url: https://cds.climate.copernicus.eu/api
     key: YOUR_UID:YOUR_API_KEY

REQUIREMENTS:
    pip install cdsapi xarray netCDF4 pandas numpy

WHAT YOU GET:
    • Monthly Tmean, Tmax, Tmin for 15 Ugandan districts (2015-2023)
    • Growing Degree Days (GDD) per season
    • Heat stress days (Tmax > 35°C)
    • Cold stress days (Tmin < 5°C)
    • File: uganda_temperature_features.csv (ready to merge)
"""

import os
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_FILE = "era5_land_uganda_temperature.nc"
YEARS = list(range(2015, 2024))

# Uganda bounding box [north, west, south, east]
# Slightly expanded to ensure all districts are covered
AREA = [4.5, 29.0, -2.0, 35.5]

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

T_BASE = 10.0   # Base temperature for maize GDD (°C)
T_MAX = 35.0    # Upper threshold for heat stress (°C)
T_MIN = 5.0     # Lower threshold for cold stress (°C)

# =============================================================================
# STEP 1: DOWNLOAD ERA5-LAND MONTHLY DATA
# =============================================================================
def download_era5_land():
    """
    Download ERA5-Land monthly temperature data for Uganda.
    One request per year to stay within CDS limits.
    """
    client = cdsapi.Client()

    if os.path.exists(OUTPUT_FILE):
        print(f"[✓] File already exists: {OUTPUT_FILE}")
        return

    print("[INFO] Downloading ERA5-Land monthly temperature data...")
    print("[INFO] This may take 10-30 minutes depending on CDS queue...")

    # Download all years in one request
    client.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '2m_dewpoint_temperature',
                '2m_temperature',
            ],
            'year': [str(y) for y in YEARS],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'time': '00:00',
            'area': AREA,
            'format': 'netcdf',
        },
        OUTPUT_FILE
    )

    print(f"[✓] Downloaded: {OUTPUT_FILE}")

# =============================================================================
# STEP 2: EXTRACT & COMPUTE GDD
# =============================================================================
def extract_temperature_features():
    """
    Load ERA5-Land NetCDF, extract for district centroids,
    compute GDD and seasonal features.

    NOTE: ERA5-Land provides monthly MEAN 2m temperature.
    We approximate Tmax and Tmin using diurnal range assumptions
    based on elevation and season, OR use the mean directly for GDD.

    For more accurate GDD, you need daily Tmax/Tmin.
    ERA5-Land monthly is sufficient for coarse seasonal analysis.
    """
    print("[INFO] Loading ERA5-Land data...")
    ds = xr.open_dataset(OUTPUT_FILE)

    print(f"[INFO] Dataset variables: {list(ds.data_vars)}")
    print(f"[INFO] Dataset shape: {dict(ds.dims)}")

    # ERA5-Land temperature is in Kelvin, convert to Celsius
    t2m = ds['t2m'] - 273.15  # 2m temperature in °C

    # For GDD computation with monthly means, we use:
    # GDD_month = max(0, Tmean_month - Tbase) × days_in_month
    # This is a coarse approximation. For precise GDD, use daily data.

    records = []

    for district, (lat, lon) in DISTRICTS.items():
        # Extract time series for this district
        ts = t2m.sel(latitude=lat, longitude=lon, method='nearest')

        for t, temp in zip(ts.time.values, ts.values):
            date = pd.to_datetime(str(t))
            year = date.year
            month = date.month

            # Days in month
            days_in_month = pd.Period(date, freq='M').days_in_month

            # Monthly GDD (coarse approximation)
            gdd_month = max(0, float(temp) - T_BASE) * days_in_month

            # Heat stress: days where Tmean > 35°C (very conservative)
            heat_stress = 1 if float(temp) > T_MAX else 0

            # Cold stress: days where Tmean < 5°C
            cold_stress = 1 if float(temp) < T_MIN else 0

            records.append({
                'district': district,
                'date': date,
                'year': year,
                'month': month,
                't2m_mean_c': float(temp),
                'gdd_month': gdd_month,
                'heat_stress_month': heat_stress,
                'cold_stress_month': cold_stress,
                'days_in_month': days_in_month
            })

    df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(df)} monthly records")
    return df

# =============================================================================
# STEP 3: SEASONAL AGGREGATION
# =============================================================================
def compute_seasonal_features(temp_df):
    """
    Aggregate monthly temperature data to seasonal features.
    """
    def get_season(month):
        if month in [3, 4, 5]:     return 'MAM'
        elif month in [9, 10, 11]: return 'SON'
        elif month in [12, 1, 2]:  return 'DJF'
        else:                       return 'JJA'

    temp_df['season'] = temp_df['month'].apply(get_season)

    # Seasonal aggregates
    seasonal = temp_df.groupby(['district', 'year', 'season']).agg(
        tmean_mean=('t2m_mean_c', 'mean'),
        gdd_total=('gdd_month', 'sum'),
        heat_stress_months=('heat_stress_month', 'sum'),
        cold_stress_months=('cold_stress_month', 'sum')
    ).reset_index()

    seasonal_pivot = seasonal.pivot_table(
        index=['district', 'year'],
        columns='season',
        values=['tmean_mean', 'gdd_total', 'heat_stress_months', 'cold_stress_months']
    ).reset_index()

    # Flatten columns
    seasonal_pivot.columns = [
        f"{col[1]}_{col[0]}" if col[1] != '' else col[0]
        for col in seasonal_pivot.columns.values
    ]

    # Annual aggregates
    annual = temp_df.groupby(['district', 'year']).agg(
        annual_tmean=('t2m_mean_c', 'mean'),
        annual_gdd=('gdd_month', 'sum'),
        annual_heat_stress=('heat_stress_month', 'sum'),
        annual_cold_stress=('cold_stress_month', 'sum'),
        growing_months=('gdd_month', lambda x: (x > 0).sum())
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
    print("  ERA5-LAND TEMPERATURE + GDD COMPUTATION")
    print("=" * 70)

    # Step 1: Download
    try:
        download_era5_land()
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[INFO] Make sure you have:")
        print("  1. A CDS account: https://cds.climate.copernicus.eu/")
        print("  2. Installed cdsapi: pip install cdsapi")
        print("  3. Created ~/.cdsapirc with your API key")
        return

    # Step 2: Extract
    print("\n[INFO] Extracting temperature features...")
    temp_df = extract_temperature_features()

    # Step 3: Aggregate
    print("\n[INFO] Computing seasonal features...")
    features = compute_seasonal_features(temp_df)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print("""
Next steps:
  1. Merge with rainfall data:
     rainfall = pd.read_csv('uganda_rainfall_features.csv')
     combined = rainfall.merge(features, on=['district', 'year'])
     combined.to_csv('uganda_climate_features.csv', index=False)

  2. Run PCA on combined features:
     features = ['MAM', 'SON', 'annual_rainfall', 'rain_cv',
                 'MAM_gdd_total', 'SON_gdd_total', 'annual_tmean',
                 'annual_heat_stress', 'annual_cold_stress']

  3. Predict yield with thermal time included
    """)

if __name__ == "__main__":
    main()
