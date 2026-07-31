#!/usr/bin/env python3
"""
================================================================================
CHIRPS FIXED DOWNLOADER — Uganda PCA Project
================================================================================
Fixes the latitude clipping issue by auto-detecting coordinate names and order.
"""

import os
import time
import urllib.request
from urllib.error import HTTPError, URLError

import requests
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
LOCAL_FILE = "chirps-v2.0.monthly.nc"
CHUNK_SIZE = 8192 * 16
MAX_RETRIES = 5
RETRY_DELAY = 10

# Uganda bounds (with small buffer for edge cases)
UGANDA_BOUNDS = {
    'lat_min': -1.5, 'lat_max': 4.2,
    'lon_min': 29.5, 'lon_max': 35.0
}

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

# =============================================================================
# DOWNLOAD (same as before)
# =============================================================================
def download_with_resume(url, filepath, chunk_size=CHUNK_SIZE, max_retries=MAX_RETRIES):
    existing_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    for attempt in range(max_retries):
        try:
            response = requests.head(url, timeout=30, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            break
        except Exception as e:
            print(f"[WARN] Head request failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(RETRY_DELAY)
    else:
        total_size = 0
        existing_size = 0

    if total_size > 0 and existing_size >= total_size:
        print(f"[✓] File already complete: {filepath} ({existing_size / 1e9:.2f} GB)")
        return True

    if existing_size > 0 and total_size > 0:
        print(f"[↻] Resuming download from {existing_size / 1e9:.2f} GB / {total_size / 1e9:.2f} GB")
    else:
        print(f"[↓] Starting fresh download...")

    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    for attempt in range(max_retries):
        try:
            print(f"[↓] Download attempt {attempt + 1}/{max_retries}...")

            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()
                mode = 'ab' if existing_size > 0 else 'wb'
                downloaded = existing_size

                pbar = tqdm(
                    total=total_size if total_size > 0 else None,
                    initial=existing_size,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc="Downloading CHIRPS", ncols=80
                )

                with open(filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

                pbar.close()

                if total_size > 0 and downloaded < total_size:
                    print(f"[WARN] Download incomplete: {downloaded}/{total_size}")
                    time.sleep(RETRY_DELAY)
                    existing_size = downloaded
                    headers['Range'] = f'bytes={existing_size}-'
                    continue

                print(f"[✓] Download complete: {filepath} ({downloaded / 1e9:.2f} GB)")
                return True

        except (HTTPError, URLError, requests.exceptions.RequestException) as e:
            print(f"[ERROR] Download failed: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                existing_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                headers['Range'] = f'bytes={existing_size}-'
            else:
                print("[FATAL] Max retries exceeded.")
                return False

    return False


# =============================================================================
# FIXED EXTRACTION — Auto-detects coordinate names and ordering
# =============================================================================
def extract_rainfall_features(filepath):
    print("\n[INFO] Loading CHIRPS data...")
    ds = xr.open_dataset(filepath)
    pr = ds['precip']

    print(f"[INFO] Global dataset shape: {pr.shape}")
    print(f"[INFO] Time range: {str(pr.time.min().values)[:10]} to {str(pr.time.max().values)[:10]}")

    # Auto-detect coordinate names
    coord_map = {}
    for coord in ds.coords:
        name = coord.lower()
        if 'lat' in name:
            coord_map['lat'] = coord
        elif 'lon' in name:
            coord_map['lon'] = coord

    lat_coord = coord_map.get('lat', 'latitude')
    lon_coord = coord_map.get('lon', 'longitude')
    print(f"[INFO] Detected coordinates: lat='{lat_coord}', lon='{lon_coord}'")

    # Check ordering
    lat_vals = ds.coords[lat_coord].values
    lat_ascending = lat_vals[1] > lat_vals[0] if len(lat_vals) > 1 else True
    lon_vals = ds.coords[lon_coord].values
    lon_ascending = lon_vals[1] > lon_vals[0] if len(lon_vals) > 1 else True

    print(f"[INFO] Latitude order: {'ascending' if lat_ascending else 'descending'}")
    print(f"[INFO] Longitude order: {'ascending' if lon_ascending else 'descending'}")

    # Build slices correctly based on ordering
    lat_min, lat_max = UGANDA_BOUNDS['lat_min'], UGANDA_BOUNDS['lat_max']
    lon_min, lon_max = UGANDA_BOUNDS['lon_min'], UGANDA_BOUNDS['lon_max']

    if lat_ascending:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    if lon_ascending:
        lon_slice = slice(lon_min, lon_max)
    else:
        lon_slice = slice(lon_max, lon_min)

    print(f"[INFO] Latitude slice: {lat_slice}")
    print(f"[INFO] Longitude slice: {lon_slice}")

    # Extract with time filter
    try:
        uganda_rain = pr.sel(**{
            lat_coord: lat_slice,
            lon_coord: lon_slice,
            'time': slice(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
        })
        print(f"[✓] Clipped shape: {uganda_rain.shape}")
    except Exception as e:
        print(f"[✗] Clipping failed: {e}")
        print("[INFO] Trying with expanded bounds (+/- 0.5°)...")

        if lat_ascending:
            lat_slice = slice(lat_min - 0.5, lat_max + 0.5)
        else:
            lat_slice = slice(lat_max + 0.5, lat_min - 0.5)

        if lon_ascending:
            lon_slice = slice(lon_min - 0.5, lon_max + 0.5)
        else:
            lon_slice = slice(lon_max + 0.5, lon_min - 0.5)

        uganda_rain = pr.sel(**{
            lat_coord: lat_slice,
            lon_coord: lon_slice,
            'time': slice(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
        })
        print(f"[✓] Expanded clipping successful. Shape: {uganda_rain.shape}")

    # Verify we got data
    if uganda_rain.shape[1] == 0 or uganda_rain.shape[2] == 0:
        print("[FATAL] Clipped dataset has zero spatial extent. Check coordinate bounds.")
        return None

    # Save clipped version
    clipped_file = "uganda_chirps_clipped.nc"
    try:
        uganda_rain.to_netcdf(clipped_file)
        print(f"[✓] Saved clipped dataset: {clipped_file}")
    except Exception as e:
        print(f"[WARN] Could not save NetCDF: {e}")
        print("[INFO] Continuing with in-memory processing...")

    # Extract point time series for districts
    print("\n[INFO] Extracting rainfall time series for districts...")
    records = []

    for district, (lat, lon) in DISTRICTS.items():
        try:
            ts = uganda_rain.sel(**{lat_coord: lat, lon_coord: lon}, method='nearest')
            for t, val in zip(ts.time.values, ts.values):
                dt = pd.to_datetime(str(t))
                records.append({
                    'district': district,
                    'date': dt,
                    'year': dt.year,
                    'month': dt.month,
                    'rainfall_mm': float(val)
                })
        except Exception as e:
            print(f"[WARN] Could not extract {district}: {e}")

    df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(df)} monthly records across {len(DISTRICTS)} districts")

    # Seasonal aggregation
    print("[INFO] Computing seasonal rainfall features...")

    def get_season(month):
        if month in [3, 4, 5]:     return 'MAM'
        elif month in [9, 10, 11]: return 'SON'
        elif month in [12, 1, 2]:  return 'DJF'
        else:                       return 'JJA'

    df['season'] = df['month'].apply(get_season)

    seasonal = df.groupby(['district', 'year', 'season'])['rainfall_mm'].sum().reset_index()
    seasonal_pivot = seasonal.pivot_table(
        index=['district', 'year'],
        columns='season',
        values='rainfall_mm'
    ).reset_index()

    annual = df.groupby(['district', 'year']).agg(
        annual_rainfall=('rainfall_mm', 'sum'),
        rain_cv=('rainfall_mm', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        max_monthly=('rainfall_mm', 'max'),
        min_monthly=('rainfall_mm', 'min'),
        rainy_months=('rainfall_mm', lambda x: (x > 50).sum())
    ).reset_index()

    features = seasonal_pivot.merge(annual, on=['district', 'year'])

    for season in ['MAM', 'SON', 'DJF', 'JJA']:
        if season not in features.columns:
            features[season] = 0

    col_order = ['district', 'year', 'MAM', 'SON', 'DJF', 'JJA',
                 'annual_rainfall', 'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']
    features = features[[c for c in col_order if c in features.columns]]

    print(f"[✓] Feature matrix shape: {features.shape}")
    print("\nPreview:")
    print(features.head(10).to_string())

    features.to_csv('uganda_rainfall_features.csv', index=False)
    print("\n[✓] Saved: uganda_rainfall_features.csv")

    return features


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CHIRPS FIXED DOWNLOADER — Uganda PCA Project")
    print("=" * 70)

    if not os.path.exists(LOCAL_FILE):
        print(f"\n[✗] File not found: {LOCAL_FILE}")
        print("[INFO] Run download first, or place the file in this directory.")
        return

    print(f"\n[INFO] Using existing file: {LOCAL_FILE} ({os.path.getsize(LOCAL_FILE)/1e9:.2f} GB)")

    features = extract_rainfall_features(LOCAL_FILE)

    if features is not None:
        print("\n" + "=" * 70)
        print("  DONE — Next Steps")
        print("=" * 70)
        print("""
1. VERIFY OUTPUT:
   python -c "import pandas as pd; df = pd.read_csv('uganda_rainfall_features.csv'); print(df.head()); print(f'Shape: {df.shape}')"

2. MERGE WITH YIELD DATA:
   ubos = pd.read_csv('ubos_maize_yield.csv')  # From ubos.org
   merged = features.merge(ubos, on=['district', 'year'])
   merged.to_csv('uganda_maize_pca_ready.csv', index=False)

3. RUN PCA:
   Use the pca_from_scratch.py script we built earlier.
        """)

if __name__ == "__main__":
    main()
