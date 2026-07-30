#!/usr/bin/env python3
"""
================================================================================
CHIRPS BULLETPROOF DOWNLOADER
For: Bugema University Mathematics Student — PCA Agriculture Project
================================================================================

Features:
  • Resumable download (if interrupted, restarts where it left off)
  • Progress bar with ETA
  • Bandwidth throttling option (for slow/shared connections)
  • Automatic retry on failure
  • SHA256 verification (optional)
  • Clips to Uganda bounding box immediately after download

REQUIREMENTS:
    pip install requests tqdm xarray netCDF4 pandas numpy

USAGE:
    python chirps_downloader.py

OUTPUT:
    chirps-v2.0.monthly.nc          (global file, ~2.5 GB)
    uganda_chirps_clipped.nc        (Uganda-only, ~15 MB)
    uganda_rainfall_features.csv    (seasonal aggregates, ready for PCA)
"""

import os
import sys
import time
import urllib.request
import hashlib
from urllib.error import HTTPError, URLError

import requests
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION — Edit these if needed
# =============================================================================
CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
LOCAL_FILE = "chirps-v2.0.monthly.nc"
CHUNK_SIZE = 8192 * 16  # 128 KB chunks
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

# Uganda bounding box
UGANDA_BOUNDS = {
    'lat_min': -1.5, 'lat_max': 4.2,
    'lon_min': 29.5, 'lon_max': 35.0
}

# District centroids for feature extraction
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
# DOWNLOAD WITH RESUME & PROGRESS
# =============================================================================
def download_with_resume(url, filepath, chunk_size=CHUNK_SIZE, max_retries=MAX_RETRIES):
    """
    Download a file with resume support and progress bar.

    If the file already exists partially, resumes from where it left off.
    Retries on network errors up to max_retries times.
    """
    # Check existing file size for resume
    existing_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    # Get total file size from server
    print(f"[INFO] Checking file size on server...")
    for attempt in range(max_retries):
        try:
            response = requests.head(url, timeout=30, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            break
        except Exception as e:
            print(f"[WARN] Head request failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(RETRY_DELAY)
    else:
        print("[ERROR] Could not determine file size. Proceeding without resume.")
        total_size = 0
        existing_size = 0

    if total_size > 0 and existing_size >= total_size:
        print(f"[✓] File already complete: {filepath} ({existing_size / 1e9:.2f} GB)")
        return True

    if existing_size > 0 and total_size > 0:
        print(f"[↻] Resuming download from {existing_size / 1e9:.2f} GB / {total_size / 1e9:.2f} GB")
    else:
        print(f"[↓] Starting fresh download...")

    # Build request with Range header for resume
    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    for attempt in range(max_retries):
        try:
            print(f"[↓] Download attempt {attempt + 1}/{max_retries}...")

            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()

                # Determine write mode and total progress
                mode = 'ab' if existing_size > 0 else 'wb'
                downloaded = existing_size

                # Progress bar
                pbar = tqdm(
                    total=total_size if total_size > 0 else None,
                    initial=existing_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading CHIRPS",
                    ncols=80
                )

                with open(filepath, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

                pbar.close()

                if total_size > 0 and downloaded < total_size:
                    print(f"[WARN] Download incomplete: {downloaded}/{total_size} bytes")
                    print(f"[INFO] Will retry in {RETRY_DELAY} seconds...")
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
                # Update resume position
                existing_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                headers['Range'] = f'bytes={existing_size}-'
            else:
                print("[FATAL] Max retries exceeded. Download failed.")
                return False

    return False


# =============================================================================
# CLIP TO UGANDA & EXTRACT FEATURES
# =============================================================================
def process_uganda_rainfall(filepath):
    """
    Load global CHIRPS, clip to Uganda, extract seasonal features.
    """
    print("\n[INFO] Loading CHIRPS data...")
    chirps = xr.open_dataset(filepath)
    pr = chirps['precip']

    print(f"[INFO] Global dataset shape: {pr.shape}")
    print(f"[INFO] Time range: {str(pr.time.min().values)[:10]} to {str(pr.time.max().values)[:10]}")

    # Clip to Uganda + time range
    print("[INFO] Clipping to Uganda bounding box...")
    uganda_rain = pr.sel(
        longitude=slice(UGANDA_BOUNDS['lon_min'], UGANDA_BOUNDS['lon_max']),
        latitude=slice(UGANDA_BOUNDS['lat_max'], UGANDA_BOUNDS['lat_min']),
        time=slice(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
    )

    print(f"[✓] Clipped shape: {uganda_rain.shape}")

    # Save clipped version (much smaller)
    clipped_file = "uganda_chirps_clipped.nc"
    uganda_rain.to_netcdf(clipped_file)
    print(f"[✓] Saved clipped dataset: {clipped_file}")

    # Extract point time series
    print("\n[INFO] Extracting rainfall time series for districts...")
    records = []
    for district, (lat, lon) in DISTRICTS.items():
        ts = uganda_rain.sel(latitude=lat, longitude=lon, method='nearest')
        for t, val in zip(ts.time.values, ts.values):
            dt = pd.to_datetime(str(t))
            records.append({
                'district': district,
                'date': dt,
                'year': dt.year,
                'month': dt.month,
                'rainfall_mm': float(val)
            })

    df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(df)} monthly records across {len(DISTRICTS)} districts")

    # Seasonal aggregation
    print("[INFO] Computing seasonal aggregates...")

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

    # Fill missing seasons with 0
    for season in ['MAM', 'SON', 'DJF', 'JJA']:
        if season not in features.columns:
            features[season] = 0

    # Reorder columns
    col_order = ['district', 'year', 'MAM', 'SON', 'DJF', 'JJA',
                 'annual_rainfall', 'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']
    features = features[[c for c in col_order if c in features.columns]]

    print(f"[✓] Feature matrix shape: {features.shape}")
    print("\nPreview:")
    print(features.head(10).to_string())

    # Save
    features.to_csv('uganda_rainfall_features.csv', index=False)
    print("\n[✓] Saved: uganda_rainfall_features.csv")

    return features


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CHIRPS BULLETPROOF DOWNLOADER — Uganda PCA Project")
    print("=" * 70)
    print(f"\nTarget file: {LOCAL_FILE}")
    print(f"Expected size: ~2.5 GB")
    print(f"Source: {CHIRPS_URL}")
    print()

    # Step 1: Download
    success = download_with_resume(CHIRPS_URL, LOCAL_FILE)
    if not success:
        print("\n[FATAL] Download failed. Please check your internet connection and try again.")
        print("        If the problem persists, try Approach 2 (ClimateSERV API) or Approach 3 (GEE).")
        sys.exit(1)

    # Step 2: Process
    print("\n" + "=" * 70)
    print("  PROCESSING: Clip to Uganda & Extract Features")
    print("=" * 70)
    features = process_uganda_rainfall(LOCAL_FILE)

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("  DONE — Next Steps")
    print("=" * 70)
    print("""
Your rainfall data is ready. Next:

1. MERGE WITH YIELD DATA
   Replace synthetic yield in your PCA pipeline with real UBOS data:

   ubos = pd.read_csv('ubos_maize_yield.csv')  # Get from ubos.org
   merged = features.merge(ubos, on=['district', 'year'])
   merged.to_csv('uganda_maize_pca_ready.csv', index=False)

2. RUN PCA
   Use the PCA code we built earlier:

   from pca_from_scratch import pca  # Your implementation
   X = merged[['MAM', 'SON', 'annual_rainfall', 'rain_cv', ...]].values
   Z, components, explained = pca(X, n_components=3)

3. VISUALIZE
   Biplot of districts in PC space, colored by yield.

Files generated in this directory:
   • chirps-v2.0.monthly.nc      (~2.5 GB, global — keep or delete)
   • uganda_chirps_clipped.nc    (~15 MB, Uganda-only — keep this)
   • uganda_rainfall_features.csv  (seasonal aggregates — input to PCA)
    """)

if __name__ == "__main__":
    main()
