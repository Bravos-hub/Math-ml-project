#!/usr/bin/env python3
"""
CHIRPS Direct Extraction for Uganda
Based on diagnostic: latitude ASCENDING, longitude ASCENDING
Run this after you have chirps-v2.0.monthly.nc downloaded.
"""

import xarray as xr
import pandas as pd
import numpy as np

FILE = "chirps-v2.0.monthly.nc"

# ============================================================
# STEP 1: LOAD & CLIP TO UGANDA
# ============================================================
print("[INFO] Loading CHIRPS data...")
ds = xr.open_dataset(FILE)
pr = ds['precip']

print(f"[INFO] Global shape: {pr.shape}")
print(f"[INFO] Time range: {str(pr.time.min().values)[:10]} to {str(pr.time.max().values)[:10]}")

# CRITICAL FIX: latitude is ASCENDING (-49.975 to +49.975)
# For ascending: slice(low, high) = slice(-1.5, 4.2)
# For longitude: slice(29.5, 35.0)
uganda_rain = pr.sel(
    latitude=slice(-1.5, 4.2),      # ASCENDING: low to high
    longitude=slice(29.5, 35.0),     # ASCENDING: low to high
    time=slice('2015-01-01', '2023-12-31')
)

print(f"[✓] Clipped shape: {uganda_rain.shape}")
print(f"[✓] Lat range: {float(uganda_rain.latitude.min()):.3f} to {float(uganda_rain.latitude.max()):.3f}")
print(f"[✓] Lon range: {float(uganda_rain.longitude.min()):.3f} to {float(uganda_rain.longitude.max()):.3f}")
print(f"[✓] Time range: {str(uganda_rain.time.min().values)[:10]} to {str(uganda_rain.time.max().values)[:10]}")

# Save clipped (optional, ~15 MB)
uganda_rain.to_netcdf("uganda_chirps_clipped.nc")
print("[✓] Saved: uganda_chirps_clipped.nc")

# ============================================================
# STEP 2: EXTRACT DISTRICT TIME SERIES
# ============================================================
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

print("\n[INFO] Extracting district time series...")
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

# ============================================================
# STEP 3: SEASONAL AGGREGATION
# ============================================================
print("\n[INFO] Computing seasonal features...")

def get_season(month):
    if month in [3, 4, 5]:     return 'MAM'
    elif month in [9, 10, 11]: return 'SON'
    elif month in [12, 1, 2]:  return 'DJF'
    else:                       return 'JJA'

df['season'] = df['month'].apply(get_season)

# Seasonal totals
seasonal = df.groupby(['district', 'year', 'season'])['rainfall_mm'].sum().reset_index()
seasonal_pivot = seasonal.pivot_table(
    index=['district', 'year'],
    columns='season',
    values='rainfall_mm'
).reset_index()

# Annual stats
annual = df.groupby(['district', 'year']).agg(
    annual_rainfall=('rainfall_mm', 'sum'),
    rain_cv=('rainfall_mm', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    max_monthly=('rainfall_mm', 'max'),
    min_monthly=('rainfall_mm', 'min'),
    rainy_months=('rainfall_mm', lambda x: (x > 50).sum())
).reset_index()

features = seasonal_pivot.merge(annual, on=['district', 'year'])

# Ensure all seasons exist
for season in ['MAM', 'SON', 'DJF', 'JJA']:
    if season not in features.columns:
        features[season] = 0

# Reorder
cols = ['district', 'year', 'MAM', 'SON', 'DJF', 'JJA',
        'annual_rainfall', 'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']
features = features[[c for c in cols if c in features.columns]]

print(f"[✓] Feature matrix shape: {features.shape}")
print("\nPreview:")
print(features.head(12).to_string())

# Save
features.to_csv('uganda_rainfall_features.csv', index=False)
print("\n[✓] Saved: uganda_rainfall_features.csv")

# ============================================================
# STEP 4: QUICK SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Districts: {features['district'].nunique()}")
print(f"Years: {features['year'].min()} to {features['year'].max()}")
print(f"Total records: {len(features)}")
print(f"\nRainfall statistics (annual, mm):")
print(features['annual_rainfall'].describe().round(1))
print(f"\nMAM rainfall statistics (mm):")
print(features['MAM'].describe().round(1))
