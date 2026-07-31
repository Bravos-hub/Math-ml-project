
# ============================================================
# APPROACH 1: DIRECT NETCDF DOWNLOAD (Recommended — No API key needed)
# ============================================================
# CHIRPS provides global monthly NetCDF files you can download directly
# and clip to Uganda's bounding box.

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Uganda bounding box (approximate)
# Latitude:  -1.5° to 4.2°N
# Longitude: 29.5°E to 35.0°E
UGANDA_BOUNDS = {
    'lat_min': -1.5, 'lat_max': 4.2,
    'lon_min': 29.5, 'lon_max': 35.0
}

# Step 1: Download the global monthly CHIRPS NetCDF
# The file is ~2.5 GB for all years (1981–present)
# You only need to do this ONCE

import urllib.request
import os

url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
local_file = "chirps-v2.0.monthly.nc"

if not os.path.exists(local_file):
    print("Downloading CHIRPS monthly data (~2.5 GB)... This may take 10-30 minutes.")
    urllib.request.urlretrieve(url, local_file)
    print("Download complete.")
else:
    print("File already exists.")

# Step 2: Load and clip to Uganda
chirps = xr.open_dataset(local_file)
pr = chirps['precip']

uganda_rain = pr.sel(
    longitude=slice(UGANDA_BOUNDS['lon_min'], UGANDA_BOUNDS['lon_max']),
    latitude=slice(UGANDA_BOUNDS['lat_max'], UGANDA_BOUNDS['lat_min']),  # Note: CHIRPS lat is descending
    time=slice('2015', '2023')
)

print(f"Clipped dataset shape: {uganda_rain.shape}")
print(f"Time range: {uganda_rain.time.min().values} to {uganda_rain.time.max().values}")

# Step 3: Extract time series for specific district centroids
districts = {
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
    'Fort_Portal':(0.671, 30.275)
}

# Extract point data using nearest neighbor
rainfall_records = []
for district, (lat, lon) in districts.items():
    ts = uganda_rain.sel(latitude=lat, longitude=lon, method='nearest')
    for t, val in zip(ts.time.values, ts.values):
        rainfall_records.append({
            'district': district,
            'date': pd.to_datetime(str(t)),
            'year': pd.to_datetime(str(t)).year,
            'month': pd.to_datetime(str(t)).month,
            'rainfall_mm': float(val)
        })

df_rain = pd.DataFrame(rainfall_records)
print(f"\nExtracted {len(df_rain)} monthly records across {len(districts)} districts")
print(df_rain.head(10))

# Step 4: Compute seasonal aggregates for PCA
# Uganda has bimodal rainfall:
#   - Long rains:  March–May (MAM)
#   - Short rains: September–November (SON)
#   - Dry seasons: December–February (DJF), June–August (JJA)

def get_season(month):
    if month in [3, 4, 5]:   return 'MAM'
    elif month in [9, 10, 11]: return 'SON'
    elif month in [12, 1, 2]:  return 'DJF'
    else:                      return 'JJA'

df_rain['season'] = df_rain['month'].apply(get_season)

# Aggregate to seasonal totals per district-year
seasonal = df_rain.groupby(['district', 'year', 'season'])['rainfall_mm'].sum().reset_index()
seasonal_pivot = seasonal.pivot_table(
    index=['district', 'year'],
    columns='season',
    values='rainfall_mm'
).reset_index()

# Also compute annual total and coefficient of variation
annual = df_rain.groupby(['district', 'year']).agg(
    annual_rainfall=('rainfall_mm', 'sum'),
    rain_cv=('rainfall_mm', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    max_monthly=('rainfall_mm', 'max'),
    min_monthly=('rainfall_mm', 'min')
).reset_index()

# Merge
features = seasonal_pivot.merge(annual, on=['district', 'year'])
print("\nSeasonal features ready for PCA:")
print(features.head())
print(f"\nShape: {features.shape}")

# Save
features.to_csv('uganda_chirps_seasonal_features.csv', index=False)
print("\nSaved: uganda_chirps_seasonal_features.csv")
