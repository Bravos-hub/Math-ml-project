#!/usr/bin/env python3
"""
CHIRPS Diagnostic & Fix Script
Run this to inspect your downloaded file and extract features correctly.
"""

import xarray as xr
import pandas as pd
import numpy as np

FILE = "chirps-v2.0.monthly.nc"

print("=" * 60)
print("INSPECTING CHIRPS FILE")
print("=" * 60)

# Load dataset
ds = xr.open_dataset(FILE)
print(f"\nDataset variables: {list(ds.data_vars)}")
print(f"Dataset dimensions: {dict(ds.dims)}")
print(f"Dataset coordinates: {list(ds.coords)}")

# Check coordinate names and ranges
for coord in ds.coords:
    print(f"\n--- Coordinate: '{coord}' ---")
    print(f"  Shape: {ds.coords[coord].shape}")
    print(f"  Min: {float(ds.coords[coord].min()):.4f}")
    print(f"  Max: {float(ds.coords[coord].max()):.4f}")
    print(f"  First 3 values: {ds.coords[coord].values[:3]}")
    print(f"  Last 3 values: {ds.coords[coord].values[-3:]}")

    # Check if monotonic
    vals = ds.coords[coord].values
    if len(vals) > 1:
        diff = vals[1] - vals[0]
        if diff > 0:
            print(f"  Order: ASCENDING (step = +{diff:.6f})")
        else:
            print(f"  Order: DESCENDING (step = {diff:.6f})")

# Check precipitation variable
pr = ds['precip']
print(f"\n--- Precipitation variable ---")
print(f"  Shape: {pr.shape}")
print(f"  Dims: {pr.dims}")
print(f"  Min value: {float(pr.min()):.4f}")
print(f"  Max value: {float(pr.max()):.4f}")

# Try to find Uganda in the data
print("\n" + "=" * 60)
print("TESTING UGANDA EXTRACTION")
print("=" * 60)

# Uganda approximate bounds
lat_min, lat_max = -1.5, 4.2
lon_min, lon_max = 29.5, 35.0

# Try different coordinate names
coord_map = {}
for coord in ds.coords:
    name = coord.lower()
    if 'lat' in name:
        coord_map['lat'] = coord
    elif 'lon' in name:
        coord_map['lon'] = coord

print(f"Detected coordinate mapping: {coord_map}")

lat_coord = coord_map.get('lat', 'latitude')
lon_coord = coord_map.get('lon', 'longitude')

# Check if latitude is ascending or descending
lat_vals = ds.coords[lat_coord].values
lat_ascending = lat_vals[1] > lat_vals[0]

print(f"\nLatitude coordinate '{lat_coord}' is {'ascending' if lat_ascending else 'descending'}")

# Adjust slice based on ordering
if lat_ascending:
    lat_slice = slice(lat_min, lat_max)
else:
    lat_slice = slice(lat_max, lat_min)

lon_vals = ds.coords[lon_coord].values
lon_ascending = lon_vals[1] > lon_vals[0]

if lon_ascending:
    lon_slice = slice(lon_min, lon_max)
else:
    lon_slice = slice(lon_max, lon_min)

print(f"\nUsing latitude slice: {lat_slice}")
print(f"Using longitude slice: {lon_slice}")

# Extract
try:
    subset = pr.sel(**{lat_coord: lat_slice, lon_coord: lon_slice, 'time': slice('2015-01-01', '2023-12-31')})
    print(f"\n[✓] Extraction successful!")
    print(f"    Shape: {subset.shape}")
    print(f"    Dims: {subset.dims}")
    print(f"    Lat range: {float(subset.coords[lat_coord].min()):.4f} to {float(subset.coords[lat_coord].max()):.4f}")
    print(f"    Lon range: {float(subset.coords[lon_coord].min()):.4f} to {float(subset.coords[lon_coord].max()):.4f}")
    print(f"    Time range: {str(subset.time.min().values)[:10]} to {str(subset.time.max().values)[:10]}")
except Exception as e:
    print(f"\n[✗] Extraction failed: {e}")

    # Try with slightly expanded bounds
    print("\nTrying with expanded bounds (+/- 0.5 degrees)...")
    lat_min_e, lat_max_e = lat_min - 0.5, lat_max + 0.5
    lon_min_e, lon_max_e = lon_min - 0.5, lon_max + 0.5

    if lat_ascending:
        lat_slice_e = slice(lat_min_e, lat_max_e)
    else:
        lat_slice_e = slice(lat_max_e, lat_min_e)

    if lon_ascending:
        lon_slice_e = slice(lon_min_e, lon_max_e)
    else:
        lon_slice_e = slice(lon_max_e, lon_min_e)

    try:
        subset = pr.sel(**{lat_coord: lat_slice_e, lon_coord: lon_slice_e, 'time': slice('2015-01-01', '2023-12-31')})
        print(f"[✓] Expanded extraction successful! Shape: {subset.shape}")
    except Exception as e2:
        print(f"[✗] Also failed: {e2}")
        subset = None

# Save diagnostic info
if subset is not None:
    subset.to_netcdf("uganda_chirps_clipped.nc")
    print(f"\n[✓] Saved clipped subset to uganda_chirps_clipped.nc")
