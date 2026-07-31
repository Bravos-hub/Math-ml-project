"""CHIRPS monthly rainfall extraction and feature engineering.

Primary source: the local global CHIRPS v2.0 monthly NetCDF
(``data/raw/chirps-v2.0.monthly.nc``, 1981-01 .. 2026-06, 0.05 deg).
A Uganda clip (2015-2023) is generated once and cached; the full history
slice is used to build the 1981-2010 climatological baseline for anomaly
and z-score features.

Rainfall features (monthly basis):
    MAM, SON, DJF, JJA (seasonal totals, mm)
    annual_rainfall (mm)
    rain_cv (coefficient of variation of monthly totals)
    max_monthly, min_monthly (mm)
    rainy_months (months with >= 50 mm)
    rainfall_anomaly, rainfall_zscore (vs 1981-2010 district climatology)
    MAM_anomaly, SON_anomaly (seasonal anomalies)

Note: dry-spell / onset / intensity features require DAILY rainfall and are
built by ``cropyield.data.climateserv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .paths import CHIRPS_GLOBAL_NC, CHIRPS_UGANDA_NC, INTERIM, RAW

RAINY_MONTH_THRESHOLD_MM = 50.0
CLIMATOLOGY_YEARS = (1981, 2010)
FEATURE_YEARS = range(2015, 2024)

OUTPUT_FEATURES = INTERIM / "uganda_rainfall_features_114.csv"
OUTPUT_CLIMATOLOGY = INTERIM / "uganda_rainfall_climatology_114.csv"

SEASONS = {"MAM": [3, 4, 5], "SON": [9, 10, 11], "DJF": [12, 1, 2], "JJA": [6, 7, 8]}


def _load_uganda_slice(years: tuple[int, int] | None = None) -> xr.DataArray:
    """Load (and build, if needed) the Uganda-clipped CHIRPS slice."""
    if not CHIRPS_UGANDA_NC.exists():
        if not CHIRPS_GLOBAL_NC.exists():
            raise FileNotFoundError(
                f"Missing CHIRPS data: neither {CHIRPS_UGANDA_NC.name} nor "
                f"{CHIRPS_GLOBAL_NC.name} is present"
            )
        print(f"[INFO] Building {CHIRPS_UGANDA_NC.name} from global NetCDF ...")
        ds = xr.open_dataset(CHIRPS_GLOBAL_NC)
        rain = ds["precip"].sel(
            latitude=slice(-1.5, 4.2),
            longitude=slice(29.5, 35.0),
            time=slice("2015-01-01", "2023-12-31"),
        )
        rain.to_netcdf(CHIRPS_UGANDA_NC)
    da = xr.open_dataarray(CHIRPS_UGANDA_NC)
    if years is not None:
        da = da.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    return da


def _extract_monthly_panel(
    rain: xr.DataArray, centroids: pd.DataFrame
) -> pd.DataFrame:
    """Long-format monthly rainfall at each district centroid (nearest pixel)."""
    data = rain.values  # materialize the (small) Uganda clip once
    lats = rain.latitude.values
    lons = rain.longitude.values
    months = pd.to_datetime(rain.time.values)

    records = []
    for i, district in enumerate(centroids["district"]):
        ilat = int(np.argmin(np.abs(lats - centroids["lat"].iloc[i])))
        ilon = int(np.argmin(np.abs(lons - centroids["lon"].iloc[i])))
        for j, dt in enumerate(months):
            records.append(
                {
                    "district": district,
                    "year": dt.year,
                    "month": dt.month,
                    "rainfall_mm": float(data[j, ilat, ilon]),
                }
            )
    return pd.DataFrame(records)


def _monthly_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the monthly panel into district-year seasonal features."""
    rows = []
    for (district, year), g in panel.groupby(["district", "year"]):
        by_month = g.set_index("month")["rainfall_mm"]
        row = {"district": district, "year": year}
        for season, months in SEASONS.items():
            row[season] = by_month.reindex(months).sum(skipna=True)
        row["annual_rainfall"] = by_month.sum(skipna=True)
        row["rain_cv"] = (
            by_month.std(ddof=1) / by_month.mean() if by_month.mean() > 0 else 0.0
        )
        row["max_monthly"] = by_month.max()
        row["min_monthly"] = by_month.min()
        row["rainy_months"] = int((by_month >= RAINY_MONTH_THRESHOLD_MM).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _climatology(centroids: pd.DataFrame) -> pd.DataFrame:
    """1981-2010 monthly mean and interannual std at each district.

    The global NetCDF is large, so the Uganda time slice is loaded into
    memory once and indexed with plain numpy (nearest pixel per centroid).
    """
    if not CHIRPS_GLOBAL_NC.exists():
        raise FileNotFoundError(f"Missing global CHIRPS for climatology: {CHIRPS_GLOBAL_NC}")
    ds = xr.open_dataset(CHIRPS_GLOBAL_NC)
    sub = ds["precip"].sel(
        latitude=slice(-1.5, 4.2),
        longitude=slice(29.5, 35.0),
        time=slice(f"{CLIMATOLOGY_YEARS[0]}-01-01", f"{CLIMATOLOGY_YEARS[1]}-12-31"),
    ).load()
    data = sub.values  # (n_month, n_lat, n_lon)
    lats = sub.latitude.values
    lons = sub.longitude.values
    months = pd.to_datetime(sub.time.values)
    month_of_year = np.array([m.month for m in months])

    rows = []
    for i, district in enumerate(centroids["district"]):
        ilat = int(np.argmin(np.abs(lats - centroids["lat"].iloc[i])))
        ilon = int(np.argmin(np.abs(lons - centroids["lon"].iloc[i])))
        ts = data[:, ilat, ilon]
        clim_mean = {}
        clim_std = {}
        for m in range(1, 13):
            values = ts[month_of_year == m]
            clim_mean[m] = float(values.mean())
            clim_std[m] = float(values.std(ddof=1))
        row = {
            "district": district,
            "clim_annual_mean": sum(clim_mean.values()),
            "clim_annual_std": float(
                np.sqrt(sum(clim_std[m] ** 2 for m in range(1, 13)))
            ),
        }
        for season, months in SEASONS.items():
            row[f"clim_{season}_mean"] = sum(clim_mean[m] for m in months)
            row[f"clim_{season}_std"] = float(
                np.sqrt(sum(clim_std[m] ** 2 for m in months))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _anomalies(features: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    out = features.merge(clim, on="district", how="left", validate="many_to_one")
    out["rainfall_anomaly"] = out["annual_rainfall"] - out["clim_annual_mean"]
    out["rainfall_zscore"] = out["rainfall_anomaly"] / out["clim_annual_std"].replace(0, np.nan)
    for season in SEASONS:
        out[f"{season}_anomaly"] = (
            out[season] - out[f"clim_{season}_mean"]
        )
        out[f"{season}_zscore"] = (
            out[f"{season}_anomaly"]
            / out[f"clim_{season}_std"].replace(0, np.nan)
        )
    return out


def build_rainfall_features(
    centroids: pd.DataFrame,
    output_features: Path = OUTPUT_FEATURES,
    output_climatology: Path = OUTPUT_CLIMATOLOGY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build district-year monthly rainfall features and climatology."""
    rain = _load_uganda_slice((FEATURE_YEARS.start, FEATURE_YEARS.stop - 1))
    panel = _extract_monthly_panel(rain, centroids)
    features = _monthly_features(panel)
    clim = _climatology(centroids)
    features = _anomalies(features, clim)
    features.to_csv(output_features, index=False)
    clim.to_csv(output_climatology, index=False)
    return features, clim


if __name__ == "__main__":
    from .districts import build_district_table

    centroids = build_district_table()[["district", "lat", "lon"]]
    features, clim = build_rainfall_features(centroids)
    print(f"[✓] Rainfall features: {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"[✓] Climatology: {clim.shape[0]} districts")
