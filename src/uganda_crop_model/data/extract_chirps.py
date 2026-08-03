"""Polygon-weighted daily CHIRPS extraction.

This is the authoritative extractor for new analyses.  It accepts an xarray
dataset/data-array and a boundary table containing polygon geometries, then
returns one area-weighted daily series per spatial unit.  The implementation
does not require geopandas at runtime: geometry objects with ``contains`` and
``bounds`` (for example shapely polygons) are sufficient.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _point_in_geometry(lon: float, lat: float, geometry: object) -> bool:
    try:
        from shapely.geometry import Point

        return bool(geometry.contains(Point(lon, lat)) or geometry.touches(Point(lon, lat)))
    except (ImportError, AttributeError):
        # Lightweight fallback for GeoJSON-like polygons.  This is intended
        # for tests and simple boundaries when shapely is not installed.
        if isinstance(geometry, Mapping):
            coordinates = geometry.get("coordinates", [])
            if geometry.get("type") == "Polygon":
                ring = coordinates[0]
            elif geometry.get("type") == "MultiPolygon":
                ring = coordinates[0][0]
            else:
                ring = []
            inside = False
            j = len(ring) - 1
            for i, (x1, y1) in enumerate(ring):
                x2, y2 = ring[j]
                intersects = ((y1 > lat) != (y2 > lat)) and (
                    lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1
                )
                if intersects:
                    inside = not inside
                j = i
            return inside
    return False


def extract_polygon_daily_mean(
    dataset: xr.Dataset | xr.DataArray | str | Path,
    boundaries: pd.DataFrame,
    *,
    variable: str = "precip",
    unit_column: str = "spatial_unit",
    geometry_column: str = "geometry",
    latitude_column: str | None = None,
    longitude_column: str | None = None,
) -> pd.DataFrame:
    """Extract area-weighted daily means for every polygon.

    The returned columns are ``spatial_unit``, ``date`` and ``rain_mm``.
    Latitude weights use ``cos(latitude)`` and are normalized separately for
    each polygon.  Empty polygons raise instead of silently creating missing
    predictors.
    """
    if unit_column not in boundaries or geometry_column not in boundaries:
        raise ValueError(f"boundaries must contain {unit_column!r} and {geometry_column!r}")

    opened = False
    if isinstance(dataset, (str, Path)):
        dataset = xr.open_dataset(dataset)
        opened = True
    try:
        if isinstance(dataset, xr.DataArray):
            data = dataset
        else:
            if variable not in dataset:
                raise ValueError(f"Variable {variable!r} not found in dataset.")
            data = dataset[variable]

        lat_name = latitude_column or next((n for n in ("latitude", "lat") if n in data.coords), None)
        lon_name = longitude_column or next((n for n in ("longitude", "lon") if n in data.coords), None)
        time_name = next((n for n in ("time", "date") if n in data.coords), None)
        if not all((lat_name, lon_name, time_name)):
            raise ValueError("Dataset must expose latitude, longitude, and time coordinates.")

        lats = np.asarray(data[lat_name].values, dtype=float)
        lons = np.asarray(data[lon_name].values, dtype=float)
        values = data.transpose(time_name, lat_name, lon_name).values
        dates = pd.to_datetime(data[time_name].values)
        records: list[dict[str, object]] = []

        for boundary in boundaries.itertuples(index=False):
            unit = getattr(boundary, unit_column)
            geometry = getattr(boundary, geometry_column)
            mask = np.zeros((len(lats), len(lons)), dtype=bool)
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    mask[i, j] = _point_in_geometry(float(lon), float(lat), geometry)
            if not mask.any():
                raise ValueError(f"Polygon {unit!r} contains no CHIRPS grid-cell centers.")
            weights = np.cos(np.deg2rad(lats))[:, None] * mask
            weights = weights / weights.sum()
            series = np.nansum(values * weights[None, :, :], axis=(1, 2))
            for date, amount in zip(dates, series):
                records.append({"spatial_unit": unit, "date": date, "rain_mm": float(amount)})

        return pd.DataFrame(records).sort_values(["spatial_unit", "date"]).reset_index(drop=True)
    finally:
        if opened:
            dataset.close()
