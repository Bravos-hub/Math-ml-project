"""SoilGrids v2.0 point soil properties extraction.

Fetches modelled soil properties (mean of prediction interval) at district
centroids from the ISRIC SoilGrids REST API. As of 2026-07 the API only
accepts a SINGLE property and a SINGLE depth per request (multi-value
requests return 500 or empty layers), so each point requires
``n_properties x n_depths`` requests; responses are cached under
``data/raw/climate_cache/soilgrids/``.

Properties are aggregated to the 0-30 cm depth-weighted mean and converted
to conventional units (``configs/features.yaml`` -> ``soil``).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from cropyield.data.paths import INTERIM
from cropyield.data.provenance import SOIL_SOILGRIDS, add_provenance

log = logging.getLogger(__name__)

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["clay", "sand", "silt", "soc", "bdod", "cec", "phh2o"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]
DEPTH_WEIGHTS = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
MAX_RETRIES = 4


def _fetch_one(district: str, lon: float, lat: float, prop: str, depth: str,
               cache: Path) -> dict | None:
    """Fetch one (property, depth) value; if the API returns a null mean
    (holes in the SoilGrids mosaic), retry at nearby offsets (up to ~9 km)
    and return the nearest valid pixel (its geometry records the offset)."""
    cache_file = cache / f"{district}__{prop}__{depth}.json"
    if cache_file.exists():
        with open(cache_file) as fh:
            return json.load(fh)
    candidates = [(lon, lat)]
    for km in (2.2, 4.4, 8.9):
        for dlon, dlat in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)):
            candidates.append((lon + dlon * km / 111.0,
                               lat + dlat * km / 111.0))
    params = {"property": prop, "depth": depth,
              "value": "mean", "type": "Point"}
    last_exc = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(5 * (2 ** (attempt - 1)))
        for clon, clat in candidates:
            try:
                r = requests.get(SOILGRIDS_URL,
                                 params={**params, "lon": clon, "lat": clat},
                                 timeout=60)
                r.raise_for_status()
                data = r.json()
                layers = data.get("properties", {}).get("layers", [])
                if not layers or layers[0]["depths"][0]["values"].get("mean") is None:
                    continue
                cache_file.write_text(json.dumps(data))
                return data
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break
        if last_exc is not None:
            log.warning("%s %s %s attempt %d: %s", district, prop, depth,
                        attempt + 1, last_exc)
    log.error("%s %s %s failed permanently", district, prop, depth)
    return None


def fetch_soil_properties(points: pd.DataFrame, workers: int = 6,
                          cache: Path | None = None) -> pd.DataFrame:
    """Fetch soil properties for each point; returns one row per point."""
    cache = cache or Path("data/raw/climate_cache/soilgrids")
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(points) * len(PROPERTIES) * len(DEPTHS)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, d, lon, lat, p, depth, cache): (d, p, depth)
            for d, lon, lat in zip(points["district"], points["lon"], points["lat"])
            for p in PROPERTIES
            for depth in DEPTHS
        }
        by_key: dict[tuple, dict] = {}
        for future in as_completed(futures):
            district, prop, depth = futures[future]
            data = future.result()
            if data is not None:
                by_key[(district, prop, depth)] = data
            done += 1
            if done % 200 == 0:
                log.info("soilgrids %d/%d requests", done, total)
    for _, row in points.iterrows():
        district = row["district"]
        record = {"district": district, "lat": row["lat"], "lon": row["lon"]}
        for prop in PROPERTIES:
            values = []
            weights = []
            d_factor = None
            for depth in DEPTHS:
                data = by_key.get((district, prop, depth))
                if data is None:
                    continue
                layer = data["properties"]["layers"][0]
                d_factor = layer["unit_measure"]["d_factor"]
                for dep in layer["depths"]:
                    if dep["label"] == depth and dep["values"].get("mean") is not None:
                        values.append(float(dep["values"]["mean"]))
                        weights.append(DEPTH_WEIGHTS[depth])
            if values and d_factor:
                wsum = sum(weights)
                record[prop] = sum(v * w / wsum for v, w in zip(values, weights)) \
                    / d_factor
            else:
                record[prop] = None
        rows.append(record)
    return pd.DataFrame(rows)


def build_soil_features(points: pd.DataFrame,
                        out_csv: Path | None = None,
                        **kwargs) -> pd.DataFrame:
    """Fetch soil properties and write the interim feature table."""
    df = fetch_soil_properties(points, **kwargs)
    if out_csv is not None:
        out = add_provenance(
            df,
            soil_source=SOIL_SOILGRIDS,
            yield_granularity="district",
            quality_note=(
                "SoilGrids v2.0 modelled mean at 0-30 cm depth-weighted "
                "average of 0-5/5-15/15-30 cm layers; point prediction at "
                "district centroid."
            ),
        )
        out.to_csv(out_csv, index=False)
    return df


def main() -> None:
    import sys

    from cropyield.data.districts import build_district_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    centroids = build_district_table()[["district", "lat", "lon"]]
    soil = build_soil_features(
        centroids,
        out_csv=INTERIM / "uganda_soil_features_114.csv",
    )
    soil.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
