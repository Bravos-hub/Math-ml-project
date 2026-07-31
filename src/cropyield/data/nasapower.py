"""NASA POWER daily temperature extraction (MERRA-2, 0.5 deg).

Fetches daily T2M_MAX/T2M_MIN for the district centroids, caches raw JSON
responses under ``data/raw/climate_cache/nasapower/`` and derives the
temperature features from ``configs/features.yaml`` -> ``temperature``:
growing degree days (GDD, base 10 C), heat stress, warm nights, heatwaves
and interactions with the daily rainfall features.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from yaml import safe_load

from cropyield.data.paths import CONFIGS, INTERIM, POWER_CACHE
from cropyield.data.provenance import TEMP_NASA_POWER, add_provenance

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M_MAX,T2M_MIN"
MAX_RETRIES = 3
FILL = -999.0


def _load_temp_cfg() -> dict:
    with open(CONFIGS / "features.yaml") as fh:
        return safe_load(fh)["temperature"]


def _fetch_one(district: str, lon: float, lat: float,
               start: str, end: str, cache: Path) -> dict | None:
    cache_file = cache / f"{district}.json"
    if cache_file.exists():
        with open(cache_file) as fh:
            return json.load(fh)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                POWER_URL,
                params=dict(parameters=PARAMS, community="AG",
                            longitude=f"{lon:.4f}", latitude=f"{lat:.4f}",
                            start=start, end=end, format="JSON"),
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()["properties"]["parameter"]
            if not data.get("T2M_MAX"):
                raise ValueError("empty parameter block")
            cache_file.write_text(json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001
            log.warning("district %s attempt %d failed: %s",
                        district, attempt + 1, exc)
            if attempt == MAX_RETRIES - 1:
                log.error("district %s failed permanently", district)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def fetch_daily_temperature(points: pd.DataFrame,
                            start: str = "20150101",
                            end: str = "20231231",
                            workers: int = 3,
                            cache: Path = POWER_CACHE) -> pd.DataFrame:
    """Fetch daily T2M_MAX/T2M_MIN per point; returns long-format frame."""
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, d, lon, lat, start, end, cache): d
            for d, lon, lat in zip(points["district"], points["lon"], points["lat"])
        }
        for i, future in enumerate(as_completed(futures), start=1):
            district = futures[future]
            data = future.result()
            if data is None:
                continue
            for day, tmax in data["T2M_MAX"].items():
                tmin = data["T2M_MIN"].get(day, FILL)
                date = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
                rows.append({
                    "district": district,
                    "date": date,
                    "tmax_c": float(tmax) if float(tmax) != FILL else None,
                    "tmin_c": float(tmin) if float(tmin) != FILL else None,
                })
            if i % 20 == 0:
                log.info("fetched %d/%d points", i, len(futures))
    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def _season_rows(daily: pd.DataFrame, daily_rain: pd.DataFrame,
                 cfg: dict) -> pd.DataFrame:
    """One row per district-year-season with temperature features."""
    season_windows = _season_windows()
    gdd_base = cfg["gdd_base_c"]
    heat_th = cfg["heat_threshold_c"]
    extreme_th = cfg["extreme_heat_threshold_c"]
    warm_night = cfg["warm_night_threshold_c"]
    heatwave_days = cfg["heatwave_days"]
    first_months = set(season_windows["first_season"])
    second_months = set(season_windows["second_season"])
    rows = []
    for (district, year), g in daily.groupby(["district", "year"]):
        g = g.set_index("date")
        base = {"district": district, "year": year}
        for name, months in (("first", first_months), ("second", second_months)):
            season = g[g["month"].isin(months)]
            if len(season) == 0:
                continue
            tmax = season["tmax_c"].dropna()
            tmin = season["tmin_c"].dropna()
            gdd = ((tmax + tmin) / 2 - gdd_base).clip(lower=0).sum()
            heat_days = int((tmax >= heat_th).sum())
            extreme_days = int((tmax >= extreme_th).sum())
            warm_nights = int((tmin >= warm_night).sum())
            # heatwaves: >= heatwave_days consecutive days with tmax >= heat_th
            heatwave_count = 0
            run = 0
            for v in tmax.to_numpy():
                run = run + 1 if v >= heat_th else 0
                if run == heatwave_days:
                    heatwave_count += 1
            row = {
                **base,
                "season": name,
                "season_gdd": float(gdd),
                "heat_days": heat_days,
                "extreme_heat_days": extreme_days,
                "warm_night_days": warm_nights,
                "heatwave_count": heatwave_count,
            }
            if daily_rain is not None:
                rain_season = daily_rain[
                    (daily_rain["district"] == district)
                    & (daily_rain["year"] == year)
                    & (daily_rain["month"].isin(months))
                ]
                if len(rain_season) and len(tmax):
                    rain_season = rain_season.set_index("date").reindex(
                        season.index).sort_index()
                    tmax_s = tmax.reindex(rain_season.index)
                    aligned = pd.DataFrame({
                        "tmax": tmax_s.to_numpy(),
                        "rain": rain_season["rain_mm"].to_numpy(),
                    }).dropna()
                    if len(aligned):
                        wet_tmax = aligned.loc[aligned["rain"] >= 1.0, "tmax"]
                        row["wet_day_tmax_mean"] = float(wet_tmax.mean())
                        row["wet_day_gdd"] = float(
                            ((wet_tmax - gdd_base).clip(lower=0)).sum()
                            if len(wet_tmax) else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _season_windows() -> dict:
    with open(CONFIGS / "features.yaml") as fh:
        return safe_load(fh)["season_windows"]


def build_temperature_features(points: pd.DataFrame,
                               daily_rain: pd.DataFrame | None = None,
                               temp: pd.DataFrame | None = None,
                               out_csv: Path | None = None,
                               **fetch_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch daily temperatures and derive seasonal features.

    Returns (daily_temp, features). ``daily_rain`` optionally enables the
    rainfall-temperature interaction features.
    """
    cfg = _load_temp_cfg()
    if temp is None:
        temp = fetch_daily_temperature(points, **fetch_kwargs)
    features = _season_rows(temp, daily_rain, cfg)
    features = features.sort_values(["district", "year", "season"]).reset_index(drop=True)
    if out_csv is not None:
        out = add_provenance(
            features,
            temperature_source=TEMP_NASA_POWER,
            yield_granularity="district",
            quality_note=(
                "NASA POWER daily T2M_MAX/T2M_MIN (MERRA-2, ~0.5 deg) at "
                "district centroid; GDD base 10C; heatwave = "
                f"{cfg['heatwave_days']}+ consecutive days >= "
                f"{cfg['heat_threshold_c']}C."
            ),
        )
        out.to_csv(out_csv, index=False)
    return temp, features


def main() -> None:
    import sys

    from cropyield.data.districts import build_district_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    centroids = build_district_table()[["district", "lat", "lon"]]
    daily, features = build_temperature_features(
        centroids,
        out_csv=INTERIM / "uganda_temperature_features_nasapower.csv",
    )
    daily.to_csv(INTERIM / "uganda_daily_temperature_nasapower.csv", index=False)
    features.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
