"""ClimateSERV CHIRPS daily rainfall extraction and feature engineering.

Fetches daily CHIRPS v2.0 (0.05 deg) rainfall time series for the district
centroids from the ClimateSERV API (``datatype=0``, daily), caches the raw
responses under ``data/raw/climate_cache/climateserv/`` and derives the
seasonal features listed in ``configs/data.yaml`` -> ``daily_required_features``.

The API requires a POST with form data (GET has been observed to return 500);
the request format was taken from the live frontend (``/static/frontend/js/map.js``).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from yaml import safe_load

from cropyield.data.paths import CLIMATESERV_CACHE, CONFIGS, INTERIM
from cropyield.data.provenance import RAIN_CHIRPS_DAILY, add_provenance

log = logging.getLogger(__name__)

BASE_URL = "https://climateserv.servirglobal.net"
API = f"{BASE_URL}/api"
DATATYPE_CHIRPS_DAILY = "0"

POLL_INTERVAL_S = 2.0
REQUEST_TIMEOUT_S = 60.0
MAX_POLL_S = 600.0
MAX_RETRIES = 3


def _load_cfg() -> dict:
    with open(CONFIGS / "features.yaml") as fh:
        return safe_load(fh)


def _new_session() -> requests.Session:
    """Session primed with the CSRF cookie the API expects."""
    s = requests.Session()
    s.get(f"{BASE_URL}/map", timeout=REQUEST_TIMEOUT_S)
    return s


def submit_request(session: requests.Session, begintime: str, endtime: str,
                   lon: float, lat: float) -> str:
    """Submit a point time-series request; return the job id."""
    geom = json.dumps({"type": "Point", "coordinates": [lon, lat]})
    form = {
        "datatype": DATATYPE_CHIRPS_DAILY,
        "ensemble": "false",
        "begintime": begintime,
        "endtime": endtime,
        "intervaltype": "0",
        "operationtype": "0",
        "dateType_Category": "default",
        "isZip_CurrentDataType": "false",
        "is_from_ui": "true",
        "geometry": geom,
    }
    resp = session.post(f"{API}/submitDataRequest/", data=form,
                        timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    job = resp.json()[0]
    return job


def _progress(session: requests.Session, job: str) -> float:
    resp = session.get(f"{API}/getDataRequestProgress/?id={job}",
                       timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    text = resp.text.strip().strip("[]")
    return float(text)


def poll_job(session: requests.Session, job: str) -> None:
    """Block until the job completes (raises on timeout)."""
    waited = 0.0
    while waited < MAX_POLL_S:
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
        if _progress(session, job) >= 100:
            return
    raise TimeoutError(f"ClimateSERV job {job} did not finish in {MAX_POLL_S}s")


def fetch_job(session: requests.Session, job: str) -> list[dict]:
    resp = session.get(f"{API}/getDataFromRequest/?id={job}",
                       timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()["data"]


def _fetch_one(district: str, lon: float, lat: float,
               begintime: str, endtime: str, cache: Path) -> dict | None:
    cache_file = cache / f"{district}.json"
    if cache_file.exists():
        with open(cache_file) as fh:
            return json.load(fh)
    session = _new_session()
    for attempt in range(MAX_RETRIES):
        try:
            job = submit_request(session, begintime, endtime, lon, lat)
            poll_job(session, job)
            data = fetch_job(session, job)
            if len(data) < 100:  # unreasonably short response
                raise ValueError(f"short response: {len(data)} records")
            cache_file.write_text(json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001
            log.warning("district %s attempt %d failed: %s",
                        district, attempt + 1, exc)
            if attempt == MAX_RETRIES - 1:
                log.error("district %s failed permanently", district)
                return None
            time.sleep(5 * (attempt + 1))
    return None


def fetch_daily_rainfall(points: pd.DataFrame,
                         begintime: str = "01/01/2015",
                         endtime: str = "12/31/2023",
                         workers: int = 3,
                         cache: Path = CLIMATESERV_CACHE) -> pd.DataFrame:
    """Fetch daily CHIRPS rainfall for each point; returns long-format frame.

    ``points`` must have columns ``district``, ``lon``, ``lat``.
    """
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, d, lon, lat, begintime, endtime, cache): d
            for d, lon, lat in zip(points["district"], points["lon"], points["lat"])
        }
        for i, future in enumerate(as_completed(futures), start=1):
            district = futures[future]
            data = future.result()
            if data is None:
                continue
            for rec in data:
                rows.append({
                    "district": district,
                    "date": pd.Timestamp(rec["isodate"]),
                    "rain_mm": float(rec["raw_value"]),
                })
            if i % 20 == 0:
                log.info("fetched %d/%d points", i, len(futures))
    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["doy"] = df["date"].dt.dayofyear
    return df.sort_values(["district", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Daily-derived seasonal features (configs/data.yaml -> daily_required_features)
# --------------------------------------------------------------------------

DRY_DAY_MM = 1.0          # a day is "dry" below this threshold
WET_DAY_MM = 1.0          # a day is "wet" at or above this threshold


def _dry_spells(rain: pd.Series) -> list[tuple[int, int]]:
    """Return (start_idx, length) for runs of dry days."""
    dry = (rain < DRY_DAY_MM).to_numpy()
    spells = []
    start = None
    for i, is_dry in enumerate(dry):
        if is_dry and start is None:
            start = i
        elif not is_dry and start is not None:
            spells.append((start, i - start))
            start = None
    if start is not None:
        spells.append((start, len(rain) - start))
    return spells


def _max_5day(rain: pd.Series) -> float:
    if len(rain) == 0:
        return 0.0
    return float(rain.rolling(5, min_periods=1).sum().max())


def _onset(rain: pd.Series, onset_cfg: dict, window_doy: tuple[int, int]) -> int | None:
    """First day meeting the onset rule; None if no valid onset in window."""
    acc_mm = onset_cfg["accumulation_mm"]
    acc_days = onset_cfg["accumulation_days"]
    dry_spell = onset_cfg["dry_spell_days"]
    followup = onset_cfg["followup_days"]
    values = rain.to_numpy()
    doys = np_doy(rain.index)
    start, end = window_doy
    for i in range(len(values) - acc_days + 1):
        d = doys[i]
        if d < start or d > end:
            continue
        if values[i:i + acc_days].sum() >= acc_mm:
            future = values[i + acc_days:i + acc_days + followup]
            ok = True
            run = 0
            for v in future:
                run = run + 1 if v < DRY_DAY_MM else 0
                if run >= dry_spell:
                    ok = False
                    break
            if ok:
                return int(d)
    return None


def _false_onset(rain: pd.Series, onset_cfg: dict, window_doy: tuple[int, int]) -> int:
    """1 if an earlier onset candidate was rejected by the dry-spell test."""
    acc_mm = onset_cfg["accumulation_mm"]
    acc_days = onset_cfg["accumulation_days"]
    dry_spell = onset_cfg["dry_spell_days"]
    followup = onset_cfg["followup_days"]
    values = rain.to_numpy()
    doys = np_doy(rain.index)
    start, end = window_doy
    for i in range(len(values) - acc_days + 1):
        d = doys[i]
        if d < start or d > end:
            continue
        if values[i:i + acc_days].sum() >= acc_mm:
            future = values[i + acc_days:i + acc_days + followup]
            run = 0
            for v in future:
                run = run + 1 if v < DRY_DAY_MM else 0
                if run >= dry_spell:
                    return 1
            return 0
    return 0


def _cessation(rain: pd.Series, onset_cfg: dict,
               window_doy: tuple[int, int], onset: int | None = None) -> int | None:
    """First day of the final 7+ day dry spell that starts at/after onset,
    closing the season; fallback to the last wet day in the window."""
    dry_spell = onset_cfg["dry_spell_days"]
    doys = np_doy(rain.index)
    start, end = window_doy
    in_window = (doys >= start) & (doys <= end)
    window = rain[in_window]
    spells = _dry_spells(window)
    for s_idx, length in reversed(spells):
        if length >= dry_spell:
            spell_start = int(doys[in_window][s_idx])
            if onset is None or spell_start >= onset:
                return spell_start
    wet = window[window >= WET_DAY_MM]
    if len(wet) == 0:
        return None
    return int(wet.index[-1].dayofyear)


def np_doy(index: pd.DatetimeIndex) -> "pd.Index":
    """Day-of-year as a numpy array (mirrors index order)."""
    return index.dayofyear.to_numpy()


def _season_rows(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per district-year-season with daily-derived features."""
    season_windows = cfg["season_windows"]
    first_months = set(season_windows["first_season"])
    second_months = set(season_windows["second_season"])
    onset_cfg = cfg["rainfall"]["onset"]
    rows = []
    for (district, year), g in daily.groupby(["district", "year"]):
        g = g.set_index("date")
        base = {"district": district, "year": year}
        for name, months, window_doy in (
            ("first", first_months, (onset_cfg["earliest_onset_doy"],
                                     onset_cfg["latest_onset_doy"])),
            ("second", second_months, (213, 334)),
        ):
            season = g[g["month"].isin(months)]
            rain = season["rain_mm"]
            if len(rain) == 0:
                continue
            rain.index = season.index
            onset = _onset(rain, onset_cfg, window_doy)
            cessation = _cessation(rain, onset_cfg, window_doy, onset)
            dry_spells = _dry_spells(rain)
            wet = rain[rain >= WET_DAY_MM]
            row = {
                **base,
                "season": name,
                "season_total_mm": float(rain.sum()),
                "longest_dry_spell_days": max((l for _, l in dry_spells), default=0),
                "dry_spell_count_7d": sum(1 for _, l in dry_spells if l >= 7),
                "dry_spell_count_10d": sum(1 for _, l in dry_spells if l >= 10),
                "rain_days_1mm": int((rain >= 1).sum()),
                "rain_days_10mm": int((rain >= 10).sum()),
                "rain_days_20mm": int((rain >= 20).sum()),
                "mean_wet_day_rainfall": float(wet.mean()) if len(wet) else 0.0,
                "maximum_5day_rainfall": _max_5day(rain),
                "season_onset_day": onset,
                "season_cessation_day": cessation,
                "season_length_days": (cessation - onset + 1
                                       if onset is not None and cessation is not None
                                       and cessation >= onset else None),
                "false_onset_flag": _false_onset(rain, onset_cfg, window_doy),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_daily_features(points: pd.DataFrame,
                         daily: pd.DataFrame | None = None,
                         out_csv: Path | None = None,
                         **fetch_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch daily rainfall for ``points`` and derive seasonal features.

    Returns (daily_long, features). If ``out_csv`` is given the features are
    also written there with provenance columns.
    """
    cfg = _load_cfg()
    if daily is None:
        daily = fetch_daily_rainfall(points, **fetch_kwargs)
    features = _season_rows(daily, cfg)
    features = features.sort_values(["district", "year", "season"]).reset_index(drop=True)
    if out_csv is not None:
        out = add_provenance(
            features,
            rainfall_source=RAIN_CHIRPS_DAILY,
            yield_granularity="district",
            quality_note=(
                "Daily CHIRPS v2.0 0.05deg nearest pixel at district centroid "
                "via ClimateSERV API; onset rule 20mm/3d + no 7d dry spell "
                "in 15 follow-up days."
            ),
        )
        out.to_csv(out_csv, index=False)
    return daily, features


def main() -> None:
    import sys

    from cropyield.data.districts import build_district_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    centroids = build_district_table()[["district", "lat", "lon"]]
    centroids = centroids.rename(columns={"lat": "lat", "lon": "lon"})
    daily, features = build_daily_features(
        centroids,
        out_csv=INTERIM / "uganda_daily_features_climateserv.csv",
    )
    features.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
