"""Seasonal temperature and thermal-stress features.

Temperature is summarised over exactly the same sub-region polygons and
season dates as rainfall (see ``features/rainfall.build_seasonal_*``).  The
thermal thresholds are configurable research assumptions documented in
``configs/final_maize_aas.yaml``; maize agronomy (base temperature 10 C,
heat stress above 30-35 C) motivates the defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEMPERATURE_SOURCE = "NASA_POWER_daily"
PREDICTOR_GEOGRAPHIC_LEVEL = "sub_region"

BASE_TEMPERATURE_C = 10.0
HEAT_THRESHOLD_C = 32.0
EXTREME_HEAT_THRESHOLD_C = 35.0
COLD_THRESHOLD_C = 10.0


def summarize_temperature_period(
    period: pd.DataFrame,
    *,
    base_temperature_c: float = BASE_TEMPERATURE_C,
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    extreme_heat_threshold_c: float = EXTREME_HEAT_THRESHOLD_C,
    cold_threshold_c: float = COLD_THRESHOLD_C,
) -> dict[str, float | int]:
    required = {"tmax_c", "tmin_c"}
    missing = required.difference(period.columns)

    if missing:
        raise ValueError(
            f"Temperature data missing columns: {sorted(missing)}"
        )

    tmax = pd.to_numeric(period["tmax_c"], errors="coerce")
    tmin = pd.to_numeric(period["tmin_c"], errors="coerce")

    valid = tmax.notna() & tmin.notna()
    tmax = tmax[valid]
    tmin = tmin[valid]

    if tmax.empty:
        raise ValueError("No valid temperature records.")

    tmean = (tmax + tmin) / 2.0

    growing_degree_days = np.maximum(
        tmean - base_temperature_c,
        0.0,
    ).sum()

    return {
        "temperature_mean_c": float(tmean.mean()),
        "temperature_maximum_c": float(tmax.max()),
        "temperature_minimum_c": float(tmin.min()),
        "temperature_range_c": float(tmax.mean() - tmin.mean()),
        "growing_degree_days": float(growing_degree_days),
        "heat_days_32c": int(tmax.ge(heat_threshold_c).sum()),
        "extreme_heat_days_35c": int(tmax.ge(extreme_heat_threshold_c).sum()),
        "cold_days_10c": int(tmin.le(cold_threshold_c).sum()),
    }


def build_seasonal_temperature_features(
    daily_temperature: pd.DataFrame,
    season_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Build one temperature-feature row per (spatial_unit, year, season).

    ``daily_temperature`` needs ``spatial_unit``, ``date``, ``tmax_c``,
    ``tmin_c``.  ``season_calendar`` needs ``spatial_unit``, ``year``,
    ``season``, ``start_date``, ``end_date``.
    """
    calendar = season_calendar.copy()
    calendar["start_date"] = pd.to_datetime(calendar["start_date"])
    calendar["end_date"] = pd.to_datetime(calendar["end_date"])
    calendar["year"] = pd.to_numeric(calendar["year"], errors="coerce")

    daily = daily_temperature.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    rows: list[dict[str, object]] = []

    for row in calendar.itertuples(index=False):
        mask = (
            daily["spatial_unit"].eq(row.spatial_unit)
            & daily["date"].between(
                row.start_date,
                row.end_date,
                inclusive="both",
            )
        )

        period = daily.loc[
            mask,
            ["date", "tmax_c", "tmin_c"],
        ]

        if period.empty:
            raise ValueError(
                "No temperature records for "
                f"{row.spatial_unit}, {row.year}, {row.season}."
            )

        features = summarize_temperature_period(period)

        rows.append(
            {
                "spatial_unit": row.spatial_unit,
                "year": int(row.year),
                "season": row.season,
                **features,
                "temperature_source": TEMPERATURE_SOURCE,
                "predictor_geographic_level": PREDICTOR_GEOGRAPHIC_LEVEL,
            }
        )

    result = pd.DataFrame(rows)

    key = ["spatial_unit", "year", "season"]
    if result.duplicated(key).any():
        raise ValueError("Duplicate seasonal temperature feature rows.")

    return result.sort_values(key).reset_index(drop=True)


if __name__ == "__main__":
    from uganda_crop_model.data.paths import INTERIM, PUBLIC

    daily = pd.read_csv(INTERIM / "subregion_daily_temperature.csv")
    calendar = pd.read_csv(PUBLIC / "season_calendar.csv")
    features = build_seasonal_temperature_features(daily, calendar)
    print(features.head(20).to_string())