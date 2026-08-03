"""Seasonal rainfall features from daily sub-region series.

Rainfall is summarised over the documented agronomic season windows in
``data/public/season_calendar.csv``.  The season calendar is a research
assumption derived from the long-run Ugandan first/second-rain season
(March-July, August-December) and must be sensitivity-tested before the
final report (see configs/final_maize_aas.yaml).

Onset/cessation use an operational rule (>= 20 mm over 3 days followed by
no 7+ day dry spell in the following 15 days), matching the rule already
documented in configs/features.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAINFALL_SOURCE = "CHIRPS_v2.0_daily_via_ClimateSERV"
PREDICTOR_GEOGRAPHIC_LEVEL = "sub_region"

DRY_DAY_MM = 1.0
WET_DAY_MM = 1.0
ONSET_ACCUMULATION_MM = 20.0
ONSET_ACCUMULATION_DAYS = 3
ONSET_FOLLOWUP_DAYS = 15
ONSET_FUTURE_DRY_SPELL_DAYS = 7


def longest_true_run(values: pd.Series) -> int:
    mask = values.fillna(False).astype(bool)

    if mask.empty or not mask.any():
        return 0

    groups = mask.ne(mask.shift()).cumsum()
    run_lengths = mask.groupby(groups).sum()

    return int(run_lengths.max())


def detect_rainfall_onset(
    rain: pd.Series,
    *,
    three_day_total_mm: float = ONSET_ACCUMULATION_MM,
    future_window_days: int = ONSET_FOLLOWUP_DAYS,
    maximum_future_dry_spell_days: int = ONSET_FUTURE_DRY_SPELL_DAYS,
    dry_day_threshold_mm: float = DRY_DAY_MM,
) -> pd.Timestamp | pd.NaT:
    """Return the day of the first onset-candidate 3-day rainy period.

    Rule: at least ``three_day_total_mm`` over 3 consecutive days,
    followed by no dry spell longer than ``maximum_future_dry_spell_days``
    within the following ``future_window_days``.
    """
    rain = rain.sort_index().fillna(0.0)

    for start in range(0, max(0, len(rain) - future_window_days - 3)):
        first_three = rain.iloc[start : start + 3]

        if first_three.sum() < three_day_total_mm:
            continue

        future = rain.iloc[start + 3 : start + 3 + future_window_days]
        future_dry = future.lt(dry_day_threshold_mm)

        if longest_true_run(future_dry) <= maximum_future_dry_spell_days:
            return pd.Timestamp(rain.index[start])

    return pd.NaT


def detect_rainfall_cessation(
    rain: pd.Series,
    *,
    dry_spell_days: int = ONSET_FUTURE_DRY_SPELL_DAYS,
    dry_day_threshold_mm: float = DRY_DAY_MM,
    onset: pd.Timestamp | pd.NaT | None = None,
) -> pd.Timestamp | pd.NaT:
    """First day of the final dry spell of ``dry_spell_days`` in the window.

    Returns the start of the final run of at least ``dry_spell_days`` dry
    days occurring at or after ``onset``.  If no such spell exists, falls
    back to the last wet day.
    """
    dry = rain.lt(dry_day_threshold_mm)

    run_starts = np.flatnonzero(
        dry.to_numpy() & ~dry.shift(1, fill_value=False).to_numpy()
    )
    candidate = None

    for position in run_starts:
        run_length = 0
        for value in dry.iloc[position:].to_numpy():
            if not value:
                break
            run_length += 1

        if run_length >= dry_spell_days:
            start_date = pd.Timestamp(dry.index[position])
            if onset is None or pd.isna(onset) or start_date >= onset:
                candidate = start_date

    if candidate is not None:
        return pd.Timestamp(candidate)

    wet = rain[rain.ge(WET_DAY_MM)]
    if wet.empty:
        return pd.NaT
    return pd.Timestamp(wet.index[-1])


def summarize_rainfall_period(
    period: pd.DataFrame,
) -> dict[str, float | int | pd.Timestamp]:
    rain = (
        period.set_index("date")["rain_mm"]
        .sort_index()
        .astype(float)
        .fillna(0.0)
    )

    wet_days = rain[rain.ge(WET_DAY_MM)]

    wet_day_cv = (
        float(wet_days.std(ddof=1) / wet_days.mean())
        if len(wet_days) >= 2 and wet_days.mean() > 0
        else np.nan
    )

    onset = detect_rainfall_onset(rain)
    cessation = detect_rainfall_cessation(rain, onset=onset)

    onset_doy = (
        int(onset.strftime("%j")) if onset is not pd.NaT else np.nan
    )
    cessation_doy = (
        int(cessation.strftime("%j"))
        if cessation is not pd.NaT
        else np.nan
    )

    season_length = (
        int((cessation - onset).days + 1)
        if onset is not pd.NaT and cessation is not pd.NaT
        else np.nan
    )

    return {
        "rain_total_mm": float(rain.sum()),
        "rain_mean_daily_mm": float(rain.mean()),
        "rainy_days_1mm": int(rain.ge(1.0).sum()),
        "rainy_days_10mm": int(rain.ge(10.0).sum()),
        "heavy_rain_days_20mm": int(rain.ge(20.0).sum()),
        "maximum_1day_rainfall_mm": float(rain.max()),
        "maximum_5day_rainfall_mm": float(
            rain.rolling(5, min_periods=1).sum().max()
        ),
        "longest_dry_spell_days": longest_true_run(rain.lt(1.0)),
        "wet_day_rainfall_cv": wet_day_cv,
        "rainfall_onset_date": onset,
        "rainfall_cessation_date": cessation,
        "onset_day_of_year": onset_doy,
        "cessation_day_of_year": cessation_doy,
        "season_length_days": season_length,
    }


def build_seasonal_rainfall_features(
    daily_rainfall: pd.DataFrame,
    season_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Build one rainfall-feature row per (spatial_unit, year, season).

    ``daily_rainfall`` needs ``spatial_unit``, ``date``, ``rain_mm``.
    ``season_calendar`` needs ``spatial_unit``, ``year``, ``season``,
    ``start_date``, ``end_date``.
    """
    calendar = season_calendar.copy()
    calendar["start_date"] = pd.to_datetime(calendar["start_date"])
    calendar["end_date"] = pd.to_datetime(calendar["end_date"])
    calendar["year"] = pd.to_numeric(calendar["year"], errors="coerce")

    daily = daily_rainfall.copy()
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

        period = daily.loc[mask, ["date", "rain_mm"]]

        if period.empty:
            raise ValueError(
                "No rainfall records for "
                f"{row.spatial_unit}, {row.year}, {row.season}."
            )

        features = summarize_rainfall_period(period)

        rows.append(
            {
                "spatial_unit": row.spatial_unit,
                "year": int(row.year),
                "season": row.season,
                **features,
                "rainfall_source": RAINFALL_SOURCE,
                "predictor_geographic_level": PREDICTOR_GEOGRAPHIC_LEVEL,
            }
        )

    result = pd.DataFrame(rows)

    key = ["spatial_unit", "year", "season"]
    if result.duplicated(key).any():
        raise ValueError("Duplicate seasonal rainfall feature rows.")

    return result.sort_values(key).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    from uganda_crop_model.data.paths import INTERIM, PUBLIC

    daily = pd.read_csv(INTERIM / "subregion_daily_rainfall.csv")
    calendar = pd.read_csv(PUBLIC / "season_calendar.csv")
    features = build_seasonal_rainfall_features(daily, calendar)
    print(features.head(20).to_string())