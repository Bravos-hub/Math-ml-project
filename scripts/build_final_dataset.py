#!/usr/bin/env python3
"""Build and validate the final maize sub-region season-year dataset.

Entry point for the milestone analysis dataset:

    python -m scripts.build_final_dataset

Steps:
1. Aggregate district daily rainfall/temperature to the sub-region series.
2. Generate the documented season calendar.
3. Build the AAS sub-region targets (2018 annual + 2020 first/second).
4. Compute seasonal climate features on the shared windows.
5. Merge into data/processed/final_maize_subregion_season_year.csv.
6. Run the final-analysis quality gate (fails honestly when the sample is
   too small for final mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uganda_crop_model.data.merge_dataset import save_final_maize_dataset  # noqa: E402
from uganda_crop_model.data.paths import FINAL_MAIZE_DATASET, TABLES  # noqa: E402
from uganda_crop_model.data.subregion_climate import (  # noqa: E402
    load_district_map,
    save_subregion_daily,
)
from uganda_crop_model.quality.dataset import (  # noqa: E402
    AnalysisPolicy,
    validate_final_dataset,
)

CLIMATE_FEATURES = [
    "rain_total_mm",
    "rain_mean_daily_mm",
    "rainy_days_1mm",
    "rainy_days_10mm",
    "heavy_rain_days_20mm",
    "maximum_1day_rainfall_mm",
    "maximum_5day_rainfall_mm",
    "longest_dry_spell_days",
    "wet_day_rainfall_cv",
    "onset_day_of_year",
    "cessation_day_of_year",
    "season_length_days",
    "temperature_mean_c",
    "temperature_maximum_c",
    "temperature_minimum_c",
    "temperature_range_c",
    "growing_degree_days",
    "heat_days_32c",
    "extreme_heat_days_35c",
    "cold_days_10c",
]


def main() -> int:
    print("[1/3] Aggregating district daily climate to sub-region ...")
    district_map = load_district_map()
    save_subregion_daily(district_map)

    print("[2/3] Building and validating the final dataset ...")
    df = save_final_maize_dataset()

    print(f"[3/3] Final dataset: {len(df)} rows x {df.shape[1]} cols")
    print(df.groupby(["year", "season"]).size().to_string())

    try:
        validate_final_dataset(df, CLIMATE_FEATURES, AnalysisPolicy())
    except ValueError as exc:
        print("\nfinal_mode gate RAISES (honest failure):")
        print(f"  {exc}")
        print("  -> The current AAS sample does not meet the blueprint's")
        print("     minimum-row/year targets; it is kept as the authoritative")
        print("     dataset but cannot yet support a final model comparison.")

    TABLES.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["year", "season"]).agg(
        spatial_units=("spatial_unit", "nunique"),
        rows=("crop", "size"),
        yield_mean_t_ha=("yield_tons_ha", "mean"),
    ).reset_index()
    summary.to_csv(TABLES / "final_maize_dataset_summary.csv", index=False)
    print(f"Summary table: {TABLES / 'final_maize_dataset_summary.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())