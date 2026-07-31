#!/usr/bin/env python3
"""
Build a district-year groundnuts yield table for Eastern Uganda from the
extracted AAS 2020 groundnuts sub-region table.

This assigns each district the groundnuts statistics reported for its UBOS AAS
2020 sub-region and preserves source granularity metadata.

Input:
  - aas2020_groundnuts_subregion.csv

Outputs:
  - aas2020_eastern_district_groundnuts_yield.csv
  - aas2020_eastern_district_groundnuts_yield_total_2020.csv
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("aas2020_groundnuts_subregion.csv")
OUTPUT_FILE = Path("aas2020_eastern_district_groundnuts_yield.csv")
TOTAL_OUTPUT_FILE = Path("aas2020_eastern_district_groundnuts_yield_total_2020.csv")

DISTRICT_TO_SUBREGION = {
    "Mbale": "Elgon",
    "Kapchorwa": "Elgon",
    "Iganga": "Busoga",
    "Jinja": "Busoga",
    "Tororo": "Bukedi",
}


def main():
    print("=" * 70)
    print("  BUILD EASTERN UGANDA 2020 GROUNDNUTS YIELD TABLE FROM AAS 2020")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    subregion = pd.read_csv(INPUT_FILE)
    subregion = subregion[subregion["sub_region"].isin(set(DISTRICT_TO_SUBREGION.values()))].copy()
    subregion["year"] = 2020

    records = []
    for district, sub_region in DISTRICT_TO_SUBREGION.items():
        subset = subregion[subregion["sub_region"] == sub_region]
        for _, row in subset.iterrows():
            records.append(
                {
                    "district": district,
                    "year": int(row["year"]),
                    "season_group": row["season_group"],
                    "crop": row["crop"],
                    "table_id": row["table_id"],
                    "source_granularity": "sub_region_assigned_to_district",
                    "sub_region": sub_region,
                    "yield_tons_ha": row["yield_mt_per_ha_production_over_harvested"],
                    "yield_tons_ha_planted": row["yield_mt_per_ha_production_over_planted"],
                    "area_planted_ha": row["area_planted_ha"],
                    "area_harvested_ha": row["area_harvested_ha"],
                    "production_mt": row["production_mt"],
                    "cv_area_planted_pct": row["cv_area_planted_pct"],
                    "cv_area_harvested_pct": row["cv_area_harvested_pct"],
                    "cv_production_pct": row["cv_production_pct"],
                }
            )

    district_yield = pd.DataFrame(records).sort_values(["district", "season_group"]).reset_index(drop=True)
    district_yield.to_csv(OUTPUT_FILE, index=False)

    total_2020 = district_yield[district_yield["season_group"] == "total_2020"].copy()
    total_2020.to_csv(TOTAL_OUTPUT_FILE, index=False)

    print(f"[✓] Saved: {OUTPUT_FILE}")
    print(f"[✓] Rows: {len(district_yield)}")
    print(district_yield.to_string(index=False))
    print()
    print(f"[✓] Saved: {TOTAL_OUTPUT_FILE}")
    print(f"[✓] Rows: {len(total_2020)}")
    print(total_2020.to_string(index=False))


if __name__ == "__main__":
    main()
