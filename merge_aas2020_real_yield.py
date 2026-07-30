#!/usr/bin/env python3
"""
Overlay real AAS 2020 maize yield onto the Eastern Uganda modeling dataset.

This creates a transparent hybrid dataset:
  - 2020 rows use real AAS 2020 yield assigned from sub-region tables
  - later years retain the existing yield source already present in the model
    table

Outputs:
  - eastern_uganda_maize_modeling_dataset_2020_2023_hybrid_yield.csv
"""

from pathlib import Path

import pandas as pd


INPUT_DATASET = Path("eastern_uganda_maize_modeling_dataset_2020_2023.csv")
AAS_YIELD_FILE = Path("aas2020_eastern_district_yield_total_2020.csv")
OUTPUT_DATASET = Path("eastern_uganda_maize_modeling_dataset_2020_2023_hybrid_yield.csv")


def main():
    print("=" * 70)
    print("  MERGE REAL AAS 2020 YIELD INTO EASTERN UGANDA DATASET")
    print("=" * 70)

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_DATASET}")
    if not AAS_YIELD_FILE.exists():
        raise FileNotFoundError(f"Missing AAS yield file: {AAS_YIELD_FILE}")

    dataset = pd.read_csv(INPUT_DATASET)
    aas = pd.read_csv(AAS_YIELD_FILE)

    aas_keep = [
        "district",
        "year",
        "sub_region",
        "source_granularity",
        "yield_tons_ha",
        "yield_tons_ha_planted",
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_area_planted_pct",
        "cv_area_harvested_pct",
        "cv_production_pct",
    ]
    aas = aas[aas_keep].rename(
        columns={
            "yield_tons_ha": "yield_tons_ha_aas2020",
            "yield_tons_ha_planted": "yield_tons_ha_planted_aas2020",
            "area_planted_ha": "area_planted_ha_aas2020",
            "area_harvested_ha": "area_harvested_ha_aas2020",
            "production_mt": "production_mt_aas2020",
            "cv_area_planted_pct": "cv_area_planted_pct_aas2020",
            "cv_area_harvested_pct": "cv_area_harvested_pct_aas2020",
            "cv_production_pct": "cv_production_pct_aas2020",
        }
    )

    merged = dataset.merge(aas, on=["district", "year"], how="left", validate="one_to_one")
    merged["yield_tons_ha_original"] = merged["yield_tons_ha"]
    merged["yield_source"] = "existing_model_table"

    mask = merged["yield_tons_ha_aas2020"].notna()
    merged.loc[mask, "yield_tons_ha"] = merged.loc[mask, "yield_tons_ha_aas2020"]
    merged.loc[mask, "yield_source"] = "AAS2020_subregion_assigned_to_district"

    ordered_front = [
        "district",
        "year",
        "yield_tons_ha",
        "yield_source",
        "yield_tons_ha_original",
        "yield_tons_ha_aas2020",
        "sub_region",
        "source_granularity",
    ]
    remainder = [c for c in merged.columns if c not in ordered_front]
    merged = merged[ordered_front + remainder]

    merged.to_csv(OUTPUT_DATASET, index=False)
    print(f"[✓] Saved: {OUTPUT_DATASET}")
    print(f"[✓] Shape: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(merged[ordered_front].to_string(index=False))


if __name__ == "__main__":
    main()
