"""Consolidated loading of UBOS AAS 2020 sub-region crop statistics.

The AAS 2020 Chapter 6 annex provides, for each of 16 crops, area planted
(ha), area harvested (ha), production (MT), yield over harvested area, and
yield over planted area, at the 14 sub-region level (plus a national row),
for the first season 2020, second season 2020, and total 2020.

These are official, survey-weighted point estimates with published CVs, so
they receive quality grade A at sub-region level.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .provenance import (
    YIELD_AAS2020_SUBREGION,
    GRANULARITY_SUBREGION,
    add_provenance,
)
from .paths import INTERIM

SUBREGION_FILES = sorted(INTERIM.glob("aas2020_*_subregion.csv"))
COMBINED_FILE = INTERIM / "aas2020_additional_crops_subregion.csv"

RENAME = {
    "yield_mt_per_ha_production_over_harvested": "yield_over_harvested",
    "yield_mt_per_ha_production_over_planted": "yield_over_planted",
}

REQUIRED = [
    "crop",
    "sub_region",
    "season_group",
    "area_planted_ha",
    "area_harvested_ha",
    "production_mt",
    "yield_over_harvested",
    "yield_over_planted",
    "cv_area_planted_pct",
    "cv_area_harvested_pct",
    "cv_production_pct",
    "year",
]


def load_aas2020_subregion() -> pd.DataFrame:
    """Load all per-crop AAS 2020 sub-region tables into one clean table."""
    frames = []
    for path in SUBREGION_FILES:
        if path == COMBINED_FILE:
            continue
        df = pd.read_csv(path)
        crop = df["crop"].dropna().unique()
        if len(crop) != 1:
            raise ValueError(f"Unexpected multiple crops in {path.name}: {crop}")
        df = df.rename(columns=RENAME)
        df["year"] = 2020
        df = df[df["entity_type"] == "sub_region"].copy()
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop(columns=[c for c in panel.columns if c == "table_id"])

    # The official tables also carry a national ("Uganda") row; exclude it
    # from the sub-region panel (documented, not dropped silently elsewhere).
    panel = panel[panel["sub_region"] != "Uganda"].copy()

    for col in REQUIRED:
        if col not in panel.columns:
            panel[col] = pd.NA

    panel["harvest_loss_ratio"] = (
        1.0 - panel["area_harvested_ha"] / panel["area_planted_ha"]
    )

    panel = panel[REQUIRED + ["harvest_loss_ratio"]].sort_values(
        ["crop", "sub_region", "season_group"]
    ).reset_index(drop=True)

    panel = add_provenance(
        panel,
        yield_source=YIELD_AAS2020_SUBREGION,
        yield_granularity=GRANULARITY_SUBREGION,
        quality_note="Official AAS 2020 Chapter 6 sub-region estimates, survey-weighted with published CVs.",
    )

    # Recover OCR junk in the source yields (e.g. a stray backtick in
    # Table 6-19 groundnuts, South Buganda, second season 2020): recompute
    # from production/area and flag the cell (grade C) AFTER provenance, so
    # the flags are not overwritten.
    for col in ("yield_over_harvested", "yield_over_planted"):
        junk = pd.to_numeric(panel[col], errors="coerce").isna() \
            & panel[col].notna()
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        if col == "yield_over_harvested" and junk.any():
            panel.loc[junk, col] = (
                panel["production_mt"] / panel["area_harvested_ha"]
            ).loc[junk]
        if col == "yield_over_planted" and junk.any():
            panel.loc[junk, col] = (
                panel["production_mt"] / panel["area_planted_ha"]
            ).loc[junk]
        panel.loc[junk, "is_imputed"] = True
        panel.loc[junk, "data_quality_score"] = "C"
        panel.loc[junk, "data_quality_note"] = (
            "Yield not legible in source table; recomputed as "
            "production / area and flagged as derived."
        )

    return panel


def save_aas2020_subregion(output: Path | None = None) -> pd.DataFrame:
    output = output or INTERIM / "aas2020_subregion_consolidated.csv"
    panel = load_aas2020_subregion()
    panel.to_csv(output, index=False)
    return panel


if __name__ == "__main__":
    panel = save_aas2020_subregion()
    print(f"[✓] AAS 2020 consolidated: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(panel.groupby("crop").size().to_string())
