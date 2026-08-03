"""Build AAS subregion-season-year crop targets at the correct analytical grain.

The official AAS yield statistics are reported at the 14-sub-region level.
The final-analysis target therefore models the grain::

    sub_region x year x season x crop

AAS 2020 provides an explicit first-season and second-season split per
sub-region; the published ``total_2020`` rows are *derived* from those two
seasonal blocks and are excluded to avoid double-counting.  AAS 2018 is only
published as an annual sub-region table (its ``area_harvested_ha`` reflects
the second-season block, production reflects the total), so 2018 rows are
tagged ``season = "annual"`` and marked ``yield_consistency_ok = False``.

Yield (t/ha) is recomputed as ``production_mt / area_harvested_ha`` exactly
as published, so that the target and its audit columns are internally
consistent for every row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED2020 = {
    "crop",
    "sub_region",
    "season_group",
    "area_planted_ha",
    "area_harvested_ha",
    "production_mt",
    "cv_area_planted_pct",
    "cv_area_harvested_pct",
    "cv_production_pct",
}

REQUIRED2018 = {
    "crop",
    "sub_region",
    "area_planted_ha",
    "area_harvested_ha",
    "production_mt",
    "cv_production_pct",
}

SEASON_NORMALISATION = {
    "first_season_2020": "first_season",
    "second_season_2020": "second_season",
}

TARGET_SOURCE = {
    2020: "AAS2020_Table_6_1",
    2018: "AAS2018_Annex4",
}


def _normalise_numbers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _valid_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid = (
        df["production_mt"].notna()
        & df["area_harvested_ha"].notna()
        & df["area_harvested_ha"].gt(0)
    )
    return df[valid].copy()


def _build_provenance(
    df: pd.DataFrame,
    *,
    year: int,
    spatial_level: str,
    target_definition: str,
) -> pd.DataFrame:
    df["spatial_unit"] = df["sub_region"]
    df["spatial_level"] = spatial_level
    df["target_source"] = TARGET_SOURCE[year]
    df["target_source_type"] = "official_aggregate"
    df["target_geographic_level"] = "sub_region"
    df["predictor_geographic_level"] = "sub_region"
    df["target_definition"] = target_definition
    df["target_year"] = year
    df["target_season"] = df["season"]
    df["is_proxy"] = False
    df["is_synthetic"] = False
    df["is_geographically_assigned"] = False
    return df


def _audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["yield_tons_ha"] = (
        df["production_mt"] / df["area_harvested_ha"]
    )
    df["yield_tons_ha_planted"] = np.where(
        df["area_planted_ha"].gt(0),
        df["production_mt"] / df["area_planted_ha"],
        np.nan,
    )
    df["harvested_fraction"] = np.where(
        df["area_planted_ha"].gt(0),
        df["area_harvested_ha"] / df["area_planted_ha"],
        np.nan,
    )
    return df


def build_aas2020_subregion_target(
    source_file: Path,
    *,
    year: int = 2020,
    crop: str | None = "maize",
) -> pd.DataFrame:
    """Build first/second-season sub-region targets from an AAS 2020 table.

    ``total_2020`` rows are dropped (they are derived from the seasonal
    blocks).  The single "Uganda" national row is dropped as well.  Pass
    ``crop=None`` to keep every crop present in the table.
    """
    df = pd.read_csv(source_file)

    missing = REQUIRED2020.difference(df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    if crop is not None:
        df = df[df["crop"].eq(crop.lower())].copy()

    df["season_group"] = df["season_group"].astype(str)
    df = df[~df["season_group"].str.startswith("total")].copy()
    df = df[df["sub_region"] != "Uganda"].copy()

    numeric = [
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_area_planted_pct",
        "cv_area_harvested_pct",
        "cv_production_pct",
    ]
    df = _normalise_numbers(df, numeric)
    df = _valid_target_rows(df)

    if df.empty:
        raise ValueError("No valid AAS 2020 seasonal target rows remain.")

    df["year"] = year
    df["season"] = df["season_group"].map(SEASON_NORMALISATION)
    if df["season"].isna().any():
        unhandled = sorted(df.loc[df["season"].isna(), "season_group"].unique())
        raise ValueError(f"Unhandled season groups: {unhandled}")

    df = _audit_columns(df)
    df = _build_provenance(
        df,
        year=year,
        spatial_level="sub_region",
        target_definition="production_mt_divided_by_area_harvested_ha",
    )
    df["season_definition"] = "surveyed_seasonal_block"
    df["yield_consistency_ok"] = True

    key = ["spatial_unit", "year", "season", "crop"]
    if df.duplicated(key).any():
        raise ValueError("AAS 2020 target contains duplicate analytical keys.")

    output_columns = [
        *key,
        "yield_tons_ha",
        "yield_tons_ha_planted",
        "harvested_fraction",
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_area_planted_pct",
        "cv_area_harvested_pct",
        "cv_production_pct",
        *[
            "target_source",
            "target_source_type",
            "target_geographic_level",
            "predictor_geographic_level",
            "target_definition",
            "target_year",
            "target_season",
            "is_proxy",
            "is_synthetic",
            "is_geographically_assigned",
        ],
        "season_definition",
        "yield_consistency_ok",
    ]
    return df[output_columns].sort_values(key).reset_index(drop=True)


def build_aas2018_subregion_target(
    source_file: Path,
    *,
    year: int = 2018,
    crop: str | None = "maize",
) -> pd.DataFrame:
    """Build annual sub-region targets from the AAS 2018 consolidated table.

    AAS 2018 does not release separate first/second targets; the consolidated
    table merges total production with the second's harvested area.  These
    rows are kept at ``season`` and flagged ``yield_consistency_ok = False``.
    Pass ``crop=None`` to keep every crop present in the table.
    """
    df = pd.read_csv(source_file)

    missing = REQUIRED2018.difference(df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    if crop is not None:
        df = df[df["crop"].eq(crop.lower())].copy()

    numeric = [
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_production_pct",
    ]
    df = _normalise_numbers(df, numeric)
    df = _valid_target_rows(df)

    for column in ("cv_area_planted_pct", "cv_area_harvested_pct"):
        if column not in df.columns:
            df[column] = np.nan

    if len(df) == 0:
        raise ValueError("No valid AAS 2018 target rows remain.")

    df["harvested_fraction"] = np.where(
        df["area_planted_ha"].gt(0),
        df["area_harvested_ha"] / df["area_planted_ha"],
        np.nan,
    )
    # production is the total block's value while area harvested comes from
    # the second's block; the published sub-region yields are kept untouched.
    df["yield_over_harvested_published"] = pd.to_numeric(
        df.get("yield_over_harvested", pd.Series(index=df.index)),
        errors="coerce",
    )
    df["yield_tons_ha"] = df["yield_over_harvested_published"]
    df["yield_tons_ha_planted"] = np.where(
        df["area_planted_ha"].gt(0),
        df["production_mt"] / df["area_planted_ha"],
        np.nan,
    )
    df = df.dropna(subset=["yield_tons_ha"])

    df["year"] = year
    df["season"] = "annual"
    df["season_definition"] = "published_annual_aggregate_total_and_s2_harvested_area"
    df["yield_consistency_ok"] = False

    df = _build_provenance(
        df,
        year=year,
        spatial_level="sub_region",
        target_definition="published_official_subregion_yield_over_harvested",
    )

    key = ["spatial_unit", "year", "season", "crop"]
    if df.duplicated(key).any():
        raise ValueError("AAS 2018 target contains duplicate analytical keys.")

    output_columns = [
        *key,
        "yield_tons_ha",
        "yield_tons_ha_planted",
        "harvested_fraction",
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_area_planted_pct",
        "cv_area_harvested_pct",
        "cv_production_pct",
        "target_source",
        "target_source_type",
        "target_geographic_level",
        "predictor_geographic_level",
        "target_definition",
        "target_year",
        "target_season",
        "is_proxy",
        "is_synthetic",
        "is_geographically_assigned",
        "season_definition",
        "yield_consistency_ok",
    ]
    return df[output_columns].sort_values(key).reset_index(drop=True)


def build_combined_maize_targets(
    aas2020_file: Path,
    aas2018_file: Path,
) -> pd.DataFrame:
    """Combine AAS 2020 seasonal and AAS 2018 annual maize targets."""
    targets = pd.concat(
        [
            build_aas2020_subregion_target(aas2020_file, year=2020),
            build_aas2018_subregion_target(aas2018_file, year=2018),
        ],
        ignore_index=True,
    )
    key = ["spatial_unit", "year", "season", "crop"]
    if targets.duplicated(key).any():
        raise ValueError("Combined targets contain duplicate analytical keys.")
    return targets.sort_values(key).reset_index(drop=True)


def build_aas2020_multi_crop_target(
    crop_tables: dict[str, Path],
    *,
    year: int = 2020,
) -> pd.DataFrame:
    """Combine first/second-season targets across the eligible food crops.

    Only crops that report both production and harvested area are eligible
    (``REQUIRED2020``), so every row has the same
    ``production / harvested area`` target definition.
    """
    frames = []
    for crop, path in crop_tables.items():
        frame = build_aas2020_subregion_target(
            path,
            year=year,
            crop=crop,
        )
        frames.append(frame)

    targets = pd.concat(frames, ignore_index=True)

    key = ["spatial_unit", "year", "season", "crop"]
    if targets.duplicated(key).any():
        raise ValueError("Multi-crop targets contain duplicate analytical keys.")
    return targets.sort_values(key).reset_index(drop=True)


def build_aas2018_multi_crop_target(
    source_file: Path,
    crops: list[str],
    *,
    year: int = 2018,
) -> pd.DataFrame:
    """Build annual targets for the same food-crop set from AAS 2018."""
    all_rows = build_aas2018_subregion_target(
        source_file,
        year=year,
        crop=None,
    )
    allowed = {crop.lower() for crop in crops}
    targets = all_rows[all_rows["crop"].isin(allowed)].copy()

    if targets.empty:
        raise ValueError("No AAS 2018 rows remain for the requested crops.")

    key = ["spatial_unit", "year", "season", "crop"]
    if targets.duplicated(key).any():
        raise ValueError("AAS 2018 multi-crop targets contain duplicate keys.")
    return targets.sort_values(key).reset_index(drop=True)


def build_combined_multi_crop_targets(
    aas2020_crop_tables: dict[str, Path],
    aas2018_file: Path,
) -> pd.DataFrame:
    """Combine AAS 2020 seasonal and AAS 2018 annual multi-crop targets."""
    crops = list(aas2020_crop_tables)
    targets = pd.concat(
        [
            build_aas2020_multi_crop_target(aas2020_crop_tables),
            build_aas2018_multi_crop_target(aas2018_file, crops),
        ],
        ignore_index=True,
    )
    key = ["spatial_unit", "year", "season", "crop"]
    if targets.duplicated(key).any():
        raise ValueError("Combined multi-crop targets contain duplicate keys.")
    return targets.sort_values(key).reset_index(drop=True)


if __name__ == "__main__":
    from uganda_crop_model.data.paths import AAS2018_MAIZE, AAS2020_MAIZE

    result = build_combined_maize_targets(AAS2020_MAIZE, AAS2018_MAIZE)
    print(result.groupby(["year", "season"]).size().to_string())
    print(result.to_string(index=False))
