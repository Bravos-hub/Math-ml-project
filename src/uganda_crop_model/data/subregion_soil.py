"""Aggregate validated district SoilGrids/elevation features to subregions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOIL_SOURCE = "SoilGrids_v2.0"
SOIL_COLUMNS = {
    "clay": "clay_pct",
    "sand": "sand_pct",
    "silt": "silt_pct",
    "soc": "soil_organic_carbon",
    "bdod": "bulk_density",
    "cec": "cation_exchange_capacity",
    "phh2o": "soil_ph",
}


def validate_district_soil(soil: pd.DataFrame, district_map: pd.DataFrame) -> None:
    required = {"district", *SOIL_COLUMNS}
    missing = required.difference(soil.columns)
    if missing:
        raise ValueError(f"Soil table missing columns: {sorted(missing)}")
    if soil["district"].duplicated().any():
        raise ValueError("Soil table contains duplicate districts.")
    if len(soil) != 114:
        raise ValueError(f"Expected 114 district soil rows, found {len(soil)}.")
    if not set(soil["district"]).issubset(set(district_map["district"])):
        raise ValueError("Soil districts are not all present in the district map.")
    values = soil[list(SOIL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Soil features contain missing or non-finite values.")
    texture_sum = values[["clay", "sand", "silt"]].sum(axis=1)
    if not texture_sum.between(99.0, 101.0).all():
        raise ValueError("Soil texture percentages do not sum to approximately 100.")
    if soil.get("is_proxy", pd.Series(False, index=soil.index)).astype(bool).any():
        raise ValueError("Proxy soil rows are not permitted in the final dataset.")


def build_subregion_soil_features(
    soil_file: Path,
    district_map_file: Path,
) -> pd.DataFrame:
    """Return one validated soil feature row per subregion."""
    soil = pd.read_csv(soil_file)
    district_map = pd.read_csv(district_map_file)[["district", "sub_region"]].drop_duplicates()
    validate_district_soil(soil, district_map)
    merged = soil.merge(district_map, on="district", how="inner", validate="one_to_one")
    rows = []
    for subregion, group in merged.groupby("sub_region", sort=True):
        row: dict[str, object] = {"spatial_unit": subregion}
        for source, target in SOIL_COLUMNS.items():
            values = pd.to_numeric(group[source], errors="raise")
            row[target] = float(values.mean())
            row[f"{target}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row.update({
            "soil_source": SOIL_SOURCE,
            "soil_source_version": "v2.0",
            "soil_depth_cm": 30,
            "soil_quality_flag": "validated_district_centroid_aggregate",
            "soil_district_count": int(len(group)),
            "predictor_geographic_level": "sub_region",
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if result["spatial_unit"].duplicated().any() or len(result) != 14:
        raise ValueError("Expected exactly one soil row for each of 14 subregions.")
    return result.sort_values("spatial_unit").reset_index(drop=True)


def build_optional_elevation_features(
    elevation_file: Path | None,
    district_map_file: Path,
) -> pd.DataFrame:
    """Aggregate an optional validated district elevation source.

    Missing files produce an empty frame; elevation is never synthesized.
    """
    columns = ["spatial_unit", "elevation_m", "elevation_m_sd", "elevation_source"]
    if elevation_file is None or not elevation_file.exists():
        return pd.DataFrame(columns=columns)
    elevation = pd.read_csv(elevation_file)
    required = {"district", "elevation_m"}
    if not required.issubset(elevation.columns):
        raise ValueError(f"Elevation table missing columns: {sorted(required - set(elevation.columns))}")
    mapping = pd.read_csv(district_map_file)[["district", "sub_region"]]
    merged = elevation.merge(mapping, on="district", how="inner", validate="one_to_one")
    merged["elevation_m"] = pd.to_numeric(merged["elevation_m"], errors="raise")
    return merged.groupby("sub_region").agg(
        elevation_m=("elevation_m", "mean"),
        elevation_m_sd=("elevation_m", "std"),
    ).rename_axis("spatial_unit").reset_index().assign(
        elevation_source="validated_external_source"
    )
