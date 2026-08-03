"""Final-analysis dataset quality gate.

Any dataset that is used for a "final" analysis must pass :func:
`validate_final_dataset`.  In final mode the pipeline raises rather than
falling back to proxy, synthetic, or geographically assigned targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

KEY_COLUMNS = [
    "spatial_unit",
    "year",
    "season",
    "crop",
]

TARGET_DERIVED_COLUMNS = {
    "yield_tons_ha",
    "yield_tons_ha_planted",
    "production_mt",
    "area_harvested_ha",
    "area_planted_ha",
    "total_production_kg",
    "total_area_ha",
}


@dataclass(frozen=True)
class AnalysisPolicy:
    minimum_rows: int = 100
    minimum_spatial_units: int = 5
    minimum_years: int = 4
    minimum_feature_coverage: float = 0.80
    allowed_target_source_types: tuple[str, ...] = (
        "official_aggregate",
        "observed_survey",
    )


def validate_final_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    policy: AnalysisPolicy = AnalysisPolicy(),
) -> None:
    """Fail fast when a dataset is not suitable for final analysis."""

    required = {
        *KEY_COLUMNS,
        "yield_tons_ha",
        "target_source_type",
        "target_geographic_level",
        "predictor_geographic_level",
        "is_proxy",
        "is_synthetic",
        "is_geographically_assigned",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df.duplicated(KEY_COLUMNS).any():
        duplicate_rows = df.loc[
            df.duplicated(KEY_COLUMNS, keep=False),
            KEY_COLUMNS,
        ]
        raise ValueError(
            "Duplicate analytical keys found:\n"
            f"{duplicate_rows.head(20).to_string(index=False)}"
        )

    if len(df) < policy.minimum_rows:
        raise ValueError(
            f"Only {len(df)} rows are available; "
            f"final mode requires at least {policy.minimum_rows}."
        )

    if df["spatial_unit"].nunique() < policy.minimum_spatial_units:
        raise ValueError("Insufficient independent spatial units.")

    if df["year"].nunique() < policy.minimum_years:
        raise ValueError("Insufficient temporal coverage.")

    target = pd.to_numeric(df["yield_tons_ha"], errors="coerce")

    if target.isna().any():
        raise ValueError("The target contains missing or non-numeric values.")

    if not np.isfinite(target).all():
        raise ValueError("The target contains infinite values.")

    if (target < 0).any():
        raise ValueError("Negative yield values are not permitted.")

    if df["is_proxy"].astype(bool).any():
        raise ValueError("Proxy targets are not permitted in final mode.")

    if df["is_synthetic"].astype(bool).any():
        raise ValueError("Synthetic targets are not permitted in final mode.")

    if df["is_geographically_assigned"].astype(bool).any():
        raise ValueError(
            "Targets assigned from a higher geography are not permitted "
            "as lower-level observations."
        )

    invalid_source_types = set(
        df["target_source_type"].dropna().unique()
    ).difference(policy.allowed_target_source_types)

    if invalid_source_types:
        raise ValueError(
            f"Invalid target source types: {sorted(invalid_source_types)}"
        )

    level_mismatch = (
        df["target_geographic_level"]
        != df["predictor_geographic_level"]
    )

    if level_mismatch.any():
        bad = df.loc[
            level_mismatch,
            [
                "spatial_unit",
                "target_geographic_level",
                "predictor_geographic_level",
            ],
        ]
        raise ValueError(
            "Target and predictor geographic levels do not match:\n"
            f"{bad.head(20).to_string(index=False)}"
        )

    leakage = set(feature_columns).intersection(TARGET_DERIVED_COLUMNS)
    if leakage:
        raise ValueError(
            "Target-derived columns were included as predictors: "
            f"{sorted(leakage)}"
        )

    unavailable = [c for c in feature_columns if c not in df.columns]
    if unavailable:
        raise ValueError(f"Requested features do not exist: {unavailable}")

    coverage = df[list(feature_columns)].notna().mean()
    low_coverage = coverage[coverage < policy.minimum_feature_coverage]

    if not low_coverage.empty:
        raise ValueError(
            "Features below the required coverage threshold:\n"
            f"{low_coverage.sort_values().to_string()}"
        )