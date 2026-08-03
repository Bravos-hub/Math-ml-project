"""Soil data coverage and quality gate.

Static soil properties may repeat across years, but this repetition must be
documented.  They explain persistent spatial differences, not year-to-year
soil change.  Do not include a soil column in the primary PCA while most
records are missing or from failed requests.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_soil_coverage(
    soil: pd.DataFrame,
    soil_columns: Sequence[str],
    *,
    minimum_coverage: float = 0.80,
) -> None:
    missing_columns = set(soil_columns).difference(soil.columns)

    if missing_columns:
        raise ValueError(
            f"Missing soil columns: {sorted(missing_columns)}"
        )

    coverage = soil[list(soil_columns)].notna().mean()
    failed = coverage[coverage < minimum_coverage]

    if not failed.empty:
        raise ValueError(
            "Soil data coverage is insufficient:\n"
            f"{failed.sort_values().to_string()}"
        )

    texture_columns = {"clay_pct", "sand_pct", "silt_pct"}

    if texture_columns.issubset(soil.columns):
        total = (
            soil["clay_pct"]
            + soil["sand_pct"]
            + soil["silt_pct"]
        )

        invalid = total.notna() & ~total.between(95, 105)

        if invalid.any():
            raise ValueError(
                "Soil texture percentages do not sum to approximately 100."
            )