"""Optional remote-sensing feature validation.

Vegetation features are deliberately opt-in.  The module records whether a
feature is available before harvest so retrospective NDVI is not silently
described as an early forecast predictor.
"""

from __future__ import annotations

import pandas as pd


def validate_vegetation_features(
    features: pd.DataFrame,
    *,
    date_column: str = "observation_date",
    harvest_column: str = "harvest_date",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Qualify vegetation observations as pre-harvest or retrospective."""
    required = {"spatial_unit", "year", date_column, harvest_column}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Vegetation data missing columns: {sorted(missing)}")
    out = features.copy()
    out[date_column] = pd.to_datetime(out[date_column])
    out[harvest_column] = pd.to_datetime(out[harvest_column])
    out["vegetation_timing"] = out[date_column].le(out[harvest_column]).map(
        {True: "pre_harvest", False: "retrospective"}
    )
    out["vegetation_source"] = "remote_sensing_optional"
    out["predictor_geographic_level"] = "sub_region"
    for column in feature_columns or [c for c in out if c.startswith(("ndvi", "evi"))]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
