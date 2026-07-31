"""Data provenance framework.

Every modeling table in this project must record, for each variable group,
where the values came from and how trustworthy they are.  This module
defines the canonical provenance columns, the source constants, and the
A-F quality grading used across the pipeline.

Quality grades
--------------
A : directly observed at the matching district-year (or subregion-year) level
    from an official survey estimate.
B : official value assigned from a larger geographic area (e.g. a
    subregion-level survey estimate copied to its member districts).
C : derived from official totals (e.g. production / harvested area computed
    from official aggregates).
D : proxy value (e.g. weather-station data used for a district, satellite
    proxy, reanalysis without local validation).
E : synthetic or demonstration value (generated for pipeline testing only).
F : failed or missing extraction (e.g. an API request that failed; the
    variable must NOT be used in PCA or modeling).
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Source constants
# ---------------------------------------------------------------------------

# Yield sources
YIELD_AAS2020_SUBREGION = "AAS2020_subregion"
YIELD_AAS2018_SUBREGION = "AAS2018_subregion"
YIELD_AAS2020_ZARDI = "AAS2020_zardi"
YIELD_AAS2018_ZARDI = "AAS2018_zardi"
YIELD_SYNTHETIC = "synthetic_generator"
YIELD_MISSING = "missing"

# Rainfall sources
RAIN_CHIRPS_MONTHLY = "CHIRPS_v2.0_monthly"
RAIN_CHIRPS_DAILY = "CHIRPS_v2.0_daily_climateserv"
RAIN_UBOS_STATION = "UBOS_station"
RAIN_SYNTHETIC = "synthetic_generator"
RAIN_MISSING = "missing"

# Temperature sources
TEMP_NASA_POWER = "NASA_POWER_daily"
TEMP_ERA5_LAND = "ERA5_Land"
TEMP_CHIRTS_ERA5 = "CHIRTS_ERA5"
TEMP_AGERA5 = "C3S_AgERA5"
TEMP_SYNTHETIC = "synthetic_generator"
TEMP_MISSING = "missing"

# Soil property sources
SOIL_SOILGRIDS = "SoilGrids_v2.0"
SOIL_WOSIS = "WoSIS"
SOIL_MISSING = "missing"

# Soil moisture sources
SM_C3S = "C3S_SOILMOISTURE"
SM_MISSING = "missing"

# ---------------------------------------------------------------------------
# Granularity constants
# ---------------------------------------------------------------------------

GRANULARITY_DISTRICT = "district"
GRANULARITY_SUBREGION = "subregion"
GRANULARITY_ZARDI = "zardi"
GRANULARITY_NATIONAL = "national"

# ---------------------------------------------------------------------------
# Provenance columns
# ---------------------------------------------------------------------------

PROVENANCE_COLUMNS = [
    "yield_source",
    "rainfall_source",
    "temperature_source",
    "soil_source",
    "soil_moisture_source",
    "yield_granularity",
    "is_proxy",
    "is_imputed",
    "data_quality_score",
    "data_quality_note",
]

QUALITY_GRADE_FROM_SOURCE = {
    # Official survey estimates at the matching level -> A.
    YIELD_AAS2020_SUBREGION: "A",
    YIELD_AAS2018_SUBREGION: "A",
    # Assigned from a larger area -> B when the unit of analysis is a district.
    YIELD_AAS2020_ZARDI: "B",
    YIELD_AAS2018_ZARDI: "B",
    # Observed weather data is grade A for climate features at matching level.
    RAIN_CHIRPS_MONTHLY: "A",
    RAIN_CHIRPS_DAILY: "A",
    RAIN_UBOS_STATION: "D",  # station data used as proxy for district coverage
    TEMP_NASA_POWER: "A",
    TEMP_ERA5_LAND: "A",
    TEMP_CHIRTS_ERA5: "A",
    TEMP_AGERA5: "A",
    SOIL_SOILGRIDS: "A",
    SOIL_WOSIS: "A",
    SM_C3S: "A",
    # Synthetic and missing values.
    YIELD_SYNTHETIC: "E",
    RAIN_SYNTHETIC: "E",
    TEMP_SYNTHETIC: "E",
    YIELD_MISSING: "F",
    RAIN_MISSING: "F",
    TEMP_MISSING: "F",
    SOIL_MISSING: "F",
    SM_MISSING: "F",
}


def quality_grade(source: str | None) -> str:
    """Return the data-quality grade for a source constant (A-F)."""
    if source is None or (isinstance(source, float) and pd.isna(source)):
        return "F"
    return QUALITY_GRADE_FROM_SOURCE.get(str(source), "F")


def add_provenance(
    df: pd.DataFrame,
    *,
    yield_source: str | None = None,
    rainfall_source: str | None = None,
    temperature_source: str | None = None,
    soil_source: str | None = None,
    soil_moisture_source: str | None = None,
    yield_granularity: str | None = None,
    is_proxy: bool = False,
    is_imputed: bool = False,
    quality_note: str = "",
    grade_override: str | None = None,
) -> pd.DataFrame:
    """Attach provenance columns to a dataframe.

    Per-row source columns (e.g. an existing ``yield_source`` column) take
    precedence over the constant values passed here.  The data-quality score
    is computed per row from the yield source (the target is what defines the
    overall observation grade), or overridden with ``grade_override``.
    """
    df = df.copy()

    assignments = {
        "yield_source": yield_source,
        "rainfall_source": rainfall_source,
        "temperature_source": temperature_source,
        "soil_source": soil_source,
        "soil_moisture_source": soil_moisture_source,
    }
    for col, value in assignments.items():
        if col in df.columns:
            df[col] = df[col].fillna(value) if value is not None else df[col]
        else:
            df[col] = value

    if yield_granularity is not None:
        if "yield_granularity" in df.columns:
            df["yield_granularity"] = df["yield_granularity"].fillna(yield_granularity)
        else:
            df["yield_granularity"] = yield_granularity
    elif "yield_granularity" not in df.columns:
        df["yield_granularity"] = None

    if "is_proxy" not in df.columns:
        df["is_proxy"] = is_proxy
    if "is_imputed" not in df.columns:
        df["is_imputed"] = is_imputed

    if grade_override is not None:
        grades = grade_override
    else:
        grades = df["yield_source"].map(quality_grade)
    df["data_quality_score"] = grades
    df["data_quality_note"] = quality_note
    return df


def provenance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize provenance columns for reporting and validation."""
    cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    rows = []
    for col in cols:
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"column": col, "value": str(value), "count": int(count)})
    return pd.DataFrame(rows).sort_values(["column", "count"], ascending=[True, False])
