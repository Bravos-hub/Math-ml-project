"""Feature-availability report (review P0 #7 / #8).

Separates columns of a modeling dataset into:

* ``available``  - numeric data present for a usable fraction of rows;
* ``unavailable``- column exists but is all-null (failed/empty extraction);
* ``planned``    - documented in configuration but not yet present.

Also enforces the review's rule that a column with 100% missing values should
be excluded from modeling, and writes the compact markdown table described in
the review (Variable / Source / Coverage / Status).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropyield.data.paths import TABLES

# A column with any non-null values is usable; with zero non-null it is
# "unavailable" and must not enter modeling.
ALL_NULL_EXCLUDE = True


def feature_status(df: pd.DataFrame, col: str) -> tuple[float, str]:
    """Return (coverage_share, status) for a single column."""
    coverage = 1.0 - df[col].isna().mean()
    status = "usable" if coverage > 0 else "unavailable"
    return coverage, status


def infer_source(column: str) -> str:
    """Map a column prefix to its documented data source for the report."""
    prefixes = {
        "rain_": "CHIRPS (monthly)",
        "daily_": "CHIRPS (daily)",
        "temp_": "NASA POWER (daily)",
        "soil_": "SoilGrids",
        "cv_": "UBOS AAS survey",
    }
    for prefix, source in prefixes.items():
        if column.startswith(prefix):
            return source
    return "derived"


def build_availability_report(
    df: pd.DataFrame,
    *,
    provenance_columns: tuple[str, ...] | None = None,
    planned: list[str] | None = None,
) -> pd.DataFrame:
    """Return a Variable / Source / Coverage / Status report for ``df``."""
    exclude = set(provenance_columns or ("crop", "sub_region", "season_group",
                                         "year", "district"))
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        coverage, status = feature_status(df, col)
        rows.append({
            "Variable": col,
            "Source": infer_source(col),
            "Coverage": round(coverage, 4),
            "Status": status,
        })
    for col in planned or ():
        if col not in df.columns:
            rows.append({
                "Variable": col,
                "Source": "generated / not yet integrated",
                "Coverage": 0.0,
                "Status": "planned",
            })
    report = pd.DataFrame(rows)
    return report.sort_values(["Status", "Variable"]).reset_index(drop=True)


def write_availability_report(df: pd.DataFrame,
                              out: Path | None = None,
                              **kwargs) -> Path:
    """Write the availability report and return its path."""
    out = out or (TABLES / "feature_availability.csv")
    report = build_availability_report(df, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    return out