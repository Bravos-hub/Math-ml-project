"""Missingness and survey-uncertainty helpers.

``missingness_report`` gives a per-column count and fraction for the
final dataset.  Survey coefficients of variation from AAS are preserved as
*metadata* only; they are not used as ordinary predictors of yield
(blueprint sections 22 and 23).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_report(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return (
        data.isna()
        .agg(["sum", "mean"])
        .T.rename(
            columns={
                "sum": "missing_count",
                "mean": "missing_fraction",
            }
        )
        .sort_values("missing_fraction", ascending=False)
    )


def build_quality_band(
    data: pd.DataFrame,
) -> pd.Series:
    """Map maximum AAS coefficient of variation to a quality band.

    The cut-offs are declared study rules rather than universal standards.
    """

    maximum_cv = data[
        [
            "cv_area_harvested_pct",
            "cv_production_pct",
        ]
    ].max(axis=1)

    return pd.cut(
        maximum_cv,
        bins=[-np.inf, 15, 30, 50, np.inf],
        labels=[
            "higher_precision",
            "moderate_precision",
            "low_precision",
            "very_low_precision",
        ],
    )