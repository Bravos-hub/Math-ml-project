"""Outlier analysis without globally deleting observations.

Descriptive Mahalanobis-style diagnostics only.  A statistically unusual
season is not automatically an error; sensitivity analyses should compare
models with and without flagged rows (blueprint section 21).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


def robust_mahalanobis_flags(
    data: pd.DataFrame,
    feature_columns: list[str],
    *,
    significance_level: float = 0.01,
) -> pd.DataFrame:
    complete = data[feature_columns].dropna()

    if len(complete) <= len(feature_columns):
        raise ValueError(
            "Insufficient observations for multivariate outlier analysis."
        )

    scaled = StandardScaler().fit_transform(complete)

    robust_covariance = MinCovDet(random_state=42).fit(scaled)

    squared_distance = robust_covariance.mahalanobis(scaled)

    threshold = chi2.ppf(
        1.0 - significance_level,
        df=len(feature_columns),
    )

    result = pd.DataFrame(
        {
            "mahalanobis_distance_squared": squared_distance,
            "mahalanobis_threshold": threshold,
            "is_multivariate_outlier": squared_distance > threshold,
        },
        index=complete.index,
    )

    return result