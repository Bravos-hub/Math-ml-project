"""Prediction intervals using split-conformal with a temporal calibration
split (blueprint section 20).

The most recent training year is used as a calibration period; the remaining
years are used for model training.  Intervals are not guaranteed to be
well calibrated until their empirical coverage is evaluated on held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone


def temporal_split_conformal_predict(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    years_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_year = years_train.max()

    calibration_year_value = (
        calibration_year if isinstance(calibration_year, (int, np.integer))
        else int(calibration_year)
    )

    proper_train_mask = years_train.astype(float).lt(
        float(calibration_year_value)
    )
    calibration_mask = years_train.astype(float).eq(
        float(calibration_year_value)
    )

    if proper_train_mask.sum() == 0:
        raise ValueError(
            "No observations remain for proper model training."
        )

    if calibration_mask.sum() < 5:
        raise ValueError("Calibration sample is too small.")

    fitted = clone(estimator)

    fitted.fit(
        X_train.loc[proper_train_mask],
        y_train.loc[proper_train_mask],
    )

    calibration_prediction = fitted.predict(
        X_train.loc[calibration_mask]
    )

    calibration_residuals = np.abs(
        y_train.loc[calibration_mask].to_numpy()
        - calibration_prediction
    )

    n = len(calibration_residuals)

    quantile_level = min(
        1.0,
        np.ceil((n + 1) * (1 - alpha)) / n,
    )

    radius = np.quantile(
        calibration_residuals,
        quantile_level,
        method="higher",
    )

    prediction = fitted.predict(X_test)

    lower = prediction - radius
    upper = prediction + radius

    return prediction, lower, upper