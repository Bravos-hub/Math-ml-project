"""Independent mathematical verification of the PCA eigen-decomposition.

Provides a hand-computed eigenvalue solution for symmetric 2x2 covariance
matrices and a check that the sklearn result reproduces it.  This is the
testable core of the project's PCA mathematics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def manual_eigenvalues_2x2(
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (2, 2):
        raise ValueError("A 2 x 2 covariance matrix is required.")

    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("The covariance matrix is not symmetric.")

    a = covariance[0, 0]
    c = covariance[0, 1]
    b = covariance[1, 1]

    discriminant = np.sqrt((a - b) ** 2 + 4.0 * c**2)

    lambda_small = ((a + b) - discriminant) / 2.0
    lambda_large = ((a + b) + discriminant) / 2.0

    return np.asarray(
        [lambda_small, lambda_large],
        dtype=float,
    )


def verify_two_feature_pca(
    data: pd.DataFrame,
    feature_one: str,
    feature_two: str,
) -> dict[str, np.ndarray]:
    subset = (
        data[[feature_one, feature_two]]
        .dropna()
        .astype(float)
    )

    if len(subset) < 3:
        raise ValueError(
            "At least three complete observations are required."
        )

    values = subset.to_numpy()
    centered = values - values.mean(axis=0)

    covariance = (centered.T @ centered) / (len(centered) - 1)

    manual = manual_eigenvalues_2x2(covariance)
    eigh_values, eigh_vectors = np.linalg.eigh(covariance)

    if not np.allclose(
        manual,
        eigh_values,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise AssertionError(
            "Manual eigenvalues do not match numpy.linalg.eigh."
        )

    if np.any(eigh_values < -1e-10):
        raise AssertionError(
            "Covariance matrix has a materially negative eigenvalue."
        )

    if not np.allclose(
        covariance @ eigh_vectors,
        eigh_vectors @ np.diag(eigh_values),
        atol=1e-10,
    ):
        raise AssertionError(
            "Eigenvector equation verification failed."
        )

    return {
        "covariance": covariance,
        "manual_eigenvalues": manual,
        "eigh_eigenvalues": eigh_values,
        "eigenvectors": eigh_vectors,
    }