"""PCA stability testing via bootstrap resampling and sign alignment.

A component should not receive a strong agricultural interpretation if its
loadings are unstable across resamples.  Loadings here are the eigenvector
coefficients (components), aligned to a reference fit by absolute absolute
matching and sign (blueprint section 24).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def align_components(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Permute and sign-align candidate components to the reference."""

    similarity = np.abs(reference @ candidate.T)

    reference_indices, candidate_indices = linear_sum_assignment(
        -similarity
    )

    aligned = np.zeros_like(reference)

    for reference_index, candidate_index in zip(
        reference_indices,
        candidate_indices,
    ):
        component = candidate[candidate_index].copy()

        if (
            np.dot(reference[reference_index], component) < 0
        ):
            component *= -1

        aligned[reference_index] = component

    return aligned


def bootstrap_pca_loadings(
    X: np.ndarray,
    *,
    n_components: int,
    iterations: int = 500,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Bootstrap the aligned loadings of the first n_components."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reference_pca = PCA(
        n_components=n_components,
        svd_solver="full",
    ).fit(X_scaled)

    reference = reference_pca.components_

    rng = np.random.default_rng(random_seed)

    bootstrapped = np.empty(
        (
            iterations,
            n_components,
            X.shape[1],
        ),
        dtype=float,
    )

    for iteration in range(iterations):
        indices = rng.integers(
            0,
            len(X_scaled),
            size=len(X_scaled),
        )

        sample = X_scaled[indices]

        candidate = (
            PCA(
                n_components=n_components,
                svd_solver="full",
            )
            .fit(sample)
            .components_
        )

        bootstrapped[iteration] = align_components(
            reference,
            candidate,
        )

    return {
        "reference_components": reference,
        "mean_components": bootstrapped.mean(axis=0),
        "lower_95": np.percentile(bootstrapped, 2.5, axis=0),
        "upper_95": np.percentile(bootstrapped, 97.5, axis=0),
    }