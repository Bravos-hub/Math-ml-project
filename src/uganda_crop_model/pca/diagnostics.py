"""Descriptive PCA diagnostics: parallel analysis and loading tables.

These are for the descriptive PCA report, not for leakage-free predictive
component counts.  For predictive modeling, component selection must happen
separately inside every training fold (blueprint sections 12 and 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array


def parallel_analysis(
    standardized_features: np.ndarray,
    *,
    iterations: int = 500,
    percentile: float = 95.0,
    random_seed: int = 42,
) -> dict[str, np.ndarray | int]:
    """Compare observed eigenvalues against a permuted-column null."""

    X = check_array(
        standardized_features,
        ensure_2d=True,
        dtype=float,
    )

    observed = PCA(svd_solver="full").fit(X).explained_variance_

    rng = np.random.default_rng(random_seed)

    null_eigenvalues = np.empty(
        (iterations, len(observed)),
        dtype=float,
    )

    for iteration in range(iterations):
        randomized = np.column_stack(
            [
                rng.permutation(X[:, column])
                for column in range(X.shape[1])
            ]
        )

        null_eigenvalues[iteration] = (
            PCA(svd_solver="full").fit(randomized).explained_variance_
        )

    threshold = np.percentile(
        null_eigenvalues,
        percentile,
        axis=0,
    )

    retained = int(np.sum(observed > threshold))
    retained = max(1, retained)

    return {
        "observed_eigenvalues": observed,
        "null_threshold": threshold,
        "retained_components": retained,
    }


def build_pca_loading_tables(
    pca: PCA,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation loadings and percentage contributions.

    For standardized variables the correlation loading is
    L_{jk} = v_{jk} * sqrt(lambda_k).  Contribution of variable j to
    component k is the squared loadings share expressed as a percentage.
    """

    eigenvectors = pca.components_.T

    correlation_loadings = (
        eigenvectors * np.sqrt(pca.explained_variance_)
    )

    components = [
        f"PC{i + 1}"
        for i in range(pca.n_components_)
    ]

    loading_table = pd.DataFrame(
        correlation_loadings,
        index=feature_names,
        columns=components,
    )

    squared = loading_table.pow(2)

    contribution = (
        squared.divide(squared.sum(axis=0), axis=1) * 100.0
    )

    return loading_table, contribution


def fit_standardized_pca(
    data: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_components: float | int | None = None,
    random_seed: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Standardize then fit PCA on the given columns (data-dependent).

    Used by the descriptive reporting path only.
    """

    subset = data[feature_columns].dropna().astype(float)

    if len(subset) == 0:
        raise ValueError("No complete observations for PCA.")

    if not isinstance(n_components, float) and n_components is not None and n_components > subset.shape[1]:
        raise ValueError(
            "Cannot retain more components than features."
        )

    scaled = StandardScaler().fit_transform(subset)

    if n_components is not None and isinstance(n_components, (float, int)):
        pca = PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=random_seed,
        )
    else:
        pca = PCA(svd_solver="full", random_state=random_seed)

    pca.fit(scaled)

    return pca, scaled