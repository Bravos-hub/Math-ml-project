"""Leakage-safe preprocessing pipelines.

The blue-print design uses three feature representations:

* ``raw``: all eligible continuous variables directly.
* ``pca``: PCA over all continuous variables, with categoricals outside PCA.
* ``hybrid``: PCA of dynamic climate variables plus original static soil and
  terrain variables plus categories.  This does not duplicate raw variables
  and their linear combinations.

Every transformation (imputation, scaling, one-hot encoding, PCA) is fitted
only on the training observations of each fold.  Imputation is median for
continuous variables and most-frequent for categorical variables
(blueprint section 15).

The class catalog is resolved from the actual columns present in the
final dataset, so a dataset without soil columns (as in the current
milestone) still runs; static features are simply omitted.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FeatureSpace = Literal["raw", "pca", "hybrid"]

CLIMATE_FEATURES = [
    "rain_total_mm",
    "rainy_days_1mm",
    "heavy_rain_days_20mm",
    "maximum_5day_rainfall_mm",
    "longest_dry_spell_days",
    "wet_day_rainfall_cv",
    "temperature_mean_c",
    "temperature_maximum_c",
    "growing_degree_days",
    "heat_days_32c",
]

STATIC_FEATURES = [
    "soil_ph",
    "soil_organic_carbon",
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "bulk_density",
    "cation_exchange_capacity",
    "elevation_m",
    "soil_ph_sd",
    "soil_organic_carbon_sd",
    "clay_pct_sd",
    "sand_pct_sd",
    "silt_pct_sd",
    "bulk_density_sd",
    "cation_exchange_capacity_sd",
    "elevation_m_sd",
]

CATEGORICAL_FEATURES = [
    "season",
    "crop",
]


def resolve_feature_columns(
    data: pd.DataFrame,
) -> dict[str, list[str]]:
    """Intersect the catalog with the columns actually present.

    Required climate features that are entirely absent raise an error so that
    an incorrectly built dataset cannot silently lose its predictors.
    """

    climate = [c for c in CLIMATE_FEATURES if c in data.columns]
    static = [c for c in STATIC_FEATURES if c in data.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in data.columns]

    if not climate:
        raise ValueError(
            "No climate feature columns are present in the dataset."
        )

    return {
        "climate": climate,
        "static": static,
        "categorical": categorical,
    }


def numeric_pipeline(
    *,
    pca_variance: float | None = None,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if pca_variance is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                ),
            )
        )

    return Pipeline(steps)


def categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(
    feature_space: FeatureSpace,
    *,
    feature_columns: dict[str, list[str]],
    pca_variance: float = 0.90,
) -> ColumnTransformer:
    """Build the preprocessing object for one feature-representation space.

    Only the columns actually present (``feature_columns``) are used; the
    imputation/scaling/PCA steps never see the test observations until the
    full pipeline is cross-validated.
    """

    climate = feature_columns["climate"]
    static = feature_columns["static"]
    categorical = feature_columns["categorical"]

    transformers: list[tuple[str, object, list[str]]] = []

    if feature_space == "raw":
        continuous_columns = climate + static
        if continuous_columns:
            transformers.append(
                (
                    "continuous_raw",
                    numeric_pipeline(),
                    continuous_columns,
                )
            )

    elif feature_space == "pca":
        # PCA is the climate-only comparison space. Static soil/terrain is
        # deliberately reserved for the hybrid branch so the scientific
        # question is whether static agroecology adds value beyond climate PCs.
        continuous_columns = climate
        if continuous_columns:
            if len(continuous_columns) < 2:
                raise ValueError(
                    "PCA space requires at least two continuous features."
                )
            transformers.append(
                (
                    "continuous_pca",
                    numeric_pipeline(pca_variance=pca_variance),
                    continuous_columns,
                )
            )

    elif feature_space == "hybrid":
        if climate:
            if len(climate) < 2:
                raise ValueError(
                    "Hybrid space requires at least two climate features "
                    "for its PCA branch."
                )
            transformers.append(
                (
                    "climate_pca",
                    numeric_pipeline(pca_variance=pca_variance),
                    climate,
                )
            )
        if static:
            transformers.append(
                (
                    "static_raw",
                    numeric_pipeline(),
                    static,
                )
            )

    else:
        raise ValueError(
            f"Unsupported feature space: {feature_space}"
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                categorical_pipeline(),
                categorical,
            )
        )

    if not transformers:
        raise ValueError(
            "No transformer could be built for the requested dataset."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
