#!/usr/bin/env python3
"""
Train yield prediction models on the canonical Uganda modeling dataset.

Default behavior:
  - Reads eastern_uganda_maize_modeling_dataset.csv
  - Selects numeric predictors with usable coverage
  - Imputes missing values with median
  - Standardizes predictors
  - Runs PCA
  - Compares raw-feature and PCA-based models with 5-fold CV

Outputs are written with a configurable prefix.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TARGET = "yield_tons_ha"
EXCLUDE_COLUMNS = {
    "district",
    "year",
    "yield_tons_ha",
    "source_status",
    "planting_date",
}
MIN_NON_NULL_FRACTION = 0.6
MAX_PCA_COMPONENTS = 6
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="eastern_uganda_maize_modeling_dataset.csv",
        help="Input CSV dataset path.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for output files, for example overlap_2020_2023_",
    )
    return parser.parse_args()


def load_dataset(input_file):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    df = pd.read_csv(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df[df[TARGET].notna()].copy()
    if df.empty:
        raise ValueError("No rows with non-missing yield_tons_ha were found.")

    return df, input_path


def select_predictors(df):
    candidate_cols = []
    coverage = []

    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null_fraction = numeric.notna().mean()
        if non_null_fraction >= MIN_NON_NULL_FRACTION:
            candidate_cols.append(col)
            coverage.append(
                {
                    "feature": col,
                    "non_null_fraction": non_null_fraction,
                }
            )

    if not candidate_cols:
        raise ValueError("No usable numeric predictors met the coverage threshold.")

    X = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    coverage_df = pd.DataFrame(coverage).sort_values(
        ["non_null_fraction", "feature"], ascending=[False, True]
    )
    return X, candidate_cols, coverage_df


def fit_pca_diagnostics(X, feature_names):
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_ready = preprocessor.fit_transform(X)

    pca = PCA()
    pca.fit(X_ready)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            "eigenvalue": pca.explained_variance_,
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    ).reset_index(names="feature")

    return explained, loadings, X_ready.shape[1]


def build_models(n_components):
    raw_models = {
        "OLS (raw)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge (raw)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "SVR (raw)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=5.0, epsilon=0.1)),
            ]
        ),
        "Random Forest (raw)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    random_state=42,
                )),
            ]
        ),
    }

    pca_models = {
        "OLS (PCA)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge (PCA)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "SVR (PCA)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", SVR(kernel="rbf", C=5.0, epsilon=0.1)),
            ]
        ),
        "Random Forest (PCA)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    random_state=42,
                )),
            ]
        ),
    }

    return raw_models | pca_models


def evaluate_models(X, y, n_components):
    models = build_models(n_components)
    results = []

    for name, pipeline in models.items():
        predictions = cross_val_predict(pipeline, X, y, cv=CV)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        results.append(
            {
                "model": name,
                "feature_space": "PCA" if "(PCA)" in name else "raw",
                "rmse": rmse,
                "r2": r2,
                "n_components": n_components if "(PCA)" in name else X.shape[1],
            }
        )

    return pd.DataFrame(results).sort_values(["r2", "rmse"], ascending=[False, True])


def write_summary(df, feature_names, explained, results, n_components, summary_file, input_path):
    best = results.iloc[0]
    lines = [
        "Uganda Yield Modeling Training Summary",
        "=" * 40,
        f"Input dataset: {input_path}",
        f"Rows used: {len(df)}",
        f"Predictor count: {len(feature_names)}",
        f"Predictors: {', '.join(feature_names)}",
        f"PCA components evaluated: {n_components}",
        f"Variance explained by {n_components} PCs: "
        f"{explained['cumulative_explained_variance'].iloc[n_components - 1]:.4f}",
        "",
        "Best model:",
        f"  {best['model']}",
        f"  R2 = {best['r2']:.4f}",
        f"  RMSE = {best['rmse']:.4f}",
        "",
        "Top 5 model results:",
    ]

    for _, row in results.head(5).iterrows():
        lines.append(
            f"  {row['model']}: R2={row['r2']:.4f}, RMSE={row['rmse']:.4f}, "
            f"features={int(row['n_components'])}"
        )

    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    prefix = args.prefix
    results_file = Path(f"{prefix}model_comparison_results.csv")
    variance_file = Path(f"{prefix}pca_explained_variance.csv")
    loadings_file = Path(f"{prefix}pca_loadings.csv")
    coverage_file = Path(f"{prefix}predictor_coverage.csv")
    summary_file = Path(f"{prefix}training_summary.txt")

    print("=" * 70)
    print("  TRAIN YIELD MODELS ON CANONICAL EASTERN UGANDA DATASET")
    print("=" * 70)

    df, input_path = load_dataset(args.input)
    X, feature_names, coverage_df = select_predictors(df)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy()

    explained, loadings, n_features = fit_pca_diagnostics(X, feature_names)
    n_components = min(MAX_PCA_COMPONENTS, n_features, len(X))
    results = evaluate_models(X, y, n_components)

    coverage_df.to_csv(coverage_file, index=False)
    explained.to_csv(variance_file, index=False)
    loadings.to_csv(loadings_file, index=False)
    results.to_csv(results_file, index=False)
    write_summary(df, feature_names, explained, results, n_components, summary_file, input_path)

    print(f"[✓] Saved: {coverage_file}")
    print(f"[✓] Saved: {variance_file}")
    print(f"[✓] Saved: {loadings_file}")
    print(f"[✓] Saved: {results_file}")
    print(f"[✓] Saved: {summary_file}")
    print()
    print(results.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
