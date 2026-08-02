#!/usr/bin/env python3
"""
Build and evaluate the first real-data benchmark for Eastern Uganda maize yield
using AAS 2020 real yield assignments.

**evaluation_status = "demonstration_only"**

This is the legacy five-row smoke test (``pipeline_smoke_test_2020``).
It is intentionally narrow and MUST NOT be treated as a performance
benchmark: with n=5 rows, seven predictors and only ~3 distinct targets
(pseudo-replicated subregion assignments), the metrics it produces are
unstable and are not evidence of district-level yield predictability.
Formal model comparison uses the subregion x season x year units in
``reports/tables/validation_all.csv``.

  - year: 2020 only
  - geography: Mbale, Kapchorwa, Iganga, Jinja, Tororo
  - target: real AAS 2020 maize yield assigned from sub-region totals

Outputs:
  - eastern_uganda_maize_real_benchmark_2020.csv
  - real_2020_benchmark_model_results.csv
  - real_2020_benchmark_pca_loadings.csv
  - real_2020_benchmark_pca_explained_variance.csv
  - real_2020_benchmark_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


INPUT_FILE = Path("eastern_uganda_maize_modeling_dataset_2020_2023_hybrid_yield.csv")
BENCHMARK_FILE = Path("eastern_uganda_maize_real_benchmark_2020.csv")
RESULTS_FILE = Path("real_2020_benchmark_model_results.csv")
LOADINGS_FILE = Path("real_2020_benchmark_pca_loadings.csv")
VARIANCE_FILE = Path("real_2020_benchmark_pca_explained_variance.csv")
SUMMARY_FILE = Path("real_2020_benchmark_summary.txt")

TARGET = "yield_tons_ha"
REAL_SOURCE = "AAS2020_subregion_assigned_to_district"
BENCHMARK_FEATURES = [
    "MAM",
    "SON",
    "annual_rainfall",
    "rain_cv",
    "annual_gdd",
    "elevation_m",
    "soil_moisture_index",
]


def load_benchmark_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    benchmark = df[(df["year"] == 2020) & (df["yield_source"] == REAL_SOURCE)].copy()
    if benchmark.empty:
        raise ValueError("No 2020 real-yield rows found for the benchmark.")

    benchmark.to_csv(BENCHMARK_FILE, index=False)
    return benchmark


def build_pca_artifacts(X):
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
        index=BENCHMARK_FEATURES,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    ).reset_index(names="feature")

    explained.to_csv(VARIANCE_FILE, index=False)
    loadings.to_csv(LOADINGS_FILE, index=False)
    return explained


def evaluate_models(X, y):
    loo = LeaveOneOut()
    n_components = min(3, X.shape[0] - 1, X.shape[1])

    models = {
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
                ("model", SVR(kernel="rbf", C=3.0, epsilon=0.1)),
            ]
        ),
        "Random Forest (raw)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)),
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
                ("model", SVR(kernel="rbf", C=3.0, epsilon=0.1)),
            ]
        ),
        "Random Forest (PCA)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("model", RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)),
            ]
        ),
    }

    results = []
    for name, pipeline in models.items():
        preds = cross_val_predict(pipeline, X, y, cv=loo)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        results.append(
            {
                "model": name,
                "feature_space": "PCA" if "(PCA)" in name else "raw",
                "rmse": rmse,
                "r2": r2,
                "n_components": n_components if "(PCA)" in name else X.shape[1],
            }
        )

    results_df = pd.DataFrame(results).sort_values(["r2", "rmse"], ascending=[False, True])
    results_df.to_csv(RESULTS_FILE, index=False)
    return results_df, n_components


def write_summary(benchmark, explained, results, n_components):
    best = results.iloc[0]
    lines = [
        "Pipeline smoke test (demonstration only)",
        "=" * 40,
        "evaluation_status = demonstration_only",
        "Label: pipeline_smoke_test_2020 — Eastern Uganda maize, 2020 only",
        "The five-row benchmark is a pipeline smoke test, NOT a performance",
        "benchmark: n=5 rows with pseudo-replicated subregion-assigned targets",
        "and ~3 distinct outcome values are too few for any generalization claim.",
        "Formal evaluation: reports/tables/validation_all.csv.",
        f"Date built: 2026-07-30",
        f"Input file: {INPUT_FILE}",
        f"Benchmark rows: {len(benchmark)}",
        f"Districts: {', '.join(benchmark['district'].tolist())}",
        f"Yield source: {REAL_SOURCE}",
        f"Features: {', '.join(BENCHMARK_FEATURES)}",
        f"PCA components used in PCA models: {n_components}",
        f"Variance explained by first {n_components} PCs: "
        f"{explained['cumulative_explained_variance'].iloc[n_components - 1]:.4f}",
        "",
        "Note:",
        "This smoke test uses only five district rows to confirm data loading,",
        "PCA execution, model fitting and output generation; it is not a valid",
        "measure of district-level yield predictability.",
        "",
        "Best model:",
        f"  {best['model']}",
        f"  R2 = {best['r2']:.4f}",
        f"  RMSE = {best['rmse']:.4f}",
        "",
        "All model results:",
    ]

    for _, row in results.iterrows():
        lines.append(
            f"  {row['model']}: R2={row['r2']:.4f}, RMSE={row['rmse']:.4f}, "
            f"features={int(row['n_components'])}"
        )

    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("=" * 70)
    print("  BUILD 2020-ONLY REAL-DATA BENCHMARK")
    print("=" * 70)

    benchmark = load_benchmark_dataset()
    print(f"[✓] Saved: {BENCHMARK_FILE}")
    print(f"[✓] Shape: {benchmark.shape[0]} rows x {benchmark.shape[1]} columns")
    print(
        benchmark[
            ["district", "year", "yield_tons_ha", "yield_source", "sub_region"] + BENCHMARK_FEATURES
        ].to_string(index=False)
    )
    print()

    X = benchmark[BENCHMARK_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(benchmark[TARGET], errors="coerce").to_numpy()

    explained = build_pca_artifacts(X)
    results, n_components = evaluate_models(X, y)
    write_summary(benchmark, explained, results, n_components)

    print(f"[✓] Saved: {VARIANCE_FILE}")
    print(f"[✓] Saved: {LOADINGS_FILE}")
    print(f"[✓] Saved: {RESULTS_FILE}")
    print(f"[✓] Saved: {SUMMARY_FILE}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
