#!/usr/bin/env python3
"""
Build and evaluate the 2020 real-data benchmark for Eastern Uganda beans yield
using AAS 2020 real yield assignments.

This benchmark is intentionally narrow:
  - year: 2020 only
  - geography: Mbale, Kapchorwa, Iganga, Jinja, Tororo
  - target: real AAS 2020 beans yield assigned from sub-region totals

Outputs:
  - eastern_uganda_beans_real_benchmark_2020.csv
  - real_2020_beans_benchmark_model_results.csv
  - real_2020_beans_benchmark_pca_loadings.csv
  - real_2020_beans_benchmark_pca_explained_variance.csv
  - real_2020_beans_benchmark_summary.txt
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


FEATURES_FILE = Path("eastern_uganda_maize_modeling_dataset_2020_2023.csv")
YIELD_FILE = Path("aas2020_eastern_district_beans_yield_total_2020.csv")
BENCHMARK_FILE = Path("eastern_uganda_beans_real_benchmark_2020.csv")
RESULTS_FILE = Path("real_2020_beans_benchmark_model_results.csv")
LOADINGS_FILE = Path("real_2020_beans_benchmark_pca_loadings.csv")
VARIANCE_FILE = Path("real_2020_beans_benchmark_pca_explained_variance.csv")
SUMMARY_FILE = Path("real_2020_beans_benchmark_summary.txt")

TARGET = "yield_tons_ha"
BENCHMARK_LABEL = "Eastern Uganda beans real-data benchmark, 2020 only"
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
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")
    if not YIELD_FILE.exists():
        raise FileNotFoundError(f"Missing yield file: {YIELD_FILE}")

    features = pd.read_csv(FEATURES_FILE)
    features = features[features["year"] == 2020].copy()

    yield_df = pd.read_csv(YIELD_FILE)
    keep_cols = [
        "district",
        "year",
        "sub_region",
        "source_granularity",
        "yield_tons_ha",
        "yield_tons_ha_planted",
        "area_planted_ha",
        "area_harvested_ha",
        "production_mt",
        "cv_area_planted_pct",
        "cv_area_harvested_pct",
        "cv_production_pct",
    ]
    yield_df = yield_df[keep_cols].copy()
    yield_df = yield_df.rename(
        columns={
            "yield_tons_ha": "yield_tons_ha_beans2020",
            "yield_tons_ha_planted": "yield_tons_ha_planted_beans2020",
            "area_planted_ha": "area_planted_ha_beans2020",
            "area_harvested_ha": "area_harvested_ha_beans2020",
            "production_mt": "production_mt_beans2020",
            "cv_area_planted_pct": "cv_area_planted_pct_beans2020",
            "cv_area_harvested_pct": "cv_area_harvested_pct_beans2020",
            "cv_production_pct": "cv_production_pct_beans2020",
        }
    )

    merged = features.merge(yield_df, on=["district", "year"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No merged 2020 beans benchmark rows were produced.")

    merged["yield_tons_ha_original_maize_proxy"] = merged["yield_tons_ha"]
    merged["yield_tons_ha"] = merged["yield_tons_ha_beans2020"]
    merged["yield_source"] = "AAS2020_subregion_assigned_to_district"

    merged.to_csv(BENCHMARK_FILE, index=False)
    return merged


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
        "Real-Data Benchmark",
        "=" * 24,
        f"Label: {BENCHMARK_LABEL}",
        "Date built: 2026-07-31",
        f"Features file: {FEATURES_FILE}",
        f"Yield file: {YIELD_FILE}",
        f"Benchmark rows: {len(benchmark)}",
        f"Districts: {', '.join(benchmark['district'].tolist())}",
        "Yield source: AAS2020_subregion_assigned_to_district",
        f"Features: {', '.join(BENCHMARK_FEATURES)}",
        f"PCA components used in PCA models: {n_components}",
        f"Variance explained by first {n_components} PCs: "
        f"{explained['cumulative_explained_variance'].iloc[n_components - 1]:.4f}",
        "",
        "Caution:",
        "This benchmark uses only five district rows and is intended as a real-data",
        "sanity check, not a final generalization claim.",
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
    print("  BUILD 2020-ONLY REAL-DATA BEANS BENCHMARK")
    print("=" * 70)

    benchmark = load_benchmark_dataset()
    print(f"[✓] Saved: {BENCHMARK_FILE}")
    print(f"[✓] Shape: {benchmark.shape[0]} rows x {benchmark.shape[1]} columns")
    print(
        benchmark[
            [
                "district",
                "year",
                "yield_tons_ha",
                "yield_source",
                "sub_region",
            ]
            + BENCHMARK_FEATURES
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
