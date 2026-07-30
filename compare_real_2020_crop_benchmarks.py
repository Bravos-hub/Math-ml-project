#!/usr/bin/env python3
"""
Compare the 2020 Eastern Uganda real-data crop benchmarks for maize, beans,
and groundnuts.

This script consolidates the benchmark datasets and model results into a
side-by-side summary for reporting and portfolio use.

Outputs:
  - real_2020_crop_benchmark_comparison.csv
  - real_2020_crop_benchmark_district_targets.csv
  - real_2020_crop_benchmark_comparison.md
"""

from pathlib import Path

import pandas as pd


MAIZE_BENCHMARK_FILE = Path("eastern_uganda_maize_real_benchmark_2020.csv")
BEANS_BENCHMARK_FILE = Path("eastern_uganda_beans_real_benchmark_2020.csv")
GROUNDNUTS_BENCHMARK_FILE = Path("eastern_uganda_groundnuts_real_benchmark_2020.csv")
MAIZE_RESULTS_FILE = Path("real_2020_benchmark_model_results.csv")
BEANS_RESULTS_FILE = Path("real_2020_beans_benchmark_model_results.csv")
GROUNDNUTS_RESULTS_FILE = Path("real_2020_groundnuts_benchmark_model_results.csv")

COMPARISON_FILE = Path("real_2020_crop_benchmark_comparison.csv")
DISTRICT_FILE = Path("real_2020_crop_benchmark_district_targets.csv")
REPORT_FILE = Path("real_2020_crop_benchmark_comparison.md")

FEATURES = [
    "MAM",
    "SON",
    "annual_rainfall",
    "rain_cv",
    "annual_gdd",
    "elevation_m",
    "soil_moisture_index",
]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_inputs():
    for path in [
        MAIZE_BENCHMARK_FILE,
        BEANS_BENCHMARK_FILE,
        GROUNDNUTS_BENCHMARK_FILE,
        MAIZE_RESULTS_FILE,
        BEANS_RESULTS_FILE,
        GROUNDNUTS_RESULTS_FILE,
    ]:
        require_file(path)

    maize = pd.read_csv(MAIZE_BENCHMARK_FILE)
    beans = pd.read_csv(BEANS_BENCHMARK_FILE)
    groundnuts = pd.read_csv(GROUNDNUTS_BENCHMARK_FILE)
    maize_results = pd.read_csv(MAIZE_RESULTS_FILE)
    beans_results = pd.read_csv(BEANS_RESULTS_FILE)
    groundnuts_results = pd.read_csv(GROUNDNUTS_RESULTS_FILE)
    return maize, beans, groundnuts, maize_results, beans_results, groundnuts_results


def build_comparison(maize, beans, groundnuts, maize_results, beans_results, groundnuts_results):
    benchmark_specs = [
        (
            "maize",
            "Eastern Uganda maize real-data benchmark, 2020 only",
            maize,
            maize_results,
        ),
        (
            "beans",
            "Eastern Uganda beans real-data benchmark, 2020 only",
            beans,
            beans_results,
        ),
        (
            "groundnuts",
            "Eastern Uganda groundnuts real-data benchmark, 2020 only",
            groundnuts,
            groundnuts_results,
        ),
    ]

    rows = []
    for crop, label, benchmark, results in benchmark_specs:
        best = results.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "crop": crop,
                "benchmark_label": label,
                "benchmark_year": 2020,
                "district_count": int(benchmark["district"].nunique()),
                "row_count": int(len(benchmark)),
                "yield_source": "AAS2020_subregion_assigned_to_district",
                "best_model": best["model"],
                "best_feature_space": best["feature_space"],
                "best_r2": float(best["r2"]),
                "best_rmse": float(best["rmse"]),
                "yield_mean_tons_ha": float(benchmark["yield_tons_ha"].mean()),
                "yield_std_tons_ha": float(benchmark["yield_tons_ha"].std(ddof=1)),
                "yield_min_tons_ha": float(benchmark["yield_tons_ha"].min()),
                "yield_max_tons_ha": float(benchmark["yield_tons_ha"].max()),
                "features_used": ", ".join(FEATURES),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("crop").reset_index(drop=True)
    comparison.to_csv(COMPARISON_FILE, index=False)
    return comparison


def build_district_table(maize, beans, groundnuts):
    maize_cols = [
        "district",
        "sub_region",
        "yield_tons_ha",
        "annual_rainfall",
        "annual_gdd",
        "elevation_m",
        "soil_moisture_index",
    ]
    beans_cols = [
        "district",
        "yield_tons_ha",
    ]

    district = maize[maize_cols].rename(columns={"yield_tons_ha": "maize_yield_tons_ha"})
    district = district.merge(
        beans[beans_cols].rename(columns={"yield_tons_ha": "beans_yield_tons_ha"}),
        on="district",
        how="inner",
        validate="one_to_one",
    )
    district = district.merge(
        groundnuts[beans_cols].rename(columns={"yield_tons_ha": "groundnuts_yield_tons_ha"}),
        on="district",
        how="inner",
        validate="one_to_one",
    )
    district["yield_gap_maize_minus_beans"] = (
        district["maize_yield_tons_ha"] - district["beans_yield_tons_ha"]
    )
    district["yield_gap_maize_minus_groundnuts"] = (
        district["maize_yield_tons_ha"] - district["groundnuts_yield_tons_ha"]
    )
    district = district.sort_values("district").reset_index(drop=True)
    district.to_csv(DISTRICT_FILE, index=False)
    return district


def markdown_table(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(comparison, district):
    maize_row = comparison[comparison["crop"] == "maize"].iloc[0]
    beans_row = comparison[comparison["crop"] == "beans"].iloc[0]
    groundnuts_row = comparison[comparison["crop"] == "groundnuts"].iloc[0]

    lines = [
        "# 2020 Real-Data Crop Benchmark Comparison",
        "",
        "This memo compares the first three Eastern Uganda `2020` real-data benchmarks",
        "built from UBOS AAS 2020 sub-region yields assigned to districts.",
        "",
        "## Scope",
        "",
        "- Geography: `Iganga`, `Jinja`, `Kapchorwa`, `Mbale`, `Tororo`",
        "- Year: `2020`",
        "- Predictors: `MAM`, `SON`, `annual_rainfall`, `rain_cv`, `annual_gdd`, `elevation_m`, `soil_moisture_index`",
        "- Yield source: `AAS2020_subregion_assigned_to_district`",
        "- Validation: leave-one-out cross-validation on `5` district rows",
        "",
        "## Benchmark Summary",
        "",
        "| Crop | Best model | Best R² | Best RMSE | Mean yield (t/ha) | Std. dev. |",
        "|---|---|---:|---:|---:|---:|",
        f"| Maize | {maize_row['best_model']} | {maize_row['best_r2']:.4f} | {maize_row['best_rmse']:.4f} | {maize_row['yield_mean_tons_ha']:.4f} | {maize_row['yield_std_tons_ha']:.4f} |",
        f"| Beans | {beans_row['best_model']} | {beans_row['best_r2']:.4f} | {beans_row['best_rmse']:.4f} | {beans_row['yield_mean_tons_ha']:.4f} | {beans_row['yield_std_tons_ha']:.4f} |",
        f"| Groundnuts | {groundnuts_row['best_model']} | {groundnuts_row['best_r2']:.4f} | {groundnuts_row['best_rmse']:.4f} | {groundnuts_row['yield_mean_tons_ha']:.4f} | {groundnuts_row['yield_std_tons_ha']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"- Maize is the stronger first benchmark in this setup: its best model reached `R² = {maize_row['best_r2']:.4f}` with `RMSE = {maize_row['best_rmse']:.4f}`.",
        f"- Beans is much weaker under the same feature set: its best model reached `R² = {beans_row['best_r2']:.4f}` with `RMSE = {beans_row['best_rmse']:.4f}`.",
        f"- Groundnuts is a stronger secondary benchmark than beans: its best model reached `R² = {groundnuts_row['best_r2']:.4f}` with `RMSE = {groundnuts_row['best_rmse']:.4f}`.",
        "- The likely reason is target structure, not a pipeline failure. The beans target has only three unique sub-region-assigned values across five districts, so there is limited learnable variation at district level.",
        "- Groundnuts also has only three sub-region-assigned target values, but with a wider spread than beans. That makes it a better test than beans, though still weaker than a true district-level panel.",
        "- The maize target separates the districts more strongly, which makes rainfall, temperature, and terrain features appear more predictive in this first benchmark.",
        "- Both results remain preliminary because the benchmark uses only five rows and sub-region-assigned yields rather than direct district microdata.",
        "",
        "## District Targets",
        "",
        markdown_table(district),
        "",
        "## Recommendation",
        "",
        "- Keep maize as the primary first real-data benchmark.",
        "- Use groundnuts as the stronger secondary benchmark because it preserves some learnable environmental variation under the same five-district setup.",
        "- Treat beans as the weakest contrast case showing that some crops need either more districts, more years, or crop-specific management variables before environmental PCA features can explain yield well.",
        "",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 70)
    print("  COMPARE 2020 REAL-DATA CROP BENCHMARKS")
    print("=" * 70)

    maize, beans, groundnuts, maize_results, beans_results, groundnuts_results = load_inputs()
    comparison = build_comparison(
        maize,
        beans,
        groundnuts,
        maize_results,
        beans_results,
        groundnuts_results,
    )
    district = build_district_table(maize, beans, groundnuts)
    build_report(comparison, district)

    print(f"[✓] Saved: {COMPARISON_FILE}")
    print(comparison.to_string(index=False))
    print()
    print(f"[✓] Saved: {DISTRICT_FILE}")
    print(district.to_string(index=False))
    print()
    print(f"[✓] Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
