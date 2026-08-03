#!/usr/bin/env python3
"""Generate a conservative technical report from current result tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model.data.paths import REPORTS, TABLES  # noqa: E402


def main() -> None:
    authoritative = TABLES / "multi_crop_spatial_model_comparison.csv"
    if authoritative.exists():
        results = pd.read_csv(authoritative)
        candidates = results[results.get("target_scale", "raw").eq("raw")] if "target_scale" in results else results
        lines = [
            "# Uganda crop-yield prediction technical report", "",
            "This report reads the authoritative 373-row multi-crop spatial results. "
            "It is descriptive and does not establish causality.", "", "## Validation results", "",
        ]
        for _, row in candidates.head(10).iterrows():
            lines.append(f"- **{row['model']} / {row['feature_space']}**: RMSE {row['rmse']:.3f}, MAE {row['mae']:.3f}, R² {row['r2']:.3f} (n={int(row['observations'])}).")
        lines += ["", "## Interpretation limits", "", "Raw tonnes/ha is primary; log1p and crop-normalized results are sensitivities. The panel covers only 2018 and 2020. AAS2019 and validated local elevation are unavailable, so temporal/stress conclusions are limited. Results are not causal or operational forecasts."]
        output = REPORTS / "technical_report" / "final_report.md"
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines) + "\n"); print(output); return
    results = pd.read_csv(TABLES / "validation_all.csv")
    candidates = results[~results["model"].isin(["mean_predictor", "historical_mean", "previous_year_yield"])]
    lines = [
        "# Uganda crop-yield prediction technical report",
        "",
        "This report summarizes held-out validation results. It is descriptive and does not establish causality.",
        "",
        "## Validation results",
        "",
    ]
    for scheme in ("group_by_subregion", "random_cv", "temporal_2018_2020", "temporal_2020_2018"):
        subset = candidates[candidates["scheme"].eq(scheme)]
        if subset.empty:
            continue
        best = subset.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
        lines.append(
            f"- **{scheme}**: best held-out result was "
            f"{best['representation']}/{best['model']} "
            f"(RMSE {best['rmse']:.3f}, R² {best['r2']:.3f})."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The pooled panel contains repeated crop, season, and subregion structure. "
        "The representation comparison supplies the same crop and season context "
        "to raw, PCA, and hybrid spaces. Results are not causal effects and should "
        "not be described as production-ready forecasts.",
        "",
        "See `reports/tables/model_agreement.csv`, `residual_diagnostics.csv`, "
        "`vif.csv`, and `survey_uncertainty_sensitivity.csv` for supplementary diagnostics.",
    ]
    output = REPORTS / "technical_report" / "final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
