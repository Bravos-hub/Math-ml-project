#!/usr/bin/env python3
"""Generate a conservative technical report from current result tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model.data.paths import REPORTS, TABLES  # noqa: E402


def main() -> None:
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
