#!/usr/bin/env python3
"""Generate the interim technical report from authoritative result tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model.data.paths import REPORTS, TABLES


def select_primary_results(results: pd.DataFrame) -> pd.DataFrame:
    """Select registered overall raw-scale results, sorted by error."""

    candidates = results.copy()
    if "registered_primary_metric" in candidates:
        selected = candidates["registered_primary_metric"].fillna(False)
        if selected.dtype != bool:
            selected = selected.astype(str).str.lower().eq("true")
        candidates = candidates[selected]
    if "result_scope" in candidates:
        candidates = candidates[candidates["result_scope"].eq("overall")]
    elif "crop" in candidates:
        candidates = candidates[candidates["crop"].isna()]
    if "target_scale" in candidates:
        candidates = candidates[candidates["target_scale"].fillna("raw").eq("raw")]
    return candidates.sort_values(["rmse", "mae"], ascending=[True, True])


def _metric_line(row: pd.Series) -> str:
    return (
        f"- **{row['model']} / {row['feature_space']}**: "
        f"RMSE {row['rmse']:.3f}, MAE {row['mae']:.3f}, "
        f"R² {row['r2']:.3f} (n={int(row['observations'])})."
    )


def build_report(results: pd.DataFrame, manifest: dict[str, object]) -> str:
    primary = select_primary_results(results)
    baselines = primary[primary["feature_space"].eq("baseline")]
    models = primary[~primary["feature_space"].eq("baseline")]
    scopes = results.get("result_scope", pd.Series(index=results.index, dtype=str))
    scales = results.get("target_scale", pd.Series("raw", index=results.index))
    per_crop = results[
        scopes.eq("per_crop") & scales.fillna("raw").eq("raw")
    ].sort_values(["rmse", "mae"])

    lines = [
        "# Interim technical report: selected food-crop yields in Uganda",
        "",
        (
            "This report is generated from the leakage-controlled authoritative run. "
            "It describes retrospective secondary-data yield estimation, not an "
            "early-warning forecast or causal analysis."
        ),
        "",
        "## Best overall model",
        "",
    ]
    if models.empty:
        lines.append("No registered primary model completed successfully.")
    else:
        lines.append(_metric_line(models.iloc[0]))

    lines += ["", "## Training-derived baselines", ""]
    if baselines.empty:
        lines.append("No applicable baseline result was available.")
    else:
        lines.extend(_metric_line(row) for _, row in baselines.iterrows())

    lines += ["", "## Per-crop diagnostic performance", ""]
    if per_crop.empty:
        lines.append("Per-crop diagnostics were unavailable.")
    else:
        for crop, group in per_crop.groupby("crop"):
            row = group.iloc[0]
            lines.append(
                f"- **{crop}**: best {row['model']} / {row['feature_space']} "
                f"had RMSE {row['rmse']:.3f}, MAE {row['mae']:.3f}, "
                f"and R² {row['r2']:.3f}."
            )

    gate_passed = bool(manifest.get("final_acceptance_gate_passed", False))
    lines += [
        "",
        "## Acceptance gates",
        "",
        f"Final acceptance gates passed: **{'yes' if gate_passed else 'no'}**.",
    ]
    if manifest.get("acceptance_gate_reason"):
        lines.append(str(manifest["acceptance_gate_reason"]))

    lines += [
        "",
        "## Limitations",
        "",
        (
            "The validated source material contains only AAS 2018 and 2020. The "
            "primary seasonal analysis currently contains only 2020, while 2018 is "
            "annual and is analyzed separately. Temporal generalization therefore "
            "cannot be established. With only 14 subregions, group-held-out "
            "calibration sets are small; interval coverage is reported empirically "
            "and should be interpreted cautiously. Pooled performance can conceal "
            "negative crop-specific R² values, so it is not evidence of uniformly "
            "successful prediction for every crop."
        ),
        "",
    ]
    return "\n".join(lines)


def write_interim_report(
    comparison_path: Path,
    manifest_path: Path,
    output: Path | None = None,
) -> Path:
    results = pd.read_csv(comparison_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    output = output or REPORTS / "technical_report" / "interim_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(results, manifest))
    return output


def main() -> None:
    options = [
        (
            TABLES / "multi_crop_seasonal_spatial_model_comparison.csv",
            TABLES / "multi_crop_seasonal_analysis_manifest.json",
        ),
        (
            TABLES / "multi_crop_spatial_model_comparison.csv",
            TABLES / "multi_crop_analysis_manifest.json",
        ),
    ]
    for comparison, manifest in options:
        if comparison.exists():
            print(write_interim_report(comparison, manifest))
            return
    raise SystemExit("No authoritative model-comparison table exists.")


if __name__ == "__main__":
    main()
