#!/usr/bin/env python3
"""Build the final-analysis model tables and diagnostic reports.

Runs on the single authoritative dataset
``data/processed/final_maize_subregion_season_year.csv``:

1. Missingness report.
2. Descriptive PCA diagnostics (parallel analysis, loading tables).
3. Leakage-safe nested model evaluation under spatial-group or rolling
   rolling-year (temporal) outer splits.

The current AAS milestone has only two years, so rolling-origin temporal
splits (needing > 3 years) cannot be built; the script reports that
explicitly rather than falling back to a leaky shuffled split.

Writes (always to ``reports/tables/``):
  * missingness_report.csv
  * pca_diagnostics.csv
  * spatial_model_comparison.csv  (or temporal_model_comparison.csv)
  * analysis_manifest.json

Run:
  PYTHONPATH=src .venv/bin/python scripts/run_final_analysis.py --mode spatial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from uganda_crop_model.data.paths import (  # noqa: E402
    FINAL_MAIZE_DATASET,
    TABLES,
)
from uganda_crop_model.evaluation.metadata import (  # noqa: E402
    missingness_report,
)
from uganda_crop_model.evaluation.nested_cv import (  # noqa: E402
    run_nested_evaluation,
    summarize_out_of_fold_predictions,
)
from uganda_crop_model.models.pipelines import (  # noqa: E402
    resolve_feature_columns,
)
from uganda_crop_model.models.registry import (  # noqa: E402
    get_model_registry,
)
from uganda_crop_model.pca.diagnostics import (  # noqa: E402
    build_pca_loading_tables,
    fit_standardized_pca,
    parallel_analysis,
)
from uganda_crop_model.pca.stability import (  # noqa: E402
    bootstrap_pca_loadings,
)
from uganda_crop_model.validation.splits import (  # noqa: E402
    rolling_origin_year_splits,
    spatial_group_splits,
)

log = logging.getLogger(__name__)

VALIDATION_MODES = ("spatial", "temporal")


def run_model_evaluation(
    data: pd.DataFrame,
    registry: dict,
    *,
    outer_splits,
    inner_mode: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_all = []
    metrics_all = []

    for feature_space in ("raw", "pca", "hybrid"):
        for model_name, model_spec in registry.items():
            preds, metrics = run_nested_evaluation(
                data,
                feature_space=feature_space,
                model_name=model_name,
                model_spec=model_spec,
                outer_splits=outer_splits,
                inner_mode=inner_mode,
                random_seed=random_seed,
            )
            predictions_all.append(preds)
            metrics_all.append(metrics)

    predictions = pd.concat(predictions_all, ignore_index=True)
    summary = summarize_out_of_fold_predictions(predictions)

    return summary, pd.concat(metrics_all, ignore_index=True)


def build_pca_report(
    data: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = resolve_feature_columns(data)
    continuous = feature_columns["climate"] + feature_columns["static"]

    pca, scaled = fit_standardized_pca(data, continuous)
    loading_table, contribution = build_pca_loading_tables(
        pca,
        continuous,
    )

    diagnostic = {
        "feature": continuous,
        "pc1_loading": loading_table.iloc[:, 0] if loading_table.shape[1] >= 1 else None,
        "pc1_contribution_pct": (
            contribution.iloc[:, 0] if contribution.shape[1] >= 1 else None
        ),
        "pc1_loading_abs": (
            loading_table.iloc[:, 0].abs() if loading_table.shape[1] >= 1 else None
        ),
    }

    report = pd.DataFrame(diagnostic).sort_values(
        "pc1_loading_abs",
        ascending=False,
    )

    report["explained_variance_ratio"] = None
    report["retained_components_parallel"] = None
    report["bootstrap_loading_mean_pc1"] = None
    report["bootstrap_loading_ci_pc1"] = None

    if pca.n_components_ is not None:
        report["explained_variance_ratio"] = pca.explained_variance_ratio_[0] \
            if pca.explained_variance_ratio_.size >= 1 else None
        pa = parallel_analysis(
            scaled,
            iterations=200,
            random_seed=42,
        )
        report["retained_components_parallel"] = pa["retained_components"]

        stability = bootstrap_pca_loadings(
            scaled,
            n_components=min(2, scaled.shape[1]),
            iterations=200,
            random_seed=42,
        )
        mean_pc1 = stability["mean_components"][0]
        lower_pc1 = stability["lower_95"][0]
        upper_pc1 = stability["upper_95"][0]
        report["bootstrap_loading_mean_pc1"] = [
            float(mean_pc1[idx])
            if idx < len(mean_pc1) else float("nan")
            for idx in range(len(continuous))
        ]
        report["bootstrap_loading_ci_pc1"] = [
            f"[{lower_pc1[idx]:.3f}, {upper_pc1[idx]:.3f}]"
            if idx < len(lower_pc1) else "nan"
            for idx in range(len(continuous))
        ]

    return report.drop(columns=["pc1_loading_abs"])


def build_temporal_outer(data: pd.DataFrame) -> list:
    try:
        return list(
            rolling_origin_year_splits(
                data,
                minimum_training_years=3,
            )
        )
    except ValueError as exc:
        log.warning("temporal splits unavailable: %s", exc)
        return []


def analysis_manifest(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    mode: str,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H%M%SZ"
        ),
        "dataset": FINAL_MAIZE_DATASET.name,
        "analysis_mode": "final",
        "rows": int(len(data)),
        "spatial_units": int(data["spatial_unit"].nunique()),
        "years": sorted(int(y) for y in data["year"].unique()),
        "validation": mode,
        "models_evaluated": (
            int(summary["model"].nunique()) if not summary.empty else 0
        ),
        "feature_spaces_evaluated": (
            sorted(summary["feature_space"].unique())
            if not summary.empty else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        default="spatial",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="restrict to these registry keys "
        "(e.g. --models dummy_mean ols ridge)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short smoke run: spatial, dummy/ols/ridge only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not FINAL_MAIZE_DATASET.exists():
        log.error("final dataset not found; run make data-final first")
        sys.exit(1)

    data = pd.read_csv(FINAL_MAIZE_DATASET)
    registry = get_model_registry(random_seed=args.random_seed)

    if args.quick:
        registry = {
            name: spec
            for name, spec in registry.items()
            if name in ("dummy_mean", "ols", "ridge")
        }
    elif args.models:
        registry = {
            name: registry[name]
            for name in args.models
            if name in registry
        }
        if not registry:
            log.error("no valid --models given")
            sys.exit(1)

    TABLES.mkdir(parents=True, exist_ok=True)

    missingness = missingness_report(data).reset_index().rename(
        columns={"index": "column"}
    )
    missingness.to_csv(
        TABLES / "missingness_report.csv",
        index=False,
    )
    log.info("wrote missingness_report.csv")

    pca_report = build_pca_report(data)
    pca_report.to_csv(
        TABLES / "pca_diagnostics.csv",
        index=False,
    )
    log.info("wrote pca_diagnostics.csv")

    summary = None
    metrics = None

    if args.mode == "temporal":
        temporal_splits = build_temporal_outer(data)
        if temporal_splits:
            summary, metrics = run_model_evaluation(
                data,
                registry,
                outer_splits=temporal_splits,
                inner_mode="temporal",
                random_seed=args.random_seed,
            )
        else:
            log.info(
                "no temporal splits available on this sample; "
                "nothing to run"
            )
    else:
        outer_splits = spatial_group_splits(
            data,
            requested_splits=5,
            random_seed=args.random_seed,
        )
        summary, metrics = run_model_evaluation(
            data,
            registry,
            outer_splits=outer_splits,
            inner_mode="spatial",
            random_seed=args.random_seed,
        )

    if summary is not None:
        comparison_name = f"{args.mode}_model_comparison.csv"
        summary.to_csv(
            TABLES / comparison_name,
            index=False,
        )
        metrics.to_csv(
            TABLES / f"{args.mode}_fold_results.csv",
            index=False,
        )
        log.info("wrote %s", comparison_name)

        manifest = analysis_manifest(data, summary, args.mode)
        with (TABLES / "analysis_manifest.json").open("w") as fh:
            json.dump(manifest, fh, indent=2)
        log.info("wrote analysis_manifest.json")

        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()