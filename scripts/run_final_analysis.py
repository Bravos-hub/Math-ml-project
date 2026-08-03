#!/usr/bin/env python3
"""Authoritative, leakage-safe multi-crop analysis runner."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model.data.paths import FINAL_MAIZE_DATASET, FINAL_MULTI_CROP_DATASET, TABLES
from uganda_crop_model.data.ubos_waves import validate_wave_source
from uganda_crop_model.evaluation.baselines import predict_training_baselines
from uganda_crop_model.evaluation.metadata import missingness_report
from uganda_crop_model.evaluation.nested_cv import run_nested_evaluation, summarize_out_of_fold_predictions, heldout_permutation_diagnostics
from uganda_crop_model.evaluation.outliers import robust_mahalanobis_flags
from uganda_crop_model.evaluation.interpretability import heldout_permutation_importance, residual_diagnostics, model_agreement
from uganda_crop_model.models.pipelines import resolve_feature_columns
from uganda_crop_model.models.registry import get_model_registry
from uganda_crop_model.pca.diagnostics import build_pca_loading_tables, fit_standardized_pca, parallel_analysis
from uganda_crop_model.pca.stability import bootstrap_pca_loadings
from uganda_crop_model.validation.splits import (
    leave_one_subregion_out_splits, rolling_origin_year_splits,
    spatial_group_splits, future_unseen_location_splits,
)

log = logging.getLogger(__name__)
VALIDATION_MODES = ("spatial", "temporal", "loso", "stress")


def build_pca_report(data):
    cols = resolve_feature_columns(data)
    continuous = cols["climate"]
    pca, scaled = fit_standardized_pca(data, continuous)
    loading, contribution = build_pca_loading_tables(pca, continuous)
    report = pd.DataFrame({
        "feature": continuous,
        "pc1_loading": loading.iloc[:, 0],
        "pc1_contribution_pct": contribution.iloc[:, 0],
    })
    report["explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
    report["retained_components_parallel"] = parallel_analysis(scaled, iterations=100)["retained_components"]
    stability = bootstrap_pca_loadings(scaled, n_components=min(2, scaled.shape[1]), iterations=100)
    report["bootstrap_loading_mean_pc1"] = stability["mean_components"][0]
    report["bootstrap_loading_ci_pc1"] = [
        f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(stability["lower_95"][0], stability["upper_95"][0])
    ]
    return report.sort_values("pc1_loading", key=lambda x: x.abs(), ascending=False)


def outer_splits(data, mode, seed):
    if mode == "spatial":
        return spatial_group_splits(data, requested_splits=5, random_seed=seed), "spatial"
    if mode == "loso":
        return leave_one_subregion_out_splits(data), "spatial"
    if mode == "temporal":
        return list(rolling_origin_year_splits(data, minimum_training_years=3)), "temporal"
    return list(future_unseen_location_splits(data, minimum_training_years=3)), "temporal"


def safe_metrics(observed, predicted):
    return {
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)) if len(observed) > 1 and np.unique(observed).size > 1 else np.nan,
    }


def crop_metrics(predictions):
    rows = []
    for (model, space, crop), group in predictions.groupby(["model", "feature_space", "crop"]):
        values = safe_metrics(group.observed_yield, group.predicted_yield)
        values.update(model=model, feature_space=space, crop=crop, observations=len(group),
                      target_scale="raw", skill_vs_crop_mean=np.nan)
        rows.append(values)
    return pd.DataFrame(rows)


def normalized_metrics(predictions, data, transform):
    """Compute sensitivity metrics using fold-independent declared transforms.

    Crop-normalized means are estimated from the training side in the
    evaluation module in future extensions; this table is deliberately marked
    sensitivity and never replaces raw tonnes/ha results.
    """
    frame = predictions.copy()
    if transform == "log1p":
        frame["observed_target"] = np.log1p(frame.observed_yield)
        frame["predicted_target"] = np.log1p(frame.predicted_yield.clip(lower=0))
    else:
        means = frame["training_crop_mean"] if "training_crop_mean" in frame else data.yield_tons_ha.mean()
        frame["observed_target"] = frame.observed_yield - means
        frame["predicted_target"] = frame.predicted_yield - means
    rows = []
    for (model, space), group in frame.groupby(["model", "feature_space"]):
        m = safe_metrics(group.observed_target, group.predicted_target)
        m.update(model=model, feature_space=space, observations=len(group), target_scale=transform)
        rows.append(m)
    return pd.DataFrame(rows)


def baseline_predictions(data, splits):
    rows, fallback_rows = [], []
    for fold, (train, test) in enumerate(splits, 1):
        for name, (values, fallbacks) in predict_training_baselines(data, train, test).items():
            part = data.iloc[test][["spatial_unit", "year", "season", "crop"]].copy()
            part["fold"] = fold; part["model"] = name; part["feature_space"] = "baseline"
            part["observed_yield"] = data.iloc[test].yield_tons_ha.to_numpy(); part["predicted_yield"] = values
            rows.append(part)
            fallback_rows.append({"model": name, "fold": fold, "fallback_count": fallbacks, "test_rows": len(test)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fallback_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALIDATION_MODES, default="spatial")
    parser.add_argument("--dataset", choices=("maize", "multi_crop"), default="maize")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = FINAL_MAIZE_DATASET if args.dataset == "maize" else FINAL_MULTI_CROP_DATASET
    prefix = "" if args.dataset == "maize" else "multi_crop_"
    if not path.exists():
        raise SystemExit(f"final dataset not found: {path}")
    data = pd.read_csv(path)
    cols = resolve_feature_columns(data)
    missingness_report(data).reset_index().rename(columns={"index": "column"}).to_csv(TABLES / f"{prefix}missingness_report.csv", index=False)
    build_pca_report(data).to_csv(TABLES / f"{prefix}pca_diagnostics.csv", index=False)
    try:
        splits, inner_mode = outer_splits(data, args.mode, args.random_seed)
    except ValueError as exc:
        log.warning("%s validation unavailable: %s", args.mode, exc)
        manifest = {"dataset": path.name, "validation": args.mode, "status": "unavailable", "reason": str(exc)}
        (TABLES / f"{prefix}analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
        return
    registry = get_model_registry(
        random_seed=args.random_seed,
        load_optional=not args.quick,
    )
    if args.quick:
        registry = {k: v for k, v in registry.items() if k in {"dummy_mean", "ols", "ridge"}}
    elif args.models:
        registry = {k: registry[k] for k in args.models if k in registry}
    predictions, fold_results, permutation, execution_status = [], [], [], []
    for space in ("raw", "pca", "hybrid"):
        for name, spec in registry.items():
            try:
                pred, folds = run_nested_evaluation(data, feature_space=space, model_name=name, model_spec=spec,
                                                    outer_splits=splits, inner_mode=inner_mode, random_seed=args.random_seed)
                predictions.append(pred); fold_results.append(folds)
                execution_status.append({"model": name, "feature_space": space, "status": "completed", "reason": ""})
                importance = heldout_permutation_diagnostics(data, feature_space=space, model_spec=spec,
                                                             outer_splits=splits, random_seed=args.random_seed)
                importance["model"] = name; importance["feature_space"] = space
                permutation.append(importance)
                pred.to_csv(TABLES / f"{prefix}{args.mode}_{name}_{space}_predictions.csv", index=False)
            except Exception as exc:
                log.warning("checkpoint skipped for %s/%s: %s", name, space, exc)
                execution_status.append({"model": name, "feature_space": space, "status": "unavailable", "reason": str(exc)})
    if not predictions:
        raise SystemExit("No model/feature-space evaluation completed")
    pred = pd.concat(predictions, ignore_index=True)
    baseline, fallback = baseline_predictions(data, splits)
    pred = pd.concat([pred, baseline], ignore_index=True)
    summary = summarize_out_of_fold_predictions(pred)
    summary = pd.concat([summary, crop_metrics(pred)], ignore_index=True, sort=False)
    summary = pd.concat([summary, normalized_metrics(pred, data, "log1p"), normalized_metrics(pred, data, "crop_normalized")], ignore_index=True, sort=False)
    fold = pd.concat(fold_results, ignore_index=True)
    pred.to_csv(TABLES / f"{prefix}{args.mode}_predictions.csv", index=False)
    summary.to_csv(TABLES / f"{prefix}{args.mode}_model_comparison.csv", index=False)
    fold.to_csv(TABLES / f"{prefix}{args.mode}_fold_results.csv", index=False)
    pd.DataFrame(execution_status).to_csv(TABLES / f"{prefix}{args.mode}_execution_status.csv", index=False)
    # Diagnostics consume the same OOF predictions and never delete rows.
    coverage_pred = pred[pred["feature_space"].ne("baseline")].copy()
    coverage = coverage_pred.assign(covered=lambda x: x.observed_yield.between(x.conformal_lower, x.conformal_upper))
    coverage.groupby(["model", "feature_space"]).agg(coverage=("covered", "mean"), observations=("covered", "size")).reset_index().to_csv(TABLES / f"{prefix}spatial_conformal_coverage.csv", index=False)
    feat = cols["climate"] + cols["static"]
    try:
        flags = robust_mahalanobis_flags(data, feat).reindex(data.index)
        flags.insert(0, "row_index", flags.index); flags.to_csv(TABLES / f"{prefix}outlier_flags.csv", index=False)
    except ValueError as exc:
        pd.DataFrame({"status": ["unavailable"], "reason": [str(exc)]}).to_csv(TABLES / f"{prefix}outlier_flags.csv", index=False)
    residual_diagnostics(pred.observed_yield, pred.predicted_yield).to_csv(TABLES / f"{prefix}residual_diagnostics.csv", index=False)
    model_agreement(pred).to_csv(TABLES / f"{prefix}model_agreement.csv", index=False)
    pd.concat(permutation, ignore_index=True).to_csv(TABLES / f"{prefix}heldout_permutation_importance.csv", index=False)
    wave_checks = [validate_wave_source(p) for p in sorted(Path("data/raw").glob("*2015-2021*.xlsx"))]
    wave_checks.append(validate_wave_source(Path("data/raw/AAS2019.pdf")))
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": path.name, "rows": len(data), "spatial_units": int(data.spatial_unit.nunique()), "years": sorted(map(int, data.year.unique())), "validation": args.mode, "status": "available", "models": sorted(pred.model.unique()), "feature_spaces": sorted(pred.feature_space.unique()), "target_primary": "raw tonnes/ha", "target_sensitivities": ["log1p", "crop_normalized"], "elevation": {"status": "unavailable", "reason": "no validated local elevation file"}, "baseline_fallbacks": fallback.to_dict(orient="records"), "execution_status": execution_status, "additional_wave_validation": wave_checks, "outputs": {"predictions": f"{prefix}{args.mode}_predictions.csv", "comparison": f"{prefix}{args.mode}_model_comparison.csv", "coverage": f"{prefix}spatial_conformal_coverage.csv", "execution_status": f"{prefix}{args.mode}_execution_status.csv"}}
    (TABLES / f"{prefix}analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
