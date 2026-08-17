#!/usr/bin/env python3
"""Authoritative, leakage-safe selected-food-crop analysis runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model import __version__
from uganda_crop_model.data.paths import (
    CONFIGS,
    FINAL_MAIZE_DATASET,
    FINAL_MULTI_CROP_ANNUAL_DATASET,
    FINAL_MULTI_CROP_DATASET,
    FINAL_MULTI_CROP_SEASONAL_DATASET,
    RUNS,
    TABLES,
)
from uganda_crop_model.data.ubos_waves import validate_wave_source
from uganda_crop_model.evaluation.baselines import (
    predict_training_baselines,
)
from uganda_crop_model.evaluation.interpretability import (
    model_agreement,
    residual_diagnostics,
)
from uganda_crop_model.evaluation.metadata import missingness_report
from uganda_crop_model.evaluation.nested_cv import (
    heldout_permutation_diagnostics,
    run_nested_evaluation,
    skill_score,
    summarize_conformal_coverage,
    summarize_out_of_fold_predictions,
)
from uganda_crop_model.evaluation.outliers import robust_mahalanobis_flags
from uganda_crop_model.models.pipelines import resolve_feature_columns
from uganda_crop_model.models.registry import get_model_registry
from uganda_crop_model.pca.diagnostics import (
    build_pca_loading_tables,
    fit_standardized_pca,
    parallel_analysis,
)
from uganda_crop_model.pca.stability import bootstrap_pca_loadings
from uganda_crop_model.quality.dataset import (
    AnalysisPolicy,
    validate_final_dataset,
)
from uganda_crop_model.validation.splits import (
    future_unseen_location_splits,
    leave_one_subregion_out_splits,
    rolling_origin_year_splits,
    spatial_group_splits,
)

log = logging.getLogger(__name__)
VALIDATION_MODES = ("spatial", "temporal", "loso", "stress")
DATASETS = {
    "maize": (FINAL_MAIZE_DATASET, ""),
    "multi_crop": (FINAL_MULTI_CROP_DATASET, "multi_crop_"),
    "multi_crop_seasonal": (
        FINAL_MULTI_CROP_SEASONAL_DATASET,
        "multi_crop_seasonal_",
    ),
    "multi_crop_annual": (
        FINAL_MULTI_CROP_ANNUAL_DATASET,
        "multi_crop_annual_",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(CONFIGS.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def environmental_pca_frame(data: pd.DataFrame, continuous: list[str]) -> pd.DataFrame:
    """Give each independent environment equal weight in descriptive PCA."""

    key = ["spatial_unit", "year", "season"]
    return data[key + continuous].drop_duplicates(key).reset_index(drop=True)


def feature_timing_table(feature_columns: dict[str, list[str]]) -> pd.DataFrame:
    """Declare when every predictor becomes available."""

    rows = []
    for feature in feature_columns["static"]:
        rows.append(
            {
                "feature": feature,
                "feature_available_by": "pre_season",
                "prediction_horizon": "season_end_retrospective",
            }
        )
    for feature in feature_columns["categorical"]:
        rows.append(
            {
                "feature": feature,
                "feature_available_by": "pre_season",
                "prediction_horizon": "season_end_retrospective",
            }
        )
    for feature in feature_columns["climate"]:
        rows.append(
            {
                "feature": feature,
                "feature_available_by": "season_end",
                "prediction_horizon": "season_end_retrospective",
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def build_pca_report(data: pd.DataFrame) -> pd.DataFrame:
    cols = resolve_feature_columns(data)
    continuous = cols["climate"]
    pca_data = environmental_pca_frame(data, continuous)
    pca, scaled = fit_standardized_pca(pca_data, continuous)
    loading, contribution = build_pca_loading_tables(pca, continuous)
    report = pd.DataFrame(
        {
            "feature": continuous,
            "pc1_loading": loading.iloc[:, 0],
            "pc1_contribution_pct": contribution.iloc[:, 0],
        }
    )
    report["descriptive_environment_count"] = len(pca_data)
    report["explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
    report["retained_components_parallel"] = parallel_analysis(scaled, iterations=100)[
        "retained_components"
    ]
    stability = bootstrap_pca_loadings(
        scaled, n_components=min(2, scaled.shape[1]), iterations=100
    )
    report["bootstrap_loading_mean_pc1"] = stability["mean_components"][0]
    report["bootstrap_loading_ci_pc1"] = [
        f"[{lo:.3f}, {hi:.3f}]"
        for lo, hi in zip(stability["lower_95"][0], stability["upper_95"][0])
    ]
    return report.sort_values("pc1_loading", key=lambda x: x.abs(), ascending=False)


def outer_splits(data: pd.DataFrame, mode: str, seed: int):
    if mode == "spatial":
        return spatial_group_splits(
            data, requested_splits=5, random_seed=seed
        ), "spatial"
    if mode == "loso":
        return leave_one_subregion_out_splits(data), "spatial"
    if mode == "temporal":
        return list(
            rolling_origin_year_splits(data, minimum_training_years=3)
        ), "temporal"
    return list(
        future_unseen_location_splits(data, minimum_training_years=3)
    ), "temporal"


def safe_metrics(observed, predicted) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted))
        if len(observed) > 1 and np.unique(observed).size > 1
        else np.nan,
    }


def crop_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["model", "feature_space", "target_scale", "crop"]
    for (model, space, scale, crop), group in predictions.groupby(group_columns):
        observed = group.get("observed_evaluation_target", group["observed_yield"])
        predicted = group.get("predicted_evaluation_target", group["predicted_yield"])
        values = safe_metrics(observed, predicted)
        values.update(
            model=model,
            feature_space=space,
            crop=crop,
            observations=len(group),
            target_scale=scale,
            result_scope="per_crop",
            registered_primary_metric=False,
            skill_vs_training_global_mean=(
                skill_score(
                    group["observed_yield"].to_numpy(),
                    group["predicted_yield"].to_numpy(),
                    group["training_global_mean"].to_numpy(),
                )
                if "training_global_mean" in group
                else np.nan
            ),
            skill_vs_training_crop_mean=(
                skill_score(
                    group["observed_yield"].to_numpy(),
                    group["predicted_yield"].to_numpy(),
                    group["training_crop_mean"].to_numpy(),
                )
                if "training_crop_mean" in group
                else np.nan
            ),
        )
        rows.append(values)
    return pd.DataFrame(rows)


def macro_average(crop_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse",
        "mae",
        "r2",
        "skill_vs_training_global_mean",
        "skill_vs_training_crop_mean",
    ]
    macro = (
        crop_summary.groupby(["model", "feature_space", "target_scale"])[metrics]
        .mean()
        .reset_index()
    )
    macro["observations"] = (
        crop_summary.groupby(["model", "feature_space", "target_scale"])["crop"]
        .nunique()
        .to_numpy()
    )
    macro["result_scope"] = "macro_average"
    macro["registered_primary_metric"] = False
    return macro


def baseline_predictions(data: pd.DataFrame, splits, validation_mode: str):
    rows, fallback_rows = [], []
    for fold, (train, test) in enumerate(splits, 1):
        training = data.iloc[train]
        global_mean = float(training["yield_tons_ha"].mean())
        crop_means = training.groupby("crop")["yield_tons_ha"].mean()
        outputs = predict_training_baselines(
            data, train, test, validation_mode=validation_mode
        )
        for name, output in outputs.items():
            part = data.iloc[test][["spatial_unit", "year", "season", "crop"]].copy()
            part["fold"] = fold
            part["model"] = name
            part["feature_space"] = "baseline"
            part["target_scale"] = "raw"
            part["observed_yield"] = data.iloc[test].yield_tons_ha.to_numpy()
            part["predicted_yield"] = output.values
            part["observed_evaluation_target"] = part["observed_yield"]
            part["predicted_evaluation_target"] = part["predicted_yield"]
            part["training_global_mean"] = global_mean
            part["training_crop_mean"] = (
                data.iloc[test]["crop"].map(crop_means).fillna(global_mean).to_numpy()
            )
            part["baseline_applicable"] = output.baseline_applicable
            part["fallback_used"] = output.fallback_used
            part["fallback_level"] = output.fallback_level
            rows.append(part)
            fallback_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "fallback_count": int(output.fallback_used.sum()),
                    "test_rows": len(test),
                    "fallback_rate": float(output.fallback_used.mean()),
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fallback_rows)


def _acceptance_gate(data: pd.DataFrame, features: list[str]) -> tuple[bool, str]:
    try:
        validate_final_dataset(data, features, AnalysisPolicy())
    except ValueError as exc:
        return False, str(exc)
    return True, "All declared final-analysis data gates passed."


def _write_run_bundle(
    *,
    run_id: str,
    manifest: dict[str, object],
    artifacts: dict[str, Path],
) -> Path:
    run_dir = RUNS / run_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for name, source in artifacts.items():
        if source.exists():
            shutil.copy2(source, run_dir / name)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALIDATION_MODES, default="spatial")
    parser.add_argument(
        "--dataset", choices=tuple(DATASETS), default="multi_crop_seasonal"
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    path, prefix = DATASETS[args.dataset]
    if not path.exists():
        raise SystemExit(
            f"final dataset not found: {path}; run scripts/build_final_dataset.py"
        )
    data = pd.read_csv(path)
    if data["target_temporal_granularity"].nunique() != 1:
        raise SystemExit("Primary modeling data mix annual and seasonal targets.")

    cols = resolve_feature_columns(data)
    features = cols["climate"] + cols["static"] + cols["categorical"]
    missingness_path = TABLES / f"{prefix}missingness_report.csv"
    pca_path = TABLES / f"{prefix}pca_diagnostics.csv"
    feature_timing_path = TABLES / f"{prefix}feature_timing.csv"
    missingness_report(data).reset_index().rename(columns={"index": "column"}).to_csv(
        missingness_path, index=False
    )
    build_pca_report(data).to_csv(pca_path, index=False)
    feature_timing_table(cols).to_csv(feature_timing_path, index=False)

    try:
        splits, inner_mode = outer_splits(data, args.mode, args.random_seed)
    except ValueError as exc:
        log.warning("%s validation unavailable: %s", args.mode, exc)
        manifest = {
            "dataset": path.name,
            "validation_scheme": args.mode,
            "analysis_status": "unavailable",
            "reason": str(exc),
        }
        (TABLES / f"{prefix}analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return 0

    registry = get_model_registry(
        random_seed=args.random_seed,
        load_optional=not args.quick,
    )
    if args.models:
        unknown = sorted(set(args.models).difference(registry))
        if unknown:
            raise SystemExit(f"Unknown models: {unknown}")
        registry = {name: registry[name] for name in args.models}
    elif args.quick:
        registry = {
            name: spec
            for name, spec in registry.items()
            if name in {"dummy_mean", "ols", "ridge"}
        }

    predictions, fold_results, permutation, execution_status = [], [], [], []
    for target_scale in ("raw", "crop_centered"):
        for space in ("raw", "pca", "hybrid"):
            for name, spec in registry.items():
                try:
                    pred, folds = run_nested_evaluation(
                        data,
                        feature_space=space,
                        model_name=name,
                        model_spec=spec,
                        outer_splits=splits,
                        inner_mode=inner_mode,
                        random_seed=args.random_seed,
                        target_scale=target_scale,
                    )
                    predictions.append(pred)
                    fold_results.append(folds)
                    execution_status.append(
                        {
                            "model": name,
                            "feature_space": space,
                            "target_scale": target_scale,
                            "status": "completed",
                            "reason": "",
                        }
                    )
                    pred.to_csv(
                        TABLES
                        / f"{prefix}{args.mode}_{name}_{space}_{target_scale}_predictions.csv",
                        index=False,
                    )
                    if target_scale == "raw":
                        importance = heldout_permutation_diagnostics(
                            data,
                            feature_space=space,
                            model_spec=spec,
                            outer_splits=splits,
                            random_seed=args.random_seed,
                        )
                        importance["model"] = name
                        importance["feature_space"] = space
                        permutation.append(importance)
                except Exception as exc:  # noqa: BLE001 - checkpoint every model failure
                    log.warning(
                        "checkpoint skipped for %s/%s/%s: %s",
                        name,
                        space,
                        target_scale,
                        exc,
                    )
                    execution_status.append(
                        {
                            "model": name,
                            "feature_space": space,
                            "target_scale": target_scale,
                            "status": "unavailable",
                            "reason": str(exc),
                        }
                    )
    if not predictions:
        raise SystemExit("No model/feature-space evaluation completed")

    model_predictions = pd.concat(predictions, ignore_index=True)
    baseline, fallback = baseline_predictions(data, splits, args.mode)
    all_predictions = pd.concat(
        [model_predictions, baseline], ignore_index=True, sort=False
    )
    overall = summarize_out_of_fold_predictions(all_predictions)
    per_crop = crop_metrics(all_predictions)
    summary = pd.concat(
        [overall, per_crop, macro_average(per_crop)], ignore_index=True, sort=False
    )
    fold = pd.concat(fold_results, ignore_index=True)
    coverage = summarize_conformal_coverage(model_predictions)

    predictions_path = TABLES / f"{prefix}{args.mode}_predictions.csv"
    comparison_path = TABLES / f"{prefix}{args.mode}_model_comparison.csv"
    fold_path = TABLES / f"{prefix}{args.mode}_fold_results.csv"
    coverage_path = TABLES / f"{prefix}{args.mode}_conformal_coverage.csv"
    execution_path = TABLES / f"{prefix}{args.mode}_execution_status.csv"
    permutation_path = TABLES / f"{prefix}heldout_permutation_importance.csv"
    all_predictions.to_csv(predictions_path, index=False)
    summary.to_csv(comparison_path, index=False)
    fold.to_csv(fold_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(execution_status).to_csv(execution_path, index=False)
    if permutation:
        pd.concat(permutation, ignore_index=True).to_csv(permutation_path, index=False)

    raw_diagnostics = model_predictions[model_predictions["target_scale"].eq("raw")]
    feat = cols["climate"] + cols["static"]
    try:
        flags = robust_mahalanobis_flags(data, feat).reindex(data.index)
        flags.insert(0, "row_index", flags.index)
        flags.to_csv(TABLES / f"{prefix}outlier_flags.csv", index=False)
    except ValueError as exc:
        pd.DataFrame({"status": ["unavailable"], "reason": [str(exc)]}).to_csv(
            TABLES / f"{prefix}outlier_flags.csv", index=False
        )
    residual_diagnostics(
        raw_diagnostics.observed_yield, raw_diagnostics.predicted_yield
    ).to_csv(TABLES / f"{prefix}residual_diagnostics.csv", index=False)
    model_agreement(raw_diagnostics).to_csv(
        TABLES / f"{prefix}model_agreement.csv", index=False
    )

    wave_checks = [
        validate_wave_source(p)
        for p in sorted(Path("data/raw").glob("*2015-2021*.xlsx"))
    ]
    wave_checks += [
        validate_wave_source(p) for p in sorted(Path(".").glob("UGA-UBOS-AAS-*.xml"))
    ]
    wave_checks.append(validate_wave_source(Path("data/raw/AAS2019.pdf")))
    gate_passed, gate_reason = _acceptance_gate(data, features)
    git_commit, dirty = _git_metadata()
    generated_at = datetime.now(UTC)
    run_id = (
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{args.dataset}-{args.mode}-{_sha256(path)[:8]}"
    )
    status_frame = pd.DataFrame(execution_status)
    environment_count = (
        data[["spatial_unit", "year", "season"]].drop_duplicates().shape[0]
    )
    manifest = {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "git_commit": git_commit,
        "working_tree_dirty": dirty,
        "dataset": path.name,
        "dataset_sha256": _sha256(path),
        "config_sha256": _config_sha256(),
        "package_version": __version__,
        "Python_version": platform.python_version(),
        "random_seed": args.random_seed,
        "row_count": len(data),
        "crop_environment_unit_count": len(data),
        "environmental_unit_count": environment_count,
        "independent_environmental_unit_count": environment_count,
        "spatial_unit_count": int(data.spatial_unit.nunique()),
        "years": sorted(map(int, data.year.unique())),
        "models_requested": sorted(registry),
        "models_completed": int(status_frame["status"].eq("completed").sum()),
        "models_failed": int(status_frame["status"].ne("completed").sum()),
        "validation_scheme": args.mode,
        "conformal_alpha": 0.10,
        "calibration_scheme": "held_out_spatial_groups_within_outer_training",
        "metric_ci_method": "spatial_unit_cluster_bootstrap",
        "metric_ci_iterations": 200,
        "target_temporal_granularity": data["target_temporal_granularity"].iloc[0],
        "analysis_status": "final" if gate_passed else "interim_spatial_only",
        "final_acceptance_gate_passed": gate_passed,
        "acceptance_gate_reason": gate_reason,
        "prediction_horizon": "season_end_retrospective",
        "feature_timing": feature_timing_path.name,
        "baseline_fallbacks": fallback.to_dict(orient="records"),
        "execution_status": execution_status,
        "additional_wave_validation": wave_checks,
    }
    manifest_path = TABLES / f"{prefix}analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    report_path = TABLES.parent / "technical_report" / "interim_report.md"
    from make_report import write_interim_report

    write_interim_report(comparison_path, manifest_path, report_path)
    run_dir = _write_run_bundle(
        run_id=run_id,
        manifest=manifest,
        artifacts={
            "model_comparison.csv": comparison_path,
            "predictions.csv": predictions_path,
            "fold_results.csv": fold_path,
            "conformal_coverage.csv": coverage_path,
            "pca_diagnostics.csv": pca_path,
            "report.md": report_path,
        },
    )
    print(summary.to_string(index=False))
    print(f"Immutable run bundle: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
