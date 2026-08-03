"""Regression tests for the updated research and engineering audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from uganda_crop_model.evaluation.baselines import (
    predict_training_baselines,
    previous_available_wave,
)
from uganda_crop_model.models import build_preprocessor, get_model_registry

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_selects_only_registered_overall_raw_results():
    report = _load_script("make_report.py")
    results = pd.DataFrame(
        [
            {
                "model": "ridge",
                "feature_space": "pca",
                "rmse": 1.0,
                "mae": 0.8,
                "result_scope": "overall",
                "target_scale": "raw",
                "registered_primary_metric": True,
            },
            {
                "model": "dummy_mean",
                "feature_space": "hybrid",
                "rmse": 0.1,
                "mae": 0.1,
                "result_scope": "per_crop",
                "target_scale": "raw",
                "registered_primary_metric": False,
            },
            {
                "model": "ols",
                "feature_space": "pca",
                "rmse": 0.5,
                "mae": 0.4,
                "result_scope": "overall",
                "target_scale": "crop_centered",
                "registered_primary_metric": False,
            },
        ]
    )
    selected = report.select_primary_results(results)
    assert selected[["model", "feature_space"]].values.tolist() == [["ridge", "pca"]]


def test_spatial_baselines_exclude_inapplicable_subregion_history():
    frame = pd.DataFrame(
        {
            "spatial_unit": ["a", "a", "b", "b"],
            "crop": ["maize", "beans", "maize", "beans"],
            "season": ["first_season"] * 4,
            "year": [2018, 2018, 2020, 2020],
            "yield_tons_ha": [1.0, 10.0, 3.0, 30.0],
        }
    )
    result = predict_training_baselines(
        frame, [0, 1], [2, 3], validation_mode="spatial"
    )
    assert set(result) == {
        "training_global_mean",
        "training_crop_mean",
        "training_crop_season_mean",
    }
    assert not result["training_crop_mean"].fallback_used.any()


def test_previous_wave_uses_closest_strictly_earlier_year():
    train = pd.DataFrame(
        {
            "spatial_unit": ["a", "a"],
            "crop": ["maize", "maize"],
            "season": ["first_season", "first_season"],
            "year": [2018, 2020],
            "yield_tons_ha": [1.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "spatial_unit": ["a"],
            "crop": ["maize"],
            "season": ["first_season"],
            "year": [2021],
            "yield_tons_ha": [3.0],
        }
    )
    output = previous_available_wave(train, test)
    assert output.values.tolist() == [2.0]
    assert output.baseline_applicable.tolist() == [True]
    assert output.fallback_level.tolist() == ["none"]


def test_public_model_api_is_callable():
    assert callable(build_preprocessor)
    assert callable(get_model_registry)


def test_descriptive_pca_frame_deduplicates_crop_repetitions():
    runner = _load_script("run_final_analysis.py")
    data = pd.DataFrame(
        {
            "spatial_unit": ["a", "a", "b"],
            "year": [2020, 2020, 2020],
            "season": ["first", "first", "first"],
            "crop": ["maize", "beans", "maize"],
            "rain_total_mm": [10.0, 10.0, 20.0],
        }
    )
    result = runner.environmental_pca_frame(data, ["rain_total_mm"])
    assert len(result) == 2


def test_accepted_manifest_matches_dataset_hash_when_present():
    manifest_path = ROOT / "reports" / "runs" / "accepted" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("No run has passed final acceptance gates yet.")
    manifest = json.loads(manifest_path.read_text())
    dataset = ROOT / "data" / "processed" / manifest["dataset"]
    assert (
        hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert manifest["git_commit"] == commit
