from pathlib import Path
import pandas as pd
import pytest

from uganda_crop_model.data.subregion_soil import (
    build_subregion_soil_features, validate_district_soil,
)
from uganda_crop_model.validation.splits import leave_one_subregion_out_splits
from uganda_crop_model.evaluation.baselines import predict_training_baselines

ROOT = Path(__file__).resolve().parents[1]


def test_soil_source_has_complete_114_district_coverage():
    soil = pd.read_csv(ROOT / "data/interim/uganda_soil_features_114.csv")
    mapping = pd.read_csv(ROOT / "data/interim/uganda_districts_114.csv")
    validate_district_soil(soil, mapping)
    result = build_subregion_soil_features(
        ROOT / "data/interim/uganda_soil_features_114.csv",
        ROOT / "data/interim/uganda_districts_114.csv",
    )
    assert len(result) == 14
    assert result["soil_district_count"].sum() == 114
    assert result[["clay_pct", "sand_pct", "silt_pct"]].notna().all().all()
    assert result["soil_source"].eq("SoilGrids_v2.0").all()


def test_loso_is_spatially_disjoint():
    frame = pd.DataFrame({"spatial_unit": ["a", "a", "b", "b", "c", "c"]})
    for train, test in leave_one_subregion_out_splits(frame):
        assert set(frame.iloc[train].spatial_unit).isdisjoint(frame.iloc[test].spatial_unit)


def test_crop_baseline_is_training_only_and_informative_under_spatial_holdout():
    frame = pd.DataFrame({
        "spatial_unit": ["a", "a", "b", "b"],
        "crop": ["maize", "beans", "maize", "beans"],
        "season": ["annual"] * 4,
        "yield_tons_ha": [1.0, 10.0, 3.0, 30.0],
    })
    result = predict_training_baselines(frame, [0, 1], [2, 3])
    crop_values, fallback_count = result["crop_mean"]
    assert fallback_count == 0
    assert crop_values.tolist() == [1.0, 10.0]
