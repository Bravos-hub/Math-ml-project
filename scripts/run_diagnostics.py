#!/usr/bin/env python3
"""Generate non-causal interpretation and sensitivity tables.

All predictive diagnostics consume held-out predictions from the validation
matrix; no training-set performance is presented as evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uganda_crop_model.data.paths import FINAL_MULTI_CROP_DATASET, TABLES  # noqa: E402
from uganda_crop_model.data.paths import INTERIM, DISTRICT_MAP  # noqa: E402
from uganda_crop_model.data.season_calendar import make_sensitivity_calendars  # noqa: E402
from uganda_crop_model.features.rainfall import build_seasonal_rainfall_features  # noqa: E402
from uganda_crop_model.evaluation.interpretability import (  # noqa: E402
    heldout_permutation_importance,
    model_agreement,
    residual_diagnostics,
    variance_inflation_factors,
)
from uganda_crop_model.models.pipelines import build_preprocessor, resolve_feature_columns  # noqa: E402
from uganda_crop_model.validation.splits import spatial_group_splits  # noqa: E402
from uganda_crop_model.evaluation.metadata import build_quality_band  # noqa: E402
from uganda_crop_model.evaluation.sensitivity import (  # noqa: E402
    complete_case_columns,
    filter_uncertain_targets,
    season_window_sensitivity,
)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(TABLES / "validation_all_predictions.csv")
    predictions = predictions.rename(
        columns={"y_true": "observed_yield", "y_pred": "predicted_yield"}
    )
    model_agreement(predictions).to_csv(TABLES / "model_agreement.csv", index=False)
    residual_diagnostics(
        predictions["observed_yield"], predictions["predicted_yield"]
    ).to_csv(TABLES / "residual_diagnostics.csv", index=False)

    data = pd.read_csv(FINAL_MULTI_CROP_DATASET)
    numeric = data.select_dtypes(include="number").drop(
        columns=[c for c in ("yield_tons_ha", "year") if c in data], errors="ignore"
    )
    variance_inflation_factors(numeric).to_csv(TABLES / "vif.csv", index=False)

    feature_columns = [
        c for c in data.columns
        if c.startswith(("rain_", "temperature_", "temp_", "daily_", "soil_"))
    ]
    pd.DataFrame({
        "feature": feature_columns,
        "complete_case_coverage": [float(data[c].notna().mean()) for c in feature_columns],
        "complete_case_eligible": [c in complete_case_columns(data, feature_columns) for c in feature_columns],
    }).to_csv(TABLES / "complete_case_sensitivity.csv", index=False)

    if "cv_production_pct" in data:
        uncertain = filter_uncertain_targets(data)
        pd.DataFrame([{
            "rule": "cv_production_pct <= 30",
            "rows_retained": len(uncertain),
            "rows_excluded": len(data) - len(uncertain),
            "retention_fraction": len(uncertain) / len(data),
        }]).to_csv(TABLES / "survey_uncertainty_sensitivity.csv", index=False)

    if "cv_production_pct" in data:
        bands = build_quality_band(data).rename("quality_band").to_frame()
        bands.insert(0, "row", bands.index)
        bands.to_csv(TABLES / "survey_quality_bands.csv", index=False)

    resolved = resolve_feature_columns(data)
    predictors = resolved["climate"] + resolved["static"] + resolved["categorical"]
    train_index, test_index = next(iter(spatial_group_splits(data, requested_splits=5)))
    X_train, X_test = data.iloc[train_index][predictors], data.iloc[test_index][predictors]
    y_train, y_test = data.iloc[train_index]["yield_tons_ha"], data.iloc[test_index]["yield_tons_ha"]
    importance_rows = []
    # A single regularized reference model per space keeps this supplementary
    # diagnostic bounded; the utility itself supports any fitted estimator.
    for space in ("raw",):
        for name, estimator in (("ridge", Ridge(alpha=1.0)),):
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(space, feature_columns=resolved)),
                ("model", clone(estimator)),
            ])
            pipeline.fit(X_train, y_train)
            table = heldout_permutation_importance(
                pipeline, X_test, y_test, predictors
            )
            table.insert(0, "feature_space", space)
            table.insert(1, "model", name)
            importance_rows.append(table)
    pd.concat(importance_rows, ignore_index=True).to_csv(
        TABLES / "heldout_permutation_importance.csv", index=False
    )

    daily = pd.read_csv(INTERIM / "subregion_daily_rainfall.csv")
    calendars = list(make_sensitivity_calendars(DISTRICT_MAP).values())
    season_window_sensitivity(
        daily,
        calendars,
        build_seasonal_rainfall_features,
    ).to_csv(TABLES / "season_window_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
