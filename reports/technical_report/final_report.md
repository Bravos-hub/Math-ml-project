# Uganda crop-yield prediction technical report

This report summarizes held-out validation results. It is descriptive and does not establish causality.

## Validation results

- **group_by_subregion**: best held-out result was pca/XGBoost (RMSE 0.574, R² 0.679).
- **random_cv**: best held-out result was raw/RandomForest (RMSE 0.613, R² 0.634).
- **temporal_2018_2020**: best held-out result was pca/XGBoost (RMSE 0.717, R² 0.570).
- **temporal_2020_2018**: best held-out result was raw/XGBoost (RMSE 0.430, R² 0.585).

## Interpretation limits

The pooled panel contains repeated crop, season, and subregion structure. The representation comparison supplies the same crop and season context to raw, PCA, and hybrid spaces. Results are not causal effects and should not be described as production-ready forecasts.

See `reports/tables/model_agreement.csv`, `residual_diagnostics.csv`, `vif.csv`, and `survey_uncertainty_sensitivity.csv` for supplementary diagnostics.
