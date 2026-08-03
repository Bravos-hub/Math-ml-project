# Interim technical report: selected food-crop yields in Uganda

This report is generated from the leakage-controlled authoritative run. It describes retrospective secondary-data yield estimation, not an early-warning forecast or causal analysis.

## Best overall model

- **ridge / pca**: RMSE 3.338, MAE 1.413, R² 0.128 (n=249).

## Training-derived baselines

- **training_crop_season_mean / baseline**: RMSE 3.221, MAE 1.032, R² 0.188 (n=249).
- **training_crop_mean / baseline**: RMSE 3.248, MAE 1.140, R² 0.174 (n=249).
- **training_global_mean / baseline**: RMSE 3.590, MAE 1.772, R² -0.009 (n=249).

## Per-crop diagnostic performance

- **beans**: best training_crop_season_mean / baseline had RMSE 0.182, MAE 0.151, and R² 0.448.
- **groundnuts**: best training_crop_season_mean / baseline had RMSE 0.179, MAE 0.130, and R² 0.087.
- **irish_potatoes**: best ridge / pca had RMSE 1.440, MAE 1.152, and R² 0.574.
- **maize**: best training_crop_season_mean / baseline had RMSE 1.198, MAE 0.856, and R² 0.164.
- **millet**: best training_crop_season_mean / baseline had RMSE 0.412, MAE 0.289, and R² -0.108.
- **rice**: best training_crop_mean / baseline had RMSE 0.754, MAE 0.520, and R² -0.217.
- **simsim**: best training_crop_mean / baseline had RMSE 0.231, MAE 0.154, and R² -0.019.
- **sorghum**: best training_crop_season_mean / baseline had RMSE 0.866, MAE 0.662, and R² 0.231.
- **soya_beans**: best training_global_mean / baseline had RMSE 8.609, MAE 3.015, and R² -0.011.
- **sweet_potatoes**: best training_crop_season_mean / baseline had RMSE 3.332, MAE 2.435, and R² 0.294.

## Acceptance gates

Final acceptance gates passed: **no**.
Insufficient independent environmental units: 28 < 50.

## Limitations

The validated source material contains only AAS 2018 and 2020. The primary seasonal analysis currently contains only 2020, while 2018 is annual and is analyzed separately. Temporal generalization therefore cannot be established. With only 14 subregions, group-held-out calibration sets are small; interval coverage is reported empirically and should be interpreted cautiously. Pooled performance can conceal negative crop-specific R² values, so it is not evidence of uniformly successful prediction for every crop.
