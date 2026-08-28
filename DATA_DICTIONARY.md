# Canonical Data Dictionary

Project: *PCA-Based Machine Learning for Predicting Selected Food-Crop Yields in Uganda*

Canonical pipeline: `uganda_crop_model`
Last reviewed: 27 August 2026

## Scope and analytical grain

The primary homogeneous analysis file is `data/processed/final_multi_crop_seasonal.csv`. Each row is one observed `spatial_unit × year × season × crop` combination. The independent environmental unit is `spatial_unit × year × season`; crop rows sharing that key are dependent because they share environmental predictors.

The current seasonal file contains 249 crop-environment rows arising from 28 independent environments, 14 UBOS subregions, and one year (2020). It is an interim spatial, season-end retrospective dataset—not evidence of future-year forecasting. Annual and mixed-granularity files are audit/sensitivity artifacts and must not be combined with seasonal targets in a primary run.

No proxy, synthetic, geographically assigned, district-level, or household-microdata target is permitted in the authoritative analysis. Legacy scripts live under `scripts/legacy/` and are not part of this schema.

## Keys, target, and provenance controls

| Field | Type / unit | Role | Definition |
|---|---|---|---|
| `spatial_unit` | string | key/group | One of 14 UBOS AAS subregions. |
| `year` | integer | key/time | Target survey year. |
| `season` | string | key/context | UBOS-compatible season label. |
| `crop` | string | key/context | Published crop category. |
| `yield_tons_ha` | float, t/ha | target | Official aggregate yield from the matched UBOS target table. |
| `target_temporal_granularity` | enum | contract | `seasonal` or `annual`; exactly one value per primary dataset. |
| `target_source_type` | enum | contract | Must be an allowed observed/official aggregate type. |
| `target_geographic_level` | string | contract | Geographic grain of the target. |
| `predictor_geographic_level` | string | contract | Must match target geography. |
| `is_proxy` | boolean | exclusion | Must be false. |
| `is_synthetic` | boolean | exclusion | Must be false. |
| `is_geographically_assigned` | boolean | exclusion | Must be false. |

Production, planted area, harvested area, and yield-derived totals are provenance/audit fields only and are prohibited as predictors by the final-data gate.

## Environmental predictors

The exact columns are frozen in each run's dataset hash and selected by `resolve_feature_columns`.

| Prefix / field | Source | Meaning and units |
|---|---|---|
| `rain_*`, `daily_*` | CHIRPS v2.0 / ClimateSERV | Seasonal rainfall totals (mm), rainy-day counts, dry spells, extremes, and anomalies. |
| `temp_*` | NASA POWER | Seasonal temperature (°C), growing degree-days, heat days, warm nights, and heatwaves. |
| `soil_*` | SoilGrids v2.0 | Subregion aggregates and within-subregion variability for texture, SOC, bulk density, CEC, and pH over documented depths. |
| `season`, `crop` | UBOS/context | Categorical variables retained outside environmental PCA. |

Climate fields use observations through season end, so every output using them must retain `season_end_retrospective`. Soil variables are static or slow-changing. Missingness and minimum feature coverage are checked before a run can pass the final-data gate.

## PCA and modeling roles

Descriptive PCA gives each unique `spatial_unit × year × season` environment equal weight. Predictive PCA, scaling, imputation, and categorical encoding are fit inside training folds. PCA components are statistical latent axes and must not be described as causal agronomic mechanisms.

Headline tables must identify result scope: pooled raw, macro-average, crop-specific, or crop-centered sensitivity. A bare “R² = X” is not a complete result description.

## Canonical files

| File | Status / grain |
|---|---|
| `final_multi_crop_seasonal.csv` | Primary interim seasonal analysis; subregion × 2020 season × crop. |
| `final_multi_crop_annual.csv` | Separate annual sensitivity artifact; never mixed with seasonal targets. |
| `final_multi_crop_subregion_season_year.csv` | Combined audit artifact; not a primary modeling input. |
| `final_maize_subregion_season_year.csv` | Supplementary small-sample maize artifact. |

See `DATA_GOVERNANCE.md` for source terms and `ETHICS_AND_DATA_GOVERNANCE.md` for privacy, institutional-review, and re-review conditions.
