# PCA-Based Machine Learning for Predicting Selected Food-Crop Yields in Uganda

This repository implements a reproducible, leakage-controlled framework for
retrospective crop-yield estimation using official Uganda Bureau of Statistics
Annual Agricultural Survey (AAS) targets, climate observations, and soil
covariates. The primary research scope is ten selected food crops; maize-only
work is retained as a small-sample supplementary analysis.

> **Research status:** interim spatial-validation framework. It is not an
> operational forecast, early-warning system, causal study, or farmer-level
> decision tool.

## Current authoritative conclusion

The project has built a geographically aligned multi-crop secondary dataset
and evaluates raw, PCA, and hybrid representations under spatially separated
training and test folds. Preprocessing, PCA, hyperparameter selection, target
centering, and uncertainty calibration are fitted without outer-test outcomes.

The defensible current conclusion is methodological: PCA-based Ridge models
show limited pooled spatial skill, crop-level performance varies substantially,
and simple training-derived crop/season baselines can outperform the fitted
models. The validated source material contains only 2018 and 2020, and those
waves have incompatible target granularity (2018 annual; 2020 seasonal).
Temporal generalization has therefore not been established.

Changing experimental metrics are not hardcoded here. See the generated
[interim technical report](reports/technical_report/interim_report.md) and its
immutable run bundle under `reports/runs/<run_id>/`. A final accepted pointer at
`reports/runs/accepted/report.md` must not be created until every declared
acceptance gate passes.

## Research questions

1. How can official agricultural, climate, and soil data be integrated at a
   consistent geographic and temporal grain for selected Ugandan food crops?
2. What latent agroclimatic factors can descriptive PCA identify when each
   independent environment receives equal weight?
3. How do raw, PCA, and hybrid representations compare under leakage-controlled
   geographic validation?
4. Do environmental predictors explain within-crop yield deviations beyond
   training-derived global, crop, and crop-season baselines?

## Authoritative data products

| Dataset | Role | Current size |
|---|---|---:|
| `data/processed/final_multi_crop_seasonal.csv` | Primary homogeneous spatial analysis | 249 crop-environment rows; 28 independent environments |
| `data/processed/final_multi_crop_annual.csv` | Separate annual analysis | 124 crop-environment rows; 14 independent environments |
| `data/processed/final_multi_crop_subregion_season_year.csv` | Combined audit artifact; never a primary modeling input | 373 rows |
| `data/processed/final_maize_subregion_season_year.csv` | Supplementary maize small-sample artifact | 42 rows |

The ten crops are maize, beans, groundnuts, sorghum, millet, rice, soya beans,
simsim, Irish potatoes, and sweet potatoes. Every row records target and
predictor geography, source type, proxy/synthetic/assignment flags, target
definition, temporal granularity, and processing version.

The final quality gate checks both crop-environment row count and independent
`spatial_unit × year × season` environments. It also rejects mixed annual and
seasonal targets, target-derived predictors, geographic mismatch, proxy or
synthetic outcomes, inadequate coverage, and insufficient years or spatial
units. The current data fail final acceptance because they provide only 28
seasonal and 14 annual independent environments, with one comparable year in
each homogeneous dataset.

## Validation and uncertainty design

The authoritative runner supports:

- grouped spatial cross-validation;
- leave-one-subregion-out validation;
- rolling-origin temporal validation when enough comparable years exist; and
- future-year/unseen-location stress validation when the data support it.

Within each outer spatial fold, subregions are split into proper-training and
untouched calibration groups. Hyperparameters are selected using only
proper-training data; the selected estimator is fitted there, calibrated on
held-out groups, and then applied to the untouched outer test fold. Reported
uncertainty outputs include nominal and actual coverage, interval width,
coverage by crop and season, calibration sample size, and calibration-group
count.

Spatial comparisons use only baselines available for unseen subregions:

- training global mean;
- training crop mean; and
- training crop-season mean.

Historical subregion and exact previous-wave baselines are reserved for
temporal designs. Every baseline prediction records applicability, whether a
fallback was used, and its fallback level. Skill is measured against
training-derived predictions, never against a mean calculated from test
outcomes.

## Performance views

Every run produces four complementary views:

1. pooled raw-yield performance;
2. per-crop diagnostics;
3. macro-averaged crop performance; and
4. a fold-local crop-centered experiment that predicts deviations from each
   crop's proper-training mean before adding that mean back to yield forecasts.

Clustered bootstrap intervals for RMSE, MAE, and both training-derived skill
scores are included in the overall comparison table. Descriptive PCA is fitted
once per unique environmental observation; predictive PCA remains inside each
training fold.

## Prediction horizon

The current experiment is a **season-end retrospective yield-estimation
framework**. Static soil and categorical context are available pre-season, but
the climate predictors include complete-season rainfall, cessation, dry-spell,
temperature, and growing-degree-day summaries. Their declared timing is written
to `reports/tables/multi_crop_seasonal_feature_timing.csv` using
`prediction_horizon` and `feature_available_by` fields.

Pre-season, day-30, and mid-season forecasts remain future experiments and must
exclude features unavailable at those horizons.

## Reproducible outputs

Each completed authoritative run creates:

```text
reports/runs/<run_id>/
├── manifest.json
├── model_comparison.csv
├── predictions.csv
├── fold_results.csv
├── conformal_coverage.csv
├── pca_diagnostics.csv
└── report.md
```

The manifest records the Git commit and dirty state, dataset and configuration
hashes, package and Python versions, seed, row and environment counts, requested
and completed models, validation scheme, prediction horizon, and analysis
status. Run directories are immutable: an existing run ID is never overwritten.

## Installation and commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/build_final_dataset.py
pytest -q
python scripts/run_authoritative_analysis.py --models dummy_mean ridge
python scripts/make_report.py
```

The broader legacy pipeline remains available through the `Makefile`, but
published research claims should use `uganda_crop_model` and the authoritative
runner above. The install configuration intentionally excludes the legacy
`cropyield` package.

## Data sources

| Source | Contribution |
|---|---|
| UBOS AAS 2018 and AAS 2020 | Official subregion crop targets and survey uncertainty fields |
| CHIRPS / ClimateSERV | Daily rainfall and seasonal rainfall summaries |
| NASA POWER | Daily temperature, heat, and growing-degree-day summaries |
| ISRIC SoilGrids v2.0 | Subregion soil means and within-subregion variability |

AAS 2019 metadata are present, but no validated comparable target table has
been integrated. The next major scientific milestone is obtaining at least two
additional harmonized target years, not adding another algorithm.

## Known limitations

- Only 14 subregions are available, so calibration groups and uncertainty
  diagnostics remain small.
- Annual and seasonal survey targets cannot be pooled as though they measured
  the same temporal quantity.
- Crop-environment rows share environmental predictors within an environment
  and are not independent environmental replicates.
- Pooled accuracy can largely reflect between-crop yield-scale differences;
  per-crop, macro, and crop-centered results are mandatory.
- Soil covariates are static and do not represent annual soil change.
- The analysis uses secondary aggregate data and has no external field
  validation.

## Citation and license

Olimi, B. (2026). *PCA-Based Machine Learning for Predicting Selected Food-Crop
Yields in Uganda Using Secondary Agricultural and Climate Data*.

MIT — see [LICENSE](LICENSE). Source data remain subject to their providers'
terms.
