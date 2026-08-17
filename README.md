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
# Uganda Crop Yield Modeling with PCA and Machine Learning

A reproducible agricultural data-science project for building, analyzing, and
evaluating crop-yield prediction pipelines for Uganda using official
agricultural statistics, climate observations, and soil covariates.

The repository focuses on subregion-level observed yields (14 UBOS AAS
subregions, crops: maize, beans, groundnuts) built from official UBOS
Annual Agricultural Survey statistics (AAS 2018, AAS 2020), CHIRPS
rainfall, NASA POWER temperature, C3S soil-moisture, and ISRIC SoilGrids
soil properties. It combines data engineering, exploratory analysis,
Principal Component Analysis (PCA), and supervised machine-learning
evaluation with grouped (geographic) and temporal validation.

> **Research status:** active undergraduate proof of concept. The current
> benchmarks are based on secondary data and are not yet field-validated
> or suitable for operational farmer-level decision-making.
> This study developed and evaluated a reproducible, leakage-controlled framework for crop-yield prediction using geographically matched agricultural and environmental data from Ugandan subregions. All preprocessing transformations were fitted within training folds, geographic validation isolated subregions between training and test data, and temporal split analyses isolated observations across survey years where the available data permitted. Raw, PCA-reduced, and hybrid environmental representations were compared across linear and tree-based models against transparent baseline predictors.
> The results demonstrate that the framework can support auditable undergraduate research on crop-yield prediction, but they do not establish production-ready forecasting or broad temporal generalization. The evidence is limited by the small number of survey years, repeated subregion structure, uneven seasonal definitions between 2018 and 2020, and the absence of a larger independent validation sample. The contribution is therefore methodological and empirical: it shows how a small agricultural dataset can be analyzed transparently, with explicit leakage controls, provenance, uncertainty diagnostics, and honest acceptance criteria.

### Current authoritative conclusion

The authoritative evidence set is the 373-row multi-crop subregion × crop ×
season × year panel. SoilGrids district-centroid observations are aggregated
to subregion means and within-subregion standard deviations, so raw, PCA, and
hybrid representations are genuinely distinct. Raw tonnes/ha remains the
primary target; log1p and fold-local crop-normalized analyses are sensitivities.

The pipeline now exposes spatial, LOSO, temporal, and future-unseen-location
stress modes and writes checkpointed out-of-fold predictions, training-only
baselines, conformal coverage, outlier, residual, and held-out permutation
diagnostics. This supports a defensible undergraduate research contribution
in data integration, leakage-controlled validation, representation comparison,
and transparent uncertainty—not operational or causal predictability.

The validated panel contains only 2018 and 2020, so temporal and stress
conclusions remain explicitly limited. AAS2019 is unparseable and no validated
local elevation source exists; neither is synthesized. Maize is separately
labeled as a 42-row small-sample analysis.

---

## Project objectives

The study is organized around three objectives:

1. **Prepare the data:** acquire, integrate, clean, and explore publicly
   available Ugandan agricultural and climate data suitable for crop-yield
   analysis.
2. **Design and interpret PCA:** mathematically verify PCA and use it to
   identify latent climatic, thermal, and soil-related factors.
3. **Evaluate predictive models:** compare crop-yield models trained on
   raw features, PCA-reduced features, and hybrid feature representations.

The corresponding research questions are:

1. How can publicly available and climate datasets be integrated and
   prepared for crop-yield analysis in Uganda?
2. What latent agroclimatic factors can be identified and interpreted
   using PCA?
3. How does predictive perform and differ across algorithms and feature
   representations under honest (geographic/temporal) validation?

---

## Scientific validity and status of results

The project separates research stages explicitly (observed / assigned /
proxy / synthetic data, grades A-F) and records provenance for every
variable group. Two rules govern interpretation:

- The five-row 2020 Eastern-Uganda benchmark is a pipeline smoke test
  (`pipeline_smoke_test_2020`, `evaluation_status = "demonstration_only"`).
  With n=5 rows and ~3 distinct pseudo-replicated targets it is NOT
  evidence of district-level predictability.
- Formal model comparison runs on the honest subregion-level samples —
  56 rows per crop and 167 rows for the pooled subregion x crop panel —
  under random and grouped-by-subregion CV plus both temporal splits.
  Every put-off run is recorded in `reports/tables/experiment_registry.csv`
  (run id, git commit, sample size, number of unique targets, feature
  set, representation, small-sample warnings).

The corrected pooled-panel comparison gives every representation the same
context variables (`crop` and `season_group`), encoded within each training
fold. Under grouped-by-subregion CV, the best result is XGBoost with PCA and
hybrid representations (identical here because no static soil variables are
available): RMSE = 0.574 and R2 = 0.679. Raw PCR is close behind (R2 = 0.641).
Under random CV, raw Random Forest is best (RMSE = 0.613, R2 = 0.634), while
PCA/hybrid Random Forest reaches R2 = 0.624. The mean baseline has grouped
R2 = -0.020 and random-CV R2 = -0.015. Therefore, the evidence does not
support the claim that hybrid is the only representation that generalizes.

---

## Data sources

| Source | Contribution | Current role |
|---|---|---|
| UBOS AAS 2018 (PDF), AAS 2020 (Excel annex) | Planted/harvested area, production, yield, coefficients of variation | Official crop-yield source (14 subregions, 2018 & 2020) |
| CHIRPS v2.0 | Rainfall seasonal, monthly, daily | Real climate predictor source |
| NASA POWER (MERRA-2) | Daily T2M_MAX/T2M_MIN, growing degree days, thermal stress | Thermal predictor layer |
| C3S SOILMOISTURE (Copernicus) | Soil moisture | Climate-predictor layer |
| ISRIC SoilGrids v2.0 | Clay, sand, silt, SOC, bulk density, CEC, pH (0-30 cm) | Agroecological predictor layer |
| geoBoundaries OCHA ADM2 | District names and centroids geometry | Geographic covariate layer |

`reports/tables/feature_availability.csv` records `Variable / Source /
Coverage / Status` for every modeling column, and the soil-loading step
refuses to build a panel unless at least 80% of soil columns present
(`MIN_SOIL_COVERAGE`).

---

## Repository layout

```
configs/            data.yaml, features.yaml, models.yaml
data/
  raw/              original downloads: UBOS, CHIRPS NetCDF, API caches
  external/         third-party datasets (C3S soil moisture, SoilGrids caches)
  interim/          partially processed feature and yield tables
  processed/
    observed/       official survey estimates at the unit of analysis (grade A)
    assigned/       official values assigned from larger areas (grade B)
    proxy/          proxy-based tables (grade D)
    synthetic/      synthetic/demonstration tables (grade E)
  public/           model-ready files intended for sharing
src/cropyield/      the pipeline package (data, features, pca, models, reporting)
scripts/            pipeline entrypoints; legacy/ holds deprecated scripts
tests/              data-quality and contract tests (pytest)
notebooks/          PCA math proof and verification notebooks
reports/            figures/ and tables/, experiments registry
```

Package ownership: `uganda_crop_model` is the authoritative final-analysis
pipeline and source of truth for published results. `cropyield` is retained
as the legacy/data-engineering and exploratory compatibility layer, including
the layout-adaptive UBOS PDF parser used to validate candidate survey waves.

## Data classes and quality grades

Every modeling table carries provenance columns: `yield_source`,
`rainfall_source`, `temperature_source`, `soil_source`,
`soil_moisture_source`, `yield_granularity`, `is_proxy`, `is_imputed`,
`data_quality_score`, `data_quality_note`.

| Grade | Meaning |
|---|---|
| A | Direct observed at the matching district/subregion-year level (official estimate) |
| B | Official value assigned from a larger area (subregion -> district) |
| C | Derived from official totals |
| D | Proxy value (station or satellite proxy) |
| E | Synthetic or demonstration value (pipeline testing only) |
| F | Failed or missing extraction — never used |

---

## Panels

| File | Rows (maize) | Grain | Grade | Content |
|---|---|---|---|---|
| `data/processed/observed/maize_subregion_panel.csv` | 56 | subregion x 2020 (3 seasons) + 2018 annual | A | AAS 2018 + 2020 official estimates + CHIRPS/POWER climate |
| `data/processed/assigned/maize_district_assigned_panel.csv` | 456 | district x 2020 (3 seasons) + 2018 annual | B | subregion values assigned to 114 districts |
| `data/processed/observed/crop_pooled_subregion_panel.csv` | 167 (all crops) | subregion x crop x season x year | A | pooled panel with crop and season context predictors |

Targets: `yield_over_harvested` (primary, t/ha = production / harvested
area), `yield_over_planted` (sensitivity), `production_mt`,
`area_harvested_ha`, plus ~46 climate features (`rain_*`, `daily_*`,
`temp_*`, `soil_*`).

---

## Validation design

Out-of-sample validation runs on the honest subregion-level samples with
two schemes — random CV and grouped-by-subregion CV (geographic
generalization), plus both temporal splits (train 2018 / test 2020 and
vice versa). Baselines are mean predictor, historical mean, and
previous-year yield; conformal prediction intervals are reported. Results
in `reports/tables/validation_all.csv`.

Run `make diagnostics` for held-out residuals, model agreement, complete-case
and survey-uncertainty sensitivity tables, season-window sensitivity, VIFs,
and held-out permutation importance. `make figures` and `make report` generate
the reproducible figures and conservative technical report under `reports/`.

### Feature representations

`make models` compares three feature representations at the modeling step:

1. **Raw** — cleaned predictors used directly.
2. **PCA-reduced** — retained principal-component scores (correlation-matrix PCA).
3. **Hybrid** — climate PC scores plus original static variables and the
   same contextual variables held outside PCA.

For a fair comparison, all three representations receive the same contextual
variables (`crop` and `season_group` in the pooled panel); only the continuous
environmental representation changes. In this milestone the hybrid and PCA
spaces coincide because static soil variables are unavailable.

All preprocessing (imputation, scaling, PCA fitting) is done inside each
training fold only, to avoid leakage. Model matrix: OLS, Ridge, PCR, PLS,
Random Forest, XGBoost + 3 baselines in `scripts/run_models.py`.

### Experiment registry

Each model run records an experiment row in `reports/tables/experiment_registry.csv`:
`run_id`, timestamp, git commit, sample size, number of unique targets,
`n_features`, `feature_set`, representation, and small-sample warnings
(sample < 100 or too few unique targets).

---

## PCA and representations

`make pca` runs a rigorous PCA of the 65-feature district-year climate matrix
(1019 rows): correlation-matrix PCA with covariance comparison, component
retention by Kaiser, cumulative variance (0.85), and parallel analysis
(95th percentile), plus bootstrap confidence intervals for eigenvalues and
loadings. Results in `reports/tables/pca_v2_*` and `reports/figures/`. The
underlying SVD is verified against a hand-computed eigendecomposition and
sklearn in `notebooks/01_pca_math.ipynb`.

---

## Final-mode milestone dataset (sub-region season-year)

The project is moving its scientific evaluation to a dedicated final-analysis
pipeline in `src/uganda_crop_model/`.  The analytical grain is

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
