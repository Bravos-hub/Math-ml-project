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

In the pooled panel, the **hybrid** representation (PC scores + one-hot
crop indicator) is the only model set that beats the mean baseline under
grouped-by-subregion CV: XGBoost R2 = 0.65 (grouped) / 0.60 (random CV);
the raw and PCA representations do not beat the mean baseline.

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
| `data/processed/observed/crop_pooled_subregion_panel.csv` | 167 (all crops) | subregion x crop x season x year | A | pooled panel with `crop` as a hybrid predictor |

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

### Feature representations

`make models` compares three feature representations at the modeling step:

1. **Raw** — cleaned predictors used directly.
2. **PCA-reduced** — retained principal-component scores (correlation-matrix PCA).
3. **Hybrid** — PC scores plus contextual variables not in the PCA
   transformation (e.g. one-hot `crop` in the pooled panel).

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

## Installation

```bash
git clone https://github.com/Bravos-hub/Math-ml-project.git
cd Math-ml-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or make setup
```

---

## Running the pipeline

```bash
make data                    # build all real-data panels (grade A/B)
make validate                # run pytest contract tests (35 tests)
make pca                     # PCA retention + stability analysis
make models                  # model comparison (raw/pca/hybrid, grouped CV)
make figures
make report
make all
```

`make data` -> `scripts/build_panels.py` (observed/assigned panels plus the
pooled sub-region x crop panel).

`make models` -> `scripts/run_models.py`. Default scope is the pooled
panel x `full_agroecological` features x raw/pca/hybrid representations
(~2 min). Add `--full` for all crops x feature sets (raw).

Check input-path constants and confirm required source tables exist first.

---

## Reproducibility requirements

A reproducible run should record source URLs and retrieval dates, dataset
versions, script/commit version, unit-conversion rules, district/subregion
mapping rules, missing-data decisions, standardization parameters,
PCA feature list and retained components, CV folds and random seeds,
model hyperparameters, output paths.

Automated checks in `tests/` include:

- Unique subregion-season-year keys
- Non-negative crop yield
- Positive harvested area where yield is computed
- No target leakage into PCA
- Symmetric covariance matrix with non-negative eigenvalues
- Agreement between manual eigenvalues and `numpy.linalg.eigh()`
- No overlap between grouped training and test entities

---

## Known limitations

1. Based on secondary data, not field-validated.
2. The 5-row benchmark is a smoke test, not a generalization claim.
3. AAS yield statistics are subregional and assigned to districts; they are
   not independent district-level attributes.
4. AAS 2018 annual rows publish total production with second-season
   harvested area; kept with an explicit `yield_consistency_ok` flag.
5. Static soil values across years do not reflect annual change.
6. PCA maximizes explained variance, not yield-prediction accuracy.
7. The framework supports association and prediction, not causal inference.
8. Outputs are not suitable for operational farm recommendations at this stage.

---

## Research ethics and responsible use

This phase uses secondary data only; no farmer recruitment, interviews, or
field measurements. When using restricted data: follow the provider's access
conditions, do not publish direct identifiers or protected coordinates,
keep restricted sources outside the public repository, and clearly
distinguish observed, derived, proxy, and synthetic values.

---

## Citation

```text
Olimi, B. (2026). Uganda Crop-Yield Prediction with PCA and Machine Learning. GitHub repository.
```

---

## License

MIT — see `LICENSE`. Data remain the property of their respective providers
(UBOS, CHC/UCSB, NASA, Copernicus, ISRIC).