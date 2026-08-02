# Changelog

All notable changes to this project are documented here, following the
principle that every experiment must be reproducible from a documented
data version.

## [Unreleased]

### Added (Phase 5 - review compliance)

- `data/processed/observed/crop_pooled_subregion_panel.csv`: cross-crop
  panel (167 rows) pooling maize/beans/groundnuts at the subregion x
  season x year level with `crop` as an explicit predictor
  (`scripts/build_panels.py`), giving 167 honest samples with 139 unique
  targets. Repeated subregion-cohort correlations (5-row vintage
  benchmark) are no longer the only evidence set.
- `src/cropyield/pca/representations.py`: `RepresentationTransformer`
  providing `raw`, `pca` and `hybrid` feature representations; PCA
  components are fit on the training fold only to avoid leakage, and the
  `hybrid` representation one-hot encodes the pooled `crop` column
  alongside PC scores (`configs/models.yaml` -> `pca.hybrid_extra`).
- `scripts/run_models.py`: driver that runs the model matrix across
  crops x feature sets x representations, rewrites
  `reports/tables/validation_all.csv` and appends to
  `reports/tables/experiment_registry.csv`. Default scope (pooled x
  `full_agroecological` x raw/pca/hybrid) runs in ~2 min; `--full`
  includes all crops x feature sets.
- `src/cropyield/reporting/feature_availability.py` +
  `reports/tables/feature_availability.csv`: Variable / Source / Coverage
  / Status per modeling column; soil loading is refused if < 80% of soil
  columns are present (`MIN_SOIL_COVERAGE`).
- Survey-uncertainty fields on panels (`add_reliability`):
  `target_cv`, `target_reliability_weight`, `high_uncertainty_flag`,
  `yield_consistency_ok`. AAS 2020 rows verify production/area equality;
  AAS 2018 annual rows are flagged rather than hidden.
- Experiment registry: each validation run records `run_id`, git commit,
  timestamp, sample size, number of unique targets, feature set,
  representation, and small-sample warnings
  (`sample_size < 100`, fewer unique targets) in
  `reports/tables/experiment_registry.csv`.
- Legacy 5-row benchmark renamed in its header/summary to
  `pipeline_smoke_test_2020` with `evaluation_status = "demonstration_only"`;
  it is explicitly not a performance benchmark. See
  `scripts/legacy/build_real_2020_benchmark.py`.

### Added (Phase 0 - scientific restructure)

- Repository reorganized into `data/`, `src/cropyield/`, `scripts/`,
  `configs/`, `tests/`, `notebooks/`, `reports/`.
- Data-provenance framework (`src/cropyield/data/provenance.py`):
  source, granularity, `is_proxy`, `is_imputed`, and A-F quality grades for
  every variable group (yield, rainfall, temperature, soil, soil moisture).
- Explicit separation of observed / assigned / proxy / synthetic datasets.
- Config-driven pipeline (`configs/*.yaml`), Makefile command interface,
  `pyproject.toml`, `requirements.txt`, `CITATION.cff`, `LICENSE`.
- Legacy exploratory scripts preserved in `scripts/legacy/` (deprecated).

### Added (Phase 1c-1g - full geography and real climate panel)

- `uganda_districts_114.csv`: official AAS district grouping (AAS 2020
  Ch. 1 Table 1-2) matched 1:1 to geoBoundaries OCHA ADM2 names; centroids
  from ADM2 geometry (`src/cropyield/data/districts.py`).
- CHIRPS v2.0 monthly extraction for all 114 districts
  (`src/cropyield/data/chirps.py`): MAM/SON/DJF/JJA/annual features, CV,
  wet months, 1981-2010 climatology and z-scores. Extraction rewritten with
  numpy nearest-pixel indexing after xarray vectorized `sel()` proved
  infeasible on the 7.7 GB global NetCDF. Values are consistent with the
  current CHIRPS v2.0 file; the legacy 15-district CSV was built from an
  older download and is superseded.
- CHIRPS daily rainfall via the ClimateSERV API for all 114 centroids
  (`src/cropyield/data/climateserv.py`): onset (20 mm/3 d + no 7 d dry
  spell in 15 follow-up days), cessation, season length, false-onset flag,
  dry-spell counts (>=7/>=10 d), rain days (>=1/10/20 mm), mean wet-day
  rainfall, maximum 5-day rainfall. The API requires POST + CSRF cookie
  (GET returns 500); raw responses cached under `data/raw/climate_cache/`.
- NASA POWER daily T2M_MAX/T2M_MIN (MERRA-2) for all 114 centroids
  (`src/cropyield/data/nasapower.py`): GDD (base 10 C), heat days
  (>=30/35 C), warm nights (>=20 C), heatwaves (>=3 consecutive days
  >=30 C), wet-day temperature interactions.
- Panels (`scripts/build_panels.py`):
  - `data/processed/observed/{crop}_subregion_panel.csv` - 14 subregions x
    (2020: 3 season groups, 2018: annual), yields grade A; climate
    features aggregated from district centroids by subregion mean.
  - `data/processed/assigned/{crop}_district_assigned_panel.csv` - 114
    districts, yields assigned from subregion (grade B).
  - Built for maize (56/456 rows), beans (55/449 - AAS 2020 beans second
    season covers 13 subregions in the official file), groundnuts (56/456).
- 2018 panel rows carry a single annual (total) season group; AAS 2018
  publishes second-season yields for annual crops and totals for perennials
  (documented in the source parser).

### Added (Phase 2 - SoilGrids)

- `src/cropyield/data/soilgrids.py`: ISRIC SoilGrids v2.0 extraction
  (clay, sand, silt, SOC, bulk density, CEC, pH) for all 114 district
  centroids, 0-30 cm depth-weighted (0-5/5-15/15-30). The API only accepts
  one property + one depth per request; `d_factor` conversion applied per
  property; jitter retries (2.2/4.4/8.9 km, 8 directions) fill mosaic
  holes (Kalangala/Wakiso centroids). Result: 0 missing values across all
  114 districts (`uganda_soil_features_114.csv`).

### Added (Phase 3 - honest validation)

- `src/cropyield/models/validate.py` + `configs/models.yaml`: validation
  matrix over 3 crops x 4 feature sets (rainfall-only, climate,
  climate+thermal, full+soil) x 4 schemes (random CV, group-by-subregion
  CV, temporal 2018->2020, temporal 2020->2018) x 6 models (OLS, Ridge,
  PCR, PLS, RF, XGBoost) + 3 baselines (mean, historical mean,
  previous-year yield). Conformal prediction intervals; negative
  predictions clipped at 0; zero-variance features filtered.
- Results in `reports/tables/validation_all.csv` (plus per-crop/per-scheme
  tables and long-form prediction CSVs). Best genuine signal: beans
  XGBoost under random CV (R2 = 0.613, fold-verified); no model beats
  previous-year yield under temporal validation for maize.

### Added (Phase 4 - PCA)

- `src/cropyield/pca/pca_analysis.py` + `scripts/run_pca.py`: PCA of the
  65-feature district-year climate matrix (1019 rows). Correlation-matrix
  PCA (SVD) with covariance comparison; retention via Kaiser-Guttman
  (12 components), cumulative variance 0.85 (13), and parallel analysis
  95th percentile (10); 500-bootstrap confidence intervals for
  eigenvalues and loadings; scree plot + loadings heatmap.
- `notebooks/01_pca_math.ipynb`: hand-computed eigendecomposition
  verified against the pipeline SVD and sklearn to machine precision.
- `tests/test_data_quality.py`: 20 schema/data-quality tests (district
  count, DOY windows, texture sums, yield plausibility, provenance
  columns, no-NaN feature panels); all pass. Expanded in Phase 5 with
  yield-consistency, survey-uncertainty, pooled-panel, feature
  availability and representation tests (35 total).
- Better pooled baseline from a repeated-cohort experiment: pooled
  `hybrid` RandomForest under grouped CV has R2 ~ 0.65 (vs mean
  predictor's ~ -0.02).

### Data quality decisions

- `uganda_temperature_features.csv` is flagged SYNTHETIC (generator-based);
  replaced by NASA POWER daily temperatures for new panels.
- `uganda_full_pipeline_data.csv` and `*_hybrid_yield.csv` moved to
  `data/processed/synthetic/` and are NOT used in modeling.
- SoilGrids extraction succeeded after the single-property API workaround;
  soil properties are complete (grade A) and included in PCA/modeling.
  (`uganda_temperature_features.csv` remains synthetic and is never used.)
- AAS 2019 PDF in `data/raw/` is corrupt; 2019 is not used. Panel years are
  2018 and 2020.

## [0.0.0] - 2026-07-30

Initial exploratory repository: 2020-only Eastern Uganda benchmark
(5 districts, subregion-assigned yields), synthetic temperature, synthetic
2021-2023 yields, all superseded by the Phase 0+ pipeline.
