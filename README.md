# Uganda Crop Yield Modeling with PCA and Machine Learning

A reproducible agricultural data-science project for building, analyzing, and evaluating crop-yield prediction pipelines for Uganda using official agricultural statistics, climate observations, terrain variables, and soil covariates.

The repository currently focuses on district-level and subregion-derived benchmarks for Eastern Uganda, with maize as the primary research crop and beans and groundnuts included as comparative crop benchmarks. It combines data engineering, exploratory analysis, Principal Component Analysis (PCA), and supervised machine-learning evaluation.

> **Research status:** active undergraduate proof of concept. The current results are computational benchmarks based on secondary data and are not yet field-validated or suitable for operational farmer-level decision-making.

---

## Project objectives

The study is organized around three objectives:

1. **Prepare the data:** acquire, integrate, clean, and explore publicly available Ugandan agricultural and climate data suitable for crop-yield analysis.
2. **Design and interpret PCA:** mathematically verify PCA and use it to identify latent climatic, thermal, terrain, and soil-related factors.
3. **Evaluate predictive models:** compare crop-yield models trained on raw features, PCA-reduced features, and selected hybrid feature representations.

The corresponding research questions are:

1. How can publicly available agricultural and climate datasets be integrated and prepared for crop-yield analysis in Uganda?
2. What latent agroclimatic factors can be identified and interpreted using PCA?
3. How does predictive performance differ across algorithms and feature representations?

---

## Current analytical scope

### Primary unit of analysis

The canonical modeling grain is:

```text
one district × one year
```

Some 2020 benchmark tables use:

```text
one district × one crop × one season group
```

### Current geographic focus

The current Eastern Uganda benchmark includes:

- Iganga
- Jinja
- Kapchorwa
- Mbale
- Tororo

These districts are mapped to AAS 2020 subregions for benchmark construction. Because the source yield statistics are subregional, districts assigned to the same subregion may share the same benchmark yield value. This is a deliberate approximation and must not be interpreted as independently observed district-level yield.

### Current crop coverage

- Maize
- Beans
- Groundnuts
- Additional AAS 2020 crop tables are included for broader agricultural comparison.

---

## Data sources

| Source | Main contribution | Current role |
|---|---|---|
| Uganda Bureau of Statistics Annual Agricultural Survey 2020 | Planted area, harvested area, production, yield, coefficients of variation | Official crop benchmark source |
| CHIRPS | Seasonal and annual rainfall | Real climate predictor source |
| ERA5-Land or schema-compatible generated temperature features | Maximum/minimum temperature, growing degree days, thermal stress | Thermal predictor layer |
| SoilGrids | Soil pH, organic carbon, texture, bulk density, cation exchange capacity | Agroecological predictor layer |
| District metadata | Coordinates and elevation | Geographic and terrain covariates |
| Derived project tables | Cleaned, harmonized, and benchmark-ready variables | Modeling inputs and outputs |

See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for the canonical variables, units, roles, source notes, and known quality limitations.

---

## Core features

### Rainfall

- `MAM`
- `SON`
- `DJF`
- `JJA`
- `annual_rainfall`
- `rain_cv`
- `max_monthly`
- `min_monthly`
- `rainy_months`

### Temperature and thermal time

- Seasonal `tmax` and `tmin`
- Seasonal growing degree days
- Seasonal heat- and cold-stress indicators
- Annual thermal summaries
- Elevation

### Soil and terrain

- Soil pH
- Soil organic carbon
- Clay, sand, and silt fractions
- Bulk density
- Cation exchange capacity
- Latitude and longitude metadata

### Target

The primary supervised-learning target is:

```text
yield_tons_ha
```

For AAS-derived crop tables, yield is calculated from production and harvested area where available.

---

## Repository contents

### Documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, setup, workflow, limitations, and reproducibility guide |
| `DATA_DICTIONARY.md` | Canonical schema, variable definitions, data quality notes, and file mapping |

### Data-construction scripts

| Script | Purpose |
|---|---|
| `build_modeling_dataset.py` | Builds the canonical district-year modeling table from climate, soil, and yield inputs |
| `build_eastern_2020_yield_from_aas.py` | Builds the Eastern Uganda maize yield benchmark from AAS 2020 |
| `build_eastern_2020_beans_yield_from_aas.py` | Builds the beans benchmark |
| `build_eastern_2020_groundnuts_yield_from_aas.py` | Builds the groundnuts benchmark |

### Benchmark and evaluation scripts

| Script | Purpose |
|---|---|
| `build_real_2020_benchmark.py` | Builds and evaluates the maize benchmark |
| `build_real_2020_beans_benchmark.py` | Builds and evaluates the beans benchmark |
| `build_real_2020_groundnuts_benchmark.py` | Builds and evaluates the groundnuts benchmark |
| `compare_real_2020_crop_benchmarks.py` | Compares maize, beans, and groundnuts benchmark results |

### Important generated datasets

| File | Grain | Description |
|---|---|---|
| `uganda_rainfall_features.csv` | district-year | CHIRPS-derived rainfall features |
| `uganda_temperature_features.csv` | district-year | Seasonal and annual thermal features |
| `uganda_soil_features.csv` | district | Soil and extraction-status variables |
| `uganda_rainfall_temperature_soil_features.csv` | district-year | Integrated climate and soil table |
| `uganda_full_modeling_dataset.csv` | district-year | Canonical model-ready dataset where yield is available |
| `eastern_uganda_maize_modeling_dataset.csv` | district-year | Eastern Uganda maize modeling panel |
| `aas2020_eastern_district_yield.csv` | district-season | Maize benchmark assigned from AAS subregions |
| `aas2020_eastern_district_beans_yield.csv` | district-season | Beans benchmark |
| `aas2020_eastern_district_groundnuts_yield.csv` | district-season | Groundnuts benchmark |

The repository also contains subregion-level AAS crop tables and generated benchmark figures.

---

## Analytical workflow

```text
Official and remote-sensing data
        ↓
Source-specific extraction and cleaning
        ↓
District and year harmonization
        ↓
Feature engineering
        ↓
Rainfall + temperature + terrain + soil integration
        ↓
AAS crop-yield benchmark construction
        ↓
Missing-data and quality checks
        ↓
Feature standardization
        ↓
Principal Component Analysis
        ↓
Raw-feature, PCA, and hybrid model comparisons
        ↓
Cross-validation and diagnostic visualizations
```

---

## PCA methodology

For a centered data matrix \(X_c\), the sample covariance matrix is:

```text
S = (X_cᵀ X_c) / (n - 1)
```

The project verifies that the covariance matrix is symmetric and positive semi-definite, then compares manually derived eigenvalues for a two-feature subset with `numpy.linalg.eigh()`.

For a two-dimensional covariance matrix:

```text
S = [[a, c],
     [c, b]]
```

the characteristic equation is:

```text
λ² - (a + b)λ + (ab - c²) = 0
```

PCA is then applied to standardized numerical predictors. Component retention should be justified using explained variance, scree-plot structure, stability, and interpretability rather than a threshold alone.

Important methodological rule:

> The yield target and variables directly used to calculate it must never be included in the PCA predictor matrix.

---

## Predictive modeling plan

The intended comparison includes:

- Ordinary Least Squares
- Ridge Regression
- Random Forest
- XGBoost

Each model should be evaluated using three feature representations:

1. **Raw:** original cleaned predictors.
2. **PCA-reduced:** retained principal-component scores.
3. **Hybrid:** PCA scores plus contextual variables that were not included in the PCA transformation.

Recommended evaluation metrics:

- Root Mean Squared Error (RMSE)
- Coefficient of determination (`R²`)
- Mean and standard deviation across cross-validation folds

All preprocessing steps must be fitted inside each training fold to prevent leakage.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bravos-hub/Math-ml-project.git
cd Math-ml-project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the scientific Python dependencies

A pinned `requirements.txt` is not yet present. The current scripts are expected to require packages similar to:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl xgboost
```

Depending on the extraction workflow, geospatial processing may additionally require:

```bash
pip install geopandas rasterio xarray netCDF4 requests
```

Create and commit a pinned dependency file after validating the complete environment:

```bash
pip freeze > requirements.txt
```

---

## Running the pipeline

Run scripts from the repository root so that relative paths resolve consistently.

### Build crop benchmarks

```bash
python build_eastern_2020_yield_from_aas.py
python build_eastern_2020_beans_yield_from_aas.py
python build_eastern_2020_groundnuts_yield_from_aas.py
```

### Build the integrated modeling dataset

```bash
python build_modeling_dataset.py
```

### Build and evaluate 2020 crop benchmarks

```bash
python build_real_2020_benchmark.py
python build_real_2020_beans_benchmark.py
python build_real_2020_groundnuts_benchmark.py
```

### Compare crop benchmarks

```bash
python compare_real_2020_crop_benchmarks.py
```

Before running a script, inspect its input-path constants or command-line options and confirm that the required source tables exist.

---

## Data management

Large climate and geospatial source files must not be committed to ordinary Git history.

The repository ignores formats such as:

```text
*.nc
*.tif
*.tiff
*.zip
```

The 7+ GiB CHIRPS NetCDF source should remain outside Git and be recreated or downloaded using documented scripts. GitHub should store code, metadata, lightweight derived tables, documentation, and publication-ready outputs—not unrestricted raw archives.

Recommended local structure:

```text
data/
├── raw/          # ignored
├── external/     # ignored where licensing requires
├── interim/      # ignored
├── processed/    # publish only when permitted
└── public/       # dictionaries, schemas, and small shareable examples
```

---

## Reproducibility requirements

A reproducible research run should record:

- Source URLs and retrieval dates
- Dataset versions
- Script and commit version
- Unit-conversion rules
- District/subregion mapping rules
- Missing-data decisions
- Standardization parameters
- PCA feature list and retained components
- Cross-validation folds and random seeds
- Model hyperparameters
- Output metrics and figure paths

Recommended automated checks include:

- Unique district-year keys
- Nonnegative crop yield
- Positive harvested area where yield is calculated
- No target leakage into PCA
- Symmetric covariance matrix
- Nonnegative covariance eigenvalues within numerical tolerance
- Agreement between manual eigenvalues and `numpy.linalg.eigh()`
- No overlap between grouped training and test entities

---

## Known limitations

The current repository should be interpreted with the following limitations:

1. The project is based on secondary data and has not yet been validated through field measurements.
2. Several benchmarks contain only five Eastern Uganda districts, which is insufficient for strong generalization claims.
3. AAS 2020 yield statistics are subregional and are assigned to selected districts for comparison; they are not independent district observations.
4. Some temperature or yield tables may be synthetic, proxy-based, or schema-compatible placeholders. Every result must identify the actual source used.
5. SoilGrids extraction records may contain `request_failed` or missing values and require quality checks before production analysis.
6. SoilGrids units must be confirmed against the exact source metadata before publication.
7. Static soil values repeated across years do not represent annual soil change.
8. PCA maximizes predictor variance, not yield-prediction accuracy.
9. The present framework supports association and prediction, not causal inference.
10. Current outputs are not suitable for operational farm recommendations or national deployment.

---

## Roadmap

### Phase 1 — Mathematical and data foundation

- Complete the handwritten PCA derivation
- Add an executable PCA verification notebook
- Pin Python dependencies
- Add automated data-contract tests

### Phase 2 — Stronger real-data integration

- Replace all proxy yield fields with documented official observations
- Replace schema-compatible synthetic thermal data with fully documented ERA5-Land extraction
- Repair and validate SoilGrids ingestion
- Add UNPS household/plot features where licensing and identifiers permit

### Phase 3 — Robust modeling

- Implement leakage-safe pipelines
- Add grouped five-fold cross-validation
- Compare OLS, Ridge, Random Forest, and XGBoost
- Evaluate raw, PCA, and hybrid representations
- Add uncertainty and sensitivity analysis

### Phase 4 — External validation

Subject to institutional approval and partnership:

- Collect primary plot-level observations
- Measure true plot area and harvested yield
- Validate satellite and survey predictors
- Test transferability across districts and seasons
- Recalibrate the final model

---

## Research ethics and responsible use

This phase uses secondary data only. No direct farmer recruitment, interviews, or field measurements are included.

When using restricted household or survey data:

- Follow the provider's access conditions.
- Do not publish direct household identifiers.
- Do not publish precise protected coordinates.
- Keep restricted source files outside the public repository.
- Publish reproducible code and aggregate outputs where licensing permits.
- Clearly distinguish observed, derived, assigned, proxy, and synthetic values.

---

## Contributing

Contributions should preserve data provenance and research reproducibility.

Before submitting changes:

1. Document new data sources and licenses.
2. Update `DATA_DICTIONARY.md` for every added field.
3. Avoid committing large raw files.
4. Add or update quality checks.
5. Use clear commit messages.
6. State whether results use real, proxy, assigned, or synthetic data.

---

## Citation

When citing this repository, use a form similar to:

```text
Kubanja, E. E. (2026). Uganda Crop Yield Modeling with PCA and Machine Learning: An undergraduate computational research project. GitHub repository.
```

A formal `CITATION.cff` file should be added before publication or archival release.

---

## Author

**Kubanja Elijah Eldred**  
Undergraduate researcher and software engineer  
Uganda

---

## License

No license file is currently present. Until a license is added, the repository remains protected by default copyright rules. Add an explicit software and data license only after reviewing the licenses and redistribution conditions of all upstream datasets.

---

## Disclaimer

This repository is an academic proof of concept. Its predictions, derived statistics, and visualizations must not be used as a substitute for official agricultural statistics, agronomic advice, field inspection, or validated operational forecasting.