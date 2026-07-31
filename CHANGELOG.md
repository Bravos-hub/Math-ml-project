# Changelog

All notable changes to this project are documented here, following the
principle that every experiment must be reproducible from a documented
data version.

## [Unreleased]

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

### Data quality decisions

- `uganda_temperature_features.csv` is flagged SYNTHETIC (generator-based);
  replaced by NASA POWER daily temperatures for new panels.
- `uganda_full_pipeline_data.csv` and `*_hybrid_yield.csv` moved to
  `data/processed/synthetic/` and are NOT used in modeling.
- SoilGrids extraction failed (`request_failed`); soil properties are
  excluded from PCA/modeling until repaired (quality grade F).
- AAS 2019 PDF in `data/raw/` is corrupt; 2019 is not used. Panel years are
  2018 and 2020.

## [0.0.0] - 2026-07-30

Initial exploratory repository: 2020-only Eastern Uganda benchmark
(5 districts, subregion-assigned yields), synthetic temperature, synthetic
2021-2023 yields, all superseded by the Phase 0+ pipeline.
