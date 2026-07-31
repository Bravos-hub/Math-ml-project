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
