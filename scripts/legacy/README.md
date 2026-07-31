# Legacy scripts

These scripts built the original flat-layout exploration pipeline
(2020-only Eastern Uganda benchmark, synthetic temperature, hybrid
datasets). They are **deprecated and superseded** by the pipeline under
`src/cropyield/` and `scripts/`.

They are kept for reference and reproducibility of the original results.
Their relative paths still point at the old root-level layout, so they may
need path adjustments if you run them. Do not use their outputs for new
modeling: load tables from `data/processed/observed/` or
`data/processed/assigned/` instead.

Key scripts and what they produced:

| Script | Purpose | Output (now in) |
|---|---|---|
| `build_eastern_2020_yield_from_aas.py` | Assign AAS 2020 subregion maize yield to 5 eastern districts | `data/interim/aas2020_eastern_district_yield.csv` |
| `build_real_2020_benchmark.py` | First 5-row real-data benchmark (maize) | `reports/tables/real_2020_benchmark_*` |
| `train_yield_models.py` | Raw vs PCA model comparison | `reports/tables/*model_comparison_results.csv` |
| `chirps_extract_direct.py` | CHIRPS monthly extraction from local NetCDF (reused logic) | `data/interim/uganda_rainfall_features.csv` |
| `chirps_approach2_climateserv.py` | ClimateSERV API prototype (reused logic) | - |
| `soil-df.py` | SoilGrids REST query (failed; being repaired) | `data/interim/uganda_soil_features.csv` |
| `uganda_full_pipeline_selfcontained.py` | Fully synthetic pipeline | `data/processed/synthetic/` |
