# Reviewer response map

| Reviewer item | Implementation / evidence | Status |
|---|---|---|
| Soil and elevation validation | `data/subregion_soil.py`, final datasets, soil contract tests; elevation adapter refuses absent/unvalidated files | Soil implemented; elevation unavailable honestly |
| Real hybrid representation | `models/pipelines.py`, `run_final_analysis.py --dataset multi_crop --mode spatial`; climate-only PCA is compared with PCA-climate + raw static soil | Implemented |
| PLS comparison | Added to the authoritative model registry with fold-local preprocessing | Implemented; included in the next full matrix |
| Uncertainty/outliers/diagnostics | OOF conformal columns, `*_spatial_conformal_coverage.csv`, outlier/residual/agreement tables; held-out permutation requires successful model checkpoints | Partial: RF/XGB execution remains to be completed |
| Crop and target sensitivity | Crop-stratified tables plus raw/log1p/crop-normalized outputs; crop means are fold-local; raw remains primary | Implemented |
| Historical baselines | Training-fold subregion, subregion×crop, crop mean, and previous-wave fallback tables; spatial subregion baselines are degenerate by design | Partial: temporal validation awaits more years |
| LOSO/stress | `--mode loso`, `--mode stress`, explicit unavailable manifest for insufficient years | Implemented |
| Additional UBOS waves | `data/ubos_waves.py` invokes the layout-adaptive AAS parser for PDFs and records DDI XML codebooks as metadata-only; AAS2019 PDF remains unparseable and workbooks lack panel grain | Documented blocker; next step is obtain matching AAS2019 microdata or published subregion target tables |
| CI and reproducibility | `.github/workflows/ci.yml`, `requirements.lock`, generated-output ignore rules | Implemented |
| Research interpretation | 373-row multi-crop evidence is primary; maize is small-sample exploratory; two-year limitation retained | Implemented |
