# Reviewer response map

| Reviewer item | Implementation / evidence | Status |
|---|---|---|
| Soil and elevation validation | `data/subregion_soil.py`, final datasets, soil contract tests; elevation adapter refuses absent/unvalidated files | Soil passed; elevation unavailable honestly |
| Real hybrid representation | `models/pipelines.py`, `run_final_analysis.py --dataset multi_crop --mode spatial`; PCA and hybrid use distinct branches | Implemented |
| Uncertainty/outliers/diagnostics | OOF conformal columns, `*_spatial_conformal_coverage.csv`, outlier/residual/agreement/permutation tables | Implemented |
| Crop and target sensitivity | Crop-stratified tables plus raw/log1p/crop-normalized outputs; raw remains primary | Implemented |
| Historical baselines | Training-fold subregion, subregion×crop, and previous-wave fallback tables | Implemented; additional waves unavailable |
| LOSO/stress | `--mode loso`, `--mode stress`, explicit unavailable manifest for insufficient years | Implemented |
| Additional UBOS waves | `data/ubos_waves.py` validates grain; AAS2019 PDF and 2015–2021 workbooks do not meet panel contract | Documented blocker |
| CI and reproducibility | `.github/workflows/ci.yml`, `requirements.lock`, generated-output ignore rules | Implemented |
| Research interpretation | 373-row multi-crop evidence is primary; maize is small-sample exploratory; two-year limitation retained | Implemented |
