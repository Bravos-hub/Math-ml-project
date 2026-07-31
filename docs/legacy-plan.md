# Plan: Uganda Agriculture + Rainfall Merged Dataset

Goal: Real CSV with ≥100 observations and ≥8 features for 3 districts (Luwero, Nakasongola, Mubende), merging UBOS AAS tables, CHIRPS/NASA POWER rainfall, and UNPS-style household farm data.

## Stage 0 — Feasibility check (now)
- Test NASA POWER API (free, no auth) for precipitation at 3 district coordinates.
- Test access to UBOS AAS tables (ubos.org) and World Bank Microdata (UNPS waves) — likely gated; document fallback.

## Stage 1 — Data acquisition (real sources first)
- Load `batch-download` skill guidance.
- Pull REAL monthly precipitation 2015–2023 for Luwero, Nakasongola, Mubende via NASA POWER API (CHIRPS-equivalent satellite/ag precip).
- Extract district-level crop/yield figures from published UBOS AAS 2018/2019 tables (PDF tables, cited).
- Attempt UNPS microdata (World Bank Microdata Library); if registration-gated, reconstruct household-level rows from published UNPS summary statistics, clearly labeled.

## Stage 2 — Merge & engineer
- District × season × year panel as backbone (real rainfall).
- Join district crop aggregates + household-farm covariates (farm size, hh size, crops grown, inputs).
- Target: ≥100 observations, ≥8 features.

## Stage 3 — Validate & deliver
- Validate row/feature counts, missingness, sanity checks.
- Deliver `/mnt/agents/output/uganda_agri_rainfall_merged.csv` + data dictionary note on which columns are real vs reconstructed.
