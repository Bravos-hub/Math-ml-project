# Final-evidence protocol

The headline scientific evidence of this project is the set of curated
tables under `reports/tables/` produced from the 373-row multi-crop
dataset (`data/processed/final_multi_crop_subregion_season_year.csv`).
This document defines when and how that evidence is regenerated so that
committed results always match the code that produced them.

## The staleness rule

Evidence is **stale** and must be regenerated whenever any of the
following changes after `manifest.generated_at`:

- anything under `src/uganda_crop_model/`
- `scripts/run_final_analysis.py` or `scripts/build_final_dataset.py`
- `configs/final_maize_aas.yaml`
- the authoritative datasets in `data/processed/final_*.csv`

A stale table committed next to newer code is worse than no table: it
attributes numbers to a pipeline that never produced them.

## Regenerating

```bash
make final-evidence
```

This target:

1. Refuses to run if the multi-crop dataset is missing (`make data-final` first).
2. Runs the **full** model registry — `dummy_mean`, `ols`, `ridge`, `pls`,
   `random_forest`, `xgboost` — across `raw` / `pca` / `hybrid`
   representations under spatial GroupKFold. It never uses `--quick`.
3. Stages the curated outputs for commit.

Then review and commit:

```bash
git status
git commit -m "Regenerate final evidence (<reason>)"
```

## Rules

1. **`--quick` output is never committed as headline evidence.** Quick
   runs (dummy/ols/ridge only) are smoke tests for development.
2. **Every registry model must appear in the committed comparison.** If
   the run log shows `checkpoint skipped` for a model, fix the cause
   (usually a missing optional dependency — install `requirements.lock`)
   and re-run. Do not commit a partial matrix.
3. **Check `multi_crop_spatial_execution_status.csv`** before committing:
   every row must read `completed`; an `unavailable` row must be explained
   in the commit message.
4. **The maize-only table is secondary.** The 42-row maize sample fails
   the project's own quality gate; it is exploratory. If it is committed,
   label it accordingly and never cite it as the primary result.
5. **Interpretation columns are part of the evidence.** The
   `target_scale` slices (`raw`, `log1p`, `crop_normalized`) and the
   per-crop rows exist so the report can state honestly how much pooled
   skill is between-crop variance. Report generators must filter on
   `crop` / `target_scale` deliberately, not read the file wholesale.

## Why this exists

On 2026-08-03 the committed comparison table and manifest were produced
at 10:44 by a `--quick` run, while the model-registry and baseline fixes
landed at 11:58 — the repository briefly contained new code paired with
evidence from an older pipeline (no PLS, no tree models, degenerate
baselines). This protocol prevents that class of error from recurring.
