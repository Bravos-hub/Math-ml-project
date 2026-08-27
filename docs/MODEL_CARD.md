# Model Card: Uganda Selected Food-Crop Yield Research Framework

## Status

- Project stage: **research**
- Current lifecycle state: **INTERIM_RESEARCH**
- Deployment authorized: **no**
- Prediction horizon: **season-end retrospective**

`RESEARCH_VALIDATED` may be recorded only when the scientific acceptance gates
pass. Neither lifecycle state means production approval.

## Purpose

This framework evaluates whether climate and soil representations can explain
spatial variation in aggregate selected food-crop yields across held-out
Ugandan subregions. It compares raw, PCA, and hybrid representations using
leakage-controlled nested validation and transparent baselines.

## Prediction unit and data

The row-level unit is `subregion × crop × season × year`; the independent
environmental unit is `subregion × season × year`. The current homogeneous
seasonal analysis contains 249 crop-environment rows from 28 environments, 14
subregions, and 2020 only. Targets are aggregate UBOS statistics.

## Required research features retained

- training-fold preprocessing and PCA;
- nested group-aware model selection;
- held-out spatial evaluation and LOSO support;
- proper-training and conservative full-outer-training baselines;
- pooled, macro-average, crop-specific, and crop-centered results;
- split-conformal intervals with calibration support counts;
- spatial-cluster bootstrap confidence intervals;
- crop, season, subregion, and yield-level error diagnostics;
- dataset/config hashes and immutable run manifests.

## Current evidence limitations

- Temporal forecasting is not established because the primary seasonal data
  contain one homogeneous year.
- Twenty-eight independent environments are below the declared minimum of 50.
- Fourteen spatial clusters and small calibration sets limit precision.
- Some crops have weak or negative held-out R².
- Simple crop or crop-season baselines may outperform fitted ML models.
- Third-party terms and the institutional ethics determination remain pending.

## Appropriate uses

- thesis presentation and methodological demonstration;
- retrospective aggregate analysis;
- leakage, PCA, validation, and reproducibility research;
- planning future data collection and temporal evaluation.

## Prohibited uses

- operational or guaranteed yield forecasts;
- individual farmer decisions, credit, or insurance pricing;
- automated fertilizer or pesticide recommendations;
- claims of causal agronomic effects;
- future-season generalization claims;
- deployment or model promotion based solely on this evidence.

## Interpretation

PCA components are statistical latent axes. Feature importance and component
loadings describe associations, not causes. Every metric must state whether it
is pooled raw, macro-averaged, crop-specific, or crop-centered.
