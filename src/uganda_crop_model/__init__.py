"""Uganda crop-yield modeling research package.

This is the canonical (final-analysis) package for the project.  It takes
over the scientific role of the earlier ``cropyield`` prototype: build an
analytically valid subregion-season-year target dataset and evaluate PCA and
predictive models on it under leakage-safe validation.

Design contract (see configs/final_maize_aas.yaml):

* Analytical grain: sub_region x year x season x crop.
* Target: official AAS production / harvested area (no proxy, no synthetic,
  no geographic assignment in final mode).
* Predictors: climate aggregated to the same subregion (daily rainfall and
  temperature summarised over documented season windows).
"""

from __future__ import annotations

__version__ = "0.2.0"