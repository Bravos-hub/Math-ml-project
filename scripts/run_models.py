#!/usr/bin/env python3
"""Run the model-comparison matrix and write experiment tracking.

Regenerates per-crop/per-feature-set/per-representation validation tables,
the combined ``reports/tables/validation_all.csv``, and an experiment
registry (``reports/tables/experiment_registry.csv``) that records run_id,
git commit, sample size, number of unique targets and small-sample warnings
(review P2 #23).

Default scope (fast, ~2 min):
  - pooled subregion x crop sample, ``full_agroecological`` feature set,
    representations raw / pca / hybrid (review P1 #18).

With ``--full`` also runs the raw-only matrix over all crops x feature sets
(the historical coverage of ``validation_all.csv``).

Run:
  python scripts/run_models.py [--full] [--crops ...] [--feature-sets ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropyield.data.paths import TABLES  # noqa: E402
from cropyield.models.validate import load_config, run_validation  # noqa: E402

log = logging.getLogger(__name__)

CROPS = ("maize", "beans", "groundnuts", "pooled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run all crops x feature sets (raw) in addition")
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--representations", nargs="*",
                        default=["raw", "pca", "hybrid"])
    parser.add_argument("--feature-sets", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config()
    feature_sets = list(cfg["feature_sets"])
    reps = args.representations
    crops = args.crops or (list(CROPS) if args.full else ["pooled"])
    fsets = args.feature_sets or (feature_sets if args.full
                                  else ["full_agroecological"])

    summaries = []
    metadatas = []
    for crop in crops:
        for feature_set in fsets:
            for representation in reps:
                log.info("validating crop=%s feature_set=%s repr=%s",
                         crop, feature_set, representation)
                summary, _preds, meta = run_validation(
                    crop, feature_set, representation)
                summaries.append(summary)
                metadatas.append(meta)
                # keep column closure of meta short for the registry
                registry_row = {k: v for k, v in meta.items()
                                if not isinstance(v, list)}
                metadatas[-1] = registry_row

    all_summary = pd.concat(summaries, ignore_index=True)
    registry = pd.DataFrame(metadatas)
    TABLES.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(TABLES / "validation_all.csv", index=False)
    registry.to_csv(TABLES / "experiment_registry.csv", index=False)
    log.info("wrote validation_all.csv (%d rows) and experiment_registry.csv",
             len(all_summary))
    key = all_summary[all_summary["scheme"] == "random_cv"] \
        [["crop", "feature_set", "representation", "model", "rmse", "r2",
          "anomaly_r2"]] \
        .sort_values(["crop", "feature_set", "rmse"])
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()