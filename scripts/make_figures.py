#!/usr/bin/env python3
"""Create the reproducible validation and residual figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uganda_crop_model.data.paths import FIGURES, TABLES  # noqa: E402


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(TABLES / "validation_all.csv")
    model_results = results[~results["model"].isin(["mean_predictor", "historical_mean", "previous_year_yield"])]
    for scheme in ("group_by_subregion", "random_cv"):
        subset = model_results[model_results["scheme"].eq(scheme)]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="model", columns="representation", values="r2", aggfunc="max")
        pivot.plot(kind="bar", figsize=(10, 5), ylabel="R²", title=f"Held-out {scheme} performance")
        plt.tight_layout()
        plt.savefig(FIGURES / f"validation_{scheme}_r2.png", dpi=180)
        plt.close()

    predictions = pd.read_csv(TABLES / "validation_all_predictions.csv")
    y_true = predictions.get("observed_yield", predictions.get("y_true"))
    y_pred = predictions.get("predicted_yield", predictions.get("y_pred"))
    if y_true is not None and y_pred is not None:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=8, alpha=0.35)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("Observed yield (t/ha)")
        plt.ylabel("Held-out predicted yield (t/ha)")
        plt.title("Held-out predictions")
        plt.tight_layout()
        plt.savefig(FIGURES / "heldout_observed_vs_predicted.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    main()
