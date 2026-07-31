#!/usr/bin/env python3
"""
Create one portfolio figure for the first three Eastern Uganda 2020
real-data crop benchmarks.

Inputs:
  - real_2020_crop_benchmark_comparison.csv
  - real_2020_crop_benchmark_district_targets.csv
  - real_2020_benchmark_model_results.csv
  - real_2020_beans_benchmark_model_results.csv
  - real_2020_groundnuts_benchmark_model_results.csv

Outputs:
  - real_2020_crop_benchmark_portfolio_figure.png
  - real_2020_crop_benchmark_portfolio_figure.svg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COMPARISON_FILE = Path("real_2020_crop_benchmark_comparison.csv")
DISTRICT_FILE = Path("real_2020_crop_benchmark_district_targets.csv")
RESULT_FILES = {
    "maize": Path("real_2020_benchmark_model_results.csv"),
    "beans": Path("real_2020_beans_benchmark_model_results.csv"),
    "groundnuts": Path("real_2020_groundnuts_benchmark_model_results.csv"),
}
PNG_FILE = Path("real_2020_crop_benchmark_portfolio_figure.png")
SVG_FILE = Path("real_2020_crop_benchmark_portfolio_figure.svg")

CROP_ORDER = ["maize", "beans", "groundnuts"]
CROP_LABELS = {"maize": "Maize", "beans": "Beans", "groundnuts": "Groundnuts"}
COLORS = {"maize": "#0B6E4F", "beans": "#D97706", "groundnuts": "#B83280"}
FEATURES = [
    "MAM",
    "SON",
    "annual_rainfall",
    "rain_cv",
    "annual_gdd",
    "elevation_m",
    "soil_moisture_index",
]


def require_files():
    paths = [COMPARISON_FILE, DISTRICT_FILE, *RESULT_FILES.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required benchmark files: " + ", ".join(missing))


def load_data():
    comparison = pd.read_csv(COMPARISON_FILE).set_index("crop").loc[CROP_ORDER].reset_index()
    district = pd.read_csv(DISTRICT_FILE).sort_values("district").reset_index(drop=True)
    results = {
        crop: pd.read_csv(path).assign(crop=crop)
        for crop, path in RESULT_FILES.items()
    }
    all_results = pd.concat(results.values(), ignore_index=True)
    return comparison, district, all_results


def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#17202A",
        va="top",
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_best_scores(ax, comparison):
    x = np.arange(len(CROP_ORDER))
    values = comparison["best_r2"].to_numpy()
    bars = ax.bar(x, values, color=[COLORS[crop] for crop in CROP_ORDER], width=0.62)
    ax.axhline(0, color="#475569", linewidth=0.9)
    ax.set_xticks(x, [CROP_LABELS[crop] for crop in CROP_ORDER])
    ax.set_ylabel("Best leave-one-out $R^2$")
    ax.set_ylim(min(-1.0, values.min() - 0.15), 1.08)
    ax.set_title("Best cross-validated score", loc="left", fontsize=12, fontweight="bold")
    ax.text(
        0,
        -0.25,
        "Higher is better; negative $R^2$ means the model loses to a mean-only baseline.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#64748B",
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.035 if value >= 0 else -0.09),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    style_axis(ax)


def plot_raw_vs_pca(ax, all_results):
    summary = (
        all_results.groupby(["crop", "feature_space"], as_index=False)["r2"]
        .max()
        .pivot(index="crop", columns="feature_space", values="r2")
        .reindex(CROP_ORDER)
    )
    x = np.arange(len(CROP_ORDER))
    width = 0.34
    raw = summary["raw"].to_numpy()
    pca = summary["PCA"].to_numpy()
    ax.bar(x - width / 2, raw, width, color="#94A3B8", label="Raw features")
    ax.bar(x + width / 2, pca, width, color="#38BDF8", label="Top 3 PCs")
    ax.axhline(0, color="#475569", linewidth=0.9)
    ax.set_xticks(x, [CROP_LABELS[crop] for crop in CROP_ORDER])
    ax.set_ylabel("Best $R^2$ within feature space")
    ax.set_ylim(-2.65, 1.08)
    ax.set_title("Does PCA help in this benchmark?", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    for i, (raw_value, pca_value) in enumerate(zip(raw, pca)):
        ax.text(i - width / 2, raw_value + (0.04 if raw_value >= 0 else -0.11), f"{raw_value:.2f}",
                ha="center", va="bottom" if raw_value >= 0 else "top", fontsize=8.5)
        ax.text(i + width / 2, pca_value + (0.04 if pca_value >= 0 else -0.11), f"{pca_value:.2f}",
                ha="center", va="bottom" if pca_value >= 0 else "top", fontsize=8.5)
    style_axis(ax)


def plot_district_targets(ax, district):
    target_specs = [
        ("maize_yield_tons_ha", "Maize"),
        ("beans_yield_tons_ha", "Beans"),
        ("groundnuts_yield_tons_ha", "Groundnuts"),
    ]
    y = np.arange(len(district))
    offsets = [-0.22, 0, 0.22]
    for offset, (column, label) in zip(offsets, target_specs):
        ax.scatter(
            district[column],
            y + offset,
            s=48,
            color=COLORS[label.lower()],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=label,
        )
    ax.set_yticks(y, district["district"])
    ax.set_xlabel("AAS 2020 yield target (tonnes/ha)")
    ax.set_title("The five-district target structure", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8.5)
    ax.invert_yaxis()
    style_axis(ax)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)


def plot_model_landscape(ax, all_results):
    for crop in CROP_ORDER:
        crop_results = all_results[all_results["crop"] == crop]
        ax.scatter(
            crop_results["rmse"],
            crop_results["r2"],
            s=58,
            color=COLORS[crop],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.8,
            label=CROP_LABELS[crop],
        )
        best = crop_results.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
        ax.scatter(
            best["rmse"],
            best["r2"],
            s=145,
            facecolors="none",
            edgecolors=COLORS[crop],
            linewidth=1.8,
            zorder=4,
        )
    ax.axhline(0, color="#475569", linewidth=0.9)
    ax.set_xlabel("RMSE (tonnes/ha)")
    ax.set_ylabel("$R^2$")
    ax.set_title("All tested models", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=8.5)
    ax.text(
        0.98,
        0.04,
        "Ring = best model for that crop",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color="#64748B",
    )
    style_axis(ax)
    ax.grid(axis="both", color="#E2E8F0", linewidth=0.8)


def add_footer(fig):
    fig.text(
        0.055,
        0.018,
        "Scope: Eastern Uganda | 2020 only | 5 districts | 7 climate/terrain predictors | "
        "AAS sub-region yields assigned to districts | Leave-one-out validation",
        fontsize=8.5,
        color="#475569",
    )
    fig.text(
        0.945,
        0.018,
        "First real-data benchmark",
        fontsize=8.5,
        color="#0B6E4F",
        ha="right",
        fontweight="bold",
    )


def main():
    require_files()
    comparison, district, all_results = load_data()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), facecolor="#F8FAFC")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12, wspace=0.25, hspace=0.42)
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")

    plot_best_scores(axes[0, 0], comparison)
    plot_raw_vs_pca(axes[0, 1], all_results)
    plot_district_targets(axes[1, 0], district)
    plot_model_landscape(axes[1, 1], all_results)

    fig.suptitle(
        "Three crops, one environmental prediction benchmark",
        x=0.055,
        y=0.955,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#17202A",
    )
    fig.text(
        0.055,
        0.915,
        "Eastern Uganda 2020 real-data benchmark | rainfall + temperature + terrain + soil moisture",
        fontsize=11.5,
        color="#64748B",
    )
    add_footer(fig)

    fig.savefig(PNG_FILE, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(SVG_FILE, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {PNG_FILE}")
    print(f"[✓] Saved: {SVG_FILE}")


if __name__ == "__main__":
    main()
