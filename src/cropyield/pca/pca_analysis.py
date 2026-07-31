"""PCA v2: rigorous exploratory analysis of the climate feature matrix.

Implements the decisions in ``configs/features.yaml`` -> ``pca``:

- primary analysis on the correlation matrix (standardized features), with a
  covariance-matrix comparison;
- component retention via Kaiser-Guttman (eigenvalue > 1), cumulative
  variance threshold (0.85) and parallel analysis (95th percentile of
  permuted-column eigenvalues, n_permutations draws);
- bootstrap confidence intervals (n_bootstrap resamples of rows) for
  eigenvalues and loadings.

The analysis runs on the district-year climate matrix (114 districts x
2015-2023; rainfall features) optionally augmented with the seasonal
daily/temperature features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from yaml import safe_load

from cropyield.data.paths import CONFIGS, FIGURES, INTERIM, TABLES

log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIGS / "features.yaml") as fh:
        return safe_load(fh)["pca"]


def build_feature_matrix(rainfall: pd.DataFrame,
                         daily: pd.DataFrame | None = None,
                         temp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wide feature matrix (district-year rows) for PCA.

    Rainfall features are annual (one row per district-year); the seasonal
    daily/temperature features enter as the first-season values plus the
    second-season values, keyed on district-year.
    """
    from cropyield.data.provenance import PROVENANCE_COLUMNS

    def strip(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in PROVENANCE_COLUMNS if c in df.columns],
                       errors="ignore")

    df = strip(rainfall).copy()
    if daily is not None:
        d = strip(daily[daily["season"] == "first"].drop(columns="season"))
        d = d.rename(columns={c: f"first_{c}" for c in d.columns
                              if c not in ("district", "year")})
        df = df.merge(d, on=["district", "year"], how="left")
        d2 = strip(daily[daily["season"] == "second"].drop(columns="season"))
        d2 = d2.rename(columns={c: f"second_{c}" for c in d2.columns
                                if c not in ("district", "year")})
        df = df.merge(d2, on=["district", "year"], how="left")
    if temp is not None:
        t = strip(temp[temp["season"] == "first"].drop(columns="season"))
        t = t.rename(columns={c: f"temp_first_{c}" for c in t.columns
                              if c not in ("district", "year")})
        df = df.merge(t, on=["district", "year"], how="left")
        t2 = strip(temp[temp["season"] == "second"].drop(columns="season"))
        t2 = t2.rename(columns={c: f"temp_second_{c}" for c in t2.columns
                                if c not in ("district", "year")})
        df = df.merge(t2, on=["district", "year"], how="left")
    df = df.dropna(how="all")
    return df


def run_pca(X: np.ndarray, center: bool = True, scale: bool = True,
            n_components: int | None = None):
    """Correlation (scale=True) or covariance (scale=False) PCA via SVD."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0) if center else X
    if scale:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = X / std
    n, p = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvals = S ** 2 / (n - 1)
    loadings = Vt.T * np.sqrt(np.maximum(eigenvals, 0))[None, :]
    return eigenvals, loadings, S


def parallel_analysis(X: np.ndarray, n_permutations: int = 200,
                      seed: int = 42) -> np.ndarray:
    """95th percentile of permuted-column eigenvalues (parallel analysis)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n = len(X)
    sim_eigs = np.zeros((n_permutations, min(X.shape)))
    for i in range(n_permutations):
        perm = X.copy()
        for j in range(perm.shape[1]):
            perm[:, j] = rng.permutation(perm[:, j])
        perm = perm - perm.mean(axis=0)
        std = perm.std(axis=0)
        std[std == 0] = 1.0
        perm = perm / std
        U, S, _ = np.linalg.svd(perm, full_matrices=False)
        sim_eigs[i] = S ** 2 / (n - 1)
    return np.quantile(sim_eigs, 0.95, axis=0)


def bootstrap_cis(X: np.ndarray, n_bootstrap: int = 500,
                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CIs for eigenvalues and loadings (row resampling)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    k = min(n, p)
    eig_draws = np.zeros((n_bootstrap, k))
    load_draws = np.zeros((n_bootstrap, p, k))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        eigs, loads, _ = run_pca(X[idx])
        eig_draws[i] = eigs[:k]
        load_draws[i] = loads[:, :k]
    eig_ci = np.stack([np.quantile(eig_draws, 0.025, axis=0),
                       np.quantile(eig_draws, 0.975, axis=0)])
    load_ci = np.stack([np.quantile(load_draws, 0.025, axis=0),
                        np.quantile(load_draws, 0.975, axis=0)])
    return eig_ci, load_ci


def analyze(feature_matrix: pd.DataFrame, out_dir: Path = TABLES,
            fig_dir: Path = FIGURES) -> dict:
    """Full PCA v2 analysis on a feature matrix; writes tables + figure."""
    cfg = load_config()
    feat_cols = [c for c in feature_matrix.columns
                 if c not in ("district", "year")]
    X = feature_matrix[feat_cols].dropna().to_numpy()
    dropped = len(feature_matrix) - len(X)
    if dropped:
        log.info("dropped %d rows with missing values", dropped)

    # Correlation PCA (primary)
    eigs, loads, _ = run_pca(X, scale=True)
    p = len(feat_cols)
    n_keep_cor = int(np.sum(np.cumsum(eigs / eigs.sum()) < cfg["variance_threshold"])) + 1
    n_keep_kaiser = int(np.sum(eigs > 1.0))
    pa = parallel_analysis(X, cfg["n_permutations"])
    n_keep_pa = int(np.sum(eigs > pa))
    eig_ci, load_ci = bootstrap_cis(X, cfg["n_bootstrap"])

    # Covariance PCA (comparison)
    eigs_cov, loads_cov, _ = run_pca(X, scale=False)

    scree = pd.DataFrame({
        "component": np.arange(1, p + 1),
        "eigenvalue_correlation": eigs,
        "eigenvalue_covariance": eigs_cov,
        "variance_explained": eigs / eigs.sum(),
        "cumulative_variance": np.cumsum(eigs / eigs.sum()),
        "parallel_analysis_95th": pa,
        "boot_ci_lower": eig_ci[0],
        "boot_ci_upper": eig_ci[1],
        "keep_correlation_0.85": np.arange(1, p + 1) <= n_keep_cor,
        "keep_kaiser_gt_1": np.arange(1, p + 1) <= n_keep_kaiser,
        "keep_parallel_analysis": np.arange(1, p + 1) <= n_keep_pa,
    })
    loadings = pd.DataFrame(loads[:, :max(n_keep_cor, 4)])
    loadings.insert(0, "feature", feat_cols)
    for i in range(loadings.shape[1] - 1):
        loadings[f"pc{i+1}_ci_lower"] = load_ci[0][:, i][:len(feat_cols)]
        loadings[f"pc{i+1}_ci_upper"] = load_ci[1][:, i][:len(feat_cols)]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    scree.to_csv(out_dir / "pca_v2_scree.csv", index=False)
    loadings.to_csv(out_dir / "pca_v2_loadings.csv", index=False)

    _plot(scree, loadings, fig_dir / "pca_v2_scree.png")

    return {
        "n_features": p,
        "n_rows": len(X),
        "n_keep_0.85": n_keep_cor,
        "n_keep_kaiser": n_keep_kaiser,
        "n_keep_parallel": n_keep_pa,
        "eigenvalues": eigs,
        "loadings": loads,
        "scree": scree,
    }


def _plot(scree: pd.DataFrame, loadings: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k = min(12, len(scree))
    axes[0].errorbar(np.arange(1, k + 1), scree["eigenvalue_correlation"][:k],
                     yerr=[scree["eigenvalue_correlation"][:k] - scree["boot_ci_lower"][:k],
                           scree["boot_ci_upper"][:k] - scree["eigenvalue_correlation"][:k]],
                     fmt="o", capsize=3, label="correlation PCA (95% CI)")
    axes[0].plot(np.arange(1, k + 1), scree["parallel_analysis_95th"][:k],
                 "--", label="parallel analysis 95th pct")
    axes[0].axhline(1.0, color="grey", lw=0.8)
    axes[0].set_xlabel("component"); axes[0].set_ylabel("eigenvalue")
    axes[0].set_title("Scree plot (correlation PCA)")
    axes[0].legend()
    lk = min(8, loadings.shape[1] - 1)
    im = axes[1].imshow(loadings.iloc[:, 1:lk + 1].to_numpy(), aspect="auto",
                        cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_yticks(range(len(loadings))); axes[1].set_yticklabels(loadings["feature"], fontsize=7)
    axes[1].set_xticks(range(lk)); axes[1].set_xticklabels(
        [f"PC{i+1}" for i in range(lk)], fontsize=8)
    axes[1].set_title("Loadings heatmap")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rainfall = pd.read_csv(INTERIM / "uganda_rainfall_features_114.csv")
    daily = pd.read_csv(INTERIM / "uganda_daily_features_climateserv.csv")
    temp = pd.read_csv(INTERIM / "uganda_temperature_features_nasapower.csv")
    matrix = build_feature_matrix(rainfall, daily, temp)
    result = analyze(matrix)
    print(f"rows: {result['n_rows']}, features: {result['n_features']}")
    print(f"keep: 0.85 variance -> {result['n_keep_0.85']}, "
          f"Kaiser -> {result['n_keep_kaiser']}, "
          f"parallel analysis -> {result['n_keep_parallel']}")
    print(f"top-5 eigenvalues: {np.round(result['eigenvalues'][:5], 2)}")


if __name__ == "__main__":
    main()
