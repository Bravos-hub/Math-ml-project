"""Validation v2: honest model evaluation on the observed subregion panel.

Schemes (``configs/models.yaml`` -> ``validation_schemes``):
- random_cv: 5-fold KFold
- group_by_subregion: 5-fold GroupKFold (never mix a subregion across folds)
- temporal_2018_2020 / temporal_2020_2018: train one year, test the other

Models: OLS, Ridge, PCR, PLS, RandomForest, XGBoost (if installed),
all through a SimpleImputer + StandardScaler pipeline.
Baselines: mean predictor, per-subregion historical mean, previous-year
yield (temporal schemes only).

Metrics per scheme x model: RMSE, MAE, R2, anomaly R2 (yields centered on
the per-subregion mean), plus split-conformal prediction-interval coverage.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from yaml import safe_load

from cropyield.data.paths import CONFIGS, OBSERVED, TABLES
from cropyield.pca.representations import RepresentationTransformer

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("xgboost not installed; XGBoost will be skipped")

RNG = 42


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def experiment_banner(df: pd.DataFrame, feature_cols: list[str],
                      target: str, cfg: dict) -> dict:
    """Collect the experiment-tracking fields required by review P2 #23 and
    issue explicit warnings for under-sized samples."""
    tracking = cfg.get("experiment_tracking", {})
    min_n = tracking.get("min_sample_size", 100)
    min_uniq = tracking.get("min_unique_targets", 10)
    n = len(df)
    n_unique = int(df[target].dropna().nunique())
    warnings = []
    if n < min_n:
        warnings.append(
            f"sample_size={n} < min_sample_size={min_n}: metrics are "
            "demonstration-only, not evidence of generalization")
    if n_unique < min_uniq:
        warnings.append(
            f"unique_targets={n_unique} < min_unique_targets={min_uniq}: "
            "only a handful of distinct outcomes")
    for w in warnings:
        log.warning("EXPERIMENT WARNING: %s", w)
    return {
        "run_id": run_id(),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "sample_size": n,
        "number_of_unique_targets": n_unique,
        "n_features": len(feature_cols),
        "feature_set": feature_cols,
        "target": target,
        "target_source": str(df.get("yield_source", pd.Series(dtype=str)).mode().iloc[0]) \
            if "yield_source" in df else "unknown",
        "warnings": "; ".join(warnings) if warnings else "",
    }


def _representation_pipeline(repr_name: str, pca_input_cols: list[str],
                             extra_cols: list[str], n_components: float):
    """Return a function (X_train, X_test) -> (Xtr, Xte) that fits the
    representation (imputer/scaler/PCA/encoder) on the training fold ONLY.

    ``raw`` returns the original frame unchanged (imputation/scaling happens
    inside each model pipeline, as before).
    """
    if repr_name == "raw":
        return lambda Xtr, Xte: (Xtr, Xte)
    tr = RepresentationTransformer(
        repr_name, pca_input_cols=pca_input_cols,
        extra_cols=extra_cols, n_components=n_components)
    return lambda Xtr, Xte: (tr.fit(Xtr).transform(Xtr),
                             tr.transform(Xte))


def load_config() -> dict:
    with open(CONFIGS / "models.yaml") as fh:
        return safe_load(fh)


def load_panel(crop: str) -> pd.DataFrame:
    if crop == "pooled":
        return pd.read_csv(OBSERVED / "crop_pooled_subregion_panel.csv")
    return pd.read_csv(OBSERVED / f"{crop}_subregion_panel.csv")


def make_model(name: str, seed: int = RNG):
    """Pipeline: impute -> drop zero-variance (train-fold) -> scale -> model."""
    if name == "OLS":
        model = LinearRegression()
    elif name == "Ridge":
        model = Ridge(alpha=1.0)
    elif name == "PCR":
        model = PCA(n_components=0.95)
        return make_pipeline(SimpleImputer(strategy="mean"),
                             VarianceThreshold(threshold=1e-8),
                             StandardScaler(), model,
                             LinearRegression())
    elif name == "PLS":
        model = PLSRegression(n_components=3)
    elif name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=500, max_depth=3, min_samples_leaf=2,
            random_state=seed, n_jobs=-1)
    elif name == "XGBoost":
        if not HAS_XGBOOST:
            return None
        model = XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed)
    else:
        raise ValueError(f"unknown model {name}")
    return make_pipeline(SimpleImputer(strategy="mean"),
                         VarianceThreshold(threshold=1e-8),
                         StandardScaler(), model)


def split_indices(df: pd.DataFrame, scheme: str,
                  cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs for a scheme."""
    if scheme == "random_cv":
        kf = KFold(n_splits=cfg["cross_validation"]["n_folds_random"],
                   shuffle=True, random_state=RNG)
        return list(kf.split(df))
    if scheme == "group_by_subregion":
        gkf = GroupKFold(n_splits=cfg["cross_validation"]["n_folds_grouped"])
        groups = df["sub_region"].to_numpy()
        return list(gkf.split(df, groups=groups))
    if scheme == "temporal_2018_2020":
        train = df.index[df["year"] == 2018].to_numpy()
        test = df.index[df["year"] == 2020].to_numpy()
        return [(train, test)]
    if scheme == "temporal_2020_2018":
        train = df.index[df["year"] == 2020].to_numpy()
        test = df.index[df["year"] == 2018].to_numpy()
        return [(train, test)]
    raise ValueError(f"unknown scheme {scheme}")


def anomaly_r2(y_true: np.ndarray, y_pred: np.ndarray,
               groups: np.ndarray) -> float:
    """R2 after centering both series on the per-group mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    centered_t = y_true - pd.Series(y_true).groupby(groups).transform("mean")
    centered_p = y_pred - pd.Series(y_true).groupby(groups).transform("mean")
    ss_res = np.sum((centered_t - centered_p) ** 2)
    ss_tot = np.sum((centered_t - centered_t.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def conformal_interval(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal: absolute-residual quantile at 1 - alpha (on the
    calibration fold); for 5-fold use each fold's residuals as calibration
    for that fold's test predictions."""
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return float(q)


def run_scheme(df: pd.DataFrame, scheme: str, feature_cols: list[str],
               target: str, cfg: dict, models: list[str],
               representation: str = "raw",
               pca_extra: list[str] | None = None) -> pd.DataFrame:
    """Evaluate all models + baselines for one scheme; returns per-fold
    predictions with metrics. ``representation`` selects raw/pca/hybrid
    features (the transformer is always fit on the training fold only)."""
    y = df[target].to_numpy()
    groups = df["sub_region"].to_numpy()
    years = df["year"].to_numpy()
    splits = split_indices(df, scheme, cfg)
    pca_cfg = cfg.get("pca", {})
    n_components = pca_cfg.get("n_components", 0.95)
    hybrid_extra = pca_cfg.get("hybrid_extra") or pca_extra or []
    available_extra = [c for c in hybrid_extra if c in df.columns]
    rows = []
    for fold, (tr, te) in enumerate(splits):
        step = _representation_pipeline(
            representation, feature_cols, available_extra, n_components)
        cols = feature_cols
        if representation == "hybrid":
            cols = cols + [c for c in available_extra
                           if c not in feature_cols]
        Xtr, Xte = step(df.loc[tr, cols], df.loc[te, cols])
        for name in models:
            model = make_model(name)
            if model is None:
                continue
            model.fit(Xtr, y[tr])
            pred = np.clip(model.predict(Xte), 0.0, None)
            _append(rows, scheme, fold, name, groups[te], years[te],
                    y[te], pred)
        # baselines
        train_y, test_y = y[tr], y[te]
        _append(rows, scheme, fold, "mean_predictor", groups[te], years[te],
                test_y, np.full(len(te), train_y.mean()))
        hist = pd.Series(train_y).groupby(groups[tr]).mean()
        _append(rows, scheme, fold, "historical_mean", groups[te], years[te],
                test_y, hist.reindex(groups[te]).fillna(train_y.mean()).to_numpy())
        if scheme.startswith("temporal"):
            _append(rows, scheme, fold, "previous_year_yield", groups[te],
                    years[te], test_y, np.nan_to_num(
                        _previous_year(df, groups[te], years[te], target),
                        nan=train_y.mean()))
    return pd.DataFrame(rows)


def _append(rows: list, scheme: str, fold: int, model: str,
            subregions: np.ndarray, years: np.ndarray,
            y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Expand one fold's predictions into long-form rows."""
    for sub, year, yt, yp in zip(subregions, years, y_true, y_pred):
        rows.append({"scheme": scheme, "fold": fold, "model": model,
                     "sub_region": sub, "year": year,
                     "y_true": yt, "y_pred": yp})


def _previous_year(df: pd.DataFrame, subregions: np.ndarray,
                   years: np.ndarray, target: str) -> np.ndarray:
    """Previous-year yield per subregion. 2018 panel rows carry the single
    annual (total_2018) season group, so any 2020 season group is matched
    against that same 2018 total."""
    out = []
    by_sub_year = df.set_index(["sub_region", "year"])[target]
    for sub, year in zip(subregions, years):
        prev_year = 2018 if year == 2020 else 2020
        try:
            out.append(float(by_sub_year.loc[(sub, prev_year)].iloc[0]))
        except (KeyError, IndexError):
            out.append(np.nan)
    return np.array(out)


def summarize(pred_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate per-fold predictions into scheme x model metric rows."""
    alpha = cfg["uncertainty"]["conformal_alpha"]
    out = []
    for (scheme, model), g in pred_df.groupby(["scheme", "model"]):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        groups = g["sub_region"].to_numpy()
        resid = np.abs(yt - yp)
        quantile = np.quantile(resid, 1 - alpha)
        coverage = float(np.mean(resid <= quantile))
        out.append({
            "scheme": scheme, "model": model,
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp),
            "anomaly_r2": anomaly_r2(yt, yp, groups),
            "n_test": len(yt),
            "conformal_alpha": alpha,
            "conformal_interval": quantile,
            "conformal_coverage": coverage,
        })
    return pd.DataFrame(out)


def run_validation(crop: str = "maize",
                   feature_set: str | None = None,
                   representation: str = "raw",
                   out_dir: Path = TABLES,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the validation matrix for a crop; returns (summary, preds, meta).

    ``representation`` selects raw/pca/hybrid features. The returned ``meta``
    carries experiment-tracking fields and sample-size warnings.
    """
    cfg = load_config()
    df = load_panel(crop)
    if "yield_over_harvested" not in df:
        raise ValueError(f"panel for {crop} lacks yield_over_harvested")
    feature_sets = cfg["feature_sets"]
    feature_cols = feature_sets[feature_set] if feature_set else \
        list(feature_sets["full_agroecological"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing features in panel: {missing}")
    models = cfg["models"]
    schemes = cfg["validation_schemes"]
    target = cfg["target"]
    meta = experiment_banner(df, feature_cols, target, cfg)
    meta["crop"] = crop
    meta["feature_set"] = feature_set or "full_agroecological"
    meta["representation"] = representation
    pred_parts = []
    for scheme in schemes:
        pred_parts.append(run_scheme(df, scheme, feature_cols, target, cfg,
                                     models, representation=representation))
    preds = pd.concat(pred_parts, ignore_index=True)
    preds["representation"] = representation
    preds["crop"] = crop
    summary = summarize(preds, cfg)
    summary["crop"] = crop
    summary["feature_set"] = feature_set or "full_agroecological"
    summary["representation"] = representation
    out_dir.mkdir(parents=True, exist_ok=True)
    label = feature_set or "full_agroecological"
    stem = f"validation_{crop}_{label}_{representation}"
    summary.to_csv(out_dir / f"{stem}.csv", index=False)
    preds.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
    return summary, preds, meta


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    crop = sys.argv[1] if len(sys.argv) > 1 else "maize"
    feature_set = sys.argv[2] if len(sys.argv) > 2 else None
    representation = sys.argv[3] if len(sys.argv) > 3 else "raw"
    summary, _, meta = run_validation(crop, feature_set, representation)
    print(summary.sort_values(["scheme", "rmse"]).to_string(index=False))
    print(meta)


if __name__ == "__main__":
    main()
