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

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("xgboost not installed; XGBoost will be skipped")

RNG = 42


def load_config() -> dict:
    with open(CONFIGS / "models.yaml") as fh:
        return safe_load(fh)


def load_panel(crop: str) -> pd.DataFrame:
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
               target: str, cfg: dict,
               models: list[str]) -> pd.DataFrame:
    """Evaluate all models + baselines for one scheme; returns per-fold
    predictions with metrics."""
    X = df[feature_cols]
    y = df[target].to_numpy()
    groups = df["sub_region"].to_numpy()
    years = df["year"].to_numpy()
    splits = split_indices(df, scheme, cfg)
    rows = []
    for fold, (tr, te) in enumerate(splits):
        for name in models:
            model = make_model(name)
            if model is None:
                continue
            model.fit(X.iloc[tr], y[tr])
            pred = np.clip(model.predict(X.iloc[te]), 0.0, None)
            rows.append({
                "scheme": scheme, "fold": fold, "model": name,
                "sub_region": groups[te], "year": years[te],
                "y_true": y[te], "y_pred": pred,
            })
        # baselines
        train_y, test_y = y[tr], y[te]
        rows.append({
            "scheme": scheme, "fold": fold, "model": "mean_predictor",
            "sub_region": groups[te], "year": years[te],
            "y_true": test_y,
            "y_pred": np.full(len(te), train_y.mean()),
        })
        hist = pd.Series(train_y).groupby(groups[tr]).mean()
        rows.append({
            "scheme": scheme, "fold": fold, "model": "historical_mean",
            "sub_region": groups[te], "year": years[te],
            "y_true": test_y,
            "y_pred": hist.reindex(groups[te]).fillna(train_y.mean()).to_numpy(),
        })
        if scheme.startswith("temporal"):
            rows.append({
                "scheme": scheme, "fold": fold, "model": "previous_year_yield",
                "sub_region": groups[te], "year": years[te],
                "y_true": test_y,
                "y_pred": np.nan_to_num(
                    _previous_year(df, groups[te], years[te], target),
                    nan=train_y.mean()),
            })
    return pd.DataFrame(rows)


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
        yt = np.concatenate(g["y_true"].to_numpy())
        yp = np.concatenate(g["y_pred"].to_numpy())
        groups = np.concatenate(g["sub_region"].to_numpy())
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
                   out_dir: Path = TABLES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full validation matrix for a crop; returns (summary, preds)."""
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
    pred_parts = []
    for scheme in schemes:
        pred_parts.append(run_scheme(df, scheme, feature_cols, target, cfg,
                                     models))
    preds = pd.concat(pred_parts, ignore_index=True)
    summary = summarize(preds, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = feature_set or "full_agroecological"
    summary.to_csv(out_dir / f"validation_{crop}_{label}.csv", index=False)
    preds.to_csv(out_dir / f"validation_{crop}_{label}_predictions.csv",
                 index=False)
    return summary, preds


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    crop = sys.argv[1] if len(sys.argv) > 1 else "maize"
    feature_set = sys.argv[2] if len(sys.argv) > 2 else None
    summary, _ = run_validation(crop, feature_set)
    print(summary.sort_values(["scheme", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
