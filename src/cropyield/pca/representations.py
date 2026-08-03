"""Feature representations for modeling (review P1 #18).

Three leak-free representations, applied inside each training fold:

* ``raw``   - the original feature columns (only imputation/scaling in-fold).
* ``pca``   - PCA scores computed from the training fold only, with n_components
              chosen by cumulative-variance (or a fixed integer).
* ``hybrid``- PCA scores from the continuous climate features plus the
              original continuous variables that were NOT included in the
              PCA. Context variables (e.g. crop and season) are kept outside
              PCA for all three representations, so comparisons are fair.

All transformations (imputer, scaler, PCA, one-hot) are fit on the training
indices and applied to the test fold, so no test-fold information leaks.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RepresentationTransformer:
    """Fit on :class:`X_train`, transform any fold's features without leakage.

    Parameters
    ----------
    representation : {"raw", "pca", "hybrid"}
        Which feature representation to build.
    pca_input_cols : list[str] | None
        Columns passed to PCA. ``None`` selects all numeric columns.
    extra_cols : list[str] | None
        Columns kept verbatim (one-hot) in the hybrid representation. Ignored
        for ``raw``/``pca``.
    n_components : int | float
        Absolute component count or cumulative-variance fraction (0 < k < 1).
    scale : bool
        Standardize PCA inputs to their correlation matrix.
    """

    def __init__(self, representation: str = "raw",
                 pca_input_cols: list[str] | None = None,
                 extra_cols: list[str] | None = None,
                 n_components: int | float = 0.95,
                 scale: bool = True) -> None:
        self.representation = representation
        self.pca_input_cols = pca_input_cols
        self.extra_cols = extra_cols
        self.n_components = n_components
        self.scale = scale
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._encoder: OneHotEncoder | None = None
        self._encoded_cols: list[str] = []
        self._pca_cols: list[str] = []
        self._extra_cols: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "RepresentationTransformer":
        if self.pca_input_cols is None:
            self.pca_input_cols = [
                c for c in X_train.columns
                if pd.api.types.is_numeric_dtype(X_train[c])
            ]
        self._pca_cols = list(self.pca_input_cols)
        # Context columns are deliberately kept outside PCA for every
        # representation.  This makes raw, PCA, and hybrid comparisons
        # differ only in how continuous environmental predictors are
        # represented.
        if self.extra_cols is None and self.representation == "hybrid":
            extra = [c for c in X_train.columns if c not in self._pca_cols]
        else:
            extra = [c for c in (self.extra_cols or []) if c in X_train.columns]
        self._extra_cols = extra
        if extra:
            self._encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            self._encoder.fit(X_train[extra])
            self._encoded_cols = (
                self._encoder.get_feature_names_out(extra).tolist()
            )
        if self.representation == "raw":
            return self
        self._imputer = SimpleImputer(strategy="median")
        cont = self._imputer.fit_transform(X_train[self._pca_cols])
        if self.scale:
            self._scaler = StandardScaler()
            cont = self._scaler.fit_transform(cont)
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(cont)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.representation == "raw":
            parts = [X[self._pca_cols].copy()]
            if self._encoder is not None:
                extra = [c for c in self._extra_cols if c in X.columns]
                encoded = self._encoder.transform(X[extra])
                parts.append(pd.DataFrame(
                    encoded, index=X.index, columns=self._encoded_cols
                ))
            return pd.concat(parts, axis=1)
        cont = X[self._pca_cols]
        if self._imputer is not None:
            cont = self._imputer.transform(cont)
        if self._scaler is not None:
            cont = self._scaler.transform(cont)
        scores = self._pca.transform(cont)
        parts = [pd.Series(scores[:, i], index=X.index, name=f"PC{i + 1}")
                 for i in range(scores.shape[1])]
        if self._encoder is not None:
            extra = [c for c in self._extra_cols if c in X.columns]
            encoded = self._encoder.transform(X[extra])
            for name, values in zip(self._encoded_cols, encoded.T):
                parts.append(pd.Series(values, index=X.index, name=name))
        return pd.concat(parts, axis=1)


def fit_representation(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       representation: str,
                       pca_input_cols: list[str] | None = None,
                       extra_cols: list[str] | None = None,
                       n_components: int | float = 0.95,
                       ) -> pd.DataFrame:
    """Convenience wrapper: fit on train, return transformed test features."""
    t = RepresentationTransformer(
        representation, pca_input_cols=pca_input_cols,
        extra_cols=extra_cols, n_components=n_components)
    return t.fit(X_train).transform(X_test)


def split_pca_and_extra(df: pd.DataFrame, extra_cols: list[str]) -> tuple:
    """Given a full feature frame, return (pca_input_cols, extra_cols) where
    extra columns are kept whole in the hybrid representation."""
    pca_cols = [c for c in df.columns
                if c not in extra_cols and
                pd.api.types.is_numeric_dtype(df[c])]
    return pca_cols, extra_cols
