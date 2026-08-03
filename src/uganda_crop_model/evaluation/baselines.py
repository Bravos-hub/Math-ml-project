"""Training-fold-only baselines for the authoritative panel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_training_baselines(data, train_index, test_index):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    y = pd.to_numeric(train["yield_tons_ha"], errors="raise")
    global_mean = float(y.mean())
    sub_mean = train.groupby("spatial_unit")["yield_tons_ha"].mean()
    sub_crop = train.groupby(["spatial_unit", "crop"])["yield_tons_ha"].mean()
    previous = train.groupby(["spatial_unit", "crop", "season"])["yield_tons_ha"].mean()

    def lookup(table, keys, fallback):
        values = []
        fallbacks = 0
        for row in test.itertuples(index=False):
            key = tuple(getattr(row, k) for k in keys)
            if len(key) == 1:
                key = key[0]
            try:
                value = table.loc[key]
                if pd.isna(value):
                    raise KeyError
            except KeyError:
                value = fallback
                fallbacks += 1
            values.append(float(value))
        return np.asarray(values), fallbacks

    # Historical means are fit only on training observations. Previous-wave
    # matching is exact on available historical rows; unseen matches use the
    # corresponding subregion/crop mean, then the global training mean.
    sub_pred, sub_fb = lookup(sub_mean, ("spatial_unit",), global_mean)
    sub_crop_pred, sc_fb = lookup(sub_crop, ("spatial_unit", "crop"), global_mean)
    prev_pred, prev_fb = lookup(previous, ("spatial_unit", "crop", "season"), global_mean)
    return {
        "historical_subregion_mean": (sub_pred, sub_fb),
        "historical_subregion_crop_mean": (sub_crop_pred, sc_fb),
        "previous_wave_yield": (prev_pred, prev_fb),
    }
