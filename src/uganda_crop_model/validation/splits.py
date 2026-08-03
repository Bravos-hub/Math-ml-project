"""Validation splits for a subregion x year x season x crop panel.

Because the data are a panel with multiple spatial units and seasons per
year, ordinary shuffled K-fold would leak a spatial unit or a year across
folds.  The blue-print therefore requires explicit year-based (rolling
origin) and spatial-group (GroupKFold) splits, plus a harder
future-unseen-location stress test (blueprint section 17).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

Split = tuple[np.ndarray, np.ndarray]


def rolling_origin_year_splits(
    metadata: pd.DataFrame,
    *,
    minimum_training_years: int = 3,
) -> Iterator[Split]:
    """Yield (train, test) index pairs where test is a single future year.

    Every observation in a test year is held out, so no test row is ever
    available during training.  Raises when there are not enough distinct
    years to form even one such split.
    """

    years = np.array(sorted(metadata["year"].dropna().unique()))

    if len(years) <= minimum_training_years:
        raise ValueError(
            "Not enough years for rolling temporal validation: "
            f"found {len(years)} year(s), need more than "
            f"{minimum_training_years}."
        )

    for position in range(minimum_training_years, len(years)):
        test_year = years[position]

        train_mask = metadata["year"].lt(test_year)
        test_mask = metadata["year"].eq(test_year)

        train_indices = np.flatnonzero(train_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())

        if train_indices.size == 0 or test_indices.size == 0:
            continue

        yield train_indices, test_indices


def spatial_group_splits(
    metadata: pd.DataFrame,
    *,
    requested_splits: int = 5,
    random_seed: int = 42,
) -> list[Split]:
    """GroupKFold splits where each fold keeps spatial units disjoint."""

    groups = metadata["spatial_unit"].astype(str)
    number_of_groups = groups.nunique()

    if number_of_groups < 2:
        raise ValueError("At least two spatial groups are required.")

    number_of_splits = min(requested_splits, number_of_groups)

    splitter = GroupKFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=random_seed,
    )

    dummy_X = np.zeros((len(metadata), 1))

    return list(
        splitter.split(
            dummy_X,
            groups=groups.to_numpy(),
        )
    )


def future_unseen_location_splits(
    metadata: pd.DataFrame,
    *,
    minimum_training_years: int = 3,
    minimum_test_rows: int = 1,
) -> Iterator[Split]:
    """Stress test: test rows are both in a future year and from a spatial
    unit that was excluded from training."""

    years = sorted(metadata["year"].dropna().unique())
    spatial_units = sorted(
        metadata["spatial_unit"].astype(str).unique()
    )

    for year_position in range(
        minimum_training_years,
        len(years),
    ):
        test_year = years[year_position]

        for test_spatial_unit in spatial_units:
            train_mask = (
                metadata["year"].lt(test_year)
                & metadata["spatial_unit"]
                .astype(str)
                .ne(test_spatial_unit)
            )

            test_mask = (
                metadata["year"].eq(test_year)
                & metadata["spatial_unit"]
                .astype(str)
                .eq(test_spatial_unit)
            )

            train_indices = np.flatnonzero(train_mask.to_numpy())
            test_indices = np.flatnonzero(test_mask.to_numpy())

            if (
                train_indices.size > 0
                and test_indices.size >= minimum_test_rows
            ):
                yield train_indices, test_indices