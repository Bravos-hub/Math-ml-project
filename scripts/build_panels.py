#!/usr/bin/env python3
"""Build the observed subregion panel and the district-assigned panel.

Panels combine official AAS yields (grade A at subregion level) with climate
features from the district centroids (CHIRPS monthly + daily via ClimateSERV,
NASA POWER temperature). District-level climate features are aggregated to
subregions by the mean over the subregion's districts (documented derivation).

Outputs
-------
data/processed/observed/{crop}_subregion_panel.csv
    14 subregions x (2020: 3 season groups, 2018: 1 annual) rows per crop.
data/processed/assigned/{crop}_district_assigned_panel.csv
    114 districts x same season structure; yields assigned from subregion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropyield.data.paths import (  # noqa: E402
    ASSIGNED,
    INTERIM,
    OBSERVED,
)
from cropyield.data.provenance import (  # noqa: E402
    PROVENANCE_COLUMNS,
    RAIN_CHIRPS_DAILY,
    RAIN_CHIRPS_MONTHLY,
    TEMP_NASA_POWER,
    YIELD_AAS2018_SUBREGION,
    YIELD_AAS2020_SUBREGION,
    add_provenance,
)

# ---------------------------------------------------------------------------
# Season-group mapping
# ---------------------------------------------------------------------------

SEASON_MAP = {
    "first_season_2020": ("first", 2020),
    "second_season_2020": ("second", 2020),
    "total_2020": ("total", 2020),
}

# Feature columns that are additive across the two rains seasons.
_ADDITIVE = [
    "season_total_mm", "rain_days_1mm", "rain_days_10mm", "rain_days_20mm",
    "dry_spell_count_7d", "dry_spell_count_10d",
]
_EXTREME = ["longest_dry_spell_days", "maximum_5day_rainfall"]
_DROP_FOR_TOTAL = [
    "season_onset_day", "season_cessation_day", "season_length_days",
    "false_onset_flag", "mean_wet_day_rainfall",
]
_TEMP_ADDITIVE = [
    "season_gdd", "heat_days", "extreme_heat_days", "warm_night_days",
    "heatwave_count", "wet_day_tmax_mean", "wet_day_gdd",
]


def load_climate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    districts = pd.read_csv(INTERIM / "uganda_districts_114.csv")
    monthly = pd.read_csv(INTERIM / "uganda_rainfall_features_114.csv")
    daily = pd.read_csv(INTERIM / "uganda_daily_features_climateserv.csv")
    temp = pd.read_csv(INTERIM / "uganda_temperature_features_nasapower.csv")
    drop = [c for c in PROVENANCE_COLUMNS if c in daily.columns]
    daily = daily.drop(columns=drop)
    drop = [c for c in PROVENANCE_COLUMNS if c in temp.columns]
    temp = temp.drop(columns=drop)
    return districts, monthly, daily, temp


def climate_for_season(district_climate: pd.DataFrame,
                       district_col: str, year_col: str,
                       season_col: str, year: int, season: str) -> pd.DataFrame:
    """Filter per-district seasonal climate for (year, season); 'total'
    combines the first and second seasons (sum except maxima/means)."""
    if season != "total":
        return district_climate[
            (district_climate[year_col] == year)
            & (district_climate[season_col] == season)
        ]
    first = district_climate[
        (district_climate[year_col] == year)
        & (district_climate[season_col] == "first")
    ].set_index(district_col)
    second = district_climate[
        (district_climate[year_col] == year)
        & (district_climate[season_col] == "second")
    ].set_index(district_col)
    combined = first.copy()
    drop_cols = [c for c in _DROP_FOR_TOTAL if c in combined]
    keep = [c for c in combined.columns
            if c not in drop_cols and c not in (year_col, season_col)]
    for col in keep:
        a = combined[col]
        b = second[col].reindex(combined.index)
        if col == "wet_day_tmax_mean":
            combined[col] = (a + b) / 2
        elif col in _EXTREME:
            combined[col] = a.combine(b, max)
        else:
            combined[col] = a + b.fillna(0)
    return combined.drop(columns=drop_cols).reset_index()


def subregion_climate(df: pd.DataFrame, districts: pd.DataFrame,
                      district_col: str = "district") -> pd.DataFrame:
    """Aggregate district-level climate features to subregions (mean)."""
    m = df.merge(districts[[district_col, "sub_region"]], on=district_col, how="left")
    feat_cols = [c for c in df.columns
                 if c not in (district_col, "sub_region")
                 and pd.api.types.is_numeric_dtype(df[c])]
    return m.groupby("sub_region")[feat_cols].mean().reset_index()


def build_panel(crop: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    aas20 = pd.read_csv(INTERIM / "aas2020_subregion_consolidated.csv")
    aas18 = pd.read_csv(INTERIM / "aas2018_subregion_consolidated.csv")
    districts, monthly, daily, temp = load_climate()

    yields = pd.concat([
        aas20[aas20["crop"] == crop][
            ["sub_region", "season_group", "area_planted_ha", "area_harvested_ha",
             "production_mt", "yield_over_harvested", "yield_over_planted",
             "cv_area_planted_pct", "cv_area_harvested_pct", "cv_production_pct",
             "harvest_loss_ratio", "year"]
        ].assign(year=2020),
        aas18[aas18["crop"] == crop][
            ["sub_region", "area_planted_ha", "area_harvested_ha", "production_mt",
             "yield_over_harvested", "yield_over_planted", "cv_production_pct",
             "harvest_loss_ratio"]
        ].assign(year=2018, season_group="total_2018",
                 cv_area_planted_pct=pd.NA, cv_area_harvested_pct=pd.NA),
    ], ignore_index=True)

    climate_rows = []
    for (year, season_group), g in yields.groupby(["year", "season_group"]):
        if season_group.startswith("total"):
            season, y = "total", year
        else:
            season, y = SEASON_MAP[season_group]
        d = climate_for_season(daily, "district", "year", "season", y, season)
        t = climate_for_season(temp, "district", "year", "season", y, season)
        m = monthly[monthly["year"] == y][
            ["district", "MAM", "SON", "DJF", "JJA", "annual_rainfall", "rain_cv",
             "max_monthly", "min_monthly", "rainy_months", "rainfall_anomaly",
             "rainfall_zscore", "MAM_anomaly", "MAM_zscore", "SON_anomaly",
             "SON_zscore"]
        ]
        for _, row in g.iterrows():
            sub = row["sub_region"]
            climate_rows.append({
                "sub_region": sub,
                "year": year,
                "season_group": season_group,
                **{f"daily_{c}": v for c, v in
                   subregion_climate(d, districts).set_index("sub_region")
                   .loc[sub].items()},
                **{f"temp_{c}": v for c, v in
                   subregion_climate(t, districts).set_index("sub_region")
                   .loc[sub].items()},
                **{f"rain_{c}": v for c, v in
                   subregion_climate(m, districts).set_index("sub_region")
                   .loc[sub].items()},
            })

    panel = yields.merge(pd.DataFrame(climate_rows),
                         on=["sub_region", "year", "season_group"], how="left")
    return panel, districts


def assigned_from_subregion(panel: pd.DataFrame,
                            districts: pd.DataFrame) -> pd.DataFrame:
    """Assign subregion yields and climate to districts (grade B)."""
    climate_cols = [c for c in panel.columns
                    if c.startswith(("daily_", "temp_", "rain_"))]
    merged = panel.merge(districts, on="sub_region", how="inner")
    return merged


def add_panel_provenance(df: pd.DataFrame, *, assigned: bool) -> pd.DataFrame:
    note = (
        "Yields from official AAS subregion estimates. Climate features from "
        "district centroids (CHIRPS monthly + CHIRPS daily via ClimateSERV + "
        "NASA POWER), aggregated to subregion by district mean."
        if not assigned else
        "Subregion panel assigned to districts via official AAS district "
        "grouping; district-level climate features used directly."
    )
    return add_provenance(
        df,
        yield_source=YIELD_AAS2020_SUBREGION,
        rainfall_source=f"{RAIN_CHIRPS_MONTHLY}+{RAIN_CHIRPS_DAILY}",
        temperature_source=TEMP_NASA_POWER,
        yield_granularity="district" if assigned else "subregion",
        quality_note=note,
    )


def main() -> None:
    crop = sys.argv[1] if len(sys.argv) > 1 else "maize"
    panel, districts = build_panel(crop)
    OBSERVED.mkdir(parents=True, exist_ok=True)
    ASSIGNED.mkdir(parents=True, exist_ok=True)

    observed = add_panel_provenance(panel, assigned=False)
    observed_out = OBSERVED / f"{crop}_subregion_panel.csv"
    observed_out.parent.mkdir(parents=True, exist_ok=True)
    observed.to_csv(observed_out, index=False)

    assigned = add_panel_provenance(assigned_from_subregion(panel, districts),
                                    assigned=True)
    assigned_out = ASSIGNED / f"{crop}_district_assigned_panel.csv"
    assigned_out.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(assigned_out, index=False)

    print(f"[{crop}] observed panel: {len(observed)} rows -> {observed_out}")
    print(f"[{crop}] assigned panel: {len(assigned)} rows -> {assigned_out}")


if __name__ == "__main__":
    main()
