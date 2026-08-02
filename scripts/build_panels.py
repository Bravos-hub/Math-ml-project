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

# CV (coefficient of variation, %) above this marks a target estimate as
# low-reliability (kept in the dataset but flagged for sensitivity analysis).
HIGH_CV_THRESHOLD_PCT = 30.0

# SoilGrids properties must be available for at least this share of units
# before soil features are allowed into the modeling dataset (review P0 #8).
MIN_SOIL_COVERAGE = 0.80

# Feature columns that are additive across the two rains seasons.
_ADDITIVE = [
    "season_total_mm", "rain_days_1mm", "rain_days_10mm", "rain_days_20mm",
    "dry_spell_count_7d", "dry_spell_count_10d",
]
_EXTREME = ["longest_dry_spell_days", "maximum_5day_rainfall", "false_onset_flag"]
# Features that cannot be naively summed across seasons; handled explicitly
# in climate_for_season for 'total' season groups.
_DROP_FOR_TOTAL = [
    "season_onset_day", "season_cessation_day", "season_length_days",
    "mean_wet_day_rainfall",
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


def load_soil() -> pd.DataFrame:
    soil = pd.read_csv(INTERIM / "uganda_soil_features_114.csv")
    drop = [c for c in PROVENANCE_COLUMNS if c in soil.columns]
    soil = soil.drop(columns=drop)
    value_cols = [c for c in soil.columns if c in
                  ("clay", "sand", "silt", "soc", "bdod", "cec", "phh2o")]
    if not value_cols:
        raise ValueError("soil table carries no SoilGrids property columns")
    coverage = soil[value_cols].notna().mean().min()
    if coverage < MIN_SOIL_COVERAGE:
        raise ValueError(
            f"SoilGrids coverage below threshold: {coverage:.2f} < "
            f"{MIN_SOIL_COVERAGE}; exclude soil from modeling or re-extract."
        )
    return soil


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
    # Carry meaningful values for the features that were dropped above.
    if "season_onset_day" in drop_cols:
        combined["season_onset_day"] = combined["season_onset_day"] \
            .combine(second["season_onset_day"].reindex(combined.index), min)
    if "season_cessation_day" in drop_cols:
        combined["season_cessation_day"] = combined["season_cessation_day"] \
            .combine(second["season_cessation_day"].reindex(combined.index), max)
    if "season_length_days" in drop_cols:
        on = combined["season_onset_day"]
        ce = combined["season_cessation_day"]
        combined["season_length_days"] = ce - on + 1
    if "mean_wet_day_rainfall" in drop_cols:
        w1 = combined["rain_days_1mm"]
        w2 = second["rain_days_1mm"].reindex(combined.index)
        m1 = combined["mean_wet_day_rainfall"]
        m2 = second["mean_wet_day_rainfall"].reindex(combined.index)
        combined["mean_wet_day_rainfall"] = \
            (m1 * w1 + m2 * w2) / (w1 + w2).replace(0, pd.NA)
    return combined.drop(columns=[c for c in drop_cols if c not in combined]).reset_index()


def subregion_climate(df: pd.DataFrame, districts: pd.DataFrame,
                      district_col: str = "district") -> pd.DataFrame:
    """Aggregate district-level climate features to subregions (mean)."""
    m = df.merge(districts[[district_col, "sub_region"]], on=district_col, how="left")
    feat_cols = [c for c in df.columns
                 if c not in (district_col, "sub_region")
                 and pd.api.types.is_numeric_dtype(df[c])]
    return m.groupby("sub_region")[feat_cols].mean().reset_index()


def add_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Derive survey uncertainty fields from the stored production CV.

    AAS tables report coefficient-of-variation percentages for area and
    production. ``cv_production_pct`` is carried through as ``target_cv``
    (percent); ``target_reliability_weight`` is an inverse-CV weight suitable
    for weighted regression; ``high_uncertainty_flag`` marks estimates whose
    CV exceeds ``HIGH_CV_THRESHOLD_PCT``.

    ``yield_consistency_ok`` records whether ``production / harvested
    area`` reproduces the published ``yield_over_harvested``. AAS 2018
    publishes total production with second-season harvested yield for annual
    crops, so its 2018 total rows legitimately fail this check and must be
    documented rather than silently removed.
    """
    df = df.copy()
    cv = df.get("cv_production_pct")
    if cv is None or cv.isna().all():
        df["target_cv"] = pd.NA
        df["target_reliability_weight"] = pd.NA
        df["high_uncertainty_flag"] = pd.NA
    else:
        cv = pd.to_numeric(cv, errors="coerce")
        df["target_cv"] = cv
        eps = 1.0
        df["target_reliability_weight"] = (eps / (cv / 100.0)).clip(upper=10.0)
        df["high_uncertainty_flag"] = (cv >= HIGH_CV_THRESHOLD_PCT).map({True: 1, False: 0})

    calc = df["production_mt"] / df["area_harvested_ha"]
    df["yield_consistency_ok"] = (calc == 0) | (
        (calc - df["yield_over_harvested"]).abs() <= 0.02 * df["yield_over_harvested"].abs()
    )
    return df


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
    soil = load_soil()
    soil_subregion = subregion_climate(
        soil, districts, district_col="district"
    ).rename(columns={c: f"soil_{c}" for c in soil.columns
                      if c not in ("district", "sub_region")})
    panel = panel.merge(soil_subregion, on="sub_region", how="left")
    return add_reliability(panel), districts


def assigned_from_subregion(panel: pd.DataFrame,
                            districts: pd.DataFrame) -> pd.DataFrame:
    """Assign subregion yields and climate to districts (grade B)."""
    panel = panel.drop(columns=[c for c in panel.columns
                                if c.startswith("soil_")])
    merged = panel.merge(districts, on="sub_region", how="inner")
    soil = load_soil()
    merged = merged.merge(
        soil.rename(columns={c: f"soil_{c}" for c in soil.columns
                             if c not in ("district", "lat", "lon")}),
        on="district", how="left",
    )
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


def build_pooled_panel() -> pd.DataFrame:
    """Stack the three crop panels into one subregion x crop x season x year
    table so that the analysis has more than 100 genuinely distinct
    analytical units (review P0 #3). ``crop`` becomes an explicit predictor."""
    parts = []
    for crop in ("maize", "beans", "groundnuts"):
        panel = pd.read_csv(OBSERVED / f"{crop}_subregion_panel.csv")
        parts.append(panel.assign(crop=crop))
    pooled = pd.concat(parts, ignore_index=True)
    return pooled.reindex(columns=["crop"] + [c for c in pooled.columns
                                              if c != "crop"])


def main() -> None:
    crop = sys.argv[1] if len(sys.argv) > 1 else "maize"
    for c in ("maize", "beans", "groundnuts"):
        panel, districts = build_panel(c)

        observed = add_panel_provenance(panel, assigned=False)
        observed_out = OBSERVED / f"{c}_subregion_panel.csv"
        observed_out.parent.mkdir(parents=True, exist_ok=True)
        observed.to_csv(observed_out, index=False)

        assigned = add_panel_provenance(assigned_from_subregion(panel, districts),
                                        assigned=True)
        assigned_out = ASSIGNED / f"{c}_district_assigned_panel.csv"
        assigned_out.parent.mkdir(parents=True, exist_ok=True)
        assigned.to_csv(assigned_out, index=False)

        print(f"[{c}] observed panel: {len(observed)} rows -> {observed_out}")
        print(f"[{c}] assigned panel: {len(assigned)} rows -> {assigned_out}")

    pooled = add_panel_provenance(build_pooled_panel(), assigned=False)
    pooled_out = OBSERVED / "crop_pooled_subregion_panel.csv"
    pooled.to_csv(pooled_out, index=False)
    print(f"[pooled] subregion x crop x season x year: {len(pooled)} rows -> "
          f"{pooled_out}")

    from cropyield.reporting.feature_availability import write_availability_report
    planned = []
    try:
        from cropyield.config import load_features_config
        planned = load_features_config().get("daily_required_features", [])
    except Exception:
        pass
    write_availability_report(
        pooled,
        provenance_columns=tuple(PROVENANCE_COLUMNS) +
        ("crop", "sub_region", "season_group", "year", "district"),
        planned=planned)


if __name__ == "__main__":
    main()
