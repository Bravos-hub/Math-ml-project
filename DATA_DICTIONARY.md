# Data Dictionary

Project: Agriculture - Predicting Crop Yield in Uganda  
Prepared: July 30, 2026

## Purpose

This document defines the canonical variables for the Uganda agriculture
modeling workflow. The main analytical table is a district-year panel built by
merging rainfall, temperature, soil, and yield data.

Primary modeling target:

- `yield_tons_ha`

Primary unit of analysis:

- one row per `district` x `year`

Core generated datasets in this repository:

- `uganda_rainfall_features.csv`
- `uganda_temperature_features.csv`
- `uganda_soil_features.csv`
- `uganda_rainfall_temperature_soil_features.csv`
- `uganda_full_modeling_dataset.csv`

## Dataset-Level Rules

- `district` is the spatial join key.
- `year` is the temporal join key.
- Soil variables are static or slowly changing and are repeated across years for
  the same district.
- Rainfall and temperature variables are year-specific.
- Yield variables may come from a real UBOS-derived table or a proxy table.
- All continuous predictors should be standardized before PCA or SVR.

## Canonical Schema

| Variable | Type | Unit | Source | Role | Description |
|---|---|---:|---|---|---|
| `district` | string | - | CHIRPS / ERA5 / SoilGrids / UBOS | key | District name used for joins across all tables. |
| `year` | integer | calendar year | CHIRPS / ERA5 / UBOS | key | Observation year. |
| `yield_tons_ha` | float | tonnes per hectare | UBOS or proxy yield table | target | District-level maize yield. This is the main supervised learning target. |

## Rainfall Variables

Source table: `uganda_rainfall_features.csv`

| Variable | Type | Unit | Source | Role | Description |
|---|---|---:|---|---|---|
| `MAM` | float | mm | CHIRPS | predictor | Total rainfall in March-April-May, the main long-rains season. |
| `SON` | float | mm | CHIRPS | predictor | Total rainfall in September-October-November, the short-rains season. |
| `DJF` | float | mm | CHIRPS | predictor | Total rainfall in December-January-February. |
| `JJA` | float | mm | CHIRPS | predictor | Total rainfall in June-July-August. |
| `annual_rainfall` | float | mm | CHIRPS | predictor | Total annual rainfall across all months in the year. |
| `rain_cv` | float | ratio | CHIRPS | predictor | Coefficient of variation of monthly rainfall within a year. Measures rainfall instability. |
| `max_monthly` | float | mm | CHIRPS | predictor | Maximum monthly rainfall observed in the year. |
| `min_monthly` | float | mm | CHIRPS | predictor | Minimum monthly rainfall observed in the year. |
| `rainy_months` | integer | count | CHIRPS | predictor | Number of months with rainfall above the script threshold used during feature engineering. |

## Temperature and Thermal-Time Variables

Source table: `uganda_temperature_features.csv`

| Variable | Type | Unit | Source | Role | Description |
|---|---|---:|---|---|---|
| `MAM_tmax` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean maximum temperature proxy for March-April-May. |
| `MAM_tmin` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean minimum temperature proxy for March-April-May. |
| `MAM_gdd` | float | degree-days | ERA5-Land or synthetic generator | predictor | Growing degree days for March-April-May. |
| `MAM_heat_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Number of heat-stress months or events in MAM, depending on source script. |
| `MAM_cold_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Number of cold-stress months or events in MAM, depending on source script. |
| `SON_tmax` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean maximum temperature proxy for September-October-November. |
| `SON_tmin` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean minimum temperature proxy for September-October-November. |
| `SON_gdd` | float | degree-days | ERA5-Land or synthetic generator | predictor | Growing degree days for September-October-November. |
| `SON_heat_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Heat-stress count for SON. |
| `SON_cold_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Cold-stress count for SON. |
| `DJF_tmax` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean maximum temperature proxy for December-January-February. |
| `DJF_tmin` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean minimum temperature proxy for December-January-February. |
| `DJF_gdd` | float | degree-days | ERA5-Land or synthetic generator | predictor | Growing degree days for DJF. |
| `DJF_heat_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Heat-stress count for DJF. |
| `DJF_cold_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Cold-stress count for DJF. |
| `JJA_tmax` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean maximum temperature proxy for June-July-August. |
| `JJA_tmin` | float | degrees C | ERA5-Land or synthetic generator | predictor | Mean minimum temperature proxy for June-July-August. |
| `JJA_gdd` | float | degree-days | ERA5-Land or synthetic generator | predictor | Growing degree days for JJA. |
| `JJA_heat_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Heat-stress count for JJA. |
| `JJA_cold_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Cold-stress count for JJA. |
| `annual_tmax` | float | degrees C | ERA5-Land or synthetic generator | predictor | Annual mean maximum temperature proxy. |
| `annual_tmin` | float | degrees C | ERA5-Land or synthetic generator | predictor | Annual mean minimum temperature proxy. |
| `annual_gdd` | float | degree-days | ERA5-Land or synthetic generator | predictor | Annual growing degree days. |
| `annual_heat_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Annual heat-stress count. |
| `annual_cold_stress` | integer | count | ERA5-Land or synthetic generator | predictor | Annual cold-stress count. |
| `elevation_m` | float | meters above sea level | district metadata / synthetic generator | predictor | District elevation used as an agroecological covariate. |

## Soil Variables

Source table: `uganda_soil_features.csv`

| Variable | Type | Unit | Source | Role | Description |
|---|---|---:|---|---|---|
| `lat` | float | decimal degrees | SoilGrids query geometry | metadata | District centroid latitude used for point extraction. |
| `lon` | float | decimal degrees | SoilGrids query geometry | metadata | District centroid longitude used for point extraction. |
| `phh2o` | float | pH | SoilGrids | predictor | Soil pH in water. Key indicator of nutrient availability. |
| `soc` | float | g/kg or SoilGrids native unit | SoilGrids | predictor | Soil organic carbon. Proxy for soil fertility and structure. |
| `clay` | float | percent or SoilGrids native unit | SoilGrids | predictor | Clay fraction. Influences water retention and nutrient holding capacity. |
| `sand` | float | percent or SoilGrids native unit | SoilGrids | predictor | Sand fraction. Influences drainage and drought sensitivity. |
| `silt` | float | percent or SoilGrids native unit | SoilGrids | predictor | Silt fraction. Part of soil texture composition. |
| `bdod` | float | SoilGrids native unit | SoilGrids | predictor | Bulk density of fine earth fraction. Related to rooting and compaction constraints. |
| `cec` | float | SoilGrids native unit | SoilGrids | predictor | Cation exchange capacity. Indicates nutrient retention potential. |
| `source_status` | string | - | local processing | quality flag | Indicates whether SoilGrids values were fetched successfully, missing, cached, or empty. |

## Yield and Survey Aggregates

Possible source tables:

- `ubos_maize_yield_district.csv`
- `ubos_district_yield_proxy.csv`

| Variable | Type | Unit | Source | Role | Description |
|---|---|---:|---|---|---|
| `yield_tons_ha` | float | tonnes per hectare | UBOS or proxy | target | District-level maize yield. |
| `n_households` | integer | count | UBOS microdata aggregation | optional predictor / metadata | Number of households contributing to the district-year aggregate. |
| `total_area_ha` | float | hectares | UBOS microdata aggregation | optional predictor / metadata | Total harvested area represented in the district-year aggregate. |
| `total_production_kg` | float | kilograms | UBOS microdata aggregation | optional predictor / metadata | Total production represented in the district-year aggregate. |

## Planned Extension Variables

These fields are part of the intended final project scope but are not yet
guaranteed in the current repository outputs.

| Variable | Type | Unit | Planned Source | Role | Description |
|---|---|---:|---|---|---|
| `fertilizer_kg_ha` | float | kg/ha | UBOS / UNPS | predictor | Fertilizer application intensity. |
| `planting_density` | float | plants/ha or stand count proxy | UNPS / survey engineering | predictor | Plant population density. |
| `ndvi_peak` | float | index value | Sentinel-2 / Landsat / GEE | predictor | Peak NDVI during the growing season. Proxy for canopy vigor. |
| `evi_peak` | float | index value | Sentinel-2 / Landsat / GEE | optional predictor | Peak enhanced vegetation index. |
| `soil_moisture_index` | float | index value | remote sensing / reanalysis | optional predictor | Soil moisture or moisture stress indicator. |
| `planting_date` | date or ordinal day | day-of-year | survey or agronomic reconstruction | predictor | Planting timing variable. |
| `solar_radiation` | float | MJ/m2/day or equivalent | NASA POWER | optional predictor | Available radiation during crop growth. |

## Modeling Roles

Recommended feature subsets:

- Rainfall-only baseline:
  - `MAM`, `SON`, `DJF`, `JJA`, `annual_rainfall`, `rain_cv`, `max_monthly`,
    `min_monthly`, `rainy_months`
- Climate-plus-thermal model:
  - rainfall-only baseline
  - seasonal temperature and `*_gdd` variables
  - `elevation_m`
- Agroecological model:
  - climate-plus-thermal model
  - `phh2o`, `soc`, `clay`, `sand`, `silt`, `bdod`, `cec`
- Full planned project model:
  - agroecological model
  - `fertilizer_kg_ha`, `planting_density`, `ndvi_peak`, and other survey or
    remote sensing extensions

## Data Quality Notes

- District naming must be standardized before merging. For example, case and
  underscore differences should be resolved consistently.
- Temperature variable semantics depend on the generating script. Current
  `uganda_temperature_features.csv` matches the synthetic schema used by the
  integrated pipeline, not the monthly ERA5-Land schema proposed elsewhere.
- SoilGrids units should be confirmed against the exact API response metadata
  before publication-grade reporting.
- If yield data is proxy rather than real UBOS-derived data, this must be
  stated explicitly in any report, notebook, or figure caption.

## Recommended Derived Variables

These are not required raw columns, but they are useful analytical additions.

| Variable | Formula or Rule | Purpose |
|---|---|---|
| `z_*` | feature-wise standardization | Required for PCA, SVR, and distance-based diagnostics. |
| `pc1`, `pc2`, ... | projection of standardized data onto eigenvectors | Principal component scores. |
| `yield_anomaly` | district-year yield minus district mean yield | Removes persistent district fixed effects. |
| `rainfall_anomaly` | annual rainfall minus district mean annual rainfall | Isolates interannual climate shocks. |
| `reconstruction_error` | norm of `X - X_k` after PCA truncation | Useful for studying information loss vs. prediction error. |

## File Mapping

| File | Expected Grain | Notes |
|---|---|---|
| `uganda_rainfall_features.csv` | district-year | Real CHIRPS-derived rainfall features. |
| `uganda_temperature_features.csv` | district-year | Temperature and GDD features; current table matches synthetic integrated pipeline schema. |
| `uganda_soil_features.csv` | district | SoilGrids point extractions and quality flags. |
| `uganda_rainfall_temperature_soil_features.csv` | district-year | Climate panel enriched with static soil variables. |
| `uganda_full_modeling_dataset.csv` | district-year | Climate + soil + yield table, if a yield file is available. |

