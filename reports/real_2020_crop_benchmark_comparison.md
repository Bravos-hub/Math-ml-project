# 2020 Real-Data Crop Benchmark Comparison

> **evaluation_status = "demonstration_only"** (review P0 #2)
>
> This document describes the first five-row 2020 benchmark (also referred
> to as `pipeline_smoke_test_2020`) that assigned AAS 2020 sub-region yields
> to five Eastern Uganda districts. It is **not** a performance benchmark:
> with `n = 5` rows, seven predictors and only ~3 distinct target values, the
> R² figures below are an artefact of the tiny, pseudo-replicated sample and
> must not be cited as evidence of district-level yield predictability.
> Formal model comparison now lives in `reports/tables/validation_all.csv`
> (subregion x season x year units, 2018 & 2020).

This memo compares the first three Eastern Uganda `2020` real-data benchmarks
built from UBOS AAS 2020 sub-region yields assigned to districts.

## Scope

- Geography: `Iganga`, `Jinja`, `Kapchorwa`, `Mbale`, `Tororo`
- Year: `2020`
- Predictors: `MAM`, `SON`, `annual_rainfall`, `rain_cv`, `annual_gdd`, `elevation_m`, `soil_moisture_index`
- Yield source: `AAS2020_subregion_assigned_to_district` (pseudo-replicated)
- Validation: leave-one-out cross-validation on `5` district rows
- evaluation_status: demonstration_only

## Benchmark Summary

| Crop | Best model | Best R² | Best RMSE | Mean yield (t/ha) | Std. dev. |
|---|---|---:|---:|---:|---:|
| Maize | Ridge (raw) | 0.9677 | 0.0864 | 2.0744 | 0.5378 |
| Beans | Random Forest (PCA) | -0.7920 | 0.0579 | 0.6563 | 0.0483 |
| Groundnuts | Ridge (raw) | 0.8541 | 0.0393 | 0.4814 | 0.1151 |

## Interpretation

- Maize is the stronger first benchmark in this setup: its best model reached `R² = 0.9677` with `RMSE = 0.0864`.
- Beans is much weaker under the same feature set: its best model reached `R² = -0.7920` with `RMSE = 0.0579`.
- Groundnuts is a stronger secondary benchmark than beans: its best model reached `R² = 0.8541` with `RMSE = 0.0393`.
- The likely reason is target structure, not a pipeline failure. The beans target has only three unique sub-region-assigned values across five districts, so there is limited learnable variation at district level.
- Groundnuts also has only three sub-region-assigned target values, but with a wider spread than beans. That makes it a better test than beans, though still weaker than a true district-level panel.
- The maize target separates the districts more strongly, which makes rainfall, temperature, and terrain features appear more predictive in this first benchmark.
- Both results remain preliminary because the benchmark uses only five rows and sub-region-assigned yields rather than direct district microdata.

## District Targets

| district | sub_region | maize_yield_tons_ha | annual_rainfall | annual_gdd | elevation_m | soil_moisture_index | beans_yield_tons_ha | groundnuts_yield_tons_ha | yield_gap_maize_minus_beans | yield_gap_maize_minus_groundnuts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Iganga | Busoga | 1.60003238244435 | 1714.1868747605577 | 5206.989260070408 | 1100 | 0.2537254927059014 | 0.6904468967705395 | 0.3649043656974847 | 0.9095854856738106 | 1.2351280167468655 |
| Jinja | Busoga | 1.60003238244435 | 1472.0532818860918 | 4850.775276137949 | 1200 | 0.2677227469782035 | 0.6904468967705395 | 0.3649043656974847 | 0.9095854856738106 | 1.2351280167468655 |
| Kapchorwa | Elgon | 2.6510166735556018 | 1355.2572250473431 | 3519.583648912396 | 1800 | 0.2378369122743606 | 0.6636769574159984 | 0.5950828682883496 | 1.9873397161396034 | 2.0559338052672524 |
| Mbale | Elgon | 2.6510166735556018 | 1536.3423973508054 | 4024.479833295983 | 1300 | 0.2037665310005346 | 0.6636769574159984 | 0.5950828682883496 | 1.9873397161396034 | 2.0559338052672524 |
| Tororo | Bukedi | 1.8697114647833253 | 1431.847238001199 | 5092.253236278846 | 1200 | 0.2507131857176621 | 0.5731757916040728 | 0.4869183122414556 | 1.2965356731792523 | 1.3827931525418697 |

## Recommendation

- Keep maize as the primary first real-data benchmark.
- Use groundnuts as the stronger secondary benchmark because it preserves some learnable environmental variation under the same five-district setup.
- Treat beans as the weakest contrast case showing that some crops need either more districts, more years, or crop-specific management variables before environmental PCA features can explain yield well.
