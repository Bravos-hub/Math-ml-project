# Data Governance and Third-Party Source Register

The MIT license applies to repository code and original documentation only. It does not relicense third-party data. Before publication, verify the exact product version, access date, applicable terms, and whether each committed extract may be redistributed. Blank verification fields are approval blockers, not implied permissions.

| Provider | Dataset/product | Version | Original URL/DOI | Access date | License/terms | Redistribution permitted? | Files committed? | Transformations | Citation | Personal-data status |
|---|---|---|---|---|---|---|---|---|---|---|
| Uganda Bureau of Statistics (UBOS) | AAS 2018/2020 published aggregates and DDI codebooks | 2018/2020 | Record exact UBOS links | **VERIFY** | **VERIFY** | **VERIFY** | Inventory before release | Parse/harmonize published crop, season, subregion targets | UBOS AAS 2018/2020 | Aggregate; no direct identifiers in analytical table |
| UCSB Climate Hazards Center | CHIRPS rainfall | v2.0 | https://www.chc.ucsb.edu/data/chirps | **VERIFY** | **VERIFY** | **VERIFY** | Inventory caches/derivatives | Spatial extraction and seasonal/daily summaries | CHIRPS v2.0 | Non-personal |
| NASA POWER | POWER meteorology | Record product/version | https://power.larc.nasa.gov/ | **VERIFY** | **VERIFY** | **VERIFY** | Inventory API caches | Daily extraction and thermal summaries | NASA POWER | Non-personal |
| Copernicus C3S | Soil moisture | Record product/version | https://cds.climate.copernicus.eu/ | **VERIFY** | **VERIFY** | **VERIFY** | Inventory inputs/derivatives | Spatial/temporal features | Exact CDS citation required | Non-personal |
| ISRIC | SoilGrids | v2.0 | https://soilgrids.org/ | **VERIFY** | **VERIFY** | **VERIFY** | Inventory caches/aggregates | District extraction and subregion aggregation | SoilGrids v2.0 | Non-personal |
| geoBoundaries | Uganda ADM2 / OCHA geometry | Record release | https://www.geoboundaries.org/ | **VERIFY** | **VERIFY** | **VERIFY** | Inventory geometry/cache | Names, centroids, spatial aggregation | Exact release citation required | Non-personal |

## Release checklist

- Resolve every `VERIFY` field and inventory every third-party raw, cached, derived, and committed file.
- Remove any file whose redistribution is not affirmatively allowed.
- Preserve provider attribution and required notices in the evidence bundle.
- Store restricted data outside the public repository with access controls.
- Re-run dataset/config hashes after any source or transformation change.

No evidence bundle is publication-ready while a relevant `VERIFY` field remains unresolved.
