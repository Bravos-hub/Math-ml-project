# Temperature Data Access Guide for Uganda
## CHIRPS Rainfall + Temperature → PCA → Yield Prediction

---

## Why Temperature Matters

Rainfall alone explains ~28% of maize yield variance in Uganda. Temperature captures:

- **Growing Degree Days (GDD):** Thermal time for crop development
  - Maize needs ~1200–1800 GDD (base 10°C) to mature
  - High altitude districts (Kabale, Kapchorwa) have fewer GDD → shorter growing season
- **Heat stress:** Tmax > 35°C during flowering causes pollen sterility
- **Cold stress:** Tmin < 5°C damages seedlings
- **Elevation effects:** 6.5°C cooling per 1000m elevation

**Expected impact:** Adding temperature should improve R² from ~0.28 to ~0.50+ with real data.

---

## Three Data Sources

### Option 1: ERA5-Land Monthly (RECOMMENDED)

**Source:** Copernicus Climate Data Store (CDS)  
**URL:** https://cds.climate.copernicus.eu/  
**Resolution:** 0.1° (finer than standard ERA5's 0.25°)  
**Variables:** 2m temperature, dewpoint temperature  
**Period:** 1950–present (3-month delay)  
**Format:** NetCDF

**Setup:**
```bash
# 1. Create free CDS account
#    → https://cds.climate.copernicus.eu/
#    → Click "Login/Register"

# 2. Get API key from your profile page
#    → Copy the UID and API key

# 3. Install CDS API client
pip install cdsapi

# 4. Create ~/.cdsapirc file
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
EOF

# 5. Run the downloader
python era5_land_downloader.py
```

**What you get:**
- One NetCDF file (~2–3 GB for Uganda, 2015–2023)
- Monthly mean 2m temperature
- We compute GDD, heat/cold stress from monthly means

**Limitation:** Monthly means are coarse for GDD. For precise GDD, you need daily Tmax/Tmin.

---

### Option 2: CHIRTS-ERA5 Daily (Most Precise for GDD)

**Source:** Climate Hazards Center (CHC), UC Santa Barbara  
**URL:** https://data.chc.ucsb.edu/products/CHIRTS-ERA5/  
**Resolution:** 0.05° (same as CHIRPS)  
**Variables:** Tmax, Tmin  
**Period:** 1980–present  
**Format:** GeoTIFF (daily files)

**Setup:**
```bash
pip install rasterio xarray netCDF4 pandas numpy
python chirts_era5_downloader.py
```

**What you get:**
- Daily Tmax/Tmin for every day 2015–2023
- Precise GDD computation: GDD = Σ(max(0, (Tmax+Tmin)/2 − 10))
- Exact heat/cold stress day counts

**Limitation:** ~6,500 individual GeoTIFF files (~15–20 GB total). Very slow download.

---

### Option 3: Google Earth Engine (Fastest if you have account)

**Source:** Google Earth Engine  
**Dataset:** ERA5-Land daily or CHIRTS-ERA5  
**Advantage:** Cloud computation — no downloads needed

**Setup:**
```bash
pip install earthengine-api
earthengine authenticate
python era5_land_gee.py  # See script in project files
```

**What you get:**
- Direct extraction of district time series
- No local storage needed
- Can fuse with MODIS NDVI in same query

---

## Growing Degree Days (GDD) Formula

For maize in Uganda:

```
Daily GDD = max(0, (Tmax + Tmin)/2 − Tbase)

Where:
  Tbase = 10°C (maize germination threshold)
  Tmax = daily maximum temperature
  Tmin = daily minimum temperature

Seasonal GDD = Σ(Daily GDD) over growing season

Maize requires:
  • Early maturing varieties: ~1200 GDD
  • Medium maturing varieties: ~1500 GDD
  • Late maturing varieties: ~1800 GDD
```

**Example for Mbale (1300m):**
- MAM season (Mar–May): ~90 days
- Mean Tmax ≈ 26°C, Tmin ≈ 15°C
- Daily GDD ≈ (26+15)/2 − 10 = 10.5°C
- Seasonal GDD ≈ 10.5 × 90 = 945 GDD
- **Conclusion:** Mbale gets enough GDD for early-maturing maize

**Example for Kabale (2000m):**
- MAM season: ~90 days
- Mean Tmax ≈ 22°C, Tmin ≈ 11°C
- Daily GDD ≈ (22+11)/2 − 10 = 6.5°C
- Seasonal GDD ≈ 6.5 × 90 = 585 GDD
- **Conclusion:** Kabale is marginal for maize — explains why potatoes dominate

---

## What You Have Now (Synthetic Temperature)

Since real downloads take time, I generated **realistic synthetic temperature** based on:
- Uganda's known climate patterns (bimodal temperature)
- Elevation lapse rate (6.5°C per 1000m)
- District-specific elevation data
- Year-to-year climate variability

**District temperature patterns:**

| District | Elevation | MAM Tmean | MAM GDD | Assessment |
|----------|-----------|-----------|---------|------------|
| Kasese | 1000m | 24.7°C | 1324 | Excellent |
| Arua | 1200m | 23.8°C | 1242 | Good |
| Iganga | 1100m | 23.5°C | 1215 | Good |
| Mbale | 1300m | 22.7°C | 1143 | Adequate |
| Kapchorwa | 1800m | 20.5°C | 945 | Marginal |
| Kabale | 2000m | 18.1°C | 729 | Marginal |

This matches real agronomy: Kabale grows potatoes, not maize, because of insufficient GDD.

---

## Timeline

| Week | Action |
|------|--------|
| Now | Use synthetic temperature data to develop pipeline |
| Week 1 | Register CDS account + request ERA5-Land data |
| Week 2 | Download ERA5-Land monthly NetCDF |
| Week 3 | Re-run pipeline with real temperature |
| Week 4 | Compare: synthetic vs real temperature predictions |

---

## Quick Start

**Run the full pipeline now (with synthetic temperature):**
```bash
python uganda_full_pipeline.py
```

**When real temperature data arrives, just swap the CSV:**
```python
# In uganda_full_pipeline.py, change:
temp = pd.read_csv("uganda_temperature_features.csv")
# To:
# temp = pd.read_csv("era5_land_temperature_features.csv")  # Your real data
```

---

## Citation

When you publish, cite:

> Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

> Muñoz-Sabater, J., et al. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13(9), 4349–4383. https://doi.org/10.5194/essd-13-4349-2021
