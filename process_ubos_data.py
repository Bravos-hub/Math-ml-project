import pandas as pd
import numpy as np

# Read UBOS microdata (Stata format)
df = pd.read_stata('AAS_2019.dta')

# Filter for maize (crop code varies — check documentation)
# Common codes: 1 = maize, 2 = beans, etc.
maize = df[df['crop_code'] == 1].copy()

# Compute yield (tons per hectare)
# Check variable names in your dataset — they may differ
maize['yield_tons_ha'] = maize['production_kg'] / (maize['area_harvested_ha'] * 1000)

# Remove outliers (unrealistic yields)
maize = maize[(maize['yield_tons_ha'] > 0.1) & (maize['yield_tons_ha'] < 10)]

# Aggregate to district-year
yield_by_district = maize.groupby(['district', 'year']).agg(
    yield_tons_ha=('yield_tons_ha', 'mean'),
    n_households=('household_id', 'nunique'),
    total_area_ha=('area_harvested_ha', 'sum'),
    total_production_kg=('production_kg', 'sum')
).reset_index()

# Also compute weighted yield (by area)
yield_by_district['yield_weighted'] = (
    yield_by_district['total_production_kg'] / (yield_by_district['total_area_ha'] * 1000)
)

# Save
yield_by_district.to_csv('ubos_maize_yield_district.csv', index=False)
print(f"Saved {len(yield_by_district)} district-year records")
print(yield_by_district.head())
