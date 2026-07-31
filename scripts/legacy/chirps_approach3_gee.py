
# ============================================================
# APPROACH 3: Google Earth Engine Python API
# ============================================================
# Best for: large-scale spatial analysis, computing zonal statistics
# over district boundaries, time-series extraction.
# Requires: free Google Earth Engine account (signup at earthengine.google.com)

import ee
import geemap
import pandas as pd

# Initialize (first time: run `earthengine authenticate` in terminal)
ee.Initialize()

# Load CHIRPS daily collection
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

# Define Uganda districts as points or polygons
# For district-level analysis, you'd use Uganda admin boundaries
# Here we use point locations for simplicity
districts = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([34.175, 1.075]), {'name': 'Mbale'}),
    ee.Feature(ee.Geometry.Point([34.450, 1.400]), {'name': 'Kapchorwa'}),
    ee.Feature(ee.Geometry.Point([33.483, 0.617]), {'name': 'Iganga'}),
    ee.Feature(ee.Geometry.Point([33.204, 0.425]), {'name': 'Jinja'}),
    ee.Feature(ee.Geometry.Point([34.181, 0.693]), {'name': 'Tororo'}),
    ee.Feature(ee.Geometry.Point([33.611, 1.715]), {'name': 'Soroti'}),
    ee.Feature(ee.Geometry.Point([32.917, 2.250]), {'name': 'Lira'}),
    ee.Feature(ee.Geometry.Point([32.299, 2.774]), {'name': 'Gulu'}),
    ee.Feature(ee.Geometry.Point([30.658, -0.607]), {'name': 'Mbarara'}),
    ee.Feature(ee.Geometry.Point([30.911, 3.020]), {'name': 'Arua'}),
])

# Filter CHIRPS to a time range
start_date = '2015-01-01'
end_date = '2023-12-31'
chirps_filtered = chirps.filterDate(start_date, end_date)

# Extract time series for each district
# Method: reduceRegion over each image, mapped over the collection
def extract_rainfall(image):
    date = image.date().format('YYYY-MM-dd')
    rainfall = image.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=5566  # CHIRPS native resolution ~5.5km
    )
    return rainfall.map(lambda f: f.set('date', date))

results = chirps_filtered.map(extract_rainfall).flatten()

# Convert to pandas DataFrame
# Note: For large collections, use getInfo() carefully or export to Drive
features = results.getInfo()['features']
records = []
for f in features:
    props = f['properties']
    records.append({
        'district': props.get('name'),
        'date': props.get('date'),
        'rainfall_mm': props.get('mean')
    })

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()
print(df.head(20))

# Aggregate to seasonal/annual for PCA
# ... (same seasonal aggregation as Approach 1)
