import requests
import numpy as np
import pandas as pd

# ISRIC SoilGrids API endpoint
# We will query point values for each district centroid

DISTRICTS = {
    'Mbale': (1.075, 34.175),
    'Kapchorwa': (1.400, 34.450),
    'Iganga': (0.617, 33.483),
    'Jinja': (0.425, 33.204),
    'Tororo': (0.693, 34.181),
    'Soroti': (1.715, 33.611),
    'Lira': (2.250, 32.917),
    'Gulu': (2.774, 32.299),
    'Mbarara': (-0.607, 30.658),
    'Arua': (3.020, 30.911),
    'Masaka': (-0.333, 31.733),
    'Fort_Portal': (0.671, 30.275),
    'Hoima': (1.433, 31.350),
    'Kabale': (-1.250, 29.983),
    'Kasese': (0.183, 30.083)
}

# SoilGrids REST API v2.0
# Properties we need for agriculture:
# - phh2o: pH in water
# - soc: soil organic carbon (g/kg)
# - clay: clay content (%)
# - sand: sand content (%)
# - silt: silt content (%)
# - bdod: bulk density (cg/cm³)
# - cec: cation exchange capacity (cmol(c)/kg)

properties = ["phh2o", "soc", "clay", "sand", "silt", "bdod", "cec"]
depths = ["0-5cm", "5-15cm", "15-30cm"]  # Topsoil (root zone)

records = []

for district, (lat, lon) in DISTRICTS.items():
    print(f"Fetching soil data for {district}...")
    
    # Build API request
    # SoilGrids v2.0 uses a properties query
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    params = {
        "lon": lon,
        "lat": lat,
        "depth": "0-30cm",  # Average topsoil
        "value": "mean"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            record = {'district': district, 'lat': lat, 'lon': lon}
            
            # Extract properties
            for prop in properties:
                if prop in data.get('properties', {}):
                    layers = data['properties'][prop].get('layers', [])
                    for layer in layers:
                        if layer.get('depth') == '0-30cm':
                            record[prop] = layer.get('values', {}).get('mean', np.nan)
            
            records.append(record)
        else:
            print(f"  [WARN] Status {response.status_code} for {district}")
    except Exception as e:
        print(f"  [ERROR] {district}: {e}")

# Create DataFrame
soil_df = pd.DataFrame(records)
print(f"\nFetched soil data for {len(soil_df)} districts")
print(soil_df.to_string())

# Save
soil_df.to_csv('uganda_soil_features.csv', index=False)
print("\nSaved: uganda_soil_features.csv")
