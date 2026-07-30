
# ============================================================
# APPROACH 2: ClimateSERV REST API — Point & Polygon Extraction
# ============================================================
# ClimateSERV provides a REST API for CHIRPS data.
# Best for: extracting time series for specific coordinates or districts.
# No API key required for basic usage.

import requests
import json
import pandas as pd
import time

BASE_URL = "https://climateserv.servirglobal.net/api"

# --- Step 1: Submit a data request ---
def request_chirps_data(geometry, start_date, end_date, datatype=0):
    """
    Submit a CHIRPS data request to ClimateSERV.

    Parameters:
    -----------
    geometry : dict
        GeoJSON geometry (Point or Polygon)
    start_date, end_date : str
        Format: "MM/DD/YYYY"
    datatype : int
        0 = CHIRPS Daily, 5 = CHIRPS Monthly
    """
    url = f"{BASE_URL}/submitDataRequest/"
    params = {
        'datatype': datatype,
        'begintime': start_date,
        'endtime': end_date,
        'intervaltype': 0,  # 0 = daily, 1 = monthly
        'geometry': json.dumps(geometry)
    }
    response = requests.get(url, params=params, timeout=60)
    if response.status_code == 200:
        return response.json()  # Returns a job ID
    else:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

# --- Step 2: Check job status ---
def check_job_status(job_id):
    url = f"{BASE_URL}/getDataRequestProgress/{job_id}/"
    response = requests.get(url, timeout=30)
    return response.json()

# --- Step 3: Retrieve results ---
def get_job_results(job_id):
    url = f"{BASE_URL}/getDataFromRequest/{job_id}/"
    response = requests.get(url, timeout=60)
    return response.json()

# --- Example: Fetch rainfall for Mbale district center ---
mbale_point = {
    "type": "Point",
    "coordinates": [34.175, 1.075]  # [lon, lat]
}

# Submit request for 2022
print("Submitting CHIRPS request for Mbale...")
job_id = request_chirps_data(mbale_point, "01/01/2022", "12/31/2022", datatype=0)
print(f"Job ID: {job_id}")

# Poll until complete
print("Polling for results...")
for _ in range(30):
    status = check_job_status(job_id)
    print(f"Progress: {status}%")
    if status == 100:
        break
    time.sleep(2)

# Get data
results = get_job_results(job_id)
print(f"Retrieved {len(results)} daily records")

# Convert to DataFrame
df = pd.DataFrame(results)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
print(df.head())

# Aggregate to monthly
monthly = df.groupby(df['date'].dt.to_period('M'))['value'].sum().reset_index()
monthly['date'] = monthly['date'].dt.to_timestamp()
print(monthly)
