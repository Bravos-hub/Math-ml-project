"""
================================================================================
END-TO-END PIPELINE: CHIRPS Rainfall + PCA for Ugandan Maize Yield Prediction
================================================================================
Author: [Your Name] | Bugema University
Description: Downloads CHIRPS data, extracts seasonal rainfall features,
             merges with yield data, runs PCA, and predicts yield.

REQUIREMENTS:
    pip install xarray netCDF4 pandas numpy matplotlib scikit-learn

DATA SOURCES:
    - CHIRPS v2.0 Monthly: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
    - Uganda yield data: UBOS Annual Agricultural Survey
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# =============================================================================
# CONFIGURATION
# =============================================================================
UGANDA_BOUNDS = {
    'lat_min': -1.5, 'lat_max': 4.2,
    'lon_min': 29.5, 'lon_max': 35.0
}

# District centroids for Eastern & Northern Uganda
DISTRICTS = {
    'Mbale':      (1.075, 34.175),
    'Kapchorwa':  (1.400, 34.450),
    'Iganga':     (0.617, 33.483),
    'Jinja':      (0.425, 33.204),
    'Tororo':     (0.693, 34.181),
    'Soroti':     (1.715, 33.611),
    'Lira':       (2.250, 32.917),
    'Gulu':       (2.774, 32.299),
    'Mbarara':    (-0.607, 30.658),
    'Arua':       (3.020, 30.911),
    'Masaka':     (-0.333, 31.733),
    'Fort_Portal':(0.671, 30.275),
    'Hoima':      (1.433, 31.350),
    'Kabale':     (-1.250, 29.983),
    'Kasese':     (0.183, 30.083)
}

YEARS = list(range(2015, 2024))  # 2015-2023
CHIRPS_FILE = "chirps-v2.0.monthly.nc"
CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"

# =============================================================================
# STEP 1: DOWNLOAD CHIRPS DATA (if not already present)
# =============================================================================
def download_chirps():
    """Download CHIRPS monthly NetCDF if not already present."""
    if os.path.exists(CHIRPS_FILE):
        print(f"[✓] CHIRPS file already exists: {CHIRPS_FILE}")
        return

    print(f"[↓] Downloading CHIRPS monthly data (~2.5 GB)...")
    print(f"    Source: {CHIRPS_URL}")
    print(f"    This will take 10-30 minutes depending on your connection.")

    try:
        urllib.request.urlretrieve(CHIRPS_URL, CHIRPS_FILE)
        print(f"[✓] Download complete: {CHIRPS_FILE}")
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        print("    Alternative: Use Google Earth Engine or ClimateSERV API.")
        raise

# =============================================================================
# STEP 2: EXTRACT RAINFALL FEATURES FOR UGANDAN DISTRICTS
# =============================================================================
def extract_rainfall_features():
    """Extract seasonal and annual rainfall features from CHIRPS for each district."""

    print("\n[📊] Loading and clipping CHIRPS data to Uganda...")
    chirps = xr.open_dataset(CHIRPS_FILE)
    pr = chirps['precip']

    # Clip to Uganda + time range
    uganda_rain = pr.sel(
        longitude=slice(UGANDA_BOUNDS['lon_min'], UGANDA_BOUNDS['lon_max']),
        latitude=slice(UGANDA_BOUNDS['lat_max'], UGANDA_BOUNDS['lat_min']),
        time=slice(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
    )

    print(f"    Clipped shape: {uganda_rain.shape}")
    print(f"    Time range: {str(uganda_rain.time.min().values)[:10]} to {str(uganda_rain.time.max().values)[:10]}")

    # Extract point time series for each district
    records = []
    for district, (lat, lon) in DISTRICTS.items():
        ts = uganda_rain.sel(latitude=lat, longitude=lon, method='nearest')
        for t, val in zip(ts.time.values, ts.values):
            dt = pd.to_datetime(str(t))
            records.append({
                'district': district,
                'date': dt,
                'year': dt.year,
                'month': dt.month,
                'rainfall_mm': float(val)
            })

    df = pd.DataFrame(records)
    print(f"\n[✓] Extracted {len(df)} monthly records across {len(DISTRICTS)} districts")

    # =============================================================================
    # STEP 3: COMPUTE SEASONAL & ANNUAL AGGREGATES
    # =============================================================================
    print("\n[📐] Computing seasonal rainfall features...")

    def get_season(month):
        if month in [3, 4, 5]:     return 'MAM'    # Long rains
        elif month in [9, 10, 11]: return 'SON'    # Short rains
        elif month in [12, 1, 2]:  return 'DJF'    # Dry (N)
        else:                       return 'JJA'    # Dry (S)

    df['season'] = df['month'].apply(get_season)

    # Seasonal totals
    seasonal = df.groupby(['district', 'year', 'season'])['rainfall_mm'].sum().reset_index()
    seasonal_pivot = seasonal.pivot_table(
        index=['district', 'year'],
        columns='season',
        values='rainfall_mm'
    ).reset_index()

    # Annual statistics
    annual = df.groupby(['district', 'year']).agg(
        annual_rainfall=('rainfall_mm', 'sum'),
        rain_cv=('rainfall_mm', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        max_monthly=('rainfall_mm', 'max'),
        min_monthly=('rainfall_mm', 'min'),
        rainy_months=('rainfall_mm', lambda x: (x > 50).sum())  # months with >50mm
    ).reset_index()

    # Merge
    features = seasonal_pivot.merge(annual, on=['district', 'year'])

    # Fill any missing seasonal values with 0
    for season in ['MAM', 'SON', 'DJF', 'JJA']:
        if season not in features.columns:
            features[season] = 0

    print(f"\n[✓] Feature matrix shape: {features.shape}")
    print(f"    Columns: {list(features.columns)}")

    return features

# =============================================================================
# STEP 4: LOAD/CREATE YIELD DATA (replace with real UBOS data when available)
# =============================================================================
def create_synthetic_yield(features_df):
    """
    Create realistic yield data based on rainfall features.
    REPLACE THIS with real yield data from UBOS when you get it.

    The synthetic model encodes real agronomic knowledge:
    - MAM rainfall is most important for maize in Uganda
    - Too much rain (high CV) can cause waterlogging
    - SON rainfall matters for second season crops
    """
    np.random.seed(42)

    def compute_yield(row):
        mam = row.get('MAM', 0)
        son = row.get('SON', 0)
        annual = row['annual_rainfall']
        cv = row['rain_cv']

        # Base yield
        base = 0.8

        # MAM rainfall contribution (optimal around 300-500mm)
        mam_effect = 0.003 * mam - 0.000002 * (mam - 400)**2

        # SON rainfall (secondary)
        son_effect = 0.001 * son

        # Annual total (diminishing returns after 1200mm)
        annual_effect = 0.0005 * annual - 0.0000003 * max(0, annual - 1200)**2

        # Rainfall variability penalty
        cv_penalty = -0.5 * cv

        # District fixed effect (some districts have better soil/management)
        district_effect = {
            'Mbale': 0.3, 'Kapchorwa': 0.2, 'Iganga': 0.1, 'Jinja': 0.15,
            'Tororo': 0.25, 'Soroti': 0.1, 'Lira': 0.05, 'Gulu': 0.0,
            'Mbarara': 0.2, 'Arua': 0.0, 'Masaka': 0.15, 'Fort_Portal': 0.1,
            'Hoima': 0.1, 'Kabale': 0.15, 'Kasese': 0.1
        }.get(row['district'], 0)

        # Year trend (slight improvement over time)
        year_trend = 0.02 * (row['year'] - 2015)

        yield_val = base + mam_effect + son_effect + annual_effect + cv_penalty + district_effect + year_trend
        yield_val += np.random.normal(0, 0.15)  # noise
        return max(0.5, min(5.0, yield_val))

    features_df['yield_tons_ha'] = features_df.apply(compute_yield, axis=1)
    return features_df

# =============================================================================
# STEP 5: PCA FROM SCRATCH
# =============================================================================
def run_pca(X, n_components=3):
    """
    Principal Component Analysis from scratch using eigendecomposition.

    Parameters:
    -----------
    X : ndarray, shape (n_samples, n_features)
        Standardized feature matrix
    n_components : int
        Number of principal components to retain

    Returns:
    --------
    Z : ndarray, shape (n_samples, n_components)
        Projected data
    components : ndarray, shape (n_features, n_components)
        Principal component loadings (eigenvectors)
    explained_ratio : ndarray
        Proportion of variance explained by each component
    """
    # 1. Center (already standardized, so mean ≈ 0)
    # 2. Covariance matrix
    cov = np.cov(X, rowvar=False)

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = eigh(cov)

    # 4. Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Select top-k
    components = eigenvectors[:, :n_components]

    # 6. Project
    Z = X @ components

    # Explained variance
    explained_ratio = eigenvalues / np.sum(eigenvalues)

    return Z, components, explained_ratio

# =============================================================================
# STEP 6: FULL PIPELINE EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("  CHIRPS + PCA PIPELINE FOR UGANDAN MAIZE YIELD PREDICTION")
    print("=" * 70)

    # 1. Download
    download_chirps()

    # 2. Extract features
    features = extract_rainfall_features()

    # 3. Add yield (synthetic for now — replace with real data)
    print("\n[🌾] Adding yield data (synthetic — replace with UBOS data)...")
    features = create_synthetic_yield(features)

    # 4. Prepare feature matrix for PCA
    feature_cols = ['MAM', 'SON', 'DJF', 'JJA', 'annual_rainfall', 
                    'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']

    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in features.columns]
    X = features[feature_cols].values
    y = features['yield_tons_ha'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Run PCA
    print("\n[📐] Running PCA from scratch...")
    Z, components, explained = run_pca(X_scaled, n_components=3)

    print(f"\n    Explained variance ratios:")
    for i, r in enumerate(explained[:3]):
        cum = sum(explained[:i+1])
        print(f"      PC{i+1}: {r:.4f} ({r*100:.1f}%)  |  Cumulative: {cum*100:.1f}%")

    # Loadings
    loadings = pd.DataFrame(components, index=feature_cols, columns=['PC1', 'PC2', 'PC3'])
    print(f"\n    Component loadings:")
    print(loadings.round(3))

    # 6. Predictive modeling
    print("\n[🤖] Predictive modeling: Raw features vs PCA components...")

    # Linear Regression on raw
    lr_raw = LinearRegression()
    r2_lr_raw = cross_val_score(lr_raw, X_scaled, y, cv=5, scoring='r2').mean()
    rmse_lr_raw = np.sqrt(-cross_val_score(lr_raw, X_scaled, y, cv=5, scoring='neg_mean_squared_error').mean())

    # Linear Regression on PCA
    lr_pca = LinearRegression()
    r2_lr_pca = cross_val_score(lr_pca, Z, y, cv=5, scoring='r2').mean()
    rmse_lr_pca = np.sqrt(-cross_val_score(lr_pca, Z, y, cv=5, scoring='neg_mean_squared_error').mean())

    # Random Forest on raw
    rf_raw = RandomForestRegressor(n_estimators=100, random_state=42)
    r2_rf_raw = cross_val_score(rf_raw, X_scaled, y, cv=5, scoring='r2').mean()
    rmse_rf_raw = np.sqrt(-cross_val_score(rf_raw, X_scaled, y, cv=5, scoring='neg_mean_squared_error').mean())

    # Random Forest on PCA
    rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
    r2_rf_pca = cross_val_score(rf_pca, Z, y, cv=5, scoring='r2').mean()
    rmse_rf_pca = np.sqrt(-cross_val_score(rf_pca, Z, y, cv=5, scoring='neg_mean_squared_error').mean())

    print(f"\n    {'Model':<30} {'RMSE':<10} {'R²':<10}")
    print(f"    {'-'*50}")
    print(f"    {'Linear Reg (raw)':<30} {rmse_lr_raw:<10.3f} {r2_lr_raw:<10.3f}")
    print(f"    {'Linear Reg (3 PCs)':<30} {rmse_lr_pca:<10.3f} {r2_lr_pca:<10.3f}")
    print(f"    {'Random Forest (raw)':<30} {rmse_rf_raw:<10.3f} {r2_rf_raw:<10.3f}")
    print(f"    {'Random Forest (3 PCs)':<30} {rmse_rf_pca:<10.3f} {r2_rf_pca:<10.3f}")

    # 7. Visualization
    print("\n[📊] Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Biplot
    ax = axes[0]
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='RdYlGn', s=60, alpha=0.7, edgecolors='k', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='Yield (t/ha)')
    ax.set_xlabel(f'PC1: {explained[0]*100:.1f}%')
    ax.set_ylabel(f'PC2: {explained[1]*100:.1f}%')
    ax.set_title('Ugandan Districts in PCA Space\n(CHIRPS Rainfall Features)')
    ax.grid(True, alpha=0.3)

    # Model comparison
    ax = axes[1]
    models = ['LR\n(raw)', 'LR\n(3 PCs)', 'RF\n(raw)', 'RF\n(3 PCs)']
    r2_vals = [r2_lr_raw, r2_lr_pca, r2_rf_raw, r2_rf_pca]
    colors = ['steelblue', 'steelblue', 'forestgreen', 'forestgreen']
    bars = ax.bar(models, r2_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R² Score (5-fold CV)', fontsize=11)
    ax.set_title('Predictive Performance: Raw vs PCA Features')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('chirps_pca_results.png', dpi=150, bbox_inches='tight')
    print("    Saved: chirps_pca_results.png")
    plt.show()

    # 8. Save outputs
    features.to_csv('uganda_chirps_pca_features.csv', index=False)
    print("\n[✓] Saved: uganda_chirps_pca_features.csv")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNEXT STEPS:")
    print("  1. Replace synthetic yield with real UBOS data")
    print("  2. Add more features: temperature (CHIRTS), soil (ISRIC), NDVI (MODIS)")
    print("  3. Try Kernel PCA or Sparse PCA for non-linear structure")
    print("  4. Publish your findings!")

if __name__ == "__main__":
    main()
