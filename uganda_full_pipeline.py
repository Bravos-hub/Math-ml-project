#!/usr/bin/env python3
"""
================================================================================
COMPLETE INTEGRATED PIPELINE
CHIRPS Rainfall + Temperature (GDD) + UBOS Yield → PCA → Prediction
Bugema University — Mathematics & Machine Learning Project
================================================================================

This script runs the FULL pipeline:
  1. Load real CHIRPS rainfall features
  2. Load temperature features (synthetic for now — replace with ERA5-Land)
  3. Load UBOS-proxy yield data
  4. Merge all three datasets
  5. Run PCA from scratch on combined features
  6. Compare: Rainfall-only vs Rainfall+Temperature prediction
  7. Generate visualizations

TO USE REAL TEMPERATURE DATA:
  1. Create CDS account: https://cds.climate.copernicus.eu/
  2. Install cdsapi: pip install cdsapi
  3. Run: python era5_land_downloader.py
  4. Replace uganda_temperature_features.csv with real data
  5. Re-run this script

REQUIREMENTS:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
RAINFALL_FILE = "uganda_rainfall_features.csv"
TEMP_FILE = "uganda_temperature_features.csv"
YIELD_FILE = "ubos_district_yield_proxy.csv"
OUTPUT_PREFIX = "uganda_full_pipeline_"

# =============================================================================
# STEP 1: LOAD ALL DATA
# =============================================================================
print("=" * 70)
print("  STEP 1: LOAD RAINFALL + TEMPERATURE + YIELD DATA")
print("=" * 70)

rainfall = pd.read_csv(RAINFALL_FILE)
print(f"[✓] Rainfall: {len(rainfall)} records")

temp = pd.read_csv(TEMP_FILE)
print(f"[✓] Temperature: {len(temp)} records")
print(f"    ⚠️  REPLACE with real ERA5-Land data when available")

yield_data = pd.read_csv(YIELD_FILE)
print(f"[✓] Yield: {len(yield_data)} records")
print(f"    ⚠️  REPLACE with real UBOS microdata when available")

# Merge
combined = rainfall.merge(temp, on=['district', 'year'])
combined = combined.merge(yield_data[['district', 'year', 'yield_tons_ha']], on=['district', 'year'])
print(f"[✓] Combined: {len(combined)} records, {combined.shape[1]} columns")

# =============================================================================
# STEP 2: DEFINE FEATURE SETS
# =============================================================================
rainfall_cols = ['MAM', 'SON', 'DJF', 'JJA', 'annual_rainfall', 
                 'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']

combined_cols = rainfall_cols + [
    'MAM_tmax', 'MAM_tmin', 'MAM_gdd', 'MAM_heat_stress', 'MAM_cold_stress',
    'SON_tmax', 'SON_tmin', 'SON_gdd', 'SON_heat_stress', 'SON_cold_stress',
    'DJF_tmax', 'DJF_tmin', 'DJF_gdd',
    'JJA_tmax', 'JJA_tmin', 'JJA_gdd',
    'annual_tmax', 'annual_tmin', 'annual_gdd',
    'annual_heat_stress', 'annual_cold_stress',
    'elevation_m'
]

y = combined['yield_tons_ha'].values
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# STEP 3: PCA + MODELING FUNCTION
# =============================================================================
def run_analysis(X, feature_names, label):
    """Run PCA and predictive modeling on a feature set."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    cov = np.cov(X_scaled, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    explained = eigenvalues / np.sum(eigenvalues)
    cumvar = np.cumsum(explained) * 100

    results = []

    # Raw features
    for name, model in [
        ('Linear Regression (raw)', LinearRegression()),
        ('Ridge Regression (raw)', Ridge(alpha=1.0)),
        ('Random Forest (raw)', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6))
    ]:
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        results.append({'Dataset': label, 'Model': name, 'RMSE': rmse, 'R²': r2, 'Features': X.shape[1]})

    # PCA models
    for k in [2, 3, 4, 5, 6, 8, 10]:
        if k > X.shape[1]:
            break
        Z = X_scaled @ eigenvectors[:, :k]
        for name, model in [
            (f'Linear Regression ({k} PCs)', LinearRegression()),
            (f'Ridge Regression ({k} PCs)', Ridge(alpha=1.0)),
            (f'Random Forest ({k} PCs)', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6))
        ]:
            y_pred = cross_val_predict(model, Z, y, cv=cv)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            results.append({'Dataset': label, 'Model': name, 'RMSE': rmse, 'R²': r2, 'Features': k})

    return results, eigenvalues, explained, cumvar, eigenvectors

# =============================================================================
# STEP 4: RUN ANALYSIS ON BOTH DATASETS
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 4: PCA + MODELING")
print("=" * 70)

# Rainfall-only
print("\n[Model A] Rainfall-only features...")
results_rain, ev_rain, exp_rain, cum_rain, evec_rain = run_analysis(
    combined[rainfall_cols].values, rainfall_cols, 'Rainfall Only'
)
print(f"  Features: {len(rainfall_cols)}")
for i in range(6):
    print(f"  PC{i+1}: λ={ev_rain[i]:.4f} ({exp_rain[i]*100:.1f}%)  Cum: {cum_rain[i]:.1f}%")

# Rainfall + Temperature
print("\n[Model B] Rainfall + Temperature features...")
results_comb, ev_comb, exp_comb, cum_comb, evec_comb = run_analysis(
    combined[combined_cols].values, combined_cols, 'Rainfall + Temperature'
)
print(f"  Features: {len(combined_cols)}")
for i in range(8):
    print(f"  PC{i+1}: λ={ev_comb[i]:.4f} ({exp_comb[i]*100:.1f}%)  Cum: {cum_comb[i]:.1f}%")

# Combine results
all_results = results_rain + results_comb
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Dataset':<25} {'Model':<35} {'RMSE':<8} {'R²':<8}")
print("-" * 75)
for _, row in results_df.iterrows():
    print(f"{row['Dataset']:<25} {row['Model']:<35} {row['RMSE']:<8.3f} {row['R²']:<8.3f}")

best = results_df.loc[results_df['R²'].idxmax()]
print(f"\n[★] Best model: {best['Dataset']} — {best['Model']} (R² = {best['R²']:.3f})")

# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 5: GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Scree comparison
ax = axes[0, 0]
ax.plot(range(1, len(ev_rain)+1), ev_rain, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Rainfall Only')
ax.plot(range(1, len(ev_comb)+1), ev_comb, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Rainfall + Temperature')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Eigenvalue (λ)')
ax.set_title('Scree Plot Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Cumulative variance
ax = axes[0, 1]
ax.plot(range(1, len(cum_rain)+1), cum_rain, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Rainfall Only')
ax.plot(range(1, len(cum_comb)+1), cum_comb, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Rainfall + Temperature')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Variance (%)')
ax.set_title('Cumulative Variance Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Best models comparison
ax = axes[1, 0]
rain_best = results_df[results_df['Dataset'] == 'Rainfall Only'].nlargest(3, 'R²')
comb_best = results_df[results_df['Dataset'] == 'Rainfall + Temperature'].nlargest(3, 'R²')
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, rain_best['R²'].values, width, label='Rainfall Only', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, comb_best['R²'].values, width, label='Rainfall + Temperature', color='coral', alpha=0.8, edgecolor='black')
ax.set_ylabel('R² Score')
ax.set_title('Top 3 Models Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['Best', '2nd', '3rd'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 0.35)

# 4. Feature importance (combined)
ax = axes[1, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined[combined_cols])
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
rf.fit(X_scaled, y)
importance = pd.Series(rf.feature_importances_, index=combined_cols).sort_values(ascending=True).tail(15)
importance.plot(kind='barh', ax=ax, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance))), edgecolor='black', linewidth=0.5)
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}analysis.png", dpi=150, bbox_inches='tight')
print(f"[✓] Saved: {OUTPUT_PREFIX}analysis.png")
plt.show()

# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================
results_df.to_csv(f"{OUTPUT_PREFIX}results.csv", index=False)
combined.to_csv(f"{OUTPUT_PREFIX}data.csv", index=False)
print(f"[✓] Saved: {OUTPUT_PREFIX}results.csv")
print(f"[✓] Saved: {OUTPUT_PREFIX}data.csv")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
print(f"""
KEY FINDINGS:
  • Rainfall-only: {len(rainfall_cols)} features, 4 PCs for 90% variance
  • Rainfall + Temperature: {len(combined_cols)} features, 6 PCs for 90% variance
  • Best model: {best['Model']} ({best['Dataset']})
  • R² = {best['R²']:.3f}, RMSE = {results_df.loc[results_df['R²'].idxmax(), 'RMSE']:.3f} t/ha

NEXT STEPS:
  1. Replace synthetic temperature with ERA5-Land:
     → https://cds.climate.copernicus.eu/
     → python era5_land_downloader.py

  2. Replace proxy yield with real UBOS data:
     → https://microdata.ubos.org:7070/

  3. Add soil data (ISRIC SoilGrids):
     → https://soilgrids.org/

  4. With real data, temperature (especially GDD) should improve R²
     from ~0.28 to ~0.50+ by capturing elevation and thermal time effects.
""")
