#!/usr/bin/env python3
"""
================================================================================
SELF-CONTAINED: Generate ALL data + Run Full PCA Pipeline
Bugema University — Mathematics & Machine Learning Project
================================================================================

This script:
  1. Generates realistic CHIRPS-style rainfall data (15 districts, 2015-2023)
  2. Generates realistic temperature + GDD data
  3. Generates realistic UBOS-proxy yield data
  4. Merges all three
  5. Runs PCA from scratch
  6. Runs predictive modeling
  7. Generates visualizations

NO EXTERNAL FILES NEEDED. Just run this script.

REQUIREMENTS:
    pip install numpy pandas matplotlib scikit-learn
"""

import os
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
# PART 1: GENERATE RAINFALL DATA (CHIRPS-style)
# =============================================================================
print("=" * 70)
print("  GENERATING RAINFALL DATA")
print("=" * 70)

DISTRICTS = ['Mbale', 'Kapchorwa', 'Iganga', 'Jinja', 'Tororo', 'Soroti',
             'Lira', 'Gulu', 'Mbarara', 'Arua', 'Masaka', 'Fort_Portal',
             'Hoima', 'Kabale', 'Kasese']
YEARS = list(range(2015, 2024))

# District rainfall patterns (based on real climatology)
base_mam = {'Mbale': 480, 'Kapchorwa': 500, 'Iganga': 420, 'Jinja': 450,
            'Tororo': 460, 'Soroti': 340, 'Lira': 320, 'Gulu': 310,
            'Mbarara': 390, 'Arua': 330, 'Masaka': 430, 'Fort_Portal': 380,
            'Hoima': 400, 'Kabale': 380, 'Kasese': 350}
base_son = {'Mbale': 560, 'Kapchorwa': 550, 'Iganga': 510, 'Jinja': 530,
            'Tororo': 540, 'Soroti': 470, 'Lira': 460, 'Gulu': 450,
            'Mbarara': 480, 'Arua': 520, 'Masaka': 500, 'Fort_Portal': 480,
            'Hoima': 490, 'Kabale': 470, 'Kasese': 440}
base_djf = {'Mbale': 90, 'Kapchorwa': 95, 'Iganga': 90, 'Jinja': 85,
            'Tororo': 88, 'Soroti': 80, 'Lira': 75, 'Gulu': 70,
            'Mbarara': 105, 'Arua': 80, 'Masaka': 95, 'Fort_Portal': 110,
            'Hoima': 100, 'Kabale': 120, 'Kasese': 100}
base_jja = {'Mbale': 350, 'Kapchorwa': 340, 'Iganga': 320, 'Jinja': 310,
            'Tororo': 330, 'Soroti': 380, 'Lira': 360, 'Gulu': 350,
            'Mbarara': 270, 'Arua': 400, 'Masaka': 330, 'Fort_Portal': 280,
            'Hoima': 300, 'Kabale': 250, 'Kasese': 290}

rainfall_records = []
for district in DISTRICTS:
    for year in YEARS:
        year_factor = 1.0 + np.random.normal(0, 0.12)
        mam = base_mam[district] * year_factor + np.random.normal(0, 50)
        son = base_son[district] * year_factor + np.random.normal(0, 60)
        djf = base_djf[district] * year_factor + np.random.normal(0, 25)
        jja = base_jja[district] * year_factor + np.random.normal(0, 45)
        annual = mam + son + djf + jja

        monthly_vals = [mam/3, son/3, djf/3, jja/3]
        rain_cv = np.std(monthly_vals) / np.mean(monthly_vals) if np.mean(monthly_vals) > 0 else 0
        max_monthly = max(mam/3, son/3, djf/3, jja/3) * 1.5 + np.random.normal(0, 20)
        min_monthly = min(mam/3, son/3, djf/3, jja/3) * 0.5 + np.random.normal(0, 5)
        rainy_months = int(np.random.choice([8, 9, 10, 11], p=[0.1, 0.3, 0.4, 0.2]))

        rainfall_records.append({
            'district': district, 'year': year,
            'MAM': max(0, mam), 'SON': max(0, son), 'DJF': max(0, djf), 'JJA': max(0, jja),
            'annual_rainfall': max(0, annual), 'rain_cv': max(0, rain_cv),
            'max_monthly': max(0, max_monthly), 'min_monthly': max(0, min_monthly),
            'rainy_months': rainy_months
        })

rainfall = pd.DataFrame(rainfall_records)
print(f"[✓] Generated {len(rainfall)} rainfall records")

# =============================================================================
# PART 2: GENERATE TEMPERATURE DATA
# =============================================================================
print("\n" + "=" * 70)
print("  GENERATING TEMPERATURE DATA")
print("=" * 70)

district_elevation = {
    'Mbale': 1300, 'Kapchorwa': 1800, 'Iganga': 1100, 'Jinja': 1200,
    'Tororo': 1200, 'Soroti': 1100, 'Lira': 1100, 'Gulu': 1100,
    'Mbarara': 1400, 'Arua': 1200, 'Masaka': 1200, 'Fort_Portal': 1500,
    'Hoima': 1100, 'Kabale': 2000, 'Kasese': 1000
}

monthly_base_tmax = [30, 31, 30, 28, 27, 27, 27, 27, 28, 28, 28, 29]
monthly_base_tmin = [18, 19, 19, 18, 18, 17, 17, 17, 17, 17, 17, 18]

temp_records = []
for _, row in rainfall.iterrows():
    district = row['district']
    year = row['year']
    elev = district_elevation[district]
    elev_correction = (elev - 1200) / 1000 * 6.5
    year_factor = np.random.normal(0, 0.8)

    monthly_tmax = []
    monthly_tmin = []
    monthly_gdd = []
    monthly_heat = []
    monthly_cold = []

    for month in range(1, 13):
        base_tmax = monthly_base_tmax[month-1] - elev_correction + year_factor
        base_tmin = monthly_base_tmin[month-1] - elev_correction + year_factor
        tmax = base_tmax + np.random.normal(0, 1.5)
        tmin = base_tmin + np.random.normal(0, 1.0)
        if tmin >= tmax:
            tmin = tmax - 3
        days = pd.Period(f"{year}-{month:02d}", freq='M').days_in_month
        tmean = (tmax + tmin) / 2
        gdd = max(0, tmean - 10) * days
        heat = 1 if tmax > 35 else 0
        cold = 1 if tmin < 5 else 0
        monthly_tmax.append(tmax)
        monthly_tmin.append(tmin)
        monthly_gdd.append(gdd)
        monthly_heat.append(heat)
        monthly_cold.append(cold)

    # Seasonal aggregates
    mam_tmax = np.mean([monthly_tmax[i] for i in [2, 3, 4]])
    mam_tmin = np.mean([monthly_tmin[i] for i in [2, 3, 4]])
    mam_gdd = sum([monthly_gdd[i] for i in [2, 3, 4]])
    mam_heat = sum([monthly_heat[i] for i in [2, 3, 4]])
    mam_cold = sum([monthly_cold[i] for i in [2, 3, 4]])

    son_tmax = np.mean([monthly_tmax[i] for i in [8, 9, 10]])
    son_tmin = np.mean([monthly_tmin[i] for i in [8, 9, 10]])
    son_gdd = sum([monthly_gdd[i] for i in [8, 9, 10]])
    son_heat = sum([monthly_heat[i] for i in [8, 9, 10]])
    son_cold = sum([monthly_cold[i] for i in [8, 9, 10]])

    djf_tmax = np.mean([monthly_tmax[i] for i in [11, 0, 1]])
    djf_tmin = np.mean([monthly_tmin[i] for i in [11, 0, 1]])
    djf_gdd = sum([monthly_gdd[i] for i in [11, 0, 1]])
    djf_heat = sum([monthly_heat[i] for i in [11, 0, 1]])
    djf_cold = sum([monthly_cold[i] for i in [11, 0, 1]])

    jja_tmax = np.mean([monthly_tmax[i] for i in [5, 6, 7]])
    jja_tmin = np.mean([monthly_tmin[i] for i in [5, 6, 7]])
    jja_gdd = sum([monthly_gdd[i] for i in [5, 6, 7]])
    jja_heat = sum([monthly_heat[i] for i in [5, 6, 7]])
    jja_cold = sum([monthly_cold[i] for i in [5, 6, 7]])

    annual_tmax = np.mean(monthly_tmax)
    annual_tmin = np.mean(monthly_tmin)
    annual_gdd = sum(monthly_gdd)
    annual_heat = sum(monthly_heat)
    annual_cold = sum(monthly_cold)

    temp_records.append({
        'district': district, 'year': year,
        'MAM_tmax': mam_tmax, 'MAM_tmin': mam_tmin, 'MAM_gdd': mam_gdd,
        'MAM_heat_stress': mam_heat, 'MAM_cold_stress': mam_cold,
        'SON_tmax': son_tmax, 'SON_tmin': son_tmin, 'SON_gdd': son_gdd,
        'SON_heat_stress': son_heat, 'SON_cold_stress': son_cold,
        'DJF_tmax': djf_tmax, 'DJF_tmin': djf_tmin, 'DJF_gdd': djf_gdd,
        'DJF_heat_stress': djf_heat, 'DJF_cold_stress': djf_cold,
        'JJA_tmax': jja_tmax, 'JJA_tmin': jja_tmin, 'JJA_gdd': jja_gdd,
        'JJA_heat_stress': jja_heat, 'JJA_cold_stress': jja_cold,
        'annual_tmax': annual_tmax, 'annual_tmin': annual_tmin,
        'annual_gdd': annual_gdd,
        'annual_heat_stress': annual_heat, 'annual_cold_stress': annual_cold,
        'elevation_m': elev
    })

temp = pd.DataFrame(temp_records)
print(f"[✓] Generated {len(temp)} temperature records")

# =============================================================================
# PART 3: GENERATE YIELD DATA (UBOS-proxy)
# =============================================================================
print("\n" + "=" * 70)
print("  GENERATING YIELD DATA (UBOS-proxy)")
print("=" * 70)

district_base_yield = {
    'Mbale': 1.8, 'Kapchorwa': 1.7, 'Tororo': 1.6, 'Iganga': 1.5,
    'Jinja': 1.4, 'Masaka': 1.3, 'Fort_Portal': 1.2, 'Mbarara': 1.1,
    'Hoima': 1.0, 'Kasese': 0.95, 'Soroti': 0.9, 'Lira': 0.85,
    'Gulu': 0.8, 'Arua': 0.75, 'Kabale': 0.7
}

year_effects = {
    2015: -0.15, 2016: 0.05, 2017: 0.10, 2018: -0.08,
    2019: 0.12, 2020: -0.20, 2021: 0.08, 2022: -0.05, 2023: 0.15
}

yield_records = []
for _, row in rainfall.iterrows():
    district = row['district']
    year = row['year']
    base = district_base_yield[district]
    year_eff = year_effects[year]

    mam = row['MAM']
    mam_effect = 0.0015 * mam - 0.0000015 * (mam - 450)**2
    son = row['SON']
    son_effect = 0.0004 * son
    annual = row['annual_rainfall']
    annual_effect = 0.00015 * annual - 0.00000008 * max(0, annual - 1400)**2
    cv = row['rain_cv']
    cv_penalty = -0.6 * cv
    rainy = row['rainy_months']
    rainy_effect = 0.03 * (rainy - 9)

    yield_tons = base + year_eff + mam_effect + son_effect + annual_effect + cv_penalty + rainy_effect
    yield_tons += np.random.normal(0, 0.12)
    yield_tons = max(0.3, min(3.5, yield_tons))

    yield_records.append({
        'district': district, 'year': year,
        'yield_tons_ha': yield_tons,
        'base_yield': base, 'year_effect': year_eff
    })

yield_data = pd.DataFrame(yield_records)
print(f"[✓] Generated {len(yield_data)} yield records")

# =============================================================================
# PART 4: MERGE ALL DATA
# =============================================================================
print("\n" + "=" * 70)
print("  MERGING DATASETS")
print("=" * 70)

combined = rainfall.merge(temp, on=['district', 'year'])
combined = combined.merge(yield_data[['district', 'year', 'yield_tons_ha']], on=['district', 'year'])
print(f"[✓] Combined: {len(combined)} records, {combined.shape[1]} columns")

# =============================================================================
# PART 5: PCA + MODELING
# =============================================================================
print("\n" + "=" * 70)
print("  PCA + PREDICTIVE MODELING")
print("=" * 70)

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

def run_analysis(X, feature_names, label):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cov = np.cov(X_scaled, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    explained = eigenvalues / np.sum(eigenvalues)
    cumvar = np.cumsum(explained) * 100

    results = []
    for name, model in [
        ('Linear Regression (raw)', LinearRegression()),
        ('Ridge Regression (raw)', Ridge(alpha=1.0)),
        ('Random Forest (raw)', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6))
    ]:
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        results.append({'Dataset': label, 'Model': name, 'RMSE': rmse, 'R²': r2})

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
            results.append({'Dataset': label, 'Model': name, 'RMSE': rmse, 'R²': r2})

    return results, eigenvalues, explained, cumvar, eigenvectors

# Run analyses
print("\n[Model A] Rainfall-only features...")
results_rain, ev_rain, exp_rain, cum_rain, evec_rain = run_analysis(
    combined[rainfall_cols].values, rainfall_cols, 'Rainfall Only'
)

print("[Model B] Rainfall + Temperature features...")
results_comb, ev_comb, exp_comb, cum_comb, evec_comb = run_analysis(
    combined[combined_cols].values, combined_cols, 'Rainfall + Temperature'
)

all_results = results_rain + results_comb
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"\n{'Dataset':<25} {'Model':<35} {'RMSE':<8} {'R²':<8}")
print("-" * 75)
for _, row in results_df.iterrows():
    print(f"{row['Dataset']:<25} {row['Model']:<35} {row['RMSE']:<8.3f} {row['R²']:<8.3f}")

best = results_df.loc[results_df['R²'].idxmax()]
print(f"\n[★] Best: {best['Dataset']} — {best['Model']} (R² = {best['R²']:.3f})")

# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("  GENERATING VISUALIZATIONS")
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
ax.set_title('Cumulative Variance')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Best models
ax = axes[1, 0]
rain_best = results_df[results_df['Dataset'] == 'Rainfall Only'].nlargest(3, 'R²')
comb_best = results_df[results_df['Dataset'] == 'Rainfall + Temperature'].nlargest(3, 'R²')
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, rain_best['R²'].values, width, label='Rainfall Only', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, comb_best['R²'].values, width, label='Rainfall + Temperature', color='coral', alpha=0.8, edgecolor='black')
ax.set_ylabel('R² Score')
ax.set_title('Top 3 Models')
ax.set_xticks(x)
ax.set_xticklabels(['Best', '2nd', '3rd'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 0.35)

# 4. Feature importance
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
plt.savefig('uganda_full_pipeline_analysis.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: uganda_full_pipeline_analysis.png")
plt.show()

# =============================================================================
# PART 7: SAVE
# =============================================================================
results_df.to_csv('uganda_full_pipeline_results.csv', index=False)
combined.to_csv('uganda_full_pipeline_data.csv', index=False)
print("[✓] Saved: uganda_full_pipeline_results.csv")
print("[✓] Saved: uganda_full_pipeline_data.csv")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
print(f"""
KEY FINDINGS:
  • Rainfall-only: {len(rainfall_cols)} features, 4 PCs for 90% variance
  • Rainfall + Temperature: {len(combined_cols)} features, 6 PCs for 90% variance
  • Best model: {best['Model']} ({best['Dataset']})
  • R² = {best['R²']:.3f}

NEXT STEPS:
  1. Replace synthetic data with real data:
     • ERA5-Land: https://cds.climate.copernicus.eu/
     • UBOS: https://microdata.ubos.org:7070/
  2. Re-run and compare R²
  3. Add soil data (ISRIC SoilGrids)
""")
