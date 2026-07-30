#!/usr/bin/env python3
"""
================================================================================
COMPLETE PCA ANALYSIS: Real CHIRPS Rainfall Data for Uganda
Bugema University — Mathematics & Machine Learning Project
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD REAL CHIRPS DATA
# ============================================================
features = pd.read_csv('uganda_rainfall_features.csv')
print(f"Loaded {len(features)} records from CHIRPS satellite data")

feature_cols = ['MAM', 'SON', 'DJF', 'JJA', 'annual_rainfall', 
                'rain_cv', 'max_monthly', 'min_monthly', 'rainy_months']
X = features[feature_cols].values

# ============================================================
# 2. STANDARDIZE
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. COVARIANCE MATRIX & EIGENDECOMPOSITION
# ============================================================
cov = np.cov(X_scaled, rowvar=False)
eigenvalues, eigenvectors = eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
explained = eigenvalues / np.sum(eigenvalues)

# ============================================================
# 4. INTERPRET COMPONENTS
# ============================================================
loadings = pd.DataFrame(eigenvectors, index=feature_cols,
                        columns=[f'PC{i+1}' for i in range(len(feature_cols))])

print("\n" + "="*60)
print("PCA RESULTS ON REAL CHIRPS DATA")
print("="*60)
for i in range(4):
    print(f"\nPC{i+1}: λ={eigenvalues[i]:.4f} ({explained[i]*100:.1f}%)")
    top = loadings.iloc[:, i].abs().sort_values(ascending=False).head(3)
    print(f"  Top features: {', '.join([f'{k}({v:.3f})' for k,v in top.items()])}")

# ============================================================
# 5. PROJECT TO 2D FOR VISUALIZATION
# ============================================================
Z = X_scaled @ eigenvectors[:, :2]

# ============================================================
# 6. VISUALIZE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1 = axes[0]
ax1.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color='#1f77b4', linewidth=2, markersize=8)
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue (λ)')
ax1.set_title('Scree Plot: Real CHIRPS Data')
ax1.grid(True, alpha=0.3)

# Biplot
ax2 = axes[1]
# Color by mean annual rainfall
rainfall_colors = features['annual_rainfall'].values
scatter = ax2.scatter(Z[:, 0], Z[:, 1], c=rainfall_colors, cmap='Blues',
                     s=60, alpha=0.7, edgecolors='k', linewidth=0.3)
plt.colorbar(scatter, ax=ax2, label='Annual Rainfall (mm)')

# Feature arrows
scale = 2.5
for i, feat in enumerate(feature_cols):
    ax2.arrow(0, 0, eigenvectors[i, 0]*scale, eigenvectors[i, 1]*scale,
             head_width=0.1, head_length=0.08, fc='darkred', ec='darkred', alpha=0.7)
    ax2.text(eigenvectors[i, 0]*scale*1.1, eigenvectors[i, 1]*scale*1.1,
            feat, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))

ax2.set_xlabel(f'PC1: Total Wetness ({explained[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2: Rain Variability ({explained[1]*100:.1f}%)')
ax2.set_title('Uganda Districts in PCA Space (CHIRPS)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', alpha=0.3)
ax2.axvline(x=0, color='gray', alpha=0.3)

plt.tight_layout()
plt.savefig('chirps_pca_final.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[✓] Saved: chirps_pca_final.png")
