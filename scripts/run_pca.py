"""Run the PCA v2 analysis (retention, stability, loadings).

Writes ``reports/tables/pca_v2_scree.csv``, ``reports/tables/pca_v2_loadings.csv``
and ``reports/figures/pca_v2_scree.png``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropyield.pca.pca_analysis import main

if __name__ == "__main__":
    main()
