"""Central path constants for the repository layout.

Layout (see README):

    data/raw/            original downloads (PDFs, xlsx, NetCDF, API caches)
    data/external/       third-party datasets not directly ingested
    data/interim/        partially processed tables
    data/processed/observed/   official estimates at the unit of analysis
    data/processed/assigned/   official values assigned from larger areas
    data/processed/proxy/      proxy-based tables
    data/processed/synthetic/  synthetic/demonstration tables
    data/public/         model-ready files intended for sharing
    reports/figures/     final figures
    reports/tables/      final result tables
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OBSERVED = PROCESSED / "observed"
ASSIGNED = PROCESSED / "assigned"
PROXY = PROCESSED / "proxy"
SYNTHETIC = PROCESSED / "synthetic"
PUBLIC = DATA / "public"
CONFIGS = REPO_ROOT / "configs"
REPORTS = REPO_ROOT / "reports"
FIGURES = REPO_ROOT / "reports" / "figures"
TABLES = REPO_ROOT / "reports" / "tables"
SCRIPTS = REPO_ROOT / "scripts"
TESTS = REPO_ROOT / "tests"
NOTEBOOKS = REPO_ROOT / "notebooks"

# Key source files ----------------------------------------------------------
AAS2020_CHAPTER6 = RAW / "AAS2020-Excel-Tables/AAS2020 -UPLOAD EXCEL TABLES/AAS2020_Chapter 6_Annex.xlsx"
AAS2020_CHAPTER1 = RAW / "AAS2020-Excel-Tables/AAS2020 -UPLOAD EXCEL TABLES/AAS 2020 Chapter 1 Annex.xlsx"
AAS2018_PDF = RAW / "AAS2018.pdf"
CHIRPS_GLOBAL_NC = RAW / "chirps-v2.0.monthly.nc"
CHIRPS_UGANDA_NC = RAW / "uganda_chirps_clipped.nc"

# Climate API caches --------------------------------------------------------
CLIMATE_CACHE = RAW / "climate_cache"
CLIMATESERV_CACHE = CLIMATE_CACHE / "climateserv"
POWER_CACHE = CLIMATE_CACHE / "nasapower"


def ensure_dirs() -> None:
    for path in (
        RAW, EXTERNAL, INTERIM, PROCESSED, OBSERVED, ASSIGNED, PROXY, SYNTHETIC,
        PUBLIC, REPORTS, FIGURES, TABLES, CLIMATE_CACHE, CLIMATESERV_CACHE, POWER_CACHE,
    ):
        path.mkdir(parents=True, exist_ok=True)
