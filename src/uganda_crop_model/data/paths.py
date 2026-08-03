"""Central path constants for the final-analysis package.

Layout (see README):

    data/raw/            original downloads (PDFs, xlsx, NetCDF, API caches)
    data/interim/        partially processed tables
    data/processed/      model-ready analytic files
    data/public/         shareable, auditable files (season calendar, etc.)
    configs/             YAML configuration
    reports/             figures/, tables/, technical_report/
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
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
TECHNICAL_REPORT = REPORTS / "technical_report"
SCRIPTS = REPO_ROOT / "scripts"
TESTS = REPO_ROOT / "tests"
NOTEBOOKS = REPO_ROOT / "notebooks"

# Source files -------------------------------------------------------------
AAS2020_MAIZE = INTERIM / "aas2020_maize_subregion.csv"
AAS2018_MAIZE = INTERIM / "aas2018_subregion_consolidated.csv"
DISTRICT_MAP = INTERIM / "uganda_districts_114.csv"
DAILY_RAINFALL = INTERIM / "uganda_daily_rainfall_climateserv.csv"
DAILY_TEMPERATURE = INTERIM / "uganda_daily_temperature_nasapower.csv"

# Final analysis outputs ---------------------------------------------------
FINAL_MAIZE_DATASET = PROCESSED / "final_maize_subregion_season_year.csv"


def ensure_dirs() -> None:
    for path in (
        DATA, RAW, EXTERNAL, INTERIM, PROCESSED, OBSERVED, ASSIGNED, PROXY,
        SYNTHETIC, PUBLIC, REPORTS, FIGURES, TABLES, TECHNICAL_REPORT,
        CONFIGS,
    ):
        path.mkdir(parents=True, exist_ok=True)