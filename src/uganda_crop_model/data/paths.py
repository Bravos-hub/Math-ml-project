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
RUNS = REPORTS / "runs"
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

# AAS 2020 per-crop sub-region tables (named tables plus the consolidated
# "additional crops" table).  Only the ten food crops that report both
# production and harvested area are eligible for the multi-crop dataset;
# perennial crops (banana, coffee, cassava) only report production over
# planted area and are therefore excluded from the yield-over-harvested
# target definition.
AAS2020_CROP_TABLES = {
    "maize": INTERIM / "aas2020_maize_subregion.csv",
    "beans": INTERIM / "aas2020_beans_subregion.csv",
    "groundnuts": INTERIM / "aas2020_groundnuts_subregion.csv",
    "sorghum": INTERIM / "aas2020_sorghum_subregion.csv",
    "millet": INTERIM / "aas2020_millet_subregion.csv",
    "rice": INTERIM / "aas2020_rice_subregion.csv",
    "soya_beans": INTERIM / "aas2020_soya_beans_subregion.csv",
    "simsim": INTERIM / "aas2020_simsim_subregion.csv",
    "irish_potatoes": INTERIM / "aas2020_irish_potatoes_subregion.csv",
    "sweet_potatoes": INTERIM / "aas2020_sweet_potatoes_subregion.csv",
}

MULTI_CROP = (
    "maize",
    "beans",
    "groundnuts",
    "sorghum",
    "millet",
    "rice",
    "soya_beans",
    "simsim",
    "irish_potatoes",
    "sweet_potatoes",
)

# Final analysis outputs ---------------------------------------------------
FINAL_MAIZE_DATASET = PROCESSED / "final_maize_subregion_season_year.csv"
FINAL_MULTI_CROP_DATASET = PROCESSED / "final_multi_crop_subregion_season_year.csv"
FINAL_MULTI_CROP_SEASONAL_DATASET = PROCESSED / "final_multi_crop_seasonal.csv"
FINAL_MULTI_CROP_ANNUAL_DATASET = PROCESSED / "final_multi_crop_annual.csv"


def ensure_dirs() -> None:
    for path in (
        DATA,
        RAW,
        EXTERNAL,
        INTERIM,
        PROCESSED,
        OBSERVED,
        ASSIGNED,
        PROXY,
        SYNTHETIC,
        PUBLIC,
        REPORTS,
        RUNS,
        FIGURES,
        TABLES,
        TECHNICAL_REPORT,
        CONFIGS,
    ):
        path.mkdir(parents=True, exist_ok=True)
