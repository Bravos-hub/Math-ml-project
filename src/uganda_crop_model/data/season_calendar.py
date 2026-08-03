"""Generate the documented season calendar for the sub-region analysis.

The calendar maps every analytical ``(spatial_unit, year, season)`` row to a
rainfall/temperature aggregation window.  For the AAS sample the window
dates come from the regional two-rains pattern encoded in
``configs/final_maize_aas.yaml`` and are a documented research assumption:

* first_season  : between 03-01 and 07-15
* second_season : between 08-15 and 12-15
* annual        : the full calendar year (AAS 2018 total rows only)

Region-specific agronomic calendars are planned sensitivity analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from yaml import safe_load

from ..config import load_seasonal_windows
from .paths import DISTRICT_MAP, PUBLIC

DEFAULT_YEARS = [2018, 2020]
DEFAULT_SEASONS = ["first_season", "second_season", "annual"]


def make_season_calendar(
    district_map_file: Path = DISTRICT_MAP,
    *,
    years: list[int] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    years = years or DEFAULT_YEARS
    seasons = seasons or DEFAULT_SEASONS
    windows = load_seasonal_windows()

    regions = pd.read_csv(district_map_file)["sub_region"].unique()

    rows = []
    for spatial_unit in sorted(regions):
        for year in years:
            for season in seasons:
                spec = windows[season]
                if spec is None:
                    start = f"{year}-01-01"
                    end = f"{year}-12-31"
                else:
                    start = f"{year}-{spec[0]}"
                    end = f"{year}-{spec[1]}"
                rows.append(
                    {
                        "spatial_unit": spatial_unit,
                        "year": int(year),
                        "season": season,
                        "start_date": start,
                        "end_date": end,
                    }
                )

    return pd.DataFrame(rows)


def generate_and_save_calendar(output: Path | None = None) -> pd.DataFrame:
    output = output or PUBLIC / "season_calendar.csv"
    calendar = make_season_calendar()
    output.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(output, index=False)
    return calendar


if __name__ == "__main__":
    from uganda_crop_model.data.paths import ensure_dirs

    ensure_dirs()
    calendar = generate_and_save_calendar()
    print(calendar.to_string(index=False))