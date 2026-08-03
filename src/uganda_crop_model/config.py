"""Configuration loading helpers for the final-analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from yaml import safe_load

from .data.paths import CONFIGS

SEASON_WINDOWS = {
    "seasons": {
        "first_season": {"start_month_day": "03-01", "end_month_day": "07-15"},
        "second_season": {"start_month_day": "08-15", "end_month_day": "12-15"},
        "annual": {"start_date_override_full_year": True},
    },
}


def load_yaml(name: str) -> dict:
    path = CONFIGS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    with open(path) as fh:
        return safe_load(fh)


def load_seasonal_windows() -> dict[str, tuple[str, str] | None]:
    """Return {season: (start_month_day, end_month_day)}.

    A season with ``None`` window (e.g. ``annual``) covers the full year.
    """
    try:
        config = load_yaml("final_maize_aas.yaml")
    except FileNotFoundError:
        config = SEASON_WINDOWS

    windows: dict[str, tuple[str, str] | None] = {}
    for name, spec in config["seasons"].items():
        if spec.get("start_date_override_full_year"):
            windows[name] = None
        else:
            windows[name] = (spec["start_month_day"], spec["end_month_day"])
    return windows