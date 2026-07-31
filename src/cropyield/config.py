"""Configuration loading.

All pipeline behavior (crops, years, feature sets, validation, seeds) is
declared in YAML files under configs/ so that experiments are auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .data.paths import CONFIGS


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from configs/ by its stem, e.g. "data"."""
    path = CONFIGS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_data_config() -> dict[str, Any]:
    return load_config("data")


def load_features_config() -> dict[str, Any]:
    return load_config("features")


def load_models_config() -> dict[str, Any]:
    return load_config("models")
