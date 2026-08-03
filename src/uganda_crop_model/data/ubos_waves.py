"""Validation hooks for additional UBOS survey waves.

Sources are accepted only when their schema proves the authoritative panel
grain and official yield definition. National/district-only or unparseable
files are documented as unavailable rather than coerced into the panel.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_GRAIN = {"subregion", "year", "season", "crop", "yield_tons_ha"}

def validate_wave_source(path: Path) -> dict:
    result = {"path": str(path), "valid": False, "reason": ""}
    if not path.exists():
        result["reason"] = "missing source"
        return result
    try:
        if path.suffix.lower() == ".pdf":
            from cropyield.data.aas2018 import load_aas2018_subregion
            panel = load_aas2018_subregion(path)
            required = {"sub_region", "year", "crop", "production_mt"}
            missing = required - set(panel.columns)
            if missing or panel.empty:
                result["reason"] = f"parsed PDF lacks required fields: {sorted(missing)}"
                return result
            result.update(valid=True, reason="validated with layout-adaptive AAS parser", rows=len(panel))
            return result
        frame = pd.read_excel(path) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path)
    except Exception as exc:
        result["reason"] = f"unparseable: {exc}"
        return result
    normalized = {str(c).strip().lower().replace(" ", "_") for c in frame.columns}
    missing = REQUIRED_GRAIN - normalized
    if missing:
        result["reason"] = f"missing required panel columns: {sorted(missing)}"
        return result
    result.update(valid=True, reason="validated authoritative panel grain", rows=len(frame))
    return result
