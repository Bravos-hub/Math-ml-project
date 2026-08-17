"""Validation hooks for additional UBOS survey waves.

Sources are accepted only when their schema proves the authoritative panel
grain and official yield definition. National/district-only or unparseable
files are documented as unavailable rather than coerced into the panel.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

REQUIRED_GRAIN = {"subregion", "year", "season", "crop", "yield_tons_ha"}


def validate_ddi_codebook(path: Path) -> dict:
    """Inspect a UBOS DDI codebook without mistaking metadata for data."""
    result = {"path": str(path), "valid": False, "status": "unavailable", "reason": ""}
    if not path.exists():
        result["reason"] = "missing codebook"
        return result
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        result["reason"] = f"unparseable XML: {exc}"
        return result
    labels = [text.text.strip() for text in root.iter() if text.tag.endswith("labl") and text.text]
    files = [text.text.strip() for text in root.iter() if text.tag.endswith("fileName") and text.text]
    title = next((text.text.strip() for text in root.iter() if text.tag.endswith("titl") and text.text), "")
    year = next((int(token) for token in title.split() if token.isdigit() and len(token) == 4), None)
    lower = " ".join(labels + files).lower()
    signals = {
        "subregion_metadata": "sub-region" in lower or "sub_region" in lower,
        "crop_metadata": "crop" in lower,
        "production_metadata": "production" in lower,
        "area_metadata": "area" in lower or "harvest" in lower,
        "season_metadata": "season" in lower or "first season" in lower,
    }
    result.update(
        valid=True,
        status="metadata_only",
        title=title,
        year=year,
        file_count=len(files),
        signals=signals,
        reason="DDI codebook confirms survey metadata; microdata or published target tables are still required",
    )
    return result

def validate_wave_source(path: Path) -> dict:
    result = {"path": str(path), "valid": False, "reason": ""}
    if not path.exists():
        result["reason"] = "missing source"
        return result
    if path.suffix.lower() == ".xml":
        return validate_ddi_codebook(path)
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
        reason = f"missing required panel columns: {sorted(missing)}"
        if path.suffix.lower() in {".xls", ".xlsx"}:
            try:
                from uganda_crop_model.data.aas_2015_2021 import detect_grain
                grain = detect_grain(path)
                reason += f"; detected grain: {grain.grain} ({grain.evidence})"
                if grain.grain != "subregion":
                    reason += (
                        " — a {0}-grain series cannot enter the subregion panel; "
                        "national abstract tables can still serve via "
                        "aas_2015_2021.build_national_yield_panel".format(grain.grain)
                    )
            except Exception:
                pass  # grain detection is advisory; never mask the real reason
        result["reason"] = reason
        return result
    result.update(valid=True, reason="validated authoritative panel grain", rows=len(frame))
    return result
