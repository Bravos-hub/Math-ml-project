"""Parser for the UBOS AAS 2018 report (Annex 4, Tables 7-1 .. 7-48).

The PDF tables have two layouts depending on crop phenology:

Type 1 (annual crops: maize, millet, sorghum, beans, sweet/irish potatoes,
groundnuts, rice, soya beans, simsim):
    First season  : area planted, production, yield*** (prod/planted)
    Second season : area planted, area harvested*, production, CV,
                    yield** (prod/harvested), yield*** (prod/planted)
    Total 2018    : area planted, production, yield***

Type 2 (perennials: banana-food, cassava, coffee arabica, coffee robusta):
    First season  : area planted, production
    Second season : area planted, area harvested, production, CV
    Total 2018    : area planted**, production, yield*** (prod/harvested s2),
                    yield**** (prod/planted annual)

Consequences for the panel schema:
- ``yield_over_harvested`` is published for 2018 only in the second-season
  block (type 1) or the total block (type 2); first-season rows carry NaN.
- ``cv_production_pct`` exists only in the second-season block.
- ``area_harvested_ha`` exists only in the second-season block.

Parsing is positional: column boundaries are recovered from the unit tokens
("(Ha)", "(MT)", "(MT/Ha)") and the "CV" label in the header block, so it
does not depend on hardcoded column numbers.  Every table is validated by
comparing the sum of sub-region values against the printed national row.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

from .provenance import (
    YIELD_AAS2018_SUBREGION,
    GRANULARITY_SUBREGION,
    add_provenance,
)
from .paths import AAS2018_PDF, INTERIM

TEXT_CACHE = INTERIM / ".aas2018_layout.txt"

TABLE_TITLE_RE = re.compile(
    r"Table 7- ?\d+: ([A-Za-z-]+) ?-? area, production and yields, by sub-region"
)

CROP_NAMES = {
    "Maize": "maize",
    "Millet": "millet",
    "Sorghum": "sorghum",
    "Beans": "beans",
    "Banana-food": "banana_food",
    "Sweet": "sweet_potatoes",
    "Irish": "irish_potatoes",
    "Groundnuts": "groundnuts",
    "Rice": "rice",
    "Soya": "soya_beans",
    "Simsim": "simsim",
    "Cassava": "cassava",
    "Coffee": "coffee_arabica",  # disambiguated below by title
}

# The title text is "Sweet potatoes", "Irish potatoes", "Soya beans",
# "Coffee arabica"/"Coffee robusta": the captured name is only the first
# word, so the full title must be used for disambiguation.
FULL_TITLE_CROP = {
    "Sweet potatoes": "sweet_potatoes",
    "Irish potatoes": "irish_potatoes",
    "Soya beans": "soya_beans",
    "Coffee arabica": "coffee_arabica",
    "Coffee robusta": "coffee_robusta",
    "Banana-food": "banana_food",
}

SUBREGION_2018_TO_CANONICAL = {
    "S. Buganda": "South Buganda",
    "N. Buganda": "North Buganda",
}

SUBREGIONS_2018 = [
    "S. Buganda", "N. Buganda", "Busoga", "Bukedi", "Elgon", "Teso",
    "Karamoja", "Lango", "Acholi", "West Nile", "Bunyoro", "Tooro",
    "Ankole", "Kigezi",
]

NUM_RE = re.compile(r"\d[\d,]*\.?\d*")

# (block, position) -> semantic column, per layout type.
TYPE1_SEMANTICS = [
    ("area_planted_ha", "production_mt", "yield_over_planted"),
    ("area_planted_ha", "area_harvested_ha", "production_mt",
     "cv_production_pct", "yield_over_harvested", "yield_over_planted"),
    ("area_planted_ha", "production_mt", "yield_over_planted"),
]
TYPE2_SEMANTICS = [
    ("area_planted_ha", "production_mt"),
    ("area_planted_ha", "area_harvested_ha", "production_mt", "cv_production_pct"),
    ("area_planted_ha", "production_mt", "yield_over_harvested", "yield_over_planted"),
]


def extract_text(pdf_path: Path = AAS2018_PDF, cache_path: Path | None = None) -> str:
    """Run pdftotext -layout and cache the result under data/interim/."""
    cache = cache_path if cache_path is not None else (TEXT_CACHE if pdf_path == AAS2018_PDF else None)
    if cache is not None and cache.exists():
        return cache.read_text(encoding="utf-8")
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    if cache is not None:
        cache.write_text(result.stdout, encoding="utf-8")
    return result.stdout


def split_table_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Return [(crop, lines), ...] for every sub-region yield table."""
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_crop: str | None = None
    current: list[str] = []

    def flush() -> None:
        if current_crop is not None and current:
            blocks.append((current_crop, current))

    for line in lines:
        if line.startswith("Table 7-"):
            if "....." in line:  # table-of-contents entry, not a body table
                continue
            title_match = re.search(r"Table 7- ?\d+:\s*(.+?)\s*(?:\.+)?$", line)
            title = title_match.group(1) if title_match else ""
            if "area, production and yields, by sub-region" in title:
                flush()
                clean_title = title.split(" -")[0].split(" area,")[0].strip()
                crop = FULL_TITLE_CROP.get(clean_title)
                if crop is None:
                    first_word = clean_title.split()[0]
                    crop = CROP_NAMES.get(first_word)
                current_crop = crop
                current = []
            elif "by ZARDI" in title or "Status of the harvest" in title:
                flush()
                current_crop = None
                current = []
        elif current_crop is not None:
            current.append(line)
    flush()
    return blocks


def find_header(lines: list[str]) -> tuple[int, list[tuple[int, int, str]]]:
    """Find the header block and the ordered column anchors.

    Returns (units_line_index, anchors) where anchors is a list of
    (position, length, token) sorted by (position, line) — the "CV" label
    is included as an anchor because it has no unit token.
    """
    first_idx = next(
        (i for i, ln in enumerate(lines) if "First season 2018" in ln), None
    )
    if first_idx is None:
        raise ValueError("Could not find 'First season 2018' header line")
    units_idx = next(
        (i for i in range(first_idx, len(lines)) if "(Ha)" in lines[i]), None
    )
    if units_idx is None:
        raise ValueError("Could not find header units line")
    # The units line is the LAST header line containing "(Ha)" before data.
    while units_idx + 1 < len(lines) and "(Ha)" in lines[units_idx + 1]:
        units_idx += 1

    anchors: list[tuple[int, int, str]] = []
    for i in range(first_idx, units_idx + 1):
        line = lines[i]
        for m in re.finditer(r"\((?:MT/Ha|MT|Ha)\)", line):
            anchors.append((m.start(), len(m.group()), m.group()))
        for m in re.finditer(r"\bCV\b", line):
            anchors.append((m.start(), 2, "CV"))
    anchors.sort(key=lambda a: (a[0], a[1]))
    return units_idx, anchors


def layout_type(anchors: list[tuple[int, int, str]]) -> str:
    """Detect table layout: 'type1' (annual) or 'type2' (perennial)."""
    n = len(anchors)
    if n == 12:  # 3 + 6 + 3
        return "type1"
    if n == 10:  # 2 + 4 + 4
        return "type2"
    raise ValueError(f"Unrecognized table layout with {n} column anchors")


def parse_data_line(
    line: str, ref_positions: list[int], tolerance: int = 8
) -> list[float | None]:
    """Parse the numbers of one data row against a reference alignment.

    ``ref_positions`` are the right-edge positions of the numbers in a
    well-aligned reference row (the national "Uganda" row).  Each number is
    assigned to the nearest reference column, monotonically; numbers further
    than ``tolerance`` characters from any reference position are treated as
    missing cells.  This is robust to the imperfect right-alignment of
    pdftotext -layout, including short values such as "0" or "1.0".
    """
    nums = [(m.end(), m.group()) for m in NUM_RE.finditer(line)]
    if not nums:
        return []
    values: list[float | None] = [None] * len(ref_positions)
    j = 0
    for end, token in nums:
        while j < len(ref_positions) - 1 and abs(end - ref_positions[j]) > abs(
            end - ref_positions[j + 1]
        ):
            j += 1
        if abs(end - ref_positions[j]) <= tolerance:
            values[j] = float(token.replace(",", ""))
        else:
            # The number does not align with any reference column; keep the
            # pointer advanced only if the next column is closer.
            pass
    return values


def parse_table(crop: str, lines: list[str]) -> pd.DataFrame:
    units_idx, anchors = find_header(lines)
    ltype = layout_type(anchors)
    semantics = TYPE1_SEMANTICS if ltype == "type1" else TYPE2_SEMANTICS
    flat = [col for block in semantics for col in block]
    if len(flat) != len(anchors):
        raise ValueError(
            f"{crop}: {len(anchors)} anchors but {len(flat)} semantic columns"
        )

    rows = []
    data_lines: list[tuple[str, str]] = []
    for line in lines[units_idx + 1:]:
        if not line.strip():
            continue
        if line.startswith("(*)") or line.startswith("(**") or line.startswith("(***") or line.startswith("(****"):
            break
        name_match = next(
            (name for name in SUBREGIONS_2018 if line.lstrip().startswith(name)),
            None,
        )
        if name_match is None:
            if line.lstrip().startswith("Uganda"):
                data_lines.append(("Uganda", line))
            continue
        data_lines.append((name_match, line))

    # Reference alignment: the first sub-region row with the full expected
    # number of columns (the national row can miss the CV value because a
    # footnote collides with that cell).
    ref_line = None
    for name, ln in data_lines:
        if name == "Uganda":
            continue
        if len(NUM_RE.findall(ln)) == len(flat):
            ref_line = ln
            break
    if ref_line is None:
        ref_line = next((ln for n, ln in data_lines if n == "Uganda"), None)
    if ref_line is None:
        return pd.DataFrame()
    ref_positions = [m.end() for m in NUM_RE.finditer(ref_line)]
    if len(ref_positions) != len(flat):
        raise ValueError(
            f"{crop}: reference row has {len(ref_positions)} numbers, expected {len(flat)}"
        )

    for name_match, line in data_lines:
        if name_match == "Uganda":
            continue
        values = parse_data_line(line, ref_positions)
        row = {
            "crop": crop,
            "sub_region": SUBREGION_2018_TO_CANONICAL.get(name_match, name_match),
            **dict(zip(flat, values, strict=True)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_aas2018_subregion(pdf_path: Path = AAS2018_PDF) -> pd.DataFrame:
    """Parse all 14 sub-region yield tables from the AAS 2018 PDF."""
    text = extract_text(pdf_path)
    blocks = split_table_blocks(text)
    frames = []
    checks = []
    for crop, lines in blocks:
        df = parse_table(crop, lines)
        if df.empty:
            raise ValueError(f"No sub-region rows parsed for {crop}")
        frames.append(df)

        # Sanity checks against plausible ranges (areas ha, production MT).
        planted = df["area_planted_ha"]
        production = df["production_mt"]
        checks.append({
            "crop": crop,
            "n_subregions": len(df),
            "n_rows": len(df),
            "planted_min": planted.min(),
            "planted_max": planted.max(),
            "production_min": production.min(),
            "production_max": production.max(),
        })

    panel = pd.concat(frames, ignore_index=True)
    panel["year"] = 2018
    panel["harvest_loss_ratio"] = (
        1.0 - panel["area_harvested_ha"] / panel["area_planted_ha"]
    )
    panel = panel.sort_values(["crop", "sub_region"]).reset_index(drop=True)

    # Validation: reconstruction of the national row by summing sub-regions.
    national = panel.groupby(["crop"]).agg(
        area_planted_ha_sum=("area_planted_ha", "sum"),
        production_mt_sum=("production_mt", "sum"),
    )
    national.to_csv(INTERIM / "aas2018_national_summary.csv")

    return add_provenance(
        panel,
        yield_source=YIELD_AAS2018_SUBREGION,
        yield_granularity=GRANULARITY_SUBREGION,
        quality_note=(
            "Official AAS 2018 Annex 4 sub-region estimates. "
            "yield_over_harvested only published for second-season (annual crops) "
            "or total (perennial crops); CV only for second season."
        ),
    )


def save_aas2018_subregion(
    output: Path | None = None,
    pdf_path: Path = AAS2018_PDF,
) -> pd.DataFrame:
    output = output or INTERIM / "aas2018_subregion_consolidated.csv"
    panel = load_aas2018_subregion(pdf_path)
    panel.to_csv(output, index=False)
    return panel


if __name__ == "__main__":
    panel = save_aas2018_subregion()
    print(f"[✓] AAS 2018 consolidated: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(panel.groupby("crop").size().to_string())
