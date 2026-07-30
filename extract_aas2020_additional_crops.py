#!/usr/bin/env python3
"""
Extract additional AAS 2020 Chapter 6 crop tables from the annex workbook
without requiring openpyxl.

Targets:
  - Cereals: millet, sorghum, rice, maize
  - Pulses/oilseeds: beans, soya_beans, simsim, groundnuts
  - Roots/tubers: cassava, sweet_potatoes, irish_potatoes
  - Bananas: banana_food, banana_sweet, banana_beer
  - Coffee: coffee_robusta, coffee_arabica

Outputs:
  - aas2020_additional_crops_subregion.csv
  - one crop-specific CSV per extracted table
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


WORKBOOK = Path("AAS2020-Excel-Tables/AAS2020 -UPLOAD EXCEL TABLES/AAS2020_Chapter 6_Annex.xlsx")
COMBINED_OUTPUT = Path("aas2020_additional_crops_subregion.csv")

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def season_value_map(groups, names):
    value_names = {}
    for cols in groups.values():
        for col, name in zip(cols, names):
            value_names[col] = name
    return value_names


def cv_value_map(groups, names):
    cv_names = {}
    for cols in groups.values():
        for col, name in zip(cols, names):
            cv_names[col] = name
    return cv_names


def make_spec(
    crop,
    sheet_xml,
    table_id,
    season_groups,
    value_field_names,
    cv_groups,
    cv_field_names,
    *,
    entity_col="B",
    value_rows=(5, 20),
    cv_entity_col=None,
    cv_rows=(5, 20),
):
    return {
        "crop": crop,
        "sheet_xml": sheet_xml,
        "table_id": table_id,
        "entity_col": entity_col,
        "value_rows": value_rows,
        "cv_entity_col": cv_entity_col or entity_col,
        "cv_rows": cv_rows,
        "season_groups": season_groups,
        "value_names": season_value_map(season_groups, value_field_names),
        "cv_groups": cv_groups,
        "cv_names": cv_value_map(cv_groups, cv_field_names),
    }


MAIZE_STYLE_GROUPS = {
    "first_season_2020": ["C", "D", "E", "F", "G"],
    "second_season_2020": ["I", "J", "K", "L", "M"],
    "total_2020": ["O", "P", "Q", "R", "S"],
}

MAIZE_STYLE_FIELDS = [
    "area_planted_ha",
    "area_harvested_ha",
    "production_mt",
    "yield_mt_per_ha_production_over_harvested",
    "yield_mt_per_ha_production_over_planted",
]

MAIZE_STYLE_CV_GROUPS = {
    "first_season_2020": ["V", "W", "X"],
    "second_season_2020": ["Z", "AA", "AB"],
    "total_2020": ["AD", "AE", "AF"],
}

MAIZE_STYLE_CV_FIELDS = [
    "cv_area_planted_pct",
    "cv_area_harvested_pct",
    "cv_production_pct",
]

BANANA_STYLE_GROUPS = {
    "first_season_2020": ["C", "D", "E"],
    "second_season_2020": ["G", "H", "I"],
    "total_2020": ["K", "L", "M"],
}

BANANA_STYLE_FIELDS = [
    "area_planted_ha",
    "production_mt",
    "yield_mt_per_ha_production_over_planted",
]

BANANA_STYLE_CV_GROUPS = {
    "first_season_2020": ["P", "Q"],
    "second_season_2020": ["S", "T"],
    "total_2020": ["V", "W"],
}

BANANA_STYLE_CV_FIELDS = [
    "cv_area_planted_pct",
    "cv_production_pct",
]

CASSAVA_STYLE_GROUPS = {
    "first_season_2020": ["C", "D", "E"],
    "second_season_2020": ["G", "H", "I"],
    "total_2020": ["K", "L", "M", "N", "O"],
}

CASSAVA_STYLE_FIELDS = [
    "area_planted_ha",
    "production_mt",
    "yield_mt_per_ha_production_over_planted",
]

CASSAVA_TOTAL_FIELDS = [
    "area_planted_ha",
    "area_harvested_ha",
    "production_mt",
    "yield_mt_per_ha_production_over_harvested",
    "yield_mt_per_ha_production_over_planted",
]

CASSAVA_STYLE_VALUE_NAMES = {
    **season_value_map(
        {
            "first_season_2020": CASSAVA_STYLE_GROUPS["first_season_2020"],
            "second_season_2020": CASSAVA_STYLE_GROUPS["second_season_2020"],
        },
        CASSAVA_STYLE_FIELDS,
    ),
    **season_value_map({"total_2020": CASSAVA_STYLE_GROUPS["total_2020"]}, CASSAVA_TOTAL_FIELDS),
}

CASSAVA_STYLE_CV_GROUPS = {
    "first_season_2020": ["R", "S", "T"],
    "second_season_2020": ["V", "W", "X"],
    "total_2020": ["Z", "AA", "AB"],
}

CASSAVA_STYLE_CV_FIELDS = [
    "cv_area_planted_pct",
    "cv_area_harvested_pct",
    "cv_production_pct",
]

COFFEE_STYLE_GROUPS = BANANA_STYLE_GROUPS
COFFEE_STYLE_FIELDS = BANANA_STYLE_FIELDS
COFFEE_STYLE_CV_GROUPS = BANANA_STYLE_CV_GROUPS
COFFEE_STYLE_CV_FIELDS = BANANA_STYLE_CV_FIELDS

CROP_SPECS = [
    make_spec(
        "millet",
        "xl/worksheets/sheet2.xml",
        "Table 6-3",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "sorghum",
        "xl/worksheets/sheet3.xml",
        "Table 6-5",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "rice",
        "xl/worksheets/sheet4.xml",
        "Table 6-7",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "beans",
        "xl/worksheets/sheet5.xml",
        "Table 6-9",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "soya_beans",
        "xl/worksheets/sheet6.xml",
        "Table 6-11",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "sweet_potatoes",
        "xl/worksheets/sheet7.xml",
        "Table 6-13",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "irish_potatoes",
        "xl/worksheets/sheet8.xml",
        "Table 6-15",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "simsim",
        "xl/worksheets/sheet9.xml",
        "Table 6-17",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "groundnuts",
        "xl/worksheets/sheet10.xml",
        "Table 6-19",
        MAIZE_STYLE_GROUPS,
        MAIZE_STYLE_FIELDS,
        MAIZE_STYLE_CV_GROUPS,
        MAIZE_STYLE_CV_FIELDS,
        cv_entity_col="U",
    ),
    make_spec(
        "banana_food",
        "xl/worksheets/sheet11.xml",
        "Table 6-21",
        BANANA_STYLE_GROUPS,
        BANANA_STYLE_FIELDS,
        BANANA_STYLE_CV_GROUPS,
        BANANA_STYLE_CV_FIELDS,
        cv_entity_col="O",
    ),
    make_spec(
        "banana_sweet",
        "xl/worksheets/sheet12.xml",
        "Table 6-23",
        BANANA_STYLE_GROUPS,
        BANANA_STYLE_FIELDS,
        BANANA_STYLE_CV_GROUPS,
        BANANA_STYLE_CV_FIELDS,
        cv_entity_col="O",
    ),
    make_spec(
        "banana_beer",
        "xl/worksheets/sheet13.xml",
        "Table 6-25",
        BANANA_STYLE_GROUPS,
        BANANA_STYLE_FIELDS,
        BANANA_STYLE_CV_GROUPS,
        BANANA_STYLE_CV_FIELDS,
        cv_entity_col="O",
    ),
    {
        "crop": "cassava",
        "sheet_xml": "xl/worksheets/sheet14.xml",
        "table_id": "Table 6-27",
        "entity_col": "B",
        "value_rows": (5, 20),
        "cv_entity_col": "Q",
        "cv_rows": (5, 20),
        "season_groups": CASSAVA_STYLE_GROUPS,
        "value_names": CASSAVA_STYLE_VALUE_NAMES,
        "cv_groups": CASSAVA_STYLE_CV_GROUPS,
        "cv_names": cv_value_map(CASSAVA_STYLE_CV_GROUPS, CASSAVA_STYLE_CV_FIELDS),
    },
    make_spec(
        "coffee_robusta",
        "xl/worksheets/sheet15.xml",
        "Table 6-29",
        COFFEE_STYLE_GROUPS,
        COFFEE_STYLE_FIELDS,
        COFFEE_STYLE_CV_GROUPS,
        COFFEE_STYLE_CV_FIELDS,
        cv_entity_col="O",
    ),
    make_spec(
        "coffee_arabica",
        "xl/worksheets/sheet16.xml",
        "Table 6-31",
        COFFEE_STYLE_GROUPS,
        COFFEE_STYLE_FIELDS,
        COFFEE_STYLE_CV_GROUPS,
        COFFEE_STYLE_CV_FIELDS,
        cv_entity_col="O",
    ),
]


def load_shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared = []
    for si in root.findall("a:si", NS):
        shared.append("".join(node.text or "" for node in si.iterfind(".//a:t", NS)))
    return shared


def load_sheet_cells(archive, shared, sheet_xml):
    root = ET.fromstring(archive.read(sheet_xml))
    cells = {}
    for row in root.find("a:sheetData", NS):
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r")
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", NS)
            value = "" if value_node is None else value_node.text or ""
            if cell_type == "s" and value != "":
                value = shared[int(value)]
            cells[ref] = value
    return cells


def to_number(value):
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def extract_values(cells, entity_col, rows, season_groups, value_names):
    start_row, end_row = rows
    records = []
    for row in range(start_row, end_row + 1):
        entity = str(cells.get(f"{entity_col}{row}", "")).strip()
        if not entity or entity.lower().startswith("notes"):
            continue
        for season_group, cols in season_groups.items():
            record = {"sub_region": entity, "season_group": season_group}
            has_value = False
            for col in cols:
                field = value_names[col]
                value = to_number(cells.get(f"{col}{row}"))
                record[field] = value
                if isinstance(value, float):
                    has_value = True
            if has_value:
                records.append(record)
    return pd.DataFrame(records)


def attach_cv(values_df, cells, entity_col, rows, cv_groups, cv_names):
    start_row, end_row = rows
    records = []
    for row in range(start_row, end_row + 1):
        entity = str(cells.get(f"{entity_col}{row}", "")).strip()
        if not entity:
            continue
        for season_group, cols in cv_groups.items():
            record = {"sub_region": entity, "season_group": season_group}
            has_value = False
            for col in cols:
                field = cv_names[col]
                value = to_number(cells.get(f"{col}{row}"))
                record[field] = value
                if isinstance(value, float):
                    has_value = True
            if has_value:
                records.append(record)
    if not records:
        return values_df
    cv_df = pd.DataFrame(records)
    return values_df.merge(cv_df, on=["sub_region", "season_group"], how="left", validate="one_to_one")


def extract_crop(archive, shared, spec):
    cells = load_sheet_cells(archive, shared, spec["sheet_xml"])
    values = extract_values(
        cells,
        entity_col=spec["entity_col"],
        rows=spec["value_rows"],
        season_groups=spec["season_groups"],
        value_names=spec["value_names"],
    )
    merged = attach_cv(
        values,
        cells,
        entity_col=spec["cv_entity_col"],
        rows=spec["cv_rows"],
        cv_groups=spec["cv_groups"],
        cv_names=spec["cv_names"],
    )
    merged.insert(0, "crop", spec["crop"])
    merged.insert(1, "table_id", spec["table_id"])
    merged.insert(2, "entity_type", "sub_region")
    merged["year"] = 2020
    return merged.sort_values(["sub_region", "season_group"]).reset_index(drop=True)


def output_name(crop):
    return Path(f"aas2020_{crop}_subregion.csv")


def main():
    print("=" * 70)
    print("  EXTRACT AAS 2020 ADDITIONAL CROP TABLES")
    print("=" * 70)

    with zipfile.ZipFile(WORKBOOK) as archive:
        shared = load_shared_strings(archive)
        frames = []
        for spec in CROP_SPECS:
            df = extract_crop(archive, shared, spec)
            out = output_name(spec["crop"])
            df.to_csv(out, index=False)
            frames.append(df)
            print(f"[✓] Saved: {out} ({len(df)} rows)")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_OUTPUT, index=False)
    print(f"[✓] Saved: {COMBINED_OUTPUT} ({len(combined)} rows)")
    print()
    print(combined.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
