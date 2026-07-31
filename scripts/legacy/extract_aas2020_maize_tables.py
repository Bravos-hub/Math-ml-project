#!/usr/bin/env python3
"""
Extract AAS 2020 maize tables from the Chapter 6 annex workbook without
requiring openpyxl.

Targets:
  - Table 6-1: Maize area and production, by sub-Region
  - Table 6-2: Maize area and production, by ZARDI

Outputs:
  - aas2020_maize_subregion.csv
  - aas2020_maize_zardi.csv
"""

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


WORKBOOK = Path("AAS2020-Excel-Tables/AAS2020 -UPLOAD EXCEL TABLES/AAS2020_Chapter 6_Annex.xlsx")
SUBREGION_OUTPUT = Path("aas2020_maize_subregion.csv")
ZARDI_OUTPUT = Path("aas2020_maize_zardi.csv")

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}

SEASON_GROUPS = {
    "first_season_2020": ["B", "C", "D", "E", "F"],
    "second_season_2020": ["H", "I", "J", "K", "L"],
    "total_2020": ["N", "O", "P", "Q", "R"],
}

CV_GROUPS = {
    "first_season_2020": ["V", "W", "X"],
    "second_season_2020": ["Z", "AA", "AB"],
    "total_2020": ["AD", "AE", "AF"],
}

VALUE_COLUMN_NAMES = {
    "B": "area_planted_ha",
    "C": "area_harvested_ha",
    "D": "production_mt",
    "E": "yield_mt_per_ha_production_over_harvested",
    "F": "yield_mt_per_ha_production_over_planted",
    "H": "area_planted_ha",
    "I": "area_harvested_ha",
    "J": "production_mt",
    "K": "yield_mt_per_ha_production_over_harvested",
    "L": "yield_mt_per_ha_production_over_planted",
    "N": "area_planted_ha",
    "O": "area_harvested_ha",
    "P": "production_mt",
    "Q": "yield_mt_per_ha_production_over_harvested",
    "R": "yield_mt_per_ha_production_over_planted",
}

CV_COLUMN_NAMES = {
    "V": "cv_area_planted_pct",
    "W": "cv_area_harvested_pct",
    "X": "cv_production_pct",
    "Z": "cv_area_planted_pct",
    "AA": "cv_area_harvested_pct",
    "AB": "cv_production_pct",
    "AD": "cv_area_planted_pct",
    "AE": "cv_area_harvested_pct",
    "AF": "cv_production_pct",
}


def load_shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared = []
    for si in root.findall("a:si", NS):
        text = "".join(node.text or "" for node in si.iterfind(".//a:t", NS))
        shared.append(text)
    return shared


def load_sheet_cells():
    with zipfile.ZipFile(WORKBOOK) as archive:
        shared = load_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    cells = {}
    for row in sheet.find("a:sheetData", NS):
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
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def extract_table(cells, key_col, start_row, end_row, value_groups, value_names):
    records = []
    for row in range(start_row, end_row + 1):
        key = str(cells.get(f"{key_col}{row}", "")).strip()
        if not key or key.lower().startswith("notes"):
            continue

        for group_name, cols in value_groups.items():
            record = {"entity": key, "season_group": group_name}
            non_null_measure = False
            for col in cols:
                field = value_names[col]
                value = to_number(cells.get(f"{col}{row}"))
                record[field] = value
                if isinstance(value, float):
                    non_null_measure = True
            if non_null_measure:
                records.append(record)

    return pd.DataFrame(records)


def attach_cv(values_df, cells, start_row, end_row):
    cv_records = []
    for row in range(start_row, end_row + 1):
        key = str(cells.get(f"U{row}", "")).strip()
        if not key:
            continue
        for group_name, cols in CV_GROUPS.items():
            record = {"entity": key, "season_group": group_name}
            non_null_measure = False
            for col in cols:
                field = CV_COLUMN_NAMES[col]
                value = to_number(cells.get(f"{col}{row}"))
                record[field] = value
                if isinstance(value, float):
                    non_null_measure = True
            if non_null_measure:
                cv_records.append(record)

    cv_df = pd.DataFrame(cv_records)
    merged = values_df.merge(cv_df, on=["entity", "season_group"], how="left", validate="one_to_one")
    return merged


def add_metadata(df, entity_type, table_id, crop):
    df = df.copy()
    df.insert(0, "crop", crop)
    df.insert(1, "table_id", table_id)
    df.insert(2, "entity_type", entity_type)
    return df


def main():
    print("=" * 70)
    print("  EXTRACT AAS 2020 MAIZE TABLES")
    print("=" * 70)

    cells = load_sheet_cells()

    subregion_values = extract_table(
        cells,
        key_col="A",
        start_row=5,
        end_row=20,
        value_groups=SEASON_GROUPS,
        value_names=VALUE_COLUMN_NAMES,
    )
    subregion = attach_cv(subregion_values, cells, start_row=5, end_row=20)
    subregion = add_metadata(subregion, entity_type="sub_region", table_id="Table 6-1", crop="maize")
    subregion = subregion.rename(columns={"entity": "sub_region"})
    subregion.to_csv(SUBREGION_OUTPUT, index=False)

    zardi_values = extract_table(
        cells,
        key_col="A",
        start_row=28,
        end_row=38,
        value_groups=SEASON_GROUPS,
        value_names=VALUE_COLUMN_NAMES,
    )
    zardi = attach_cv(zardi_values, cells, start_row=28, end_row=38)
    zardi = add_metadata(zardi, entity_type="zardi", table_id="Table 6-2", crop="maize")
    zardi = zardi.rename(columns={"entity": "zardi"})
    zardi.to_csv(ZARDI_OUTPUT, index=False)

    print(f"[✓] Saved: {SUBREGION_OUTPUT}")
    print(f"[✓] Rows: {len(subregion)}")
    print(subregion.head(9).to_string(index=False))
    print()
    print(f"[✓] Saved: {ZARDI_OUTPUT}")
    print(f"[✓] Rows: {len(zardi)}")
    print(zardi.head(9).to_string(index=False))


if __name__ == "__main__":
    main()
