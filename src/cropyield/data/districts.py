"""District-level reference table for the AAS 2020 frame.

Builds ``data/interim/uganda_districts_114.csv`` with columns:

    district, sub_region, lat, lon

from two sources:
  - district -> sub-region mapping: AAS 2020 Chapter 1, Table 1-2 (official);
  - centroids: UN OCHA admin-2 boundaries (geoBoundaries gbHumanitarian,
    2020), polygon centroid (mean of boundary coordinates).

All 114 districts of the AAS frame are present in the OCHA layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import AAS2020_CHAPTER1, INTERIM, RAW

OCHA_GEOJSON = RAW / "uganda_adm2_2020.geojson"
OUTPUT = INTERIM / "uganda_districts_114.csv"


def read_subregion_mapping() -> pd.DataFrame:
    t12 = pd.read_excel(AAS2020_CHAPTER1, sheet_name="Table 1-2", header=None)
    rows = []
    for j, sub in enumerate(t12.iloc[1].tolist()):
        for i in range(2, t12.shape[0]):
            value = t12.iloc[i, j]
            if pd.notna(value):
                rows.append({"district": str(value).strip(), "sub_region": sub})
    return pd.DataFrame(rows)


def read_ochoa_centroids() -> pd.DataFrame:
    import json

    with OCHA_GEOJSON.open("r", encoding="utf-8") as fh:
        geojson = json.load(fh)
    rows = []
    for feature in geojson["features"]:
        name = feature["properties"]["shapeName"]
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            rings = geom["coordinates"]
        else:  # MultiPolygon: use the largest ring of the largest polygon
            rings = max(geom["coordinates"], key=len)[0] if len(geom["coordinates"]) > 1 else geom["coordinates"][0]
        largest = max(rings, key=len) if geom["type"] == "Polygon" else rings
        if geom["type"] == "MultiPolygon":
            poly = max(geom["coordinates"], key=lambda p: len(p[0]))
            largest = poly[0]
        xs = [p[0] for p in largest]
        ys = [p[1] for p in largest]
        rows.append({"district": name, "lat": sum(ys) / len(ys), "lon": sum(xs) / len(xs)})
    return pd.DataFrame(rows)


def build_district_table(output: Path | None = None) -> pd.DataFrame:
    mapping = read_subregion_mapping()
    centroids = read_ochoa_centroids()
    table = mapping.merge(centroids, on="district", how="left", validate="one_to_one")
    missing = table["lat"].isna().sum()
    if missing:
        raise ValueError(f"{missing} districts missing centroids")
    table = table.sort_values(["sub_region", "district"]).reset_index(drop=True)
    output = output or OUTPUT
    table.to_csv(output, index=False)
    return table


if __name__ == "__main__":
    table = build_district_table()
    print(f"[✓] {OUTPUT.name}: {table.shape[0]} districts x {table.shape[1]} cols")
    print(table.groupby("sub_region").size().to_string())
