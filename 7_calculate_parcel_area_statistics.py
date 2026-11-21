"""Calculate parcel areas from crop-specific parcel-ID rasters.

Replace ROOT and each filename in NAMES with the corresponding input location.
Inputs must be unrotated EPSG:4326 rasters with positive integer parcel IDs.
Use the parcel rasters for the spatial extent to be summarized.
"""

from pathlib import Path
import json
import numpy as np
import rasterio
from pyproj import Geod

# Directory containing the crop-specific parcel rasters, including subdirectories.
ROOT = Path("PATH_TO_DIRECTORY_CONTAINING_CROP_PARCEL_RASTERS")

# Filenames of the parcel-ID rasters for each site and cropping pattern.
NAMES = {
    "Site 1 Wheat-rice": "SITE_1_WHEAT_RICE_PARCEL_ID_RASTER.tif",
    "Site 1 Rapeseed-rice": "SITE_1_RAPESEED_RICE_PARCEL_ID_RASTER.tif",
    "Site 1 Double rice": "SITE_1_DOUBLE_RICE_PARCEL_ID_RASTER.tif",
    "Site 2 Wheat-rice": "SITE_2_WHEAT_RICE_PARCEL_ID_RASTER.tif",
    "Site 2 Rapeseed-rice": "SITE_2_RAPESEED_RICE_PARCEL_ID_RASTER.tif",
    "Site 3 Wheat-rice": "SITE_3_WHEAT_RICE_PARCEL_ID_RASTER.tif",
    "Site 3 Rapeseed-rice": "SITE_3_RAPESEED_RICE_PARCEL_ID_RASTER.tif",
    "Site 3 Double rice": "SITE_3_DOUBLE_RICE_PARCEL_ID_RASTER.tif",
}


def inspect(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        valid = (data > 0) & src.read_masks(1).astype(bool)
        ids, counts = np.unique(data[valid], return_counts=True)
        if src.crs.to_epsg() != 4326 or src.transform.b != 0 or src.transform.d != 0:
            raise ValueError(f"Expected an unrotated WGS84 grid: {path}")
        geod = Geod(ellps="WGS84")
        t = src.transform
        row_areas = np.zeros(src.height, dtype=np.float64)
        for row in np.flatnonzero(valid.any(axis=1)):
            north = t.f + row * t.e
            south = north + t.e
            area, _ = geod.polygon_area_perimeter(
                [t.c, t.c + t.a, t.c + t.a, t.c],
                [north, north, south, south],
            )
            row_areas[row] = abs(area)
        rr, cc = np.nonzero(valid)
        areas = np.bincount(data[rr, cc].astype(np.int64), weights=row_areas[rr])
        parcel_ha = areas[ids.astype(np.int64)] / 10000.0
        return {
            "path": str(path), "shape": list(data.shape), "crs": str(src.crs),
            "resolution": list(src.res), "positive_pixels": int(counts.sum()),
            "unique_positive_values": len(ids), "minimum_value": int(ids.min()),
            "maximum_value": int(ids.max()),
            "naive_id_mean_ha": float(counts.mean() * .01),
            "naive_id_median_ha": float(np.median(counts) * .01),
            "mean_parcel_area_ha": float(parcel_ha.mean()),
            "median_parcel_area_ha": float(np.median(parcel_ha)),
            "total_area_ha": float(parcel_ha.sum()),
            "parcel_areas_ha": dict(zip(map(str, ids), map(float, parcel_ha))),
        }


if __name__ == "__main__":
    results = []
    all_paths = list(ROOT.rglob("*.tif"))
    for label, name in NAMES.items():
        paths = [p for p in all_paths if p.name == name]
        for path in paths:
            result = {"label": label, **inspect(path)}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "parcel_areas_ha"}, ensure_ascii=True), flush=True)
    Path("parcel_area_source_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Parcel areas calculated from the original crop-specific rasters, grouping pixels by positive parcel ID. WGS84 ellipsoidal pixel areas are calculated separately for each latitude row; units: ha.", "", "| Dataset | Parcels | Pixels | Mean area (ha) | Median area (ha) | Total area (ha) |", "|---|---:|---:|---:|---:|---:|"]
    for x in results:
        lines.append(f"| {x['label']} | {x['unique_positive_values']} | {x['positive_pixels']} | {x['mean_parcel_area_ha']:.4f} | {x['median_parcel_area_ha']:.4f} | {x['total_area_ha']:.4f} |")
    lines.extend(["", "Input files:"] + [f"- {x['label']}: `{x['path']}`" for x in results])
    Path("parcel_area_recalculated.md").write_text("\n".join(lines), encoding="utf-8")
