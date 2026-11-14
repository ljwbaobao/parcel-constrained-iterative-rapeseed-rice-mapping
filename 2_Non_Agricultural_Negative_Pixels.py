"""Export random pixel-center coordinates outside the supplied field boundaries.

The sampling region is the study-area mask minus the field-parcel mask.
The supplied parcel layer determines which pixels are treated as non-agricultural.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy


RASTER_TEMPLATE_PATH = r"<PATH_TO_SAR_GRID_TEMPLATE_TIFF>"
STUDY_AREA_VECTOR_PATH = r"<PATH_TO_STUDY_AREA_BOUNDARY_VECTOR>"
FIELD_PARCEL_VECTOR_PATH = r"<PATH_TO_FIELD_PARCEL_BOUNDARY_VECTOR>"
OUTPUT_DIRECTORY = r"<PATH_TO_NEGATIVE_PIXEL_COORDINATE_OUTPUT_DIRECTORY>"
NUMBER_OF_NEGATIVE_PIXELS = "<NUMBER_OF_NON_AGRICULTURAL_NEGATIVE_PIXELS_TO_SAMPLE>"
RANDOM_SEED = 42


def main():
    try:
        requested = int(NUMBER_OF_NEGATIVE_PIXELS)
    except (ValueError, TypeError) as error:
        raise ValueError("Set NUMBER_OF_NEGATIVE_PIXELS to a positive integer.") from error
    if requested <= 0:
        raise ValueError("NUMBER_OF_NEGATIVE_PIXELS must be positive.")
    paths = [RASTER_TEMPLATE_PATH, STUDY_AREA_VECTOR_PATH,
             FIELD_PARCEL_VECTOR_PATH, OUTPUT_DIRECTORY]
    if any("<PATH_TO_" in path for path in paths):
        raise ValueError("Replace the descriptive path placeholders before running.")

    with rasterio.open(RASTER_TEMPLATE_PATH) as source:
        transform, crs, shape = source.transform, source.crs, source.shape
    boundary = gpd.read_file(STUDY_AREA_VECTOR_PATH).to_crs(crs)
    parcels = gpd.read_file(FIELD_PARCEL_VECTOR_PATH).to_crs(crs)
    boundary_shapes = [(g, 1) for g in boundary.geometry if g is not None and not g.is_empty]
    parcel_shapes = [(g, 1) for g in parcels.geometry if g is not None and not g.is_empty]
    if not boundary_shapes:
        raise ValueError("The study-area layer contains no valid geometries.")
    boundary_mask = rasterize(boundary_shapes, out_shape=shape, transform=transform,
                              fill=0, all_touched=False, dtype="uint8")
    parcel_mask = (rasterize(parcel_shapes, out_shape=shape, transform=transform,
                             fill=0, all_touched=False, dtype="uint8")
                   if parcel_shapes else np.zeros(shape, dtype=np.uint8))
    rows, cols = np.where((boundary_mask == 1) & (parcel_mask == 0))
    if not len(rows):
        raise ValueError("No candidate pixels remain outside the field-parcel mask.")
    rng = np.random.RandomState(RANDOM_SEED)
    chosen = (rng.choice(len(rows), size=requested, replace=False)
              if len(rows) > requested else np.arange(len(rows)))
    x, y = xy(transform, rows[chosen], cols[chosen], offset="center")
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs).to_crs(4326)
    output = Path(OUTPUT_DIRECTORY) / "non_agricultural_negative_pixel_coordinates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    points["longitude"] = points.geometry.x
    points["latitude"] = points.geometry.y
    points[["longitude", "latitude"]].to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Candidate pixels outside field boundaries: {len(rows):,}")
    print(f"Exported {len(points):,} pixel-center coordinates to {output}")


if __name__ == "__main__":
    main()
