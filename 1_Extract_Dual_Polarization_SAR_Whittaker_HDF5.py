"""Extract aligned VV/VH pixel time series and Whittaker-smooth to HDF5.

The TIFFs are read on their shared pixel grid. No full-size aligned TIFF copies
or intermediate raw HDF5 files are created.
"""

import os
import re
from datetime import datetime

import h5py
import numpy as np
from osgeo import gdal
from scipy.linalg import solve
from tqdm import tqdm


TIFF_FOLDER = r"<PATH_TO_DUAL_BAND_SAR_TIFF_DIRECTORY>"
OUTPUT_DIR = r"<PATH_TO_OUTPUT_HDF5_DIRECTORY>"
VV_OUTPUT_H5 = os.path.join(OUTPUT_DIR, "VV_whittaker.h5")
VH_OUTPUT_H5 = os.path.join(OUTPUT_DIR, "VH_whittaker.h5")

VV_BAND = 1
VH_BAND = 2
LAMBDA = 1.0
CHUNK_ROWS = 64
CHUNK_COLS = 2048
NODATA_SENTINELS = (-9999.0,)
COMPRESSION_LEVEL = 5


def date_from_name(filename):
    match = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", filename)
    if not match:
        raise ValueError(f"TIFF filename does not contain a YYYY-MM-DD or YYYY_MM_DD date: {filename}")
    value = match.group(1).replace("_", "-")
    return datetime.strptime(value, "%Y-%m-%d"), value


def valid_values(array, nodata):
    mask = np.isfinite(array)
    for sentinel in NODATA_SENTINELS:
        mask &= array != sentinel
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def smooth_rows(values, lmbd):
    """Whittaker smooth each row, retaining NaNs at their original dates."""
    result = values.copy()
    patterns, inverse = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    for pattern_id, pattern in enumerate(patterns):
        valid_times = np.flatnonzero(pattern)
        if valid_times.size < 3:
            continue
        row_ids = np.flatnonzero(inverse == pattern_id)
        count = valid_times.size
        d2 = np.diff(np.eye(count), n=2, axis=0)
        system = np.eye(count) + lmbd * (d2.T @ d2)
        observations = values[np.ix_(row_ids, valid_times)].astype(np.float64, copy=False)
        result[np.ix_(row_ids, valid_times)] = solve(
            system, observations.T, assume_a="pos", check_finite=False
        ).T.astype(np.float32)
    return result


def inspect_and_align(paths):
    """Find the shared north-up grid and each source's row offset on it."""
    datasets = []
    for path in paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Cannot open TIFF: {path}")
        datasets.append(ds)

    reference = datasets[0]
    gt0 = reference.GetGeoTransform()
    projection = reference.GetProjection()
    if abs(gt0[2]) > 1e-12 or abs(gt0[4]) > 1e-12 or gt0[5] >= 0:
        raise ValueError("This script requires north-up geographic-coordinate rasters")
    pixel_x, pixel_y = gt0[1], abs(gt0[5])
    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("Invalid raster pixel size")

    # Shared intersection in pixel coordinates. These assets have the same
    # horizontal extent; row origins differ by integer pixels.
    offsets = []
    for ds, path in zip(datasets, paths):
        gt = ds.GetGeoTransform()
        if ds.GetProjection() != projection:
            raise ValueError(f"Raster projections do not match: {os.path.basename(path)}")
        if (ds.RasterXSize != reference.RasterXSize or
                not np.allclose([gt[0], gt[1], gt[2], gt[4], gt[5]],
                                [gt0[0], gt0[1], gt0[2], gt0[4], gt0[5]],
                                rtol=0, atol=1e-10)):
            raise ValueError(f"Horizontal grids or pixel sizes do not match: {os.path.basename(path)}")
        if ds.RasterCount < max(VV_BAND, VH_BAND):
            raise ValueError(f"Insufficient raster bands: {os.path.basename(path)}")
        vv_name = ds.GetRasterBand(VV_BAND).GetDescription().strip().upper()
        vh_name = ds.GetRasterBand(VH_BAND).GetDescription().strip().upper()
        if vv_name != "VV" or vh_name != "VH":
            raise ValueError(
                f"Band-order check failed: {os.path.basename(path)} has Band {VV_BAND}="
                f"{vv_name!r}, Band {VH_BAND}={vh_name!r} (expected VV and VH)"
            )
        row_offset_float = (gt[3] - gt0[3]) / gt0[5]
        row_offset = int(round(row_offset_float))
        if not np.isclose(row_offset_float, row_offset, atol=1e-6):
            raise ValueError(f"Raster rows are not aligned: {os.path.basename(path)}")
        offsets.append(row_offset)

    common_start = max(offsets)
    common_end = min(offset + ds.RasterYSize
                     for offset, ds in zip(offsets, datasets))
    common_rows = common_end - common_start
    if common_rows <= 0:
        raise ValueError("The TIFF images have no common spatial coverage")
    common_gt = list(gt0)
    common_gt[3] = gt0[3] + common_start * gt0[5]
    source_starts = [common_start - offset for offset in offsets]
    return datasets, common_gt, common_rows, source_starts, projection


def run():
    names = [n for n in os.listdir(TIFF_FOLDER)
             if n.lower().endswith((".tif", ".tiff"))]
    if not names:
        raise FileNotFoundError(f"No TIFF images found in directory: {TIFF_FOLDER}")
    dated = [(date_from_name(name), os.path.join(TIFF_FOLDER, name)) for name in names]
    dated.sort(key=lambda item: item[0][0])
    dates = [item[0][1] for item in dated]
    paths = [item[1] for item in dated]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.abspath(VV_OUTPUT_H5) == os.path.abspath(VH_OUTPUT_H5):
        raise ValueError("The VV and VH output paths must be different")

    datasets, gt, rows, source_starts, projection = inspect_and_align(paths)
    cols = datasets[0].RasterXSize
    reference_band = datasets[0].GetRasterBand(VV_BAND)
    reference_nodata = reference_band.GetNoDataValue()
    coord_count = 0

    print(f"Number of images: {len(paths)}; common grid: {rows} rows x {cols} columns")
    print(f"Rows cropped from the northern edge of the reference grid: {source_starts[0]} (other images are read using their grid offsets)")
    print(f"Band-order check passed: Band {VV_BAND}=VV，Band {VH_BAND}=VH")
    print(f"Whittaker smoothing: lambda={LAMBDA:g}; second-order differences; block size={CHUNK_ROWS} x {CHUNK_COLS}")

    # Count reference VV-valid pixels first so HDF5 dimensions are known.
    for y in tqdm(range(0, rows, CHUNK_ROWS), desc="Count valid pixels"):
        h = min(CHUNK_ROWS, rows - y)
        src_y = source_starts[0] + y
        for x in range(0, cols, CHUNK_COLS):
            w = min(CHUNK_COLS, cols - x)
            block = reference_band.ReadAsArray(x, src_y, w, h)
            if block is None:
                raise RuntimeError(f"Failed to read reference TIFF: x={x}, y={src_y}")
            coord_count += int(np.count_nonzero(valid_values(block, reference_nodata)))
    if coord_count == 0:
        raise RuntimeError("The reference TIFF contains no valid VV pixels")
    print(f"Number of extracted pixels: {coord_count:,}; number of time points: {len(paths)}")

    files = [h5py.File(VV_OUTPUT_H5, "w"), h5py.File(VH_OUTPUT_H5, "w")]
    try:
        h5sets = []
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for f, pol in zip(files, ("VV", "VH")):
            f.create_dataset("latitude", (coord_count,), dtype="f4", chunks=True,
                             compression="gzip", compression_opts=COMPRESSION_LEVEL)
            f.create_dataset("longitude", (coord_count,), dtype="f4", chunks=True,
                             compression="gzip", compression_opts=COMPRESSION_LEVEL)
            f.create_dataset("value", (coord_count, len(paths)), dtype="f4",
                             chunks=(min(2048, coord_count), len(paths)),
                             compression="gzip", compression_opts=COMPRESSION_LEVEL,
                             shuffle=True, fillvalue=np.nan)
            f.attrs["polarization"] = pol
            f.attrs["timestamps"] = np.asarray(dates, dtype=string_dtype)
            f.attrs["num_timestamps"] = len(paths)
            f.attrs["total_valid_pixels"] = coord_count
            f.attrs["reference_tiff"] = os.path.basename(paths[0])
            f.attrs["geotransform"] = gt
            f.attrs["projection"] = projection
            f.attrs["coordinate_system"] = "Pixel Center"
            f.attrs["processing"] = "Aligned extraction + Whittaker smoothing"
            f.attrs["ws_lambda"] = LAMBDA
            f.attrs["ws_difference_order"] = 2
            f.attrs["nodata_sentinels_excluded"] = np.asarray(
                NODATA_SENTINELS, dtype=np.float32)
            h5sets.append(f["value"])

        write_offset = 0
        for y in tqdm(range(0, rows, CHUNK_ROWS), desc="Extract and smooth time series"):
            h = min(CHUNK_ROWS, rows - y)
            for x in range(0, cols, CHUNK_COLS):
                w = min(CHUNK_COLS, cols - x)
                ref_y = source_starts[0] + y
                ref = reference_band.ReadAsArray(x, ref_y, w, h)
                valid = valid_values(ref, reference_nodata)
                local_rows, local_cols = np.where(valid)
                n = local_rows.size
                if n == 0:
                    continue
                vv = np.full((n, len(paths)), np.nan, dtype=np.float32)
                vh = np.full_like(vv, np.nan)
                for t, (ds, offset) in enumerate(zip(datasets, source_starts)):
                    src_y = offset + y
                    band_vv = ds.GetRasterBand(VV_BAND)
                    band_vh = ds.GetRasterBand(VH_BAND)
                    arr_vv = band_vv.ReadAsArray(x, src_y, w, h)
                    arr_vh = band_vh.ReadAsArray(x, src_y, w, h)
                    if arr_vv is None or arr_vh is None:
                        raise RuntimeError(f"Failed to read raster block: {paths[t]}")
                    vals_vv = np.asarray(arr_vv[local_rows, local_cols], dtype=np.float32)
                    vals_vh = np.asarray(arr_vh[local_rows, local_cols], dtype=np.float32)
                    ok_vv = valid_values(vals_vv, band_vv.GetNoDataValue())
                    ok_vh = valid_values(vals_vh, band_vh.GetNoDataValue())
                    vv[ok_vv, t] = vals_vv[ok_vv]
                    vh[ok_vh, t] = vals_vh[ok_vh]

                vv = smooth_rows(vv, LAMBDA)
                vh = smooth_rows(vh, LAMBDA)
                end = write_offset + n
                lon = gt[0] + (x + local_cols + 0.5) * gt[1]
                lat = gt[3] + (y + local_rows + 0.5) * gt[5]
                for f, values in zip(files, (vv, vh)):
                    f["longitude"][write_offset:end] = lon.astype(np.float32)
                    f["latitude"][write_offset:end] = lat.astype(np.float32)
                    f["value"][write_offset:end, :] = values
                write_offset = end
        if write_offset != coord_count:
            raise RuntimeError(f"Number of pixels written {write_offset} differs from the counted total {coord_count}")
        for f in files:
            f.attrs["smooth_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.flush()
        print(f"Smoothed VV HDF5 file: {VV_OUTPUT_H5}")
        print(f"Smoothed VH HDF5 file: {VH_OUTPUT_H5}")
    finally:
        for f in files:
            f.close()
        datasets.clear()


if __name__ == "__main__":
    run()
