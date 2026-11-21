import os
import gc
from datetime import datetime
import numpy as np
import h5py
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from psutil import virtual_memory
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from itertools import product


# -------------------------
# Utility functions
# -------------------------
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def get_unified_block_size(bytes_per_pixel, safety_factor=0.1, min_block_size=10000, max_block_size=200000):
    available_mem = virtual_memory().available
    max_pixels_per_block = int(available_mem * safety_factor / bytes_per_pixel)
    return clamp(max_pixels_per_block, min_block_size, max_block_size)


def log_memory_usage(stage):
    mem = virtual_memory()
    used_gb = mem.used / 1024 / 1024 / 1024
    available_gb = mem.available / 1024 / 1024 / 1024
    print(f"[{datetime.now()}] Memory usage - {stage}: used {used_gb:.2f} GB, available {available_gb:.2f} GB")


# -------------------------
# Raster and HDF5 data access
# -------------------------
def get_timestamps_from_merged(merged_h5_path):
    print(f"[{datetime.now()}] Reading HDF5 metadata: {os.path.basename(merged_h5_path)}")
    with h5py.File(merged_h5_path, "r") as f:
        if "timestamps" not in f.attrs:
            raise ValueError(f"HDF5 file is missing the 'timestamps' attribute: {merged_h5_path}")
        timestamps = f.attrs["timestamps"].tolist()
        try:
            [datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {e}")
        if "value" not in f:
            raise ValueError(f"HDF5 file is missing the 'value' dataset: {merged_h5_path}")
        total_pixels = f["value"].shape[0]
    print(f"  Time points: {len(timestamps)}, pixels: {total_pixels:,}")
    return timestamps, total_pixels


def read_h5_data_from_merged(merged_h5_path, indices, dataset_name="value"):
    indices = np.asarray(indices, dtype=np.int64).flatten()
    if len(indices) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    sorted_idx = np.argsort(indices)
    reversed_idx = np.argsort(sorted_idx)
    sorted_indices = indices[sorted_idx]

    sub_batch_size = 10000
    num_sub_batches = (len(sorted_indices) + sub_batch_size - 1) // sub_batch_size
    data_list = []

    with h5py.File(merged_h5_path, "r") as f:
        dataset = f[dataset_name]
        for i in tqdm(range(num_sub_batches), desc="Read HDF5 data", leave=False):
            start = i * sub_batch_size
            end = min(start + sub_batch_size, len(sorted_indices))
            sub_idx = sorted_indices[start:end]
            data_list.append(dataset[sub_idx, :].astype(np.float32))
            gc.collect()

    if not data_list:
        return np.zeros((0, 0), dtype=np.float32)

    data = np.vstack(data_list)[reversed_idx]
    del data_list, sorted_idx, reversed_idx, sorted_indices
    gc.collect()
    return data


def load_npy_mapping(mapping_path, template_tif_path):
    print(f"\n[{datetime.now()}] Loading NumPy mapping: {os.path.basename(mapping_path)}")
    log_memory_usage("before loading mapping")

    with rasterio.open(template_tif_path) as src:
        tif_rows, tif_cols = src.height, src.width
        tif_transform = src.transform
        tif_crs = src.crs

    rowcol_to_h5idx = np.load(mapping_path, mmap_mode='r')
    if rowcol_to_h5idx.shape != (tif_rows, tif_cols):
        raise ValueError(f"NumPy mapping and raster dimensions differ: NumPy {rowcol_to_h5idx.shape} vs raster ({tif_rows}, {tif_cols})")

    max_h5idx = -1
    block_rows = 256
    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Find maximum HDF5 index"):
        row_end = min(row_start + block_rows, tif_rows)
        block = rowcol_to_h5idx[row_start:row_end, :]
        block_max = block[block != -1].max() if (block != -1).any() else -1
        if block_max > max_h5idx:
            max_h5idx = block_max
        gc.collect()
    if max_h5idx == -1:
        raise ValueError("No valid HDF5 indices found in the NumPy mapping")
    print(f"  Maximum HDF5 index: {max_h5idx:,}")

    h5idx_to_row = np.full((max_h5idx + 1), -1, dtype=np.int32)
    h5idx_to_col = np.full((max_h5idx + 1), -1, dtype=np.int32)
    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Build reverse mapping"):
        row_end = min(row_start + block_rows, tif_rows)
        mapping_block = rowcol_to_h5idx[row_start:row_end, :]
        valid_row, valid_col = np.where(mapping_block != -1)
        if len(valid_row) == 0:
            continue
        global_row = valid_row + row_start
        valid_h5idx = mapping_block[valid_row, valid_col].astype(np.int32)
        valid_mask = valid_h5idx <= max_h5idx
        h5idx_to_row[valid_h5idx[valid_mask]] = global_row[valid_mask]
        h5idx_to_col[valid_h5idx[valid_mask]] = valid_col[valid_mask]
        gc.collect()

    print(f"  Valid mapping entries: {(h5idx_to_row != -1).sum():,}")
    log_memory_usage("after loading mapping")
    return rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_transform, tif_crs


def get_field_h5idx(field_data, field_nodata, rowcol_to_h5idx, h5idx_to_row, tif_rows, tif_cols):
    print(f"\n[{datetime.now()}] Selecting field-parcel pixels")
    log_memory_usage("before pixel selection")

    field_h5idx = []
    block_rows, block_cols = 128, 128
    num_blocks = (tif_rows // block_rows + 1) * (tif_cols // block_cols + 1)
    pbar = tqdm(total=num_blocks, desc="Map field pixels to HDF5 indices")

    for row_start in range(0, tif_rows, block_rows):
        row_end = min(row_start + block_rows, tif_rows)
        for col_start in range(0, tif_cols, block_cols):
            col_end = min(col_start + block_cols, tif_cols)
            field_block = field_data[row_start:row_end, col_start:col_end]
            mapping_block = rowcol_to_h5idx[row_start:row_end, col_start:col_end]
            valid_mask = (field_block != field_nodata) & (field_block > 0) & (mapping_block != -1)
            if valid_mask.any():
                block_h5 = mapping_block[valid_mask].astype(np.int32)
                block_h5 = block_h5[(block_h5 >= 0) & (block_h5 < len(h5idx_to_row))]
                if len(block_h5) > 0:
                    field_h5idx.append(block_h5)
            del field_block, mapping_block, valid_mask
            gc.collect()
            pbar.update(1)
    pbar.close()

    if not field_h5idx:
        raise ValueError("No valid pixels found inside field parcels")
    field_h5idx = np.unique(np.concatenate(field_h5idx))
    print(f"  Valid field pixels: {len(field_h5idx):,}")
    log_memory_usage("after pixel selection")
    return field_h5idx


def preprocess_field_data(label_tif_path, tif_rows, tif_cols):
    print(f"\n[{datetime.now()}] Reading parcel raster: {os.path.basename(label_tif_path)}")
    with rasterio.open(label_tif_path) as src:
        if (src.height, src.width) != (tif_rows, tif_cols):
            raise ValueError(f"Parcel raster and template dimensions differ: parcel {src.height}x{src.width} vs template {tif_rows}x{tif_cols}")
        field_data = src.read(1, out_dtype=np.int32)
        field_nodata = src.nodata
    print(f"  Parcel raster NoData value: {field_nodata}")
    return field_data, field_nodata


# -------------------------
# Field and pixel mapping functions
# -------------------------
def compute_field_zones(current_result, field_h5idx, h5_to_field, all_field_ids):
    print(f"\n[{datetime.now()}] Assigning parcels to target-proportion intervals from the current pixel-wise map")
    field_total = np.bincount(h5_to_field, minlength=max(all_field_ids) + 1)
    field_rot = np.bincount(h5_to_field[current_result == 1], minlength=max(all_field_ids) + 1)

    field_total = np.nan_to_num(field_total, nan=0, posinf=0, neginf=0)
    field_rot = np.nan_to_num(field_rot, nan=0, posinf=0, neginf=0)
    field_ratio = np.where(field_total > 0, field_rot / field_total, 0)

    zero_ratio_fields = [fid for fid in all_field_ids if field_ratio[fid] == 0]
    field_zone = {}
    for fid in all_field_ids:
        if fid in zero_ratio_fields:
            continue
        ratio = field_ratio[fid]
        if 0 < ratio <= 0.2:
            field_zone[fid] = 1
        elif 0.2 < ratio <= 0.4:
            field_zone[fid] = 2
        elif 0.4 < ratio <= 0.6:
            field_zone[fid] = 3
        elif 0.6 < ratio <= 0.8:
            field_zone[fid] = 4
        else:
            field_zone[fid] = 5

    zone_fields = {1: [], 2: [], 3: [], 4: [], 5: []}
    for fid, zone in field_zone.items():
        zone_fields[zone].append(fid)

    print(f"  Parcels with 0% target pixels: {len(zero_ratio_fields)}")
    print(f"  Parcels with >0% target pixels: {len(all_field_ids) - len(zero_ratio_fields)}")
    interval_labels = {1: "(0, 20%]", 2: "(20, 40%]", 3: "(40, 60%]", 4: "(60, 80%]", 5: "(80, 100%]"}
    for zone in range(1, 6):
        print(f"  Interval {zone} {interval_labels[zone]}: {len(zone_fields[zone])} parcels")

    total_valid_fields = sum(len(zone) for zone in zone_fields.values())
    print(f"  Check: parcels with >0% target pixels={len(all_field_ids) - len(zero_ratio_fields)}, parcels assigned to intervals={total_valid_fields}")
    return zone_fields, np.asarray(zero_ratio_fields, dtype=np.int32)


def load_initial_positive_h5idx(initial_pos_tif, field_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols):
    print(f"\n[{datetime.now()}] Loading initial positive samples: {os.path.basename(initial_pos_tif)}")
    initial_pos_h5idx = []
    with rasterio.open(initial_pos_tif) as src:
        batch_size = 50000
        for i in tqdm(range(0, len(field_h5idx), batch_size), desc="Read initial positive samples"):
            batch_h5 = field_h5idx[i:i + batch_size]
            r = h5idx_to_row[batch_h5]
            c = h5idx_to_col[batch_h5]
            valid_rc = (r >= 0) & (r < tif_rows) & (c >= 0) & (c < tif_cols)
            if not valid_rc.any():
                continue
            r_valid = r[valid_rc]
            c_valid = c[valid_rc]
            tif_values = src.read(1)[r_valid, c_valid]
            pos_mask_in_batch = tif_values == 1
            initial_pos_h5idx.extend(batch_h5[valid_rc][pos_mask_in_batch])

    initial_pos_h5idx = np.unique(initial_pos_h5idx)
    if len(initial_pos_h5idx) == 0:
        raise ValueError("No valid pixels with value 1 were found in the initial positive-sample raster")
    print(f"  Valid initial positive samples: {len(initial_pos_h5idx):,}")
    return initial_pos_h5idx


def build_features(vh_data, vv_data):
    vh_data = np.nan_to_num(vh_data, nan=1e-6, posinf=10, neginf=1e-6)
    vv_data = np.nan_to_num(vv_data, nan=1e-6, posinf=10, neginf=1e-6)

    vh = np.clip(vh_data, 1e-6, 10)
    vv = np.clip(vv_data, 1e-6, 10)

    vh_vv = vh / vv
    vh_vv = np.nan_to_num(vh_vv, nan=1e-6, posinf=10, neginf=1e-6)

    vh_vv_sum = vh + vv
    rvi = np.where(vh_vv_sum > 1e-6, 4 * vh / vh_vv_sum, 1e-6)
    rvi = np.nan_to_num(rvi, nan=1e-6, posinf=10, neginf=1e-6)

    return np.hstack([vh, vv, vh_vv, rvi])


# -------------------------
# Field-level majority voting for output maps
# -------------------------
def adjust_result_by_field_ratio(pred_result, h5_to_field, all_field_ids):
    """Apply majority voting to produce a field-level classification map."""
    print(f"\n[{datetime.now()}] Applying field-level majority voting (>50% target pixels: target; otherwise: non-target)")

    max_field_id = max(all_field_ids) if all_field_ids.size > 0 else 0
    field_total = np.bincount(h5_to_field, minlength=max_field_id + 1)
    field_rot = np.bincount(h5_to_field, weights=pred_result, minlength=max_field_id + 1)

    field_ratio = np.zeros_like(field_total, dtype=np.float32)
    valid_mask = field_total > 0
    field_ratio[valid_mask] = field_rot[valid_mask] / field_total[valid_mask]

    adjusted_result = pred_result.copy()
    for fid in tqdm(all_field_ids, desc="Apply field-level majority voting"):
        if field_total[fid] == 0:
            continue
        field_mask = h5_to_field == fid
        ratio = field_ratio[fid]
        if ratio >= 0.5:
            adjusted_result[field_mask] = 1
        else:
            adjusted_result[field_mask] = 0

    before_rot = np.sum(pred_result == 1)
    after_rot = np.sum(adjusted_result == 1)
    print(f"  Target pixels before voting: {before_rot:,}")
    print(f"  Target pixels after voting: {after_rot:,}")
    print(f"  Change: {after_rot - before_rot:,}")

    return adjusted_result


# -------------------------
# Model training and pixel classification
# -------------------------
def train_and_predict(initial_pos_h5idx, cumulative_new_pos_h5idx, neg_h5idx, field_h5idx, vh_path, vv_path, common_ts):
    combined_pos_h5idx = np.unique(np.concatenate([initial_pos_h5idx, cumulative_new_pos_h5idx]))
    pos_count = len(combined_pos_h5idx)
    print(
        f"\n[{datetime.now()}] Training random forest: {len(initial_pos_h5idx):,} initial positives + {len(cumulative_new_pos_h5idx):,} added positives = {pos_count:,} unique positives"
    )

    if len(neg_h5idx) < pos_count:
        raise ValueError(f"Insufficient negative samples: {len(neg_h5idx):,} available; {pos_count:,} required")
    np.random.seed(42)
    sampled_neg_h5idx = np.random.choice(neg_h5idx, size=pos_count, replace=False)
    print(f"  Negative samples: {len(neg_h5idx):,} available, {len(sampled_neg_h5idx):,} sampled")

    vh_pos = read_h5_data_from_merged(vh_path, combined_pos_h5idx)
    vv_pos = read_h5_data_from_merged(vv_path, combined_pos_h5idx)
    vh_neg = read_h5_data_from_merged(vh_path, sampled_neg_h5idx)
    vv_neg = read_h5_data_from_merged(vv_path, sampled_neg_h5idx)

    X_pos = build_features(vh_pos, vv_pos)
    X_neg = build_features(vh_neg, vv_neg)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.uint8)
    X, y = shuffle(X, y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print(f"  OOB \u888b\u5916\u51c6\u786e\u7387\u4f30\u8ba1: {rf.oob_score_:.4f}")

    print(f"[{datetime.now()}] Predicting all field-parcel pixels")
    bytes_per_pixel = 4 * len(common_ts) * 4
    batch_size = get_unified_block_size(bytes_per_pixel)
    pred_result = np.zeros(len(field_h5idx), dtype=np.uint8)

    for i in tqdm(range(0, len(field_h5idx), batch_size), desc="Prediction progress"):
        batch_h5 = field_h5idx[i:i + batch_size]
        vh_batch = read_h5_data_from_merged(vh_path, batch_h5)
        vv_batch = read_h5_data_from_merged(vv_path, batch_h5)
        X_batch = build_features(vh_batch, vv_batch)
        pred_result[i:i + batch_size] = rf.predict(X_batch)
        gc.collect()

    del vh_pos, vv_pos, vh_neg, vv_neg, X_pos, X_neg, X, y, rf, combined_pos_h5idx, sampled_neg_h5idx
    gc.collect()
    return pred_result


def build_h5_to_field(field_data, field_nodata, field_h5idx, rowcol_to_h5idx, h5idx_to_row, tif_rows):
    print(f"\n[{datetime.now()}] Building HDF5-index to parcel-ID mapping")
    h5_to_field = np.full(len(field_h5idx), -1, dtype=np.int32)
    block_rows = 256

    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Build parcel mapping"):
        row_end = min(row_start + block_rows, tif_rows)
        field_block = field_data[row_start:row_end, :]
        mapping_block = rowcol_to_h5idx[row_start:row_end, :]
        valid_mask = (field_block != field_nodata) & (mapping_block != -1)
        if not valid_mask.any():
            continue
        block_h5 = mapping_block[valid_mask].astype(np.int32)
        block_fid = field_block[valid_mask].astype(np.int32)
        pos_in_field = np.searchsorted(field_h5idx, block_h5)
        valid_pos = field_h5idx[pos_in_field] == block_h5
        h5_to_field[pos_in_field[valid_pos]] = block_fid[valid_pos]
        gc.collect()

    all_field_ids = np.unique(h5_to_field[h5_to_field != -1])
    print(f"  Valid parcels: {len(all_field_ids):,}")
    return h5_to_field, all_field_ids


# -------------------------
# Read a previous pixel-wise classification map
# -------------------------
def read_prev_result(prev_tif_path, field_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols):
    """Read a prior classification map and return predictions for field pixels."""
    print(f"\n[{datetime.now()}] Reading initial classification: {os.path.basename(prev_tif_path)}")
    result = np.zeros(len(field_h5idx), dtype=np.uint8)
    with rasterio.open(prev_tif_path) as src:
        batch_size = 50000
        for i in tqdm(range(0, len(field_h5idx), batch_size), desc="Read classification"):
            batch_h5 = field_h5idx[i:i + batch_size]
            r = h5idx_to_row[batch_h5]
            c = h5idx_to_col[batch_h5]
            valid_rc = (r >= 0) & (r < tif_rows) & (c >= 0) & (c < tif_cols)
            if not valid_rc.any():
                continue
            r_valid = r[valid_rc]
            c_valid = c[valid_rc]
            result[i:i + batch_size][valid_rc] = (src.read(1)[r_valid, c_valid] == 1).astype(np.uint8)
            gc.collect()
    print(f"  Initial map: {np.sum(result == 1):,} target pixels, {np.sum(result == 0):,} non-target pixels")
    return result


def write_new_result(result, field_h5idx, output_path, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_crs,
                     tif_transform):
    print(f"\n[{datetime.now()}] Writing classification output: {os.path.basename(output_path)}")
    log_memory_usage("before writing output")

    meta = {
        "driver": "GTiff",
        "height": tif_rows,
        "width": tif_cols,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": tif_crs,
        "transform": tif_transform,
        "nodata": 255
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        init_block = np.full((1024, tif_cols), 255, dtype=np.uint8)
        for row_start in tqdm(range(0, tif_rows, 1024), desc="Initialize output raster"):
            row_end = min(row_start + 1024, tif_rows)
            dst.write(init_block[:row_end - row_start], 1, window=Window(0, row_start, tif_cols, row_end - row_start))

    with rasterio.open(output_path, "r+") as dst:
        batch_size = 50000
        for i in tqdm(range(0, len(field_h5idx), batch_size), desc="Write output pixels"):
            batch_h5 = field_h5idx[i:i + batch_size]
            batch_res = result[i:i + batch_size]
            r = h5idx_to_row[batch_h5]
            c = h5idx_to_col[batch_h5]
            valid_rc = (r >= 0) & (r < tif_rows) & (c >= 0) & (c < tif_cols)
            if not valid_rc.any():
                continue
            r_valid = r[valid_rc]
            c_valid = c[valid_rc]
            res_valid = batch_res[valid_rc]

            for row in np.unique(r_valid):
                row_mask = r_valid == row
                cols = c_valid[row_mask]
                res = res_valid[row_mask]
                if len(cols) == 0:
                    continue

                min_col = min(cols)
                max_col = max(cols)
                window_width = max_col - min_col + 1
                window = Window(min_col, row, window_width, 1)

                window_data = np.full(window_width, 255, dtype=np.uint8)
                col_offsets = cols - min_col
                window_data[col_offsets] = res

                dst.write(window_data.reshape(1, -1), 1, window=window)

            gc.collect()

    log_memory_usage("after writing output")
    return output_path


# -------------------------
# Run iterative sample supplementation and classification
# -------------------------
def main():
    overall_start = datetime.now()
    print("=" * 80)
    print(f"[{overall_start}] Starting automated evaluation of 125 three-interval strategies (cache pixel-wise predictions; write majority-vote maps)")
    print("=" * 80)

    # -------------------------
    # Configure input and output paths.
    # -------------------------
    config = {
        "vh_h5": r"<PATH_TO_TONGXIANG_MERGED_WHITTAKER_VH_HDF5>",
        "vv_h5": r"<PATH_TO_TONGXIANG_MERGED_WHITTAKER_VV_HDF5>",
        "npy_mapping": r"<PATH_TO_TONGXIANG_PIXEL_TO_HDF5_INDEX_MAPPING_NPY>",
        "template_tif": r"<PATH_TO_TONGXIANG_SENTINEL1_TEMPLATE_TIF>",
        "initial_tif": r"<PATH_TO_TONGXIANG_INITIAL_PIXELWISE_RF_CLASSIFICATION_TIF>",
        "initial_positive_tif": r"<PATH_TO_TONGXIANG_RULE_SCREENED_INITIAL_POSITIVE_TIF>",
        "field_tif": r"<PATH_TO_TONGXIANG_FIELD_PARCEL_ID_RASTER_TIF>",
        "output_dir": r"<PATH_TO_TONGXIANG_ITERATIVE_CLASSIFICATION_OUTPUT_DIRECTORY>"
    }

    for key, path in config.items():
        if key != "output_dir" and not os.path.exists(path):
            raise FileNotFoundError(f"Input path does not exist: {key}={path}")
    os.makedirs(config["output_dir"], exist_ok=True)

    # -------------------------
    # Generate all three-interval strategy sequences.
    # -------------------------
    all_cmds = [''.join(map(str, combo)) for combo in product([1, 2, 3, 4, 5], repeat=3)]
    print(f"\nGenerated {len(all_cmds)} three-interval strategies (111-555)")

    # -------------------------
    # Load raster, parcel, and time-series data.
    # -------------------------
    print(f"\n" + "=" * 60)
    print(f"[{datetime.now()}] Loading shared input data")
    print("=" * 60)
    rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_transform, tif_crs = load_npy_mapping(
        config["npy_mapping"], config["template_tif"]
    )
    field_data, field_nodata = preprocess_field_data(config["field_tif"], tif_rows, tif_cols)
    field_h5idx = get_field_h5idx(field_data, field_nodata, rowcol_to_h5idx, h5idx_to_row, tif_rows, tif_cols)
    h5_to_field, all_field_ids = build_h5_to_field(field_data, field_nodata, field_h5idx, rowcol_to_h5idx, h5idx_to_row,
                                                   tif_rows)
    initial_pos_h5idx = load_initial_positive_h5idx(
        config["initial_positive_tif"], field_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols
    )
    vh_ts, _ = get_timestamps_from_merged(config["vh_h5"])
    vv_ts, _ = get_timestamps_from_merged(config["vv_h5"])
    common_ts = sorted(list(set(vh_ts) & set(vv_ts)), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    print(f"\nCommon time points: {len(common_ts)}")

    # -------------------------
    # Cache raw pixel-wise results for strategy-prefix reuse.
    # -------------------------
    # Load the initial classification and initialize the cache.
    initial_raw_result = read_prev_result(
        config["initial_tif"], field_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols
    )
    result_cache = {
        "": {
            "raw_result": initial_raw_result,
            "output_path": config["initial_tif"],
            "cumulative_pos": np.array([], dtype=np.int32)
        }
    }

    # -------------------------
    # Run each strategy, reusing cached prefixes where possible.
    # -------------------------
    completed_cmds = 0
    for full_cmd in all_cmds:
        cmd_idx = all_cmds.index(full_cmd) + 1
        print(f"\n" + "=" * 100)
        print(f"[{datetime.now()}] Processing strategy {full_cmd} ({cmd_idx}/{len(all_cmds)})")
        print("=" * 100)

        cmd_list = [int(ch) for ch in full_cmd]
        total_iter = len(cmd_list)
        current_prefix = ""

        # Find the longest strategy prefix already in the cache.
        for prefix_len in range(total_iter-1, -1, -1):
            candidate_prefix = full_cmd[:prefix_len]
            if candidate_prefix in result_cache:
                current_prefix = candidate_prefix
                current_cache = result_cache[current_prefix]
                # Reuse cached pixel-wise predictions and accumulated samples.
                current_raw_result = current_cache["raw_result"]
                current_cumulative_pos = current_cache["cumulative_pos"]
                print(f"  Reusing cached strategy prefix '{current_prefix}': {len(current_raw_result):,} field pixels")
                print(f"  Reusing cumulative samples for prefix '{current_prefix}': {len(current_cumulative_pos):,}")
                break

        # Process the remaining iterations in this strategy.
        start_iter = len(current_prefix)
        for iter_step in range(start_iter, total_iter):
            target_zone = cmd_list[iter_step]
            iter_num = iter_step + 1
            print(f"\n" + "-" * 80)
            print(f"  Iteration {iter_num}/{total_iter}: selected interval {target_zone}")
            print("-" * 80)

            # Calculate parcel intervals from the current pixel-wise map.
            zone_fields, zero_ratio_fields = compute_field_zones(
                current_raw_result, field_h5idx, h5_to_field, all_field_ids
            )
            # Build the negative-sample pool from pixels in 0% target parcels.
            zero_field_mask = np.isin(h5_to_field, zero_ratio_fields)
            negative_pool_h5idx = field_h5idx[zero_field_mask]
            print(f"  0% \u76ee\u6807\u50cf\u5143\u6bd4\u4f8b\u7530\u5757\u7684\u8d1f\u6837\u672c\u6c60: {len(negative_pool_h5idx):,} \u4e2a\u50cf\u5143")
            if len(zone_fields[target_zone]) == 0:
                print(f"  Interval {target_zone} contains no parcels; skipping this strategy")
                break

            # Select parcels in the chosen interval.
            target_fids = zone_fields[target_zone]
            new_pos_mask = np.isin(h5_to_field, target_fids) & (current_raw_result == 0)
            new_pos_h5idx = field_h5idx[new_pos_mask]
            if len(new_pos_h5idx) == 0:
                print(f"  Interval {target_zone} yielded no new positive samples; skipping this strategy")
                break
            print(f"  New positive pseudo-labels this iteration: {len(new_pos_h5idx):,}")

            # Add the new pseudo-labels to the cumulative positive sample set.
            new_cumulative_pos = np.unique(np.concatenate([current_cumulative_pos, new_pos_h5idx]))
            print(f"  Cumulative added positive samples: {len(new_cumulative_pos):,}")

            # Train the RF model and classify field pixels.
            try:
                new_raw_result = train_and_predict(
                    initial_pos_h5idx=initial_pos_h5idx,
                    cumulative_new_pos_h5idx=new_cumulative_pos,
                    neg_h5idx=negative_pool_h5idx,
                    field_h5idx=field_h5idx,
                    vh_path=config["vh_h5"],
                    vv_path=config["vv_h5"],
                    common_ts=common_ts
                )
            except Exception as e:
                print(f"  Training or prediction failed: {str(e)}; skipping this strategy")
                break

            # Apply majority voting when writing the output map.
            adjusted_result = adjust_result_by_field_ratio(new_raw_result, h5_to_field, all_field_ids)

            # Create the output file path.
            current_step_prefix = full_cmd[:iter_num]
            output_name = f"Tongxiang_strategy_{full_cmd}_iteration_{iter_num}.tif"
            output_path = os.path.join(config["output_dir"], output_name)

            # Write the majority-vote classification map.
            try:
                current_output_path = write_new_result(
                    adjusted_result, field_h5idx, output_path,
                    h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_crs, tif_transform
                )
            except Exception as e:
                print(f"  Output writing failed: {str(e)}; skipping this strategy")
                break

            # Cache pixel-wise predictions and accumulated positive samples.
            result_cache[current_step_prefix] = {
                "raw_result": new_raw_result,
                "output_path": current_output_path,
                "cumulative_pos": new_cumulative_pos
            }
            print(f"  Cached pixel-wise results for prefix '{current_step_prefix}'")

            # Release temporary arrays.
            del current_raw_result, new_raw_result, adjusted_result, new_pos_h5idx, zone_fields, new_cumulative_pos
            gc.collect()
            # Continue from the latest pixel-wise predictions.
            current_raw_result = result_cache[current_step_prefix]["raw_result"]
            current_cumulative_pos = result_cache[current_step_prefix]["cumulative_pos"]

        # Record whether the full strategy completed.
        if full_cmd in result_cache:
            completed_cmds += 1
            print(
                f"\n[{datetime.now()}] Strategy {full_cmd} completed. Output: {os.path.basename(result_cache[full_cmd]['output_path'])}")
        else:
            print(f"\n[{datetime.now()}] Strategy {full_cmd} failed or was skipped")

    # -------------------------
    # Summarize completed strategies and output locations.
    # -------------------------
    overall_elapsed = (datetime.now() - overall_start).total_seconds()
    print(f"\n" + "=" * 80)
    print(f"[{datetime.now()}] All strategies processed")
    print(f"  Total strategies: {len(all_cmds)}")
    print(f"  Completed strategies: {completed_cmds}")
    print(f"  Elapsed time: {overall_elapsed / 3600:.2f} h ({overall_elapsed:.2f} s)")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Cached strategy prefixes: {len(result_cache)}")
    print("=" * 80)

    # Release remaining arrays.
    del field_data, field_h5idx, h5_to_field, initial_pos_h5idx, initial_raw_result
    del rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, result_cache
    gc.collect()


if __name__ == "__main__":
    main()
