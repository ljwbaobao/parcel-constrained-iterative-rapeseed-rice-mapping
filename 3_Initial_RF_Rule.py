import os
import re
import glob
import numpy as np
import h5py
from scipy.signal import argrelextrema
from datetime import datetime
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import joblib
from psutil import virtual_memory
import gc
import pandas as pd


# -------------------------
# Global configuration and utility functions
# -------------------------
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def get_unified_block_size(bytes_per_pixel, safety_factor=0.1, min_block_size=10000, max_block_size=500000):
    available_mem = virtual_memory().available
    max_pixels_per_block = int(available_mem * safety_factor / bytes_per_pixel)
    return clamp(max_pixels_per_block, min_block_size, min(max_block_size, 200000))


def match_timestamps(ts_list1, ts_list2):
    common_ts = list(set(ts_list1) & set(ts_list2))
    common_ts.sort(key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return common_ts


def doy_from_date(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.timetuple().tm_yday


def log_memory_usage(stage):
    mem = virtual_memory()
    used_gb = mem.used / 1024 / 1024 / 1024
    available_gb = mem.available / 1024 / 1024 / 1024
    print(f"[{datetime.now()}] Memory usage - {stage}:  used, {used_gb:.2f}GB,  available {available_gb:.2f}GB")


# -------------------------
# Core mapping and field-pixel selection functions
# -------------------------
def load_npy_mapping(mapping_path, template_tif_path):
    print(f"[{datetime.now()}] Loading precomputed NumPy mapping: {mapping_path}")

    with rasterio.open(template_tif_path) as src:
        tif_rows, tif_cols = src.height, src.width
        tif_transform = src.transform
        tif_crs = src.crs

    rowcol_to_h5idx = np.load(mapping_path, mmap_mode='r')
    if rowcol_to_h5idx.shape != (tif_rows, tif_cols):
        raise ValueError(
            f"NumPy mapping shape does not match the template raster!npy: {rowcol_to_h5idx.shape}, TIF: ({tif_rows}, {tif_cols})"
        )

    print(f"[{datetime.now()}] Scanning the maximum HDF5 index by blocks...")
    max_h5idx = -1
    block_rows = 256
    num_blocks = (tif_rows + block_rows - 1) // block_rows

    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Scan maximum HDF5 index", total=num_blocks):
        row_end = min(row_start + block_rows, tif_rows)
        mapping_block = rowcol_to_h5idx[row_start:row_end, :]
        block_max = mapping_block[mapping_block != -1].max() if (mapping_block != -1).any() else -1
        if block_max > max_h5idx:
            max_h5idx = block_max
        del mapping_block
        gc.collect()

    if max_h5idx == -1:
        raise ValueError("No valid HDF5 indices were found in the NumPy mapping.")
    print(f"[{datetime.now()}] Maximum HDF5 index: {max_h5idx:,}")

    print(f"[{datetime.now()}] Building the reverse mapping by blocks (int32)...")
    h5idx_to_row = np.full((max_h5idx + 1), -1, dtype=np.int32)
    h5idx_to_col = np.full((max_h5idx + 1), -1, dtype=np.int32)

    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Build reverse mapping", total=num_blocks):
        row_end = min(row_start + block_rows, tif_rows)
        mapping_block = rowcol_to_h5idx[row_start:row_end, :]
        valid_row, valid_col = np.where(mapping_block != -1)
        if len(valid_row) == 0:
            del mapping_block
            continue
        global_row = valid_row + row_start
        valid_h5idx = mapping_block[valid_row, valid_col].astype(np.int32)
        valid_mask = valid_h5idx <= max_h5idx
        h5idx_to_row[valid_h5idx[valid_mask]] = global_row[valid_mask]
        h5idx_to_col[valid_h5idx[valid_mask]] = valid_col[valid_mask]
        del mapping_block, valid_row, valid_col, global_row, valid_h5idx, valid_mask
        gc.collect()

    valid_map_count = (h5idx_to_row != -1).sum()
    print(
        f"[{datetime.now()}] Reverse mapping built; valid entries: {valid_map_count:,} (memory usage: {h5idx_to_row.nbytes / 1024 / 1024:.2f} MB)"
    )
    return rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_transform, tif_crs


def get_field_h5idx(field_data, field_nodata, rowcol_to_h5idx, h5idx_to_row, tif_rows, tif_cols):
    print(f"[{datetime.now()}] Selecting valid HDF5 indices inside field parcels...")
    field_h5idx = []
    block_rows = 256
    num_blocks = (tif_rows + block_rows - 1) // block_rows

    for row_start in tqdm(range(0, tif_rows, block_rows), desc="Select field-pixel HDF5 indices"):
        row_end = min(row_start + block_rows, tif_rows)
        field_block = field_data[row_start:row_end, :]
        mapping_block = rowcol_to_h5idx[row_start:row_end, :]

        valid_mask = (field_block != field_nodata) & (mapping_block != -1)
        if not valid_mask.any():
            del field_block, mapping_block, valid_mask
            continue

        block_h5idx = mapping_block[valid_mask].astype(np.int32)
        block_h5idx = block_h5idx[(block_h5idx >= 0) & (block_h5idx < len(h5idx_to_row))]
        if len(block_h5idx) > 0:
            field_h5idx.append(block_h5idx)

        del field_block, mapping_block, valid_mask, block_h5idx
        gc.collect()

    if not field_h5idx:
        raise ValueError("No valid pixels were found inside field parcels; processing stopped.")
    field_h5idx = np.unique(np.concatenate(field_h5idx, dtype=np.int32))  # ensure sorted order
    print(f"[{datetime.now()}] Number of valid field pixels: {len(field_h5idx):,}")
    return field_h5idx


def generate_batch_indices_from_array(h5_indices, batch_size=100000):
    print(f"[{datetime.now()}] Preparing pixel batches ({batch_size:,} pixels per batch)...")
    h5_indices_sorted = np.sort(h5_indices)
    batch_list = []
    batch_start_rel = [0]
    total_pixels = 0

    total = len(h5_indices_sorted)
    for start in tqdm(range(0, total, batch_size), desc="Create batches"):
        end = min(start + batch_size, total)
        batch = h5_indices_sorted[start:end]
        batch_list.append(batch)
        total_pixels += len(batch)
        batch_start_rel.append(total_pixels)

    batch_start_rel.pop()
    print(f"[{datetime.now()}] Batches ready: {len(batch_list)} batches; total pixels: {total_pixels:,}")
    return batch_list, batch_start_rel, total_pixels


# -------------------------
# Rule-based screening function for rapeseed-rice pixels
# -------------------------
def process_rape_rice(vh_arr, timestamps):
    """Return 1 for a selected positive seed and 0 for an unselected pixel.

    The selected VH peak must occur during DOY 85-160 and exceed 0.06.
    A VH local minimum must occur during DOY 20-90 before that peak.
    """
    try:
        n = len(vh_arr)
        if n == 0 or len(timestamps) != n:
            return 0
        if n <= 12:
            return 0

        doy_list = [doy_from_date(ts) for ts in timestamps]
        doy_arr = np.asarray(doy_list)

        vh_local_maxima = argrelextrema(vh_arr, np.greater)[0]
        if len(vh_local_maxima) == 0:
            return 0
        valid_maxima = vh_local_maxima[
            (vh_local_maxima >= 6) & (vh_local_maxima <= n - 7)
        ]
        if len(valid_maxima) == 0:
            return 0

        max_idx = valid_maxima[np.argmax(vh_arr[valid_maxima])]
        max_val = vh_arr[max_idx]
        max_doy = doy_list[max_idx]
        if not (85 <= max_doy <= 160) or not (max_val > 0.06):
            return 0

        # Positive seeds require a VH local minimum during DOY 20-90,
        # before the candidate podding peak.
        pre_mask = (doy_arr >= 20) & (doy_arr <= 90) & (doy_arr < max_doy)
        pre_indices = np.flatnonzero(pre_mask)
        if len(pre_indices) == 0:
            return 0
        local_minima = argrelextrema(vh_arr, np.less)[0]
        pass_pre_minimum = bool(np.isin(local_minima, pre_indices).any())
        if not pass_pre_minimum:
            return 0

        post_min_candidates = argrelextrema(vh_arr[max_idx:], np.less)[0]
        if len(post_min_candidates) == 0:
            return 0
        post_min_val = vh_arr[max_idx + post_min_candidates[0]]
        if post_min_val >= 0.02:
            return 0

        sum_val = max_val + post_min_val
        if sum_val <= 0:
            return 0
        norm_diff = (max_val - post_min_val) / sum_val
        if norm_diff < 0.6:
            return 0

        return 1
    except Exception:
        return 0

def get_timestamps_from_merged(merged_h5_path):
    print(f"[{datetime.now()}] Reading timestamps from merged HDF5 file: {merged_h5_path}")
    with h5py.File(merged_h5_path, "r") as f:
        if "timestamps" not in f.attrs:
            raise ValueError(f"Merged HDF5 file is missing the 'timestamps' attribute; check the HDF5 preparation step.")
        timestamps = f.attrs["timestamps"].tolist()
        try:
            [datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format (expected %Y-%m-%d): {e}")
        if "value" not in f:
            raise ValueError(f"Merged HDF5 file is missing the 'value' dataset.")
        total_pixels = f["value"].shape[0]
        time_count = f["value"].shape[1]
        if time_count != len(timestamps):
            raise ValueError(f"The number of time points in the value dataset ({time_count} ) does not match the number of timestamps ({len(timestamps)} )")
    print(f"[{datetime.now()}] Timestamps in merged file: {len(timestamps)}; examples: {timestamps[:3]}...{timestamps[-3:]}")
    return timestamps, total_pixels


def read_h5_data_from_merged(merged_h5_path, indices, dataset_name="value"):
    if len(indices) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # 1. Clean indices: remove duplicates and sort in ascending order
    unique_indices = np.unique(indices)
    sorted_indices_values = np.sort(unique_indices)

    # 2. Record sorted positions to restore the original order
    sorted_pos = np.searchsorted(sorted_indices_values, indices)
    valid_pos = sorted_pos < len(sorted_indices_values)
    reversed_indices = np.where(valid_pos)[0]

    sub_batch_size = 10000
    num_sub_batches = (len(sorted_indices_values) + sub_batch_size - 1) // sub_batch_size
    data_list = []

    with h5py.File(merged_h5_path, "r") as f:
        dataset = f[dataset_name]
        max_dataset_idx = dataset.shape[0] - 1
        if len(sorted_indices_values) > 0 and sorted_indices_values[-1] > max_dataset_idx:
            raise ValueError(
                f"Index exceeds the HDF5 dataset range. Maximum requested index={sorted_indices_values[-1]}, dataset size={max_dataset_idx + 1}")

        for i in range(num_sub_batches):
            start = i * sub_batch_size
            end = min(start + sub_batch_size, len(sorted_indices_values))
            sub_indices = sorted_indices_values[start:end]

            if not np.all(sub_indices[1:] > sub_indices[:-1]):
                problematic = [sub_indices[j] for j in range(1, len(sub_indices)) if
                               sub_indices[j] <= sub_indices[j - 1]]
                print(f"Warning: sub-batch {i} contains non-increasing indices. Example values: {problematic[:5]}")

            sub_data = dataset[sub_indices, :].astype(np.float32)
            data_list.append(sub_data)
            del sub_data, sub_indices
            gc.collect()

    if not data_list:
        return np.zeros((0, 0), dtype=np.float32)

    # Merge data and restore the original order
    data = np.vstack(data_list)
    time_dim = data.shape[1]
    result = np.zeros((len(indices), time_dim), dtype=np.float32)
    result[reversed_indices] = data[sorted_pos[valid_pos]]

    del data_list, sorted_indices_values, unique_indices, sorted_pos, valid_pos, reversed_indices, data
    gc.collect()
    return result


def write_field_results_to_tif(pix_results, field_h5idx, output_tif, output_meta, h5idx_to_row, h5idx_to_col, tif_rows,
                               tif_cols):
    print(f"\n[{datetime.now()}] Writing field-area results to GeoTIFF: {output_tif}")

    init_block_rows = 1024
    with rasterio.open(output_tif, 'w', **output_meta) as dst:
        for row_start in tqdm(range(0, tif_rows, init_block_rows), desc="Initialize output raster"):
            row_end = min(row_start + init_block_rows, tif_rows)
            window = Window(0, row_start, tif_cols, row_end - row_start)
            empty_block = np.zeros((window.height, window.width), dtype=np.uint8)
            dst.write(empty_block, 1, window=window)
            del empty_block

    write_batch_size = 50000
    total = len(field_h5idx)
    num_batches = (total + write_batch_size - 1) // write_batch_size

    with rasterio.open(output_tif, 'r+') as dst:
        for batch_idx in tqdm(range(num_batches), desc="GeoTIFF write progress", total=num_batches):
            start = batch_idx * write_batch_size
            end = min(start + write_batch_size, total)

            batch_h5 = field_h5idx[start:end]
            batch_res = pix_results[start:end]

            r = h5idx_to_row[batch_h5]
            c = h5idx_to_col[batch_h5]
            valid_rc = (r >= 0) & (r < tif_rows) & (c >= 0) & (c < tif_cols)
            if not valid_rc.any():
                del batch_h5, batch_res, r, c, valid_rc
                continue

            r_valid = r[valid_rc]
            c_valid = c[valid_rc]
            res_valid = batch_res[valid_rc]

            for row in np.unique(r_valid):
                row_mask = r_valid == row
                cols = c_valid[row_mask]
                res = res_valid[row_mask]
                window = Window(0, row, tif_cols, 1)
                tif_row = dst.read(1, window=window).flatten()
                tif_row[cols] = res
                dst.write(tif_row.reshape(1, tif_cols), 1, window=window)

            del batch_h5, batch_res, r, c, valid_rc, r_valid, c_valid, res_valid
            gc.collect()

    print(f"[{datetime.now()}] GeoTIFF writing complete: {output_tif}")


def build_4features(vh_data, vv_data):
    vh = np.clip(vh_data, 1e-6, 10)
    vv = np.clip(vv_data, 1e-6, 10)

    vh_vv = vh / vv
    vh_vv = np.clip(vh_vv, 1e-6, 10)

    vh_vv_sum = vh + vv
    vh_vv_sum = np.clip(vh_vv_sum, 1e-6, None)
    rvi = (4 * vh) / vh_vv_sum
    rvi = np.clip(rvi, 1e-6, 10)

    features = np.hstack([vh, vv, vh_vv, rvi])
    return features


def load_negative_samples(csv_path, template_tif_path, rowcol_to_h5idx):
    """Load negative-sample coordinates from CSV and map them to HDF5 indices"""
    print(f"[{datetime.now()}] Loading negative-sample CSV: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        raise ValueError("The CSV must contain longitude and latitude columns.")

    # Read coordinate-transform information from the template raster
    with rasterio.open(template_tif_path) as src:
        transform = src.transform
        tif_rows, tif_cols = src.height, src.width

    # Convert longitude and latitude to raster row and column indices
    negative_h5idx = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Map negative-sample coordinates"):
        lon, lat = row['longitude'], row['latitude']
        # Convert coordinates to row and column indices
        r, c = rowcol(transform, lon, lat)
        # Check whether the pixel is within the valid raster extent
        if 0 <= r < tif_rows and 0 <= c < tif_cols:
            h5idx = rowcol_to_h5idx[r, c]
            if h5idx != -1:  # ensure that the HDF5 index is valid
                negative_h5idx.append(h5idx)

    # Remove duplicates
    negative_h5idx = np.unique(negative_h5idx)
    print(f"[{datetime.now()}] Negative-sample mapping complete; valid samples: {len(negative_h5idx)}")
    return negative_h5idx


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Overall accuracy (OA)
    oa = accuracy_score(y_true, y_pred)

    # Producer accuracy (PA) and user accuracy (UA)
    pa = np.diag(cm) / np.sum(cm, axis=1)  # row sums (reference classes)
    ua = np.diag(cm) / np.sum(cm, axis=0)  # column sums (predicted classes)

    # F1 score
    f1 = f1_score(y_true, y_pred, average=None)

    # Kappa coefficient
    kappa = cohen_kappa_score(y_true, y_pred)

    return oa, pa, ua, f1, kappa


def main():
    # Replace each placeholder with the corresponding local file path.
    vh_merged_path = r"<PATH_TO_MERGED_WHITTAKER_VH_H5>"
    vv_merged_path = r"<PATH_TO_MERGED_WHITTAKER_VV_H5>"
    template_tif = r"<PATH_TO_TEMPLATE_SENTINEL1_TIF>"
    output_directory = r"<PATH_TO_OUTPUT_DIRECTORY>"
    output_rule_tif = os.path.join(output_directory, "initial_rule_classification.tif")
    output_rf_tif = os.path.join(output_directory, "initial_rf_classification.tif")
    field_tif = r"<PATH_TO_FIELD_PARCEL_ID_RASTER_TIF>"
    rf_model_path = os.path.join(output_directory, "initial_random_forest.joblib")
    npy_mapping_path = r"<PATH_TO_PIXEL_TO_H5_INDEX_MAPPING_NPY>"
    negative_sample_csv = r"<PATH_TO_NON_AGRICULTURAL_NEGATIVE_SAMPLE_COORDINATES_CSV>"
    positive_sample_upper_limit = 50000
    os.makedirs(output_directory, exist_ok=True)

    start_total = datetime.now()
    print(f"[{datetime.now()}] Starting the workflow (CSV negative samples and updated rapeseed-rice rules)...")
    log_memory_usage("at startup")

    # 2. Load the pixel-to-HDF5 mapping.
    rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, tif_rows, tif_cols, tif_transform, tif_crs = load_npy_mapping(
        mapping_path=npy_mapping_path,
        template_tif_path=template_tif
    )
    log_memory_usage("after loading the mapping")

    # 3. Read field-parcel raster
    print(f"[{datetime.now()}] Reading the field-parcel raster...")
    with rasterio.open(field_tif, sharing=True) as field_src:
        if (field_src.height, field_src.width) != (tif_rows, tif_cols):
            raise ValueError("The field-parcel raster dimensions do not match the template raster.")
        field_data = field_src.read(1, out_dtype=np.int32, window=Window(0, 0, tif_cols, tif_rows))
        field_nodata = field_src.nodata
    log_memory_usage("after reading the field-parcel raster")

    # 4. Select pixels inside field parcels
    field_h5idx = get_field_h5idx(
        field_data=field_data,
        field_nodata=field_nodata,
        rowcol_to_h5idx=rowcol_to_h5idx,
        h5idx_to_row=h5idx_to_row,
        tif_rows=tif_rows,
        tif_cols=tif_cols
    )
    log_memory_usage("after selecting field pixels")

    # 5. Load timestamps
    vh_timestamps, vh_total_pixels = get_timestamps_from_merged(vh_merged_path)
    vv_timestamps, vv_total_pixels = get_timestamps_from_merged(vv_merged_path)
    common_timestamps = match_timestamps(vh_timestamps, vv_timestamps)
    if len(common_timestamps) < 13:
        print(f"Error: fewer than 13 common timestamps (found {len(common_timestamps)}).")
        return
    if vh_total_pixels != vv_total_pixels:
        raise ValueError(f"VH merged-file pixel count ({vh_total_pixels}) does not match VV merged-file pixel count ({vv_total_pixels}).")
    print(
        f"[{datetime.now()}] Common timestamps: {len(common_timestamps)} | examples: "
        f"{common_timestamps[:3]}...{common_timestamps[-3:]}"
    )
    print(
        f"[{datetime.now()}] Feature dimensions: four feature groups x "
        f"{len(common_timestamps)} timestamps = {4 * len(common_timestamps)}"
    )
    log_memory_usage("after loading timestamps")

    # 6. Create batches
    batch_size = get_unified_block_size(bytes_per_pixel=4 * len(common_timestamps) * 4)
    print(f"[{datetime.now()}] Calculated batch size: {batch_size}")
    batch_list, batch_start_rel, total_pixels = generate_batch_indices_from_array(
        h5_indices=field_h5idx,
        batch_size=batch_size
    )
    log_memory_usage("after creating batches")

    # 7. Apply rule-based screening.
    print(f"\n[{datetime.now()}] Starting rule screening (updated rapeseed-rice rules; {len(batch_list)} batches; field pixels only)...")
    pix_to_rule_path = os.path.join(output_directory, "rule_pixel_predictions.dat")
    pix_to_rule = np.memmap(pix_to_rule_path, dtype=np.uint8, mode="w+", shape=(total_pixels,))
    pix_to_rule[:] = 0
    pix_to_rule.flush()

    current_idx = 0
    for batch_idx, batch in enumerate(tqdm(batch_list, desc="Rule screening", total=len(batch_list))):
        if batch_idx % 10 == 0:
            gc.collect()
            log_memory_usage(f"before rule-screening batch {batch_idx}")

        batch_size = len(batch)
        vh_block = read_h5_data_from_merged(vh_merged_path, batch)
        rule_res = np.fromiter(
            (process_rape_rice(vh_arr, common_timestamps) for vh_arr in vh_block),
            dtype=np.uint8,
            count=batch_size,
        )

        pix_to_rule[current_idx:current_idx + batch_size] = rule_res
        pix_to_rule.flush()

        current_idx += batch_size
        del vh_block, rule_res, batch
        gc.collect()

    rotation_count = (pix_to_rule == 1).sum()
    print(
        f"[{datetime.now()}] Rule screening complete; positive seeds inside parcels={rotation_count:,}, unselected pixels={total_pixels - rotation_count:,}")
    log_memory_usage("after rule screening")

    # 8. Write rule-based classification raster
    rule_output_meta = {
        "driver": "GTiff",
        "height": tif_rows,
        "width": tif_cols,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": tif_crs,
        "transform": tif_transform,
        "nodata": 0
    }
    write_field_results_to_tif(
        pix_results=pix_to_rule,
        field_h5idx=field_h5idx,
        output_tif=output_rule_tif,
        output_meta=rule_output_meta,
        h5idx_to_row=h5idx_to_row,
        h5idx_to_col=h5idx_to_col,
        tif_rows=tif_rows,
        tif_cols=tif_cols
    )
    log_memory_usage("after writing the rule-based raster")

    # 9. Train random forest (CSV negative samples)
    print(f"\n[{datetime.now()}] Preparing random-forest training with CSV negative samples...")

    # 9.1 Get positive samples (pixels selected by the rule)
    pos_h5idx = field_h5idx[pix_to_rule == 1]
    pos_count = len(pos_h5idx)
    if pos_count == 0:
        print("Warning: no valid positive samples; training cannot proceed.")
        return

    # 9.2 Load negative samples from CSV
    neg_h5idx = load_negative_samples(negative_sample_csv, template_tif, rowcol_to_h5idx)
    neg_count = len(neg_h5idx)
    if neg_count == 0:
        print("Warning: no valid negative samples; training cannot proceed.")
        return

    print(f"[{datetime.now()}] Sample counts: positive={pos_count:,}, negative={neg_count:,}")

    # 9.3 Balance the class sample counts
    sample_size = min(pos_count, neg_count, positive_sample_upper_limit)
    print(f"[{datetime.now()}] Training and validation samples: positive={sample_size:,}, negative={sample_size:,}, total={2 * sample_size:,}")

    # 9.4 Randomly select samples
    rng = np.random.RandomState(42)
    pos_selected = rng.choice(pos_h5idx, size=sample_size, replace=False)
    neg_selected = rng.choice(neg_h5idx, size=sample_size, replace=False)

    # 9.5 Combine samples and split them into training (60%) and validation (40%) sets
    all_samples = np.concatenate([pos_selected, neg_selected], dtype=np.int32)
    all_labels = np.array([1] * sample_size + [0] * sample_size, dtype=np.uint8)

    # Split into training and validation sets
    X_train_indices, X_val_indices, y_train, y_val = train_test_split(
        all_samples, all_labels,
        test_size=0.4,  # 40% for validation
        random_state=42,
        stratify=all_labels  # preserve class proportions
    )

    print(f"[{datetime.now()}] Data split: training={len(X_train_indices):,}, validation={len(X_val_indices):,}")

    # 9.6 Build training features
    print(f"[{datetime.now()}] Building training features (VH, VV, VH/VV, and RVI)...")
    X_train = []
    train_block_size = get_unified_block_size(bytes_per_pixel=4 * len(common_timestamps) * 4)
    num_train_blocks = (len(X_train_indices) + train_block_size - 1) // train_block_size
    for block_idx in tqdm(range(num_train_blocks), desc="Build training features"):
        block_start = block_idx * train_block_size
        block_end = min(block_start + train_block_size, len(X_train_indices))
        batch_h5idx = X_train_indices[block_start:block_end]

        vh_batch = read_h5_data_from_merged(vh_merged_path, batch_h5idx)
        vv_batch = read_h5_data_from_merged(vv_merged_path, batch_h5idx)
        X_batch = build_4features(vh_batch, vv_batch)

        X_train.append(X_batch)
        del vh_batch, vv_batch, X_batch, batch_h5idx
        gc.collect()
    X_train = np.vstack(X_train)
    print(f"[{datetime.now()}] Training feature shape: {X_train.shape}")

    # 9.7 Build validation features
    print(f"[{datetime.now()}] Building validation features...")
    X_val = []
    val_block_size = get_unified_block_size(bytes_per_pixel=4 * len(common_timestamps) * 4)
    num_val_blocks = (len(X_val_indices) + val_block_size - 1) // val_block_size
    for block_idx in tqdm(range(num_val_blocks), desc="Build validation features"):
        block_start = block_idx * val_block_size
        block_end = min(block_start + val_block_size, len(X_val_indices))
        batch_h5idx = X_val_indices[block_start:block_end]

        vh_batch = read_h5_data_from_merged(vh_merged_path, batch_h5idx)
        vv_batch = read_h5_data_from_merged(vv_merged_path, batch_h5idx)
        X_batch = build_4features(vh_batch, vv_batch)

        X_val.append(X_batch)
        del vh_batch, vv_batch, X_batch, batch_h5idx
        gc.collect()
    X_val = np.vstack(X_val)
    print(f"[{datetime.now()}] Validation feature shape: {X_val.shape}")

    # 9.8 Train the model
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, oob_score=True, random_state=42, n_jobs=-1
    )
    print(f"[{datetime.now()}] Training random forest...")
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, rf_model_path)
    print(f"[{datetime.now()}] Model training complete. OOB score: {rf_model.oob_score_:.4f}")

    # 9.9 Evaluate the model
    print(f"\n[{datetime.now()}] Model accuracy on the validation set:")
    y_pred = rf_model.predict(X_val)

    # Calculate evaluation metrics
    oa, pa, ua, f1, kappa = calculate_metrics(y_val, y_pred)

    # Print evaluation results
    print(f"Overall accuracy (OA): {oa:.4f}")
    print(f"Producer accuracy (PA) - target: {pa[1]:.4f}, non-target: {pa[0]:.4f}")
    print(f"User accuracy (UA) - target: {ua[1]:.4f}, non-target: {ua[0]:.4f}")
    print(f"F1 score - target: {f1[1]:.4f}, non-target: {f1[0]:.4f}")
    print(f"Kappa coefficient:  {kappa:.4f}")

    del X_train, X_val, y_train, y_val, y_pred, all_samples, all_labels
    del X_train_indices, X_val_indices, pos_selected, neg_selected, pos_h5idx, neg_h5idx
    gc.collect()
    log_memory_usage("after RF training and validation")

    # 10. Predict pixels not already selected by the rule.
    print(f"\n[{datetime.now()}] Starting RF prediction for {len(batch_list)} batches...")
    pix_to_rf_path = os.path.join(output_directory, "rf_pixel_predictions.dat")
    pix_to_rf = np.memmap(pix_to_rf_path, dtype=np.uint8, mode="w+", shape=(total_pixels,))
    pix_to_rf[:] = 0

    current_idx = 0
    for batch in tqdm(batch_list, desc="RF prediction", total=len(batch_list)):
        batch_size = len(batch)
        current_rule_res = pix_to_rule[current_idx:current_idx + batch_size]
        pix_to_rf[current_idx:current_idx + batch_size] = current_rule_res

        # Predict every field pixel not selected as a positive seed.
        pred_mask = current_rule_res == 0
        if pred_mask.any():
            pred_h5idx = batch[pred_mask]
            pred_rf_idx = current_idx + np.where(pred_mask)[0]

            vh_pred = read_h5_data_from_merged(vh_merged_path, pred_h5idx)
            vv_pred = read_h5_data_from_merged(vv_merged_path, pred_h5idx)
            X_pred = build_4features(vh_pred, vv_pred)

            y_pred = rf_model.predict(X_pred).astype(np.uint8)
            pix_to_rf[pred_rf_idx] = y_pred
            pix_to_rf.flush()

            del vh_pred, vv_pred, X_pred, y_pred
        current_idx += batch_size
        del batch, batch_size, current_rule_res, pred_mask
        gc.collect()

    log_memory_usage("RF prediction completed")

    # 11. Write the RF classification raster.
    rf_output_meta = {
        "driver": "GTiff",
        "height": tif_rows,
        "width": tif_cols,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": tif_crs,
        "transform": tif_transform,
        "nodata": 0
    }
    write_field_results_to_tif(
        pix_results=pix_to_rf,
        field_h5idx=field_h5idx,
        output_tif=output_rf_tif,
        output_meta=rf_output_meta,
        h5idx_to_row=h5idx_to_row,
        h5idx_to_col=h5idx_to_col,
        tif_rows=tif_rows,
        tif_cols=tif_cols
    )

    # 12. Remove temporary files
    print(f"\n[{datetime.now()}] Total runtime: {datetime.now() - start_total}")
    del pix_to_rule, pix_to_rf, field_h5idx, batch_list, batch_start_rel
    del rowcol_to_h5idx, h5idx_to_row, h5idx_to_col, field_data
    gc.collect()
    for path in [pix_to_rule_path, pix_to_rf_path]:
        if os.path.exists(path):
            os.unlink(path)
    print(f"[{datetime.now()}] Processing finished.")


if __name__ == "__main__":
    main()

