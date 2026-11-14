"""Train a joint source-region RF and classify the source and target regions."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm


OUTPUT_DIR = Path("<PATH_TO_TRANSFER_CLASSIFICATION_OUTPUT_DIRECTORY>")
SOURCE_KEYS = ("site2", "site3")
TARGET_KEY = "site1"
TIME_POINTS = 29
BATCH_SIZE = 10000
INITIAL_SAMPLE_CAP_PER_CLASS_PER_SOURCE = 50000
RF_PARAMS = dict(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

SITE_CONFIGS = {
    key: {
        "vh_h5": f"<PATH_TO_{key.upper()}_29_TIME_POINT_VH_HDF5>",
        "vv_h5": f"<PATH_TO_{key.upper()}_29_TIME_POINT_VV_HDF5>",
        "mapping_npy": f"<PATH_TO_{key.upper()}_RASTER_ROW_COLUMN_TO_HDF5_INDEX_NPY>",
        "template_tif": f"<PATH_TO_{key.upper()}_SAR_GRID_TEMPLATE_TIFF>",
        "parcel_tif": f"<PATH_TO_{key.upper()}_UNLABELLED_PARCEL_ID_TIFF>",
    }
    for key in (*SOURCE_KEYS, TARGET_KEY)
}
for _key in SOURCE_KEYS:
    SITE_CONFIGS[_key]["negative_csv"] = (
        f"<PATH_TO_{_key.upper()}_NON_AGRICULTURAL_NEGATIVE_PIXEL_COORDINATES_CSV>"
    )


@dataclass
class SiteData:
    key: str
    config: dict
    dates: list
    mapping: np.ndarray
    inverse_row: np.ndarray
    inverse_col: np.ndarray
    field_indices: np.ndarray
    parcel_ids: np.ndarray
    rows: int
    cols: int
    transform: object
    crs: object


def validate_configuration():
    values = [str(OUTPUT_DIR)]
    values.extend(str(value) for cfg in SITE_CONFIGS.values() for value in cfg.values())
    if any("<PATH_TO_" in value for value in values):
        raise ValueError("Replace the descriptive path placeholders in 6_Transfer_Initial_RF.py before running.")


def inspect_h5(path):
    with h5py.File(path, "r") as handle:
        raw_dates = handle.attrs["timestamps"]
        dates = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw_dates]
        shape = handle["value"].shape
    if len(shape) != 2 or shape[1] != TIME_POINTS or len(dates) != TIME_POINTS:
        raise ValueError(f"Expected a pixel-by-{TIME_POINTS} HDF5 matrix: {path}")
    parsed = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    if parsed != sorted(parsed) or len(set(dates)) != TIME_POINTS:
        raise ValueError(f"HDF5 timestamps must be unique and chronologically ordered: {path}")
    return dates, shape[0]


def load_site(key):
    cfg = SITE_CONFIGS[key]
    dates, pixel_count = inspect_h5(cfg["vh_h5"])
    vv_dates, vv_count = inspect_h5(cfg["vv_h5"])
    if dates != vv_dates or pixel_count != vv_count:
        raise ValueError(f"VV and VH must use matching pixel indices and timestamps within {key}.")
    with rasterio.open(cfg["template_tif"]) as src:
        rows, cols, transform, crs = src.height, src.width, src.transform, src.crs
    mapping = np.load(cfg["mapping_npy"], mmap_mode="r")
    if mapping.shape != (rows, cols):
        raise ValueError(f"Mapping and SAR template dimensions do not match for {key}.")
    inverse_row = np.full(pixel_count, -1, dtype=np.int32)
    inverse_col = np.full(pixel_count, -1, dtype=np.int32)
    for start in range(0, rows, 512):
        block = np.asarray(mapping[start:start + 512])
        rr, cc = np.where(block >= 0)
        indices = block[rr, cc].astype(np.int64)
        if np.any(indices >= pixel_count):
            raise ValueError(f"The mapping contains indices outside the HDF5 matrix for {key}.")
        inverse_row[indices] = rr + start
        inverse_col[indices] = cc
    with rasterio.open(cfg["parcel_tif"]) as src:
        if src.shape != (rows, cols) or src.transform != transform or src.crs != crs:
            raise ValueError(f"Parcel IDs must use the SAR template grid for {key}.")
        field = src.read(1)
        valid = (field > 0) & src.read_masks(1).astype(bool) & (mapping >= 0)
        field_indices = np.unique(np.asarray(mapping[valid], dtype=np.int64))
    if not len(field_indices):
        raise ValueError(f"No valid field-parcel pixels were found for {key}.")
    parcel_ids = field[inverse_row[field_indices], inverse_col[field_indices]].astype(np.int32)
    print(f"Loaded {key}: {len(field_indices):,} field pixels, {len(np.unique(parcel_ids)):,} parcels, {TIME_POINTS} observations")
    return SiteData(key, cfg, dates, mapping, inverse_row, inverse_col,
                    field_indices, parcel_ids, rows, cols, transform, crs)


def read_h5(path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    output = np.empty((len(unique), TIME_POINTS), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        for start in range(0, len(unique), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(unique))
            output[start:stop] = handle["value"][unique[start:stop], :]
    return output[inverse]


# The rule function and initial feature builder below are copied from the local initial-RF script.

def doy_from_date(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.timetuple().tm_yday


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


def features_for(site, indices, builder):
    vh = read_h5(site.config["vh_h5"], indices)
    vv = read_h5(site.config["vv_h5"], indices)
    return builder(vh, vv).astype(np.float32, copy=False)


def screen_source_seeds(site):
    selected = []
    for start in tqdm(range(0, len(site.field_indices), BATCH_SIZE), desc=f"Screen {site.key} positive seeds"):
        indices = site.field_indices[start:start + BATCH_SIZE]
        vh = read_h5(site.config["vh_h5"], indices)
        keep = np.fromiter((process_rape_rice(series, site.dates) == 1 for series in vh),
                           dtype=bool, count=len(indices))
        selected.append(indices[keep])
    seeds = np.unique(np.concatenate(selected))
    if not len(seeds):
        raise ValueError(f"Rule screening selected no positive seeds in {site.key}.")
    return seeds


def load_non_agricultural_negatives(site):
    frame = pd.read_csv(site.config["negative_csv"])
    if not {"longitude", "latitude"}.issubset(frame.columns):
        raise ValueError("Negative-pixel CSV files must contain longitude and latitude columns.")
    rr, cc = rowcol(site.transform, frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    rr, cc = np.asarray(rr), np.asarray(cc)
    inside = (rr >= 0) & (rr < site.rows) & (cc >= 0) & (cc < site.cols)
    indices = np.asarray(site.mapping[rr[inside], cc[inside]], dtype=np.int64)
    indices = np.unique(indices[(indices >= 0) & (indices < len(site.inverse_row))])
    if not len(indices):
        raise ValueError(f"No valid non-agricultural negative pixels were found for {site.key}.")
    return indices


def predict_site(model, site, builder):
    predictions = np.empty(len(site.field_indices), dtype=np.uint8)
    for start in tqdm(range(0, len(predictions), BATCH_SIZE), desc=f"Classify {site.key}"):
        stop = min(start + BATCH_SIZE, len(predictions))
        predictions[start:stop] = model.predict(features_for(site, site.field_indices[start:stop], builder))
    return predictions


def write_tif(site, predictions, path):
    array = np.full((site.rows, site.cols), 255, dtype=np.uint8)
    array[site.inverse_row[site.field_indices], site.inverse_col[site.field_indices]] = predictions
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=site.rows, width=site.cols,
                       count=1, dtype="uint8", crs=site.crs, transform=site.transform,
                       nodata=255, compress="lzw") as dst:
        dst.write(array, 1)
    print(f"Wrote {path}")


def cache_directory():
    return OUTPUT_DIR / ".transfer_cache"


def evaluate_source_holdout(model, sites, groups, indices, labels, validation_rows):
    """Evaluate the joint model on source pixels withheld from initial fitting.

    Labels are inherited from rule-positive seeds and non-agricultural negatives.
    These diagnostics do not evaluate target-region map accuracy.
    """
    references, predictions = [], []
    for i, key in enumerate(SOURCE_KEYS):
        rows = validation_rows[groups[validation_rows] == i]
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            predictions.append(model.predict(features_for(sites[key], indices[batch], build_4features)))
            references.append(labels[batch])
    reference = np.concatenate(references)
    prediction = np.concatenate(predictions)
    tn = int(np.count_nonzero((reference == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((reference == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((reference == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((reference == 1) & (prediction == 1)))
    metrics = {
        "OA": (tn + tp) / len(reference),
        "UA": tp / (tp + fp) if tp + fp else 0.0,
        "PA": tp / (tp + fn) if tp + fn else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    print(f"Source-region 40% hold-out: {len(reference):,} pixels; seed/negative-pool labels")
    print(f"Confusion matrix (reference rows, prediction columns): [[{tn}, {fp}], [{fn}, {tp}]]")
    for name, value in metrics.items():
        print(f"Source hold-out {name}: {value:.4f}")
    return metrics


def main():
    validate_configuration()
    sites = {key: load_site(key) for key in (*SOURCE_KEYS, TARGET_KEY)}
    rng = np.random.RandomState(42)
    seeds, selected, selected_labels = {}, {}, {}
    for key in SOURCE_KEYS:
        site = sites[key]
        seeds[key] = screen_source_seeds(site)
        negative = load_non_agricultural_negatives(site)
        count = min(len(seeds[key]), len(negative), INITIAL_SAMPLE_CAP_PER_CLASS_PER_SOURCE)
        pos = rng.choice(seeds[key], count, replace=False)
        neg = rng.choice(negative, count, replace=False)
        selected[key] = np.concatenate([pos, neg])
        selected_labels[key] = np.concatenate([np.ones(count, dtype=np.uint8), np.zeros(count, dtype=np.uint8)])
        seed_map = np.isin(site.field_indices, seeds[key]).astype(np.uint8)
        write_tif(site, seed_map, OUTPUT_DIR / f"{key}_rule_positive_seeds.tif")

    # Use 60% of the balanced source pool for fitting and 40% for internal diagnostics.
    groups = np.concatenate([np.full(len(selected[key]), i, dtype=np.uint8)
                             for i, key in enumerate(SOURCE_KEYS)])
    indices = np.concatenate([selected[key] for key in SOURCE_KEYS])
    labels = np.concatenate([selected_labels[key] for key in SOURCE_KEYS])
    train_rows, validation_rows = train_test_split(np.arange(len(labels)), test_size=0.4,
                                      random_state=42, stratify=labels)
    x_parts, y_parts = [], []
    for i, key in enumerate(SOURCE_KEYS):
        rows = train_rows[groups[train_rows] == i]
        x_parts.append(features_for(sites[key], indices[rows], build_4features))
        y_parts.append(labels[rows])
    x, y = shuffle(np.vstack(x_parts), np.concatenate(y_parts), random_state=42)
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(x, y)
    del x, y, x_parts, y_parts
    evaluate_source_holdout(model, sites, groups, indices, labels, validation_rows)
    cache = cache_directory()
    cache.mkdir(parents=True, exist_ok=True)
    for key, site in sites.items():
        if key in SOURCE_KEYS:
            seed_mask = np.isin(site.field_indices, seeds[key])
            predictions = np.ones(len(site.field_indices), dtype=np.uint8)
            remaining = np.flatnonzero(~seed_mask)
            for start in tqdm(range(0, len(remaining), BATCH_SIZE), desc=f"Classify unselected {key} pixels"):
                positions = remaining[start:start + BATCH_SIZE]
                predictions[positions] = model.predict(features_for(site, site.field_indices[positions], build_4features))
        else:
            predictions = predict_site(model, site, build_4features)
        write_tif(site, predictions, OUTPUT_DIR / f"{key}_initial_RF_pixelwise.tif")
        np.savez_compressed(cache / f"{key}_initial_state.npz", field_indices=site.field_indices,
                            predictions=predictions, positive_seeds=seeds.get(key, np.array([], dtype=np.int64)))


if __name__ == "__main__":
    main()
