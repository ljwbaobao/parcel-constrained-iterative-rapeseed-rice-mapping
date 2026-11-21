"""Input configuration and shared functions for 13_Supervised Classification.py.

Replace the descriptive input paths and seasonal attribute field names.
Training labels and features are read from the existing prepared.npz and
features.npy cache files. No old training/test partition is used.
If vector attribute values use another language, replace the RAW_ATTRIBUTE
placeholders in ATTRIBUTE_VALUE_ALIASES with their exact recorded values.
The normalized categories below describe agricultural parcel records only.
"""

from pathlib import Path
import hashlib
import os

import geopandas as gpd
import h5py
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject


RF_PARAMS = dict(n_estimators=100, max_depth=10, max_features="sqrt", n_jobs=min(8, os.cpu_count() or 1))
NAMES = {1: "Rapeseed-rice", 2: "Wheat-rice", 3: "Double rice", 4: "Other agricultural patterns"}

# Keep the site keys consistent with the prepared cache subdirectories.
CONFIGS = {
    key: {
        "site": label,
        "field": f"<PATH_TO_{label.replace(' ', '_').upper()}_UNLABELLED_PARCEL_ID_TIFF>",
        "mapping": f"<PATH_TO_{label.replace(' ', '_').upper()}_RASTER_TO_HDF5_INDEX_NPY>",
        "vh": f"<PATH_TO_{label.replace(' ', '_').upper()}_SMOOTHED_VH_HDF5>",
        "vv": f"<PATH_TO_{label.replace(' ', '_').upper()}_SMOOTHED_VV_HDF5>",
        "parcels": f"<PATH_TO_{label.replace(' ', '_').upper()}_SEASONAL_CROP_LABELLED_PARCEL_VECTOR>",
        "spring": "<SPRING_CROP_ATTRIBUTE_FIELD_NAME>",
        "autumn": "<AUTUMN_CROP_ATTRIBUTE_FIELD_NAME>",
        "target": f"<PATH_TO_{label.replace(' ', '_').upper()}_RAPESEED_RICE_REFERENCE_VECTOR>",
        # This input is checked against the historical preparation manifest only.
        "best_map": f"<PATH_TO_{label.replace(' ', '_').upper()}_ORIGINAL_SELECTED_STRATEGY_MAP_IN_CACHE_MANIFEST>",
        "dce_map": f"<PATH_TO_{label.replace(' ', '_').upper()}_DCE_MAJORITY_VOTED_CLASSIFICATION_TIFF>",
    }
    for key, label in (("tongxiang", "Site 1"), ("haiyan", "Site 2"), ("yuhang", "Site 3"))
}

LABEL_RASTERS = {
    reference_key: {
        "Rapeseed-rice": f"<PATH_TO_{site_key}_RAPESEED_RICE_REFERENCE_TIFF>",
        "Wheat-rice": f"<PATH_TO_{site_key}_WHEAT_RICE_REFERENCE_TIFF>",
        "Double rice": None if reference_key == "Haiyan" else f"<PATH_TO_{site_key}_DOUBLE_RICE_REFERENCE_TIFF>",
    }
    for reference_key, site_key in (("Tongxiang", "SITE_1"), ("Haiyan", "SITE_2"), ("Yuhang", "SITE_3"))
}

# Multiple recorded values may map to the same English category.
# Copy an entry with a different key when the source attributes use synonyms.
ATTRIBUTE_VALUE_ALIASES = {
    "<RAW_ATTRIBUTE_VALUE_FOR_RAPESEED>": "rapeseed",
    "<RAW_ATTRIBUTE_VALUE_FOR_WHEAT_OR_BARLEY>": "wheat",
    "<RAW_ATTRIBUTE_VALUE_FOR_EARLY_RICE>": "early_rice",
    "<RAW_ATTRIBUTE_VALUE_FOR_LATE_OR_SINGLE_SEASON_RICE>": "rice",
    "<RAW_ATTRIBUTE_VALUE_FOR_OTHER_CROP>": "other_crop",
    "<RAW_ATTRIBUTE_VALUE_FOR_OTHER_SEASONAL_STATUS>": "other_seasonal_status",
    "<RAW_ATTRIBUTE_VALUE_FOR_GREENHOUSE>": "greenhouse",
    "<RAW_ATTRIBUTE_VALUE_FOR_LOTUS_POND>": "lotus_pond",
    "<RAW_ATTRIBUTE_VALUE_FOR_FALLOW>": "fallow",
    "<RAW_ATTRIBUTE_VALUE_FOR_RICE_SEEDLINGS>": "rice_seedlings",
    "<RAW_ATTRIBUTE_VALUE_FOR_OTHER>": "other",
    "<RAW_ATTRIBUTE_VALUE_WITH_UNKNOWN_CLASS>": "unknown",
}


def normalized_label(value):
    text = str(value).strip()
    if text in ATTRIBUTE_VALUE_ALIASES:
        return ATTRIBUTE_VALUE_ALIASES[text]
    if not text.isascii():
        raise ValueError(f"Add the recorded attribute value to ATTRIBUTE_VALUE_ALIASES: {text!r}")
    return text.lower().replace("-", "_").replace(" ", "_")


def seasonal_class(spring, autumn):
    spring, autumn = normalized_label(spring), normalized_label(autumn)
    known_spring_crops = {"rapeseed", "wheat", "early_rice", "other_crop"}
    if "other_seasonal_status" in {spring, autumn}:
        return 4
    if "greenhouse" in {spring, autumn}:
        return 4
    if autumn == "lotus_pond" and spring == "wheat":
        return 4
    if autumn == "rice":
        if spring == "rapeseed":
            return 1
        if spring == "wheat":
            return 2
        if spring == "early_rice":
            return 3
        if spring in {"other_crop", "fallow", "rice_seedlings", "other"}:
            return 4
    if autumn in {"other", "other_crop"} and spring in known_spring_crops:
        return 4
    return 0



def decode(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def load_grid(cfg):
    with rasterio.open(cfg["field"]) as src:
        field = src.read(1).astype(np.int32)
        profile = src.profile.copy()
        valid = (field > 0) & (src.read_masks(1) > 0)
    mapping = np.load(cfg["mapping"], mmap_mode="r")
    if mapping.shape != field.shape:
        raise ValueError("Field raster and HDF5 mapping dimensions differ")
    with h5py.File(cfg["vh"], "r") as src:
        dates = decode(src.attrs["timestamps"].tolist())
        h5_shape = src["value"].shape
    with h5py.File(cfg["vv"], "r") as src:
        if decode(src.attrs["timestamps"].tolist()) != dates or src["value"].shape != h5_shape:
            raise ValueError("Local VH and VV dimensions or observation dates differ")
    valid &= (mapping >= 0) & (mapping < h5_shape[0])
    flat = np.flatnonzero(valid.ravel())
    h5_indices = np.asarray(mapping).ravel()[flat].astype(np.int64)
    if len(np.unique(h5_indices)) != len(h5_indices):
        raise ValueError("Duplicate HDF5 indices map to different field pixels")
    ids = field.ravel()[flat]
    unique_ids, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    return field, profile, flat, h5_indices, unique_ids, inverse, sizes, dates


def proportional_sequence(capacities, n):
    """Create nested class allocations, covering each available class first."""
    keys = sorted(k for k, value in capacities.items() if value > 0)
    if sum(capacities.values()) < n:
        raise ValueError("Insufficient negative training parcels")
    if len(keys) > n:
        raise ValueError("Parcel budget cannot cover every negative crop class")
    counts = {k: 0 for k in keys}
    sequence = []
    for k in keys:
        sequence.append(k)
        counts[k] += 1
    weights = {k: capacities[k] / sum(capacities.values()) for k in keys}
    while len(sequence) < n:
        eligible = [k for k in keys if counts[k] < capacities[k]]
        best = max(eligible, key=lambda k: (weights[k] * (len(sequence) + 1) - counts[k], -k))
        sequence.append(best)
        counts[best] += 1
    return sequence


def sample_pixels(positions, group_values, n, rng):
    """Sample proportionally by group and retain each represented group."""
    positions = np.asarray(positions, dtype=np.int64)
    if n >= len(positions):
        return positions.copy()
    groups, inverse, sizes = np.unique(group_values[positions], return_inverse=True, return_counts=True)
    desired = sizes * n / sizes.sum()
    quotas = np.minimum(np.maximum(np.floor(desired).astype(int), 1), sizes)
    if n < len(groups):
        # This branch is used only for evaluation, never to omit training parcels.
        return np.sort(rng.choice(positions, n, replace=False))
    while quotas.sum() > n:
        eligible = np.flatnonzero(quotas > 1)
        idx = eligible[np.argmax((quotas - desired)[eligible])]
        quotas[idx] -= 1
    while quotas.sum() < n:
        eligible = np.flatnonzero(quotas < sizes)
        idx = eligible[np.argmax((desired - quotas)[eligible])]
        quotas[idx] += 1
    chunks = []
    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[0, np.cumsum(sizes)]
    for i, count in enumerate(quotas):
        pool = positions[order[boundaries[i]:boundaries[i + 1]]]
        chunks.append(rng.choice(pool, int(count), replace=False))
    return np.sort(np.concatenate(chunks))


def metric_values(predicted, y):
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    oa = (tp + tn) / len(y)
    ua = tp / (tp + fp) if tp + fp else 0.
    pa = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, OA=100 * oa, UA=100 * ua, PA=100 * pa, F1=100 * f1)


def warp_presence(path, meta):
    dst = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)
    if path:
        with rasterio.open(path) as src:
            present = ((src.read(1) > 0) & (src.read_masks(1) > 0)).astype(np.uint8)
            reproject(present, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=meta["transform"], dst_crs=meta["crs"],
                      resampling=Resampling.nearest, num_threads=2)
    return dst.astype(bool)


def reference_labels(cfg, reference_paths, meta):
    masks = [warp_presence(p, meta) for p in reference_paths.values()]
    memberships = sum(m.astype(np.uint8) for m in masks)
    conflicts = memberships > 1
    labels = np.zeros(masks[0].shape, np.uint8)
    # This order preserves the target definition used in the published PA.
    for code, mask in reversed(list(enumerate(masks, 1))):
        labels[mask] = code

    vectors = gpd.read_file(cfg["parcels"]).to_crs(meta["crs"])
    codes = np.array([seasonal_class(a, b) for a, b in
                      zip(vectors[cfg["spring"]], vectors[cfg["autumn"]])])
    extras = np.zeros(labels.shape, np.uint8)
    extra_memberships = np.zeros(labels.shape, np.uint8)
    for code in (2, 3, 4):
        geoms = [g for g in vectors.geometry[codes == code]
                 if g is not None and not g.is_empty]
        if not geoms:
            continue
        mask = rasterize(((g, 1) for g in geoms), out_shape=labels.shape,
                         transform=meta["transform"], fill=0,
                         all_touched=False, dtype="uint8").astype(bool)
        extra_memberships += mask
        extras[mask] = code
    extra_conflicts = extra_memberships > 1
    conflicts |= (labels == 0) & extra_conflicts
    supplement = (labels == 0) & ~extra_conflicts
    labels[supplement] = extras[supplement]
    return labels, conflicts, {
        "canonical_class_pixels": {NAMES[c]: int(m.sum()) for c, m in enumerate(masks, 1)},
        "canonical_reference_overlap_pixels": int((memberships > 1).sum()),
        "supplemented_reference_pixels": {NAMES[c]: int(((extras == c) & supplement).sum())
                                           for c in (2, 3, 4)},
        "unknown_policy": "No target labels inferred outside the canonical target raster; unlabelled pixels excluded",
        "other_policy": "Other non-target parcel records inside the supplied farmland mask, including seasonal status labels; no off-farm land cover inferred",
    }


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
