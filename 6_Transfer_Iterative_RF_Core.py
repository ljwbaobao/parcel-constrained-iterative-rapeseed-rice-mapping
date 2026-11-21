"""Supplement source-region pseudo-labels and export target-region majority-vote maps."""

import gc
import importlib
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

base = importlib.import_module("6_Transfer_Initial_RF")


# Construct the four time-series feature groups using the corrected RVI definition.

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


def parcel_proportions(predictions, parcel_ids):
    ids, inverse, counts = np.unique(parcel_ids, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=(predictions == 1), minlength=len(ids))
    return ids, inverse, positives / counts


def interval_parcels(predictions, parcel_ids, interval):
    ids, _, ratios = parcel_proportions(predictions, parcel_ids)
    # Match the local iterative script: (0, .2], (.2, .4], ..., (.8, 1].
    lower = (interval - 1) / 5
    upper = interval / 5
    return ids[(ratios > lower) & (ratios <= upper)], ids[ratios == 0]


def majority_vote(predictions, parcel_ids):
    _, inverse, ratios = parcel_proportions(predictions, parcel_ids)
    return (ratios[inverse] >= 0.5).astype(np.uint8)


def fit_model(sites, positives, negative_pools):
    count = sum(len(v) for v in positives.values())
    groups = np.concatenate([np.full(len(negative_pools[key]), i, dtype=np.uint8)
                             for i, key in enumerate(base.SOURCE_KEYS)])
    indices = np.concatenate([negative_pools[key] for key in base.SOURCE_KEYS])
    if len(indices) < count:
        raise ValueError(f"Insufficient zero-proportion negative pixels: {len(indices):,} available; {count:,} required.")
    chosen = np.random.RandomState(42).choice(len(indices), size=count, replace=False)
    x_parts, y_parts = [], []
    for i, key in enumerate(base.SOURCE_KEYS):
        positive = positives[key]
        negative = indices[chosen[groups[chosen] == i]]
        x_parts.extend([base.features_for(sites[key], positive, build_features),
                        base.features_for(sites[key], negative, build_features)])
        y_parts.extend([np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8)])
    x, y = shuffle(np.vstack(x_parts), np.concatenate(y_parts), random_state=42)
    model = RandomForestClassifier(**base.RF_PARAMS)
    model.fit(x, y)
    return model


def run_step(prefix, parent, sites, seeds):
    interval = int(prefix[-1])
    cumulative, negatives, positives = {}, {}, {}
    candidate_count = 0
    for key in base.SOURCE_KEYS:
        site = sites[key]
        raw = parent["predictions"][key]
        selected, zero = interval_parcels(raw, site.parcel_ids, interval)
        added = site.field_indices[np.isin(site.parcel_ids, selected) & (raw == 0)]
        candidate_count += len(added)
        cumulative[key] = np.union1d(parent["cumulative"][key], added)
        positives[key] = np.union1d(seeds[key], cumulative[key])
        negatives[key] = site.field_indices[np.isin(site.parcel_ids, zero)]
    new_count = sum(len(cumulative[key]) - len(parent["cumulative"][key]) for key in base.SOURCE_KEYS)
    # Match the local workflow's skip when the selected interval yields no candidate pixels.
    if candidate_count == 0:
        print(f"Skipping strategy {prefix}: no positive supplementation candidates.")
        return None
    print(f"Strategy {prefix}: {new_count:,} previously unused positive pixels added across source regions")
    try:
        model = fit_model(sites, positives, negatives)
    except ValueError as error:
        print(f"Skipping strategy {prefix}: {error}")
        return None
    source_raw = {key: base.predict_site(model, sites[key], build_features) for key in base.SOURCE_KEYS}
    target = sites[base.TARGET_KEY]
    target_raw = base.predict_site(model, target, build_features)
    target_final = majority_vote(target_raw, target.parcel_ids)
    base.write_tif(target, target_final, base.OUTPUT_DIR / f"Site1_strategy_{prefix}_iter_{len(prefix)}.tif")
    del model, target_raw, target_final
    gc.collect()
    return {"predictions": source_raw, "cumulative": cumulative}


def run():
    base.validate_configuration()
    sites = {key: base.load_site(key) for key in (*base.SOURCE_KEYS, base.TARGET_KEY)}
    initial, seeds = {}, {}
    for key, site in sites.items():
        path = base.cache_directory() / f"{key}_initial_state.npz"
        if not path.exists():
            raise FileNotFoundError(f"Run 6_Transfer_Initial_RF.py first; missing cache: {path}")
        with np.load(path) as saved:
            if not np.array_equal(saved["field_indices"], site.field_indices):
                raise ValueError(f"Initial cache and current field-pixel mapping differ for {key}.")
            initial[key] = saved["predictions"].copy()
            if key in base.SOURCE_KEYS:
                seeds[key] = saved["positive_seeds"].copy()
    target = sites[base.TARGET_KEY]
    base.write_tif(target, majority_vote(initial[base.TARGET_KEY], target.parcel_ids),
                   base.OUTPUT_DIR / "Site1_initial_RF.tif")
    states = {"": {"predictions": {key: initial[key] for key in base.SOURCE_KEYS},
                    "cumulative": {key: np.array([], dtype=np.int64) for key in base.SOURCE_KEYS}}}
    for iteration in range(1, 4):
        next_states = {}
        for combination in product("12345", repeat=iteration):
            prefix = "".join(combination)
            parent = states.get(prefix[:-1])
            if parent is None:
                continue
            result = run_step(prefix, parent, sites, seeds)
            if result is not None and iteration < 3:
                next_states[prefix] = result
        states = next_states
        gc.collect()

