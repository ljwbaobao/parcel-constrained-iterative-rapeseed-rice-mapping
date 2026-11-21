"""Run RF10/RF20/RF50 without reserving local test parcels.

Validation uses every target-reference pixel with valid SAR features plus one
fixed, equally sized sample of labelled non-target farmland pixels per site.
Configure paths and reference-label aliases in 9_Supervised_Classification_Support.py.
The existing prepared.npz, features.npy, and design.json caches are required.
Its Site 1 validation pixels are identical to the Site 1 local sample.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier


HERE = Path(__file__).resolve().parent
OUT = Path("<PATH_TO_SUPERVISED_RF_COMPARISON_OUTPUT_DIRECTORY>")
CACHE = Path("<PATH_TO_PREPARED_PARCEL_LABEL_AND_FEATURE_CACHE_DIRECTORY>")
TRANSFER_DCE_MAP = Path("<PATH_TO_SITE_1_TRANSFER_DCE_MAJORITY_VOTED_CLASSIFICATION_TIFF>")
TIME_POINTS = 29
FEATURE_GROUPS = ("VH", "VV", "VH/VV", "RVI")
FEATURE_COUNT = TIME_POINTS * len(FEATURE_GROUPS)
EVALUATION_SEED = 20261121
REPEAT_SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [10, 20, 50]
METRICS = ("OA", "UA", "PA", "F1")


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


base = load_module("supervised_rf_support", HERE / "9_Supervised_Classification_Support.py")
full = base


def validate_configuration():
    values = [str(OUT), str(CACHE), str(TRANSFER_DCE_MAP)]
    values.extend(str(value) for cfg in base.CONFIGS.values() for value in cfg.values())
    values.extend(str(value) for cfg in base.LABEL_RASTERS.values() for value in cfg.values() if value is not None)
    if any("<PATH_TO_" in value or "<SPRING_" in value or "<AUTUMN_" in value for value in values):
        raise ValueError("Replace the descriptive paths and attribute field placeholders in both English scripts before running.")


def save_csv(file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(file, index=False, encoding="utf-8-sig")


def load_site(key):
    cfg = base.CONFIGS[key]
    folder = CACHE / key
    old_design = json.loads((folder / "design.json").read_text(encoding="utf-8"))
    for name, saved in old_design["inputs"].items():
        st = Path(cfg[name]).stat()
        assert st.st_size == saved["size"] and st.st_mtime_ns == saved["mtime_ns"], (key, name)
    field, profile, flat, h5_indices, ids, inverse, sizes, dates = base.load_grid(cfg)
    if len(dates) != TIME_POINTS:
        raise ValueError(f"{cfg['site']} must contain {TIME_POINTS} already aligned time points; found {len(dates)}.")
    with np.load(folder / "prepared.npz") as prepared:
        # Reuse reference eligibility, but not the old training/test partition.
        pixel_classes = prepared["pixel_classes"]
        parcel_classes = prepared["parcel_classes"]
        eligible = prepared["eligible"]
    x = np.load(folder / "features.npy", mmap_mode="r")
    if x.shape != (len(flat), FEATURE_COUNT):
        raise ValueError(f"{cfg['site']} feature cache must contain {FEATURE_COUNT} columns in the order VH, VV, VH/VV, RVI.")
    refs = base.LABEL_RASTERS
    refkey = {"tongxiang": "Tongxiang", "haiyan": "Haiyan", "yuhang": "Yuhang"}[key]
    labels, conflicts, diagnostics = full.reference_labels(cfg, refs[refkey], profile)
    classes = labels.ravel()[flat]
    assert np.all(classes > 0)
    positive = np.flatnonzero(classes == 1)
    negative_pool = np.flatnonzero(classes > 1)
    expected = {"tongxiang": 19860, "haiyan": 5326, "yuhang": 8022}[key]
    assert len(positive) == expected
    rng = np.random.default_rng(EVALUATION_SEED)
    negative = np.sort(rng.choice(negative_pool, len(positive), replace=False))
    validation_positions = np.r_[positive, negative]
    validation_y = np.r_[np.ones(len(positive), np.uint8), np.zeros(len(negative), np.uint8)]
    assert len(np.unique(validation_positions)) == len(validation_positions)
    dest = OUT / key
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "validation_pixels.npz", positions=validation_positions,
                        y=validation_y, flat_indices=flat[validation_positions],
                        reference_classes=classes[validation_positions])
    yy, xx = np.unravel_index(flat[validation_positions], field.shape)
    tr = profile["transform"]
    frame = pd.DataFrame(dict(Study_area=cfg["site"], Row=yy, Column=xx,
                              Parcel_ID=ids[inverse[validation_positions]],
                              HDF5_index=h5_indices[validation_positions],
                              Reference_class=[full.NAMES[int(c)] for c in classes[validation_positions]],
                              Binary_label=validation_y,
                              X=tr.c + (xx + .5) * tr.a, Y=tr.f + (yy + .5) * tr.e))
    frame.to_csv(dest / "balanced_validation_pixels.csv", index=False, encoding="utf-8-sig")
    print(cfg["site"], "validation", len(positive), "+", len(negative),
          "; eligible parcels", {i: int((parcel_classes == i).sum()) for i in (1, 2, 3, 4)}, flush=True)
    return dict(key=key, cfg=cfg, field=field, profile=profile, flat=flat,
                h5_indices=h5_indices, ids=ids, inverse=inverse, sizes=sizes,
                dates=dates, x=x, pixel_classes=pixel_classes,
                parcel_classes=parcel_classes, eligible=eligible,
                reference_classes=classes, conflicts=conflicts.ravel()[flat],
                validation_positions=validation_positions, validation_y=validation_y,
                reference_diagnostics=diagnostics, folder=dest)


def sample_training(state, seed, budget):
    classes = state["parcel_classes"]
    positive_pool = np.flatnonzero(classes == 1)
    negative_pools = {c: np.flatnonzero(classes == c) for c in (2, 3, 4)}
    negative_pools = {c: v for c, v in negative_pools.items() if len(v)}
    assert len(positive_pool) >= 25 and sum(map(len, negative_pools.values())) >= 25
    rng = np.random.default_rng(seed)
    positive_order = rng.permutation(positive_pool)[:25]
    orders = {c: rng.permutation(pool) for c, pool in negative_pools.items()}
    sequence = base.proportional_sequence({c: len(pool) for c, pool in negative_pools.items()}, 25)
    used = {c: 0 for c in negative_pools}
    negative_order = []
    for c in sequence:
        negative_order.append(orders[c][used[c]])
        used[c] += 1
    half = budget // 2
    parcels = np.r_[positive_order[:half], np.array(negative_order[:half])]
    included = np.isin(state["inverse"], parcels)
    pc = state["pixel_classes"]
    pos = np.flatnonzero(included & (pc == 1))
    neg = np.flatnonzero(included & (pc > 1))
    n = min(len(pos), len(neg), 50000)
    prng = np.random.default_rng(1000 + seed + budget)
    groups = state["ids"][state["inverse"]]
    pos = base.sample_pixels(pos, groups, n, prng)
    neg = base.sample_pixels(neg, groups, n, prng)
    positions = np.r_[pos, neg]
    y = np.r_[np.ones(n, np.uint8), np.zeros(n, np.uint8)]
    assert len(np.unique(groups[positions])) == budget
    assert np.all((state["reference_classes"][positions] == 1) == y)
    assert len(np.unique(positions)) == len(positions)
    return positions, y, state["ids"][parcels]


def predict_voted(model, state):
    counts = np.zeros(len(state["ids"]), np.int64)
    x = state["x"]
    for lo in range(0, len(x), 50000):
        hi = min(lo + 50000, len(x))
        raw = model.predict(x[lo:hi])
        counts += np.bincount(state["inverse"][lo:hi], weights=raw,
                              minlength=len(counts)).astype(np.int64)
    return (2 * counts >= state["sizes"]).astype(np.uint8)[state["inverse"]]


def save_map(path, state, values):
    out = np.full(state["field"].shape, 255, np.uint8)
    out.ravel()[state["flat"]] = values
    profile = dict(state["profile"], dtype="uint8", count=1, nodata=255, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def map_values(path, state):
    with rasterio.open(path) as src:
        assert src.shape == state["field"].shape and src.transform == state["profile"]["transform"]
        assert src.crs == state["profile"]["crs"]
        values = src.read(1).ravel()[state["flat"]]
    assert np.isin(values, (0, 1)).all()
    return values


def score(state, values):
    return base.metric_values(values[state["validation_positions"]], state["validation_y"])


def validate_time_series(states):
    for state in states.values():
        if len(state["dates"]) != TIME_POINTS or state["x"].shape[1] != FEATURE_COUNT:
            raise ValueError("All sites must have 29 corresponding observations and 116 feature columns.")


def sample_joint_training(states, seed, budget, folder):
    pools, pool_labels, pool_sources, pool_groups = [], [], [], []
    selected_parcels = {}
    half = budget // 2
    for source_index, key in enumerate(("haiyan", "yuhang")):
        state = states[key]
        classes = state["parcel_classes"]
        rng = np.random.default_rng(seed)
        positive_order = rng.permutation(np.flatnonzero(classes == 1))[:25]
        negative_pools = {c: np.flatnonzero(classes == c) for c in (2, 3, 4)}
        negative_pools = {c: v for c, v in negative_pools.items() if len(v)}
        negative_orders = {c: rng.permutation(v) for c, v in negative_pools.items()}
        sequence = base.proportional_sequence({c: len(v) for c, v in negative_pools.items()}, 25)
        taken = {c: 0 for c in negative_pools}
        negative_order = []
        for c in sequence:
            negative_order.append(negative_orders[c][taken[c]])
            taken[c] += 1
        # Equal total parcel budgets per source, with complementary class quotas.
        n_positive = half // 2 if source_index == 0 else half - half // 2
        n_negative = half - n_positive
        selected = np.r_[positive_order[:n_positive], np.array(negative_order[:n_negative])]
        selected_parcels[key] = state["ids"][selected]
        positions = np.flatnonzero(np.isin(state["inverse"], selected))
        y = (state["pixel_classes"][positions] == 1).astype(np.uint8)
        assert np.all((state["reference_classes"][positions] == 1) == y)
        pools.append(positions)
        pool_labels.append(y)
        pool_sources.append(np.full(len(positions), source_index, np.uint8))
        pool_groups.append(state["ids"][state["inverse"][positions]].astype(np.int64) + source_index * 1000000000)
    pool_positions = np.concatenate(pools)
    y = np.concatenate(pool_labels)
    source_codes = np.concatenate(pool_sources)
    groups = np.concatenate(pool_groups)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n = min(len(pos), len(neg), 50000)
    rng = np.random.default_rng(1000 + seed + budget)
    chosen = np.r_[base.sample_pixels(pos, groups, n, rng),
                   base.sample_pixels(neg, groups, n, rng)]
    assert len(np.unique(groups[chosen])) == budget
    x = np.empty((len(chosen), FEATURE_COUNT), np.float32)
    saved = {}
    for source_index, key in enumerate(("haiyan", "yuhang")):
        local_rows = np.flatnonzero(source_codes[chosen] == source_index)
        positions = pool_positions[chosen[local_rows]]
        x[local_rows] = states[key]["x"][positions]
        saved[key + "_positions"] = positions
        saved[key + "_y"] = y[chosen[local_rows]]
        saved[key + "_parcel_ids"] = selected_parcels[key]
    np.savez_compressed(folder / "source_training_pixels.npz", **saved)
    return x, y[chosen], {k: v.tolist() for k, v in selected_parcels.items()}


def main(transfer_only=False):
    validate_configuration()
    OUT.mkdir(parents=True, exist_ok=True)
    states = {k: load_site(k) for k in ("tongxiang", "haiyan", "yuhang")}
    validate_time_series(states)
    results, samples, audit = [], {}, []
    if transfer_only:
        results = pd.read_csv(OUT / "accuracy_by_run.csv").query("Study_area != 'Site 1*'").to_dict("records")
        previous = json.loads((OUT / "experiment_design.json").read_text(encoding="utf-8"))
        audit = [r for r in previous["Runs"] if r["Study_area"] != "Site 1*"]
    comparisons = {}
    for key, state in states.items():
        cfg = state["cfg"]
        path = Path(cfg["dce_map"])
        comparisons[cfg["site"]] = dict(Method="DCE", Classification_map=str(path), **score(state, map_values(path, state)))
        if transfer_only:
            continue
        for repetition, seed in enumerate(REPEAT_SEEDS, 1):
            for budget in BUDGETS:
                folder = state["folder"] / ("RF" + str(budget)) / ("repeat_" + str(repetition))
                folder.mkdir(parents=True, exist_ok=True)
                pos, y, parcels = sample_training(state, seed, budget)
                np.savez_compressed(folder / "training_pixels.npz", positions=pos, y=y,
                                    parcel_ids=parcels, h5_indices=state["h5_indices"][pos])
                samples[(key, budget, repetition)] = (pos, y)
                start = time.monotonic()
                model = RandomForestClassifier(**base.RF_PARAMS, random_state=seed)
                model.fit(state["x"][pos], y)
                values = predict_voted(model, state)
                path = folder / "majority_voted_classification.tif"
                save_map(path, state, values)
                overlap = int(np.intersect1d(pos, state["validation_positions"]).size)
                metrics = score(state, values)
                results.append(dict(Study_area=cfg["site"], Method="RF" + str(budget),
                                    Parcels_per_training_site=budget, Total_training_parcels=budget,
                                    Repeat=repetition, Seed=seed, Training_pixels_per_class=len(y) // 2,
                                    Validation_pixels_per_class=len(state["validation_y"]) // 2,
                                    Training_validation_overlap_pixels=overlap,
                                    **metrics, **{"DCE_" + m: comparisons[cfg["site"]][m] for m in METRICS},
                                    Classification_map=str(path)))
                audit.append(dict(Study_area=cfg["site"], Budget=budget, Repeat=repetition,
                                  Actual_feature_count=state["x"].shape[1], Training_parcel_ids=parcels.tolist(),
                                  Training_manifest=str(folder / "training_pixels.npz"),
                                  Seconds=time.monotonic() - start))
                print(cfg["site"], "RF" + str(budget), "repeat", repetition,
                      "OA", round(metrics["OA"], 4), "overlap", overlap, flush=True)
    target = states["tongxiang"]
    comparisons["Site 1*"] = dict(Method="DCE", Classification_map=str(TRANSFER_DCE_MAP), **score(target, map_values(TRANSFER_DCE_MAP, target)))
    for repetition, seed in enumerate(REPEAT_SEEDS, 1):
        for budget in BUDGETS:
            folder = OUT / "transfer" / ("RF" + str(budget) + "_total") / ("repeat_" + str(repetition))
            folder.mkdir(parents=True, exist_ok=True)
            x, y, selected_sources = sample_joint_training(states, seed, budget, folder)
            assert x.shape[1] == FEATURE_COUNT and len(y) == 2 * int(y.sum())
            start = time.monotonic()
            model = RandomForestClassifier(**base.RF_PARAMS, random_state=seed)
            model.fit(x, y)
            values = predict_voted(model, target)
            path = folder / "Site1_majority_voted_classification.tif"
            save_map(path, target, values)
            metrics = score(target, values)
            results.append(dict(Study_area="Site 1*", Method="RF" + str(budget),
                                Parcels_per_training_site=budget // 2, Total_training_parcels=budget,
                                Repeat=repetition, Seed=seed, Training_pixels_per_class=len(y) // 2,
                                Validation_pixels_per_class=len(target["validation_y"]) // 2,
                                Training_validation_overlap_pixels=0,
                                **metrics, **{"DCE_" + m: comparisons["Site 1*"][m] for m in METRICS},
                                Classification_map=str(path)))
            audit.append(dict(Study_area="Site 1*", Total_training_parcels=budget, Repeat=repetition,
                              Source_training_parcel_ids=selected_sources,
                              Source_training_manifest=str(folder / "source_training_pixels.npz"),
                              Actual_feature_count=FEATURE_COUNT, Seconds=time.monotonic() - start))
            print("Site 1*", "RF" + str(budget), "total; repeat", repetition,
                  "OA", round(metrics["OA"], 4), flush=True)
    save_csv(OUT / "accuracy_by_run.csv", results)
    frame = pd.DataFrame(results)
    summary = []
    for (site, method), part in frame.groupby(["Study_area", "Method"], sort=False):
        assert len(part) == 5
        row = dict(Study_area=site, Method=method,
                   Parcels_per_training_site=int(part.Parcels_per_training_site.iloc[0]),
                   Total_training_parcels=int(part.Total_training_parcels.iloc[0]),
                   Validation_pixels_per_class=int(part.Validation_pixels_per_class.iloc[0]), Runs=5)
        for m in METRICS:
            row[m + "_mean_percent"] = float(part[m].mean())
            row[m + "_SD_percent"] = float(part[m].std(ddof=1))
        summary.append(row)
    save_csv(OUT / "accuracy_summary.csv", summary)
    wide = []
    for site in ("Site 1", "Site 2", "Site 3", "Site 1*"):
        row = dict(Study_area=site)
        for method in ("RF10", "RF20", "RF50"):
            part = frame[(frame.Study_area == site) & (frame.Method == method)]
            row[method + "_OA_mean_percent"] = float(part.OA.mean())
            row[method + "_OA_SD_percent"] = float(part.OA.std(ddof=1))
        for m in METRICS:
            row["DCE_" + m + "_percent"] = comparisons[site][m]
        wide.append(row)
    save_csv(OUT / "oa_comparison.csv", wide)
    design = dict(Evaluation_seed=EVALUATION_SEED, Repetition_seeds=REPEAT_SEEDS,
                  Local_test_parcel_holdout=False, Training_budgets_total_per_experiment=BUDGETS,
                  Model_parameters=base.RF_PARAMS,
                  Training_pool="All previously verified, single-class labelled parcels; former test parcels are not withheld",
                  Validation="All available target-reference pixels with valid features plus an equal, fixed random sample of labelled non-target farmland pixels",
                  Local_evaluation="Non-independent: local training parcels and pixels may occur in validation",
                  Transfer_evaluation="Site 1 only; neither its samples nor its labels enter source RF training",
                  Transfer_total_labelled_parcels=BUDGETS,
                  Transfer_sampling="Equal total parcel counts per source, with complementary positive and negative quotas to give equal global class counts",
                  Feature_alignment=dict(Method="Direct use of already aligned input columns",
                                         Time_points=TIME_POINTS, Feature_groups=FEATURE_GROUPS,
                                         Feature_count=FEATURE_COUNT,
                                         Observation_dates_by_site={k: s["dates"] for k, s in states.items()}),
                  Validation_manifests={k: dict(CSV=full.fingerprint(s["folder"] / "balanced_validation_pixels.csv"),
                                               NPZ=str(s["folder"] / "validation_pixels.npz"),
                                               Per_class=len(s["validation_y"]) // 2,
                                               Conflicting_validation_pixels=int(s["conflicts"][s["validation_positions"]].sum()),
                                               Reference_diagnostics=s["reference_diagnostics"]) for k, s in states.items()},
                  Fixed_DCE_comparisons=comparisons, Runs=audit)
    (OUT / "experiment_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed", len(results), "models:", OUT, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-only", action="store_true")
    main(parser.parse_args().transfer_only)
