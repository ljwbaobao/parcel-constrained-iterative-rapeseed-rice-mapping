"""Evaluate majority-voted maps using all labelled pixels with valid predictions.

Configure input paths, crop attribute fields, and recorded-value aliases below.
Geodesic areas provide auxiliary checks on the selected final maps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject


OUTPUT_DIRECTORY = Path("<PATH_TO_FULL_REFERENCE_ACCURACY_OUTPUT_DIRECTORY>")
TRANSFER_CLASSIFICATION_MAP = Path(
    "<PATH_TO_SITE_1_TRANSFER_DCE_MAJORITY_VOTED_CLASSIFICATION_TIFF>"
)

NAMES = {
    1: "Rapeseed-rice",
    2: "Wheat-rice",
    3: "Double rice",
    4: "Other agricultural patterns",
}

SITE_CONFIGS = {
    key: {
        "site": label,
        "strategy": strategy,
        "field": f"<PATH_TO_{key.upper()}_ALL_FARMLAND_PARCEL_ID_TIFF>",
        "parcels": f"<PATH_TO_{key.upper()}_SEASONAL_CROP_LABELLED_PARCEL_VECTOR>",
        "spring": "<SPRING_CROP_ATTRIBUTE_FIELD_NAME>",
        "autumn": "<AUTUMN_CROP_ATTRIBUTE_FIELD_NAME>",
        "best_map": f"<PATH_TO_{key.upper()}_{strategy}_MAJORITY_VOTED_CLASSIFICATION_TIFF>",
    }
    for key, label, strategy in (
        ("site1", "Site 1", "ADC"),
        ("site2", "Site 2", "CEE"),
        ("site3", "Site 3", "BCE"),
    )
}

# Reference rasters for the three cropping patterns.
LABEL_RASTERS = {
    key: {
        "Rapeseed-rice": f"<PATH_TO_{key.upper()}_RAPESEED_RICE_REFERENCE_TIFF>",
        "Wheat-rice": f"<PATH_TO_{key.upper()}_WHEAT_RICE_REFERENCE_TIFF>",
        "Double rice": (
            None if key == "site2"
            else f"<PATH_TO_{key.upper()}_DOUBLE_RICE_REFERENCE_TIFF>"
        ),
    }
    for key in SITE_CONFIGS
}

# Replace each placeholder key with a recorded attribute value. Add entries
# when several recorded values represent the same English category.
# Attributes already using the canonical English values need no alias.
ATTRIBUTE_VALUE_ALIASES = {
    "<RECORDED_ATTRIBUTE_VALUE_FOR_RAPESEED>": "rapeseed",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_WHEAT_OR_BARLEY>": "wheat",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_EARLY_RICE>": "early_rice",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_LATE_OR_SINGLE_SEASON_RICE>": "rice",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_OTHER_CROP>": "other_crop",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_OTHER_SEASONAL_STATUS>": "other_seasonal_status",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_GREENHOUSE>": "greenhouse",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_LOTUS_POND>": "lotus_pond",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_FALLOW>": "fallow",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_RICE_SEEDLINGS>": "rice_seedlings",
    "<RECORDED_ATTRIBUTE_VALUE_FOR_OTHER>": "other",
    "<RECORDED_ATTRIBUTE_VALUE_WITH_UNKNOWN_CLASS>": "unknown",
}

# Consistency checks for the reference experiment. Set a value to None when
# evaluating different data without a corresponding reference result.
EXPECTED_RESULTS = {
    "Site 1": {"mapped_area_ha": 183.29, "pa_percent": 96.50},
    "Site 2": {"mapped_area_ha": 60.22, "pa_percent": 99.81},
    "Site 3": {"mapped_area_ha": 70.70, "pa_percent": 87.27},
    "Site 1*": {"mapped_area_ha": 191.50, "pa_percent": 92.37},
}


def validate_configuration(output_dir, transfer_map):
    values = [str(output_dir), str(transfer_map)]
    values.extend(str(value) for cfg in SITE_CONFIGS.values() for value in cfg.values())
    values.extend(
        str(value) for cfg in LABEL_RASTERS.values()
        for value in cfg.values() if value is not None
    )
    if any("<PATH_TO_" in value or "<SPRING_" in value or "<AUTUMN_" in value
           for value in values):
        raise ValueError("Replace the descriptive input paths and crop attribute fields before running.")
    paths = [transfer_map]
    paths.extend(cfg[key] for cfg in SITE_CONFIGS.values()
                 for key in ("field", "parcels", "best_map"))
    paths.extend(value for cfg in LABEL_RASTERS.values()
                 for value in cfg.values() if value is not None)
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)


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


def warp_presence(path, meta):
    destination = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)
    if path:
        with rasterio.open(path) as source:
            present = ((source.read(1) > 0) & (source.read_masks(1) > 0)).astype(np.uint8)
            reproject(
                present, destination,
                src_transform=source.transform, src_crs=source.crs,
                dst_transform=meta["transform"], dst_crs=meta["crs"],
                resampling=Resampling.nearest, num_threads=2,
            )
    return destination.astype(bool)


def reference_labels(cfg, reference_paths, meta):
    labels = np.zeros((meta["height"], meta["width"]), np.uint8)
    canonical_counts = {}
    for code in (1, 2, 3):
        mask = warp_presence(reference_paths[NAMES[code]], meta)
        labels[mask] = code
        canonical_counts[NAMES[code]] = int(mask.sum())

    vectors = gpd.read_file(cfg["parcels"]).to_crs(meta["crs"])
    codes = np.array([
        seasonal_class(spring, autumn)
        for spring, autumn in zip(vectors[cfg["spring"]], vectors[cfg["autumn"]])
    ])
    supplemented_counts = {}
    for code in (2, 3, 4):
        geometries = [geometry for geometry in vectors.geometry[codes == code]
                      if geometry is not None and not geometry.is_empty]
        if not geometries:
            supplemented_counts[NAMES[code]] = 0
            continue
        mask = rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=labels.shape, transform=meta["transform"],
            fill=0, all_touched=False, dtype="uint8",
        ).astype(bool)
        supplement = (labels == 0) & mask
        labels[supplement] = code
        supplemented_counts[NAMES[code]] = int(supplement.sum())
    return labels, {
        "canonical_class_pixels": canonical_counts,
        "supplemented_reference_pixels": supplemented_counts,
        "unknown_policy": "No target labels inferred outside the canonical target raster; unlabelled pixels excluded",
        "other_policy": "Other non-target parcel records inside the supplied farmland mask, including seasonal status labels; no off-farm land cover inferred",
    }


def row_areas(meta):
    transform = meta["transform"]
    if not meta["crs"].is_geographic or transform.b != 0 or transform.d != 0:
        raise ValueError("Area calculation requires a north-up geographic raster.")
    geod = Geod(ellps="WGS84")
    west, east = transform.c, transform.c + transform.a
    return np.array([
        abs(geod.polygon_area_perimeter(
            [west, east, east, west],
            [transform.f + row * transform.e, transform.f + row * transform.e,
             transform.f + (row + 1) * transform.e,
             transform.f + (row + 1) * transform.e],
        )[0]) / 10000
        for row in range(meta["height"])
    ])


def measure(pred, labels, domain, weights):
    actual = labels == 1
    positive = pred == 1
    masks = {
        "TP": domain & actual & positive,
        "FN": domain & actual & ~positive,
        "FP": domain & ~actual & positive,
        "TN": domain & ~actual & ~positive,
    }
    counts = {key: int(mask.sum()) for key, mask in masks.items()}
    areas = {key: float(np.dot(mask.sum(axis=1), weights)) for key, mask in masks.items()}

    def metrics(values):
        tp, tn, fp, fn = (values[key] for key in ("TP", "TN", "FP", "FN"))

        def ratio(numerator, denominator):
            return 100 * numerator / denominator if denominator else None

        return {
            "OA_percent": ratio(tp + tn, tp + tn + fp + fn),
            "UA_percent": ratio(tp, tp + fp),
            "PA_percent": ratio(tp, tp + fn),
            "F1_percent": ratio(2 * tp, 2 * tp + fp + fn),
        }

    return {
        **counts, **metrics(counts),
        **{key + "_ha": value for key, value in areas.items()},
        **{"Area_" + key: value for key, value in metrics(areas).items()},
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run(output_dir, transfer_map=None):
    output_dir = Path(output_dir)
    transfer_map = Path(transfer_map) if transfer_map is not None else TRANSFER_CLASSIFICATION_MAP
    validate_configuration(output_dir, transfer_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(key, cfg["site"], cfg["strategy"], cfg["best_map"])
            for key, cfg in SITE_CONFIGS.items()]
    jobs.append(("site1", "Site 1*", "DCE", transfer_map))
    summary, details, audit = [], [], []
    for key, site, strategy, map_path in jobs:
        cfg = SITE_CONFIGS[key]
        map_path = Path(map_path)
        with rasterio.open(map_path) as source:
            pred, meta = source.read(1), source.profile
        with rasterio.open(cfg["field"]) as source:
            if (source.shape != pred.shape or source.crs != meta["crs"]
                    or source.transform != meta["transform"]):
                raise ValueError(f"The farmland raster and classification grid differ: {site}")
            fields = (source.read(1) > 0) & (source.read_masks(1) > 0)
        labels, diagnostics = reference_labels(cfg, LABEL_RASTERS[key], meta)
        valid = np.isin(pred, (0, 1))
        if np.any(valid & ~fields):
            raise ValueError(f"Valid classification pixels occur outside the supplied farmland mask: {site}")
        domain = (labels > 0) & valid
        weights = row_areas(meta)
        metrics = measure(pred, labels, domain, weights)
        mapped_area = float(np.dot((pred == 1).sum(axis=1), weights))
        expected = EXPECTED_RESULTS[site]
        area_check, pa_check = None, None
        if expected["mapped_area_ha"] is not None:
            area_check = abs(mapped_area - expected["mapped_area_ha"]) < 0.011
            if not area_check:
                raise ValueError(f"Mapped area differs from the configured check: {site}, {mapped_area}")
        if expected["pa_percent"] is not None:
            pa_check = (metrics["PA_percent"] is not None
                        and round(metrics["PA_percent"], 2) == expected["pa_percent"])
            if not pa_check:
                raise ValueError(f"PA differs from the configured check: {site}, {metrics['PA_percent']}")
        row = {
            "Study_area": site, "Strategy": strategy, **metrics,
            "Reference_pixels": int(domain.sum()),
            "Unlabelled_predicted_target_pixels": int(((pred == 1) & (labels == 0)).sum()),
            "Labelled_pixels_without_prediction": int(((labels > 0) & ~valid).sum()),
            "Mapped_target_area_ha": mapped_area,
        }
        if sum(metrics[name] for name in ("TP", "TN", "FP", "FN")) != row["Reference_pixels"]:
            raise ValueError(f"Confusion-matrix counts do not match the evaluation domain: {site}")
        summary.append(row)
        for code, name in NAMES.items():
            mask = domain & (labels == code)
            n_target = int((mask & (pred == 1)).sum())
            details.append({
                "Study_area": site, "Strategy": strategy, "Reference_class": name,
                "Reference_pixels": int(mask.sum()), "Predicted_target_pixels": n_target,
                "Predicted_nontarget_pixels": int((mask & (pred == 0)).sum()),
                "Predicted_target_area_ha": float(np.dot((mask & (pred == 1)).sum(1), weights)),
                "Share_of_FP_percent": (
                    100 * n_target / metrics["FP"] if code != 1 and metrics["FP"] else None
                ),
            })
        total_fp = sum(item["Predicted_target_pixels"] for item in details[-len(NAMES):]
                       if item["Reference_class"] != NAMES[1])
        if total_fp != metrics["FP"]:
            raise ValueError(f"False-positive class counts do not match the confusion matrix: {site}")
        inputs = [map_path, cfg["field"], cfg["parcels"]]
        inputs.extend(value for value in LABEL_RASTERS[key].values() if value)
        for suffix in (".dbf", ".shx", ".prj", ".cpg"):
            sidecar = Path(cfg["parcels"]).with_suffix(suffix)
            if sidecar.exists():
                inputs.append(sidecar)
        audit.append({
            "site": site, "strategy": strategy,
            "pixel_and_geodesic_metrics": metrics,
            "prediction": fingerprint(map_path),
            "inputs": [fingerprint(value) for value in inputs],
            "grid": {"shape": list(pred.shape), "crs": str(meta["crs"]),
                     "transform": list(meta["transform"]), "nodata": meta["nodata"]},
            "reference_diagnostics": diagnostics,
            "map_and_reference_consistency": {
                "mapped_area_ha": mapped_area,
                "expected_mapped_area_ha": expected["mapped_area_ha"],
                "mapped_area_check_passed": area_check,
                "expected_pa_percent": expected["pa_percent"],
                "pa_check_passed": pa_check,
            },
            "coverage": {
                "classified_pixels": int(valid.sum()),
                "field_pixels": int(fields.sum()),
                "labelled_classified_pixels": int(domain.sum()),
                "classified_unlabelled_pixels": int((valid & (labels == 0)).sum()),
                "classified_other_parcel_pixels": int((domain & (labels == 4)).sum()),
            },
        })
        print(site, strategy, {
            name: row[name] for name in
            ("TP", "FP", "FN", "TN", "UA_percent", "PA_percent", "F1_percent")
        }, flush=True)
    main_fields = (
        "Study_area", "Strategy", "Reference_pixels", "TN", "FP", "FN", "TP",
        "OA_percent", "UA_percent", "PA_percent", "F1_percent",
        "Unlabelled_predicted_target_pixels", "Labelled_pixels_without_prediction",
    )
    write_csv(output_dir / "full_reference_accuracy.csv",
              [{name: row[name] for name in main_fields} for row in summary])
    write_csv(output_dir / "commission_errors_by_class.csv", details)
    (output_dir / "evaluation_sources.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Output:", output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--transfer-map", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.transfer_map)
