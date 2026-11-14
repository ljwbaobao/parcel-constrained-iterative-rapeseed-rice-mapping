"""Assess classification TIFFs using target and non-target farmland reference pixels."""

from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix


RESULT_DIRECTORY = r"<PATH_TO_ITERATIVE_CLASSIFICATION_TIFF_DIRECTORY>"
TARGET_REFERENCE_TIFF = r"<PATH_TO_RAPESEED_RICE_REFERENCE_PARCEL_RASTER>"
ALL_FIELDS_TIFF = r"<PATH_TO_ALL_FIELD_PARCEL_RASTER>"
STUDY_BOUNDARY_TIFF = r"<PATH_TO_CLASSIFICATION_SUBREGION_BOUNDARY_RASTER>"
OUTPUT_DIRECTORY = r"<PATH_TO_ACCURACY_RESULT_OUTPUT_DIRECTORY>"

# Replace the placeholder with the total number of positive reference pixels
# in the classification subregion. With enough negative candidates, this
# selects all positive reference pixels and an equal number of negatives.
TARGET_REFERENCE_PIXEL_COUNT = "<TOTAL_NUMBER_OF_TARGET_REFERENCE_PIXELS_IN_THE_CLASSIFICATION_SUBREGION>"
RANDOM_SEED = 42

# Select one filename family to avoid mixing old and new runs in one folder.
# Use "english" for Site1_strategy_435_iter_3.tif or
# Tongxiang_strategy_435_iteration_3.tif; use "legacy" for older outputs.
FILENAME_STYLE = "english"
ENGLISH_FILENAME_PATTERN = re.compile(
    r"^.*_strategy_(?P<strategy>[1-5]{1,3})_(?:iter|iteration)_(?P<iteration>[1-3])\.tiff?$",
    flags=re.IGNORECASE,
)
LEGACY_FILENAME_PATTERN = re.compile(
    r"^.*_\u6307\u4ee4(?P<strategy>[1-5]{1,3})_\u7b2c(?P<iteration>[1-3])\u6b21\u8fed\u4ee3\u7ed3\u679c\.tiff?$",
    flags=re.IGNORECASE,
)


def load_tif_as_array(path):
    with rasterio.open(path) as source:
        return source.read(1)


def evaluate_single_tif(classification_path, rotation_mask_path, all_fields_path,
                        boundary_path, sample_count, strategy, iteration):
    """Select reference samples and calculate the confusion matrix and metrics."""
    classification = load_tif_as_array(classification_path)
    target_mask = load_tif_as_array(rotation_mask_path)
    fields = load_tif_as_array(all_fields_path)
    boundary = load_tif_as_array(boundary_path)
    if not (classification.shape == target_mask.shape == fields.shape == boundary.shape):
        raise ValueError("All input rasters must have matching dimensions.")

    positive_rows, positive_cols = np.where(
        (fields != 0) & (target_mask != 0) & (boundary != 0)
    )
    negative_rows, negative_cols = np.where(
        (fields != 0) & (target_mask == 0) & (boundary != 0)
    )
    if not len(positive_rows) or not len(negative_rows):
        raise ValueError("Both positive and negative reference candidates are required.")

    positive_count = min(sample_count, len(positive_rows))
    negative_count = min(sample_count, len(negative_rows))
    positive_selection = (random.sample(range(len(positive_rows)), positive_count)
                          if len(positive_rows) > positive_count else range(len(positive_rows)))
    negative_selection = (random.sample(range(len(negative_rows)), negative_count)
                          if len(negative_rows) > negative_count else range(len(negative_rows)))
    positive_selection = np.asarray(list(positive_selection), dtype=np.int64)
    negative_selection = np.asarray(list(negative_selection), dtype=np.int64)
    rows = np.concatenate([positive_rows[positive_selection], negative_rows[negative_selection]])
    cols = np.concatenate([positive_cols[positive_selection], negative_cols[negative_selection]])
    reference = np.concatenate([np.ones(positive_count), np.zeros(negative_count)])
    prediction = (classification[rows, cols] == 1).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(reference, prediction, labels=[0, 1]).ravel()
    ua = tp / (tp + fp) if tp + fp else 0.0
    pa = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * ua * pa / (ua + pa) if ua + pa else 0.0
    return {
        "Filename": Path(classification_path).name,
        "Strategy": strategy,
        "Iteration": iteration,
        "Sample_count": len(reference),
        "Positive_count": positive_count,
        "Negative_count": negative_count,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "OA": round(float(accuracy_score(reference, prediction)), 4),
        "UA": round(float(ua), 4),
        "PA": round(float(pa), 4),
        "F1": round(float(f1), 4),
    }


def batch_evaluate():
    try:
        sample_count = int(TARGET_REFERENCE_PIXEL_COUNT)
    except (ValueError, TypeError) as error:
        raise ValueError("Set TARGET_REFERENCE_PIXEL_COUNT to the positive reference-pixel total.") from error
    if sample_count <= 0:
        raise ValueError("TARGET_REFERENCE_PIXEL_COUNT must be positive.")
    paths = [RESULT_DIRECTORY, TARGET_REFERENCE_TIFF, ALL_FIELDS_TIFF,
             STUDY_BOUNDARY_TIFF, OUTPUT_DIRECTORY]
    if any("<PATH_TO_" in path for path in paths):
        raise ValueError("Replace the descriptive path placeholders before running.")
    if FILENAME_STYLE not in ("english", "legacy"):
        raise ValueError('FILENAME_STYLE must be "english" or "legacy".')
    pattern = ENGLISH_FILENAME_PATTERN if FILENAME_STYLE == "english" else LEGACY_FILENAME_PATTERN
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    output = Path(OUTPUT_DIRECTORY)
    output.mkdir(parents=True, exist_ok=True)
    results = {1: [], 2: [], 3: []}
    errors = []
    matched = 0
    # Preserve the original workflow: samples are selected for each map,
    # using one random sequence seeded at the start of the batch.
    for path in Path(RESULT_DIRECTORY).iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        matched += 1
        try:
            iteration = int(match.group("iteration"))
            record = evaluate_single_tif(path, TARGET_REFERENCE_TIFF, ALL_FIELDS_TIFF,
                                         STUDY_BOUNDARY_TIFF, sample_count,
                                         match.group("strategy"), iteration)
            results[iteration].append(record)
            print(f"Assessed {path.name}: OA={record['OA']:.4f}")
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    if matched == 0:
        raise ValueError("No result filenames match FILENAME_STYLE in RESULT_DIRECTORY.")
    for iteration, records in results.items():
        if records:
            frame = pd.DataFrame(records).sort_values("Strategy")
            path = output / f"iteration_{iteration}_accuracy_results.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"Exported {len(records)} records to {path}")
    if errors:
        (output / "assessment_errors.txt").write_text("\n".join(errors), encoding="utf-8")


if __name__ == "__main__":
    batch_evaluate()
