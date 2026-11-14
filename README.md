# Parcel-constrained iterative mapping of rapeseed–rice cropping patterns

Code accompanying the study on weakly supervised mapping of rapeseed–rice cropping patterns using Sentinel-1 time series and field-parcel-constrained iterative sample supplementation.

## Overview

The method combines rule-based selection of high-confidence seed samples with an initial random forest (RF) classification. Target-pixel proportions within field parcels guide iterative supplementation of positive pseudo-labels, while negative samples are resampled to maintain class balance. Parcel-level majority voting is applied to generate the final iterative classification maps.

The study includes three local experiments and a cross-site transfer experiment. In the transfer experiment, Sites 2 and 3 provide the training samples and parcels for iterative supplementation, and the trained models are applied to Site 1. Crop-type reference labels are used for accuracy assessment and strategy evaluation; they are not used to assign pseudo-labels during parcel-constrained supplementation.

## Code files

| File | Purpose |
| --- | --- |
| `0_Supplementary_GEE_Sentinel1_Preprocessing.md` | GEE script for Sentinel-1 selection, sigma-nought to gamma-nought conversion, conversion to linear intensity, median filtering, and VV/VH export. |
| `1_Extract_Dual_Polarization_SAR_Whittaker_HDF5.py` | Extracts aligned pixel time series from dual-band VV/VH TIFFs and applies second-order Whittaker smoothing with lambda = 1 before writing HDF5 files. |
| `2_Non_Agricultural_Negative_Pixels.py` | Exports random pixel-center coordinates outside the supplied farmland parcel boundaries for the initial negative-sample pool. |
| `3_Initial_RF_Rule.py` | Selects rule-based positive seeds, trains the local initial RF with non-agricultural negative samples, and exports rule-screened and initial RF classification maps. |
| `4_Iterative_RF.py` | Evaluates parcel-proportion interval sequences, accumulates positive pseudo-labels, samples negatives from current 0%-target parcels, and exports majority-vote classification maps. |
| `5_Accuracy_Assessment.py` | Assesses iterative classification maps using target reference pixels and sampled non-target farmland pixels; exports confusion-matrix counts and OA, UA, PA, and F1. |
| `6_Transfer_Initial_RF.py` | Trains a joint initial model using Sites 2 and 3, reports source-region internal validation metrics, and exports initial maps and caches for transfer iterations. |
| `6_Transfer_Iterative_RF.py` | Entry point for source-region iterative supplementation and target-region majority-vote classification. |
| `6_Transfer_Iterative_RF_Core.py` | Implements the transfer iterations; imports the transfer initial-RF module and is called by the transfer entry point. |

## Requirements

The Python scripts use NumPy, SciPy, h5py, Rasterio, GDAL (`osgeo`), GeoPandas, scikit-learn, joblib, psutil, pandas, and tqdm. Install these packages in a compatible geospatial Python environment. The GEE script runs in the Google Earth Engine Code Editor and requires access to the configured study-area asset.

## Input data and configuration

Replace the descriptive path and sample-count placeholders before running a script. Depending on the processing stage, the inputs include:

- Study-area boundaries and field-parcel boundaries or parcel-ID rasters.
- Dual-band SAR TIFFs with band 1 named `VV` and band 2 named `VH` for time-series extraction.
- Smoothed VV and VH HDF5 files containing a `value` matrix and a `timestamps` attribute.
- A SAR template raster and a precomputed raster-row/column-to-HDF5-index NumPy mapping.
- Initial negative-sample coordinate CSVs with `longitude` and `latitude` columns.
- Rapeseed–rice reference rasters, all-field rasters, and classification-subregion masks for external accuracy assessment.

The classification rasters, parcel rasters, masks, and index mapping must refer to the same pixel grid. The transfer scripts require 29 chronologically ordered observations per polarization at each site, with matching VV/VH pixel indices and timestamps within each site.

Initial RF negative samples and accuracy-assessment negatives serve different purposes: the former are drawn from outside the supplied farmland boundaries, while the latter are agricultural pixels outside the rapeseed–rice reference class. During iterative training, negative samples are drawn from parcels with a current target-pixel proportion of 0%.

## Running the local experiments

1. Run the GEE script after configuring the study-area asset, acquisition period, and relative orbit.
2. Prepare the dual-band SAR TIFFs and run `1_Extract_Dual_Polarization_SAR_Whittaker_HDF5.py`.
3. Prepare the parcel rasters and pixel-index mapping, then run `2_Non_Agricultural_Negative_Pixels.py` to create the initial negative-pixel CSV.
4. Configure and run `3_Initial_RF_Rule.py` for the selected site.
5. Set the initial pixel-wise RF map, rule-screened positive map, and other inputs in `4_Iterative_RF.py`, then run the iterative classification.
6. Configure `5_Accuracy_Assessment.py` to assess the exported iterative maps. Set `TARGET_REFERENCE_PIXEL_COUNT` to the total number of target reference pixels within the classification subregion and select the appropriate filename style.

The local scripts are configured separately for each study site. Initial RF maps are pixel-wise outputs; the local iterative script writes parcel-level majority-vote maps and uses pixel-wise predictions to determine the intervals for subsequent iterations.

## Running the transfer experiment

Keep all three `6_Transfer_*.py` files in the same directory. Configure the source and target inputs and the output directory in `6_Transfer_Initial_RF.py`, then run:

```bash
python 6_Transfer_Initial_RF.py
python 6_Transfer_Iterative_RF.py
```

The first script generates the initial maps and the `.transfer_cache` files required by the second script. Iterative supplementation uses source-region parcels only. The iterative script exports Site 1 majority-vote maps and does not calculate target-region accuracy. Use `5_Accuracy_Assessment.py` separately for that assessment.

## Strategy notation

Intervals A–E correspond to target-pixel proportions `(0, 20%]`, `(20, 40%]`, `(40, 60%]`, `(60, 80%]`, and `(80, 100%]`, respectively. Parcels with 0% target pixels are excluded from positive supplementation.

Output filenames encode A–E as digits 1–5. For example, `143` denotes ADC, `355` denotes CEE, `235` denotes BCE, and `435` denotes DCE. Successive digits identify the intervals used in successive iterations. Majority voting assigns a parcel to the target class when at least 50% of its classified pixels are target pixels.

## Data availability

This repository contains code only. The government-provided field-parcel boundaries and crop-type labels are confidential and are not distributed with the repository. Users must supply their own input data and configure the descriptive placeholders. Results depend on the input data, study-area settings, software environment, and random seeds.
