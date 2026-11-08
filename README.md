# 论文方法

Code accompanying the study on weakly supervised mapping of rapeseed–rice cropping patterns using Sentinel-1 time series and field-parcel-constrained iterative sample supplementation.

## Overview

The method combines rule-based selection of high-confidence seed samples with an initial random forest (RF) classification. It then uses crop-unlabelled field-parcel boundaries to identify candidate omitted target pixels and iteratively supplement the positive pseudo-label set, while resampling negative pixels to maintain class balance. The workflow is designed to improve the representation of target-class variability when only a limited initial labelled sample set is available.

The study evaluates the method at three sites in Zhejiang Province, China, and includes a cross-site transfer experiment. Reference labels are used for accuracy assessment and strategy evaluation; they are not required as crop labels for the unlabelled parcels used during pseudo-label selection.

## Code files

| File | Purpose |
| --- | --- |
| `0_Supplementary_GEE_Sentinel1_Preprocessing.md` | Google Earth Engine script for Sentinel-1 preprocessing and export of VV/VH time series. |
| `1_Whittaker_Smoothing.py` | Applies second-order Whittaker smoothing to merged pixel time-series data stored in HDF5 files. |
| `2_Initial_RF_Rule.py` | Selects initial seed samples using the rule-based criteria and trains/applies the initial RF classifier. |
| `3_Iterative_RF.py` | Performs parcel-constrained iterative pseudo-label supplementation and RF classification. |

## Requirements

The Python scripts use common scientific Python packages, including NumPy, SciPy, h5py, Rasterio, scikit-learn, joblib, psutil, pandas, and tqdm. Install the packages required by the script you intend to run in your Python environment.

## Data and configuration

This repository contains code only. Study-area rasters, HDF5 time series, parcel vectors, and reference labels are not included. Configure the input and output paths, raster metadata, and study-specific settings in each script before running it. The Google Earth Engine script also requires access to the corresponding study-area asset.

## Reproducibility note

The scripts reflect the processing and classification workflow used in the study. Results depend on the input data, study-area configuration, software environment, and random seeds specified in the code.

