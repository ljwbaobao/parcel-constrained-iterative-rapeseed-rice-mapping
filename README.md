# Parcel-constrained iterative mapping of rapeseed-rice cropping patterns

Code accompanying the study on weakly supervised mapping of rapeseed-rice cropping patterns using Sentinel-1 time series and field-parcel-constrained iterative sample supplementation.

## Method

The workflow combines rule-based selection of high-confidence seed samples with an initial random forest (RF) classification. Within-parcel target-pixel proportions guide iterative supplementation of positive pseudo-labels. Negative samples are updated to maintain class balance, and parcel-level majority voting produces the final maps.

Three local experiments are configured separately for Sites 1, 2, and 3. In the transfer experiment, initial samples from Sites 2 and 3 are combined, iterative supplementation uses source-region parcels only, and the trained models are applied to Site 1. Reference crop labels are used for assessment and strategy evaluation, rather than for assigning the positive pseudo-labels in the proposed iterative method.

## Files

| File | Purpose |
| --- | --- |
| `0_Supplementary_GEE_Sentinel1_Preprocessing.md` | GEE code for Sentinel-1A selection, conversion from sigma-nought to gamma-nought using the ellipsoid incidence angle, conversion to linear intensity, circular median filtering, and VV/VH export. Sentinel-2 imagery is used for visual reference. |
| `1_Extract_Dual_Polarization_SAR_Whittaker_HDF5.py` | Extracts aligned VV/VH pixel time series from dual-band TIFFs, applies a second-order Whittaker smoother with lambda = 1, and writes the smoothed HDF5 files. |
| `2_Non_Agricultural_Negative_Pixels.py` | Samples pixel-centre coordinates inside the study-area boundary and outside the supplied farmland boundaries, producing the initial negative-sample CSV. |
| `3_Initial_RF_Rule.py` | Selects rule-based positive seeds, trains the local initial RF with non-agricultural negative samples, and exports seed and initial classification rasters. |
| `4_Iterative_RF.py` | Runs parcel-proportion interval sequences, accumulates positive pseudo-labels, draws negatives from current 0%-target parcels, and exports majority-voted classification maps. |
| `5_Accuracy_Assessment.py` | Evaluates classification maps using target reference pixels and sampled non-target farmland pixels; exports confusion-matrix counts and OA, UA, PA, and F1. |
| `6_Transfer_Initial_RF.py` | Trains the joint source-region initial RF, reports internal source-sample validation results, and exports initial maps and transfer caches. |
| `6_Transfer_Iterative_RF.py` | Entry point for iterative source-region training and target-region classification. |
| `6_Transfer_Iterative_RF_Core.py` | Implements source-region pseudo-label supplementation and target-region majority voting; imports the transfer initial-RF module. |
| `7_calculate_parcel_area_statistics.py` | Groups crop-specific raster pixels by parcel ID and calculates parcel count, mean, median, and total area using WGS84 ellipsoidal pixel areas that vary by latitude. |
| `8_analyze_selected_strategy_positive_sample_classes.py` | Reconstructs newly added positive samples for selected strategies and counts their reference classes: rapeseed-rice, wheat-rice, double rice, and other/unclassified. |
| `9_Supervised Classification.py` | Runs RF10/RF20/RF50 for the three local experiments and the transfer experiment, repeats training five times, and compares majority-voted predictions with existing DCE maps on fixed class-balanced validation pixels. |
| `9_Supervised_Classification_Support.py` | Configuration, reference-label handling, and shared functions required by the supervised RF comparison. Keep it beside the supervised main script. |
| `10_evaluate_full_reference_confusion.py` | Evaluates selected local and transfer maps using all labelled pixels with valid predictions; reports confusion matrices, false-positive classes, coverage diagnostics, and auxiliary geodesic area checks. |

## Software requirements

The Python scripts use NumPy, SciPy, h5py, Rasterio, GDAL (`osgeo`), GeoPandas, pyproj, scikit-learn, pandas, joblib, psutil, and tqdm. Install the dependencies needed by the scripts you intend to run in a compatible geospatial Python environment.

Run the GEE JavaScript in the Google Earth Engine Code Editor with access to your study-area boundary asset.

## Inputs and configuration

Replace descriptive path placeholders, crop-attribute field names, and recorded-value aliases before running the relevant scripts. Inputs depend on the analysis:

- Study-area boundaries and field-parcel boundaries or positive-integer parcel-ID rasters.
- Dual-band SAR TIFFs with VV in band 1 and VH in band 2, and acquisition dates in their filenames.
- Smoothed VV/VH HDF5 files with a `value` matrix and a `timestamps` attribute.
- A SAR template raster and a raster-row/column-to-HDF5-index NumPy mapping.
- Initial negative-sample CSVs with `longitude` and `latitude` columns.
- Crop-specific reference rasters, seasonal crop-labelled parcel vectors, and classification-subregion masks.
- Existing classification maps, archived intermediate predictions, or prepared caches required by the corresponding analysis.

Classification maps, parcel rasters, masks, and index mappings must use consistent grids. The transfer and supervised comparison scripts expect 29 already aligned time points per polarization, producing 116 features from VH, VV, VH/VV, and RVI. They do not remove a site-specific observation to align the inputs.

The GEE example exports VV and VH separately. Prepare a dual-band image for each acquisition before using script 1.

### Negative samples

The three negative-sample definitions serve different purposes:

1. **Initial RF training:** pixels outside the supplied farmland parcel boundaries, within the study-area mask.
2. **Iterative RF training:** pixels in parcels whose current pixel-wise target proportion is 0%.
3. **Accuracy assessment:** non-target agricultural reference pixels, including wheat-rice, double rice, and other agricultural patterns as available.

The non-agricultural sampling script relies on the supplied parcel boundaries to define farmland.

## Local classification

1. Configure the study-area asset, acquisition period, and relative orbit in the GEE code and export the SAR observations.
2. Prepare dual-band TIFFs and run `1_Extract_Dual_Polarization_SAR_Whittaker_HDF5.py`.
3. Prepare the parcel rasters and raster-to-HDF5 mapping. Run `2_Non_Agricultural_Negative_Pixels.py` to generate the initial negative coordinates.
4. Configure and run `3_Initial_RF_Rule.py`.
5. Supply the initial pixel-wise map, positive-seed map, and remaining inputs to `4_Iterative_RF.py`.
6. Configure `5_Accuracy_Assessment.py` for class-balanced evaluation. Set `TARGET_REFERENCE_PIXEL_COUNT` to the number of target reference pixels in the classification subregion and choose the correct filename family.

Configure the local scripts separately for each site. Pixel-wise predictions determine the intervals for subsequent iterations; majority voting generates the parcel-level output maps.

## Cross-site transfer

Keep the three `6_Transfer_*.py` files together. Configure source and target inputs and the output directory in `6_Transfer_Initial_RF.py`, then run:

```bash
python 6_Transfer_Initial_RF.py
python 6_Transfer_Iterative_RF.py
```

The initial script generates the `.transfer_cache` inputs required by the iterative stage. Its internal validation uses held-out source samples; it is separate from target-site map assessment. The iterative transfer code exports target-site majority-voted TIFFs. Use script 5 or script 10 for external map assessment.

## Additional analyses

### Parcel areas

Configure `ROOT` and `NAMES` in script 7 and run:

```bash
python 7_calculate_parcel_area_statistics.py
```

The inputs must be unrotated EPSG:4326 parcel-ID rasters. Outputs are `parcel_area_source_audit.json` and `parcel_area_recalculated.md`. These summarize crop-parcel areas; they are distinct from mapped target-class areas.

Geographic raster cells have angular dimensions. Calculate areas from the raster transform and latitude rather than assigning a fixed 100 square metres to each cell. Script 10 also calculates areas for the selected final classification maps as auxiliary consistency checks.

### Reference classes of added pseudo-labels

Configure script 8 and run:

```bash
python 8_analyze_selected_strategy_positive_sample_classes.py
```

Local additions require archived pixel-wise predictions. Transfer additions can use saved positive-sample indices (`saved_indices`) or replay the configured transfer core (`replay_core`). Replaying training is not a substitute for the archived indices when auditing historical outputs.

### Supervised RF comparison

Configure paths in both `9_Supervised Classification.py` and `9_Supervised_Classification_Support.py`, then run:

```bash
python "9_Supervised Classification.py"
```

RF10, RF20, and RF50 use 10, 20, and 50 labelled parcels per local experiment, respectively, with equal target and non-target parcel counts. For transfer, each budget is the combined total from Sites 2 and 3. Five training repetitions use randomly reselected parcels.

The comparison requires existing `prepared.npz`, `features.npy`, and `design.json` caches. The support module provides configuration and shared functions; it does not create these preparation caches. Supply compatible caches and configure the reference-label fields and aliases before running.

Validation uses all target reference pixels with valid features and an equally sized random sample of labelled non-target farmland pixels, fixed across budgets, repetitions, and methods. Existing majority-voted DCE maps are re-evaluated on those same pixels. No independent local test-parcel hold-out is used.

Outputs include majority-voted maps, training/validation sample records, `accuracy_by_run.csv`, `oa_comparison.csv`, and `experiment_design.json`.

### Full-reference map assessment

Configure the reference rasters, seasonal label attributes, selected classification maps, and output directory in script 10, then run:

```bash
python 10_evaluate_full_reference_confusion.py
```

This evaluates all labelled pixels with valid binary predictions within the supplied farmland domain, rather than a balanced subset. The outputs are:

- `full_reference_accuracy.csv`: confusion-matrix counts, OA, UA, PA, F1, and counts of unlabelled predictions or missing predictions.
- `commission_errors_by_class.csv`: predictions and false-positive composition by reference crop class.
- `evaluation_sources.json`: input fingerprints, grid details, coverage diagnostics, and area consistency checks.

Unknown reference pixels are excluded from accuracy calculations. The script reports their coverage separately.

## Strategy notation and voting

Intervals A-E correspond to within-parcel target-pixel proportions `(0, 20%]`, `(20, 40%]`, `(40, 60%]`, `(60, 80%]`, and `(80, 100%]`. Parcels with 0% target pixels are excluded from positive supplementation.

Filenames encode A-E as digits 1-5: `143` is ADC, `355` is CEE, `235` is BCE, and `435` is DCE. Each position specifies the interval selected at that iteration. For ADC, the successive sequences are A, AD, and ADC.

Majority voting assigns a parcel to the target class when at least 50% of its valid classified pixels are target pixels.

## Interpretation and data availability

Class-balanced OA describes performance with equal class contributions; it does not measure prevalence-weighted map accuracy. Full-reference assessment complements it by including all labelled pixels with valid predictions. Local training and evaluation parcels may overlap, so local estimates are not independent. In transfer, Site 1 samples are excluded from source-region model training.

Government-provided parcel boundaries and crop labels are confidential and are not distributed. Users must supply their own data, intermediate products, and configuration. The repository contains code and documentation only. Results depend on the input data, configuration, software environment, and random seeds.
