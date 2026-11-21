# Supplementary Code: Sentinel-1 Preprocessing in Google Earth Engine

The following Google Earth Engine (GEE) script filters Sentinel-1A imagery for the study area, applies the incidence-angle normalization and median filtering described in the manuscript, and exports the processed VV and VH time-series images to Google Drive.

```javascript

//---------------------------Part 1: Initialization---------------------------------
// Study period
var start_time = '2023-12-1';  
var end_time = '2024-11-25'; 

// Study area
var StudyArea = ee.FeatureCollection("projects/assets/shape/yuhang");
var relativeOrbit = 69;
Map.centerObject(StudyArea, 14);
Map.addLayer(StudyArea, {}, "StudyArea", false);

// Sentinel-1 data and visualization parameters
var Sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD");
var S1Param = {"opacity": 1, "bands": ["VV"], "min": -23.44022392940915, "max": 17.314610481899116, "gamma": 1.385};

// Reference optical imagery (Sentinel-2) and visualization parameters
var S2 = ee.ImageCollection("COPERNICUS/S2");
var S2Param = {"opacity": 1, "bands": ["B4", "B3", "B2"], "min": 731, "max": 2339, "gamma": 1.1};

// Filter Sentinel-1 imagery
var S1 = Sentinel1.filterBounds(StudyArea)
                  .filterDate(start_time, end_time)
                  .filterMetadata('platform_number', 'equals', 'A')
                  .filterMetadata('instrumentMode', 'equals', 'IW')
                  .filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING')
                  .filterMetadata('resolution_meters', 'equals', 10)
                  .filterMetadata('relativeOrbitNumber_start', 'equals', relativeOrbit)
                  .select(["VV", "VH", "angle"]);
print("Sentinel1:", S1);

// Clip the imagery to the study area
function maskBorder(image) {
  return image.clip(StudyArea);
}
var Clip_S1 = S1.map(maskBorder);
Map.addLayer(Clip_S1.first(), S1Param, "S1Image", false);

// Filter reference optical imagery (Sentinel-2)
var S2_image = S2.filterBounds(StudyArea)
                 .filterDate('2024-5-1', '2024-5-30')
                 .filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'less_than', 5);
Map.addLayer(S2_image.first().clip(StudyArea), S2Param, "S2image");

//---------------------------Part 2: Sentinel-1 preprocessing------------------------
// 1. Incidence-angle normalization
function toGamma0(img) {
  var gamma0 = img.expression('i - 10 * log10(cos(angle * pi / 180))', {
    i: img.select(['VV', 'VH']),
    angle: img.select('angle'),
    pi: Math.PI
  });
  return img.addBands(gamma0, null, true);
}

// 2. Convert backscatter from decibels to linear intensity
function toNatural(img) {
  img = ee.Image(img);
  var vvimg = img.select("VV");
  var vhimg = img.select("VH");
  var vvlinear = ee.Image(10.0).pow(vvimg.divide(10.0)).rename("VV_linear");
  var vhlinear = ee.Image(10.0).pow(vhimg.divide(10.0)).rename("VH_linear");
  var linearimg = vvlinear.addBands(vhlinear);
  return img.addBands(linearimg, null, true).select(["VV_linear", "VH_linear"]);
}

// 3. Speckle-noise reduction using a median filter
function medianFilter(img) {
  img = ee.Image(img);
  var radius = 3;
  var kernelType = "circle";
  var units = "pixels";
  var imgbn = img.bandNames().get(0);
  var img_Filtered = img.focalMedian(radius, kernelType, units);
  return img.addBands(img_Filtered.rename(ee.String(imgbn).cat("_spk")), null, true);
}

// Preprocess Sentinel-1 images
var Pre_Clip_S1 = Clip_S1.map(toGamma0).map(toNatural);
var PreV_spk_Clip_S1 = Pre_Clip_S1.select(["VV_linear"]).map(medianFilter).select("VV_linear_spk");
var PreH_spk_Clip_S1 = Pre_Clip_S1.select(["VH_linear"]).map(medianFilter).select("VH_linear_spk");
var Pre_spk_Clip_S1 = PreV_spk_Clip_S1.combine(PreH_spk_Clip_S1);

//---------------------------Part 3: Export the processed images---------------------

// Convert the image collection to a list for iteration
var listOfImages = Pre_spk_Clip_S1.toList(Pre_spk_Clip_S1.size());

// Get the number of images
var n = listOfImages.size().getInfo();

// Iterate over each image and export it using its acquisition date in the filename
for (var i = 0; i < n; i++) {
  // Get a single image
  var image = ee.Image(listOfImages.get(i));
  
  // Retrieve and format the acquisition date from the image properties
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo();
  
  // Export the VV band
  Export.image.toDrive({
    image: image.select('VV_linear_spk'),
    description: 'S1_xianxing_VV_yuhang_' + date,
    folder: 'SAR_yuhang_xianxing',
    fileNamePrefix: 'S1_xianxing_VV_yuhang_' + date,
    region: StudyArea.geometry(),
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
  
  // Export the VH band
  Export.image.toDrive({
    image: image.select('VH_linear_spk'),
    description: 'S1_xianxing_VH_yuhang_' + date,
    folder: 'SAR_yuhang_xianxing',
    fileNamePrefix: 'S1_xianxing_VH_yuhang_' + date,
    region: StudyArea.geometry(),
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}
```

