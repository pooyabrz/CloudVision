# Dataset card

**cloudcast-gridsat-europe-2017-v1** contains 252 collocated scenes over Europe during 2017. Every month contributes observations on the 3rd, 7th, 11th, 15th, 19th, 23rd and 27th at 00, 09 and 18 UTC. Each scene contains a 128 × 128 float32 grid of brightness temperatures and a uint8 grid of reference classes. Each of the 84 UTC days appears in exactly one partition.

CloudCast distributes a derived cloud-type product originally developed for forecasting. Its semantic masks serve as the training targets. Using these class codes as predictor channels would reveal the answer. CloudVision pairs them with a separate physical observation, NOAA GridSat-B1 infrared-window brightness temperature (`irwin_cdr`). The two products have different footprints and sensor processing. [CloudCast](https://vision.eng.au.dk/cloudcast-dataset/) · [Author tutorial](https://github.com/holmdk/CloudCast_Dataset/blob/main/CloudCast_Tutorial.ipynb) · [GridSat-B1 climate data record](https://doi.org/10.7289/V59P2ZKR)

## Pair construction

The acquisition uses the uncollapsed 768 × 768 cloud-type arrays in Christian Lillelund's mirror. The monthly timestamps are minutes since the start of that month. The code selects the closest available observation within 7.5 minutes of the scheduled UTC time, verifies the NOAA time and records the signed difference. It samples native centers 3, 9, …, 765 along each axis. This release's center sampling is derived from the 768 grid and differs from the separately distributed small CloudCast product. [CloudCast mirror](https://www.kaggle.com/datasets/christianlillelund/the-cloudcast-dataset)

The mirror's GEO arrays named `lats` and `lons` contain projected x and y axes in metres respectively. The polar stereographic projection uses longitude of origin 5° E, latitude of true scale 50° N, latitude of origin 90° N and WGS84. Transforming the grid to geographic coordinates and sampling the nearest NOAA grid center preserves categorical geometry. Native CloudCast spacing is approximately 3 km, giving approximately 18 km sample spacing here. The NOAA packed infrared scale and offset are 0.01 K and 200 K. Raw fill and values outside 150–350 K are missing. Invalid infrared forces the paired target to ignore.

As an independent orientation check, the January 1 00 UTC clear-land/clear-sea reference agreed with Natural Earth land polygons on 3,926 of 3,966 pixels (98.99%). Transposing the axes reduced agreement to 78.52%. The check addresses grid orientation and coastline agreement for clear pixels. Moving or elevated clouds can still have a position offset. Separately, all 16,384 temperature samples on 3 January 2017 at 00 UTC matched an independent center lookup in the complete NOAA NetCDF file, with maximum absolute difference 0.0 K. [Independent audit](../experiments/alignment.json)

## Taxonomy

| Raw CloudCast | Exported mask | Reference class |
|---:|---:|---|
| 1 | 0 | Clear land |
| 2 | 1 | Clear sea |
| 3 | 2 | Snow over land |
| 4 | 3 | Sea ice |
| 5 | 4 | Very low cloud |
| 6 | 5 | Low cloud |
| 7 | 6 | Mid-level cloud |
| 8 | 7 | High opaque cloud |
| 9 | 8 | Very high opaque cloud |
| 10 | 9 | Fractional cloud |
| 11 | 10 | Thin semitransparent cloud |
| 12 | 11 | Moderate semitransparent cloud |
| 13 | 12 | Thick semitransparent cloud |
| 14 | 13 | Semitransparent above low or mid cloud |
| 0, 255 or missing | 255 | Unprocessed or missing |
 
The CNN's eleven classes consist of ‘No cloud’ plus the ten cloud types. The target uses the most frequent cloud among valid cloudy pixels, even when clear surfaces occupy most of a tile. A tile with valid observations and no cloud receives class zero. A fully ignored tile is excluded. Ties use the smallest code.

## Temporal partitions

| Split | Dates in 2017 (inclusive) | UTC days | Scenes | 64 × 64 tiles |
|---|---|---:|---:|---:|
| Training | Jan 3–Sep 11 | 59 | 177 | 708 |
| Validation | Sep 15–Nov 3 | 12 | 36 | 144 |
| Test | Nov 7–Dec 27 | 13 | 39 | 156 |

A 24-hour embargo is enforced at both boundaries. The actual gaps are 78 hours. Partitioning precedes tiling and augmentation. The audit checks shape, value ranges, paired valid support, duplicate times, duplicate radiometry and per-scene array hashes. Twelve training pixels have ignored targets. The 638,976 test pixels all have valid labels. Training has just 115 sea-ice pixels and the test has zero. There are 594,203 clear-land training pixels. Class weights use only training counts. [Frequency counts by split](../experiments/data_audit.json)

## Importing other data

For projected NetCDF files already sharing the same spatial grid, use the native importer. It verifies time and matching spatial coordinates, dimensions and units, and compares declared grid-mapping metadata. Choose source variables explicitly and use uncollapsed raw labels.

```bash
python -m scripts.import_netcdf labels.nc infrared.nc data/custom ct irwin_cdr my-collection-v1
```

Inputs need a CF-readable time axis, two spatial coordinate axes and a declared Kelvin image variable. Optional `--tolerance-minutes` allows a 0–15 minute offset, with a unique input per label. Use the acquisition script to reproject the specific CloudCast/GridSat pair. The generic importer expects coordinates already matched. Array hash and source configuration identify each processed collection.
