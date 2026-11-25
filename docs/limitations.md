# Research limitations

The selected model assigned a cloud class to 74.9% of 122,069 labeled surface pixels. It misclassified 64.3% of boundary pixels and 67.1% of small-cloud pixels.

CloudCast is a retrieved reference product and GridSat is an independent observation. Residual timing differences, parallax, mixed pixels, sensor footprints and reference errors can all affect agreement. The land-mask check supports native grid orientation.

Sampling covers 84 dates in one European year and three UTC hours per sampled date. The test covers November and December 2017. Seasons and chronology change together. Tiles from the same scene are correlated. Single-seed, single-region estimates have no confidence intervals here. Cross-year and geographic generalization remain unmeasured.

The single calibrated infrared band carries limited information about cloud height, optical thickness and surface type. Thin cloud and snow often have weak separation. Sea ice has zero reference pixels in the test, and the model recovered no pixels of the semitransparent-above-low/mid class. Confidence calibration can shift in a new season.

Small components crossing tile edges can be split by the component analysis. Disjoint inference tiles can introduce seams. A CNN applied to a larger frame has a coarser effective input scale. Percentages measure the fraction of observed projected-grid pixels and do not correct for geographic area.

For research and demonstration, inspect the corresponding reference, class support, calibration and failure examples before interpreting a scene's estimated cloud coverage. Operational weather decisions require further validation. A separate GPU run is needed for accelerator latency, memory and concurrency measurements.

Forecasting and continuous ingestion are future extensions in the PRD. The release analyzes individual observations.
