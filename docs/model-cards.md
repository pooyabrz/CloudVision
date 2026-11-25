# Model cards

These models support research on calibrated European satellite infrared data. All four use the same fixed Kelvin normalization and validity channel. See the [data card](data-card.md) for class definitions and the [methodology](methodology.md) for selection and scoring.

## CNN baseline

Three convolution, batch-normalization, ReLU and pooling stages; global pooling and an eleven-class head.

**Version** `25288f257b24` · **Run** `cnn` · **Parameters** 19,451 · **Checkpoint** 85,657 bytes.

Selected at epoch 9 on validation macro_f1 of 0.325. Test accuracy 0.423, macro precision 0.304, macro recall 0.311 and macro F1 0.277.

Validation temperature 0.613. ECE before/after 0.173 / 0.081. Held-out ECE 0.166.

 | Class | Reference support | Predicted | Precision | Recall | F1 | 
 | --- | --- | --- | --- | --- | --- | 
 | No cloud | 0 | 0 | — | — | — | 
 | Very low cloud | 45 | 37 | 0.568 | 0.467 | 0.512 | 
 | Low cloud | 22 | 26 | 0.154 | 0.182 | 0.167 | 
 | Mid-level cloud | 18 | 39 | 0.385 | 0.833 | 0.526 | 
 | High opaque cloud | 23 | 37 | 0.405 | 0.652 | 0.500 | 
 | Very high opaque cloud | 1 | 0 | 0.000 | 0.000 | 0.000 | 
 | Fractional cloud | 4 | 0 | 0.000 | 0.000 | 0.000 | 
 | Thin semitransparent cloud | 0 | 0 | — | — | — | 
 | Moderate semitransparent cloud | 12 | 5 | 0.000 | 0.000 | 0.000 | 
 | Thick semitransparent cloud | 31 | 12 | 0.917 | 0.355 | 0.512 | 
 | Semitransparent above low or mid cloud | 0 | 0 | — | — | — | 

Grad-CAM highlights spatial evidence for the chosen image label. An entirely clear tile is rare or absent on this sampled benchmark. Whole-frame resizing changes the physical scale.

## SegFormer

Four overlapping patch stages, spatial-reduction attention, depthwise Mix-FFNs and a multiscale decoder. Trained from scratch.

**Version** `2053c043d1e4` · **Run** `segformer` · **Parameters** 140,198 · **Checkpoint** 591,825 bytes.

Selected at epoch 11 on validation miou of 0.188. Test accuracy 0.351, macro precision 0.293, macro recall 0.303 and macro F1 0.270. Test mIoU 0.168, mean Dice 0.270 and derived dominant-tile macro F1 0.310.

Validation temperature 0.810. ECE before/after 0.060 / 0.011. Held-out ECE 0.081.

 | Class | Reference support | Predicted | Precision | Recall | IoU | Dice | 
 | --- | --- | --- | --- | --- | --- | --- | 
 | Clear land | 92675 | 11905 | 0.445 | 0.057 | 0.053 | 0.101 | 
 | Clear sea | 26634 | 47068 | 0.355 | 0.627 | 0.293 | 0.453 | 
 | Snow over land | 2760 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 
 | Sea ice | 0 | 0 | — | — | — | — | 
 | Very low cloud | 93036 | 56329 | 0.423 | 0.256 | 0.190 | 0.319 | 
 | Low cloud | 68950 | 147983 | 0.255 | 0.547 | 0.210 | 0.348 | 
 | Mid-level cloud | 66448 | 107729 | 0.351 | 0.569 | 0.277 | 0.434 | 
 | High opaque cloud | 69560 | 107083 | 0.415 | 0.639 | 0.336 | 0.503 | 
 | Very high opaque cloud | 11058 | 11614 | 0.393 | 0.413 | 0.252 | 0.403 | 
 | Fractional cloud | 44596 | 34595 | 0.189 | 0.147 | 0.090 | 0.165 | 
 | Thin semitransparent cloud | 19497 | 3601 | 0.148 | 0.027 | 0.024 | 0.046 | 
 | Moderate semitransparent cloud | 56396 | 38842 | 0.260 | 0.179 | 0.119 | 0.212 | 
 | Thick semitransparent cloud | 78050 | 69072 | 0.526 | 0.465 | 0.328 | 0.494 | 
 | Semitransparent above low or mid cloud | 9316 | 3155 | 0.054 | 0.018 | 0.014 | 0.027 | 

False-cloud rate 73.6%; small-cloud pixel error 75.5%. Classes absent from both reference and prediction have undefined overlap. See the per-class support before interpreting a zero or missing score.

## U-Net

Two encoder levels and a bridge, skip concatenation, bilinear upsampling and a pixel classifier.

**Version** `362cb2a2d996` · **Run** `unet_no_augmentation` · **Parameters** 66,974 · **Checkpoint** 288,285 bytes.

Selected at epoch 11 on validation miou of 0.209. Test accuracy 0.362, macro precision 0.301, macro recall 0.339 and macro F1 0.283. Test mIoU 0.176, mean Dice 0.283 and derived dominant-tile macro F1 0.339.

Validation temperature 0.699. ECE before/after 0.080 / 0.020. Held-out ECE 0.128.

 | Class | Reference support | Predicted | Precision | Recall | IoU | Dice | 
 | --- | --- | --- | --- | --- | --- | --- | 
 | Clear land | 92675 | 10577 | 0.247 | 0.028 | 0.026 | 0.051 | 
 | Clear sea | 26634 | 11601 | 0.486 | 0.212 | 0.173 | 0.295 | 
 | Snow over land | 2760 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 
 | Sea ice | 0 | 0 | — | — | — | — | 
 | Very low cloud | 93036 | 102356 | 0.363 | 0.400 | 0.235 | 0.381 | 
 | Low cloud | 68950 | 109668 | 0.291 | 0.464 | 0.218 | 0.358 | 
 | Mid-level cloud | 66448 | 84296 | 0.444 | 0.564 | 0.331 | 0.497 | 
 | High opaque cloud | 69560 | 96772 | 0.465 | 0.647 | 0.371 | 0.541 | 
 | Very high opaque cloud | 11058 | 42328 | 0.249 | 0.952 | 0.246 | 0.394 | 
 | Fractional cloud | 44596 | 38838 | 0.253 | 0.220 | 0.133 | 0.235 | 
 | Thin semitransparent cloud | 19497 | 26645 | 0.145 | 0.199 | 0.092 | 0.168 | 
 | Moderate semitransparent cloud | 56396 | 74541 | 0.319 | 0.421 | 0.222 | 0.363 | 
 | Thick semitransparent cloud | 78050 | 41223 | 0.572 | 0.302 | 0.246 | 0.395 | 
 | Semitransparent above low or mid cloud | 9316 | 131 | 0.084 | 0.001 | 0.001 | 0.002 | 

False-cloud rate 87.8%; small-cloud pixel error 68.8%. Classes absent from both reference and prediction have undefined overlap. See the per-class support before interpreting a zero or missing score.

## Residual U-Net

A widened U-Net with learned residual projections, batch normalization, skip connections and bottleneck dropout.

**Version** `215cafe4cfb1` · **Run** `unet_plus` · **Parameters** 125,102 · **Checkpoint** 522,111 bytes.

Selected at epoch 9 on validation miou of 0.237. Test accuracy 0.393, macro precision 0.328, macro recall 0.370 and macro F1 0.323. Test mIoU 0.206, mean Dice 0.323 and derived dominant-tile macro F1 0.298.

Validation temperature 0.801. ECE before/after 0.056 / 0.009. Held-out ECE 0.077.

 | Class | Reference support | Predicted | Precision | Recall | IoU | Dice | 
 | --- | --- | --- | --- | --- | --- | --- | 
 | Clear land | 92675 | 23300 | 0.429 | 0.108 | 0.094 | 0.173 | 
 | Clear sea | 26634 | 21708 | 0.558 | 0.455 | 0.335 | 0.501 | 
 | Snow over land | 2760 | 1214 | 0.002 | 0.001 | 0.001 | 0.002 | 
 | Sea ice | 0 | 0 | — | — | — | — | 
 | Very low cloud | 93036 | 89835 | 0.396 | 0.383 | 0.242 | 0.389 | 
 | Low cloud | 68950 | 103110 | 0.340 | 0.508 | 0.256 | 0.407 | 
 | Mid-level cloud | 66448 | 87643 | 0.453 | 0.597 | 0.347 | 0.515 | 
 | High opaque cloud | 69560 | 97929 | 0.495 | 0.698 | 0.408 | 0.579 | 
 | Very high opaque cloud | 11058 | 29857 | 0.334 | 0.901 | 0.322 | 0.487 | 
 | Fractional cloud | 44596 | 52005 | 0.266 | 0.310 | 0.167 | 0.286 | 
 | Thin semitransparent cloud | 19497 | 21159 | 0.168 | 0.182 | 0.096 | 0.175 | 
 | Moderate semitransparent cloud | 56396 | 75491 | 0.330 | 0.442 | 0.233 | 0.378 | 
 | Thick semitransparent cloud | 78050 | 35722 | 0.495 | 0.226 | 0.184 | 0.311 | 
 | Semitransparent above low or mid cloud | 9316 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 

False-cloud rate 74.9%; small-cloud pixel error 67.1%. Classes absent from both reference and prediction have undefined overlap. See the per-class support before interpreting a zero or missing score.
