# Comparative experiments

Residual U-Net ranked first on validation mIoU (0.237) and achieved 0.206 mIoU and 0.323 mean Dice on the held-out test. The test covers 39 later scenes and 156 tiles.

 | Model | Task | Accuracy | Image macro F1 | mIoU | Mean Dice | Median ms | p95 ms | MiB | Train s | 
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | CNN baseline | classification | 0.423 | 0.277 | — | — | 1.06 | 1.31 | 0.082 | 9.6 | 
 | SegFormer | segmentation | 0.351 | — | 0.168 | 0.270 | 3.03 | 4.17 | 0.564 | 20.3 | 
 | U-Net | segmentation | 0.362 | — | 0.176 | 0.283 | 4.70 | 5.79 | 0.275 | 56.9 | 
 | Residual U-Net | segmentation | 0.393 | — | 0.206 | 0.323 | 6.14 | 18.91 | 0.498 | 66.0 | 

The CNN scores individual tile labels. Segmentation accuracy counts pixels. Each architecture appears with its strongest validation run. Model selection and temperature fitting use validation data.

## Eight validation runs

 | Run | Loss | Augmentation | Best epoch | Validation metric | Best score | 
 | --- | --- | --- | --- | --- | --- | 
 | cnn | weighted_ce | on | 9 | macro_f1 | 0.3250 | 
 | segformer | combined | on | 11 | miou | 0.1884 | 
 | unet | combined | on | 11 | miou | 0.2012 | 
 | unet_ce | ce | on | 12 | miou | 0.1949 | 
 | unet_dice | dice | on | 12 | miou | 0.1832 | 
 | unet_no_augmentation | combined | off | 11 | miou | 0.2095 | 
 | unet_plus | combined | on | 9 | miou | 0.2373 | 
 | unet_weighted_ce | weighted_ce | on | 11 | miou | 0.2002 | 

## Temporal and spatial errors

The selected model assigned a cloud class to 74.9% of 122,069 labeled surface pixels. It misclassified 64.3% of boundary pixels and 67.1% of small-cloud pixels.

 | Test month | Valid pixels | Pixel accuracy | mIoU | 
 | --- | --- | --- | --- | 
 | 2017-11 | 294,912 | 0.407 | 0.210 | 
 | 2017-12 | 344,064 | 0.380 | 0.198 | 

 | Reference | Predicted | Pixels | 
 | --- | --- | --- | 
 | Thick semitransparent cloud | High opaque cloud | 38,480 | 
 | Clear land | Very low cloud | 37,869 | 
 | Very low cloud | Low cloud | 24,457 | 

See the lowest, median and highest accuracy tiles for each of the three segmenters in [the figures directory](../experiments/figures/).

## Calibration and compute

Fitting a temperature of 0.801 changed validation ECE from 0.056 to 0.009. Test ECE was 0.077.

AMD EPYC 9V74 80-Core Processor, 3 threads, cpu. Timing covers one 64 × 64 forward pass. Peak GPU allocation is unmeasured for this CPU run. All counts, calibration bins, confusion matrices and fingerprints are in [comparison.json](../experiments/comparison.json).

## Metric leaders

 | Metric | Model | 
 | --- | --- | 
 | segmentation miou | Residual U-Net | 
 | segmentation accuracy | Residual U-Net | 
 | segmentation mean dice | Residual U-Net | 
 | classification macro f1 | CNN baseline | 
 | fastest forward | CNN baseline | 
 | smallest | CNN baseline | 
 | shortest training | CNN baseline | 
