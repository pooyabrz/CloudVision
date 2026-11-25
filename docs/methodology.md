# Technical methodology

## Study question

How accurately can physically calibrated single-band infrared observations recover CloudCast semantic cloud labels on later European observations? A second task classifies the dominant cloud in each tile. The benchmark separates image-level and pixel-level targets throughout.

## Inputs and sampling

CloudCast class fields provide references and independent GridSat-B1 measurements provide predictors. The collection is sampled across an entire calendar year. The 128 × 128 scenes are split chronologically by UTC date, then divided into 64 × 64 tiles. All selections use validation scores. The later November–December test observations are opened for the frozen comparison. The [data card](data-card.md) gives the projection, grid validation, label map, sampling, missing-value conventions and provenance.

Both training and serving scale the infrared channel by (K − 265) / 25. Missing measurements receive normalized value zero and a second binary channel identifies valid pixels. These fixed physical constants require no fitting. Training-only augmentation uses a horizontal flip with probability 0.5 and a spatial crop with probability 0.2. A shared crop factor in [0.8, 1] is resized with nearest-neighbor interpolation on both bands and masks. Optional vertical flips and quarter turns are configurable. They were disabled for this regional benchmark. No intensity jitter is applied.

## Optimization and selection

All architectures start from seeded random initialization. The CNN uses three convolution/pooling stages and an adaptive-pooling linear classifier. U-Net has two skip-connected spatial scales and a bridge. The residual variant widens the backbone and adds projected residual connections. SegFormer uses four hierarchical overlapping patch embeddings, spatial-reduction attention, mixed depthwise feed-forward layers and a multiscale projection decoder. It is a compact implementation trained from scratch.

Eight runs compare the CNN, the three segmentation architectures and four additional U-Net variants. The ablations isolate plain cross-entropy, inverse-square-root frequency-weighted cross-entropy, Dice alone, and disabling augmentation. The default joint objective assigns 75% weight to weighted cross-entropy and 25% to soft Dice over classes with batch support. Ignored targets do not contribute. Optimizer state, scheduler, gradient scaler and history are saved after each epoch. Seeding uses an explicit epoch-dependent seed and sampler generator so a recovered run repeats the batch sequence.

The release uses seed 42, AdamW, learning rate 0.0015, batch size 24, weight decay 0.0002 and gradient-norm clipping at 2. It permits up to 12 epochs and stops after four consecutive validation checks without improvement. The plateau scheduler reduces the learning rate when validation stalls. AMP is enabled on CUDA-capable configurations. Training and timing in this release used CPU. Each architecture contributes its best validation checkpoint across its runs. Image classification selects by macro F1. Segmentation selects by mIoU.

## Scoring and error attribution

Confusion matrices pool all valid held-out observations. Reference classes form rows, predicted classes columns. Macro scores average class scores with nonempty union. A class with references and no predictions scores zero. A class absent on both sides has an undefined score and is omitted from that average. Every table includes support. Accuracy is weighted by the number of valid observations. Mean Dice and macro F1 use the same overlap formula within a task, while image and pixel scores measure different outcomes.

Per-month results quantify variation on the two test months. A boundary pixel has a valid four-neighbor of a different reference class. A small cloud is a four-connected same-class cloud component of 16 pixels or fewer on a 64 × 64 evaluation tile. The false-cloud rate is the share of reference clear-land, clear-sea, snow or ice pixels predicted as any cloud. The qualitative triplets use the lowest, median and highest accuracy tiles for each segmenter, with their UTC scene and pixel offsets recorded in the benchmark.

Temperature scaling minimizes validation negative log-likelihood over one positive scalar in [0.25, 8]. The validation routine uses all classification tiles and a deterministic strided sample of up to 8,192 valid segmentation pixels per validation batch. A worse fitted loss falls back to temperature one. Held-out calibration uses every seventeenth valid segmentation pixel per tile and all CNN examples. ECE uses 15 equal-width confidence bins weighted by observed frequency. Entropy is divided by log(number of classes), giving a 0–1 scale. Monte Carlo inference activates dropout alone, averages probabilities and restores evaluation mode. CNN Grad-CAM weights final convolutional feature maps by the target logit's spatially averaged gradients.

## Timing and reproducibility

Latency measures batch-one 64 × 64 forward passes with eight warm-ups and 60 timed runs. Median and p95 exclude decoding, HTTP, exports and model load. CUDA synchronization and peak-memory collection are enabled on GPU. Model size is the on-disk checkpoint byte count. Training seconds cover the epoch training and validation loops. The comparison records hardware, device, threads, Python and PyTorch version. Exact timing varies with hardware and load. Split, model configuration, checkpoint and dataset fingerprints are available in the release.

## References

- Nielsen, Iosifidis & Karstoft (2021). CloudCast: A Satellite-Based Dataset and Baseline for Forecasting Clouds. [DOI](https://doi.org/10.1109/JSTARS.2021.3062936)
- Knapp et al. NOAA Climate Data Record of Gridded Satellite Data from ISCCP B1, Version 2. [Dataset DOI](https://doi.org/10.7289/V59P2ZKR)
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. [Paper](https://arxiv.org/abs/1505.04597)
- Xie et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. [Paper](https://arxiv.org/abs/2105.15203)
- Guo et al. (2017). On Calibration of Modern Neural Networks. [Paper](https://arxiv.org/abs/1706.04599)
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. [Paper](https://arxiv.org/abs/1610.02391)
