# Release verification

The delivered benchmark uses Python 3.12.14 and PyTorch 2.8.0+cpu. All 40 automated tests passed. The cases cover supported and malformed input formats, nodata propagation, time embargoes, paired augmentation, NetCDF alignment, all four architectures, loss masking, deterministic checkpoint recovery, calibration, metrics, model-file verification, REST errors and Streamlit navigation. The pinned development environment also passed dependency consistency checks.

A live browser uploaded a real 128 × 128 infrared PNG, ran the default model through the independent HTTP service, opened the pixel inspector, viewed uncertainty and model comparison, then downloaded the raw PNG mask. The downloaded image retained its original dimensions and identified the expected model in the embedded metadata. A 390 × 844 mobile viewport completed an example prediction and displayed the page within its width. The captioned demonstration shows the live desktop workflow.

All four notebooks executed against the bundled data and models. The eight training histories and best checkpoints were imported into MLflow; repeating the import retained eight entries. The alignment audit checked an independent coastline reference and compared one complete NOAA source file with every packaged infrared pixel for its scene.

Docker Compose v2.39.4 validated the configuration. The API and dashboard ran together as native processes. A Docker Engine was unavailable in the verification environment. The supplied CI job builds the image, starts both containers, checks their health endpoints and shuts them down on a Docker-capable runner.
