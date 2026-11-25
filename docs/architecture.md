# Architecture and serving

| Component | Responsibility |
|---|---|
| scripts/fetch_pairs.py, scripts/import_netcdf.py | Acquire, align, validate and atomically publish observation collections |
| src/schema.py, src/data.py | Versioned taxonomy and input validation, temporal splits, checksums, joint transforms and tile sampling |
| src/models.py, src/train.py | Four architectures, configurable training, recoverable optimizer state, validation and calibration |
| src/evaluate.py, src/metrics.py | Frozen held-out measurements, per-month and per-class reporting, spatial error analysis and benchmark figures |
| src/inference.py | Verify and cache weights, tile input, average dropout samples, compute calibrated probabilities, Grad-CAM and exports |
| api/main.py | Bound request sizes, decode supported inputs, run inference and return documented HTTP responses |
| app/dashboard.py | Browser-facing Streamlit workspace. Prediction and benchmark data are requested through the API |

## Model lifecycle

Each process reads `checkpoints/registry.json`. A per-checkpoint lock protects first load, model mode changes and Grad-CAM. The loader verifies SHA-256 before loading tensors in PyTorch's weights-only mode. An unknown model returns 404, malformed observations return 422, excessive uploads return 413 and unavailable model assets return 503. The container runs as a non-root user. Uploaded files are processed in memory and the UI keeps the latest result in browser-session state.

Set `CLOUDVISION_CHECKPOINTS` and `CLOUDVISION_METRICS` on the API to serve a new frozen run. Set `CLOUDVISION_DEVICE=cpu` or `cuda` and `CLOUDVISION_THREADS` to choose compute. Set `CLOUDVISION_API` on the dashboard to its reachable API base URL. A future published project can set `CLOUDVISION_REPO_URL` to enable the optional project link. Compose binds both ports to localhost.

## Inference geometry

The segmenters apply 64 × 64 non-overlapping tiles, with missing input padding on the right and bottom. Predictions are assembled and trimmed to original height and width. Padding lies outside the reporting denominator. Native tiles are processed in batches of up to 16. A per-image class-probability tensor is retained when computing statistics. The CNN resizes the whole frame to 64 × 64 with bilinear interpolation, then rescales its Grad-CAM to input dimensions. Large CNN inputs consequently have a different effective scale from the training tiles.

```python
from pathlib import Path
from src.inference import Registry
from src.schema import decode

registry = Registry()
paths = sorted(Path("examples").glob("*.png"))
images = (decode(p.read_bytes(), p.name) for p in paths)
for probabilities, gradcam, entry in registry.predict_batch(images, "unet_plus"):
    print(entry["version"], probabilities.shape)
```

The dashboard exposes original and overlay comparison, per-pixel and per-class probability, color legend, predicted coverage, entropy and a downloadable raw class-index mask. The classifier returns per-image probabilities. The PNG overlay blends 45% grayscale intensity and 55% assigned class color.
