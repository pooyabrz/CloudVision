"""Verified cached checkpoints, bounded native tiling, uncertainty and self-describing exports."""

import base64
import csv
import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torch import nn
from torch.nn import functional as F

from src.models import build
from src.schema import (
    COLORS,
    IMAGE_NAMES,
    NAMES,
    ROOT,
    color_mask,
    dominant,
    kelvin,
    preview,
    sha,
    tensor,
)


def default_model(entries):
    candidates = [e for e in entries if e["task"] == "segmentation"]
    if not candidates:
        raise RuntimeError("No segmentation model is available.")
    return max(candidates, key=lambda e: e["best_validation_score"])["architecture"]


class Registry:
    def __init__(self, directory=None, device=None):
        self.folder = Path(
            directory or os.environ.get("CLOUDVISION_CHECKPOINTS", ROOT / "checkpoints")
        )
        self.device = torch.device(device or os.environ.get("CLOUDVISION_DEVICE", "cpu"))
        torch.set_num_threads(int(os.environ.get("CLOUDVISION_THREADS", "3")))
        self.cache = {}
        self.lock = threading.RLock()

    def catalog(self):
        try:
            return json.loads((self.folder / "registry.json").read_text())
        except (OSError, ValueError):
            raise RuntimeError("The model registry is unavailable.") from None

    def get(self, name=None):
        with self.lock:
            entries = self.catalog()
            name = name if name is not None else default_model(entries.values())
            if name not in entries:
                raise KeyError(name)
            e = entries[name]
            key = (name, e["sha256"])
            if key not in self.cache:
                try:
                    path = self.folder / e["file"]
                    if sha(path) != e["sha256"]:
                        raise ValueError()
                    model = build(name, e["model_config"])
                    model.load_state_dict(
                        torch.load(path, weights_only=True, map_location="cpu")["weights"]
                    )
                    model.to(self.device).eval()
                except (OSError, ValueError, RuntimeError, KeyError, EOFError):
                    raise RuntimeError("Checkpoint is unavailable.") from None
                self.cache[key] = (model, threading.RLock())
            model, lock = self.cache[key]
            return model, e, lock

    @staticmethod
    def segment(model, x, tile, temp, mc):
        _, h, w = x.shape
        xx = F.pad(x, (0, (-w) % tile, 0, (-h) % tile))
        _, hh, ww = xx.shape
        spots = [(i, j) for i in range(0, hh, tile) for j in range(0, ww, tile)]
        out = np.empty((len(NAMES), hh, ww), np.float32)
        for start in range(0, len(spots), 16):
            batch = spots[start : start + 16]
            z = torch.stack([xx[:, y : y + tile, x : x + tile] for y, x in batch])
            p = (sum((model(z) / temp).softmax(1) for _ in range(mc)) / mc).cpu().numpy()
            for (y, x), sub in zip(batch, p):
                out[:, y : y + tile, x : x + tile] = sub
        return out[:, :h, :w]

    def predict(self, a, name=None, mc=1):
        if not 1 <= mc <= 12:
            raise ValueError("Use 1–12 dropout passes.")
        a = kelvin(a)
        model, e, lock = self.get(name)
        x = tensor(a, e["preprocessing"]).to(self.device)
        t = e["tile"]
        temp = e["calibration"]["temperature"]
        cls = e["task"] == "classification"
        cam = None
        with lock:
            try:
                if mc > 1:
                    for layer in model.modules():
                        if isinstance(layer, (nn.Dropout, nn.Dropout2d)):
                            layer.train()
                with torch.inference_mode():
                    if cls:
                        z = F.interpolate(x[None], (t, t), mode="bilinear", align_corners=False)
                        p = (
                            (sum((model(z) / temp).softmax(1) for _ in range(mc)) / mc)[0]
                            .cpu()
                            .numpy()
                        )
                    else:
                        p = self.segment(model, x, t, temp, mc)
            finally:
                model.eval()
            if cls:
                with torch.enable_grad():
                    z = F.interpolate(x[None], (t, t), mode="bilinear", align_corners=False)
                    f = model.features(z)
                    (g,) = torch.autograd.grad(model.head(f)[0, int(p.argmax())], f)
                    c = (f * g.mean((-2, -1), keepdim=True)).sum(1, keepdim=True).relu()
                    c = F.interpolate(c, a.shape, mode="bilinear", align_corners=False)[0, 0]
                    cam = (c / c.max().clamp_min(1e-8)).detach().cpu().numpy()
                    cam[~np.isfinite(a)] = 0
        return p, cam, e

    def predict_batch(self, images, name=None, mc=1):
        for a in images:
            yield self.predict(a, name, mc)


def png(a, tags):
    f = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("cloudvision", json.dumps(tags, separators=(",", ":")))
    Image.fromarray(a).save(f, format="PNG", pnginfo=info)
    return base64.b64encode(f.getvalue()).decode()


def unit_map(a, valid):
    return np.where(valid, np.rint(np.clip(a, 0, 1) * 65534) + 1, 0).astype(np.uint16)


def summary(p, v):
    m = p.argmax(0).astype(np.uint8)
    m[~v] = 255
    c = p.max(0)
    u = -(p * np.log(p.clip(1e-12))).sum(0) / np.log(len(p))
    counts = np.bincount(m[v], minlength=len(NAMES))
    n = int(v.sum())
    rows = [
        dict(
            id=i,
            name=name,
            color=COLORS[i],
            cloud=i >= 4,
            pixels=int(counts[i]),
            coverage_pct=100 * int(counts[i]) / n,
            probability=float(p[i, v].mean()),
            mean_confidence=float(c[m == i].mean()) if counts[i] else None,
        )
        for i, name in enumerate(NAMES)
    ]
    d = dominant(m)
    i = d + 3 if d else None
    s = dict(
        dominant=NAMES[i] if i is not None else "No cloud",
        dominant_confidence=rows[i]["mean_confidence"] if i is not None else float(c[v].mean()),
        cloud_coverage_pct=float(100 * counts[4:].sum() / n),
        noncloud_coverage_pct=float(100 * counts[:4].sum() / n),
        mean_confidence=float(c[v].mean()),
        maximum_probability=float(c[v].max()),
        mean_entropy=float(u[v].mean()),
    )
    return m, c, u, rows, s


def format_result(a, p, cam, e, input_digest, mc=1, include_maps=True):
    valid = np.isfinite(a)
    gray = preview(a)
    meta = dict(
        model=e["architecture"],
        version=e["version"],
        task=e["task"],
        dataset=e["dataset"],
        checkpoint_sha256=e["sha256"],
        input_sha256=input_digest,
        input_shape=list(a.shape),
        units="K",
        valid_pixels=int(valid.sum()),
        ignored_pixels=int((~valid).sum()),
        preprocessing=e["preprocessing"],
        model_config=e["model_config"],
        temperature=e["calibration"]["temperature"],
        mc_passes=mc,
        tile=e["tile"],
        generated_utc=datetime.now(timezone.utc).isoformat(),
        transform="whole frame bilinear-resized to train tile"
        if cam is not None
        else "native non-overlapping tiles, right/bottom nodata padding then trim",
        continuous_png="0 = nodata; (code-1)/65534 otherwise",
        mask_encoding="0..13 = taxonomy order, 255 = nodata",
    )
    images = {"original": png(gray, meta)}
    if cam is not None:
        rows = [dict(id=i, name=n, probability=float(p[i])) for i, n in enumerate(IMAGE_NAMES)]
        s = dict(
            dominant=IMAGE_NAMES[int(p.argmax())],
            dominant_confidence=float(p.max()),
            maximum_probability=float(p.max()),
            mean_entropy=float(-(p * np.log(p.clip(1e-12))).sum() / np.log(len(p))),
        )
        images["gradcam"] = png(unit_map(cam, valid), meta)
    else:
        m, c, u, rows, s = summary(p, valid)
        colors = color_mask(m)
        overlay = np.where(valid[:, :, None], (0.45 * gray + 0.55 * colors).round(), gray).astype(
            np.uint8
        )
        images.update(
            mask=png(m, meta),
            segmentation=png(colors, meta),
            overlay=png(overlay, meta),
            confidence=png(unit_map(c, valid), meta),
            uncertainty=png(unit_map(u, valid), meta),
        )
        if include_maps:
            images["probabilities"] = [png(unit_map(v, valid), meta) for v in p]
    fp = io.StringIO()
    assets = dict(
        model=e["architecture"],
        version=e["version"],
        input_sha256=input_digest,
        preprocessing_version=e["preprocessing"]["version"],
        input_height=a.shape[0],
        input_width=a.shape[1],
    )
    w = csv.DictWriter(fp, fieldnames=[*assets, *rows[0]])
    w.writeheader()
    for row in rows:
        w.writerow({**assets, **row})
    return dict(metadata=meta, summary=s, classes=rows, images=images, statistics_csv=fp.getvalue())
