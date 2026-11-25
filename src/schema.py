"""Versioned input and label contracts shared across training and serving."""

import hashlib
import io
import json
import os
import struct
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "Clear land",
    "Clear sea",
    "Snow over land",
    "Sea ice",
    "Very low cloud",
    "Low cloud",
    "Mid-level cloud",
    "High opaque cloud",
    "Very high opaque cloud",
    "Fractional cloud",
    "Thin semitransparent cloud",
    "Moderate semitransparent cloud",
    "Thick semitransparent cloud",
    "Semitransparent above low or mid cloud",
)
IMAGE_NAMES = ("No cloud",) + NAMES[4:]
COLORS = (
    "#9eae69",
    "#2b5975",
    "#f4efe3",
    "#a4d5dc",
    "#edbb68",
    "#ed8f50",
    "#cc655b",
    "#aa637f",
    "#795d90",
    "#c6a385",
    "#74d1c6",
    "#4fa8be",
    "#758fd4",
    "#b4a1dc",
)
RGB = np.array([tuple(bytes.fromhex(x[1:])) for x in COLORS], np.uint8)
MODELS = {
    "cnn": "CNN baseline",
    "unet": "U-Net",
    "unet_plus": "Residual U-Net",
    "segformer": "SegFormer",
}
IGNORE = 255
MAX_BYTES = 8 * 1024**2


def config(name):
    import yaml

    return yaml.safe_load((ROOT / "configs" / (name + ".yaml")).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def encode_raw(raw):
    a = np.ma.asarray(raw, dtype=np.float32).filled(np.nan)
    if np.any(~np.isnan(a) & ~np.isin(a, [*range(15), 255])):
        raise ValueError("Invalid raw cloud code.")
    valid = (a >= 1) & (a <= 14)
    m = np.full(a.shape, 255, np.uint8)
    m[valid] = a[valid].astype(np.uint8) - 1
    return m


def dominant(mask):
    v = mask[mask != 255]
    if not v.size:
        return 255
    c = v[v >= 4]
    return int(np.bincount(c - 4).argmax() + 1) if c.size else 0


def check_shape(shape):
    if len(shape) != 2 or min(shape) < 16 or max(shape) > 1024:
        raise ValueError("Expected a single infrared band, 16–1024 pixels on each side.")


def kelvin(a):
    check_shape(a.shape)
    if a.dtype.kind != "f":
        raise ValueError("Kelvin arrays must contain floating-point measurements.")
    ok = ~np.isnan(a)
    if not ok.any() or np.any(ok & ((a < 150) | (a > 350) | ~np.isfinite(a))):
        raise ValueError(
            "Expected 150–350 K with NaN for missing pixels and at least one valid measurement."
        )
    return a.astype(np.float32, copy=False)


def decode(body, filename):
    if not body or len(body) > MAX_BYTES:
        raise ValueError("Upload a nonempty file of at most 8 MiB.")
    ext = Path(filename or "").suffix.lower()
    fp = io.BytesIO(body)
    try:
        if ext == ".npy":
            ver = np.lib.format.read_magic(fp)
            if ver == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
            elif ver == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                raise ValueError("Unsupported NPY version.")
            check_shape(shape)
            if (
                dtype.kind != "f"
                or dtype.itemsize not in (2, 4, 8)
                or fp.tell() + int(np.prod(shape)) * dtype.itemsize != len(body)
            ):
                raise ValueError("Invalid floating-point NPY payload.")
            fp.seek(0)
            a = np.load(fp, allow_pickle=False)
        elif ext in (".png", ".tif", ".tiff"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(fp, formats=["PNG" if ext == ".png" else "TIFF"]) as im:
                    check_shape((im.height, im.width))
                    if im.mode not in ("I;16", "I;16L", "I;16B") or getattr(im, "n_frames", 1) != 1:
                        raise ValueError("Use a single 16-bit grayscale centikelvin image.")
                    a = np.array(im, dtype=np.float32) / 100.0
                    a[a == 0] = np.nan
        else:
            raise ValueError("Use floating-point Kelvin NPY or 16-bit centikelvin PNG/TIFF.")
    except (
        OSError,
        EOFError,
        TypeError,
        struct.error,
        OverflowError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError("Invalid or damaged infrared file.") from None
    return kelvin(a)


def tensor(a, pre):
    import torch

    valid = np.isfinite(a)
    return torch.from_numpy(
        np.ascontiguousarray(
            np.stack([np.where(valid, (a - pre["center"]) / pre["scale"], 0), valid]),
            dtype=np.float32,
        )
    )


def preview(a):
    gray = np.uint8(np.clip((300 - np.nan_to_num(a, nan=300)) / 90, 0, 1) * 255)
    return np.repeat(gray[:, :, None], 3, axis=2)


def color_mask(m):
    valid = m != 255
    out = np.zeros((*m.shape, 3), np.uint8)
    out[valid] = RGB[m[valid]]
    return out
