"""Audit observations and split whole UTC days before sampling any tiles."""

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.schema import IGNORE, NAMES, config, dominant, kelvin, sha, tensor


def rows(root):
    with (Path(root) / "manifest.csv").open() as f:
        r = list(csv.DictReader(f))
    if not r:
        raise ValueError("Empty manifest.")
    return sorted(r, key=lambda v: datetime.fromisoformat(v["timestamp"]))


def scene(root, r):
    path = (Path(root) / r["file"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Invalid scene path.")
    with np.load(path, allow_pickle=False) as z:
        a, m = kelvin(z["kelvin"]), z["mask"]
    if m.dtype != np.uint8 or m.shape != a.shape or np.any((m >= len(NAMES)) & (m != IGNORE)):
        raise ValueError("Invalid label encoding or dimensions.")
    if not np.any(m != IGNORE) or np.any((m != IGNORE) & ~np.isfinite(a)):
        raise ValueError("Invalid paired support.")
    return a, m


def split(records, fractions=(0.70, 0.15, 0.15), gap_hours=24):
    if (
        len(fractions) != 3
        or min(fractions) <= 0
        or not np.isclose(sum(fractions), 1)
        or gap_hours < 0
    ):
        raise ValueError("Invalid split specification.")
    times = [datetime.fromisoformat(r["timestamp"]) for r in records]
    if any(t.utcoffset() != timedelta(0) for t in times):
        raise ValueError("Timestamps must be UTC.")
    if len(set(times)) != len(times) or len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate observation.")
    days = sorted({t.date() for t in times})
    a, b = round(len(days) * fractions[0]), round(len(days) * sum(fractions[:2]))
    if not 0 < a < b < len(days):
        raise ValueError("Too few distinct dates.")
    cuts = [min(t for t in times if t.date() == day) for day in (days[a], days[b])]
    groups = {k: [] for k in ("train", "val", "test", "embargo")}
    for t, r in sorted(zip(times, records), key=lambda v: v[0]):
        label = "train" if t < cuts[0] else "val" if t < cuts[1] else "test"
        if any(c - timedelta(hours=gap_hours) < t < c for c in cuts):
            label = "embargo"
        groups[label].append(r)
    if not all(groups[k] for k in ("train", "val", "test")):
        raise ValueError("Empty split after embargo.")
    return groups


def audit(root):
    cfg = config("dataset")
    parts = split(rows(root), cfg["fractions"], cfg["gap_hours"])
    seen = set()
    stats = {}
    for k, group in parts.items():
        counts = np.zeros(14, np.int64)
        ignored = 0
        for r in group:
            a, m = scene(root, r)
            h = hashlib.sha256(a.tobytes()).digest()
            if h in seen:
                raise ValueError("Duplicate radiometry.")
            seen.add(h)
            if (
                r.get("array_sha256")
                and hashlib.sha256(a.tobytes() + m.tobytes()).hexdigest() != r["array_sha256"]
            ):
                raise ValueError("Array checksum mismatch.")
            counts += np.bincount(m[m != 255], minlength=14)
            ignored += int((m == 255).sum())
        stats[k] = dict(
            scenes=len(group),
            days=len({r["timestamp"][:10] for r in group}),
            start=group[0]["timestamp"] if group else None,
            end=group[-1]["timestamp"] if group else None,
            counts=dict(zip(NAMES, counts.tolist())),
            ignored=ignored,
        )
    return dict(
        manifest_sha256=sha(Path(root) / "manifest.csv"), splits=stats, gap_hours=cfg["gap_hours"]
    )


def augment(a, m, cfg):
    if cfg["horizontal"] and np.random.rand() < 0.5:
        a, m = np.fliplr(a), np.fliplr(m)
    if cfg["vertical"] and np.random.rand() < 0.5:
        a, m = np.flipud(a), np.flipud(m)
    if cfg["rotate"]:
        k = np.random.randint(4)
        a, m = np.rot90(a, k), np.rot90(m, k)
    if np.random.rand() < cfg["crop_probability"]:
        h, w = a.shape
        fac = np.random.uniform(cfg["minimum_crop"], 1)
        ch, cw = max(1, round(h * fac)), max(1, round(w * fac))
        y, x = np.random.randint(h - ch + 1), np.random.randint(w - cw + 1)
        sl = np.s_[y : y + ch, x : x + cw]
        if np.any(m[sl] != 255):
            a, m = [
                np.asarray(Image.fromarray(v[sl]).resize((w, h), Image.Resampling.NEAREST))
                for v in (a, m)
            ]
    return a, m


class Tiles(Dataset):
    def __init__(self, root, records, tile, pre, augmentation=None):
        self.tile, self.pre, self.aug = tile, pre, augmentation
        self.scenes = []
        self.index = []
        self.counts = np.zeros(14, np.int64)
        self.image_counts = np.zeros(11, np.int64)
        for r in records:
            a, m = scene(root, r)
            h, w = m.shape
            p = ((0, (-h) % tile), (0, (-w) % tile))
            a = np.pad(a, p, constant_values=np.nan)
            m = np.pad(m, p, constant_values=255)
            i = len(self.scenes)
            self.scenes.append((a, m, r))
            self.counts += np.bincount(m[m != 255], minlength=14)
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    d = dominant(m[y : y + tile, x : x + tile])
                    if d != 255:
                        self.image_counts[d] += 1
                        self.index.append((i, y, x))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i, y, x = self.index[n]
        a, m, _ = self.scenes[i]
        t = self.tile
        a, m = a[y : y + t, x : x + t], m[y : y + t, x : x + t]
        if self.aug:
            a, m = augment(a, m, self.aug)
        return tensor(a, self.pre), torch.tensor(m.copy(), dtype=torch.long), dominant(m)
