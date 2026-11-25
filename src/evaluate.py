"""Evaluate frozen checkpoints on held-out observations, with per-class, spatial and timing diagnostics."""

import argparse
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

from src.data import Tiles, audit, rows, split
from src.inference import Registry, default_model
from src.metrics import calibration, cm, scores, spatial
from src.schema import (
    IMAGE_NAMES,
    MODELS,
    NAMES,
    ROOT,
    color_mask,
    config,
    dominant,
    preview,
    sha,
    write_json,
)


def figure(a, m, p, path, title):
    pred = p.argmax(0).astype(np.uint8)
    pred[~np.isfinite(a)] = 255
    entropy = -(p * np.log(p.clip(1e-12))).sum(0) / np.log(len(p))
    entropy[~np.isfinite(a)] = np.nan
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3), layout="constrained", facecolor="#112332")
    for ax, name, img in zip(
        axes,
        ("Thermal IR", "CloudCast reference", "Prediction", "Normalized entropy"),
        (preview(a), color_mask(m), color_mask(pred), entropy),
    ):
        ax.imshow(img, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, color="#dce8ee", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12, color="#eff6f4")
    fig.savefig(path, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)


def timing(model, x, device):
    x = x[None].to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(8):
            model(x)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(60):
            t = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t) * 1000)
    return dict(
        median_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        batch_size=1,
        warmups=8,
        repetitions=60,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )


def one(name, reg, ds, figs):
    net, e, _ = reg.get(name)
    cls = e["task"] == "classification"
    names = IMAGE_NAMES if cls else NAMES
    k = len(names)
    matrix = np.zeros((k, k), np.int64)
    months = defaultdict(lambda: np.zeros_like(matrix))
    err = np.zeros(6, np.int64)
    ps = []
    ys = []
    examples = []
    idx = 0
    image_cm = np.zeros((11, 11), np.int64)
    with torch.inference_mode():
        for image, mask, label in DataLoader(ds, batch_size=16):
            probs = softmax(
                net(image.to(reg.device)).cpu().numpy() / e["calibration"]["temperature"], axis=1
            )
            for j, p in enumerate(probs):
                y = label[j].numpy() if cls else mask[j].numpy()
                pred = p.argmax(0)
                conf = cm(y, pred, k)
                matrix += conf
                sid, _, _ = ds.index[idx]
                month = ds.scenes[sid][2]["timestamp"][:7]
                months[month] += conf
                if cls:
                    ps.append(p[None])
                    ys.append(np.array([y]))
                else:
                    v = y != 255
                    err += spatial(y, pred)
                    ps.append(p[:, v].T[::17])
                    ys.append(y[v][::17])
                    examples.append((float(np.trace(conf) / conf.sum()), idx))
                    image_cm[
                        label[j].item(), dominant(np.where(v, pred, 255).astype(np.uint8))
                    ] += 1
                idx += 1
    s = scores(matrix, names)
    if cls:
        s.pop("miou")
        s.pop("mean_dice")
        for r in s["per_class"]:
            r["f1"] = r.pop("dice")
            r.pop("iou")
    else:
        s["dominant_image_macro_f1"] = scores(image_cm, IMAGE_NAMES)["macro_f1"]
    s.update(
        model=name,
        task=e["task"],
        version=e["version"],
        run=e["run"],
        dataset=e["dataset"],
        evaluation_split="test",
        validation_score=e["best_validation_score"],
        parameters=e["parameters"],
        size_bytes=e["size_bytes"],
        training_seconds=e["training_seconds"],
        inference=timing(net, ds[0][0], reg.device),
        calibration_validation=e["calibration"],
        calibration_test=calibration(np.concatenate(ps), np.concatenate(ys)),
        temporal=[],
    )
    for month, c in sorted(months.items()):
        metrics = scores(c, names)
        s["temporal"].append(
            dict(
                month=month,
                count=int(c.sum()),
                accuracy=metrics["accuracy"],
                score=metrics["macro_f1" if cls else "miou"],
            )
        )
    if not cls:
        s["error_analysis"] = {}
        for i, (den, rate) in enumerate(
            [
                ("boundary_pixels", "boundary_error_rate"),
                ("small_cloud_pixels", "small_cloud_error_rate"),
                ("clear_surface_pixels", "false_cloud_rate"),
            ]
        ):
            n = int(err[2 * i])
            s["error_analysis"][den] = n
            s["error_analysis"][rate] = float(err[2 * i + 1] / n) if n else None
        examples.sort()
        s["examples"] = []
        for key, (acc, i) in zip(
            ("lowest", "median", "highest"),
            (examples[0], examples[len(examples) // 2], examples[-1]),
        ):
            sid, y, x = ds.index[i]
            a, m, r = ds.scenes[sid]
            sl = np.s_[y : y + ds.tile, x : x + ds.tile]
            p, _, _ = reg.predict(a[sl], name)
            file = f"{name}_{key}.png"
            figure(a[sl], m[sl], p, figs / file, f"{MODELS[name]}  /  {r['id']}")
            s["examples"].append(
                dict(scene=r["id"], x=x, y=y, accuracy=acc, figure="figures/" + file)
            )
    return s


def evaluate(root, work):
    reg = Registry(Path(work) / "checkpoints")
    entries = reg.catalog()
    dataset_cfg = config("dataset")
    fp = sha(Path(root) / "manifest.csv")
    if any(e["manifest_sha256"] != fp for e in entries.values()):
        raise ValueError("Evaluation manifest differs from training.")
    held = split(rows(root), dataset_cfg["fractions"], dataset_cfg["gap_hours"])["test"]
    ds = Tiles(root, held, dataset_cfg["tile"], dataset_cfg["preprocessing"])
    out = Path(work) / "experiments"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ms = [one(n, reg, ds, figs) for n in sorted(entries)]
    seg = [m for m in ms if m["task"] == "segmentation"]
    cls = [m for m in ms if m["task"] == "classification"]
    leaders = {
        "segmentation_" + k: max(seg, key=lambda m: m[k])["model"]
        for k in ("miou", "accuracy", "mean_dice")
        if seg
    }
    if cls:
        leaders["classification_macro_f1"] = max(cls, key=lambda m: m["macro_f1"])["model"]
    for key, fn in [
        ("fastest_forward", lambda m: m["inference"]["median_ms"]),
        ("smallest", lambda m: m["size_bytes"]),
        ("shortest_training", lambda m: m["training_seconds"]),
    ]:
        leaders[key] = min(ms, key=fn)["model"]
    cpu = platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    result = dict(
        dataset=ms[0]["dataset"],
        manifest_sha256=fp,
        test_scenes=len(held),
        test_tiles=len(ds),
        tile=ds.tile,
        test_start=held[0]["timestamp"],
        test_end=held[-1]["timestamp"],
        hardware=dict(
            cpu=cpu,
            threads=torch.get_num_threads(),
            device=str(reg.device),
            gpu=torch.cuda.get_device_name(reg.device) if reg.device.type == "cuda" else None,
            torch=str(torch.__version__),
            python=platform.python_version(),
        ),
        leaders=leaders,
        recommended_segmentation=default_model(entries.values()) if seg else None,
        models=ms,
    )
    write_json(out / "data_audit.json", audit(root))
    write_json(out / "comparison.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed")
    p.add_argument("--workdir", type=Path, default=ROOT)
    a = p.parse_args()
    evaluate(a.data, a.workdir)
