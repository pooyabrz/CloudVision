"""Seeded, recoverable model training with validation-only calibration and selection."""

import argparse
import fcntl
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import Tiles, rows, split
from src.metrics import calibration, cm, scores
from src.models import build
from src.schema import IMAGE_NAMES, NAMES, ROOT, config, sha, write_json

RUNS = [
    ("cnn", "cnn", "weighted_ce"),
    ("unet", "unet", "combined"),
    ("unet_plus", "unet_plus", "combined"),
    ("segformer", "segformer", "combined"),
    ("unet_ce", "unet", "ce"),
    ("unet_weighted_ce", "unet", "weighted_ce"),
    ("unet_dice", "unet", "dice"),
    ("unet_no_augmentation", "unet", "combined"),
]


def seed(value, threads):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def weight(counts):
    c = torch.as_tensor(counts, dtype=torch.float32)
    valid = c > 0
    w = torch.zeros_like(c)
    w[valid] = (c[valid].sum() / c[valid]).sqrt()
    return w / w[valid].mean()


def objective(z, y, loss, w, dice_fraction=0.25):
    if loss not in ("ce", "weighted_ce", "dice", "combined"):
        raise ValueError("Unknown loss.")
    v = y != 255
    if not v.any():
        return z.sum() * 0.0
    if loss != "dice":
        ce = F.cross_entropy(z, y, weight=None if loss == "ce" else w, ignore_index=255)
        if loss != "combined":
            return ce
    p = z.float().softmax(1) * v.unsqueeze(1)
    labels = F.one_hot(y.masked_fill(~v, 0), num_classes=z.shape[1]).movedim(-1, 1) * v.unsqueeze(1)
    dims = (0, *range(2, p.ndim))
    support = labels.sum(dims)
    dice = (2 * (p * labels).sum(dims) + 1e-6) / (p.sum(dims) + support + 1e-6)
    loss_d = 1 - dice[support > 0].mean()
    return loss_d if loss == "dice" else (1 - dice_fraction) * ce + dice_fraction * loss_d


def save(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def validation(model, loader, device, classifier, collect=False):
    names = IMAGE_NAMES if classifier else NAMES
    matrix = np.zeros((len(names), len(names)), np.int64)
    zs = []
    ys = []
    model.eval()
    with torch.inference_mode():
        for x, m, d in loader:
            logits = model(x.to(device)).cpu()
            y = d if classifier else m
            matrix += cm(y.numpy(), logits.argmax(1).numpy(), len(names))
            if collect:
                z = logits.movedim(1, -1).reshape(-1, len(names))
                y = y.flatten()
                v = y != 255
                z, y = z[v], y[v]
                stride = max(1, int(np.ceil(len(y) / 8192)))
                zs.append(z[::stride].numpy())
                ys.append(y[::stride].numpy())
    return (np.concatenate(zs), np.concatenate(ys)) if collect else scores(matrix, names)


def train(name, root, out, cfg, resume=True):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    with (out / ".training.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(name, Path(root), out, cfg, resume)


def _train(name, root, out, cfg, resume):
    seed(cfg["seed"], cfg["threads"])
    ds = config("dataset")
    groups = split(rows(root), ds["fractions"], ds["gap_hours"])
    pre, tile = ds["preprocessing"], ds["tile"]
    classifier = name == "cnn"
    model_cfg = config("models")[name]
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto"
        else cfg["device"]
    )
    tr = Tiles(root, groups["train"], tile, pre, cfg["augment"])
    va = Tiles(root, groups["val"], tile, pre)
    meta = dict(
        architecture=name,
        model_config=model_cfg,
        task="classification" if classifier else "segmentation",
        dataset=json.loads((root / "source.json").read_text())["dataset"],
        manifest_sha256=sha(root / "manifest.csv"),
        preprocessing=pre,
        tile=tile,
        training=cfg,
        train_tiles=len(tr),
        val_tiles=len(va),
    )
    if (out / "run.json").exists():
        old = json.loads((out / "run.json").read_text())
        if (
            resume
            and all(old.get(k) == v for k, v in meta.items())
            and sha(out / "best.pt") == old["checkpoint_sha256"]
        ):
            return old
        raise ValueError("Existing run differs. Choose a new run directory.")
    model = build(name, model_cfg).to(device)
    weights = weight(tr.image_counts if classifier else tr.counts).to(device)
    optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg["optimizer"]](
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg["scheduler_patience"], factor=0.5
    )
    amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    hist = []
    best = -1.0
    stale = 0
    best_epoch = 0
    duration = 0.0
    if (out / "last.pt").exists():
        if not resume:
            raise FileExistsError("Use --resume or choose a fresh directory.")
        st = torch.load(out / "last.pt", weights_only=True, map_location="cpu")
        if st["meta"] != meta:
            raise ValueError("Configuration or data changed.")
        model.load_state_dict(st["weights"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        hist, best, stale, best_epoch, duration = (
            st[k] for k in ("history", "best", "stale", "best_epoch", "duration")
        )
    gen = torch.Generator()
    common = dict(
        batch_size=cfg["batch_size"], num_workers=cfg["workers"], pin_memory=device.type == "cuda"
    )
    train_loader = DataLoader(tr, shuffle=True, generator=gen, **common)
    val_loader = DataLoader(va, shuffle=False, **common)
    metric = "macro_f1" if classifier else "miou"
    for ep in range(len(hist) + 1, cfg["epochs"] + 1):
        if stale >= cfg["patience"]:
            break
        start = time.perf_counter()
        seed(cfg["seed"] + ep, cfg["threads"])
        gen.manual_seed(cfg["seed"] + ep)
        model.train()
        total = 0.0
        for x, m, d in train_loader:
            optimizer.zero_grad(set_to_none=True)
            y = (d if classifier else m).to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = objective(model(x.to(device)), y, cfg["loss"], weights, cfg["dice_fraction"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        stats = validation(model, val_loader, device, classifier)
        score = stats[metric]
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(score)
        duration += time.perf_counter() - start
        hist.append(
            dict(
                epoch=ep,
                loss=total / len(tr),
                lr=lr,
                val_score=score,
                val_accuracy=stats["accuracy"],
            )
        )
        if score > best:
            best, stale, best_epoch = score, 0, ep
            save(out / "best.pt", dict(weights=model.state_dict(), meta=meta))
        else:
            stale += 1
        save(
            out / "last.pt",
            dict(
                meta=meta,
                weights=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                history=hist,
                best=best,
                stale=stale,
                best_epoch=best_epoch,
                duration=duration,
            ),
        )
        print(out.name, ep, metric, round(score, 4), flush=True)
    model.load_state_dict(
        torch.load(out / "best.pt", weights_only=True, map_location=device)["weights"]
    )
    z, y = validation(model, val_loader, device, classifier, True)

    def nll(t):
        return -np.log(softmax(z / t, axis=1)[np.arange(len(y)), y].clip(1e-12)).mean()

    temp = float(minimize_scalar(nll, bounds=(0.25, 8.0), method="bounded").x)
    if nll(temp) > nll(1.0):
        temp = 1.0
    cal = dict(
        temperature=temp,
        samples=len(y),
        split="val",
        before=calibration(softmax(z, axis=1), y),
        after=calibration(softmax(z / temp, axis=1), y),
    )
    record = dict(
        **meta,
        history=hist,
        best_epoch=best_epoch,
        validation_metric=metric,
        best_validation_score=best,
        calibration=cal,
        training_seconds=duration,
        parameters=sum(p.numel() for p in model.parameters()),
        checkpoint="best.pt",
        checkpoint_sha256=sha(out / "best.pt"),
        device=str(device),
        torch_version=str(torch.__version__),
    )
    write_json(out / "run.json", record)
    return record


def freeze(work):
    work = Path(work)
    candidates = {}
    out = work / "checkpoints"
    out.mkdir(exist_ok=True, parents=True)
    for f in sorted((work / "experiments").glob("*/run.json")):
        r = json.loads(f.read_text())
        name = r["architecture"]
        if (
            name not in candidates
            or r["best_validation_score"] > candidates[name][0]["best_validation_score"]
        ):
            candidates[name] = (r, f.parent)
    if not candidates or len({r["manifest_sha256"] for r, _ in candidates.values()}) != 1:
        raise ValueError("A single dataset and completed runs are required.")
    entries = {}
    for name, (r, folder) in candidates.items():
        src = folder / r["checkpoint"]
        h = sha(src)
        if h != r["checkpoint_sha256"]:
            raise ValueError("Trained checkpoint hash mismatch.")
        dest = out / (name + "-" + h[:12] + ".pt")
        shutil.copyfile(src, dest)
        entries[name] = {
            **{
                k: r[k]
                for k in (
                    "architecture",
                    "task",
                    "dataset",
                    "model_config",
                    "preprocessing",
                    "tile",
                    "calibration",
                    "parameters",
                    "manifest_sha256",
                    "training_seconds",
                    "best_validation_score",
                    "validation_metric",
                )
            },
            "run": folder.name,
            "file": dest.name,
            "version": h[:12],
            "sha256": h,
            "size_bytes": dest.stat().st_size,
        }
    write_json(out / "registry.json", entries)
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed")
    p.add_argument("--workdir", type=Path, default=ROOT)
    p.add_argument("--only", nargs="+", choices=[r[0] for r in RUNS])
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    for tag, name, loss in RUNS:
        if args.only and tag not in args.only:
            continue
        cfg = config("training")
        cfg["loss"] = loss
        if tag == "unet_no_augmentation":
            cfg["augment"] = None
        for item in args.set:
            k, v = item.split("=", 1)
            if k not in cfg:
                p.error("Unknown setting " + k)
            cfg[k] = yaml.safe_load(v)
        train(name, args.data, args.workdir / "experiments" / tag, cfg)
    if args.freeze:
        freeze(args.workdir)


if __name__ == "__main__":
    main()
