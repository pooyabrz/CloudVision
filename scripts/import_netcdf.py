"""Import independently co-registered infrared and native CloudCast class NetCDF files."""

import argparse
import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from src.schema import encode_raw, kelvin, write_json


def convert(label_path, image_path, out, label_var, image_var, dataset_id, tolerance_minutes=0):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a fresh output directory.")
    if not 0 <= tolerance_minutes <= 15:
        raise ValueError("Use a time tolerance of 0–15 minutes.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=out.parent) as tmp,
        xr.open_dataset(label_path) as lds,
        xr.open_dataset(image_path) as ids,
    ):
        stage = Path(tmp) / "collection"
        (stage / "scenes").mkdir(parents=True)
        labels, images = lds[label_var], ids[image_var]
        dims = tuple(d for d in labels.dims if d != "time")
        if len(dims) != 2 or "time" not in labels.dims or set(images.dims) != {"time", *dims}:
            raise ValueError("Expected matching spatial dimensions and time.")
        for d in dims:
            if (
                d not in labels.coords
                or d not in images.coords
                or not np.array_equal(labels[d], images[d])
                or labels[d].attrs.get("units") != images[d].attrs.get("units")
            ):
                raise ValueError("Spatial coordinates or units differ.")
        lcrs, icrs = [z.attrs.get("grid_mapping") for z in (labels, images)]
        if bool(lcrs) != bool(icrs) or (
            lcrs and not lds[lcrs].rename("crs").identical(ids[icrs].rename("crs"))
        ):
            raise ValueError("Grid mapping mismatch.")
        if images.attrs.get("units", "").lower() not in ("kelvin", "kelvins", "k"):
            raise ValueError("Declare Kelvin units.")
        ts = images.time.values.astype("datetime64[s]")
        lt = labels.time.values.astype("datetime64[s]")
        if any(np.isnat(z).any() or len(np.unique(z)) != len(z) for z in (lt, ts)) or not len(ts):
            raise ValueError("Missing or duplicate acquisition times.")
        order = np.argsort(ts)
        sorted_ts = ts[order]
        used = set()
        records = []
        for j, t in enumerate(lt):
            k = np.searchsorted(sorted_ts, t)
            i = int(
                order[
                    min(
                        (i for i in (k - 1, k) if 0 <= i < len(ts)),
                        key=lambda i: abs(sorted_ts[i] - t),
                    )
                ]
            )
            offset = int((ts[i] - t) / np.timedelta64(1, "s"))
            if abs(offset) > 60 * tolerance_minutes or i in used:
                raise ValueError("No unique matching infrared observation.")
            used.add(i)
            a = kelvin(np.asarray(images.isel(time=i).transpose(*dims), np.float32))
            m = encode_raw(labels.isel(time=j).transpose(*dims).values)
            m[~np.isfinite(a)] = 255
            if not (m != 255).any():
                raise ValueError("No valid target pixels.")
            stamp = np.datetime_as_string(t, unit="s") + "Z"
            sid = stamp.replace("-", "").replace(":", "")
            file = "scenes/" + sid + ".npz"
            np.savez_compressed(stage / file, kelvin=a, mask=m)
            records.append(
                dict(
                    id=sid,
                    timestamp=stamp,
                    file=file,
                    offset_seconds=offset,
                    array_sha256=hashlib.sha256(a.tobytes() + m.tobytes()).hexdigest(),
                )
            )
        if not records:
            raise ValueError("Empty collection.")
        with (stage / "manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0])
            w.writeheader()
            w.writerows(records)
        write_json(
            stage / "source.json",
            dict(
                dataset=dataset_id,
                label_variable=label_var,
                image_variable=image_var,
                units="K",
                tolerance_minutes=tolerance_minutes,
            ),
        )
        stage.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("label_path", "image_path", "out", "label_var", "image_var", "dataset_id"):
        p.add_argument(k)
    p.add_argument("--tolerance-minutes", type=int, default=0)
    convert(**vars(p.parse_args()))
