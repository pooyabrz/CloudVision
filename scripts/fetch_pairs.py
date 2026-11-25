"""Build an independently co-registered CloudCast and GridSat-B1 dataset."""

import argparse
import calendar
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from pyproj import Transformer

from src.schema import encode_raw, write_json

MIRROR = "https://api.kaggle.com/v1/datasets/download/christianlillelund/the-cloudcast-dataset/"
NOAA = "https://www.ncei.noaa.gov/thredds/dodsC/cdr/gridsat"
CRS = "+proj=stere +lat_0=90 +lat_ts=50 +lon_0=5 +ellps=WGS84"


def fetch(url, path):
    if path.is_file():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urlopen(url, timeout=80) as r:
                b = r.read()
            if b[:3] not in (b"\x93NU", b"PK\x03", b"Dat", b"\x89HD"):
                raise ValueError("Unexpected upstream format.")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                f.write(b)
            os.replace(f.name, path)
            return b
        except (OSError, ValueError):
            if i == 4:
                raise
            time.sleep(2**i)


def cloud(cache, folder, f):
    return fetch(MIRROR + quote(folder + "/" + f, safe=""), cache / folder / f)


def grid(geo, side):
    with np.load(io.BytesIO(geo), allow_pickle=False) as z:
        x, y = z["lats"].ravel(), z["lons"].ravel()
    if x.size != 768 or y.size != 768 or side not in (128, 768):
        raise ValueError("Unknown CloudCast grid.")
    step = 768 // side
    idx = np.arange(step // 2, 768, step)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(
        *np.meshgrid(x[idx], y[idx])
    )
    return idx, lat, lon


def sample(body, lat, lon, t):
    head, data = body.split(b"\nData:\n", 1)
    dimensions = re.search(rb"Int16 irwin_cdr\[time = 1\]\[lat = (\d+)\]\[lon = (\d+)\]", head)
    if not dimensions:
        raise ValueError("Unexpected DAP2 grid.")
    h, w = map(int, dimensions.groups())
    offset = 0

    def vec(n, dtype):
        nonlocal offset
        if not np.all(np.frombuffer(data, ">u4", 2, offset) == n):
            raise ValueError("Truncated grid.")
        a = np.frombuffer(data, dtype, n, offset + 8)
        offset += 8 + 4 * n
        return a

    raw = vec(h * w, ">i4").reshape(h, w)
    if abs(int(vec(1, ">i4")[0]) * 3600 - t.timestamp()) > 60:
        raise ValueError("Wrong GridSat time.")
    yy, xx = vec(h, ">f4").astype(float), vec(w, ">f4").astype(float)
    if offset != len(data):
        raise ValueError("Unexpected DAP2 response length.")

    def nearest(v, axis):
        idx = np.rint((v - axis[0]) / ((axis[-1] - axis[0]) / (len(axis) - 1))).astype(int)
        if idx.min() < 0 or idx.max() >= len(axis):
            raise ValueError("Target outside source grid.")
        return idx

    packed = raw[nearest(lat, yy), nearest(lon, xx)]
    a = packed.astype(np.float32) * 0.01 + 200.0
    a[(packed == -31999) | (a < 150) | (a > 350)] = np.nan
    return a


NC_LOCK = threading.Lock()
FULL = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access"


def infrared(cache, name, t, lat, lon):
    query = quote("irwin_cdr[0][1570:1:1930][2370:1:3075]", safe=":")
    part = cache / "dap" / (name + ".dods")
    try:
        body = fetch(f"{NOAA}/{t.year}/{name}.dods?{query}", part)
        return sample(body, lat, lon, t), hashlib.sha256(body).hexdigest()
    except (OSError, ValueError):
        # Some OPeNDAP responses stop after the header. The complete packed NetCDF
        # is an independent, equivalent source for the same values and grid centers.
        import netCDF4

        full = fetch(f"{FULL}/{t.year}/{name}", cache / "full" / name)
        with NC_LOCK, netCDF4.Dataset("satellite.nc", memory=full) as z:
            v = z.variables["irwin_cdr"]
            v.set_auto_maskandscale(False)
            source_time = z.variables["time"]
            if (
                abs(
                    netCDF4.date2num(t.replace(tzinfo=None), source_time.units)
                    - float(source_time[0])
                )
                > 1e-5
            ):
                raise ValueError("Wrong full-file UTC time.")
            raw = v[0, 1570:1931, 2370:3076]
            yy = z["lat"][1570:1931].astype(float)
            xx = z["lon"][2370:3076].astype(float)
            yi = np.rint((lat - yy[0]) / ((yy[-1] - yy[0]) / (len(yy) - 1))).astype(int)
            xi = np.rint((lon - xx[0]) / ((xx[-1] - xx[0]) / (len(xx) - 1))).astype(int)
            vraw = raw[yi, xi]
            a = vraw.astype(np.float32) * float(v.scale_factor) + float(v.add_offset)
            a[(vraw == v._FillValue) | (a < 150) | (a > 350)] = np.nan
        return a, hashlib.sha256(full).hexdigest()


def build(out, cache, months, days, hours, year, side, workers):
    out, cache = Path(out), Path(cache)
    days, months, hours = [sorted(set(x)) for x in (days, months, hours)]
    spec = dict(year=year, size=side, months=months, date_days=days, utc_hours=hours)
    if (out / "source.json").exists():
        existing = json.loads((out / "source.json").read_text())
        if all(existing.get(k) == v for k, v in spec.items()) and (out / "manifest.csv").exists():
            return
        raise FileExistsError("Use a new directory for a different acquisition.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        dest = Path(temp) / "collection"
        (dest / "scenes").mkdir(parents=True)

        def month_jobs(m):
            month = f"{year}M{m:02}"
            ts = np.load(
                io.BytesIO(cloud(cache, month, "TIMESTAMPS.npy")), allow_pickle=False
            ).ravel()
            g = grid(cloud(cache, month, "GEO.npz"), side)
            jobs = []
            for d in days:
                if d > calendar.monthrange(year, m)[1]:
                    continue
                for h in hours:
                    t = datetime(year, m, d, h, tzinfo=timezone.utc)
                    minutes = (d - 1) * 1440 + h * 60
                    ix = int(np.abs(ts - minutes).argmin())
                    offs = float(ts[ix] - minutes)
                    if abs(offs) <= 7.5:
                        jobs.append((t, month, ix, offs, *g))
            return jobs

        def pair(job):
            t, month, ix, offs, idx, lat, lon = job
            sid = t.strftime("%Y%m%dT%H%MZ")
            name = t.strftime("GRIDSAT-B1.%Y.%m.%d.%H.v02r01.nc")
            a, input_hash = infrared(cache, name, t, lat, lon)
            labels = cloud(cache, month, f"{ix}.npy")
            raw = np.load(io.BytesIO(labels), allow_pickle=False)
            if raw.shape != (768, 768):
                raise ValueError("Unexpected label shape.")
            m = encode_raw(raw[np.ix_(idx, idx)])
            m[~np.isfinite(a)] = 255
            if not np.any(m != 255):
                raise ValueError("No paired valid pixels.")
            file = f"scenes/{sid}.npz"
            np.savez_compressed(dest / file, kelvin=a, mask=m)
            return dict(
                id=sid,
                timestamp=t.isoformat(),
                file=file,
                label_member=f"{month}/{ix}.npy",
                nominal_offset_minutes=offs,
                radiance_product=name,
                label_sha256=hashlib.sha256(labels).hexdigest(),
                input_sha256=input_hash,
                array_sha256=hashlib.sha256(a.tobytes() + m.tobytes()).hexdigest(),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [v for month in pool.map(month_jobs, months) for v in month]
            rows = []
            for r in pool.map(pair, jobs):
                rows.append(r)
                if len(rows) % 30 == 0:
                    print(len(rows), "/", len(jobs), flush=True)
        if not rows:
            raise ValueError("No collocated observations.")
        with (dest / "manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
        write_json(
            dest / "source.json",
            dict(
                dataset=f"cloudcast-gridsat-europe-{year}-v1",
                **spec,
                variable="irwin_cdr",
                units="K",
                projection=CRS,
                mirror=MIRROR,
                noaa=NOAA,
                downsample="nearest center, source index step//2 + k*step",
            ),
        )
        os.replace(dest, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--cache", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--months", nargs="+", type=int, choices=range(1, 13), default=list(range(1, 13))
    )
    p.add_argument(
        "--days", nargs="+", type=int, choices=range(1, 32), default=[3, 7, 11, 15, 19, 23, 27]
    )
    p.add_argument("--hours", nargs="+", type=int, choices=range(0, 24, 3), default=[0, 9, 18])
    p.add_argument("--year", type=int, choices=[2017, 2018], default=2017)
    p.add_argument("--side", type=int, choices=[128, 768], default=128)
    p.add_argument("--workers", type=int, default=8)
    build(**vars(p.parse_args()))
