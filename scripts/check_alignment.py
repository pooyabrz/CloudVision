"""Check the class-grid orientation against an independent land/water reference."""

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from matplotlib.path import Path as Polygon

from scripts.fetch_pairs import FULL, cloud, fetch, grid
from src.schema import ROOT, write_json

LAND = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"


def check_radiometry(cache, lat, lon, data):
    """Read the provider's complete CF file and check every matched sample."""
    import netCDF4

    product = "GRIDSAT-B1.2017.01.03.00.v02r01.nc"
    source = cache / "full" / product
    fetch(f"{FULL}/2017/{product}", source)
    with netCDF4.Dataset(source) as dataset:
        latitudes = np.asarray(dataset["lat"][:], dtype=float)
        longitudes = np.asarray(dataset["lon"][:], dtype=float)
        # Locate each midpoint independently of the arithmetic in the DAP sampler.
        iy = np.searchsorted((latitudes[:-1] + latitudes[1:]) / 2, lat)
        ix = np.searchsorted((longitudes[:-1] + longitudes[1:]) / 2, lon)
        y0, y1 = int(iy.min()), int(iy.max()) + 1
        x0, x1 = int(ix.min()), int(ix.max()) + 1
        variable = dataset["irwin_cdr"]
        # The provider stores valid_range in physical Kelvin; netCDF4 masking
        # compares packed integers to it. Retain CF scaling, then mask in Kelvin.
        variable.set_auto_mask(False)
        slab = variable[0, y0:y1, x0:x1]
        values = np.array(slab[iy - y0, ix - x0], dtype=np.float32)
        values[(values < 150) | (values > 350)] = np.nan
    with np.load(Path(data) / "scenes/20170103T0000Z.npz", allow_pickle=False) as pair:
        observed = pair["kelvin"]
    np.testing.assert_allclose(observed, values, rtol=0, atol=0.00005, equal_nan=True)
    return {
        "product": product,
        "pixels_compared": int(observed.size),
        "valid_pixels": int(np.isfinite(values).sum()),
        "max_abs_error_kelvin": float(np.nanmax(np.abs(values - observed))),
    }


def check(cache, output, data=None):
    cache = Path(cache)
    geo = cloud(cache, "2017M01", "GEO.npz")
    idx, lat, lon = grid(geo, 128)
    labels = np.load(io.BytesIO(cloud(cache, "2017M01", "0.npy")), allow_pickle=False)[
        np.ix_(idx, idx)
    ]
    file = cache / "ne_50m_land.geojson"
    if not file.exists():
        with urlopen(LAND, timeout=90) as response:
            file.write_bytes(response.read())
    points = np.column_stack([lon.ravel(), lat.ravel()])
    land = np.zeros(len(points), bool)
    for feature in json.loads(file.read_text())["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for shell, *holes in polygons:
            ring = np.asarray(shell)
            if (ring.max(0) < points.min(0)).any() or (ring.min(0) > points.max(0)).any():
                continue
            inside = Polygon(ring).contains_points(points)
            for hole in holes:
                inside &= ~Polygon(hole).contains_points(points)
            land |= inside

    def compare(y):
        valid = np.isin(y, [1, 2])
        correct = int(((y == 1)[valid] == land.reshape(y.shape)[valid]).sum())
        return {
            "pixels": int(valid.sum()),
            "correct": correct,
            "agreement": correct / int(valid.sum()),
        }

    result = {
        "scene": "2017-01-01T00:00:00Z",
        "reference": LAND,
        "native": compare(labels),
        "transpose_control": compare(labels.T),
    }
    if result["native"]["agreement"] <= result["transpose_control"]["agreement"]:
        raise ValueError("Native grid orientation did not outperform the transpose control.")
    if data is not None:
        result["independent_radiometry"] = check_radiometry(cache, lat, lon, data)
    write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "data/raw")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/alignment.json")
    p.add_argument(
        "--compare-data",
        type=Path,
        help="Also verify saved infrared against a complete NOAA source file.",
    )
    args = p.parse_args()
    print(json.dumps(check(args.cache, args.out, args.compare_data), indent=2))
