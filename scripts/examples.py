"""Build ready-to-upload observations and a georeferenced map from measured brightness temperatures."""

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from scripts.fetch_pairs import CRS
from src.schema import ROOT


def generate():
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    samples = ["20170107T1800Z", "20170115T0900Z", "20170211T0000Z"]
    for stamp in samples:
        with np.load(ROOT / "data/processed/scenes" / (stamp + ".npz")) as z:
            a = z["kelvin"]
        Image.fromarray(np.rint(np.nan_to_num(a) * 100).astype(np.uint16)).save(
            output / (stamp + ".png")
        )
    with np.load(ROOT / "data/processed/scenes" / (samples[0] + ".npz")) as z:
        a = z["kelvin"]
    w, h = 760, 540
    y, x = np.mgrid[:h, :w]
    scale = w / 128
    offset = (128 - h / scale) / 2
    t = map_coordinates(
        np.nan_to_num(a, nan=285), [y / scale + offset, x / scale], order=1, mode="nearest"
    )
    colors = np.array(
        [
            [248, 245, 218],
            [224, 240, 226],
            [177, 213, 218],
            [115, 165, 184],
            [65, 105, 137],
            [40, 65, 91],
            [26, 44, 58],
            [16, 27, 40],
        ]
    )
    im = Image.fromarray(
        np.stack(
            [
                np.interp(t, [210, 225, 238, 252, 267, 280, 293, 307], colors[:, i])
                for i in range(3)
            ],
            -1,
        ).astype(np.uint8)
    )
    draw = ImageDraw.Draw(im, "RGBA")
    transform = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    def coords(lon, lat):
        x, y = transform.transform(lon, lat)
        return list(
            zip(
                (np.asarray(x) + 853600.436345 - 9000) / 18000 * scale,
                ((-2639500.0 - 9000 - np.asarray(y)) / 18000 - offset) * scale,
            )
        )

    for lon in range(-20, 41, 10):
        lat = np.linspace(38, 66, 80)
        draw.line(coords(np.full_like(lat, lon), lat), fill=(222, 245, 240, 62), width=1)
    for lat in range(40, 66, 5):
        lon = np.linspace(-18, 38, 100)
        draw.line(coords(lon, np.full_like(lon, lat)), fill=(222, 245, 240, 62), width=1)
    for n, lo, la in [
        ("LONDON", -0.13, 51.51),
        ("BERLIN", 13.4, 52.52),
        ("PARIS", 2.35, 48.86),
        ("COPENHAGEN", 12.57, 55.68),
        ("WARSAW", 21.01, 52.23),
    ]:
        xx, yy = coords([lo], [la])[0]
        draw.ellipse((xx - 2, yy - 2, xx + 2, yy + 2), fill="white")
        draw.text((xx + 5, yy - 6), n, fill="white", stroke_width=1, stroke_fill="#284153")
    im.save(output / "europe-thermal.jpg", quality=94)


if __name__ == "__main__":
    generate()
