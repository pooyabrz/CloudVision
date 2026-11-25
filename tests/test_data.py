import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from scripts.import_netcdf import convert
from src.data import Tiles, audit, augment, rows, scene, split
from src.schema import ROOT, config, decode, dominant, encode_raw, tensor


def test_raw_codes_and_masked_fill():
    raw = np.ma.array([0, 1, 4, 5, 14, 255, 2], mask=[0, 0, 0, 0, 0, 0, 1])
    assert encode_raw(raw).tolist() == [255, 0, 3, 4, 13, 255, 255]
    with pytest.raises(ValueError):
        encode_raw([15])


@pytest.mark.parametrize("name,expected", [("all_missing", 255), ("clear", 0), ("small_cloud", 2)])
def test_dominant_excludes_surfaces(name, expected):
    mask = {"all_missing": [255, 255], "clear": [0, 1, 2], "small_cloud": [0, 0, 0, 5, 255]}[name]
    assert dominant(np.array(mask)) == expected


def test_npy_missing_tensor(array_file):
    x = np.full((32, 17), 265.0, dtype=np.float32)
    x[0, 0] = np.nan
    a = decode(array_file(x), "measurement.npy")
    t = tensor(a, config("dataset")["preprocessing"])
    assert t.shape == (2, 32, 17) and t[1].sum() == 543 and not t[0].any()


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((3, 24, 24), np.float32),
        np.full((16, 16), 400, np.float32),
        np.ones((16, 16), int),
        np.full((16, 16), np.nan, np.float32),
        np.full((16, 16), np.inf, np.float32),
        np.ones((15, 16), np.float32),
    ],
)
def test_reject_bad_arrays(array_file, bad):
    with pytest.raises(ValueError):
        decode(array_file(bad), "scan.npy")


@pytest.mark.parametrize("fmt", ["PNG", "TIFF"])
def test_raster_encoding(fmt):
    x = np.full((21, 23), 24050, np.uint16)
    x[0, 0] = 0
    out = io.BytesIO()
    Image.fromarray(x).save(out, format=fmt)
    a = decode(out.getvalue(), "scan." + fmt.lower())
    assert np.isnan(a[0, 0]) and a[1, 1] == 240.5


def test_import_netcdf_atomic_and_units(tmp_path):
    times = np.array(["2017-01-01", "2017-01-05"], dtype="datetime64[ns]")
    coords = {"time": times, "y": np.arange(20), "x": np.arange(16)}
    dims = ("time", "y", "x")
    m = np.full((2, 20, 16), 5, dtype=np.uint8)
    x = np.full_like(m, 255, dtype=np.float32)
    x[:, 0, 0] = np.nan
    labels = xr.Dataset({"cloud": (dims, m)}, coords=coords)
    ir = xr.Dataset({"ir": (dims, x, {"units": "K"})}, coords=coords)
    la, im, out = tmp_path / "label.nc", tmp_path / "input.nc", tmp_path / "out"
    labels.to_netcdf(la)
    ir.transpose("time", "x", "y").to_netcdf(im)
    convert(la, im, out, "cloud", "ir", "fixture")
    records = rows(out)
    a, mask = scene(out, records[0])
    assert (
        len(records) == 2 and np.all(mask[:, 1:] == 4) and mask[0, 0] == 255 and a.shape == (20, 16)
    )
    ir.ir.attrs["units"] = "C"
    ir.to_netcdf(im)
    with pytest.raises(ValueError, match="Kelvin"):
        convert(la, im, tmp_path / "failed", "cloud", "ir", "fixture")
    assert not (tmp_path / "failed").exists()
    ir.ir.attrs["units"] = "K"
    ir.assign_coords(x=np.arange(1, 17)).to_netcdf(im)
    with pytest.raises(ValueError, match="coordinates"):
        convert(la, im, tmp_path / "coords", "cloud", "ir", "fixture")


def test_splits_and_checksum_of_every_scene():
    root = ROOT / "data/processed"
    result = audit(root)
    parts = result["splits"]
    assert [parts[k]["scenes"] for k in ["train", "val", "test"]] == [177, 36, 39]
    for left, right in [("train", "val"), ("val", "test")]:
        gap = datetime.fromisoformat(parts[right]["start"]) - datetime.fromisoformat(
            parts[left]["end"]
        )
        assert gap >= timedelta(hours=24)
    assert sum(parts["test"]["counts"].values()) + parts["test"]["ignored"] == 39 * 128**2


def test_split_embargo_and_duplicates():
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    records = [
        dict(id=str(i), timestamp=(start + timedelta(hours=i * 12)).isoformat()) for i in range(40)
    ]
    parts = split(records)
    assert len(parts["embargo"]) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        split(records + [records[-1]])


def test_mask_and_input_stay_aligned():
    m = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 14
    m = m.astype(np.uint8)
    a = 230 + m.astype(np.float32)
    for _ in range(12):
        b, n = augment(
            a,
            m,
            dict(
                horizontal=True, vertical=True, rotate=True, crop_probability=1, minimum_crop=0.65
            ),
        )
        np.testing.assert_array_equal(b - 230, n)


def test_fractional_tiles_pad_without_spatial_drift(tmp_path):
    root = tmp_path
    a = np.full((65, 83), 246.0, dtype=np.float32)
    m = np.full(a.shape, 6, np.uint8)
    np.savez(root / "x.npz", kelvin=a, mask=m)
    d = Tiles(root, [dict(file="x.npz")], 64, config("dataset")["preprocessing"])
    _, target, label = d[3]
    assert len(d) == 4 and (target != 255).sum().item() == 19 and label == 3


def test_malformed_bytes_and_name_are_rejected():
    for contents, name in [
        (b"bad", "broken.png"),
        (b"bad", "bad.npy"),
        (b"pixels", "sky.jpg"),
        (b"", "empty.tif"),
    ]:
        with pytest.raises(ValueError):
            decode(contents, name)
