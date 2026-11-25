import json

import numpy as np
import pytest
import torch

from src.inference import Registry, default_model, summary
from src.metrics import calibration, cm, scores, spatial
from src.models import build
from src.schema import MODELS, ROOT, config, sha
from src.train import objective, weight


@pytest.mark.parametrize("name", list(MODELS))
def test_architecture_shapes_and_gradients(name):
    net = build(name, config("models")[name])
    z = net(torch.randn(2, 2, 64, 64))
    assert tuple(z.shape) == ((2, 11) if name == "cnn" else (2, 14, 64, 64))
    z.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("loss", ["ce", "weighted_ce", "dice", "combined"])
def test_ignored_pixels_never_contribute(loss):
    z = torch.randn(1, 3, 3, 3, requires_grad=True)
    y = torch.ones((1, 3, 3), dtype=torch.long)
    y[0, 0, 0] = 255
    objective(z, y, loss, weight([5, 2, 1])).backward()
    assert not z.grad[0, :, 0, 0].any()
    assert objective(z, torch.full_like(y, 255), loss, torch.ones(3)).item() == 0


def test_metric_conventions():
    mat = cm(np.array([0, 0, 1, 255]), np.array([0, 1, 1, 1]), 3)
    s = scores(mat, ["A", "B", "C"])
    assert (
        s["accuracy"] == pytest.approx(2 / 3)
        and s["miou"] == 0.5
        and s["mean_dice"] == pytest.approx(2 / 3)
    )
    assert s["per_class"][2]["iou"] is None
    c = calibration(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1]))
    assert c["ece"] == pytest.approx(0.1)


def test_cloud_pixel_denominator_and_no_cloud():
    p = np.zeros((14, 2, 4), np.float32)
    p[0] = 1
    p[:, 0, 1] = 0
    p[5, 0, 1] = 1
    valid = np.ones((2, 4), bool)
    valid[:, 3] = False
    m, _, _, classes, s = summary(p, valid)
    assert (
        s["cloud_coverage_pct"] == pytest.approx(100 / 6)
        and s["dominant"] == classes[5]["name"]
        and (m[:, 3] == 255).all()
    )
    p[0] = 1
    p[5] = 0
    assert summary(p, valid)[-1]["dominant"] == "No cloud"


def test_spatial_error_is_confined_to_valid_support():
    y = np.array([[0, 4, 255], [0, 4, 255]], np.uint8)
    p = np.array([[5, 4, 3], [0, 5, 13]], np.uint8)
    np.testing.assert_array_equal(spatial(y, p), [4, 2, 2, 1, 2, 1])


def test_registry_and_metrics_share_frozen_dataset():
    reg = Registry()
    entries = reg.catalog()
    r = json.loads((ROOT / "experiments/comparison.json").read_text())
    assert set(entries) == set(MODELS) and r["recommended_segmentation"] == default_model(
        entries.values()
    )
    for e in entries.values():
        assert (
            sha(reg.folder / e["file"]) == e["sha256"]
            and e["manifest_sha256"] == r["manifest_sha256"]
        )
        assert e["calibration"]["after"]["nll"] <= e["calibration"]["before"]["nll"] + 1e-6
    assert len(list((ROOT / "experiments").glob("*/run.json"))) == 8


def test_real_stochastic_inference_and_model_restoration():
    reg = Registry()
    a = np.full((50, 70), 255.0, np.float32)
    p, _, _ = reg.predict(a, "unet_plus", 3)
    q = next(reg.predict_batch([a], "unet_plus"))[0]
    assert p.shape == (14, 50, 70) and np.allclose(p.sum(0), 1, atol=1e-5)
    assert abs(p - q).max() > 1e-6
    model, _, _ = reg.get("unet_plus")
    assert all(not layer.training for layer in model.modules())


def test_epoch_restart_reproduces_uninterrupted_weights(tmp_path, monkeypatch):
    import csv
    import shutil

    import src.train as trainer
    from src.data import rows

    original = ROOT / "data/processed"
    subset = tmp_path / "small"
    (subset / "scenes").mkdir(parents=True)
    records = rows(original)[::3][:9]
    for row in records:
        shutil.copyfile(original / row["file"], subset / row["file"])
    shutil.copyfile(original / "source.json", subset / "source.json")
    with (subset / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    cfg = {
        **config("training"),
        "epochs": 2,
        "batch_size": 8,
        "threads": 1,
        "loss": "weighted_ce",
        "device": "cpu",
    }
    original_save = trainer.save

    def stop_after_first_epoch(path, payload):
        original_save(path, payload)
        if path.name == "last.pt":
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "save", stop_after_first_epoch)
        with pytest.raises(RuntimeError, match="interruption"):
            trainer.train("cnn", subset, tmp_path / "resumed", cfg)
    resumed = trainer.train("cnn", subset, tmp_path / "resumed", cfg)
    clean = trainer.train("cnn", subset, tmp_path / "clean", cfg)
    assert resumed["history"] == clean["history"]
    resumed_weights = torch.load(tmp_path / "resumed/best.pt", weights_only=True)["weights"]
    clean_weights = torch.load(tmp_path / "clean/best.pt", weights_only=True)["weights"]
    assert all(torch.equal(value, clean_weights[key]) for key, value in resumed_weights.items())
