import base64
import csv
import hashlib
import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from src.inference import Registry
from src.schema import MAX_BYTES, MODELS, NAMES


def test_health_models_metrics_and_swagger(client):
    assert client.get("/health").json() == {"status": "ok", "models": 4}
    catalog = client.get("/models").json()
    assert {e["architecture"] for e in catalog["models"]} == set(MODELS)
    assert len(client.get("/metrics").json()["models"]) == 4
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_export_matches_uploaded_bytes_and_configuration(client, array_file):
    a = np.full((53, 81), 245, np.float32)
    a[:2] = np.nan
    body = array_file(a)
    out = client.post("/predict", files={"file": ("scene.npy", body)}, data={"model": "unet_plus"})
    assert out.status_code == 200, out.text[:250]
    r = out.json()
    meta = r["metadata"]
    assert meta["input_sha256"] == hashlib.sha256(body).hexdigest()
    mask = Image.open(io.BytesIO(base64.b64decode(r["images"]["mask"])))
    assert json.loads(mask.info["cloudvision"])["version"] == meta["version"]
    assert mask.size == (81, 53) and (np.array(mask)[:2] == 255).all()
    c = list(csv.DictReader(io.StringIO(r["statistics_csv"])))
    assert [x["name"] for x in c] == list(NAMES) and sum(int(x["pixels"]) for x in c) == 51 * 81
    assert abs(sum(float(x["coverage_pct"]) for x in c) - 100) < 1e-5
    assert len(r["images"]["probabilities"]) == 14


def test_cnn_gradcam_and_class_probabilities(client, example):
    r = client.post(
        "/predict", files={"file": ("earth.png", example)}, data={"model": "cnn"}
    ).json()
    assert len(r["classes"]) == 11 and abs(sum(x["probability"] for x in r["classes"]) - 1) < 1e-5
    assert "gradcam" in r["images"] and "mask" not in r["images"]


def test_api_failure_modes(client, example):
    def predict(name, data, form):
        return client.post("/predict", files={"file": (name, data)}, data=form)

    assert predict("sky.png", b"broken", {}).status_code == 422
    assert predict("a.png", example, {"model": "unknown"}).status_code == 404
    assert predict("a.png", example, {"mc_passes": 13}).status_code == 422
    assert predict("a.png", bytes(MAX_BYTES + 1), {}).status_code == 413
    r = predict("a.png", example, {"model": "segformer", "include_maps": "false"}).json()
    assert "probabilities" not in r["images"]


def test_unavailable_assets_return_503(tmp_path):
    with TestClient(create_app(Registry(tmp_path), tmp_path / "absent")) as c:
        for path in ["/health", "/models", "/metrics"]:
            assert c.get(path).status_code == 503
