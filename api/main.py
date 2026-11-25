import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.inference import Registry, default_model, format_result
from src.schema import MAX_BYTES, ROOT, decode


class UploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/predict":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunk = msg.get("body", b"")
            if len(body) + len(chunk) > MAX_BYTES + 65536:
                return await JSONResponse({"detail": "Upload exceeds 8 MiB."}, 413)(
                    scope, receive, send
                )
            body.extend(chunk)
            if not msg.get("more_body"):
                break

        async def replay():
            nonlocal body
            if body is None:
                return await receive()
            data, body = body, None
            return {"type": "http.request", "body": bytes(data), "more_body": False}

        return await self.app(scope, replay, send)


class Prediction(BaseModel):
    """A self-describing prediction for an image classifier or a segmenter."""

    metadata: dict[str, Any] = Field(
        description="Checkpoint and input SHA-256, dimensions, valid support, units, model settings, calibration and spatial transform."
    )
    summary: dict[str, Any] = Field(
        description="Dominant cloud type, its confidence, maximum probability and normalized entropy. Segmenters also return cloud and non-cloud percentages."
    )
    classes: list[dict[str, Any]] = Field(
        description="Taxonomy-order probabilities. Segmenters add pixel counts, coverage, colors and assigned-class mean confidence."
    )
    images: dict[str, str | list[str]] = Field(
        description="Base64 PNGs. Segmenters return the original, mask, segmentation, overlay, confidence, entropy and optional per-class maps. The CNN returns the original and Grad-CAM."
    )
    statistics_csv: str = Field(
        description="Downloadable class statistics with input fingerprint and model version."
    )


def create_app(reg=None, metrics=None):
    reg = reg or Registry()
    metrics = Path(
        metrics or os.environ.get("CLOUDVISION_METRICS", ROOT / "experiments/comparison.json")
    )
    app = FastAPI(
        title="CloudVision",
        version="1.0.0",
        description="Satellite infrared cloud classification and segmentation",
    )
    app.add_middleware(UploadLimit)

    @app.get("/health")
    def health():
        try:
            e = reg.catalog()
            if not e or any(not (reg.folder / v["file"]).is_file() for v in e.values()):
                raise RuntimeError()
        except (RuntimeError, KeyError):
            raise HTTPException(503, "Model assets unavailable.") from None
        return dict(status="ok", models=len(e))

    @app.get("/models")
    def models():
        try:
            entries = list(reg.catalog().values())
        except RuntimeError:
            raise HTTPException(503, "Model registry unavailable.") from None
        return dict(
            models=[
                {
                    k: e[k]
                    for k in (
                        "architecture",
                        "task",
                        "version",
                        "dataset",
                        "tile",
                        "model_config",
                        "preprocessing",
                        "size_bytes",
                        "best_validation_score",
                        "validation_metric",
                    )
                }
                for e in entries
            ],
            default=default_model(entries)
            if any(e["task"] == "segmentation" for e in entries)
            else None,
        )

    @app.get("/metrics")
    def metric_report():
        try:
            return json.loads(metrics.read_text())
        except (ValueError, OSError):
            raise HTTPException(503, "Evaluation report unavailable.") from None

    @app.post(
        "/predict",
        response_model=Prediction,
        responses={
            404: {"description": "Unknown model identifier"},
            413: {"description": "Upload exceeds 8 MiB"},
            422: {"description": "Invalid thermal image, file format or form parameter"},
            503: {"description": "Model assets or inference unavailable"},
        },
    )
    async def predict(
        file: Annotated[
            UploadFile, File(description="Floating-point Kelvin NPY or 16-bit centikelvin PNG/TIFF")
        ],
        model: Annotated[str | None, Form()] = None,
        mc_passes: Annotated[int, Form(ge=1, le=12)] = 1,
        include_maps: Annotated[bool, Form()] = True,
    ):
        try:
            raw = await file.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise HTTPException(413, "Upload exceeds 8 MiB.")

            def run():
                a = decode(raw, file.filename)
                p, cam, e = reg.predict(a, model, mc_passes)
                return format_result(
                    a, p, cam, e, hashlib.sha256(raw).hexdigest(), mc_passes, include_maps
                )

            return await run_in_threadpool(run)
        except KeyError:
            raise HTTPException(404, "Unknown model. See /models.") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except (OSError, RuntimeError):
            raise HTTPException(503, "Model inference unavailable.") from None
        finally:
            await file.close()

    return app


app = create_app()
