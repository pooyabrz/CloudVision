"""Replay the persisted eight run histories into MLflow; skip already imported versions."""

import argparse
import json
from pathlib import Path

from src.schema import ROOT


def track(uri, workdir=ROOT):
    import mlflow

    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment("CloudVision")
    for path in sorted((Path(workdir) / "experiments").glob("*/run.json")):
        r = json.loads(path.read_text())
        h = r["checkpoint_sha256"]
        if not mlflow.search_runs(
            [exp.experiment_id], filter_string=f"tags.checkpoint_sha256 = '{h}'"
        ).empty:
            continue
        with mlflow.start_run(run_name=path.parent.name, tags=dict(checkpoint_sha256=h)):
            mlflow.log_params(
                {
                    k: r[k]
                    for k in (
                        "architecture",
                        "dataset",
                        "best_epoch",
                        "manifest_sha256",
                        "parameters",
                    )
                }
            )
            mlflow.log_params({k: str(v) for k, v in r["training"].items()})
            for epoch in r["history"]:
                mlflow.log_metrics(
                    {k: epoch[k] for k in ("loss", "val_score", "lr", "val_accuracy")},
                    step=epoch["epoch"],
                )
            mlflow.log_metrics(
                dict(
                    training_seconds=r["training_seconds"],
                    validation_score=r["best_validation_score"],
                    temperature=r["calibration"]["temperature"],
                )
            )
            mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(path.parent / "best.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default=(ROOT / "mlruns").as_uri())
    p.add_argument("--workdir", type=Path, default=ROOT)
    args = p.parse_args()
    track(args.uri, args.workdir)
