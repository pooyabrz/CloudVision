"""Regenerate the narrative benchmark, model cards and PDF from frozen measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.schema import MODELS, ROOT


def fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def percent(value):
    return f"{value:.1%}"


def confusions(model, n=3):
    data = np.array(model["confusion_matrix"])
    np.fill_diagonal(data, 0)
    classes = model["per_class"]
    result = []
    for index in np.argsort(data.ravel())[-n:][::-1]:
        i, j = np.unravel_index(index, data.shape)
        result.append((classes[i]["name"], classes[j]["name"], int(data[i, j])))
    return result


def markdown_tables(headings, rows):
    return "\n".join(
        " | ".join(["", *map(str, r), ""]) for r in [headings, ["---"] * len(headings), *rows]
    )


def write_markdown(docs, report, runs, winner):
    models = report["models"]
    headline = (
        f"{MODELS[winner['model']]} ranked first on validation mIoU ({fmt(winner['validation_score'])}) "
        f"and achieved {fmt(winner['miou'])} mIoU and {fmt(winner['mean_dice'])} mean Dice on the held-out test. "
        f"The test covers {report['test_scenes']} later scenes and {report['test_tiles']} tiles."
    )
    measures = markdown_tables(
        [
            "Model",
            "Task",
            "Accuracy",
            "Image macro F1",
            "mIoU",
            "Mean Dice",
            "Median ms",
            "p95 ms",
            "MiB",
            "Train s",
        ],
        [
            [
                MODELS[m["model"]],
                m["task"],
                fmt(m["accuracy"]),
                fmt(m["macro_f1"] if m["task"] == "classification" else None),
                fmt(m.get("miou")),
                fmt(m.get("mean_dice")),
                fmt(m["inference"]["median_ms"], 2),
                fmt(m["inference"]["p95_ms"], 2),
                fmt(m["size_bytes"] / 2**20, 3),
                fmt(m["training_seconds"], 1),
            ]
            for m in models
        ],
    )
    runs_table = markdown_tables(
        ["Run", "Loss", "Augmentation", "Best epoch", "Validation metric", "Best score"],
        [
            [
                tag,
                r["training"]["loss"],
                "on" if r["training"]["augment"] else "off",
                r["best_epoch"],
                r["validation_metric"],
                fmt(r["best_validation_score"], 4),
            ]
            for tag, r in runs.items()
        ],
    )
    errors = winner["error_analysis"]
    failure = (
        f"The selected model assigned a cloud class to {percent(errors['false_cloud_rate'])} of "
        f"{errors['clear_surface_pixels']:,} labeled surface pixels. It misclassified "
        f"{percent(errors['boundary_error_rate'])} of boundary pixels and "
        f"{percent(errors['small_cloud_error_rate'])} of small-cloud pixels."
    )
    calibration = winner["calibration_validation"]
    main = [
        "# Comparative experiments",
        headline,
        measures,
        "The CNN scores individual tile labels. Segmentation accuracy counts pixels. Each architecture appears with its strongest validation run. Model selection and temperature fitting use validation data.",
        "## Eight validation runs",
        runs_table,
        "## Temporal and spatial errors",
        failure,
        markdown_tables(
            ["Test month", "Valid pixels", "Pixel accuracy", "mIoU"],
            [
                [t["month"], f"{t['count']:,}", fmt(t["accuracy"]), fmt(t["score"])]
                for t in winner["temporal"]
            ],
        ),
        markdown_tables(
            ["Reference", "Predicted", "Pixels"],
            [[i, j, f"{v:,}"] for i, j, v in confusions(winner)],
        ),
        "See the lowest, median and highest accuracy tiles for each of the three segmenters in [the figures directory](../experiments/figures/).",
        "## Calibration and compute",
        f"Fitting a temperature of {fmt(calibration['temperature'])} changed validation ECE from {fmt(calibration['before']['ece'])} to {fmt(calibration['after']['ece'])}. Test ECE was {fmt(winner['calibration_test']['ece'])}.",
        f"{report['hardware']['cpu']}, {report['hardware']['threads']} threads, {report['hardware']['device']}. Timing covers one 64 × 64 forward pass. Peak GPU allocation is unmeasured for this CPU run. All counts, calibration bins, confusion matrices and fingerprints are in [comparison.json](../experiments/comparison.json).",
        "## Metric leaders",
        markdown_tables(
            ["Metric", "Model"],
            [[k.replace("_", " "), MODELS[v]] for k, v in report["leaders"].items()],
        ),
    ]
    (docs / "experiments.md").write_text("\n\n".join(main) + "\n")
    architectures = {
        "cnn": "Three convolution, batch-normalization, ReLU and pooling stages; global pooling and an eleven-class head.",
        "unet": "Two encoder levels and a bridge, skip concatenation, bilinear upsampling and a pixel classifier.",
        "unet_plus": "A widened U-Net with learned residual projections, batch normalization, skip connections and bottleneck dropout.",
        "segformer": "Four overlapping patch stages, spatial-reduction attention, depthwise Mix-FFNs and a multiscale decoder. Trained from scratch.",
    }
    cards = [
        "# Model cards",
        "These models support research on calibrated European satellite infrared data. All four use the same fixed Kelvin normalization and validity channel. See the [data card](data-card.md) for class definitions and the [methodology](methodology.md) for selection and scoring.",
    ]
    for m in models:
        c = m["calibration_validation"]
        cls = m["task"] == "classification"
        cards += [
            "## " + MODELS[m["model"]],
            architectures[m["model"]],
            f"**Version** `{m['version']}` · **Run** `{m['run']}` · **Parameters** {m['parameters']:,} · **Checkpoint** {m['size_bytes']:,} bytes.",
            f"Selected at epoch {runs[m['run']]['best_epoch']} on validation {runs[m['run']]['validation_metric']} of {fmt(m['validation_score'])}. Test accuracy {fmt(m['accuracy'])}, macro precision {fmt(m['macro_precision'])}, macro recall {fmt(m['macro_recall'])} and macro F1 {fmt(m['macro_f1'])}."
            + (
                ""
                if cls
                else f" Test mIoU {fmt(m['miou'])}, mean Dice {fmt(m['mean_dice'])} and derived dominant-tile macro F1 {fmt(m['dominant_image_macro_f1'])}."
            ),
            f"Validation temperature {fmt(c['temperature'])}. ECE before/after {fmt(c['before']['ece'])} / {fmt(c['after']['ece'])}. Held-out ECE {fmt(m['calibration_test']['ece'])}.",
            markdown_tables(
                [
                    "Class",
                    "Reference support",
                    "Predicted",
                    "Precision",
                    "Recall",
                    *(["F1"] if cls else ["IoU", "Dice"]),
                ],
                [
                    [
                        v["name"],
                        v["support"],
                        v["predicted"],
                        fmt(v["precision"]),
                        fmt(v["recall"]),
                        *([fmt(v["f1"])] if cls else [fmt(v["iou"]), fmt(v["dice"])]),
                    ]
                    for v in m["per_class"]
                ],
            ),
            (
                "Grad-CAM highlights spatial evidence for the chosen image label. An entirely clear tile is rare or absent on this sampled benchmark. Whole-frame resizing changes the physical scale."
                if cls
                else f"False-cloud rate {percent(m['error_analysis']['false_cloud_rate'])}; small-cloud pixel error {percent(m['error_analysis']['small_cloud_error_rate'])}. Classes absent from both reference and prediction have undefined overlap. See the per-class support before interpreting a zero or missing score."
            ),
        ]
    (docs / "model-cards.md").write_text("\n\n".join(cards) + "\n")
    (docs / "limitations.md").write_text(
        "\n\n".join(
            [
                "# Research limitations",
                failure,
                "CloudCast is a retrieved reference product and GridSat is an independent observation. Residual timing differences, parallax, mixed pixels, sensor footprints and reference errors can all affect agreement. The land-mask check supports native grid orientation.",
                "Sampling covers 84 dates in one European year and three UTC hours per sampled date. The test covers November and December 2017. Seasons and chronology change together. Tiles from the same scene are correlated. Single-seed, single-region estimates have no confidence intervals here. Cross-year and geographic generalization remain unmeasured.",
                "The single calibrated infrared band carries limited information about cloud height, optical thickness and surface type. Thin cloud and snow often have weak separation. Sea ice has zero reference pixels in the test, and the model recovered no pixels of the semitransparent-above-low/mid class. Confidence calibration can shift in a new season.",
                "Small components crossing tile edges can be split by the component analysis. Disjoint inference tiles can introduce seams. A CNN applied to a larger frame has a coarser effective input scale. Percentages measure the fraction of observed projected-grid pixels and do not correct for geographic area.",
                "For research and demonstration, inspect the corresponding reference, class support, calibration and failure examples before interpreting a scene's estimated cloud coverage. Operational weather decisions require further validation. A separate GPU run is needed for accelerator latency, memory and concurrency measurements.",
                "Forecasting and continuous ingestion are future extensions in the PRD. The release analyzes individual observations.",
            ]
        )
        + "\n"
    )
    return headline


class PDF:
    def __init__(self, path):
        for name, weight in [("Interpretable", "normal"), ("InterpretableBold", "bold")]:
            font = font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Sans", weight=weight)
            )
            pdfmetrics.registerFont(TTFont(name, font))
        self.w, self.h = A4
        self.c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
        self.c.setTitle("CloudVision | European cloud classification and segmentation benchmark")
        self.c.setAuthor("CloudVision")
        self.c.setSubject("Held-out classification, segmentation, calibration and error analysis")
        self.c.setCreator("CloudVision benchmark reporting")
        self.left = 42
        self.width = self.w - 84
        self.ink = "#152c37"
        self.teal = "#136e63"
        self.muted = "#435d68"

    def text(self, x, y, txt, size=10, bold=False, color=None):
        self.c.setFillColor(HexColor(color or self.ink))
        self.c.setFont("InterpretableBold" if bold else "Interpretable", size)
        self.c.drawString(x, self.h - y - size, str(txt))

    def rect(self, x, y, w, h, color):
        self.c.setFillColor(HexColor(color))
        self.c.rect(x, self.h - y - h, w, h, fill=1, stroke=0)

    def para(self, y, txt, *, x=None, w=None, size=10, leading=15):
        x = self.left if x is None else x
        w = self.width if w is None else w
        line = ""
        for word in txt.split():
            candidate = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(candidate, "Interpretable", size) > w:
                self.text(x, y, line, size)
                y += leading
                line = word
            else:
                line = candidate
        if line:
            self.text(x, y, line, size)
            y += leading
        return y

    def start(self, page, kicker, title):
        self.rect(0, 0, self.w, 7, "#207f72")
        self.text(self.left, 28, "CLOUDVISION   /   RESEARCH BENCHMARK", 8, True, self.teal)
        self.text(self.w - 120, 28, "EUROPE  ·  2017", 8, True, self.muted)
        self.text(self.left, 65, kicker.upper(), 9, True, self.teal)
        self.text(self.left, 82, title, 24, True)
        self.rect(self.left, 798, self.width, 0.8, "#bdd0ce")
        self.text(
            self.left,
            808,
            "CloudCast × NOAA GridSat-B1    /    cloudcast-gridsat-europe-2017-v1",
            7,
            color=self.muted,
        )
        self.text(self.w - 72, 806, f"{page} / 4", 8, True, self.teal)

    def heading(self, y, title):
        self.text(self.left, y, title, 14, True)

    def table(self, y, headers, rows, cols, step=25, size=9):
        self.rect(self.left, y, self.width, step, "#173a43")
        for x, h in zip(cols, headers):
            self.text(self.left + x, y + 6, h, 8, True, "#ffffff")
        y += step
        for i, row in enumerate(rows):
            self.rect(self.left, y, self.width, step, "#eaf2ef" if i % 2 == 0 else "#f6f8f6")
            for x, txt in zip(cols, row):
                self.text(self.left + x, y + (step - size) / 2 - 1, txt, size)
            y += step
        return y

    def image(self, path, y, width=None):
        width = width or self.width
        with Image.open(path) as im:
            height = width * im.height / im.width
        self.c.drawImage(str(path), self.left, self.h - y - height, width, height)
        return y + height

    def metric(self, x, y, value, label):
        self.rect(x, y, 161, 59, "#e8f3ef")
        self.text(x + 11, y + 7, value, 20, True, self.teal)
        self.text(x + 11, y + 38, label, 8, True)

    def finish(self):
        self.c.showPage()


def generate(work):
    work = Path(work)
    out = work / "experiments"
    docs = work / "docs"
    docs.mkdir(exist_ok=True, parents=True)
    report = json.loads((out / "comparison.json").read_text())
    audit = json.loads((out / "data_audit.json").read_text())
    runs = {f.parent.name: json.loads(f.read_text()) for f in sorted(out.glob("*/run.json"))}
    models = report["models"]
    winner = next(m for m in models if m["model"] == report["recommended_segmentation"])
    cnn = next(m for m in models if m["model"] == "cnn")
    headline = write_markdown(docs, report, runs, winner)
    pdf = PDF(docs / "CloudVision-Benchmark.pdf")
    left = pdf.left
    pdf.start(1, "Independent test observations", "Satellite cloud classification")
    pdf.para(129, headline)
    pdf.metric(left, 192, fmt(winner["miou"]), "Mean intersection over union")
    pdf.metric(left + 175, 192, fmt(winner["mean_dice"]), "Mean Dice coefficient")
    pdf.metric(left + 350, 192, fmt(cnn["macro_f1"]), "CNN image-level macro F1")
    pdf.heading(276, "Held-out performance")
    pdf.table(
        305,
        ["Model", "Task", "Accuracy", "Image F1", "mIoU", "Dice"],
        [
            [
                MODELS[m["model"]],
                "Image" if m["task"] == "classification" else "Pixel",
                fmt(m["accuracy"]),
                fmt(m["macro_f1"] if m["task"] == "classification" else None),
                fmt(m.get("miou")),
                fmt(m.get("mean_dice")),
            ]
            for m in models
        ],
        [9, 134, 227, 303, 382, 445],
    )
    pdf.para(
        450,
        "Image accuracy counts correctly classified tiles. Pixel accuracy counts correctly labeled pixels. Among 122,069 labeled surface pixels, the selected model predicted cloud on "
        + percent(winner["error_analysis"]["false_cloud_rate"])
        + ". Its cloud cover estimates require caution.",
    )
    pdf.heading(518, "A representative held-out tile")
    example = winner["examples"][1]
    end = pdf.image(out / example["figure"], 548)
    pdf.para(
        end + 8,
        f"Median-accuracy tile from {example['scene']}, upper-left coordinate ({example['x']}, {example['y']}). Pixel accuracy {percent(example['accuracy'])}. Lighter values in the entropy panel indicate a less decisive class distribution.",
        size=8.8,
        leading=13,
    )
    pdf.finish()

    pdf.start(2, "Data and experimental design", "From observations to targets")
    pdf.para(
        131,
        "CloudCast provides the semantic reference field. NOAA GridSat-B1 provides an independently measured infrared-window brightness temperature. Pairing these products makes the inputs and targets distinct.",
    )
    pdf.heading(198, "Sampling and temporal separation")
    pdf.table(
        228,
        ["Partition", "First UTC date", "Last UTC date", "Scenes", "Tiles"],
        [
            [name, val["start"][:10], val["end"][:10], str(val["scenes"]), str(val["scenes"] * 4)]
            for name, val in audit["splits"].items()
            if val["scenes"]
        ],
        [9, 114, 236, 365, 434],
        step=25,
    )
    pdf.para(
        348,
        "Each 128 × 128 scene is sampled at centers on the native projected grid and paired to the nearest NOAA grid center. There are three observations per sampled day. Whole UTC days are assigned before tiling, with 78 hours between the latest scene in one split and the earliest scene in the next.",
    )
    pdf.heading(433, "Four surfaces, ten cloud types")
    pdf.para(
        464,
        "The uncollapsed source codes preserve clear land, clear sea, snow and sea ice. Reference IDs 4–13 describe cloud categories. Ignore code 255 is excluded from training, scoring and the coverage denominator. For the CNN, the most common cloudy class defines the tile target. Clear-only tiles have their own target.",
    )
    pdf.heading(555, "Independent alignment check")
    pdf.para(
        586,
        "On 1 January 2017, 3,926 of 3,966 clear-land/clear-sea pixels agreed with independent Natural Earth polygons (98.99%). Transposing the class grid gave 78.52%. This test checks coastline alignment. Sensor footprints, cloud movement and parallax can still affect individual cloudy pixels.",
    )
    pdf.heading(676, "Replicating the experiment")
    pdf.para(
        707,
        "Each scene has a checksum, UTC time, label source and radiance product ID. The release includes the measurements, masks, manifest and acquisition code. A separate audit verifies valid support, per-split frequencies, duplicate radiometry and array checksums.",
    )
    pdf.finish()

    pdf.start(3, "Residual U-Net / error analysis", "Where predictions break down")
    pdf.para(
        130,
        "Per-class scores pool all valid test pixels. A missing value means the class was absent from both the reference and model predictions.",
    )
    pdf.table(
        183,
        ["Reference class", "Support", "Precision", "Recall", "IoU", "Dice"],
        [
            [
                v["name"],
                f"{v['support']:,}",
                fmt(v["precision"]),
                fmt(v["recall"]),
                fmt(v["iou"]),
                fmt(v["dice"]),
            ]
            for v in winner["per_class"]
        ],
        [6, 203, 272, 341, 400, 455],
        step=20,
        size=7.6,
    )
    pdf.heading(506, "Spatial failure modes")
    err = winner["error_analysis"]
    pdf.table(
        538,
        ["Pixel subset", "Reference pixels", "Error measure", "Rate"],
        [
            [
                "Class boundaries",
                f"{err['boundary_pixels']:,}",
                "Incorrect class",
                percent(err["boundary_error_rate"]),
            ],
            [
                "Small cloud regions",
                f"{err['small_cloud_pixels']:,}",
                "Incorrect class",
                percent(err["small_cloud_error_rate"]),
            ],
            [
                "Clear, snow or ice",
                f"{err['clear_surface_pixels']:,}",
                "Predicted as cloud",
                percent(err["false_cloud_rate"]),
            ],
        ],
        [8, 168, 287, 442],
        step=22,
        size=8.5,
    )
    pdf.heading(648, "Frequent confusions")
    a, b = confusions(winner, 2)
    pdf.para(
        681,
        f"The largest off-diagonal entry was {a[0].lower()} classified as {a[1].lower()} ({a[2]:,} pixels). Next came {b[0].lower()} classified as {b[1].lower()} ({b[2]:,}).",
    )
    pdf.para(
        736,
        "Boundary edges use four-neighbor class changes. Small regions contain at most 16 four-connected pixels of one cloud class within a tile.",
        size=8.6,
        leading=13,
    )
    pdf.finish()

    pdf.start(4, "Model selection and calibration", "What the experiments support")
    pdf.para(
        130,
        "Eight runs used the same chronological split. The table compares U-Net-family choices using validation mIoU. Each row uses the best epoch under that configuration.",
    )
    tags = ["unet_ce", "unet_weighted_ce", "unet_dice", "unet", "unet_no_augmentation", "unet_plus"]
    labels = [
        "Cross-entropy",
        "Weighted cross-entropy",
        "Soft Dice",
        "Weighted CE + Dice",
        "Combined, augmentation off",
        "Residual U-Net, combined",
    ]
    pdf.table(
        180,
        ["Validation configuration", "Best epoch", "Validation mIoU"],
        [
            [n, str(runs[t]["best_epoch"]), fmt(runs[t]["best_validation_score"], 4)]
            for t, n in zip(tags, labels)
        ],
        [8, 268, 395],
        step=21,
        size=8.5,
    )
    pdf.heading(350, "Calibration across seasons")
    cal = winner["calibration_validation"]
    tem = winner["temporal"]
    pdf.para(
        378,
        f"Validation temperature scaling (T = {fmt(cal['temperature'])}) moved ECE from {fmt(cal['before']['ece'])} to {fmt(cal['after']['ece'])}. Held-out ECE was {fmt(winner['calibration_test']['ece'])}. Residual U-Net mIoU was {fmt(tem[0]['score'])} in {tem[0]['month']} and {fmt(tem[-1]['score'])} in {tem[-1]['month']}. These months provide one seasonal comparison.",
    )
    pdf.heading(456, "Compute cost")
    pdf.table(
        486,
        ["Model", "Median ms", "p95 ms", "Checkpoint MiB", "Train s"],
        [
            [
                MODELS[m["model"]],
                fmt(m["inference"]["median_ms"], 2),
                fmt(m["inference"]["p95_ms"], 2),
                fmt(m["size_bytes"] / 2**20, 3),
                fmt(m["training_seconds"], 1),
            ]
            for m in models
        ],
        [8, 151, 241, 314, 447],
        step=21,
        size=8.5,
    )
    pdf.para(
        609,
        f"{report['hardware']['cpu']} · {report['hardware']['threads']} CPU threads · PyTorch {report['hardware']['torch']}. Batch size one, 64 × 64 pixels, eight warm-ups and 60 timed forward passes. GPU memory was unmeasured.",
        size=8.7,
        leading=13,
    )
    pdf.heading(668, "Data and methods")
    pdf.para(
        697,
        "Nielsen, Iosifidis & Karstoft (2021), CloudCast, DOI 10.1109/JSTARS.2021.3062936. NOAA GridSat-B1 v2, DOI 10.7289/V59P2ZKR. The release includes method details, model cards, confusion matrices, calibration bins, training histories and reproducible figures.",
        size=9.1,
        leading=14,
    )
    pdf.finish()
    pdf.c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    generate(parser.parse_args().workdir)
