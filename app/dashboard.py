"""Streamlit research workspace. All inference and performance figures come from the REST API."""

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from src.schema import COLORS, MODELS, NAMES, ROOT, decode, preview

BASE = os.environ.get("CLOUDVISION_API", "http://127.0.0.1:8000").rstrip("/")
PAGES = [
    "Overview",
    "Cloud analysis",
    "Segmentation explorer",
    "Model comparison",
    "Uncertainty",
    "Documentation",
]
SAMPLES = ["20170107T1800Z.png", "20170115T0900Z.png", "20170211T0000Z.png"]
st.set_page_config(
    page_title="CloudVision · Satellite cloud analysis",
    page_icon=str(ROOT / "app/mark.png"),
    layout="wide",
)
st.markdown(
    """<style>.block-container{max-width:1360px;padding-top:4rem;padding-bottom:3rem}
[data-testid=stSidebar]{background:#10202e;border-right:1px solid #29414d}
h1,h2,h3{letter-spacing:-.035em}.brand{font-size:1.9rem;font-weight:750;letter-spacing:-.06em}.brand b{color:#82d9bd}
.kicker{color:#82d9bd;font:600 .76rem monospace;letter-spacing:.08em;text-transform:uppercase;margin:0}
[data-testid=stSidebar] label[data-baseweb=radio]{padding:.26rem 0;white-space:nowrap}
[data-testid=stMetricValue]{font-size:1.6rem}.chip{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;margin-right:.45rem}
@media(max-width:650px){.block-container{padding:4rem 1rem 3rem}}</style>""",
    unsafe_allow_html=True,
)


def go(name):
    st.session_state.nav = name


def nav(label, name, key=None):
    st.button(label, on_click=go, args=(name,), key=key)


def title(label, text):
    st.markdown(f'<p class="kicker">{label}</p>', unsafe_allow_html=True)
    st.title(text)


@st.cache_data(ttl=45, show_spinner=False)
def get(path):
    try:
        r = requests.get(BASE + path, timeout=15)
        r.raise_for_status()
        return r.json()
    except (ValueError, requests.RequestException):
        raise ValueError(
            "The analysis service is unavailable. Check the API connection and retry."
        ) from None


def predict(data, filename, name, mc):
    try:
        r = requests.post(
            BASE + "/predict",
            files={"file": (filename, data)},
            data={"model": name, "mc_passes": mc, "include_maps": "true"},
            timeout=180,
        )
        if r.status_code >= 400:
            detail = f"The analysis service returned HTTP {r.status_code}."
            try:
                detail = r.json().get("detail", detail)
            except ValueError:
                pass
            raise ValueError(detail)
        return r.json()
    except requests.RequestException:
        raise ValueError("The inference service could not be reached.") from None


def pixels(enc):
    with Image.open(io.BytesIO(base64.b64decode(enc))) as im:
        return np.asarray(im).copy()


def unit(enc):
    a = pixels(enc)
    return np.where(a > 0, (a.astype(float) - 1) / 65534, np.nan)


def heat(a, cmap):
    out = (matplotlib.colormaps[cmap](np.nan_to_num(a))[:, :, :3] * 255).astype(np.uint8)
    out[~np.isfinite(a)] = [20, 33, 44]
    return out


def bars(data, field, label):
    d = pd.DataFrame(data)
    palette = d.color.tolist() if "color" in d else ["#71cab7"] * len(d)
    chart = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(field + ":Q", title=label),
            y=alt.Y("name:N", title=None, sort="-x", axis=alt.Axis(labelLimit=225)),
            color=alt.Color(
                "name:N", legend=None, scale=alt.Scale(domain=d.name.tolist(), range=palette)
            ),
            tooltip=["name", alt.Tooltip(field + ":Q", format=".3f")],
        )
        .properties(height=max(180, len(d) * 22))
    )
    st.altair_chart(chart, width="stretch")


def exports(r):
    prefix = "-".join(
        (r["metadata"]["model"], r["metadata"]["version"], r["metadata"]["input_sha256"][:8])
    )
    a = r["images"]
    with st.expander("Download results"):
        left, h = st.columns(2)
        left.download_button(
            "Metadata · JSON",
            json.dumps({k: r[k] for k in ("metadata", "summary", "classes")}, indent=2),
            prefix + ".json",
            "application/json",
        )
        h.download_button(
            "Class statistics · CSV", r["statistics_csv"], prefix + ".csv", "text/csv"
        )
        if "mask" in a:
            left.download_button(
                "Class IDs · PNG", base64.b64decode(a["mask"]), prefix + "-mask.png", "image/png"
            )
            h.download_button(
                "Overlay · PNG",
                base64.b64decode(a["overlay"]),
                prefix + "-overlay.png",
                "image/png",
            )
        else:
            left.download_button(
                "Grad-CAM · PNG",
                base64.b64decode(a["gradcam"]),
                prefix + "-gradcam.png",
                "image/png",
            )


def home():
    title("European observations · 2017", "CloudVision")
    a, b = st.columns([0.9, 1.1], gap="large", vertical_alignment="center")
    with a:
        st.header("Cloud cover over Europe")
        st.write(
            "Analyze satellite infrared measurements, inspect cloud types at individual pixels, and compare predictions with held-out observations."
        )
        st.button("Analyze an observation", type="primary", on_click=go, args=("Cloud analysis",))
        left, h = st.columns(2)
        left.metric("Cloud types", "10")
        h.metric("Surface classes", "4")
    b.image(
        str(ROOT / "examples/europe-thermal.jpg"),
        caption="GridSat-B1 · 7 January 2017, 18 UTC. Colder regions appear lighter.",
        width="stretch",
    )
    st.divider()
    for col, n, desc in zip(
        st.columns(3),
        ["Cloud analysis", "Model comparison", "Uncertainty"],
        [
            "Upload calibrated infrared data or use a real observation.",
            "Inspect class overlap, confusion and measured latency.",
            "Explore entropy, calibration and class probability.",
        ],
    ):
        with col:
            st.subheader(n)
            st.write(desc)
            nav("Open " + n.lower(), n, key="go" + n)


def analysis():
    title("Observation", "Cloud analysis")
    catalog = get("/models")
    names = [r["architecture"] for r in catalog["models"]]
    left, h = st.columns([1, 1], gap="large")
    with left:
        source = st.radio("Input source", ["Example observation", "Upload file"], horizontal=True)
        if source == "Example observation":
            filename = st.selectbox("Observation", SAMPLES)
            data = (ROOT / "examples" / filename).read_bytes()
        else:
            f = st.file_uploader("Thermal infrared observation", type=["npy", "png", "tif", "tiff"])
            st.caption(
                "Float Kelvin NPY or 16-bit centikelvin PNG/TIFF. Use NaN or integer zero for missing data."
            )
            if f is None:
                return
            filename, data = f.name, f.getvalue()
        a = decode(data, filename)
        name = st.selectbox(
            "Model",
            names,
            index=names.index(catalog["default"]) if catalog["default"] in names else 0,
            format_func=MODELS.get,
        )
        with st.expander("Uncertainty settings"):
            mc = st.slider("Monte Carlo dropout passes", 1, 12, 1)
        run = st.button("Run analysis", type="primary")
    h.image(
        preview(a),
        caption=f"Infrared input · {a.shape[1]} × {a.shape[0]} px · {np.nanmin(a):.1f}–{np.nanmax(a):.1f} K",
        width="stretch",
    )
    if run:
        with st.spinner("Running model inference…"):
            st.session_state.result = predict(data, filename, name, mc)
            st.session_state.filename = filename
    r = st.session_state.get("result")
    if (
        not r
        or r["metadata"]["model"] != name
        or r["metadata"]["input_sha256"] != hashlib.sha256(data).hexdigest()
        or r["metadata"]["mc_passes"] != mc
    ):
        return
    s = r["summary"]
    st.divider()
    st.subheader(s["dominant"])
    st.caption(
        f"{MODELS[name]} · {r['metadata']['version']} · {r['metadata']['valid_pixels']:,} valid pixels"
    )
    cols = st.columns(3)
    cols[0].metric("Dominant-class confidence", f"{s['dominant_confidence']:.1%}")
    if "mask" in r["images"]:
        cols[1].metric("Cloud coverage", f"{s['cloud_coverage_pct']:.1f}%")
        cols[2].metric("Non-cloud coverage", f"{s['noncloud_coverage_pct']:.1f}%")
        st.image(base64.b64decode(r["images"]["overlay"]), caption="Prediction overlay", width=330)
        nav("Explore segmentation", "Segmentation explorer")
    else:
        cols[1].metric("Normalized entropy", f"{s['mean_entropy']:.3f}")
        bars(r["classes"], "probability", "Image-label probability")
        st.image(
            heat(unit(r["images"]["gradcam"]), "inferno"),
            caption="Grad-CAM · positive attribution",
            width=310,
        )
    exports(r)


def explorer():
    title("Spatial analysis", "Segmentation explorer")
    r = st.session_state.get("result")
    if not r or "mask" not in r["images"]:
        st.info("Run a segmentation model first.")
        nav("Go to cloud analysis", "Cloud analysis")
        return
    v = r["images"]
    st.caption(
        f"{st.session_state.filename} · {MODELS[r['metadata']['model']]} · {r['metadata']['version']}"
    )
    compare, inspect, distribution = st.tabs(
        ["Compare layers", "Pixel inspector", "Class distribution"]
    )
    with compare:
        pos = st.slider("Original / overlay divider", 0, 100, 50, format="%d%%")
        a = pixels(v["original"])
        o = pixels(v["overlay"])
        i = round(a.shape[1] * pos / 100)
        o[:, :i] = a[:, :i]
        if 0 < i < a.shape[1]:
            o[:, i] = [255, 251, 210]
        left, h = st.columns(2)
        left.image(o, caption="Original infrared / prediction overlay", width="stretch")
        h.image(base64.b64decode(v["segmentation"]), caption="Predicted class map", width="stretch")
        with st.expander("Class legend", expanded=True):
            cols = st.columns(2)
            for i, (n, c) in enumerate(zip(NAMES, COLORS)):
                cols[i % 2].markdown(
                    f'<span class="chip" style="background:{c}"></span>{n}', unsafe_allow_html=True
                )
    with inspect:
        h, w = r["metadata"]["input_shape"]
        a, b = st.columns(2)
        x = a.number_input("Column (x)", 0, w - 1, w // 2)
        y = b.number_input("Row (y)", 0, h - 1, h // 2)
        idx = int(pixels(v["mask"])[y, x])
        if idx == 255:
            st.info("No observation at this pixel.")
        else:
            st.metric("Pixel class", NAMES[idx])
            bars(
                [
                    dict(name=n, color=c, probability=unit(p)[y, x])
                    for n, c, p in zip(NAMES, COLORS, v["probabilities"])
                ],
                "probability",
                "Pixel probability",
            )
    with distribution:
        bars(r["classes"], "coverage_pct", "Share of observed pixels (%)")
        st.dataframe(pd.DataFrame(r["classes"]), hide_index=True, width="stretch")
    exports(r)


def compare():
    title("Held-out evaluation", "Model comparison")
    r = get("/metrics")
    st.caption(
        f"{r['test_scenes']} later scenes · {r['test_tiles']} tiles · {r['test_start'][:10]} to {r['test_end'][:10]}"
    )
    names = [m["model"] for m in r["models"]]
    selection = st.multiselect("Models to compare", names, default=names, format_func=MODELS.get)
    ms = [m for m in r["models"] if m["model"] in selection]
    if not ms:
        st.info("Select an architecture.")
        return
    df = pd.DataFrame(
        [
            dict(
                Model=MODELS[m["model"]],
                Task=m["task"],
                Accuracy=m["accuracy"],
                Image_F1=m["macro_f1"] if m["task"] == "classification" else None,
                mIoU=m.get("miou"),
                Dice=m.get("mean_dice"),
                Median_ms=m["inference"]["median_ms"],
                MiB=m["size_bytes"] / 2**20,
            )
            for m in ms
        ]
    )
    st.dataframe(df, hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.subheader("Inference time")
    a.bar_chart(df, x="Model", y="Median_ms", color="#73cbb9", height=220)
    sdf = df.dropna(subset=["mIoU"])
    b.subheader("Segmentation overlap")
    if not sdf.empty:
        b.bar_chart(sdf, x="Model", y=["mIoU", "Dice"], stack=False, height=220)
    n = st.selectbox("Inspect model", selection, format_func=MODELS.get)
    m = next(m for m in ms if m["model"] == n)
    a, b = st.tabs(["Per-class metrics", "Confusion matrix"])
    with a:
        st.dataframe(pd.DataFrame(m["per_class"]), hide_index=True, width="stretch")
    with b:
        labels = [row["name"] for row in m["per_class"]]
        d = pd.DataFrame(
            [
                dict(reference=labels[i], predicted=labels[j], count=c, share=c / max(1, sum(row)))
                for i, row in enumerate(m["confusion_matrix"])
                for j, c in enumerate(row)
            ]
        )
        st.altair_chart(
            alt.Chart(d)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", sort=labels),
                y=alt.Y("reference:N", sort=labels),
                color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], scheme="tealblues")),
                tooltip=["reference", "predicted", "count", "share"],
            )
            .properties(height=365),
            width="stretch",
        )
    if m.get("examples"):
        ex = st.selectbox(
            "Representative tile",
            m["examples"],
            format_func=lambda e: f"{e['scene']} · {e['accuracy']:.1%} correct",
        )
        path = ROOT / "experiments" / ex["figure"]
        if path.exists():
            st.image(str(path), width="stretch")
    st.caption(
        f"Warm batch-one forward pass · 64 × 64 pixels · {r['hardware']['cpu']} · {r['hardware']['threads']} threads"
    )


def uncertainty():
    title("Prediction diagnostics", "Uncertainty")
    r = st.session_state.get("result")
    if not r:
        st.info("Analyze an observation first.")
        nav("Go to cloud analysis", "Cloud analysis")
        return
    v = r["images"]
    if "uncertainty" in v:
        u = unit(v["uncertainty"])
        conf = unit(v["confidence"])
        threshold = st.slider(
            "Highlight pixels with normalized entropy above", 0.0, 1.0, 0.65, 0.01
        )
        a = pixels(v["original"])
        hit = u > threshold
        a[hit] = (0.45 * a[hit] + 0.55 * np.array([255, 145, 100])).astype(np.uint8)
        left, h = st.columns(2)
        left.image(
            heat(u, "magma"), caption="Normalized entropy · brighter means higher", width="stretch"
        )
        h.image(
            a,
            caption=f"Highlighted · {hit.sum() / np.isfinite(u).sum():.1%} of valid pixels",
            width="stretch",
        )
        counts, edges = np.histogram(conf[np.isfinite(conf)], bins=12, range=(0, 1))
        st.subheader("Maximum class probability")
        st.bar_chart(
            pd.DataFrame(dict(Probability=edges[:-1].round(2), Pixels=counts)),
            x="Probability",
            y="Pixels",
            height=200,
            color="#7bd4ba",
        )
        cls = st.selectbox("Class probability map", range(14), format_func=NAMES.__getitem__)
        st.image(
            heat(unit(v["probabilities"][cls]), "viridis"),
            caption=NAMES[cls] + " probability",
            width=300,
        )
    else:
        st.image(heat(unit(v["gradcam"]), "inferno"), caption="Grad-CAM", width=320)
    m = next(m for m in get("/metrics")["models"] if m["model"] == r["metadata"]["model"])
    cal = m["calibration_validation"]
    st.subheader("Temperature scaling")
    cols = st.columns(3)
    for col, (label, n) in zip(
        cols,
        [
            ("Validation ECE before", cal["before"]["ece"]),
            ("Validation ECE after", cal["after"]["ece"]),
            ("Temperature", cal["temperature"]),
        ],
    ):
        col.metric(label, f"{n:.3f}")
    st.caption(
        "ECE compares observed correctness with confidence across 15 bins. Individual predictions can still be wrong."
    )


def docs():
    title("Reference", "Documentation")
    for label, desc in [
        (
            "Input and output",
            "Upload near-11 µm brightness temperatures in Kelvin or 16-bit centikelvin. Integer zero and NaN denote missing pixels. Mask IDs 0–3 represent surfaces, 4–13 cloud classes and 255 missing data.",
        ),
        (
            "Data and split",
            "The release pairs NOAA GridSat-B1 infrared measurements with CloudCast raw labels. All chips from one UTC day stay in the same partition. The latest observations are held out.",
        ),
        (
            "Architectures",
            "CNN predicts an image label. U-Net, residual U-Net and a compact four-stage SegFormer predict spatial classes. Each prediction identifies the checkpoint and preprocessing version.",
        ),
        (
            "Coverage and confidence",
            "Coverage divides predicted cloud pixels by observed pixels. Confidence averages softmax values over pixels assigned to each class. Grad-CAM explains the CNN, while entropy and class maps show segmentation uncertainty.",
        ),
        (
            "Limitations",
            "Single-band observations cannot reliably distinguish all surface and cloud types. Results apply to a temporally sampled European 2017 benchmark. Consult the measured error rates and model cards before interpreting coverage.",
        ),
    ]:
        with st.expander(label, expanded=label == "Input and output"):
            st.write(desc)
    st.markdown(
        "[CloudCast · Aarhus University](https://vision.eng.au.dk/cloudcast-dataset/)  \n[NOAA GridSat-B1](https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature)  \n[Nielsen, Iosifidis & Karstoft (2021)](https://doi.org/10.1109/JSTARS.2021.3062936)"
    )
    if os.getenv("CLOUDVISION_REPO_URL"):
        st.link_button("Project repository", os.environ["CLOUDVISION_REPO_URL"])


with st.sidebar:
    st.image(str(ROOT / "app/mark.png"), width=43)
    st.markdown('<div class="brand">Cloud<b>Vision</b></div>', unsafe_allow_html=True)
    st.caption("Satellite cloud analysis")
    st.divider()
    st.radio("Workspace", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    st.caption("CloudCast × GridSat-B1")
try:
    dict(zip(PAGES, (home, analysis, explorer, compare, uncertainty, docs)))[st.session_state.nav]()
except ValueError as exc:
    st.error(str(exc))
