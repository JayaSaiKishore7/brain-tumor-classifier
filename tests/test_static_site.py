"""Static site release gate: the ONNX model the browser runs must agree with the Keras
model, and the build must produce a self-contained site with working relative links."""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

from conftest import ROOT, sample_path

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_static

ONNX_PATH = os.path.join(ROOT, "model", "brain_tumor_cnn.onnx")


def test_onnx_model_matches_keras(app_module, samples):
    """Fails if the model was retrained but scripts/export_onnx.py wasn't re-run."""
    model = app_module.get_model()
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    batch = np.stack(
        [
            np.array(Image.open(sample_path(s["file"])).convert("RGB").resize((128, 128)))
            for s in samples
        ]
    ).astype(np.float32)
    keras_p = np.asarray(model(batch, training=False))
    onnx_p = session.run(None, {session.get_inputs()[0].name: batch})[0]
    assert np.max(np.abs(keras_p - onnx_p)) < 1e-4
    assert (keras_p.argmax(1) == onnx_p.argmax(1)).all()


def test_static_build(tmp_path):
    out = build_static.main(tmp_path / "site")
    for rel in (
        "index.html",
        "predict.html",
        "README.md",
        "model/brain_tumor_cnn.onnx",
        "static/style.css",
        "static/js/predict.js",
        "static/js/preprocess.js",
        "static/results/confusion_matrix.png",
    ):
        assert os.path.isfile(os.path.join(out, rel)), rel

    with open(os.path.join(ROOT, "model", "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(out, "index.html"), encoding="utf-8") as f:
        index = f.read()
    assert f"{metrics['test_accuracy'] * 100:.2f}" in index
    assert 'href="predict.html"' in index

    with open(os.path.join(out, "predict.html"), encoding="utf-8") as f:
        predict = f.read()
    assert f'integrity="{build_static.ORT_INTEGRITY}"' in predict
    assert f"onnxruntime-web@{build_static.ORT_VERSION}" in predict
    assert json.dumps(metrics["class_names"]) in predict
    assert "not a diagnostic tool" in predict

    # Every relative src/href in the pages must point at a file in the build.
    import re

    for page in (index, predict):
        for ref in re.findall(r'(?:src|href)="([^"#:]+)"', page):
            assert os.path.isfile(os.path.join(out, ref)), f"broken link: {ref}"
