"""Grad-CAM checks. The app (and the browser, via the ONNX "cam" output) compute the
heatmap forward-only as feature_map @ dense_weights. That's only valid because the
model ends in GlobalAveragePooling -> Dense; these tests prove it against a real
gradient computation, so the shortcut can't silently become wrong."""

import io

import numpy as np
import pytest
from PIL import Image

from conftest import sample_path


@pytest.fixture(scope="module")
def head(app_module):
    model = app_module.get_model()
    try:
        backbone = model.get_layer(app_module.BACKBONE_LAYER)
        dense = model.get_layer(app_module.CLASSIFIER_LAYER)
    except ValueError:
        pytest.skip("model has no GlobalAveragePooling -> Dense head (CNN): no Grad-CAM")
    return model, backbone, dense


def sample_array(app_module, model, name):
    size = app_module.input_size(model)
    img = Image.open(sample_path(name)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)[None]


def test_forward_cam_equals_gradient_gradcam(app_module, head, samples):
    import tensorflow as tf

    model, backbone, dense = head
    weights, bias = dense.get_weights()
    for s in samples[:4]:
        arr = sample_array(app_module, model, s["file"])
        feats = tf.convert_to_tensor(backbone(arr, training=False))
        with tf.GradientTape() as tape:
            tape.watch(feats)
            logits = model.get_layer("gap")(feats) @ weights + bias  # pre-softmax scores
            # Sanity: this really is the model's own head (dropout is off at inference).
            np.testing.assert_allclose(
                tf.nn.softmax(logits).numpy(), model(arr, training=False).numpy(), atol=1e-5
            )
            c = int(tf.argmax(logits[0]))
            score = logits[0, c]
        grads = tape.gradient(score, feats)
        alpha = tf.reduce_mean(grads, axis=(1, 2))  # Grad-CAM channel weights
        gradcam = tf.nn.relu(tf.reduce_sum(alpha[:, None, None, :] * feats, axis=-1))[0].numpy()
        gradcam /= gradcam.max()

        cam = app_module.explain(model, arr, c)
        np.testing.assert_allclose(cam, gradcam, atol=1e-5)


def test_api_and_page_include_heatmap(client, head, samples):
    with open(sample_path(samples[0]["file"]), "rb") as f:
        data = f.read()
    r = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(data), "scan.jpg")},
        content_type="multipart/form-data",
    )
    heatmap = r.get_json()["heatmap"]
    assert heatmap["class"] == r.get_json()["label"]
    grid = np.array(heatmap["grid"])
    assert grid.ndim == 2 and grid.min() >= 0 and abs(grid.max() - 1) < 1e-6

    r = client.post(
        "/predict",
        data={"image": (io.BytesIO(data), "scan.jpg")},
        content_type="multipart/form-data",
    )
    assert "Where the model looked" in r.get_data(as_text=True)
