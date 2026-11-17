"""
Export the trained Keras model to ONNX for in-browser inference (static site), and
verify the ONNX model gives the same results as the Keras original.

    python scripts/export_onnx.py

src/train.py runs this automatically at the end of training, so a retrain always
produces a matching ONNX model. Writes model/brain_tumor_cnn.onnx (committed, like the
.keras file; the tests fail if the two models disagree). Fails (non-zero exit) if any of the dashboard's 12
sample images gets a different label, or probabilities differ by more than 1e-4.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "brain_tumor_cnn.keras")
RESULTS = os.path.join(ROOT, "web", "static", "results")
OUT_DIR = os.path.join(ROOT, "model")
OUT_PATH = os.path.join(OUT_DIR, "brain_tumor_cnn.onnx")
TOLERANCE = 1e-4


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.float32)


def write_onnx(fn, path):
    """Freeze a tf.function and convert it to ONNX with named input/outputs.

    Freezing bakes every captured value into the graph as a constant (EfficientNet's
    input-normalisation constants, the Grad-CAM weights); without it tf2onnx turns them
    into extra model *inputs* the browser can't supply. Outputs are named after the
    function's output dict keys ("probabilities", "cam") so callers never rely on order.
    """
    import onnx
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    concrete = fn.get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)
    tf_inputs = [t.name for t in frozen.inputs]
    tf_outputs = [t.name for t in frozen.outputs]
    # tf.nest flattens dicts in sorted key order, the same order as frozen.outputs.
    names = dict(zip(tf_outputs, sorted(concrete.structured_outputs), strict=True))
    names[tf_inputs[0]] = "image"
    proto, _ = tf2onnx.convert.from_graph_def(
        frozen.graph.as_graph_def(), input_names=tf_inputs, output_names=tf_outputs, opset=17
    )
    for value in list(proto.graph.input) + list(proto.graph.output):
        value.name = names.get(value.name, value.name)
    for node in proto.graph.node:
        node.input[:] = [names.get(n, n) for n in node.input]
        node.output[:] = [names.get(n, n) for n in node.output]
    onnx.checker.check_model(proto)
    onnx.save(proto, path)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    IMG_SIZE = int(model.input_shape[1])
    with open(os.path.join(ROOT, "model", "class_names.json")) as f:
        class_names = json.load(f)

    # Inference-only graph: training=False removes the augmentation layers. Input is
    # float32 pixels in 0-255 (the model's own Rescaling layer normalises them).
    signature = [tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="image")]

    # Models with a GlobalAveragePooling -> Dense head (EfficientNet) also export a
    # Grad-CAM output: for that head, Grad-CAM is exactly the class activation map
    # feature_map @ dense_weights (see explain() in web/app.py), so the browser gets
    # heatmaps from a plain forward pass. Shape [batch, h, w, classes], unnormalised.
    try:
        backbone = model.get_layer("efficientnetb0")
        cam_weights = tf.constant(model.get_layer("classifier").get_weights()[0])
    except ValueError:
        backbone = None

    def keras_cam(x):
        return np.einsum("bhwk,kc->bhwc", np.asarray(backbone(x, training=False)), cam_weights)

    @tf.function(input_signature=signature)
    def serve(image):
        outputs = {"probabilities": model(image, training=False)}
        if backbone is not None:
            feats = backbone(image, training=False)
            outputs["cam"] = tf.einsum("bhwk,kc->bhwc", feats, cam_weights)
        return outputs

    os.makedirs(OUT_DIR, exist_ok=True)
    write_onnx(serve, OUT_PATH)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")

    session = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print("ONNX outputs:", output_names)

    def onnx_probs(x):
        return session.run(["probabilities"], {input_name: x})[0]

    with open(os.path.join(RESULTS, "samples.json")) as f:
        samples = json.load(f)
    batch = np.stack(
        [load_image(os.path.join(RESULTS, "samples", s["file"]), IMG_SIZE) for s in samples]
    )
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(8, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)

    failures = []
    for name, x in (("samples", batch), ("random", noise)):
        keras_p = np.asarray(model(x, training=False))
        onnx_p = onnx_probs(x)
        max_diff = float(np.max(np.abs(keras_p - onnx_p)))
        same = bool(np.all(keras_p.argmax(1) == onnx_p.argmax(1)))
        print(f"{name:8} max |keras - onnx| = {max_diff:.2e}   same labels: {same}")
        if max_diff > TOLERANCE or not same:
            failures.append(name)
        if backbone is not None:
            keras_c = keras_cam(x)
            onnx_c = session.run(["cam"], {input_name: x})[0]
            # Relative to the map's range: CAM values are unnormalised logits.
            cam_diff = float(np.max(np.abs(keras_c - onnx_c)) / (np.ptp(keras_c) + 1e-9))
            print(f"{name:8} Grad-CAM max relative diff = {cam_diff:.2e}")
            if cam_diff > 1e-3:
                failures.append(f"{name} cam")

    onnx_labels = onnx_probs(batch).argmax(1)
    for s, label in zip(samples, onnx_labels, strict=True):
        if class_names[label] != s["predicted"]:
            failures.append(f"{s['file']}: {class_names[label]} != {s['predicted']}")
    print(f"dashboard samples reproduced: {len(samples) - len(failures)}/{len(samples)}")

    if failures:
        sys.exit(f"ONNX export does NOT match Keras: {failures}")
    print("ONNX export verified.")


if __name__ == "__main__":
    main()
