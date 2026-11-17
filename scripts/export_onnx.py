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
IMG_SIZE = 128
TOLERANCE = 1e-4


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(os.path.join(ROOT, "model", "class_names.json")) as f:
        class_names = json.load(f)

    # Inference-only graph: training=False removes the augmentation layers. Input is
    # float32 pixels in 0-255 (the model's own Rescaling layer normalises them).
    signature = [tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="image")]

    @tf.function(input_signature=signature)
    def serve(image):
        return {"probabilities": model(image, training=False)}

    os.makedirs(OUT_DIR, exist_ok=True)
    tf2onnx.convert.from_function(serve, input_signature=signature, opset=17, output_path=OUT_PATH)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")

    session = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(os.path.join(RESULTS, "samples.json")) as f:
        samples = json.load(f)
    batch = np.stack([load_image(os.path.join(RESULTS, "samples", s["file"])) for s in samples])
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(8, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)

    failures = []
    for name, x in (("samples", batch), ("random", noise)):
        keras_p = np.asarray(model(x, training=False))
        onnx_p = session.run(None, {input_name: x})[0]
        max_diff = float(np.max(np.abs(keras_p - onnx_p)))
        same = bool(np.all(keras_p.argmax(1) == onnx_p.argmax(1)))
        print(f"{name:8} max |keras - onnx| = {max_diff:.2e}   same labels: {same}")
        if max_diff > TOLERANCE or not same:
            failures.append(name)

    onnx_labels = session.run(None, {input_name: batch})[0].argmax(1)
    for s, label in zip(samples, onnx_labels, strict=True):
        if class_names[label] != s["predicted"]:
            failures.append(f"{s['file']}: {class_names[label]} != {s['predicted']}")
    print(f"dashboard samples reproduced: {len(samples) - len(failures)}/{len(samples)}")

    if failures:
        sys.exit(f"ONNX export does NOT match Keras: {failures}")
    print("ONNX export verified.")


if __name__ == "__main__":
    main()
