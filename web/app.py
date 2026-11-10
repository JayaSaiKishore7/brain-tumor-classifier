"""
Flask app that displays the brain tumor CNN's training results (accuracy,
confusion matrix, sample predictions) and lets a visitor upload an MRI image
to get a live prediction from the trained model.

Uploaded images are never written to disk: the preview is sent back inline as a
data URI, so nothing accumulates on the server.
"""

import base64
import io
import json
import os
import threading

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
STATIC_RESULTS = os.path.join(ROOT, "web", "static", "results")
IMG_SIZE = 128
PREVIEW_SIZE = 512  # max edge of the inline preview shown back to the user
MAX_UPLOAD_MB = 8

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Model + class names are cached in memory, keyed by file modification time, so a
# retrain is picked up on the next request without restarting the server.
_lock = threading.Lock()
_cache = {}


def _load_cached(path, loader):
    """Return loader(path), reloading only when the file's mtime changes.

    Returns None if the file doesn't exist. If a reload fails (e.g. train.py is
    mid-write on the checkpoint), the previously loaded value is kept.
    """
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(path)
        if cached is None or cached[0] != mtime:
            try:
                _cache[path] = (mtime, loader(path))
            except Exception:
                if cached is None:
                    raise
                app.logger.exception("Reload of %s failed; keeping previous copy", path)
                # Don't retry (and re-log) on every request; wait for the file to change.
                _cache[path] = (mtime, cached[1])
        return _cache[path][1]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_keras_model(path):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def get_model():
    return _load_cached(MODEL_PATH, _load_keras_model)


def get_class_names():
    return _load_cached(CLASS_NAMES_PATH, _load_json)


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    return _load_json(path) if os.path.exists(path) else None


def load_samples():
    path = os.path.join(STATIC_RESULTS, "samples.json")
    return _load_json(path) if os.path.exists(path) else []


def to_data_uri(img):
    preview = img.copy()
    preview.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=load_metrics(),
        samples=load_samples(),
        has_model=os.path.exists(MODEL_PATH),
        has_history=os.path.exists(os.path.join(STATIC_RESULTS, "training_history.png")),
        has_confusion=os.path.exists(os.path.join(STATIC_RESULTS, "confusion_matrix.png")),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    preview = None

    if request.method == "POST":
        file = request.files.get("image")
        model = get_model()
        class_names = get_class_names()
        if not file or file.filename == "":
            error = "Please choose an MRI image to upload."
        elif model is None or class_names is None:
            error = "No trained model found yet. Run src/train.py first, then try again."
        else:
            # Only image decoding is guarded here; errors from the model itself
            # should surface as real errors, not "bad image" messages.
            try:
                img = Image.open(io.BytesIO(file.read())).convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                img = None
                error = "Couldn't read that file as an image. Please upload a JPG or PNG."

            if img is not None:
                resized = img.resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(resized, dtype=np.uint8)[None, ...]
                # Direct call instead of model.predict(): predict() sets up a batched
                # data pipeline on every call, which is pure overhead for one image.
                # training=False keeps the augmentation layers switched off.
                probs = np.asarray(model(arr, training=False))[0]

                ranked = sorted(
                    zip(class_names, probs.tolist()), key=lambda p: p[1], reverse=True
                )
                result = {
                    "label": ranked[0][0],
                    "confidence": round(ranked[0][1] * 100, 2),
                    "breakdown": [(name, round(p * 100, 2)) for name, p in ranked],
                }
                preview = to_data_uri(img)

    return render_template("predict.html", result=result, error=error, preview=preview)


@app.errorhandler(413)
def upload_too_large(_err):
    # Without this, an oversized upload gets Flask's bare "Request Entity Too Large" page.
    error = f"That file is too large. The upload limit is {MAX_UPLOAD_MB} MB."
    return render_template("predict.html", result=None, error=error, preview=None), 413


if __name__ == "__main__":
    # The Werkzeug debugger allows arbitrary code execution from the browser, so it
    # is opt-in only: set FLASK_DEBUG=1 for local development.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False, port=5500)
