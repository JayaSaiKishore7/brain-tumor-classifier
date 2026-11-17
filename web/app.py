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
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
STATIC_RESULTS = os.path.join(ROOT, "web", "static", "results")
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


def model_is_loaded():
    with _lock:
        return MODEL_PATH in _cache and CLASS_NAMES_PATH in _cache


def warm_up():
    """Load the model ahead of the first request (TensorFlow import + load can take
    20-60 s on a cold start). Called from serve.py in a background thread."""
    try:
        get_class_names()
        get_model()
        app.logger.info("Model warm-up complete")
    except Exception:
        app.logger.exception("Model warm-up failed")


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


MSG_NO_FILE = "Please choose an MRI image to upload."
MSG_NO_MODEL = "No trained model found yet. Run src/train.py first, then try again."
MSG_BAD_IMAGE = "Couldn't read that file as an image. Please upload a JPG or PNG."
DISCLAIMER = "Educational project only. Not a diagnostic tool; not for medical decisions."


class PredictionError(Exception):
    """A request problem with a user-facing message and an HTTP status for the API."""

    def __init__(self, message, status):
        super().__init__(message)
        self.message, self.status = message, status


def input_size(model):
    """The model's own input size (e.g. 128 for the CNN, 160 for EfficientNet)."""
    return int(model.input_shape[1])


BACKBONE_LAYER, CLASSIFIER_LAYER = "efficientnetb0", "classifier"
# Heatmap colours, low -> high. web/static/js/predict.js uses the same stops.
HEATMAP_STOPS = [(0, 0, 128), (0, 160, 255), (60, 220, 120), (255, 220, 0), (230, 30, 30)]


def explain(model, arr, class_index):
    """Grad-CAM heatmap for one class, as a small grid normalised to 0-1.

    The EfficientNet model ends in GlobalAveragePooling -> Dense, so the gradient of a
    class score w.r.t. the last feature map is just that class's Dense weights / (h*w):
    Grad-CAM reduces exactly to the class activation map sum_k w_k * A_k (Zhou et al.
    2016), with no backprop needed (tests/test_gradcam.py checks this against a real
    gradient computation). Returns None for models without that head (the CNN).
    """
    try:
        backbone = model.get_layer(BACKBONE_LAYER)
        weights = model.get_layer(CLASSIFIER_LAYER).get_weights()[0]
    except ValueError:
        return None
    feats = np.asarray(backbone(arr, training=False))[0]  # (h, w, channels)
    cam = np.maximum(feats @ weights[:, class_index], 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def heatmap_overlay(img, cam):
    """Blend an upsampled, colour-mapped heatmap over the uploaded image (preview size)."""
    base = img.copy()
    base.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    grid = Image.fromarray(np.uint8(cam * 255), "L").resize(base.size, Image.Resampling.BILINEAR)
    t = np.asarray(grid, dtype=np.float32) / 255 * (len(HEATMAP_STOPS) - 1)
    lo = np.clip(np.floor(t).astype(int), 0, len(HEATMAP_STOPS) - 2)
    stops = np.array(HEATMAP_STOPS, dtype=np.float32)
    frac = (t - lo)[..., None]
    colours = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    # Low-attention areas stay mostly see-through so the anatomy remains visible.
    alpha = 0.15 + 0.5 * np.asarray(grid, dtype=np.float32)[..., None] / 255
    blended = np.asarray(base, dtype=np.float32) * (1 - alpha) + colours * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def run_prediction(file):
    """Shared by the web page and the JSON API: uploaded file -> (PIL image, ranked, cam).

    ranked is [(class name, probability)], most likely first; cam is the Grad-CAM grid
    for the top class (None if the model doesn't support it). Raises PredictionError.
    """
    # Check for a file before touching the model, so an empty submission never
    # triggers the (slow, cold-start) model load.
    if not file or file.filename == "":
        raise PredictionError(MSG_NO_FILE, 400)
    model = get_model()
    class_names = get_class_names()
    if model is None or class_names is None:
        raise PredictionError(MSG_NO_MODEL, 503)
    # Only image decoding is guarded here; errors from the model itself should surface
    # as real errors, not "bad image" messages.
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise PredictionError(MSG_BAD_IMAGE, 400) from e

    size = input_size(model)
    arr = np.array(img.resize((size, size)), dtype=np.uint8)[None, ...]
    # Direct call instead of model.predict(): predict() sets up a batched data pipeline
    # on every call, which is pure overhead for one image. training=False keeps the
    # augmentation layers switched off.
    probs = np.asarray(model(arr, training=False))[0]
    # strict=True: a class-count mismatch between model and class_names.json must fail
    # loudly, not silently drop a class.
    ranked = sorted(zip(class_names, probs.tolist(), strict=True), key=lambda p: p[1], reverse=True)
    cam = explain(model, arr, int(np.argmax(probs)))
    return img, ranked, cam


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    preview = None

    heatmap = None
    if request.method == "POST":
        try:
            img, ranked, cam = run_prediction(request.files.get("image"))
        except PredictionError as e:
            error = e.message
        else:
            result = {
                "label": ranked[0][0],
                "confidence": round(ranked[0][1] * 100, 2),
                "breakdown": [(name, round(p * 100, 2)) for name, p in ranked],
            }
            preview = to_data_uri(img)
            if cam is not None:
                heatmap = to_data_uri(heatmap_overlay(img, cam))

    return render_template(
        "predict.html", result=result, error=error, preview=preview, heatmap=heatmap
    )


@app.post("/api/predict")
def api_predict():
    """JSON API for scripts and agents.

        curl -F image=@scan.jpg http://127.0.0.1:5500/api/predict

    200: {"label", "confidence", "probabilities": {class: p}, "heatmap", "model",
          "disclaimer"}; heatmap is {"class", "grid": rows of 0-1 values} (Grad-CAM for
          the predicted class, grid over the whole image) or null if unsupported.
    400 / 413 / 503: {"error": message}. Probabilities are 0-1, not percentages.
    """
    try:
        _img, ranked, cam = run_prediction(request.files.get("image"))
    except PredictionError as e:
        return jsonify(error=e.message), e.status
    metrics = load_metrics() or {}
    heatmap = None
    if cam is not None:
        heatmap = {"class": ranked[0][0], "grid": np.round(cam, 3).tolist()}
    return jsonify(
        heatmap=heatmap,
        label=ranked[0][0],
        confidence=round(ranked[0][1], 6),
        probabilities={name: round(p, 6) for name, p in ranked},
        model={
            "arch": metrics.get("arch", "cnn"),
            "input_size": input_size(get_model()),
            "test_accuracy": metrics.get("test_accuracy"),
        },
        disclaimer=DISCLAIMER,
    )


@app.route("/healthz")
def healthz():
    """Readiness check: 200 only once the model is in memory and can serve predictions.

    Deliberately doesn't trigger a model load itself, so health probes stay cheap.
    """
    if model_is_loaded():
        return {"status": "ok", "model_loaded": True}, 200
    return {"status": "starting", "model_loaded": False}, 503


@app.errorhandler(413)
def upload_too_large(_err):
    # Without this, an oversized upload gets Flask's bare "Request Entity Too Large" page.
    error = f"That file is too large. The upload limit is {MAX_UPLOAD_MB} MB."
    if request.path.startswith("/api/"):
        return jsonify(error=error), 413
    return render_template("predict.html", result=None, error=error, preview=None), 413


if __name__ == "__main__":
    # The Werkzeug debugger allows arbitrary code execution from the browser, so it
    # is opt-in only: set FLASK_DEBUG=1 for local development.
    # This is the development server; production uses serve.py (waitress).
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5500")),
    )
