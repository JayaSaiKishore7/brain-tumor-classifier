"""
Flask app that displays the brain tumor CNN's training results (accuracy,
confusion matrix, sample predictions) and lets a visitor upload an MRI image
to get a live prediction from the trained model.
"""

import io
import json
import os
import uuid

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
STATIC_RESULTS = os.path.join(ROOT, "web", "static", "results")
UPLOAD_DIR = os.path.join(ROOT, "web", "static", "uploads")
IMG_SIZE = 128

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap

_model = None
_class_names = None


def get_class_names():
    global _class_names
    if _class_names is None:
        path = os.path.join(MODEL_DIR, "class_names.json")
        if os.path.exists(path):
            with open(path) as f:
                _class_names = json.load(f)
        else:
            _class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    return _class_names


def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
        if not os.path.exists(model_path):
            return None
        import tensorflow as tf

        _model = tf.keras.models.load_model(model_path)
    return _model


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_samples():
    path = os.path.join(STATIC_RESULTS, "samples.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@app.route("/")
def index():
    metrics = load_metrics()
    samples = load_samples()
    has_history = os.path.exists(os.path.join(STATIC_RESULTS, "training_history.png"))
    has_confusion = os.path.exists(os.path.join(STATIC_RESULTS, "confusion_matrix.png"))
    return render_template(
        "index.html",
        metrics=metrics,
        samples=samples,
        has_history=has_history,
        has_confusion=has_confusion,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    uploaded_file = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please choose an MRI image to upload."
        else:
            model = get_model()
            if model is None:
                error = "The model hasn't finished training yet. Try again shortly."
            else:
                try:
                    raw = file.read()
                    img = Image.open(io.BytesIO(raw)).convert("RGB")

                    fname = f"{uuid.uuid4().hex}.jpg"
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    img.save(os.path.join(UPLOAD_DIR, fname))
                    uploaded_file = fname

                    resized = img.resize((IMG_SIZE, IMG_SIZE))
                    arr = np.array(resized, dtype=np.uint8)[None, ...]
                    probs = model.predict(arr, verbose=0)[0]

                    class_names = get_class_names()
                    ranked = sorted(
                        zip(class_names, probs.tolist()), key=lambda p: p[1], reverse=True
                    )
                    result = {
                        "label": ranked[0][0],
                        "confidence": round(ranked[0][1] * 100, 2),
                        "breakdown": [(name, round(p * 100, 2)) for name, p in ranked],
                    }
                except Exception as exc:
                    error = f"Couldn't read that file as an image ({exc})."

    return render_template(
        "predict.html", result=result, error=error, uploaded_file=uploaded_file
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5500)
