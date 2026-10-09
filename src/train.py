"""
Trains a basic CNN to classify brain MRI scans into four categories:
glioma, meningioma, no tumor, pituitary.

Dataset: Hemg/Brain-Tumor-MRI-Dataset (Hugging Face), 7023 images, already
downloaded to data/brain_tumor.parquet.

Outputs (all under model/ and web/static/results/):
  model/brain_tumor_cnn.keras   - trained model
  model/class_names.json        - label index -> class name
  model/metrics.json            - test accuracy + per-class precision/recall/f1
  web/static/results/training_history.png
  web/static/results/confusion_matrix.png
  web/static/results/samples/   - a handful of labeled test images for the site
  web/static/results/samples.json
"""

import io
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "brain_tumor.parquet")
MODEL_DIR = os.path.join(ROOT, "model")
RESULTS_DIR = os.path.join(ROOT, "web", "static", "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")

IMG_SIZE = 128
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset():
    print("Loading parquet...")
    df = pd.read_parquet(DATA_PATH)

    images = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(len(df), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        img = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        images[i] = np.array(img, dtype=np.uint8)
        labels[i] = row.label
        if (i + 1) % 1000 == 0:
            print(f"  decoded {i + 1}/{len(df)}")

    print(f"Loaded {len(images)} images.")
    return images, labels


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=130)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)
    return cm


def save_sample_predictions(x_test, y_true, y_pred, probs, n=12):
    idx = np.random.choice(len(x_test), size=n, replace=False)
    samples = []
    for rank, i in enumerate(idx):
        fname = f"sample_{rank:02d}.jpg"
        Image.fromarray(x_test[i]).save(os.path.join(SAMPLES_DIR, fname))
        samples.append(
            {
                "file": fname,
                "actual": CLASS_NAMES[int(y_true[i])],
                "predicted": CLASS_NAMES[int(y_pred[i])],
                "confidence": round(float(np.max(probs[i])), 4),
                "correct": bool(y_true[i] == y_pred[i]),
            }
        )
    with open(os.path.join(RESULTS_DIR, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)


def main():
    images, labels = load_dataset()

    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    model = build_model()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "brain_tumor_cnn.keras"),
        monitor="val_accuracy",
        save_best_only=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )

    save_training_history(history)

    probs = model.predict(x_test, batch_size=32)
    y_pred = np.argmax(probs, axis=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    cm = save_confusion_matrix(y_test, y_pred)
    save_sample_predictions(x_test, y_test, y_pred, probs)

    metrics = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "epochs_run": len(history.history["loss"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    model.save(os.path.join(MODEL_DIR, "brain_tumor_cnn.keras"))
    print("Done. Model + metrics + plots saved.")


if __name__ == "__main__":
    main()
