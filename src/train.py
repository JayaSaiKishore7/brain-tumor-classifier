"""
Trains a classifier for brain MRI scans: glioma, meningioma, no tumor, pituitary.

    python src/train.py                         # the CNN (the adopted model), writes model/
    python src/train.py --arch efficientnet     # transfer learning (experiment, not adopted)
    python src/train.py --arch efficientnet --out build/exp/effnet
                                                # experiment: write everything elsewhere
    python src/train.py --arch efficientnet --out build/exp/effnet --resume 11 --batch-size 16
                                                # continue an interrupted run, less memory
    python src/train.py --arch resnet50 --out build/exp/resnet50
                                                # ResNet-50 experiment (--resume works too)

Architectures:
  efficientnet  EfficientNetB0 pretrained on ImageNet (transfer learning), 160x160 input,
                head = GlobalAveragePooling -> Dropout -> Dense. That head makes Grad-CAM
                exact and forward-only (see scripts/export_onnx.py), so heatmaps also
                work in the browser.
  resnet50      ResNet-50 pretrained on ImageNet (transfer learning, experiment only),
                224x224 input, same head and same 2-phase training as efficientnet.
                ResNet-50 has no built-in input scaling, so the model starts with a
                ResNetPreprocess layer; it still takes raw 0-255 pixels like the others.
                Loading a saved resnet50 model needs that layer, i.e. `import train`.
  cnn           4-block CNN trained from scratch, 128x128 input. The default and the
                deployed model.

Dataset: Hemg/Brain-Tumor-MRI-Dataset (Hugging Face), 7023 images, downloaded to
data/brain_tumor.parquet. The 70/15/15 split is identical for every architecture and
input size (fixed seed, stratified), so results are directly comparable.

Outputs (default: model/ and web/static/results/; with --out: <out>/model, <out>/results):
  model/brain_tumor_cnn.keras   - trained model
  model/brain_tumor_cnn.onnx    - same model for the in-browser site (default output only;
                                  verified to match)
  model/class_names.json        - label index -> class name
  model/metrics.json            - architecture, input size, val + test metrics
  results/training_history.png, results/confusion_matrix.png
  results/samples/ + samples.json - labeled test images for the site
"""

import argparse
import io
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "brain_tumor.parquet")
DEFAULT_MODEL_DIR = os.path.join(ROOT, "model")
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "web", "static", "results")
MODEL_FILE = "brain_tumor_cnn.keras"

# The CNN is the adopted model (see the README's transfer-learning experiment), so a plain
# `python src/train.py` retrains it; EfficientNet is opt-in with --arch efficientnet.
DEFAULT_ARCH = "cnn"
IMG_SIZES = {"efficientnet": 160, "resnet50": 224, "cnn": 128}
SEED = 42
BATCH_SIZE = 32  # --batch-size

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def read_class_names():
    """Label index -> class name, read from the dataset's own Hugging Face metadata.

    This is the single source of truth for class names: it's written to
    model/class_names.json, which is what the Flask app reads.
    """
    meta = pq.read_schema(DATA_PATH).metadata[b"huggingface"]
    return json.loads(meta)["info"]["features"]["label"]["names"]


def load_dataset(img_size):
    print("Loading parquet...")
    df = pd.read_parquet(DATA_PATH)

    images = np.zeros((len(df), img_size, img_size, 3), dtype=np.uint8)
    labels = np.zeros(len(df), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        # Same resize as the app and the browser (Pillow bicubic): no train/serve skew.
        img = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
        img = img.resize((img_size, img_size))
        images[i] = np.array(img, dtype=np.uint8)
        labels[i] = row.label
        if (i + 1) % 1000 == 0:
            print(f"  decoded {i + 1}/{len(df)}")

    print(f"Loaded {len(images)} images at {img_size}x{img_size}.")
    return images, labels


def augmentation():
    # Mild on purpose: MRI orientation and intensity carry diagnostic signal (see README).
    # These layers are only active during fit(); inference is unaffected.
    return [
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.RandomBrightness(0.05, value_range=(0, 255)),
    ]


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_cnn(num_classes, img_size):
    model = models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            *augmentation(),
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
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    compile_model(model, 1e-3)
    return model


def build_efficientnet(num_classes, img_size):
    """EfficientNetB0 backbone (ImageNet weights, frozen at first) + a linear head.

    The backbone includes its own input scaling, so the model takes raw 0-255 pixels,
    exactly like the CNN; the app and browser preprocessing don't change.
    """
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    for layer in augmentation():
        x = layer(x)
    # training=False keeps BatchNorm statistics frozen, also during fine-tuning (the
    # standard recipe; updating them on a small dataset hurts accuracy).
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = models.Model(inputs, outputs, name="efficientnet_mri")
    compile_model(model, 1e-3)
    return model


@tf.keras.utils.register_keras_serializable(package="brain_tumor")
class ResNetPreprocess(layers.Layer):
    """Runs resnet50.preprocess_input inside the model: RGB -> BGR, then subtract the
    ImageNet mean per channel (no division). That is the input ResNet-50's ImageNet
    weights expect. A named, registered layer (not a Lambda) so the model can be saved
    and loaded again (ModelCheckpoint, --resume) without Keras's unsafe-Lambda error."""

    def call(self, x):
        # Cast first: uint8 pixels minus a float mean would fail or wrap around.
        x = tf.cast(x, self.compute_dtype)
        return tf.keras.applications.resnet50.preprocess_input(x)


def build_resnet50(num_classes, img_size):
    """ResNet-50 backbone (ImageNet weights, frozen at first) + the same head as
    build_efficientnet. Takes raw 0-255 pixels; ResNetPreprocess does the scaling."""
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    for layer in augmentation():
        x = layer(x)
    # Preprocess after augmentation: RandomBrightness works on 0-255 values.
    x = ResNetPreprocess(name="preprocess")(x)
    x = backbone(x, training=False)  # BatchNorm frozen, as in build_efficientnet
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = models.Model(inputs, outputs, name="resnet50_mri")
    compile_model(model, 1e-3)
    return model


# Per transfer-learning architecture: how to build it, the backbone's layer name inside
# the model, and the first block that gets unfrozen for fine-tuning.
TRANSFER = {
    "efficientnet": {
        "build": build_efficientnet,
        "backbone": "efficientnetb0",
        "first_block": "block6a",
    },
    "resnet50": {"build": build_resnet50, "backbone": "resnet50", "first_block": "conv5_block1"},
}


def unfreeze_top(model, backbone_name="efficientnetb0", first_block="block6a"):
    """Fine-tuning phase: make the backbone's top blocks trainable (BatchNorm stays frozen)."""
    backbone = model.get_layer(backbone_name)
    backbone.trainable = True
    trainable = False
    for layer in backbone.layers:
        if layer.name.startswith(first_block):
            trainable = True
        layer.trainable = trainable and not isinstance(layer, layers.BatchNormalization)
    compile_model(model, 1e-4)


HISTORY_KEYS = ("accuracy", "loss", "val_accuracy", "val_loss")


def save_history_json(history, boundary, path):
    with open(path, "w") as f:
        json.dump({"history": history, "boundary": boundary}, f)


def fit(model, data, epochs, patience, checkpoint_path, run, initial_best=None):
    """model.fit with early stopping + best-checkpointing. Each epoch is appended to
    run["history"] and saved to run["path"] immediately, so an interrupted run can be
    resumed (--resume) with its full training curve. The checkpoint is only overwritten
    by an epoch that beats initial_best (when resuming, the checkpoint's own score)."""
    x_train, y_train, x_val, y_val = data

    def log_epoch(_epoch, logs):
        for key in HISTORY_KEYS:
            run["history"][key].append(float(logs[key]))
        save_history_json(run["history"], run["boundary"], run["path"])

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=patience, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                initial_value_threshold=initial_best,
            ),
            callbacks.LambdaCallback(on_epoch_end=log_epoch),
        ],
        verbose=2,
    ).history


def train_head_on_cached_features(
    model, data, num_classes, backbone_name="efficientnetb0", epochs=40
):
    """Phase 1, fast path: while the backbone is frozen its pooled features never change,
    so compute them once and train the same Dropout -> Dense head on them (seconds per
    epoch instead of minutes). No augmentation in this phase; fine-tuning uses it.
    The trained head weights are copied into the full model's "classifier" layer."""
    x_train, y_train, x_val, y_val = data
    backbone = model.get_layer(backbone_name)
    gap = layers.GlobalAveragePooling2D()
    # The backbone alone skips any preprocessing layer that sits in front of it in the
    # model (ResNet-50 has one, EfficientNet doesn't), so apply that here as well.
    layer_names = {layer.name for layer in model.layers}
    if "preprocess" in layer_names:
        preprocess = model.get_layer("preprocess")
    else:
        preprocess = lambda x: x

    def features(x):
        return np.concatenate(
            [
                gap(backbone(preprocess(x[i : i + 64]), training=False)).numpy()
                for i in range(0, len(x), 64)
            ]
        )

    print("  computing backbone features once...")
    f_train, f_val = features(x_train), features(x_val)
    head = models.Sequential(
        [
            layers.Input(shape=(f_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    compile_model(head, 1e-3)
    history = head.fit(
        f_train,
        y_train,
        validation_data=(f_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
        ],
        verbose=2,
    ).history
    model.get_layer("classifier").set_weights(head.layers[-1].get_weights())
    return history


def val_accuracy(model, data):
    _, _, x_val, y_val = data
    probs = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    return float(np.mean(np.argmax(probs, 1) == y_val))


def train(arch, num_classes, img_size, data, checkpoint_path, history_path):
    """Returns (model, history dict of per-epoch lists, phase boundary epoch or None)."""
    if arch == "cnn":
        model = build_cnn(num_classes, img_size)
        model.summary()
        run = {"history": {k: [] for k in HISTORY_KEYS}, "boundary": None, "path": history_path}
        fit(model, data, 35, 8, checkpoint_path, run)
        return model, run["history"], None

    # efficientnet and resnet50: the same 2-phase transfer-learning flow.
    cfg = TRANSFER[arch]
    model = cfg["build"](num_classes, img_size)
    model.summary()
    print("Phase 1: train the head on the frozen pretrained backbone")
    history = train_head_on_cached_features(model, data, num_classes, cfg["backbone"])
    boundary = len(history["loss"])
    save_history_json(history, boundary, history_path)
    phase1_best, phase1_weights = max(history["val_accuracy"]), model.get_weights()
    print("Phase 2: fine-tune the top of the backbone")
    unfreeze_top(model, cfg["backbone"], cfg["first_block"])
    run = {"history": history, "boundary": boundary, "path": history_path}
    fine = fit(model, data, 25, 6, checkpoint_path, run)
    if max(fine["val_accuracy"]) < phase1_best:
        print("Fine-tuning didn't beat phase 1 on validation; keeping phase 1 weights.")
        model.set_weights(phase1_weights)
    return model, history, boundary


def resume(arch, data, checkpoint_path, history_path, epochs):
    """Continue an interrupted EfficientNet or ResNet-50 fine-tuning run from its best
    checkpoint.

    Only an epoch that beats the checkpoint's validation accuracy replaces it; if none
    does, the checkpoint itself is the result."""
    cfg = TRANSFER[arch]
    model = tf.keras.models.load_model(checkpoint_path)
    with open(history_path) as f:
        saved = json.load(f)
    history, boundary = saved["history"], saved["boundary"]
    unfreeze_top(model, cfg["backbone"], cfg["first_block"])
    best = val_accuracy(model, data)
    print(f"Resuming fine-tuning from {checkpoint_path}: val accuracy {best:.4f}")
    run = {"history": history, "boundary": boundary, "path": history_path}
    fine = fit(model, data, epochs, 6, checkpoint_path, run, initial_best=best)
    if max(fine["val_accuracy"]) <= best:
        print("No epoch beat the checkpoint; using the checkpoint.")
        model = tf.keras.models.load_model(checkpoint_path)
    return model, history, boundary


def save_training_history(history, boundary, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(history[metric], label="train")
        ax.plot(history["val_" + metric], label="val")
        if boundary:
            ax.axvline(boundary - 0.5, color="gray", linestyle="--", linewidth=1)
            ax.text(boundary - 0.4, ax.get_ylim()[1], " fine-tune", va="top", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_history.png"), dpi=130)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, class_names, results_dir):
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)
    return cm


def save_sample_predictions(model, x_test, y_true, class_names, results_dir, n=12):
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    # Remove images from previous runs so a smaller n doesn't leave stale files behind.
    for old in os.listdir(samples_dir):
        if old.startswith("sample_") and old.endswith(".jpg"):
            os.remove(os.path.join(samples_dir, old))

    idx = np.random.choice(len(x_test), size=n, replace=False)
    files = [f"sample_{rank:02d}.jpg" for rank in range(n)]
    for fname, i in zip(files, idx, strict=True):
        Image.fromarray(x_test[i]).save(os.path.join(samples_dir, fname))

    # Predict on the saved JPEGs, not the raw arrays: JPEG is lossy, so the gallery must
    # show the confidence a visitor gets when they classify that exact picture.
    shown = np.stack(
        [np.array(Image.open(os.path.join(samples_dir, f)).convert("RGB")) for f in files]
    )
    probs = np.asarray(model(shown, training=False))
    samples = []
    for fname, i, p in zip(files, idx, probs, strict=True):
        predicted = int(np.argmax(p))
        samples.append(
            {
                "file": fname,
                "actual": class_names[int(y_true[i])],
                "predicted": class_names[predicted],
                "confidence": round(float(np.max(p)), 4),
                "correct": bool(y_true[i] == predicted),
            }
        )
    with open(os.path.join(results_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(IMG_SIZES), default=DEFAULT_ARCH)
    parser.add_argument("--img-size", type=int, help="default: per architecture")
    parser.add_argument("--out", help="write outputs here instead of model/ + web/static/")
    parser.add_argument(
        "--resume",
        type=int,
        metavar="EPOCHS",
        help="continue an interrupted efficientnet/resnet50 fine-tuning run (same --out) "
        "from its best checkpoint, for up to EPOCHS more epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="lower it (e.g. 16) to cut peak memory"
    )
    args = parser.parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    img_size = args.img_size or IMG_SIZES[args.arch]
    if args.out:
        model_dir = os.path.join(args.out, "model")
        results_dir = os.path.join(args.out, "results")
    else:
        model_dir, results_dir = DEFAULT_MODEL_DIR, DEFAULT_RESULTS_DIR
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    class_names = read_class_names()
    print(f"Classes: {class_names}   arch={args.arch}   img_size={img_size}")
    images, labels = load_dataset(img_size)
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError("Dataset labels don't match the class names in its metadata.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    del images
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    model_path = os.path.join(model_dir, MODEL_FILE)
    history_path = os.path.join(model_dir, "history.json")
    data = (x_train, y_train, x_val, y_val)
    if args.resume:
        if args.arch not in TRANSFER:
            sys.exit("--resume is only for efficientnet/resnet50 fine-tuning runs.")
        model, history, boundary = resume(args.arch, data, model_path, history_path, args.resume)
    else:
        model, history, boundary = train(
            args.arch, len(class_names), img_size, data, model_path, history_path
        )
    save_training_history(history, boundary, results_dir)

    # Model selection used validation data only; the test set is touched once, here.
    val_acc = val_accuracy(model, data)
    probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    test_acc = float(np.mean(y_pred == y_test))
    test_loss = float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(y_test, probs)))
    print(f"Validation accuracy: {val_acc:.4f}   Test accuracy: {test_acc:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
    )
    cm = save_confusion_matrix(y_test, y_pred, class_names, results_dir)
    save_sample_predictions(model, x_test, y_test, class_names, results_dir)

    metrics = {
        "arch": args.arch,
        "img_size": img_size,
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "epochs_run": len(history["loss"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    model.save(model_path)
    print(f"Model + metrics + plots saved to {model_dir} and {results_dir}.")

    if not args.out:
        # Keep the browser (ONNX) model in sync: export it and verify it matches Keras.
        # Exits non-zero if they disagree, so a retrain can't leave the static site stale.
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import export_onnx

        export_onnx.main()
    print("Done.")


if __name__ == "__main__":
    main()
