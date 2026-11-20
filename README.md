# Brain Tumor MRI Classifier

A CNN that classifies brain MRI scans as **glioma**, **meningioma**, **pituitary tumor**
or **no tumor**.

**Live demo:** https://jayasaikishore-brain-tumor-classifier.static.hf.space
(the model runs in your browser; your scan never leaves your device)

> **Educational project only, not a diagnostic tool.**

## Results

**97.53% test accuracy** on 1,054 held-out images.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 98.3% | 95.5% | 96.9% |
| meningioma | 95.2% | 96.0% | 95.6% |
| no tumor | 98.4% | 100.0% | 99.2% |
| pituitary | 98.1% | 98.1% | 98.1% |

![Confusion matrix](web/static/results/confusion_matrix.png)

## How it works

- **Data:** [Hemg/Brain-Tumor-MRI-Dataset](https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset)
  (7,023 images), split 70% train / 15% validation / 15% test.
- **Model:** a small CNN (4 conv blocks, 1.3M parameters), 128×128 input, trained from
  scratch with Adam and early stopping. Augmentation is kept mild (±7° rotation, ±5% zoom
  and brightness), because stronger augmentation lowered accuracy on MRI scans.
- **Transfer learning was tried too.** ResNet-50 got 97.06% and EfficientNetB0 got
  96.77% on the same test set. Both are within noise of the CNN and much larger, so the
  CNN was kept.

| Model | Parameters | Test accuracy |
|---|---|---|
| **CNN (deployed)** | 1.3 M | **97.53%** |
| ResNet-50 | 23.6 M | 97.06% |
| EfficientNetB0 | 4.1 M | 96.77% |

## Run it

Requires Python 3.13.

```bash
pip install -r requirements-dev.txt

# Download the dataset (~155 MB)
curl -L -o data/brain_tumor.parquet \
  https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset/resolve/main/data/train-00000-of-00001.parquet

python src/train.py          # train the CNN (~15-20 min on CPU)
python web/app.py            # open http://localhost:5500
pytest                       # run the tests
```

Other architectures: `python src/train.py --arch resnet50 --out build/experiments/resnet50`
(or `--arch efficientnet`). Always pass `--out` so the deployed model isn't replaced.

## Project layout

```
src/train.py     training
model/           trained model (.keras + .onnx for the browser)
web/             Flask app and static-site templates
scripts/         ONNX export, static build, Hugging Face deploy
tests/           pytest suite
```

## Limitations

- Small public dataset, single train/test split, no external validation.
- Any image gets one of the four labels, even if it isn't a brain MRI.
