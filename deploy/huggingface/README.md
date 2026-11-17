---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Educational brain MRI tumor classifier (not diagnostic)
---

# Brain Tumor MRI Classifier

A small CNN trained from scratch to classify brain MRI scans as glioma, meningioma,
pituitary tumor, or no tumor (97.53% test accuracy on a held-out set).

The model runs entirely in your browser (ONNX Runtime Web): uploaded images are never
sent to a server.

**Educational project only — not a diagnostic tool. Do not use it for medical decisions.**

Source, training details and limitations: https://github.com/JayaSaiKishore7/brain-tumor-classifier

<!-- This file is the Space's README (Spaces read their config from the YAML header
above). It lives at deploy/huggingface/README.md in the GitHub repo and is copied into
the static site build by scripts/build_static.py. -->
