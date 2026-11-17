"""
Build the static site (for a free Hugging Face *static* Space) into build/static_site/.

    python scripts/build_static.py

The dashboard is rendered by the real Flask app (same templates, same data), and the
"try it" page runs the ONNX model in the visitor's browser (web/static/js/predict.js).
No TensorFlow needed: only the Flask templates, the result files and model/*.onnx.

Preview locally:  python -m http.server 8000 --directory build/static_site
"""

import os
import shutil
import sys

from flask import render_template

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web"))

import app as webapp

OUT = os.path.join(ROOT, "build", "static_site")
ONNX_MODEL = os.path.join(ROOT, "model", "brain_tumor_cnn.onnx")
SPACE_README = os.path.join(ROOT, "deploy", "huggingface", "README.md")

# onnxruntime-web, loaded from jsDelivr. Pinned version + Subresource Integrity hash, so
# the browser refuses the script if the CDN ever serves different bytes. To upgrade:
# bump the version, recompute: curl -s <url> | openssl dgst -sha384 -binary | base64
ORT_VERSION = "1.31.1"
ORT_INTEGRITY = "sha384-tNxIwEqSlcdvKtUte98AsccPbDKM958LBXB/+3BLS42TTihXHKTWgdqPqXI/lot8"


def relink(html):
    """Flask URLs are root-absolute (/static/..., /predict); make them relative files."""
    html = html.replace('="/static/', '="static/')
    html = html.replace('href="/predict"', 'href="predict.html"')
    html = html.replace('href="/"', 'href="index.html"')
    return html


def write(out, name, html):
    leftover = [s for s in ('="/static/', 'href="/"', 'href="/predict"') if s in html]
    if leftover:
        sys.exit(f"{name}: unconverted absolute links {leftover}")
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write(html)


def main(out=OUT):
    if not os.path.exists(ONNX_MODEL):
        sys.exit(f"{ONNX_MODEL} missing; run scripts/export_onnx.py first.")
    class_names = webapp.get_class_names()
    if class_names is None:
        sys.exit("model/class_names.json missing.")

    out = str(out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "model"))

    r = webapp.app.test_client().get("/")
    if r.status_code != 200:
        sys.exit(f"Rendering the dashboard failed: HTTP {r.status_code}")
    write(out, "index.html", relink(r.get_data(as_text=True)))

    with webapp.app.test_request_context("/predict"):
        predict = render_template(
            "predict_static.html",
            class_names=class_names,
            img_size=webapp.IMG_SIZE,
            max_upload_mb=webapp.MAX_UPLOAD_MB,
            ort_version=ORT_VERSION,
            ort_integrity=ORT_INTEGRITY,
        )
    write(out, "predict.html", relink(predict))

    shutil.copytree(
        os.path.join(ROOT, "web", "static"),
        os.path.join(out, "static"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(ONNX_MODEL, os.path.join(out, "model", "brain_tumor_cnn.onnx"))
    shutil.copy2(SPACE_README, os.path.join(out, "README.md"))

    files = sum(len(f) for _, _, f in os.walk(out))
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"Built {out}: {files} files, {size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    main()
