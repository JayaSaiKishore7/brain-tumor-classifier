"""
Post-deploy smoke check for the static site on Hugging Face.

    python scripts/smoke_check.py              # the Space (URL looked up via the HF API)
    python scripts/smoke_check.py <site-url>   # any other copy, e.g. a local preview

Waits (up to --timeout seconds) for the site to come up, then checks that the dashboard
shows this commit's accuracy, and that the prediction page and every file it needs (the
ONNX model, the JavaScript, the stylesheet) are served, with the model byte-identical to
the local one. Exit code 0 = healthy; non-zero = broken, roll back.

The prediction itself runs in the visitor's browser, so it can't be exercised from here:
open the "try it" page and classify one image after each deploy (about 10 seconds).
Standard library only.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("HF_SPACE_ID", "JayaSaiKishore/brain-tumor-classifier")
HEADERS = {"User-Agent": "brain-tumor-classifier-smoke-check"}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def space_url():
    status, body = fetch(f"https://huggingface.co/api/spaces/{SPACE_ID}")
    if status != 200:
        sys.exit(f"Couldn't look up Space {SPACE_ID}: HTTP {status}")
    info = json.loads(body)
    host = info.get("host") or f"https://{info['subdomain']}.static.hf.space"
    return host if host.startswith("http") else f"https://{host}"


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?")
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for the site")
    args = parser.parse_args()
    base = (args.url or space_url()).rstrip("/")
    print("Checking", base)

    start = time.time()
    status, body = None, b""
    while time.time() - start < args.timeout:
        try:
            status, body = fetch(base + "/index.html", timeout=15)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status = None
        if status == 200:
            break
        time.sleep(10)
    check("site up", status == 200, f"status {status} after {time.time() - start:.0f}s")

    with open(os.path.join(ROOT, "model", "metrics.json")) as f:
        accuracy = f"{json.load(f)['test_accuracy'] * 100:.2f}%"
    check("dashboard shows this model", accuracy.encode() in body, f"expected {accuracy}")

    status, body = fetch(base + "/predict.html")
    check("prediction page", status == 200 and b"onnxruntime-web@" in body, f"status {status}")

    for path in ("static/style.css", "static/js/predict.js", "static/js/preprocess.js"):
        status, _ = fetch(f"{base}/{path}")
        check(path, status == 200, f"status {status}")

    with open(os.path.join(ROOT, "model", "brain_tumor_cnn.onnx"), "rb") as f:
        local = hashlib.sha256(f.read()).hexdigest()
    status, body = fetch(base + "/model/brain_tumor_cnn.onnx", timeout=120)
    remote = hashlib.sha256(body).hexdigest() if status == 200 else None
    check("model served, matches local", remote == local, f"status {status}")

    print("Smoke check passed:", base)
    print(f"Now open {base}/predict.html and classify one image.")


if __name__ == "__main__":
    main()
