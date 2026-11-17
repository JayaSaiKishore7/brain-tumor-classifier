"""
Post-deploy smoke check: exercises the real user journey against a running site.

    python scripts/smoke_check.py                              # the Hugging Face Space
    python scripts/smoke_check.py http://127.0.0.1:7860        # any other deployment

1. waits (up to --timeout seconds) for /healthz to return 200 (Space build + model load)
2. loads the results dashboard
3. uploads a known sample MRI and checks a prediction comes back

Exit code 0 = healthy; non-zero = the deployment is broken and should be rolled back.
Standard library only, so it runs anywhere.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "https://jayasaikishore-brain-tumor-classifier.hf.space"
SAMPLES = os.path.join(ROOT, "web", "static", "results")


def fetch(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for /healthz")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    start = time.time()
    status = None
    while time.time() - start < args.timeout:
        try:
            status, _ = fetch(base + "/healthz", timeout=15)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status = None
        if status == 200:
            break
        time.sleep(10)
    check("healthz", status == 200, f"status {status} after {time.time() - start:.0f}s")

    status, body = fetch(base + "/")
    check("dashboard", status == 200 and "accuracy" in body.lower(), f"status {status}")

    with open(os.path.join(SAMPLES, "samples.json")) as f:
        sample = json.load(f)[0]
    with open(os.path.join(SAMPLES, "samples", sample["file"]), "rb") as f:
        image = f.read()
    boundary = uuid.uuid4().hex
    payload = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{sample["file"]}"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, body = fetch(
        base + "/predict",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    check(
        "prediction",
        status == 200 and "data:image/jpeg;base64," in body and sample["predicted"] in body,
        f"status {status}, expected '{sample['predicted']}' in result",
    )
    print("Smoke check passed:", base)


if __name__ == "__main__":
    main()
