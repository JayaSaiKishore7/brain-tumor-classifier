"""
User-journey check in a real (headless) browser: open the "try it" page, wait for the
model to load, classify sample scans, and check the results match the dashboard.

    python scripts/browser_check.py               # the live Hugging Face Space
    python scripts/browser_check.py <site-url>    # e.g. a local preview

This is what smoke_check.py can't do: the prediction runs in the visitor's browser, so
only a browser can prove it works. Also fails on any JavaScript error on the page.
Runs after every deploy and on a schedule (.github/workflows/monitor.yml).

Setup (once): pip install -r requirements-dev.txt && playwright install chromium
"""

import argparse
import json
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SAMPLES_DIR = os.path.join(ROOT, "web", "static", "results")
CHECK_FILES = ("sample_00.jpg", "sample_02.jpg", "sample_04.jpg")  # incl. a low-confidence one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?")
    args = parser.parse_args()
    if args.url:
        base = args.url.rstrip("/")
    else:
        import smoke_check

        base = smoke_check.space_url().rstrip("/")

    with open(os.path.join(SAMPLES_DIR, "samples.json")) as f:
        expected = {s["file"]: s for s in json.load(f)}

    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        js_errors = []
        page.on("pageerror", lambda e: js_errors.append(str(e)))
        page.on("console", lambda m: js_errors.append(m.text) if m.type == "error" else None)

        page.goto(base + "/predict.html", wait_until="domcontentloaded")
        # Model download + WebAssembly start-up; generous for a slow CDN.
        page.wait_for_selector("#submit:not([disabled])", timeout=120_000)
        print(f"[PASS] model loaded in browser ({base})")

        for name in CHECK_FILES:
            page.evaluate("document.getElementById('result').hidden = true")
            page.set_input_files("#file", os.path.join(SAMPLES_DIR, "samples", name))
            page.click("#submit")
            page.wait_for_selector("#result:not([hidden]), #error:not([hidden])", timeout=30_000)
            if page.is_visible("#error"):
                failures.append(f"{name}: page error '{page.inner_text('#error')}'")
                continue
            label = page.inner_text("#label").strip()
            confidence = float(page.inner_text("#confidence").split("%")[0])
            want = expected[name]
            want_conf = round(want["confidence"] * 100, 2)
            ok = label == want["predicted"] and abs(confidence - want_conf) <= 0.02
            print(
                f"[{'PASS' if ok else 'FAIL'}] {name}: {label} {confidence}% "
                f"(expected {want['predicted']} {want_conf}%)"
            )
            if not ok:
                failures.append(name)

        page.set_input_files(
            "#file",
            files=[{"name": "junk.jpg", "mimeType": "image/jpeg", "buffer": b"not an image"}],
        )
        page.click("#submit")
        page.wait_for_selector("#error:not([hidden])", timeout=30_000)
        ok = "Couldn't read that file" in page.inner_text("#error")
        print(f"[{'PASS' if ok else 'FAIL'}] non-image upload shows a friendly error")
        if not ok:
            failures.append("non-image error")
        browser.close()

    if js_errors:
        failures.append(f"JavaScript errors: {js_errors}")
        print("[FAIL] JavaScript errors:", js_errors)
    if failures:
        sys.exit(f"Browser check FAILED: {failures}")
    print("Browser check passed.")


if __name__ == "__main__":
    main()
