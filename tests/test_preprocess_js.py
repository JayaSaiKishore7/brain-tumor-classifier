"""The static site resizes uploads in JavaScript (web/static/js/preprocess.js). It must
produce exactly the pixels Pillow does, or the browser model sees different input than
the model was trained and evaluated on (train/serve skew)."""

import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, sample_path

NODE = shutil.which("node")
CLI = os.path.join(ROOT, "tests", "js", "resize_cli.mjs")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def js_resize(arr, out_w, out_h, tmp_path):
    h, w, _ = arr.shape
    src, dst = tmp_path / "in.rgb", tmp_path / "out.rgb"
    src.write_bytes(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
    subprocess.run(
        [NODE, CLI, str(src), str(w), str(h), str(out_w), str(out_h), str(dst)], check=True
    )
    return np.frombuffer(dst.read_bytes(), dtype=np.uint8).reshape(out_h, out_w, 3)


def pil_resize(arr, out_w, out_h):
    return np.array(Image.fromarray(arr, "RGB").resize((out_w, out_h)))


def realistic(w, h):
    """An MRI-like image at an arbitrary size (a dashboard sample, rescaled)."""
    img = Image.open(sample_path("sample_03.jpg")).convert("RGB")
    return np.array(img.resize((w, h), Image.Resampling.LANCZOS))


@pytest.mark.parametrize(
    "w,h",
    [(512, 512), (630, 630), (225, 300), (1000, 129), (128, 128), (90, 64), (128, 512)],
)
def test_js_resize_matches_pillow_exactly(w, h, tmp_path):
    rng = np.random.default_rng(w * 7919 + h)
    for arr in (rng.integers(0, 256, (h, w, 3), dtype=np.uint8), realistic(w, h)):
        expected = pil_resize(arr, 128, 128)
        actual = js_resize(arr, 128, 128, tmp_path)
        diff = np.abs(expected.astype(int) - actual.astype(int))
        assert diff.max() == 0, f"{w}x{h}: {np.count_nonzero(diff)} pixels differ"
