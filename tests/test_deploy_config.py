"""Deployment config checks: catch Space misconfiguration locally, before Hugging Face
rejects a deploy or serves a broken Space."""

import os
import re

from conftest import ROOT


def space_metadata():
    with open(os.path.join(ROOT, "deploy", "huggingface", "README.md"), encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "Space README must start with a YAML header"
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def test_space_metadata_valid():
    meta = space_metadata()
    # Static Spaces are the free tier; Docker/Gradio need PRO on free accounts.
    assert meta["sdk"] == "static"
    assert meta["app_file"] == "index.html"
    # Hugging Face rejects the whole upload if these are over 60 characters.
    assert len(meta["short_description"]) <= 60
    assert len(meta["title"]) <= 60
