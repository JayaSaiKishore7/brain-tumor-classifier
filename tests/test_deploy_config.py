"""Deployment config checks: catch Space/Docker misconfiguration locally, before
Hugging Face rejects a deploy or builds a container that can't be reached."""

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
    assert meta["sdk"] == "docker"
    # Hugging Face rejects the whole upload if this is over 60 characters.
    assert len(meta["short_description"]) <= 60
    assert len(meta["title"]) <= 60


def test_space_port_matches_dockerfile():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        dockerfile = f.read()
    port = space_metadata()["app_port"]
    assert f"PORT={port}" in dockerfile
    assert f"EXPOSE {port}" in dockerfile
