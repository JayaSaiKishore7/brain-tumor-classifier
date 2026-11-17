import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SAMPLES_DIR = os.path.join(ROOT, "web", "static", "results", "samples")


@pytest.fixture(scope="session")
def app_module():
    import app

    return app


@pytest.fixture(scope="session")
def client(app_module):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def samples():
    with open(os.path.join(ROOT, "web", "static", "results", "samples.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def metrics():
    with open(os.path.join(ROOT, "model", "metrics.json")) as f:
        return json.load(f)


def sample_path(name):
    return os.path.join(SAMPLES_DIR, name)
