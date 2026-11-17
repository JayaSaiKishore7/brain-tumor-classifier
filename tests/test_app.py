"""Route tests: the user journeys a release must not break."""

import io

from conftest import sample_path


def upload(client, data, filename="scan.jpg"):
    return client.post(
        "/predict",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_dashboard_renders_metrics(client, metrics):
    r = client.get("/")
    assert r.status_code == 200
    accuracy = f"{metrics['test_accuracy'] * 100:.2f}"
    assert accuracy in r.get_data(as_text=True)


def test_dashboard_assets_served(client):
    for path in (
        "/static/style.css",
        "/static/results/confusion_matrix.png",
        "/static/results/training_history.png",
    ):
        assert client.get(path).status_code == 200, path


def test_predict_page_renders(client):
    r = client.get("/predict")
    assert r.status_code == 200
    assert "not a diagnostic tool" in r.get_data(as_text=True)


def test_predict_without_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 200
    assert "Please choose an MRI image" in r.get_data(as_text=True)


def test_predict_rejects_non_image(client):
    r = upload(client, b"definitely not an image")
    assert r.status_code == 200
    assert "Couldn&#39;t read that file as an image" in r.get_data(as_text=True)


def test_predict_rejects_oversized_upload(client, app_module):
    r = upload(client, b"0" * (app_module.MAX_UPLOAD_MB * 1024 * 1024 + 1))
    assert r.status_code == 413
    assert "too large" in r.get_data(as_text=True)


def test_predict_returns_a_class(client, samples):
    with open(sample_path(samples[0]["file"]), "rb") as f:
        r = upload(client, f.read())
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "data:image/jpeg;base64," in body  # preview rendered inline


def api_upload(client, data, filename="scan.jpg"):
    return client.post(
        "/api/predict",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_api_predict_returns_json(client, samples, metrics):
    with open(sample_path(samples[0]["file"]), "rb") as f:
        r = api_upload(client, f.read())
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == samples[0]["predicted"]
    assert abs(body["confidence"] - samples[0]["confidence"]) < 1e-4
    probs = body["probabilities"]
    assert set(probs) == set(metrics["class_names"])
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert body["confidence"] == max(probs.values())
    assert body["model"]["input_size"] > 0
    assert "Not a diagnostic tool" in body["disclaimer"]


def test_api_errors_are_json(client, app_module):
    r = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "choose an MRI image" in r.get_json()["error"]

    r = api_upload(client, b"definitely not an image")
    assert r.status_code == 400 and "Couldn't read" in r.get_json()["error"]

    r = api_upload(client, b"0" * (app_module.MAX_UPLOAD_MB * 1024 * 1024 + 1))
    assert r.status_code == 413 and "too large" in r.get_json()["error"]


def test_api_rejects_get(client):
    assert client.get("/api/predict").status_code == 405


def test_healthz_not_ready_before_model_load(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "model_is_loaded", lambda: False)
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.get_json()["model_loaded"] is False


def test_healthz_ready_after_model_load(client, app_module):
    app_module.warm_up()
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "model_loaded": True}
