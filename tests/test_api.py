"""
Integration tests for the FastAPI service. These require a trained model
at models/model.pkl (produced by `python src/train.py`); if it's absent,
the tests are skipped rather than failed, so the "unit test" CI stage
(which runs before the "model training" stage) isn't blocked by it.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.pkl not found -- run `python src/train.py` first",
)

SAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_well_formed_response(client):
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease_present", "no_disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["model_name"], str)


def test_predict_rejects_invalid_payload(client):
    bad_payload = dict(SAMPLE_PATIENT)
    bad_payload["sex"] = 7  # out of allowed range [0, 1]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    bad_payload = dict(SAMPLE_PATIENT)
    del bad_payload["age"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_metrics_endpoint_is_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests" in resp.content or b"# HELP" in resp.content
