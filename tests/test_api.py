"""§S8 acceptance tests: monkeypatched model + fake store, via httpx TestClient."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_gnn.serve import app as app_module


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.threshold = 0.5
        self.model_version = "test-version"

    def score_known(self, tx_id, feature_client=None):
        return 0.7

    def score_inductive(self, features, neighbor_features):
        return 0.3


def fake_get_redis_client(*args, **kwargs):
    return None


def fake_get_features(tx_id, client=None):
    return np.zeros(165, dtype=np.float32)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "InferenceEngine", FakeEngine)
    monkeypatch.setattr(app_module, "get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(app_module, "get_features", fake_get_features)
    with TestClient(app_module.app) as c:
        yield c


def test_health_returns_model_version(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test-version"
    assert body["status"] == "ok"


def test_score_with_txid(client):
    resp = client.post("/score", json={"txId": 123})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["illicit_probability"] <= 1.0
    assert body["label"] in ("illicit", "licit")


def test_score_with_features(client):
    resp = client.post("/score", json={"features": [0.0] * 165})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["illicit_probability"] <= 1.0


def test_score_with_neither_field_returns_422(client):
    resp = client.post("/score", json={})
    assert resp.status_code == 422


def test_metrics_exposes_drift_and_http_latency(client):
    client.post("/score", json={"txId": 1})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "fraud_drift_psi" in resp.text
    assert "http_request_duration_seconds" in resp.text
