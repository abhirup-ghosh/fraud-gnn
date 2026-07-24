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


def _metric_value(text: str, pattern: str) -> float:
    import re

    match = re.search(re.escape(pattern) + r" ([0-9.eE+-]+)", text)
    assert match, f"metric line matching {pattern!r} not found"
    return float(match.group(1))


def test_metrics_after_many_scores_exposes_model_performance_metrics(client):
    before = client.get("/metrics").text
    count_before = _metric_value(before, 'fraud_predictions_total{label="licit"}')
    hist_before = _metric_value(before, "fraud_score_histogram_count")

    n_calls = 210  # > 200, matching §S9's acceptance criterion
    for _ in range(n_calls):
        client.post("/score", json={"features": [0.0] * 165})  # FakeEngine -> 0.3 -> licit

    after = client.get("/metrics").text
    count_after = _metric_value(after, 'fraud_predictions_total{label="licit"}')
    hist_after = _metric_value(after, "fraud_score_histogram_count")

    assert count_after - count_before == n_calls
    assert hist_after - hist_before == n_calls
    assert "fraud_flagged_ratio" in after
    assert "fraud_drift_psi" in after


def test_feedback_matched_prediction_updates_rolling_metrics(client):
    client.post("/score", json={"txId": 42})  # FakeEngine.score_known -> 0.7 -> illicit

    resp = client.post("/feedback", json={"txId": 42, "true_label": "illicit"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched"] is True
    assert body["rolling_precision"] == 1.0
    assert body["rolling_recall"] == 1.0
    assert body["rolling_f1"] == 1.0

    metrics_text = client.get("/metrics").text
    assert "fraud_rolling_precision 1.0" in metrics_text


def test_feedback_unmatched_txid_returns_matched_false(client):
    resp = client.post("/feedback", json={"txId": 999999, "true_label": "licit"})
    assert resp.status_code == 200
    assert resp.json() == {
        "matched": False,
        "rolling_precision": None,
        "rolling_recall": None,
        "rolling_f1": None,
    }


def test_feedback_invalid_label_returns_422(client):
    resp = client.post("/feedback", json={"txId": 1, "true_label": "fraud"})
    assert resp.status_code == 422
