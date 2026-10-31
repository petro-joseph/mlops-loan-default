"""API behaviour with a stand-in model, so no MLflow server is needed."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import serve


class FakeModel:
    def predict_proba(self, df):
        p = (df["credit_amount"] / 10000).clip(0, 1).to_numpy()
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    serve.state.update(model=FakeModel(), version="7", columns=["credit_amount", "purpose"], reference=None)
    yield TestClient(serve.app)  # not a context manager: the alias watcher doesn't start
    serve.state.update(model=None, version=None, columns=[], reference=None)


def test_predict_returns_score_decision_and_version(client):
    r = client.post("/predict", json={"loans": [{"credit_amount": 1000, "purpose": "car"},
                                                 {"credit_amount": 8000, "purpose": "car"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "7"
    assert [x["decision"] for x in body["results"]] == ["approve", "refer"]


def test_missing_field_is_rejected(client):
    r = client.post("/predict", json={"loans": [{"credit_amount": 1000}]})
    assert r.status_code == 422 and "purpose" in r.json()["detail"]


def test_health_reports_version(client):
    assert client.get("/health").json()["version"] == "7"


def test_no_model_means_503():
    serve.state.update(model=None, version=None)
    c = TestClient(serve.app)
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json={"loans": [{"a": 1}]}).status_code == 503


def test_metrics_exposed(client):
    client.post("/predict", json={"loans": [{"credit_amount": 1000, "purpose": "car"}]})
    text = client.get("/metrics/").text
    assert "loan_predictions_total" in text and "loan_model_version" in text
