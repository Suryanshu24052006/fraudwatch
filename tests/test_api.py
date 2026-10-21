import pytest
from fastapi.testclient import TestClient

from fraudwatch import api

NORMAL = {"amount": 25.0, "hour": 14, **{f"v{i}": 0.0 for i in range(1, 29)}}


@pytest.fixture()
def client(model):
    original = api.load
    api.load = lambda: (model, {"review": 0.3})
    yield TestClient(api.app)
    api.load = original


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_normal_transaction_allowed(client):
    r = client.post("/score", json=NORMAL)
    assert r.status_code == 200
    assert r.json()["decision"] == "allow"
    assert r.json()["latency_ms"] >= 0


def test_suspicious_transaction_flagged(client):
    r = client.post("/score", json={**NORMAL, "v14": -8.0, "v4": 5.0}).json()
    assert r["decision"] in {"review", "block"}
    assert r["fraud_probability"] > 0.3


def test_bad_input(client):
    assert client.post("/score", json={**NORMAL, "hour": 30}).status_code == 422
    assert client.post("/score", json={**NORMAL, "amount": -5}).status_code == 422
    missing = dict(NORMAL)
    missing.pop("v7")
    assert client.post("/score", json=missing).status_code == 422
