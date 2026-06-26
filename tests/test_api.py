from fastapi.testclient import TestClient

from readmission.api.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_bad_types():
    # time_in_hospital must be an int; a non-numeric string is a 422.
    response = client.post("/predict", json={"time_in_hospital": "not-a-number"})
    assert response.status_code == 422


def test_predict_returns_score_or_503():
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "time_in_hospital": 5,
        "num_medications": 12,
        "diag_1": "428",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["risk_band"] in {"Low", "Medium", "High"}
