from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {
        "Gender": 0,
        "Age": 35,
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
        "lead_days": 3.0,
        "appointment_weekday": 2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability_no_show" in body
    assert "model_version" in body