from fastapi.testclient import TestClient
from src.canteen_api.main import app, model

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction(monkeypatch):
    monkeypatch.setattr(
        model,
        "predict",
        lambda data: [450.0]
    )

    payload = {
        "students_on_campus": 3500,
        "menu_price_rs": 75,
        "temp_c": 30,
        "exam_days_left": 15,
        "special_dishes": 3,
        "queue_minutes_yesterday": 20,
        "day_type": "weekday",
        "block": "A-Block"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 450.0


def test_invalid_temperature():
    payload = {
        "students_on_campus": 3500,
        "menu_price_rs": 75,
        "temp_c": 80,
        "exam_days_left": 15,
        "special_dishes": 3,
        "queue_minutes_yesterday": 20,
        "day_type": "weekday",
        "block": "A-Block"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_menu_price():
    payload = {
        "students_on_campus": 3500,
        "menu_price_rs": 10,
        "temp_c": 30,
        "exam_days_left": 15,
        "special_dishes": 3,
        "queue_minutes_yesterday": 20,
        "day_type": "weekday",
        "block": "A-Block"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422