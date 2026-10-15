import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _png_bytes() -> bytes:
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_detections():
    response = client.post(
        "/predict", files={"file": ("blank.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["detections"], list)
    assert body["latency_ms"] > 0


def test_predict_rejects_wrong_type():
    response = client.post(
        "/predict", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_rejects_corrupt_image():
    response = client.post(
        "/predict", files={"file": ("bad.png", b"not an image", "image/png")}
    )
    assert response.status_code == 400
