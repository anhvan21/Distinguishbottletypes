from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    type: str | None = None  # from the optional classifier, e.g. "Wine Bottle"
    type_confidence: float | None = None


class PredictionResponse(BaseModel):
    detections: list[Detection]
    latency_ms: float
    annotated_image: str
