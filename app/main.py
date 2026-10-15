import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.detector import SmartVisionDetector
from app.schemas import PredictionResponse

app = FastAPI(
    title="Smart Vision Inspection API",
    version="0.1.0",
)

# Đặt MODEL_PATH=models/product_best.pt để dùng model đã fine-tune.
# Đặt TARGET_CLASSES="bottle" (nhiều lớp: "bottle,cup") để chỉ giữ các lớp đó.
target_classes = [
    name.strip() for name in os.getenv("TARGET_CLASSES", "").split(",") if name.strip()
]
# Đặt CLASSIFIER_PATH=models/bottle_types_best.pt để phân loại từng vật thể tìm được.
detector = SmartVisionDetector(
    os.getenv("MODEL_PATH", "yolo26n.pt"),
    target_classes=target_classes,
    classifier_path=os.getenv("CLASSIFIER_PATH") or None,
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Plain `def` (not async): FastAPI runs it in a thread pool, so the
# blocking YOLO inference does not freeze the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG and WEBP images are supported",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image must be non-empty and no larger than 10 MB",
        )

    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(content)
        temp_path = Path(temp.name)

    try:
        return detector.predict(str(temp_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image inference failed") from exc
    finally:
        temp_path.unlink(missing_ok=True)
