import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

OUTPUT_DIR = Path("outputs")


class SmartVisionDetector:
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        conf: float = 0.25,
        target_classes: list[str] | None = None,
        classifier_path: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_ids = self._resolve_class_ids(target_classes)
        # Stage 2 (optional): classify each detected crop, e.g. bottle -> beer/wine/water
        self.classifier = YOLO(classifier_path, task="classify") if classifier_path else None

    def _resolve_class_ids(self, names: list[str] | None) -> list[int] | None:
        """Map class names (e.g. ["bottle"]) to model class ids; None = all classes."""
        if not names:
            return None
        name_to_id = {name: idx for idx, name in self.model.names.items()}
        unknown = [name for name in names if name not in name_to_id]
        if unknown:
            raise ValueError(f"Unknown classes for this model: {unknown}")
        return [name_to_id[name] for name in names]

    def _classify_crop(self, crop: np.ndarray) -> tuple[str, float]:
        probs = self.classifier.predict(source=crop, verbose=False)[0].probs
        return self.classifier.names[probs.top1], round(float(probs.top1conf), 4)

    def predict(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read the input image")

        start = time.perf_counter()
        results = self.model.predict(
            source=image, conf=self.conf, classes=self.class_ids, verbose=False
        )
        result = results[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detection = {
                "class_id": class_id,
                "label": self.model.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
            if self.classifier is not None:
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size:
                    detection["type"], detection["type_confidence"] = self._classify_crop(crop)
            detections.append(detection)
        latency_ms = (time.perf_counter() - start) * 1000

        annotated = (
            self._draw(image.copy(), detections) if self.classifier is not None else result.plot()
        )

        # Unique file per request so concurrent calls don't overwrite each other
        output_path = OUTPUT_DIR / f"prediction_{uuid.uuid4().hex[:12]}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError("Cannot save annotated image")

        return {
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "annotated_image": str(output_path),
        }

    @staticmethod
    def _draw(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            name = det.get("type", det["label"])
            score = det.get("type_confidence", det["confidence"])
            text = f"{name} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1 - th - 6, 0)
            cv2.rectangle(image, (x1, top), (x1 + tw + 4, top + th + 6), (0, 200, 0), -1)
            cv2.putText(image, text, (x1 + 2, top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image
