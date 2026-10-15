import pytest
from ultralytics.utils import ASSETS

from app.detector import SmartVisionDetector

BUS_IMAGE = str(ASSETS / "bus.jpg")


def test_target_classes_keeps_only_requested_class():
    detector = SmartVisionDetector(target_classes=["person"])
    labels = {d["label"] for d in detector.predict(BUS_IMAGE)["detections"]}
    assert labels == {"person"}


def test_bottle_filter_drops_other_objects():
    detector = SmartVisionDetector(target_classes=["bottle"])
    assert detector.predict(BUS_IMAGE)["detections"] == []


def test_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown classes"):
        SmartVisionDetector(target_classes=["chai"])
