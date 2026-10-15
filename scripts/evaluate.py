"""Đánh giá mô hình phân loại chai và toàn bộ pipeline (detect -> classify).

1. Classifier: confusion matrix + precision/recall từng lớp trên <data>/val.
2. Pipeline: chạy SmartVisionDetector trên ảnh gốc CHƯA dùng khi chuẩn bị dữ liệu
   (prepare_classifier_data.py lấy max-per-class ảnh đầu sau khi shuffle với cùng seed),
   đo tỉ lệ tìm thấy chai, độ chính xác loại chai và latency p50/p95.

Chạy: uv run python scripts/evaluate.py --src "D:/.../Bottle Images" --used-per-class 2000
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import SmartVisionDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def evaluate_classifier(model_path: str, val_dir: Path) -> None:
    model = YOLO(model_path, task="classify")
    names = [model.names[i] for i in sorted(model.names)]
    y_true, y_pred = [], []
    for class_dir in sorted(d for d in val_dir.iterdir() if d.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for i in range(0, len(files), 64):
            for r in model.predict([str(f) for f in files[i:i + 64]], verbose=False):
                y_true.append(class_dir.name)
                y_pred.append(model.names[r.probs.top1])

    print("=== 1. Classifier trên ảnh đã cắt (val) ===")
    print(classification_report(y_true, y_pred, labels=names, digits=3))
    print("Confusion matrix (hàng = thật, cột = dự đoán):")
    width = max(len(n) for n in names)
    print(" " * width + " | " + " ".join(f"{n[:6]:>6}" for n in names))
    for name, row in zip(names, confusion_matrix(y_true, y_pred, labels=names)):
        print(f"{name:>{width}} | " + " ".join(f"{v:>6}" for v in row))


def unused_images(src: Path, used_per_class: int, seed: int) -> dict[str, list[Path]]:
    """Tái tạo đúng thứ tự shuffle của prepare_classifier_data.py để lấy ảnh chưa dùng."""
    rng = random.Random(seed)
    result = {}
    for class_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        images = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        rng.shuffle(images)
        result[class_dir.name] = images[used_per_class:]
    return result


def evaluate_pipeline(args) -> None:
    detector = SmartVisionDetector(
        args.detector, conf=args.conf, target_classes=["bottle"], classifier_path=args.classifier
    )
    rng = random.Random(0)
    latencies, rows = [], []
    for class_name, images in unused_images(Path(args.src), args.used_per_class, args.seed).items():
        sample = rng.sample(images, min(args.test_per_class, len(images)))
        found = correct = 0
        for path in sample:
            out = detector.predict(str(path))
            Path(out["annotated_image"]).unlink(missing_ok=True)
            latencies.append(out["latency_ms"])
            types = [d["type"] for d in out["detections"] if d.get("type")]
            if types:
                found += 1
                # Mỗi ảnh tổng hợp chỉ chứa 1 loại chai -> lấy loại xuất hiện nhiều nhất
                correct += Counter(types).most_common(1)[0][0] == class_name
        rows.append((class_name, len(sample), found, correct))

    print(f"\n=== 2. Pipeline trên ảnh gốc chưa dùng (conf={args.conf}) ===")
    print(f"{'Lớp':<16} {'Ảnh':>5} {'Tìm thấy chai':>14} {'Đúng loại*':>11}")
    for name, n, found, correct in rows:
        acc = f"{correct / found:.1%}" if found else "-"
        print(f"{name:<16} {n:>5} {found / n:>14.1%} {acc:>11}")
    n_all = sum(r[1] for r in rows)
    found_all = sum(r[2] for r in rows)
    correct_all = sum(r[3] for r in rows)
    print(f"{'Tổng':<16} {n_all:>5} {found_all / n_all:>14.1%} {correct_all / max(found_all, 1):>11.1%}")
    print("* Tính trên các ảnh đã tìm thấy chai.")
    print(f"Đúng hoàn toàn (tìm thấy + đúng loại): {correct_all / n_all:.1%}")
    lat = np.array(latencies[5:] or latencies)  # bỏ vài lần đầu (khởi động model)
    print(f"Latency detect+classify: p50={np.percentile(lat, 50):.1f} ms, p95={np.percentile(lat, 95):.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Thư mục ảnh gốc (mỗi lớp 1 thư mục con)")
    parser.add_argument("--data", default="datasets/bottle_types")
    parser.add_argument("--classifier", default="models/bottle_types_best.pt")
    parser.add_argument("--detector", default="yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--used-per-class", type=int, default=2000)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_classifier(args.classifier, Path(args.data) / "val")
    evaluate_pipeline(args)


if __name__ == "__main__":
    main()
