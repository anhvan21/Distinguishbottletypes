"""Huấn luyện model phân loại loại chai (YOLO classification).

Chạy: uv run python scripts/train_classifier.py --data datasets/bottle_types --epochs 20
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/bottle_types")
    parser.add_argument("--model", default="yolo26n-cls.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="models/bottle_types_best.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="bottle_types",
        exist_ok=True,
    )

    best_path = Path(model.trainer.best)
    metrics = YOLO(best_path).val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"Top-1 accuracy: {metrics.top1:.3f}")
    print(f"Top-5 accuracy: {metrics.top5:.3f}")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, target)
    print(f"Saved best model -> {target}")


if __name__ == "__main__":
    main()
