"""Fine-tune YOLO để nhận diện 1 sản phẩm cụ thể.

Chạy: uv run python scripts/train.py --epochs 50
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/product/data.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="product",
        exist_ok=True,
    )

    metrics = model.val(data=args.data, device=args.device)
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP@50:    {metrics.box.map50:.3f}")
    print(f"mAP@50:95: {metrics.box.map:.3f}")

    best = Path(model.trainer.best)
    target = Path("models/product_best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)
    print(f"Saved best model -> {target}")


if __name__ == "__main__":
    main()
