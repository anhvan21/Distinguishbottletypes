"""Chia ảnh phân loại thành train/val cho YOLO classification.

Đầu vào: thư mục có mỗi lớp là 1 thư mục con, ví dụ
    raw/Beer Bottles/*.jpg, raw/Wine Bottles/*.jpg, ...
Đầu ra: <out>/train/<lớp>/*.jpg và <out>/val/<lớp>/*.jpg

Với --crop, mỗi ảnh được chạy qua YOLO detector và chỉ lưu các vùng cắt quanh
vật thể (mặc định "bottle"), để dữ liệu train giống lúc inference trong API.
Train/val được chia theo ảnh gốc TRƯỚC khi cắt nên không bị rò rỉ dữ liệu.

Chạy: uv run python scripts/prepare_classifier_data.py --src data_raw/bottles --out datasets/bottle_types --crop
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def crop_objects(detector, image_path: Path, class_ids: list[int], conf: float,
                 pad: float = 0.05) -> list:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    h, w = image.shape[:2]
    result = detector.predict(source=image, conf=conf, classes=class_ids, verbose=False)[0]
    crops = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(int(x1 - dx), 0), max(int(y1 - dy), 0)
        x2, y2 = min(int(x2 + dx), w), min(int(y2 + dy), h)
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            crops.append(image[y1:y2, x1:x2])
    return crops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Thư mục chứa các thư mục con theo lớp")
    parser.add_argument("--out", default="datasets/bottle_types")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Giới hạn số ảnh gốc mỗi lớp để train nhanh trên CPU")
    parser.add_argument("--crop", action="store_true",
                        help="Cắt vật thể bằng YOLO detector thay vì dùng cả ảnh")
    parser.add_argument("--detector", default="yolo26n.pt")
    parser.add_argument("--crop-class", default="bottle")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"Không tìm thấy thư mục lớp nào trong {src}")

    detector = class_ids = None
    if args.crop:
        from ultralytics import YOLO

        detector = YOLO(args.detector)
        class_ids = [i for i, n in detector.names.items() if n == args.crop_class]
        if not class_ids:
            raise SystemExit(f"Detector không có lớp '{args.crop_class}'")

    rng = random.Random(args.seed)
    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        rng.shuffle(images)
        if args.max_per_class:
            images = images[: args.max_per_class]
        n_val = max(1, int(len(images) * args.val_ratio))
        stats = []
        for split, files in (("val", images[:n_val]), ("train", images[n_val:])):
            target = out / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            saved = skipped = 0
            for f in files:
                if detector is None:
                    shutil.copy2(f, target / f.name)
                    saved += 1
                    continue
                crops = crop_objects(detector, f, class_ids, args.conf)
                skipped += not crops
                for i, crop in enumerate(crops):
                    cv2.imwrite(str(target / f"{f.stem}_{i}.jpg"), crop)
                    saved += 1
            stats.append(f"{split}: {saved} ảnh" + (f" ({skipped} ảnh gốc không tìm thấy vật thể)" if detector else ""))
        print(f"{class_dir.name}: " + ", ".join(stats))


if __name__ == "__main__":
    main()
