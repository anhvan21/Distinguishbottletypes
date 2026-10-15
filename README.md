# SmartVisionInspection

Hệ thống kiểm tra lỗi sản phẩm bằng Computer Vision, xây dựng với **Python, PyTorch, YOLO, OpenCV và FastAPI**. Hệ thống nhận ảnh sản phẩm qua REST API, phát hiện đối tượng/lỗi bằng YOLO và trả về nhãn, độ tin cậy, tọa độ bounding box cùng ảnh đã được vẽ khung.

> **Trạng thái hiện tại (v0.1.0):** đang dùng model YOLO26n huấn luyện sẵn trên COCO, nên mới nhận diện được vật thể thông thường (người, xe, ...). Để phát hiện lỗi sản phẩm thật (vết nứt, trầy xước, rỉ sét, ...), cần fine-tune trên dataset lỗi có nhãn bounding box — xem [Lộ trình](#lộ-trình).

## Tính năng

- `POST /predict`: tải ảnh lên (JPEG/PNG/WEBP, tối đa 10 MB) để nhận diện.
- Kết quả gồm `label`, `confidence`, `bbox [x1, y1, x2, y2]` và thời gian inference `latency_ms`.
- Lưu ảnh có vẽ bounding box vào `outputs/`, mỗi request một file riêng.
- `GET /health`: kiểm tra server đang hoạt động.
- Kiểm tra đầu vào: sai định dạng → `415`, ảnh rỗng/hỏng/quá lớn → `400`.

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Model | YOLO26 (Ultralytics), PyTorch |
| Xử lý ảnh | OpenCV |
| API | FastAPI, Uvicorn |
| Kiểm thử | pytest, httpx |
| Quản lý môi trường | uv |

## Cấu trúc dự án

```
SmartVisionInspection/
├── app/
│   ├── main.py        # FastAPI app: /health, /predict
│   ├── detector.py    # SmartVisionDetector: chạy YOLO, vẽ khung, đo latency
│   └── schemas.py     # Pydantic schema cho response
├── tests/
│   └── test_api.py    # Test API
├── outputs/           # Ảnh kết quả (không đưa lên git)
└── pyproject.toml
```

## Cài đặt và chạy

Yêu cầu: Python 3.12+ và [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Mở http://127.0.0.1:8000/docs, chọn `POST /predict` → **Try it out** → chọn ảnh → **Execute**.

Lần chạy đầu tiên sẽ tự tải trọng số `yolo26n.pt` nên cần có Internet.

Gọi API bằng `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@bus.jpg;type=image/jpeg"
```

Ví dụ kết quả:

```json
{
  "detections": [
    {"class_id": 5, "label": "bus", "confidence": 0.8806, "bbox": [0.0, 230.48, 803.2, 750.7]},
    {"class_id": 0, "label": "person", "confidence": 0.872, "bbox": [48.61, 397.7, 240.34, 902.41]}
  ],
  "latency_ms": 64.54,
  "annotated_image": "outputs/prediction_d6a3cc259b55.jpg"
}
```

## Kiểm thử

```bash
uv run pytest -q
```

## Lưu ý về latency

`latency_ms` chỉ đo thời gian gọi model, chưa gồm upload, giải mã ảnh và trả response. Lần dự đoán đầu tiên chậm hơn hẳn vì phải nạp model; khi benchmark cần chạy khởi động trước rồi mới báo cáo p50/p95.

## Lộ trình

- [ ] **Dataset:** chọn dataset lỗi có nhãn bounding box (ví dụ NEU-DET, GC10-DET). MVTec AD phù hợp với bài toán anomaly detection hơn là detection.
- [ ] **Training:** fine-tune YOLO trên dataset lỗi.
- [ ] **Evaluation:** đo Precision, Recall, mAP@50, mAP@50:95 và latency p50/p95 trên tập test độc lập.
- [ ] **Backend:** thêm logging, cấu hình qua biến môi trường, xử lý nhiều request đồng thời.
- [ ] **Deployment:** đóng gói bằng Docker.
