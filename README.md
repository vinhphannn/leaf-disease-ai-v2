---
title: Phân loại bệnh lá cây (CAFormer/CNN)
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: web/app.py
pinned: false
license: apache-2.0
---
# Phân loại bệnh lá cây AI — Ứng dụng CAFormer/CNN

Demo trực tuyến (Hugging Face Spaces): https://huggingface.co/spaces/Zynh/leaf-disease-v2

Dự án này là một ứng dụng web hoàn chỉnh để phân loại bệnh lá cây. Hỗ trợ hai mô hình và cung cấp trực quan hóa Grad-CAM:

- CAFormer S18 (PyTorch/TIMM) — mô hình chính với độ chính xác cao và Grad-CAM
- CNN (Keras/TensorFlow) — mô hình tham chiếu sử dụng file .h5

Ứng dụng được xây dựng bằng Gradio và thiết kế để dễ sử dụng cục bộ hoặc triển khai lên Hugging Face Spaces.

## Tính năng

- Chuyển đổi mô hình: CAFormer (PyTorch) hoặc CNN (.h5, TensorFlow)
- Bản đồ nhiệt Grad-CAM/EigenCAM cho CAFormer để giải thích dự đoán
- Bảng điều khiển tổng quan: thống kê dataset, metrics, thông tin mô hình, thư viện mẫu
- Suy luận tương tác: tải lên/dán/webcam hình ảnh đầu vào
- Cấu trúc sẵn sàng cho checkpoint và metrics để huấn luyện/đánh giá

## Demo trực tuyến

- Ứng dụng web: https://huggingface.co/spaces/Zynh/leaf-disease-v2

## Công nghệ sử dụng

- PyTorch + TIMM (CAFormer S18)
- TensorFlow/Keras (CNN baseline, .h5)
- Gradio 5 cho giao diện
- OpenCV (headless), Pillow, NumPy
- Grad-CAM (package `grad-cam`) cho trực quan hóa

## Dataset

- New Plant Diseases Dataset (Kaggle): hình ảnh bệnh lá đa lớp
- Cấu trúc thông thường (cục bộ):

```
data/
  train/  valid/  test/
  # mỗi thư mục chứa một thư mục con cho mỗi lớp
```

## Mô hình

- CAFormer S18 (TIMM): file `models/model_best.pth`
- CNN baseline: file `models/plant_disease_cnn_256.h5`

Lưu ý: theo dõi các file mô hình lớn bằng Git LFS.

## Cấu trúc dự án

```
.
├─ app.py                      # wrapper nhỏ để expose web/app.py cho Spaces
├─ web/
│  └─ app.py                   # ứng dụng Gradio Blocks chính
├─ models/
│  ├─ model_best.pth           # CAFormer (PyTorch)
│  └─ plant_disease_cnn_256.h5 # CNN (Keras)
├─ data/                       # layout dữ liệu cục bộ tùy chọn (train/valid/test)
├─ outputs/                    # metrics/confusion matrix/gradcam tùy chọn
├─ requirements.txt
└─ README.md
```

## Cài đặt cục bộ

Yêu cầu:

- Python 3.10 được khuyến nghị (Spaces sử dụng py3.10 base)

Cài đặt dependencies:

```
pip install -r requirements.txt
```

Chạy cục bộ:

```
python web/app.py
```

Mở URL hiển thị trong terminal (ví dụ: http://127.0.0.1:7860).

## Sử dụng ứng dụng

1) Mở tab "Tổng quan" để xem thống kê dataset và hướng dẫn
2) Mở tab "Suy luận (Grad-CAM)"
3) Chọn mô hình:
   - CAFormer (khuyến nghị; hỗ trợ Grad-CAM/EigenCAM)
   - CNN (.h5; không có overlay Grad-CAM)
4) Tải lên/dán/chụp hình ảnh và nhấp "Dự đoán"

Mẹo: Grad-CAM chỉ áp dụng cho mô hình PyTorch (CAFormer). Giữ CAFormer được chọn để có bản đồ nhiệt.

## Triển khai Hugging Face Spaces

- Đảm bảo `requirements.txt` tối thiểu và tương thích CPU (đã được cấu hình)
- Giữ mô hình trong `models/` và sử dụng Git LFS
- Entrypoint Spaces: `app.py` ở root repo (wrapper) hoặc `web/app.py` nếu được cấu hình tương ứng

### Git LFS cho Mô hình

```
git lfs install
git lfs track "models/*"
git add .gitattributes models/*
git commit -m "Track model files via LFS"
```

## Khắc phục sự cố

- Thiếu file app trên Spaces: đảm bảo `app.py` tồn tại ở root hoặc đặt `app_file: web/app.py` trong cấu hình Space
- Grad-CAM không hiển thị: chọn CAFormer; đảm bảo `grad-cam` và `opencv-python-headless` được cài đặt
- TensorFlow thiếu trên Spaces: có trong `requirements.txt` dưới dạng `tensorflow-cpu`; rebuild Space
- File mô hình lớn: phải được theo dõi bởi Git LFS hoặc tải lên qua HF Space files UI

## Giấy phép

Apache-2.0 (khuyến nghị cho tương thích và cấp bằng sáng chế).

# Phân loại bệnh lá cây – Dự án hoàn chỉnh

Dự án huấn luyện và triển khai mô hình phân loại bệnh lá cây với pipeline chuyên nghiệp: huấn luyện 3 giai đoạn, quản lý checkpoint thông minh, đánh giá đầy đủ (Accuracy, Precision/Recall/F1, Confusion Matrix), Grad-CAM/EigenCAM giải thích mô hình, và web demo 2 tab (tổng quan + suy luận/Grad-CAM).

## 1) Dữ liệu

- Nguồn: New Plant Diseases Dataset (Augmented) – KaggleLink: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download
- Cấu trúc thư mục (sau khi tải về và tách):

```
./data/
  ├── train/
  ├── valid/
  └── test/
```

- Số lượng ảnh hiển thị tự động trên web (đếm từ thư mục). Bạn có thể cập nhật và bấm "Làm mới" ở tab Tổng quan.

## 2) Mô hình

- Kiến trúc: `caformer_s18.sail_in1k` (MetaFormer family – TIMM), tiền huấn luyện ImageNet-1K.
- Head (fully-connected) được điều chỉnh theo số lượng lớp (classes) của dataset.
- Triển khai bằng TIMM, PyTorch.

## 3) Chiến lược huấn luyện (Transfer Learning 3 giai đoạn)

- Stage 1 (Head-only): Freeze backbone, chỉ train head với LR cao để thích nghi nhanh (CosineAnnealingWarmRestarts).
- Stage 2 (Full fine-tune): Unfreeze toàn bộ, LR nhỏ (ReduceLROnPlateau), chuyển stage khi plateau.
- Stage 3 (Ultra fine-tune): LR rất thấp để "đánh bóng", dừng khi plateau (Cosine ngắn).
- Checkpoint thông minh: lưu `best`, `latest`, và `epoch_XXX`; hỗ trợ resume đa stage và đặt tên theo stage để không ghi đè.

## 4) Đánh giá (Validation/Test)

Có 2 cách:

- Một nút chạy (cấu hình trong file, không cần tham số):

  ```bash
  py run_evaluate_model.py
  ```

  Kết quả lưu tại `outputs/`:

  - `runner_metrics.txt` (Accuracy, Macro Precision/Recall/F1)
  - `runner_classification_report.txt`
  - `runner_confusion_matrix.csv`
- Dùng script linh hoạt (có tham số):

  ```bash
  py src/evaluation/evaluate_model.py --model_path models/model_best.pth --data_dir ./data/valid --batch_size 32 --num_workers 4
  ```

## 5) Xuất model tốt nhất

- Tự tìm checkpoint tốt nhất (ưu tiên `checkpoint_stageX_best.pth`) và xuất model gọn:
  ```bash
  py export_best_model.py --checkpoint_dirs . ./checkpoint ./checkpoints --model_name caformer_s18.sail_in1k --out models/model_best.pth
  ```

## 6) Grad-CAM / EigenCAM (giải thích mô hình)

- Tạo overlay cho một thư mục ảnh (mặc định `./data/test`):
  ```bash
  py run_gradcam.py
  ```
- Ảnh overlay lưu ở `outputs/gradcam/`.
- Lưu ý Python 3.13: cài từ GitHub nếu PyPI chưa hỗ trợ:
  ```bash
  py -m pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
  ```

## 7) Web demo (Gradio – 2 tab)

- Tab "Tổng quan" (Tiếng Việt): Nguồn dữ liệu (Kaggle), thống kê Train/Valid/Test, mô hình, chiến lược huấn luyện, pipeline; đọc kết quả đánh giá từ `outputs/*`.
- Tab "Suy luận (Grad-CAM)": Upload/dán/webcam ảnh → Dự đoán lớp + hiển thị Grad-CAM.
- Cách chạy:

  ```bash
  ./scripts/run_web.ps1
  ```

  hoặc:
  ```bash
  py web/app.py
  ```
- Biến môi trường (tùy chọn): `MODEL_PATH` (mặc định `models/model_best.pth`), `IMG_SIZE`.

## 8) Cài đặt phụ thuộc (Windows/Python)

- Yêu cầu chính: PyTorch, torchvision, timm, tqdm, pillow, numpy, scikit-learn, gradio.
- Cài nhanh:
  ```bash
  py -m pip install -U pip
  py -m pip install torch torchvision timm tqdm pillow numpy scikit-learn gradio
  # (tuỳ chọn Grad-CAM – 3.13 dùng GitHub)
  py -m pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
  ```

## 9) Cấu trúc dự án

```
.
├── src/
│   ├── data/
│   │   ├── datasets.py           # Dataset utilities
│   │   └── transforms.py         # Train/val/test transforms
│   ├── models/
│   │   └── build_model.py        # Model factory (TIMM)
│   ├── training/
│   │   ├── loop.py               # (placeholder) migrate logic từ notebook
│   │   ├── checkpoints.py        # Helper quản lý checkpoint
│   │   ├── metrics.py            # Metric helpers
│   │   └── utils.py              # Seeding, helpers
│   ├── evaluation/
│   │   └── evaluate.py           # (placeholder) entry eval batch
│   └── inference/
│       └── predict.py            # (placeholder) inference đơn lẻ
│
├── web/
│   └── app.py                    # Gradio app (2 tab)
├── scripts/
│   ├── run_web.ps1               # Launch web demo
│   └── eval.ps1                  # Wrapper đánh giá (gọi evaluate_best.py)
│
├── models/                       # model_best.pth (xuất)
├── checkpoints/ / checkpoint/    # checkpoint huấn luyện (từ notebook)
├── outputs/                      # metrics, báo cáo, confusion matrix, gradcam
├── data/                         # Train/Valid/Test
├── evaluate_best.py              # Đánh giá checkpoint tốt nhất (linh hoạt)
├── export_best_model.py          # Xuất model_best.pth
├── run_evaluate_model.py         # Đánh giá 1-click (config trong file)
├── run_gradcam.py                # Tạo overlay Grad-CAM 1-click
├── leaf-diseases.ipynb           # Notebook huấn luyện 3 giai đoạn
├── .cursorignore                 # Bỏ qua ảnh data khi index
└── README.md
```

## 10) Gợi ý hiệu năng

- CPU chậm: tăng `--batch_size` nếu đủ RAM; giảm `--img_size` (192) để đánh giá nhanh hơn.
- GPU: cài PyTorch CUDA và driver phù hợp để tăng tốc đáng kể.

---

Mọi thứ đã cấu hình để bạn “Run là chạy”. Nếu muốn giảm tham số khi chạy, dùng các file runner (`run_evaluate_model.py`, `run_gradcam.py`) với cấu hình viết sẵn ngay trong file.
