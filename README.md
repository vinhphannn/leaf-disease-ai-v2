---
title: Plant Disease Classifier (CAFormer/CNN)
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: web/app.py
pinned: false
license: apache-2.0
---

# Leaf Disease AI — CAFormer/CNN Web App

Demo (Hugging Face Spaces): https://huggingface.co/spaces/Zynh/leaf-disease-v2

This project is a production-ready web application for plant leaf disease classification. It supports two models and provides Grad-CAM visualization:
- CAFormer S18 (PyTorch/TIMM) — main model with high accuracy and Grad-CAM
- CNN (Keras/TensorFlow) — reference baseline using a .h5 model

The app is built with Gradio and designed for easy local usage or deployment to Hugging Face Spaces.

## Features

- Model switcher: CAFormer (PyTorch) or CNN (.h5, TensorFlow)
- Grad-CAM/EigenCAM heatmap for CAFormer to explain predictions
- Overview dashboard: dataset stats, metrics, model info, sample gallery
- Interactive inference: upload/paste/webcam image input
- Checkpoints and metrics-ready structure for training/evaluation

## Live Demo

- Web app: https://huggingface.co/spaces/Zynh/leaf-disease-v2

## Tech Stack

- PyTorch + TIMM (CAFormer S18)
- TensorFlow/Keras (CNN baseline, .h5)
- Gradio 5 for UI
- OpenCV (headless), Pillow, NumPy
- Grad-CAM (package `grad-cam`) for visualization

## Dataset

- New Plant Diseases Dataset (Kaggle): multi-class leaf disease images
- Typical structure (local):
```
data/
  train/  valid/  test/
  # each contains one subfolder per class
```

## Models

- CAFormer S18 (TIMM): file `models/model_best.pth`
- CNN baseline: file `models/plant_disease_cnn_256.h5`

Note: track large model files with Git LFS.

## Project Structure

```
.
├─ app.py                      # tiny wrapper that exposes web/app.py for Spaces
├─ web/
│  └─ app.py                   # main Gradio Blocks app
├─ models/
│  ├─ model_best.pth           # CAFormer (PyTorch)
│  └─ plant_disease_cnn_256.h5 # CNN (Keras)
├─ data/                       # optional local data layout (train/valid/test)
├─ outputs/                    # optional metrics/confusion matrix/gradcam
├─ requirements.txt
└─ README.md
```

## Local Setup

Prerequisites:
- Python 3.10 recommended (Spaces uses py3.10 base)

Install dependencies:
```
pip install -r requirements.txt
```

Run locally:
```
python web/app.py
```
Open the URL shown in the terminal (e.g., http://127.0.0.1:7860).

## Using the App

1) Open tab "Tổng quan" for dataset stats and guidance
2) Open tab "Suy luận (Grad-CAM)"
3) Select model:
   - CAFormer (recommended; supports Grad-CAM/EigenCAM)
   - CNN (.h5; no Grad-CAM overlay)
4) Upload/paste/capture an image and click "Dự đoán"

Tip: Grad-CAM only applies to the PyTorch model (CAFormer). Keep CAFormer selected for heatmaps.

## Hugging Face Spaces Deployment

- Ensure `requirements.txt` is minimal and CPU-compatible (already configured)
- Keep models under `models/` and use Git LFS
- Spaces entrypoint: `app.py` at repo root (wrapper) or `web/app.py` if configured accordingly

### Git LFS for Models

```
git lfs install
git lfs track "models/*"
git add .gitattributes models/*
git commit -m "Track model files via LFS"
```

## Push to GitHub

Repository (empty at the time of writing): https://github.com/vinhphannn/leaf-disease-ai-v2.git

Commands to initialize and push this project:
```
git init
git remote add origin https://github.com/vinhphannn/leaf-disease-ai-v2.git
git lfs install
git lfs track "models/*"
git add .
git commit -m "Initial commit: leaf-disease AI web app (CAFormer/CNN)"
git push -u origin main
```

If the remote has no default branch, create main:
```
git branch -M main
git push -u origin main
```

## Troubleshooting

- Missing app file on Spaces: ensure `app.py` exists at root or set `app_file: web/app.py` in Space config
- Grad-CAM not showing: select CAFormer; ensure `grad-cam` and `opencv-python-headless` installed
- TensorFlow missing on Spaces: present in `requirements.txt` as `tensorflow-cpu`; rebuild Space
- Large model files: must be tracked by Git LFS or upload via HF Space files UI

## License

Apache-2.0 (recommended for compatibility and patent grant).

# Phân loại bệnh lá cây – Dự án hoàn chỉnh

Dự án huấn luyện và triển khai mô hình phân loại bệnh lá cây với pipeline chuyên nghiệp: huấn luyện 3 giai đoạn, quản lý checkpoint thông minh, đánh giá đầy đủ (Accuracy, Precision/Recall/F1, Confusion Matrix), Grad-CAM/EigenCAM giải thích mô hình, và web demo 2 tab (tổng quan + suy luận/Grad-CAM).

## 1) Dữ liệu
- Nguồn: New Plant Diseases Dataset (Augmented) – Kaggle  
  Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download
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
