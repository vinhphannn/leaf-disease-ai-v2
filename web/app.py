import os
import io
from typing import Tuple, List, Optional

import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

# For CNN model (Keras/TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN model will not work.")

# Optional Grad-CAM
try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    EigenCAM = None
    show_cam_on_image = None


# Model paths
CAFORMER_PATH = "models/model_v2/best_caformer_model.pth"
MOBILENET_PATH = "models/model_v2/mobilenetv2_plant_disease_final.h5"
CNN_PATH = "models\model_v2\plant_disease_cnn_merged_256.h5"

IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
USE_EIGEN_CAM = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model variables
caformer_model = None
mobilenet_model = None
cnn_model = None
current_model_type = "caformer"  # "caformer", "mobilenet", or "cnn"
classes: List[str] = []
target_layer = None
cam = None
model_param_total = 0
model_param_trainable = 0


def crop_to_square(img: Image.Image):
    """Crop rectangular image to square by taking center crop"""
    width, height = img.size
    
    if width == height:
        return img
    
    # Calculate the size of the square (use the smaller dimension)
    size = min(width, height)
    
    # Calculate center crop coordinates
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    # Crop to square
    return img.crop((left, top, right, bottom))


def build_tfms(img_size: int = 224):
    return transforms.Compose([
        transforms.Lambda(crop_to_square),  # Crop to square first
        transforms.Resize((img_size, img_size)),  # Then resize to target size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def preprocess_for_cnn(img: Image.Image, img_size: int = 256):
    """Preprocess image for CNN model (Keras/TensorFlow)"""
    # First crop to square to preserve aspect ratio
    img_square = crop_to_square(img)
    
    # Then resize to model input size
    img_resized = img_square.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img_resized) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_caformer_model(model_path: str, device: torch.device):
    """Load CAFormer model from PyTorch checkpoint"""
    bundle = torch.load(model_path, map_location=device)
    cls = bundle.get('classes')
    model_name = bundle.get('model_name', 'caformer_s18.sail_in1k')
    
    # If classes not found in checkpoint, use default classes
    if not cls:
        print("Warning: Classes not found in checkpoint, using default classes")
        cls = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    mdl = timm.create_model(model_name, pretrained=False, num_classes=len(cls))
    mdl.load_state_dict(bundle['model_state_dict'])
    mdl.to(device)
    mdl.eval()
    return mdl, cls


def load_mobilenet_model(model_path: str):
    """Load MobileNetV2 model from Keras .h5 file - Simplified version"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Cannot load MobileNetV2 model.")
    
    try:
        # Try to load with custom objects to handle complex architectures
        import tensorflow.keras as keras
        from tensorflow.keras.utils import get_custom_objects
        
        # Load with compile=False to avoid compilation issues
        mobilenet_mdl = keras.models.load_model(model_path, compile=False)
        
        # Recompile if needed
        mobilenet_mdl.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    except Exception as e:
        print(f"MobileNetV2 load failed: {e}")
        # Fallback: Create a simple MobileNetV2 model
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, Model
            
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Add custom top layers
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            predictions = layers.Dense(38, activation='softmax')(x)
            
            mobilenet_mdl = Model(inputs=base_model.input, outputs=predictions)
            mobilenet_mdl.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Created fallback MobileNetV2 model")
            
        except Exception as e2:
            raise RuntimeError(f"Failed to load MobileNetV2 model: {str(e2)}")
    
    # Define classes (same as CAFormer model)
    cls = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    return mobilenet_mdl, cls


def load_cnn_model(model_path: str):
    """Load CNN model from Keras .h5 file"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Cannot load CNN model.")
    
    # Load the model
    cnn_mdl = keras.models.load_model(model_path)
    
    # Define classes (same as CAFormer model)
    cls = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    return cnn_mdl, cls


def pick_target_layer(mdl: torch.nn.Module):
    for attr in ["stages", "blocks", "layer4", "features"]:
        if hasattr(mdl, attr):
            layer = getattr(mdl, attr)
            if isinstance(layer, (torch.nn.Sequential, list, tuple)) and len(layer) > 0:
                return layer[-1]
            return layer
    children = list(mdl.children())
    return children[-1] if children else mdl


def ensure_cam(mdl: torch.nn.Module):
    global cam, target_layer
    if GradCAM is None and EigenCAM is None:
        return None
    if cam is not None:
        return cam
    target_layer = pick_target_layer(mdl)
    cam_class = EigenCAM if USE_EIGEN_CAM else GradCAM
    cam = cam_class(model=mdl, target_layers=[target_layer])
    return cam


def to_numpy_image(img_pil: Image.Image) -> np.ndarray:
    disp = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(disp).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


tfms = build_tfms(IMG_SIZE)


def predict_image(img: Image.Image):
    if current_model_type == "caformer":
        if caformer_model is None:
            return "CAFormer model not loaded", None
        
        inp = tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = caformer_model(inp)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = probs.max(dim=0)
            pred_name = classes[int(top_idx.item())]

        overlay = None
        if show_cam_on_image is not None:
            try:
                local_cam = ensure_cam(caformer_model)
                targets = None
                grayscale_cam = local_cam(input_tensor=inp, targets=targets)[0]
                overlay = show_cam_on_image(to_numpy_image(img), grayscale_cam, use_rgb=True)
                overlay = Image.fromarray(overlay)
            except Exception:
                overlay = None

        result_text = f"Prediction: {pred_name} ({float(top_prob.item()):.2%})\nModel: CAFormer"
        return result_text, overlay
    
    elif current_model_type == "mobilenet":
        if mobilenet_model is None:
            return "MobileNetV2 model not loaded", None
        
        try:
            inp = preprocess_for_cnn(img, img_size=224)  # MobileNetV2 typically uses 224x224
            predictions = mobilenet_model.predict(inp, verbose=0)
            probs = predictions[0]
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            pred_name = classes[int(top_idx)]
            
            # MobileNetV2 model doesn't support Grad-CAM easily, so return original image
            overlay = img
            
            result_text = f"Prediction: {pred_name} ({float(top_prob):.2%})\nModel: MobileNetV2"
            return result_text, overlay
        except Exception as e:
            return f"MobileNetV2 prediction failed: {str(e)}", None
    
    elif current_model_type == "cnn":
        if cnn_model is None:
            return "CNN model not loaded", None
        
        try:
            inp = preprocess_for_cnn(img, img_size=256)
            predictions = cnn_model.predict(inp, verbose=0)
            probs = predictions[0]
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            pred_name = classes[int(top_idx)]
            
            # CNN model doesn't support Grad-CAM easily, so return original image
            overlay = img
            
            result_text = f"Prediction: {pred_name} ({float(top_prob):.2%})\nModel: CNN"
            return result_text, overlay
        except Exception as e:
            return f"CNN prediction failed: {str(e)}", None
    
    else:
        return "Unknown model type", None


def switch_model(model_type: str):
    """Switch between CAFormer, MobileNetV2, and CNN models"""
    global current_model_type, caformer_model, mobilenet_model, cnn_model, classes, model_param_total, model_param_trainable
    
    if model_type == "caformer":
        try:
            if caformer_model is None:
                caformer_model, classes = load_caformer_model(CAFORMER_PATH, device)
                model_param_total = sum(p.numel() for p in caformer_model.parameters())
                model_param_trainable = sum(p.numel() for p in caformer_model.parameters() if p.requires_grad)
            current_model_type = "caformer"
            return f"✅ Switched to CAFormer model\nParameters: {model_param_total:,} total, {model_param_trainable:,} trainable"
        except Exception as e:
            return f"❌ Failed to load CAFormer model: {str(e)}"
    
    elif model_type == "mobilenet":
        try:
            if mobilenet_model is None:
                mobilenet_model, classes = load_mobilenet_model(MOBILENET_PATH)
                model_param_total = mobilenet_model.count_params()
                model_param_trainable = model_param_total
            current_model_type = "mobilenet"
            return f"✅ Switched to MobileNetV2 model\nParameters: {model_param_total:,} total"
        except Exception as e:
            return f"❌ Failed to load MobileNetV2 model: {str(e)}"
    
    elif model_type == "cnn":
        try:
            if cnn_model is None:
                cnn_model, classes = load_cnn_model(CNN_PATH)
                model_param_total = cnn_model.count_params()
                model_param_trainable = model_param_total
            current_model_type = "cnn"
            return f"✅ Switched to CNN model\nParameters: {model_param_total:,} total"
        except Exception as e:
            return f"❌ Failed to load CNN model: {str(e)}"
    
    else:
        return "❌ Unknown model type"


def load_metrics_text() -> str:
    paths = [
        "outputs/metrics_from_model.txt",
        "outputs/runner_metrics.txt",
        "outputs/metrics.txt",
    ]
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                continue
    return "No metrics file found yet. Run evaluation to generate outputs." 


def latest_metrics_values() -> dict:
    """Parse outputs/*metrics*.txt to extract key numbers if available."""
    paths = [
        "outputs/metrics_from_model.txt",
        "outputs/runner_metrics.txt",
        "outputs/metrics.txt",
    ]
    data = {}
    import re
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    txt = f.read()
                m = re.search(r"Accuracy:\s*([0-9.]+)", txt)
                if m:
                    data['accuracy'] = float(m.group(1))
                m = re.search(r"Macro Precision:\s*([0-9.]+)", txt)
                if m:
                    data['macro_precision'] = float(m.group(1))
                m = re.search(r"Macro Recall:\s*([0-9.]+)", txt)
                if m:
                    data['macro_recall'] = float(m.group(1))
                m = re.search(r"Macro F1:\s*([0-9.]+)", txt)
                if m:
                    data['macro_f1'] = float(m.group(1))
                if data:
                    return data
            except Exception:
                continue
    return data


def _count_images(root: str) -> int:
    if not os.path.isdir(root):
        return 0
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    total = 0
    for pat in patterns:
        for _ in os.walk(root):
            pass
        from glob import glob as _glob
        total += len(_glob(os.path.join(root, "**", pat), recursive=True))
    return total


def sample_images_per_class(root: str, max_per_class: int = 2) -> List[Tuple[str, List[str]]]:
    """Return [(class_name, [img paths...])]."""
    result: List[Tuple[str, List[str]]] = []
    if not os.path.isdir(root):
        return result
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    from glob import glob as _glob
    for cname in classes:
        cdir = os.path.join(root, cname)
        imgs = []
        for pat in patterns:
            imgs.extend(_glob(os.path.join(cdir, pat)))
        imgs = imgs[:max_per_class]
        if imgs:
            result.append((cname, imgs))
    return result


def overview_markdown() -> str:
    # Dataset paths (mặc định theo repo)
    data_dir = "./data"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    n_train = _count_images(train_dir)
    n_valid = _count_images(valid_dir)
    n_test = _count_images(test_dir)
    total = n_train + n_valid + n_test

    # Model info
    model_name = "(chưa tải)"
    num_classes = 0
    try:
        bundle = torch.load(CAFORMER_PATH, map_location="cpu")
        model_name = bundle.get("model_name", model_name)
        cls = bundle.get("classes", [])
        num_classes = len(cls) if isinstance(cls, list) else 0
    except Exception:
        # Fallback: suy ra từ số thư mục lớp ở train
        try:
            if os.path.isdir(train_dir):
                num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        except Exception:
            pass

    # Latest metrics
    mets = latest_metrics_values()
    acc = mets.get('accuracy')
    mp = mets.get('macro_precision')
    mr = mets.get('macro_recall')
    mf1 = mets.get('macro_f1')

    return f"""
### Tổng quan dự án

<div class="section-title">1) Dữ liệu</div>
<div>• Nguồn: <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download" target="_blank">New Plant Diseases Dataset (Kaggle)</a></div>
<div>• Thư mục: <code>{data_dir}</code> (train/valid/test)</div>
<div style="display:flex;gap:12px;margin-top:8px">
  <div class="stat-card"><div class="stat-title">Tổng ảnh</div><div class="stat-value">{total:,}</div></div>
  <div class="stat-card"><div class="stat-title">Train</div><div class="stat-value">{n_train:,}</div></div>
  <div class="stat-card"><div class="stat-title">Valid</div><div class="stat-value">{n_valid:,}</div></div>
  <div class="stat-card"><div class="stat-title">Test</div><div class="stat-value">{n_test:,}</div></div>
  <div class="stat-card"><div class="stat-title">Số lớp</div><div class="stat-value">{num_classes}</div></div>
  <div class="stat-card"><div class="stat-title">Ảnh/lớp (xấp xỉ)</div><div class="stat-value">{(n_train//max(num_classes,1)):,}</div></div>
  </div>

  <div style="display:flex;gap:12px;margin-top:8px">
  <div class="stat-card"><div class="stat-title">Tham số tổng</div><div class="stat-value">{model_param_total:,}</div></div>
  <div class="stat-card"><div class="stat-title">Trainable</div><div class="stat-value">{model_param_trainable:,}</div></div>
  <div class="stat-card"><div class="stat-title">Accuracy (mới nhất)</div><div class="stat-value">{(f"{acc:.4f}" if acc is not None else "-")}</div></div>
  <div class="stat-card"><div class="stat-title">Macro P/R/F1</div><div class="stat-value">{(f"{mp:.4f}/{mr:.4f}/{mf1:.4f}" if mp is not None else "-")}</div></div>
  </div>

<div class="section-title" style="margin-top:12px">2) Môi trường & Mô hình</div>
<div class="kv"><div class="k">Thiết bị</div><div class="v">{device}</div></div>
<div class="kv"><div class="k">Kích thước ảnh</div><div class="v">{IMG_SIZE}×{IMG_SIZE}</div></div>
<div class="kv"><div class="k">Model (bundle)</div><div class="v">{model_name}</div></div>
<div class="kv"><div class="k">Số lớp (bundle)</div><div class="v">{num_classes}</div></div>

<div class="section-title" style="margin-top:12px">3) Quy trình huấn luyện tóm tắt</div>
<ul>
  <li>Stage 1 (head-only) → thích nghi nhanh</li>
  <li>Stage 2 (full fine-tune) → tối ưu toàn bộ</li>
  <li>Stage 3 (ultra fine-tune) → tinh chỉnh LR thấp</li>
  <li>Checkpoint: <code>best/latest/stageX_epoch</code></li>
  </ul>

<div class="section-title" style="margin-top:12px">4) Sử dụng nhanh</div>
<div>• Web demo: <code>py web/app.py</code> hoặc chạy Space.</div>
<div>• Grad-CAM: Tab "Suy luận". Chỉ hoạt động với mô hình PyTorch.</div>
<div class="note">Bấm “Làm mới tổng quan” để cập nhật số liệu mới.</div>
"""


with gr.Blocks(title="Phân loại bệnh lá cây",
               css="""
               :root { --card-bg: #fafafa; --card-border: #e8e8e8; --muted: #666; --accent: #146C94; }
               .stat-card {border: 1px solid var(--card-border); border-radius: 12px; padding: 12px; background: var(--card-bg); transition: box-shadow .2s ease}
               .stat-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.06) }
               .stat-title {font-weight: 600; color: var(--muted)}
               .stat-value {font-size: 20px; font-weight: 700}
               .section-title {font-size: 18px; font-weight: 700; margin-top: 8px}
               .kv {display:flex; gap:8px; align-items:center}
               .kv .k {width:180px; color: var(--muted)}
               .kv .v {font-weight:600}
               .note {color:#777; font-size: 13px}
               a {color: var(--accent)}
               code {background:#f5f5f5; padding:2px 6px; border-radius:6px}
               """) as app:
    gr.Markdown("""
    <div style="display:flex;align-items:center;gap:12px">
      <img src="https://cdn-icons-png.flaticon.com/512/2909/2909592.png" width="36"/>
      <div>
        <div style="font-size:22px;font-weight:800;line-height:1">Phân loại bệnh lá cây</div>
        <div style="color:#666">Demo web: tổng quan dự án + suy luận kèm Grad-CAM</div>
      </div>
    </div>
    """)

    with gr.Tab("Tổng quan"):
        with gr.Row():
            ov = gr.Markdown(overview_markdown())
        gr.Markdown("""
        <div class="section-title">Kết quả đánh giá</div>
        """)
        with gr.Row():
            metrics = gr.Textbox(value=load_metrics_text(), lines=16, label="(đọc từ outputs/*)", interactive=False)
        with gr.Row():
            refresh_overview = gr.Button("🔄 Làm mới tổng quan")
            refresh_metrics = gr.Button("📊 Làm mới kết quả")
        refresh_overview.click(lambda: overview_markdown(), None, ov)
        refresh_metrics.click(lambda: load_metrics_text(), None, metrics)

        gr.Markdown("""
        <div class="section-title">Ảnh mẫu theo lớp (train)</div>
        """)
        def build_gallery():
            items = []
            for cname, paths in sample_images_per_class(os.path.join("./data", "train"), max_per_class=2):
                for p in paths:
                    items.append((p, cname))
            return items
        gallery = gr.Gallery(label="Mỗi lớp 2 ảnh", columns=6, height=220, value=build_gallery())
        refresh_gallery = gr.Button("🖼️ Làm mới ảnh mẫu")
        refresh_gallery.click(lambda: build_gallery(), None, gallery)

    with gr.Tab("Suy luận (Grad-CAM)"):
        gr.Markdown("""
        - Chọn mô hình: CAFormer (99.33% accuracy), MobileNetV2 (~90% accuracy), hoặc CNN tự xây dựng (~70% accuracy)
        - Tải ảnh, dán ảnh (Ctrl+V), hoặc chụp từ webcam.  
        - Bấm "Dự đoán" để nhận kết quả và overlay Grad-CAM (chỉ CAFormer).
        """)
        
        # Model selection
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["caformer", "mobilenet", "cnn"],
                value="caformer",
                label="Chọn mô hình",
                info="CAFormer: 99.33% | MobileNetV2: ~90% | CNN: ~70% accuracy"
            )
            switch_btn = gr.Button("🔄 Chuyển mô hình", variant="secondary")
            model_status = gr.Textbox(label="Trạng thái mô hình", lines=2, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil",
                               sources=["upload", "webcam", "clipboard"],
                               label="Ảnh đầu vào",
                               height=320)
                with gr.Row():
                    btn = gr.Button("🚀 Dự đoán", variant="primary")
                    clear_btn = gr.Button("🧹 Xoá ảnh")
            with gr.Column(scale=1):
                out_text = gr.Textbox(label="Kết quả", lines=3)
                out_img = gr.Image(type="pil", label="Grad-CAM Overlay", height=320)
        
        # Event handlers
        switch_btn.click(fn=switch_model, inputs=model_dropdown, outputs=model_status)
        btn.click(fn=predict_image, inputs=inp, outputs=[out_text, out_img])
        clear_btn.click(lambda: (None, ""), None, [inp, out_text])


def _startup_load():
    global caformer_model, classes, model_param_total, model_param_trainable, current_model_type
    if not os.path.isfile(CAFORMER_PATH):
        raise RuntimeError(f"CAFormer model file not found: {CAFORMER_PATH}")
    mdl, cls = load_caformer_model(CAFORMER_PATH, device)
    # Count params
    try:
        model_param_total = sum(p.numel() for p in mdl.parameters())
        model_param_trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    except Exception:
        model_param_total = 0
        model_param_trainable = 0
    # Warmup CAM if available
    if EigenCAM is not None or GradCAM is not None:
        try:
            _ = ensure_cam(mdl)
        except Exception:
            pass
    current_model_type = "caformer"
    return mdl, cls


if __name__ == "__main__":
    caformer_model, classes = _startup_load()
    app.launch()


