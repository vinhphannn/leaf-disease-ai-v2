"""
One-click Grad-CAM/EigenCAM generator for the saved model.

Usage: open and Run this file in your IDE. Configure INPUT_DIR below.
Saves overlays to outputs/gradcam/.
"""
import os
from glob import glob
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm

# Optional: pip install pytorch-grad-cam
try:
	from pytorch_grad_cam import GradCAM, EigenCAM
	from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
	GradCAM = None
	EigenCAM = None
	show_cam_on_image = None


# ===== In-file config =====
MODEL_PATH = "models/model_best.pth"
INPUT_DIR = "./data/test"   # folder chứa ảnh cần visualize; có thể trỏ vào class folder
IMG_SIZE = 224
OUTPUT_DIR = "./outputs/gradcam"
USE_EIGEN_CAM = True          # EigenCAM (phù hợp nhiều kiến trúc); False => GradCAM
MAX_IMAGES = 100              # giới hạn số ảnh để chạy nhanh
# ==========================


def build_tfms(img_size: int = 224):
	return transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])


def load_clean_model(model_path: str, device: torch.device):
	bundle = torch.load(model_path, map_location=device)
	classes = bundle.get('classes')
	model_name = bundle.get('model_name', 'caformer_s18.sail_in1k')
	if not classes:
		raise SystemExit("Model file missing 'classes'.")
	model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
	model.load_state_dict(bundle['model_state_dict'])
	model.to(device)
	model.eval()
	return model, classes


def pick_target_layer(model: torch.nn.Module):
	"""Pick a suitable last conv-like layer for CAM.
	For many TIMM vision models, a good default is the last block of the feature extractor.
	"""
	# Try common attributes
	for attr in ["stages", "blocks", "layer4", "features"]:
		if hasattr(model, attr):
			layer = getattr(model, attr)
			# pick the last submodule if it's a sequence-like
			if isinstance(layer, (torch.nn.Sequential, list, tuple)) and len(layer) > 0:
				return layer[-1]
			return layer
	# Fallback: last child
	children = list(model.children())
	return children[-1] if children else model


def to_numpy_image(img_pil: Image.Image) -> np.ndarray:
	arr = np.array(img_pil).astype(np.float32) / 255.0
	if arr.ndim == 2:
		arr = np.stack([arr, arr, arr], axis=-1)
	return arr


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")
	if GradCAM is None and EigenCAM is None:
		raise SystemExit("Please install pytorch-grad-cam: pip install git+https://github.com/jacobgil/pytorch-grad-cam.git")

	if not os.path.isfile(MODEL_PATH):
		raise SystemExit(f"Model file not found: {MODEL_PATH}")
	if not os.path.isdir(INPUT_DIR):
		raise SystemExit(f"Input directory not found: {INPUT_DIR}")

	model, classes = load_clean_model(MODEL_PATH, device)
	tfms = build_tfms(IMG_SIZE)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# Collect images
	exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
	images: List[str] = []
	for ext in exts:
		images.extend(glob(os.path.join(INPUT_DIR, "**", ext), recursive=True))
	images = images[:MAX_IMAGES]
	if not images:
		raise SystemExit("No images found in INPUT_DIR.")

	# Build CAM object (remove unsupported args for compatibility)
	target_layer = pick_target_layer(model)
	cam_class = EigenCAM if USE_EIGEN_CAM else GradCAM
	try:
		cam = cam_class(model=model, target_layers=[target_layer])
	except TypeError:
		cam = cam_class(model=model, target_layers=[target_layer])

	for img_path in tqdm(images, desc="Grad-CAM"):
		img_pil = Image.open(img_path).convert('RGB')
		# Resize the display image to match CAM size
		display_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
		rgb_np = to_numpy_image(display_pil)
		inp = tfms(img_pil).unsqueeze(0).to(device)

		with torch.no_grad():
			logits = model(inp)
			pred_idx = int(logits.argmax(dim=1).item())
			pred_name = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)

		targets = None  # default uses predicted class; EigenCAM ignores targets
		grayscale_cam = cam(input_tensor=inp, targets=targets)
		grayscale_cam = grayscale_cam[0]

		overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
		out_rel = os.path.relpath(img_path, INPUT_DIR).replace(os.sep, "_")
		out_file = os.path.join(OUTPUT_DIR, f"cam_{pred_idx}_{out_rel}")
		Image.fromarray(overlay).save(out_file)

	print(f"Saved overlays to: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()


