# pip install grad-cam opencv-python torch torchvision pillow

import torch
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# 1. Load model
model = timm.create_model('caformer_s18.sail_in1k', pretrained=False, num_classes=38)
model.load_state_dict(torch.load('models/model_v2/best_caformer_model.pth')['model_state_dict'])
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2. Target layer (chọn layer cuối)
target_layers = [model.stages[-1]]  # hoặc model.head

# 3. Init GradCAM
cam = GradCAM(model=model, target_layers=target_layers)

# 4. Preprocess ảnh
img = Image.open("test_leaf.jpg").convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0).to(device)

# 5. Tính CAM
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]  # shape: (H, W)

# 6. Chuẩn bị ảnh RGB
rgb_img = np.array(img.resize((224, 224))) / 255.0
rgb_img = rgb_img.astype(np.float32)

# 7. Overlay
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cam_image = Image.fromarray(cam_image)

# 8. Lưu
cam_image.save("gradcam_output.jpg")
print("Đã lưu Grad-CAM: gradcam_output.jpg")
