import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import BreastCancerModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = BreastCancerModel()

model.load_state_dict(
    torch.load(
        "../models/best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

target_layers = [
    model.model.features[-1]
]

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")

    rgb_img = np.array(
        image.resize((224, 224))
    ) / 255.0

    tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs("../uploads", exist_ok=True)

    output_path = "../uploads/gradcam.jpg"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path