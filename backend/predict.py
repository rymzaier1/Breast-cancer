import torch

from PIL import Image
from torchvision import transforms

from model import BreastCancerModel

classes = [
    "Benign",
    "Malignant"
]

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

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    return {
        "class": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }