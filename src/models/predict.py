"""Inference helper for a trained anatomical-plane model bundle."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(image: Image.Image, model_path=None):
    path = Path(model_path) if model_path else ROOT / "models" / "anatomical_plane_mobilenetv2.pth"
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {"available": False, "message": "No trained model bundle found. Train the anatomical-plane model first."}

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    classes = bundle["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    score, idx = torch.max(probabilities, dim=0)
    return {
        "available": True,
        "task": "anatomical_plane_classification",
        "predicted_class": classes[idx.item()],
        "confidence": round(score.item(), 4),
        "probabilities": {classes[i]: round(probabilities[i].item(), 4) for i in range(len(classes))},
        "clinical_limit": "This is not a Spina Bifida detector and must not be used for diagnosis.",
    }
