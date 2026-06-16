"""
Inférence DenseNet-121 — classification de cellules sanguines.

Usage :
    python -m src.models.predict_model --image path/to/cell.jpg
    python -m src.models.predict_model --image path/to/cell.jpg --model models/best_densenet121.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import timm
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CLASSES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
CRITICAL = {"erythroblast", "ig"}
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(model_path: Path):
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    model = timm.create_model("densenet121", pretrained=False, num_classes=len(CLASSES))
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


def _preprocess(image_path: Path) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(Image.open(image_path).convert("RGB")).unsqueeze(0)


def predict(image_path: Path, model_path: Path) -> dict:
    """Retourne la prédiction pour une image.

    Returns:
        {
            "predicted_class": str,
            "confidence": float,
            "is_critical": bool,
            "top3": [{"class": str, "probability": float}, ...],
            "all_probabilities": {class: float, ...},
        }
    """
    model, device = load_model(model_path)
    tensor = _preprocess(image_path).to(device)

    with torch.no_grad():
        proba = torch.softmax(model(tensor).float(), dim=1).cpu().numpy()[0]

    top3_idx = np.argsort(proba)[::-1][:3]
    predicted_class = CLASSES[top3_idx[0]]

    return {
        "predicted_class": predicted_class,
        "confidence": float(proba[top3_idx[0]]),
        "is_critical": predicted_class in CRITICAL,
        "top3": [
            {"class": CLASSES[i], "probability": float(proba[i])}
            for i in top3_idx
        ],
        "all_probabilities": {cls: float(proba[i]) for i, cls in enumerate(CLASSES)},
    }


def _parse_args():
    default_model = ROOT / os.getenv("MODELS_DIR", "models") / "best_densenet121.pth"
    parser = argparse.ArgumentParser(description="Classifie une image de cellule sanguine")
    parser.add_argument("--image", required=True, help="Chemin vers l'image (.jpg/.png/.tiff)")
    parser.add_argument("--model", default=str(default_model), help="Chemin vers le .pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Lancez d'abord : python -m src.train.training"
        )

    result = predict(image_path, model_path)

    warn = "  ⚠️  CRITIQUE" if result["is_critical"] else ""
    print(f"\nPrédiction : {result['predicted_class'].upper()}{warn}")
    print(f"Confiance  : {result['confidence']*100:.1f}%")
    print("\nTop 3 :")
    for item in result["top3"]:
        critical = " ⚠️" if item["class"] in CRITICAL else ""
        print(f"  {item['class']:15s} {item['probability']*100:.1f}%{critical}")
