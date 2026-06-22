"""Tests d'intégration — pipeline de prédiction DenseNet-121.

Ces tests chargent le vrai modèle .pth et vérifient l'intégralité du pipeline
(chargement → prétraitement → inférence → softmax).
Skippés automatiquement en CI si le fichier modèle est absent.
"""
import pytest
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

MODELS_DIR = Path(__file__).parents[3] / "models"
MODEL_PATH = MODELS_DIR / "best_DenseNet_121.pth"
CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]


@pytest.mark.integration
def test_model_file_exists():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle non disponible en CI : {MODEL_PATH}")
    assert MODEL_PATH.exists()


@pytest.mark.integration
def test_metadata_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non disponible en CI")
    meta = MODELS_DIR / "metadata.json"
    assert meta.exists(), "metadata.json introuvable"


@pytest.mark.integration
def test_predict_returns_8_classes(fixtures_dir):
    import timm

    if not MODEL_PATH.exists():
        pytest.skip("Modèle non disponible en CI")

    model = timm.create_model("densenet121", pretrained=False, num_classes=8)
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    images = list(fixtures_dir.glob("*.jpg"))
    assert images
    img = Image.open(images[0]).convert("RGB")

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1).numpy()[0]

    assert len(proba) == 8
    assert abs(proba.sum() - 1.0) < 1e-5
    assert proba.max() <= 1.0 and proba.min() >= 0.0
