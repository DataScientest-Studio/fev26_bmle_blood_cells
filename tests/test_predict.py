"""Tests du pipeline de prédiction sur les images fixtures."""
import pytest
from pathlib import Path
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
MODELS_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODELS_DIR / "best_DenseNet_121.pth"
CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]


def test_model_file_exists():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modèle non disponible en CI : {MODEL_PATH}")
    assert MODEL_PATH.exists()


def test_metadata_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non disponible en CI")
    meta = MODELS_DIR / "metadata.json"
    assert meta.exists(), "metadata.json introuvable"


def test_predict_returns_8_classes():
    import torch
    import timm
    import torchvision.transforms as T

    if not MODEL_PATH.exists():
        pytest.skip("Modèle non disponible en CI")
    pth = MODEL_PATH

    model = timm.create_model("densenet121", pretrained=False, num_classes=8)
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    images = list(FIXTURES.glob("*.jpg"))
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
