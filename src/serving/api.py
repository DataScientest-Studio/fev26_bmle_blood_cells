#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend — prédictions ML/DL + interface MLflow
Lancement :
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms
from torchvision.models import densenet121

# Détection du device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Classes de cellules sanguines — ordre identique à configs/densenet121.yaml et tests/
CLASSES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "IG",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]

NUM_CLASSES = len(CLASSES)

# Initialiser FastAPI
app = FastAPI(
    title="Blood Cell Classifier API",
    description="DL predictions + MLflow tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
model_device = None

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    """Charge le modèle DenseNet-121 une seule fois"""
    global model, model_device

    if model is not None:
        return model

    # Chemins où chercher le modèle (en ordre de priorité)
    model_paths = [
        "models/best_DenseNet_121.pth",
        "/app/models/best_DenseNet_121.pth",
        "reports/pour_mac/best_DenseNet_121.pth",
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Model found at: {model_path}")
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model not found. Searched in: {', '.join(model_paths)}"
        )

    # Charger le modèle
    model = densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    model_device = DEVICE

    return model


@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage"""
    try:
        load_model()
        print("Model loaded successfully at startup")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")


@app.get("/health")
async def health():
    """Vérifier que l'API est accessible"""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Route root"""
    return {
        "message": "Blood Cell Classifier API",
        "version": "1.0.0",
        "model": "DenseNet-121",
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédiction DL sur une image uploadée.

    Returns:
    {
        "class": "Lymphocyte",
        "confidence": 0.987,
        "all_probas": {
            "Basophil": 0.001,
            "Eosinophil": 0.002,
            ...
        }
    }
    """
    try:
        # Charger le modèle si nécessaire
        model = load_model()

        # Lire l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prétraiter
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prédire
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)

        # Préparer la réponse
        pred_idx = pred_class.item()
        pred_label = CLASSES[pred_idx]
        pred_confidence = confidence.item()

        # Toutes les probabilités
        all_probas = {CLASSES[i]: float(probs[0, i].item()) for i in range(NUM_CLASSES)}

        return {
            "class": pred_label,
            "confidence": round(pred_confidence, 3),
            "all_probas": all_probas
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Prédiction échouée"
        }


@app.get("/classes")
async def get_classes():
    """Retourner les 8 classes de cellules"""
    return {"classes": CLASSES, "num_classes": NUM_CLASSES}


@app.get("/model-info")
async def model_info():
    """Info sur le modèle chargé"""
    return {
        "model": "DenseNet-121",
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "classes": CLASSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
