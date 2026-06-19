#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend — prédictions ML/DL + interface MLflow
Lancement :
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import sys
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
    description="API de classification de cellules sanguines — DenseNet-121 (8 classes)",
    version="1.0.0",
    openapi_tags=[
        {"name": "Inférence",    "description": "Prédiction sur une image de cellule sanguine"},
        {"name": "Entraînement", "description": "Lancement et suivi de l'entraînement du modèle"},
        {"name": "Info",         "description": "État de l'API et informations sur le modèle"},
    ],
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


MLFLOW_MODEL_NAME = "blood-cell-densenet121"


def _load_from_registry():
    """Charge le modèle @production depuis le MLflow Registry."""
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@production"
    loaded = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    loaded = loaded.to(DEVICE)
    loaded.eval()
    print(f"[MLflow] Modèle chargé : {model_uri}")
    return loaded


def _load_from_file():
    """Charge le modèle depuis un .pth local (fallback)."""
    model_paths = [
        "models/best_densenet121.pth",
        "models/best_DenseNet_121.pth",
        "/app/models/best_densenet121.pth",
        "/app/models/best_DenseNet_121.pth",
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"Modèle introuvable. Chemins testés : {model_paths}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes_ckpt = state_dict["classifier.weight"].shape[0]

    m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes_ckpt)
    m.load_state_dict(state_dict)
    m = m.to(DEVICE)
    m.eval()
    print(f"[Fallback] Modèle chargé depuis fichier : {model_path}")
    return m


def load_model():
    """Charge le modèle — MLflow Registry @production en priorité, .pth local en fallback."""
    global model, model_device

    if model is not None:
        return model

    try:
        model = _load_from_registry()
    except Exception as exc:
        print(f"[warn] MLflow Registry indisponible ({exc}) — fallback .pth local")
        model = _load_from_file()

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


@app.get("/health", tags=["Info"])
async def health():
    """Vérifier que l'API est accessible."""
    return {"status": "ok"}


@app.get("/", tags=["Info"])
async def root():
    """Informations générales sur l'API."""
    return {
        "message": "Blood Cell Classifier API",
        "version": "1.0.0",
        "model": "DenseNet-121",
        "classes": CLASSES,
    }


@app.post("/predict", tags=["Inférence"])
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


@app.get("/classes", tags=["Info"])
async def get_classes():
    """Retourner les 8 classes de cellules sanguines."""
    return {"classes": CLASSES, "num_classes": NUM_CLASSES}


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Informations sur le modèle chargé."""
    return {
        "model": "DenseNet-121",
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "classes": CLASSES,
    }


class TrainingParams(BaseModel):
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    epochs_head: Optional[int] = 5
    epochs_full: Optional[int] = 10
    batch_size: Optional[int] = 32


@app.post("/training", tags=["Entraînement"])
async def run_training(params: TrainingParams = TrainingParams()):
    """
    Lance l'entraînement DenseNet-121 et retourne les métriques de base.

    Returns:
    {
        "status": "ok",
        "val_acc": 0.961,
        "test_acc": 0.958,
        "model_path": "models/best_densenet121.pth"
    }
    """
    try:
        from src.train.training import train, DEFAULT_CFG

        data_dir = (
            Path(params.data_dir) if params.data_dir
            else ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
        )
        output_dir = (
            Path(params.output_dir) if params.output_dir
            else ROOT / os.getenv("MODELS_DIR", "models")
        )

        cfg = {
            **DEFAULT_CFG,
            "epochs_head": params.epochs_head,
            "epochs_full": params.epochs_full,
            "batch_size": params.batch_size,
        }

        metrics = train(data_dir, output_dir, cfg)

        return {
            "status": "ok",
            "val_acc": round(metrics["best_val_acc"], 4),
            "test_acc": round(metrics["test_acc"], 4),
            "model_path": str(output_dir / "best_densenet121.pth"),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
