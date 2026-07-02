#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend — prédictions ML/DL + interface MLflow
Lancement :
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
import sys
import time
import numpy as np

# Force CPU — MPS hang sur macOS récent avec certaines versions de PyTorch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_NO_MPS"] = "1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Security, Depends  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.security import APIKeyHeader  # noqa: E402
from pathlib import Path  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import Optional  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.models import densenet121  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Force CPU (MPS désactivé — hang sur macOS récent)
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
model_version = None  # version MLflow Registry actuellement chargée (None si fallback .pth)

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

# ── Authentification API Key ─────────────────────────────────────────────────
# Si API_SECRET_KEY n'est pas définie (dev local, CI), l'auth est désactivée.
# En production (docker-compose), la variable DOIT être définie dans le .env.
_API_SECRET_KEY = os.getenv("API_SECRET_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)):
    if _API_SECRET_KEY is None:
        return  # auth désactivée si variable non configurée
    if api_key != _API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide ou absente.")


def _extract_image_stats(image: Image.Image) -> dict:
    """Extrait les stats image pour le monitoring de data drift."""
    import numpy as np
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    return {
        "mean_brightness": float(gray.mean()),
        "std_brightness":  float(gray.std()),
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
        "image_width":  image.size[0],
        "image_height": image.size[1],
    }


def _log_prediction(
    image_name: str, predicted_class: str, confidence: float, image: Image.Image = None,
    patient_id: Optional[int] = None, patient_name: Optional[str] = None,
    triggered_by: Optional[str] = None,
) -> Optional[int]:
    """Logue la prédiction dans Supabase et retourne son id (None si indisponible).
    Échec silencieux — ne doit jamais faire échouer une prédiction réelle."""
    try:
        from src.auth.db import get_connection
        stats = _extract_image_stats(image) if image is not None else {}
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions
                (image_name, predicted_class, confidence, model_version,
                 mean_brightness, std_brightness, mean_r, mean_g, mean_b,
                 image_width, image_height, patient_id, patient_name, triggered_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                image_name, predicted_class, confidence, model_version,
                stats.get("mean_brightness"), stats.get("std_brightness"),
                stats.get("mean_r"), stats.get("mean_g"), stats.get("mean_b"),
                stats.get("image_width"), stats.get("image_height"), patient_id, patient_name,
                triggered_by,
            ),
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"[warn] Supabase (predictions) indisponible : {e}")
        return None


def _load_from_registry():
    """Charge le modèle @production depuis le MLflow Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    global model_version
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@production"
    loaded = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    loaded = loaded.to(DEVICE)
    loaded.eval()

    try:
        mv = MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        model_version = mv.version
    except Exception:
        model_version = None

    print(f"[MLflow] Modèle chargé : {model_uri} (version {model_version})")
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
    except Exception:
        try:
            model = _load_from_file()
        except Exception as exc:
            raise RuntimeError(f"Impossible de charger le modèle : {exc}")

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


@app.post("/predict", tags=["Inférence"], dependencies=[Depends(require_api_key)])
async def predict(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    patient_name: Optional[str] = Form(None),
    triggered_by: Optional[str] = Form(None),
):
    """
    Prédiction DL sur une image uploadée.

    Returns:
    {
        "prediction_id": 42,
        "predicted_class": "Lymphocyte",
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
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Préparer la réponse
        pred_idx = pred_class.item()
        pred_label = CLASSES[pred_idx]
        pred_confidence = confidence.item()

        # Toutes les probabilités
        all_probas = {CLASSES[i]: float(probs[0, i].item()) for i in range(NUM_CLASSES)}

        prediction_id = _log_prediction(
            file.filename, pred_label, round(pred_confidence, 3), image,
            patient_id=patient_id, patient_name=patient_name, triggered_by=triggered_by,
        )

        return {
            "prediction_id": prediction_id,
            "predicted_class": pred_label,
            "confidence": round(pred_confidence, 3),
            "is_critical": pred_label.lower() in {"erythroblast", "ig"},
            "inference_ms": inference_ms,
            "top3": sorted(all_probas.items(), key=lambda x: x[1], reverse=True)[:3],
            "all_probas": all_probas,
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Prédiction échouée"
        }


@app.post("/gradcam", tags=["Inférence"], dependencies=[Depends(require_api_key)])
async def gradcam_predict(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    patient_name: Optional[str] = Form(None),
    triggered_by: Optional[str] = Form(None),
):
    """
    Prédiction + GradCAM pour une image.

    Returns:
    {
        "prediction_id": 42,
        "predicted_class": "Lymphocyte",
        "confidence": 0.987,
        "is_critical": false,
        "all_probas": {...},
        "gradcam_b64": "<base64 PNG 224x224>"
    }
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus as GradCAMLib
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        m = load_model()

        image_data = await file.read()
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")

        img_224 = pil_img.resize((224, 224))
        img_array = np.array(img_224, dtype=np.float32) / 255.0

        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = m(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[0, pred_idx].item())

        all_probas = {CLASSES[i]: float(probs[0, i].item()) for i in range(NUM_CLASSES)}

        try:
            target_layer = m.features.norm5
        except AttributeError:
            target_layer = list(m.features.children())[-1]

        cam_obj = GradCAMLib(model=m, target_layers=[target_layer])
        grayscale_cam = cam_obj(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)],
        )
        del cam_obj

        visualization = show_cam_on_image(img_array, grayscale_cam[0], use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(visualization).save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode()

        prediction_id = _log_prediction(
            file.filename, CLASSES[pred_idx], round(confidence, 3), pil_img,
            patient_id=patient_id, patient_name=patient_name, triggered_by=triggered_by,
        )

        return {
            "prediction_id": prediction_id,
            "predicted_class": CLASSES[pred_idx],
            "confidence": round(confidence, 3),
            "is_critical": CLASSES[pred_idx] in {"Erythroblast", "IG"},
            "all_probas": all_probas,
            "gradcam_b64": gradcam_b64,
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


class FeedbackPayload(BaseModel):
    prediction_id: int
    agrees: bool
    corrected_class: Optional[str] = None
    comment: Optional[str] = None


@app.post("/feedback", tags=["Inférence"], dependencies=[Depends(require_api_key)])
async def feedback(payload: FeedbackPayload):
    """
    Enregistre le désaccord (ou l'accord) d'un médecin sur une prédiction,
    relié à predictions.id (renvoyé par /predict sous prediction_id).
    """
    if payload.corrected_class is not None and payload.corrected_class not in CLASSES:
        raise HTTPException(status_code=422, detail=f"corrected_class doit être l'une de {CLASSES}")

    try:
        from src.auth.db import get_connection
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_feedback (prediction_id, agrees, corrected_class, comment)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (payload.prediction_id, payload.agrees, payload.corrected_class, payload.comment),
        )
        feedback_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"feedback_id": feedback_id, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Supabase indisponible : {e}")


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


@app.post("/training", tags=["Entraînement"], dependencies=[Depends(require_api_key)])
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
            "mlflow_run_id": metrics.get("run_id", ""),
            "model_path": str(output_dir / "best_densenet121.pth"),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
