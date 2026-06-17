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
MLFLOW_MODEL_NAME = "blood-cell-densenet121"
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_from_registry(device: str):
    """Charge le modèle @production depuis le MLflow Registry."""
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@production"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device).eval()
    print(f"  [MLflow] Modele charge : {model_uri}")
    return model, device


def load_model(model_path: Path = None):
    """Charge le modèle depuis le MLflow Registry (@production) ou un .pth local."""
    device = _device()

    if model_path is None:
        try:
            return _load_from_registry(device)
        except Exception as exc:
            raise FileNotFoundError(
                f"MLflow Registry indisponible ({exc})\n"
                "Passez --model path/to/best_densenet121.pth ou démarrez le serveur MLflow."
            ) from exc

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


def predict(image_path: Path, model_path: Path = None) -> dict:
    """Retourne la prédiction pour une image.

    model_path=None  -> charge depuis MLflow Registry @production
    model_path=Path  -> charge depuis le fichier .pth local

    Returns:
        {
            "predicted_class": str,
            "confidence": float,
            "is_critical": bool,
            "inference_ms": float,
            "top3": [{"class": str, "probability": float}, ...],
            "all_probabilities": {class: float, ...},
        }
    """
    import time
    model, device = load_model(model_path)
    tensor = _preprocess(image_path).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        proba = torch.softmax(model(tensor).float(), dim=1).cpu().numpy()[0]
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    top3_idx = np.argsort(proba)[::-1][:3]
    predicted_class = CLASSES[top3_idx[0]]

    return {
        "predicted_class": predicted_class,
        "confidence": float(proba[top3_idx[0]]),
        "is_critical": predicted_class in CRITICAL,
        "inference_ms": inference_ms,
        "top3": [
            {"class": CLASSES[i], "probability": float(proba[i])}
            for i in top3_idx
        ],
        "all_probabilities": {cls: float(proba[i]) for i, cls in enumerate(CLASSES)},
    }


def _log_to_supabase(image_name: str, result: dict, mlflow_run_id: str = "") -> None:
    """Logue la prédiction dans la table Supabase predictions. Silencieux si indisponible."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("SUPABASE_HOST"),
            port=int(os.getenv("SUPABASE_PORT", 5432)),
            dbname=os.getenv("SUPABASE_DB"),
            user=os.getenv("SUPABASE_USER"),
            password=os.getenv("SUPABASE_PASSWORD"),
            connect_timeout=5,
            sslmode="require",
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (image_name, predicted_class, confidence, mlflow_run_id)"
            " VALUES (%s, %s, %s, %s)",
            (image_name, result["predicted_class"], round(result["confidence"], 4), mlflow_run_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"  [warn] Supabase logging skipped: {exc}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Classifie une image de cellule sanguine")
    parser.add_argument("--image", required=True, help="Chemin vers l'image (.jpg/.png/.tiff)")
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers un .pth local (défaut : MLflow Registry @production)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model) if args.model else None

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}\n"
            "Lancez d'abord : python -m src.train.training"
        )

    result = predict(image_path, model_path)

    warn = "  [CRITIQUE]" if result["is_critical"] else ""
    print(f"\nPrediction : {result['predicted_class'].upper()}{warn}")
    print(f"Confiance  : {result['confidence']*100:.1f}%")
    print(f"Inference  : {result['inference_ms']} ms")
    print("\nTop 3 :")
    for item in result["top3"]:
        critical = " [!]" if item["class"] in CRITICAL else ""
        print(f"  {item['class']:15s} {item['probability']*100:.1f}%{critical}")

    _log_to_supabase(image_path.name, result)
