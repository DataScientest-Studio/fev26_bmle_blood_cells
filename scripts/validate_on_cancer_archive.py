"""
Validation des modèles DL (DenseNet-121, ConvNeXt-Tiny) sur le dataset
CancerImagingArchive regroupé en 7 classes.

Usage:
    python validate_on_cancer_archive.py
"""

from pathlib import Path
import json
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Chemins ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/Users/fredericdelabot/Documents/DataScientest/Projet CHU Lyon"
                   "/CancerImagingArchive/regrouped")
MODELS_DIR  = Path("reports/Fred_DL_pipeline_report_full")
OUT_DIR     = Path("reports/validation_cancer_archive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constantes (identiques à folder_report.py) ────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_CFGS = {
    "DenseNet-121":  {"timm_name": "densenet121",   "input_size": 224, "file": "best_DenseNet_121.pth"},
    "ConvNeXt-Tiny": {"timm_name": "convnext_tiny", "input_size": 224, "file": "best_ConvNeXt_Tiny.pth"},
}

# Classes du modèle — 9 sorties (8 réelles + "output" spurieux, ordre alphabétique)
MODEL_CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
                 "lymphocyte", "monocyte", "neutrophil", "output", "platelet"]
VALID_CLASSES = [c for c in MODEL_CLASSES if c != "output"]


def load_model(cfg: dict, model_dir: Path):
    pth   = model_dir / cfg["file"]
    state = torch.load(pth, map_location="cpu", weights_only=True)
    head_keys = [k for k in state
                 if k.startswith(("classifier", "head", "fc"))
                 and k.endswith(".weight") and state[k].ndim == 2]
    num_classes = state[head_keys[0]].shape[0] if head_keys else len(MODEL_CLASSES)
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=num_classes)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, img_path: Path, input_size: int) -> str:
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        proba = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    # Masquer la classe "output" spurieuse et renormaliser
    valid_mask  = np.array([c in set(VALID_CLASSES) for c in MODEL_CLASSES])
    proba       = proba * valid_mask
    total       = proba.sum()
    if total > 0:
        proba /= total
    return MODEL_CLASSES[int(proba.argmax())]


def collect_dataset():
    """Retourne (paths, true_labels) pour toutes les images du dataset regroupé."""
    paths, labels = [], []
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".tiff", ".tif", ".jpg", ".jpeg", ".png"}:
                paths.append(img_path)
                labels.append(cls_dir.name)
    return paths, labels


def run_validation(model_name: str, model):
    input_size = MODEL_CFGS[model_name]["input_size"]
    paths, y_true = collect_dataset()

    y_pred = []
    for p in tqdm(paths, desc=model_name, unit="img"):
        y_pred.append(predict(model, p, input_size))

    # Classes présentes dans le dataset (pas platelet)
    present_classes = sorted(set(y_true))

    report = classification_report(y_true, y_pred, labels=present_classes, digits=3)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print('='*60)
    print(report)

    # Sauvegarde texte
    (OUT_DIR / f"report_{model_name.replace('-','_')}.txt").write_text(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=present_classes)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"{model_name} — CancerImagingArchive")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"cm_{model_name.replace('-','_')}.png", dpi=150)
    plt.close(fig)
    print(f"  → Sauvegardé dans {OUT_DIR}")


def main():
    print(f"Dataset : {DATA_DIR}")
    paths, labels = collect_dataset()
    print(f"Images  : {len(paths)}  |  Classes : {sorted(set(labels))}\n")

    for name, cfg in MODEL_CFGS.items():
        pth = MODELS_DIR / cfg["file"]
        if not pth.exists():
            print(f"[SKIP] {name} — fichier introuvable : {pth}")
            continue
        model = load_model(cfg, MODELS_DIR)
        run_validation(name, model)

    print(f"\nRapports complets : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
