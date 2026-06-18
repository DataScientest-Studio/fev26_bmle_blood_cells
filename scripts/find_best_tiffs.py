#!/usr/bin/env python3
"""
Trouve les 3 meilleurs tiffs du CancerImagingArchive avec confiance ~98%
sur les modèles DL fold 1.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")

if not os.getenv("CROSSVAL_CACHE_DIR") or not os.getenv("CANCER_ARCHIVE_DIR"):
    raise EnvironmentError(
        "CROSSVAL_CACHE_DIR et CANCER_ARCHIVE_DIR doivent être définis dans ton .env local "
        "(chemins personnels, propres à chaque machine)."
    )
CROSSVAL_DIR = Path(os.environ["CROSSVAL_CACHE_DIR"])
ARCHIVE_DIR  = Path(os.environ["CANCER_ARCHIVE_DIR"])

CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "ig",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]

# Correspondance dossiers → classe modèle
FOLDER_TO_CLASS = {
    "BAS": "basophil",
    "EOS": "eosinophil",
    "EBO": "erythroblast",
    "LYT": "lymphocyte",
    "LYA": "lymphocyte",
    "MON": "monocyte",
    "NGB": "neutrophil",
    "NGS": "neutrophil",
}

MODELS_CFG = {
    "DenseNet_121":    {"timm_name": "densenet121",        "input_size": 224},
    "ConvNeXt_Tiny":   {"timm_name": "convnext_tiny",      "input_size": 224},
    "EfficientNet_B3": {"timm_name": "tf_efficientnet_b3", "input_size": 300},
    "ResNet_50":       {"timm_name": "resnet50",            "input_size": 224},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_CONF   = 98.0
TOLERANCE     = 2.0  # ±2%


def load_fold1_models(device):
    models = {}
    for key, cfg in MODELS_CFG.items():
        pth = CROSSVAL_DIR / "fold_1" / f"best_fold1_{key}.pth"
        state = torch.load(pth, map_location="cpu", weights_only=True)
        head_keys = [k for k in state
                     if k.startswith(("classifier", "head", "fc"))
                     and k.endswith(".weight") and state[k].ndim == 2]
        n_cls = state[head_keys[0]].shape[0] if head_keys else len(CLASS_NAMES)
        m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=n_cls)
        m.load_state_dict(state)
        m.eval().to(device)
        models[key] = m
        print(f"  OK {key}")
    return models


def predict(model, img_path, input_size, device):
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        proba = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(proba.argmax())
    return CLASS_NAMES[idx], round(float(proba[idx]) * 100, 2)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Chargement modèles fold 1...")
    models = load_fold1_models(device)

    # Collecter tous les tiffs des dossiers retenus
    all_images = []
    for folder, true_class in FOLDER_TO_CLASS.items():
        folder_path = ARCHIVE_DIR / folder
        tiffs = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
        for p in tiffs:
            all_images.append((p, true_class, folder))

    print(f"\n{len(all_images)} tiffs à évaluer dans {len(FOLDER_TO_CLASS)} dossiers\n")

    densenet_cfg = {"DenseNet_121": MODELS_CFG["DenseNet_121"]}

    results = []
    for img_path, true_class, folder in tqdm(all_images, unit="img"):
        pred_class, conf = predict(models["DenseNet_121"], img_path, densenet_cfg["DenseNet_121"]["input_size"], device)
        if pred_class == true_class:
            dist = abs(conf - TARGET_CONF)
            results.append({
                "fichier":   img_path.name,
                "dossier":   folder,
                "classe":    true_class,
                "modele":    "DenseNet_121",
                "confiance": conf,
                "dist_98":   round(dist, 2),
            })

    if not results:
        print("Aucun résultat trouvé.")
        return

    # Trier par distance à 98%
    results.sort(key=lambda x: x["dist_98"])

    top3 = results[:3]

    print("\n" + "=" * 60)
    print("TOP 3 — Tiffs les plus proches de 98% sur DenseNet-121")
    print("=" * 60)
    for i, r in enumerate(top3, 1):
        print(f"\n#{i}")
        print(f"  Fichier   : {r['fichier']}")
        print(f"  Dossier   : {r['dossier']} → {r['classe']}")
        print(f"  Modèle    : {r['modele']}")
        print(f"  Confiance : {r['confiance']}%  (écart 98% : {r['dist_98']}%)")
        print(f"  Chemin    : {ARCHIVE_DIR / r['dossier'] / r['fichier']}")

    in_range = [r for r in results if r["dist_98"] <= TOLERANCE]
    print(f"\n{len(in_range)} résultats dans la fenêtre 96-100% (DenseNet uniquement)")


if __name__ == "__main__":
    main()
