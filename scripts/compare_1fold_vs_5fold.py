#!/usr/bin/env python3
"""
Compare predictions 1-fold vs 5-fold ensemble sur les images test_cropped.
Trouve les images mieux classifiées par l'ensemble 5 folds.
"""
import os
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")

# ── Chemins ──────────────────────────────────────────────────────────────────
if not os.getenv("ONEDRIVE_CACHE_DIR"):
    raise EnvironmentError(
        "ONEDRIVE_CACHE_DIR doit être défini dans ton .env local (chemin personnel)."
    )
ONEDRIVE = Path(os.environ["ONEDRIVE_CACHE_DIR"])
TEST_DIR     = ONEDRIVE / "MendereyAmeliore" / "test_cropped"
CROSSVAL_DIR = ONEDRIVE / "DL_crossval_ameliorees"
PRED_1FOLD   = ONEDRIVE / "Rapports" / "predictions_test_full.xlsx"
OUT_CSV      = ONEDRIVE / "Rapports" / "compare_1fold_vs_5fold.csv"

# ── Constantes ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["basophil", "eosinophil", "erythroblast", "ig",
                 "lymphocyte", "monocyte", "neutrophil", "platelet"]
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

MODELS_CFG = {
    "DenseNet_121":    {"timm_name": "densenet121",        "input_size": 224},
    "ConvNeXt_Tiny":   {"timm_name": "convnext_tiny",      "input_size": 224},
    "EfficientNet_B3": {"timm_name": "tf_efficientnet_b3", "input_size": 300},
    "ResNet_50":       {"timm_name": "resnet50",            "input_size": 224},
}


def load_fold_models(crossval_dir: Path, n_folds: int):
    """Charge les modèles pour chaque fold. Retourne dict[model_key -> list[model]]."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    all_models = {}
    for key, cfg in MODELS_CFG.items():
        fold_models = []
        for i in range(1, n_folds + 1):
            pth = crossval_dir / f"fold_{i}" / f"best_fold{i}_{key}.pth"
            if not pth.exists():
                print(f"  MANQUANT: {pth.name}")
                continue
            state = torch.load(pth, map_location="cpu", weights_only=True)
            head_keys = [k for k in state
                         if k.startswith(("classifier", "head", "fc"))
                         and k.endswith(".weight") and state[k].ndim == 2]
            n_cls = state[head_keys[0]].shape[0] if head_keys else len(CLASS_NAMES)
            m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=n_cls)
            m.load_state_dict(state)
            m.eval().to(device)
            fold_models.append(m)
        print(f"  {key}: {len(fold_models)}/{n_folds} folds chargés")
        all_models[key] = fold_models
    return all_models, device


def predict_ensemble(fold_models, img_path: Path, input_size: int, device: str):
    """Retourne (classe_prédite, confiance%) via moyenne des probas sur les folds."""
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probas = [torch.softmax(m(tensor), dim=1)[0].cpu().numpy() for m in fold_models]
    proba = np.mean(probas, axis=0)
    idx = int(proba.argmax())
    return CLASS_NAMES[idx], round(float(proba[idx]) * 100, 1)


def main():
    print("=" * 60)
    print("Comparaison 1-fold vs 5-fold — images test_cropped")
    print("=" * 60)

    # Charger les prédictions 1-fold existantes
    df_1fold = pd.read_excel(PRED_1FOLD)
    print(f"\nPrédictions 1-fold chargées : {len(df_1fold)} lignes")

    # Lister les images de test
    images = sorted([p for p in TEST_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    print(f"Images test_cropped : {len(images)}")

    # Charger les 5 folds
    print("\nChargement des modèles 5 folds...")
    all_models, device = load_fold_models(CROSSVAL_DIR, n_folds=5)

    # Inférence 5 folds sur toutes les images
    print(f"\nInférence 5 folds sur {len(images)} images (device={device})...")
    rows_5fold = {}
    t0 = time.time()

    for img_path in tqdm(images, unit="img"):
        row = {}
        for key, cfg in MODELS_CFG.items():
            fold_models = all_models[key]
            if not fold_models:
                row[f"{key}_5fold_pred"] = None
                row[f"{key}_5fold_conf"] = None
                continue
            pred, conf = predict_ensemble(fold_models, img_path, cfg["input_size"], device)
            row[f"{key}_5fold_pred"] = pred
            row[f"{key}_5fold_conf"] = conf
        rows_5fold[img_path.name] = row

    elapsed = time.time() - t0
    print(f"Terminé en {elapsed:.1f}s")

    # Fusionner avec les prédictions 1-fold
    df_5fold = pd.DataFrame.from_dict(rows_5fold, orient="index").reset_index()
    df_5fold.rename(columns={"index": "fichier"}, inplace=True)

    df = df_1fold.merge(df_5fold, on="fichier", how="inner")

    # Calculer correct/incorrect pour chaque modèle × chaque configuration
    model_keys_1fold = {
        "DenseNet_121": "DenseNet_121",
        "ConvNeXt_Tiny": "ConvNeXt_Tiny",
        "EfficientNet_B3": "EfficientNet_B3",
        "ResNet_50": "ResNet_50",
    }
    col_map_1fold = {
        "DenseNet_121":    "DenseNet_121_pred",
        "ConvNeXt_Tiny":   "ConvNeXt_Tiny_pred",
        "EfficientNet_B3": "EfficientNet_B3_pred",
        "ResNet_50":       "ResNet_50_pred",
    }

    for key in MODELS_CFG:
        col_1 = col_map_1fold[key]
        col_5 = f"{key}_5fold_pred"
        if col_1 in df.columns and col_5 in df.columns:
            df[f"{key}_ok_1fold"] = df[col_1] == df["classe_reelle"]
            df[f"{key}_ok_5fold"] = df[col_5] == df["classe_reelle"]
            df[f"{key}_better_5fold"] = (~df[f"{key}_ok_1fold"]) & df[f"{key}_ok_5fold"]
            df[f"{key}_worse_5fold"]  = df[f"{key}_ok_1fold"] & (~df[f"{key}_ok_5fold"])

    df.to_csv(OUT_CSV, index=False)
    print(f"\nRésultats sauvegardés : {OUT_CSV}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ — Mieux classifiés par 5-fold vs 1-fold")
    print("=" * 60)
    total = len(df)
    for key in MODELS_CFG:
        col_b = f"{key}_better_5fold"
        col_w = f"{key}_worse_5fold"
        if col_b not in df.columns:
            continue
        n_better = df[col_b].sum()
        n_worse  = df[col_w].sum()
        n_ok1    = df[f"{key}_ok_1fold"].sum()
        n_ok5    = df[f"{key}_ok_5fold"].sum()
        print(f"\n{key}:")
        print(f"  Précision 1-fold : {n_ok1}/{total} = {100*n_ok1/total:.2f}%")
        print(f"  Précision 5-fold : {n_ok5}/{total} = {100*n_ok5/total:.2f}%")
        print(f"  Mieux par 5-fold : {n_better} images")
        print(f"  Moins bien       : {n_worse} images")
        if n_better > 0:
            better_imgs = df[df[col_b]][["fichier", "classe_reelle",
                                         col_map_1fold[key], f"{key}_5fold_pred"]].head(10)
            print(f"  Exemples (jusqu'à 10) :")
            print(better_imgs.to_string(index=False))

    # ── Images mieux classifiées par TOUS les modèles en 5-fold ──────────────
    print("\n" + "=" * 60)
    print("Images mieux classifiées par AU MOINS 1 modèle en 5-fold")
    print("=" * 60)
    better_cols = [f"{k}_better_5fold" for k in MODELS_CFG if f"{k}_better_5fold" in df.columns]
    df["n_models_better_5fold"] = df[better_cols].sum(axis=1)
    best_imgs = df[df["n_models_better_5fold"] > 0].sort_values("n_models_better_5fold", ascending=False)
    print(f"Total : {len(best_imgs)} images")
    if len(best_imgs) > 0:
        cols_show = ["fichier", "classe_reelle", "n_models_better_5fold"] + better_cols
        print(best_imgs[cols_show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
