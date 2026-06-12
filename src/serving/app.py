#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Streamlit : Classification de cellules sanguines
ML (SVM RBF) + DL (EfficientNet-B3 · ConvNeXt-Tiny · DenseNet-121 · ResNet-50)
Dataset : Mendeley PBC — 8 classes

Lancement :
    streamlit run src/streamlit/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import streamlit as st
from PIL import Image

# ── Chemin racine du projet ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
CRITICAL = {"erythroblast", "ig"}

MODELS_CONFIG = {
    "EfficientNet-B3": {"timm_name": "tf_efficientnet_b3", "input_size": 300},
    "ConvNeXt-Tiny":   {"timm_name": "convnext_tiny",       "input_size": 224},
    "DenseNet-121":    {"timm_name": "densenet121",          "input_size": 224},
    "ResNet-50":       {"timm_name": "resnet50",             "input_size": 224},
}

MODEL_COLORS = {
    "EfficientNet-B3": "#D85A30",
    "ConvNeXt-Tiny":   "#7F77DD",
    "DenseNet-121":    "#1D9E75",
    "ResNet-50":       "#378ADD",
    "SVM (RBF)":       "#E67E22",
}

CLASS_EMOJI = {
    "basophil":     "🔵",
    "eosinophil":   "🟠",
    "erythroblast": "🔴",
    "ig":           "🟣",
    "lymphocyte":   "🟢",
    "monocyte":     "🟡",
    "neutrophil":   "⚪",
    "platelet":     "🩷",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Chemin par défaut des modèles (peut être surchargé dans la sidebar)
_ONEDRIVE_CACHE   = Path(r"C:\Users\Dume\Documents\OneDriveDL\OneDrive\BloodCellCaches")
_MODEL_CANDIDATES = [
    _ONEDRIVE_CACHE / "DL_crossval_ameliorees" / "fold_1",
    ROOT / "reports" / "pour_mac",
]
DEFAULT_MODEL_DIR = next(
    (p for p in _MODEL_CANDIDATES if (p / "best_fold1_EfficientNet_B3.pth").exists()),
    ROOT / "reports" / "pour_mac",
)
_ML_PKL_CANDIDATES = [
    _ONEDRIVE_CACHE / "ML_learn" / "best_ml_model.pkl",
    ROOT / "reports" / "pour_mac" / "best_ml_model.pkl",
]
DEFAULT_ML_PKL = next((p for p in _ML_PKL_CANDIDATES if p.exists()), _ML_PKL_CANDIDATES[-1])


# ── Chargement modèles (caché par Streamlit) ──────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_dl_model(model_key: str, pth_path: str):
    import torch
    import timm
    cfg = MODELS_CONFIG[model_key]
    device = (
        "mps"  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=len(CLASSES))
    ckpt  = torch.load(pth_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_ml_model(pkl_path: str):
    import joblib
    bundle = joblib.load(pkl_path)
    return bundle["model"], bundle["scaler"], bundle["classes"]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_dl(img: Image.Image, input_size: int):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


def extract_features_ml(img: Image.Image) -> np.ndarray:
    import cv2
    IMG_SIZE = (128, 128)
    N_BINS   = 16
    LBP_BINS = 10

    img_rgb = np.array(img.convert("RGB").resize(IMG_SIZE))
    feats: list[float] = list(img_rgb.mean(axis=(0, 1))) + list(img_rgb.std(axis=(0, 1)))

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    feats += list(hsv.mean(axis=(0, 1))) + list(hsv.std(axis=(0, 1)))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    feats += list(lab.mean(axis=(0, 1))) + list(lab.std(axis=(0, 1)))

    for c in range(3):
        h, _ = np.histogram(img_rgb[:, :, c], bins=N_BINS, range=(0, 256), density=True)
        feats += list(h)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_gray, _ = np.histogram(gray, bins=N_BINS, range=(0, 256), density=True)
    feats += list(h_gray)

    try:
        from skimage.feature import local_binary_pattern
        lbp  = local_binary_pattern(gray, P=8, R=1, method="uniform")
        h_lbp, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
        feats += list(h_lbp)
    except Exception:
        feats += [0.0] * LBP_BINS

    return np.array(feats, dtype=np.float32)


# ── Inférence ─────────────────────────────────────────────────────────────────

def predict_dl(model, device: str, img: Image.Image, input_size: int) -> np.ndarray:
    import torch
    tensor = preprocess_dl(img, input_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        proba  = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
    return proba


def predict_ml(ml_model, scaler, img: Image.Image) -> np.ndarray:
    feats = extract_features_ml(img).reshape(1, -1)
    feats_sc = scaler.transform(feats)
    return ml_model.predict_proba(feats_sc)[0]


# ── Interface ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Blood Cell Classifier",
        page_icon="🔬",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🔬 Blood Cell Classifier")
        st.caption("Mendeley PBC — 8 classes — ML + DL")
        st.divider()

        st.subheader("📁 Dossier des modèles")
        model_dir_str = st.text_input(
            "Chemin vers les modèles",
            value=str(DEFAULT_MODEL_DIR),
            help="Dossier contenant les .pth et best_ml_model.pkl",
        )
        model_dir = Path(model_dir_str)

        st.divider()
        st.subheader("🤖 Modèles actifs")
        use_svm = st.checkbox("SVM (RBF)", value=True)
        use_eff = st.checkbox("EfficientNet-B3", value=True)
        use_cnx = st.checkbox("ConvNeXt-Tiny", value=True)
        use_dns = st.checkbox("DenseNet-121", value=True)
        use_res = st.checkbox("ResNet-50", value=True)

        st.divider()
        st.subheader("⚙️ Options")
        show_all_proba = st.checkbox("Afficher toutes les probabilités", value=False)
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)

        st.divider()
        st.caption("Classes critiques cliniques :\n⚠️ erythroblast · ig")

    # ── Titre principal ───────────────────────────────────────────────────────
    st.title("🔬 Classification de Cellules Sanguines")
    st.markdown(
        "**5 modèles** — SVM RBF · EfficientNet-B3 · ConvNeXt-Tiny · DenseNet-121 · ResNet-50 · "
        "Mendeley PBC — 8 classes · Images améliorées (Cellpose + augmentation)"
    )
    st.divider()

    # ── Upload image ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Dépose une image de cellule sanguine",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Image individuelle d'une cellule sanguine",
    )

    if uploaded is None:
        st.info("⬆️ Charge une image pour lancer la classification.")
        st.divider()
        _show_class_reference()
        return

    img = Image.open(uploaded).convert("RGB")

    col_img, col_results = st.columns([1, 2])
    with col_img:
        st.image(img, caption=uploaded.name, use_container_width=True)
        st.caption(f"Taille : {img.size[0]}×{img.size[1]} px")

    # ── Chargement et inférence ───────────────────────────────────────────────
    all_probas: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    active_dl = {
        "EfficientNet-B3": use_eff,
        "ConvNeXt-Tiny":   use_cnx,
        "DenseNet-121":    use_dns,
        "ResNet-50":       use_res,
    }

    with col_results:
        with st.spinner("Inférence en cours..."):

            # SVM
            if use_svm:
                # Cherche le pkl dans model_dir, sinon fallback OneDrive/local
                pkl_path = model_dir / "best_ml_model.pkl"
                if not pkl_path.exists():
                    pkl_path = DEFAULT_ML_PKL
                if pkl_path.exists():
                    try:
                        ml_model, scaler, _ = load_ml_model(str(pkl_path))
                        all_probas["SVM (RBF)"] = predict_ml(ml_model, scaler, img)
                    except Exception as e:
                        errors["SVM (RBF)"] = str(e)
                else:
                    errors["SVM (RBF)"] = f"Fichier introuvable : {pkl_path.name}"

            # DL
            for model_key, active in active_dl.items():
                if not active:
                    continue
                fold = 1
                pth_name = f"best_fold{fold}_{model_key.replace('-', '_').replace(' ', '_')}.pth"
                pth_path = model_dir / pth_name
                if pth_path.exists():
                    try:
                        model, device = load_dl_model(model_key, str(pth_path))
                        cfg = MODELS_CONFIG[model_key]
                        all_probas[model_key] = predict_dl(model, device, img, cfg["input_size"])
                    except Exception as e:
                        errors[model_key] = str(e)
                else:
                    errors[model_key] = f"Fichier introuvable : {pth_name}"

        # ── Résultats ─────────────────────────────────────────────────────────
        if errors:
            for name, err in errors.items():
                st.warning(f"⚠️ {name} : {err}")

        if not all_probas:
            st.error("Aucun modèle n'a pu produire de prédiction.")
            return

        # Ensemble (moyenne des probabilités)
        ensemble_proba = np.mean(list(all_probas.values()), axis=0)
        ensemble_class = CLASSES[np.argmax(ensemble_proba)]
        ensemble_conf  = ensemble_proba.max()

        # Verdict principal
        is_critical = ensemble_class in CRITICAL
        icon = "⚠️" if is_critical else "✅"
        color = "red" if is_critical else "green"
        st.markdown(
            f"### {icon} Prédiction consensus : "
            f"<span style='color:{color};font-weight:bold'>"
            f"{CLASS_EMOJI.get(ensemble_class,'')} {ensemble_class.upper()}</span> "
            f"— {ensemble_conf*100:.1f}%",
            unsafe_allow_html=True,
        )
        if is_critical:
            st.error("⚠️ Classe critique clinique — vérification humaine recommandée.")

        if ensemble_conf < confidence_threshold:
            st.warning(f"⚠️ Confiance faible ({ensemble_conf*100:.1f}% < {confidence_threshold*100:.0f}%)")

        st.divider()

        # Résultats par modèle
        st.subheader("Prédictions par modèle")
        for model_name, proba in all_probas.items():
            pred_idx   = np.argmax(proba)
            pred_class = CLASSES[pred_idx]
            pred_conf  = proba[pred_idx]
            color_hex  = MODEL_COLORS.get(model_name, "#888")

            with st.expander(
                f"**{model_name}** → {CLASS_EMOJI.get(pred_class,'')} **{pred_class}** "
                f"({pred_conf*100:.1f}%)",
                expanded=True,
            ):
                # Barre de progression top-3
                top3_idx = np.argsort(proba)[::-1][:3]
                for i in top3_idx:
                    cls  = CLASSES[i]
                    conf = proba[i]
                    warn = " ⚠️" if cls in CRITICAL else ""
                    st.progress(
                        float(conf),
                        text=f"{CLASS_EMOJI.get(cls,'')} {cls}{warn} — {conf*100:.1f}%",
                    )

                if show_all_proba:
                    st.caption("Toutes les classes :")
                    for i, cls in enumerate(CLASSES):
                        warn = " ⚠️" if cls in CRITICAL else ""
                        st.progress(
                            float(proba[i]),
                            text=f"{CLASS_EMOJI.get(cls,'')} {cls}{warn} — {proba[i]*100:.2f}%",
                        )

        # Tableau récap
        st.divider()
        st.subheader("Tableau récapitulatif")
        rows = []
        for model_name, proba in all_probas.items():
            pred_idx = np.argmax(proba)
            rows.append({
                "Modèle": model_name,
                "Prédiction": CLASSES[pred_idx],
                "Confiance": f"{proba[pred_idx]*100:.1f}%",
                "Critique": "⚠️" if CLASSES[pred_idx] in CRITICAL else "✅",
            })
        rows.append({
            "Modèle": "🏆 CONSENSUS",
            "Prédiction": ensemble_class,
            "Confiance": f"{ensemble_conf*100:.1f}%",
            "Critique": "⚠️" if ensemble_class in CRITICAL else "✅",
        })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _show_class_reference() -> None:
    st.subheader("📋 Référence des 8 classes")
    cols = st.columns(4)
    descriptions = {
        "basophil":     "Basophile — rare, granules foncés",
        "eosinophil":   "Éosinophile — granules orangés",
        "erythroblast": "Érythroblaste ⚠️ — précurseur GR, critical",
        "ig":           "IG ⚠️ — granulocytes immatures, critical",
        "lymphocyte":   "Lymphocyte — petit noyau rond",
        "monocyte":     "Monocyte — grand noyau en fer à cheval",
        "neutrophil":   "Neutrophile — noyau multilobé",
        "platelet":     "Plaquette — très petite, sans noyau",
    }
    for i, (cls, desc) in enumerate(descriptions.items()):
        with cols[i % 4]:
            st.metric(
                label=f"{CLASS_EMOJI.get(cls,'')} {cls}",
                value="⚠️ critique" if cls in CRITICAL else "normale",
            )
            st.caption(desc)


if __name__ == "__main__":
    main()
