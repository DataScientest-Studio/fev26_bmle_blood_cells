#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Streamlit : Classification de cellules sanguines
Appelle l'API FastAPI (DenseNet-121 uniquement)
"""

import os
import streamlit as st
from PIL import Image
import requests
import io

API_URL = os.getenv("API_URL", "http://api:8000")

CLASSES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
    "Promyelocyte"
]

CRITICAL = {"Erythroblast"}

CLASS_EMOJI = {
    "Basophil": "B",
    "Eosinophil": "E",
    "Erythroblast": "R",
    "Lymphocyte": "L",
    "Monocyte": "M",
    "Neutrophil": "N",
    "Platelet": "P",
    "Promyelocyte": "Pr",
}


def predict_with_api(image: Image.Image) -> dict:
    """Appelle l'API FastAPI pour prédire la classe d'une image."""
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.png", img_bytes, "image/png")},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "message": response.text}
    
    except Exception as e:
        return {"error": str(e), "message": "Failed to call API"}


def show_class_reference() -> None:
    """Affiche la reference des 8 classes."""
    st.subheader("Reference des 8 classes")
    cols = st.columns(4)
    descriptions = {
        "Basophil": "Basophile — rare, granules fonces",
        "Eosinophil": "Eosinophile — granules oranges",
        "Erythroblast": "Erythroblaste [CRITICAL] — precurseur GR",
        "Lymphocyte": "Lymphocyte — petit noyau rond",
        "Monocyte": "Monocyte — grand noyau en fer a cheval",
        "Neutrophil": "Neutrophile — noyau multilobes",
        "Platelet": "Plaquette — tres petite, sans noyau",
        "Promyelocyte": "Promyelocyte — stade intermediaire",
    }
    for i, (cls, desc) in enumerate(descriptions.items()):
        with cols[i % 4]:
            status = "[CRITICAL]" if cls in CRITICAL else "normal"
            st.metric(label=f"{CLASS_EMOJI.get(cls, '')} {cls}", value=status)
            st.caption(desc)


def main() -> None:
    """Fonction principale Streamlit."""
    st.set_page_config(
        page_title="Blood Cell Classifier",
        page_icon="microscope",
        layout="wide",
    )

    st.title("Classification de Cellules Sanguines")
    st.markdown("**DenseNet-121** — Mendeley PBC — 8 classes")
    st.divider()

    uploaded = st.file_uploader(
        "Depose une image de cellule sanguine",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Image individuelle d'une cellule sanguine",
    )

    if uploaded is None:
        st.info("Charge une image pour lancer la classification.")
        st.divider()
        show_class_reference()
        return

    img = Image.open(uploaded).convert("RGB")

    col_img, col_results = st.columns([1, 2])
    
    with col_img:
        st.image(img, caption=uploaded.name, use_container_width=True)
        st.caption(f"Taille : {img.size[0]}x{img.size[1]} px")

    with col_results:
        with st.spinner("Classification en cours..."):
            result = predict_with_api(img)

        if "error" in result:
            st.error(f"Erreur : {result.get('message', result['error'])}")
            return

        pred_class = result.get("class", "Unknown")
        confidence = result.get("confidence", 0.0)
        all_probas = result.get("all_probas", {})

        is_critical = pred_class in CRITICAL
        icon = "WARN" if is_critical else "OK"
        color = "red" if is_critical else "green"
        
        st.markdown(
            f"### [{icon}] Prediction : "
            f"<span style='color:{color};font-weight:bold'>"
            f"{CLASS_EMOJI.get(pred_class, '')} {pred_class.upper()}</span> "
            f"— {confidence*100:.1f}%",
            unsafe_allow_html=True,
        )
        
        if is_critical:
            st.error("Classe critique clinique — verification humaine recommandee.")

        st.divider()

        st.subheader("Top 3 predictions")
        sorted_probs = sorted(all_probas.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs[:3]:
            warn = " [CRITICAL]" if cls in CRITICAL else ""
            st.progress(
                float(prob),
                text=f"{CLASS_EMOJI.get(cls, '')} {cls}{warn} — {prob*100:.1f}%",
            )

        with st.expander("Voir toutes les classes"):
            st.subheader("Toutes les probabilites")
            for cls, prob in sorted(all_probas.items(), key=lambda x: x[1], reverse=True):
                warn = " [CRITICAL]" if cls in CRITICAL else ""
                st.progress(
                    float(prob),
                    text=f"{CLASS_EMOJI.get(cls, '')} {cls}{warn} — {prob*100:.2f}%",
                )


if __name__ == "__main__":
    main()