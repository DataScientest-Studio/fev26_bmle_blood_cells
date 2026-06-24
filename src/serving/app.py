#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Streamlit : Classification de cellules sanguines
Appelle l'API FastAPI (DenseNet-121 uniquement)
"""

import os
import sys
import io
from pathlib import Path

import mlflow
import pandas as pd
import psycopg2
import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.auth.users import verify_user  # noqa: E402

API_URL = os.getenv("API_URL", "http://api:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_MODEL_NAME = "blood-cell-densenet121"

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

CRITICAL = {"Erythroblast"}

CLASS_EMOJI = {
    "Basophil": "B",
    "Eosinophil": "E",
    "Erythroblast": "R",
    "IG": "IG",
    "Lymphocyte": "L",
    "Monocyte": "M",
    "Neutrophil": "N",
    "Platelet": "P",
}


def _api_headers() -> dict:
    if API_SECRET_KEY:
        return {"X-API-Key": API_SECRET_KEY}
    return {}


def predict_with_api(image: Image.Image) -> dict:
    """Appelle l'API FastAPI pour prédire la classe d'une image."""
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.png", img_bytes, "image/png")},
            headers=_api_headers(),
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "message": response.text}

    except Exception as e:
        return {"error": str(e), "message": "Failed to call API"}


def send_feedback(prediction_id: int, agrees: bool, corrected_class: str = None, comment: str = None) -> dict:
    """Envoie l'avis du médecin (accord/désaccord) sur une prédiction à l'API."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "prediction_id": prediction_id,
                "agrees": agrees,
                "corrected_class": corrected_class,
                "comment": comment,
            },
            headers=_api_headers(),
            timeout=15,
        )
        if response.status_code == 200:
            return {"ok": True}
        return {"ok": False, "message": response.text}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def show_class_reference() -> None:
    """Affiche la reference des 8 classes."""
    st.subheader("Reference des 8 classes")
    cols = st.columns(4)
    descriptions = {
        "Basophil": "Basophile — rare, granules fonces",
        "Eosinophil": "Eosinophile — granules oranges",
        "Erythroblast": "Erythroblaste [CRITICAL] — precurseur GR",
        "IG": "Granulocyte immature — precurseur immature",
        "Lymphocyte": "Lymphocyte — petit noyau rond",
        "Monocyte": "Monocyte — grand noyau en fer a cheval",
        "Neutrophil": "Neutrophile — noyau multilobes",
        "Platelet": "Plaquette — tres petite, sans noyau",
    }
    for i, (cls, desc) in enumerate(descriptions.items()):
        with cols[i % 4]:
            status = "[CRITICAL]" if cls in CRITICAL else "normal"
            st.metric(label=f"{CLASS_EMOJI.get(cls, '')} {cls}", value=status)
            st.caption(desc)


@st.cache_data(ttl=60)
def fetch_training_runs() -> pd.DataFrame:
    """Lit tous les runs d'entrainement loggues dans Supabase (training_runs)."""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"), port=int(os.getenv("SUPABASE_PORT", 5432)),
        dbname=os.getenv("SUPABASE_DB"), user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"), connect_timeout=10, sslmode="require",
    )
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT mlflow_run_id, model_name, generation, fold, device, status, started_at,
                   duration_seconds, cpu_percent_avg, ram_used_mb_avg,
                   gpu_name, gpu_util_percent_avg, gpu_mem_used_mb_avg
            FROM training_runs ORDER BY started_at DESC LIMIT 500
        """)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


@st.cache_data(ttl=60)
def fetch_mlflow_metrics() -> dict:
    """Recupere macro_f1/accuracy pour tous les runs connus de MLflow, en une
    seule recherche par experience (plutot qu'un get_run par ligne)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    metrics_by_run = {}
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000):
            metrics_by_run[run.info.run_id] = {
                "macro_f1": run.data.metrics.get("macro_f1"),
                "accuracy": run.data.metrics.get("accuracy"),
            }
    return metrics_by_run


@st.cache_data(ttl=60)
def fetch_production_version() -> dict | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        run = client.get_run(mv.run_id)
        return {
            "version": mv.version,
            "generation": mv.tags.get("generation"),
            "macro_f1": run.data.metrics.get("macro_f1"),
        }
    except Exception:
        return None


def show_logs_tab() -> None:
    """Onglet Logs : tous les runs d'entrainement et leurs stats (GPU, temps, accuracy...)."""
    st.subheader("Historique des entrainements")

    if st.button("Rafraichir"):
        fetch_training_runs.clear()
        fetch_mlflow_metrics.clear()
        fetch_production_version.clear()

    try:
        runs = fetch_training_runs()
    except Exception as e:
        st.error(f"Supabase indisponible : {e}")
        return

    if runs.empty:
        st.info("Aucun run d'entrainement logge pour le moment.")
        return

    prod = fetch_production_version()
    metrics_by_run = fetch_mlflow_metrics()
    runs["macro_f1"] = runs["mlflow_run_id"].map(lambda r: metrics_by_run.get(r, {}).get("macro_f1"))
    runs["accuracy"] = runs["mlflow_run_id"].map(lambda r: metrics_by_run.get(r, {}).get("accuracy"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs loggues", len(runs))
    col2.metric("Taux de succes", f"{(runs['status'] == 'success').mean() * 100:.0f}%")
    col3.metric("Duree moyenne", f"{runs['duration_seconds'].mean():.0f}s")
    if prod:
        col4.metric(
            "@production",
            f"v{prod['version']} ({prod['generation']})",
            f"macro_f1={prod['macro_f1']:.4f}" if prod["macro_f1"] is not None else None,
        )
    else:
        col4.metric("@production", "—")

    st.divider()

    generations = ["Toutes"] + sorted(runs["generation"].dropna().unique().tolist())
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        gen_filter = st.selectbox("Generation", generations)
    with col_f2:
        status_filter = st.selectbox("Statut", ["Tous", "success", "failed"])

    filtered = runs
    if gen_filter != "Toutes":
        filtered = filtered[filtered["generation"] == gen_filter]
    if status_filter != "Tous":
        filtered = filtered[filtered["status"] == status_filter]

    st.dataframe(
        filtered,
        column_config={
            "mlflow_run_id": "Run ID",
            "model_name": "Modele",
            "generation": "Generation",
            "fold": "Fold",
            "device": "Device",
            "status": "Statut",
            "started_at": st.column_config.DatetimeColumn("Demarre le"),
            "duration_seconds": st.column_config.NumberColumn("Duree (s)", format="%.0f"),
            "cpu_percent_avg": st.column_config.NumberColumn("CPU %", format="%.1f"),
            "ram_used_mb_avg": st.column_config.NumberColumn("RAM (Mo)", format="%.0f"),
            "gpu_name": "GPU",
            "gpu_util_percent_avg": st.column_config.NumberColumn("GPU %", format="%.1f"),
            "gpu_mem_used_mb_avg": st.column_config.NumberColumn("GPU mem (Mo)", format="%.0f"),
            "macro_f1": st.column_config.NumberColumn("macro_f1", format="%.4f"),
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.4f"),
        },
        use_container_width=True,
        hide_index=True,
    )


def login_screen() -> bool:
    """Affiche le formulaire de connexion. Retourne True si l'utilisateur est authentifie."""
    if st.session_state.get("authenticated"):
        return True

    st.title("Connexion")
    with st.form("login_form"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        if verify_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Identifiant ou mot de passe incorrect.")

    return False


def show_classification_tab() -> None:
    """Onglet Classification : upload d'image, prediction, avis du medecin."""
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

        prediction_id = result.get("prediction_id")
        pred_class = result.get("predicted_class", "Unknown")
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

        st.divider()
        st.subheader("Avis du medecin")

        feedback_done_key = f"feedback_done_{prediction_id}"
        if st.session_state.get(feedback_done_key):
            st.success("Merci, votre avis a ete enregistre.")
        elif prediction_id is None:
            st.caption("Avis indisponible (prediction non enregistree — Supabase hors ligne ?).")
        else:
            agrees_label = st.radio(
                "Etes-vous d'accord avec cette prediction ?",
                ["Oui", "Non"],
                horizontal=True,
                index=None,
                key=f"agrees_{prediction_id}",
            )
            corrected_class = None
            if agrees_label == "Non":
                corrected_class = st.selectbox(
                    "Quelle est la classe correcte selon vous ?",
                    CLASSES,
                    key=f"corrected_{prediction_id}",
                )
            comment = st.text_area(
                "Commentaire (optionnel)", key=f"comment_{prediction_id}",
            )

            if st.button("Envoyer mon avis", key=f"submit_{prediction_id}"):
                if agrees_label is None:
                    st.warning("Precise si tu es d'accord ou non avant d'envoyer.")
                else:
                    res = send_feedback(
                        prediction_id,
                        agrees=(agrees_label == "Oui"),
                        corrected_class=corrected_class,
                        comment=comment or None,
                    )
                    if res["ok"]:
                        st.session_state[feedback_done_key] = True
                        st.rerun()
                    else:
                        st.error(f"Erreur lors de l'envoi : {res['message']}")


def main() -> None:
    """Fonction principale Streamlit."""
    st.set_page_config(
        page_title="Blood Cell Classifier",
        page_icon="microscope",
        layout="wide",
    )

    if not login_screen():
        return

    with st.sidebar:
        st.caption(f"Connecte : {st.session_state['username']}")
        if st.button("Se deconnecter"):
            st.session_state.clear()
            st.rerun()

    tab_classify, tab_logs = st.tabs(["Classification", "Logs"])
    with tab_classify:
        show_classification_tab()
    with tab_logs:
        show_logs_tab()


if __name__ == "__main__":
    main()
