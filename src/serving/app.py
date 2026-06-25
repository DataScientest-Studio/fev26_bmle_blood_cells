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
load_dotenv(ROOT / ".env", override=True)

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
        host=os.getenv("SUPABASE_HOST"), port=int(os.getenv("SUPABASE_PORT", 6543)),
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


_DRIFT_LEVELS = {
    "normal":   ("✅ Normal",   "green"),
    "warning":  ("⚠️ Warning",  "orange"),
    "alerte":   ("🔴 Alerte",   "red"),
    "critique": ("🚨 Critique", "red"),
    "unknown":  ("— Inconnu",   "gray"),
}


def _level_badge(level: str) -> str:
    label, color = _DRIFT_LEVELS.get(level, ("—", "gray"))
    return f"<span style='color:{color};font-weight:bold'>{label}</span>"


def show_drift_tab() -> None:
    """Onglet Drift : rapports Evidently de data drift et model drift."""
    st.subheader("Monitoring du drift (IVDR 2017/746)")
    st.caption(
        "Seuils d'alerte : warning > 0.10 | alerte > 0.20 | critique > 0.30"
    )

    col_gen, col_ver = st.columns([2, 1])
    with col_ver:
        model_version_input = st.text_input(
            "Version modele (vide = toutes)", value="", key="drift_model_version"
        )
    with col_gen:
        generate = st.button("Generer le rapport de drift", type="primary")

    if generate:
        with st.spinner("Generation du rapport Evidently en cours..."):
            try:
                from src.evidently.drift_report import generate_report
                result = generate_report(
                    model_version=model_version_input.strip() or None
                )
                if "error" in result:
                    st.error(result["error"])
                    return
                st.session_state["last_drift_result"] = result
                st.success(f"Rapport genere (id={result['report_id']}) et sauvegarde dans Supabase.")
            except Exception as e:
                st.error(f"Erreur : {e}")
                return

    # Charger le dernier rapport (genere ou depuis Supabase)
    result = st.session_state.get("last_drift_result")
    if result is None:
        try:
            from src.evidently.drift_report import load_last_report
            result = load_last_report()
        except Exception:
            result = None

    if result is None:
        st.info("Aucun rapport disponible. Clique sur 'Generer le rapport de drift'.")
        return

    st.divider()

    # ── Metriques cles ────────────────────────────────────────────────────────
    st.subheader("Resume")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reference", f"{result.get('n_reference', 0)} imgs")
    c2.metric("Production", f"{result.get('n_current', 0)} predictions")
    c3.metric("Features driftees", result.get("n_drifted_features", 0))

    data_level = result.get("data_drift_level", "unknown")
    pred_level = result.get("pred_drift_level", "unknown")
    c4.metric(
        "Data drift",
        f"{result.get('data_drift_score', 0):.3f}",
        delta=_DRIFT_LEVELS[data_level][0],
        delta_color="off",
    )
    c5.metric(
        "Prediction drift",
        f"{result.get('pred_drift_score', 0):.3f}",
        delta=_DRIFT_LEVELS[pred_level][0],
        delta_color="off",
    )

    # Model drift (feedback medecin)
    model_score = result.get("model_drift_score")
    metrics = result.get("metrics", {})
    model_metrics = metrics.get("model_drift", {})
    if model_score is not None:
        st.divider()
        st.subheader("Model drift (feedback medecin)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Feedbacks recus", model_metrics.get("n_feedback", 0))
        mc2.metric("Taux accord medecin", f"{model_metrics.get('accuracy', 0)*100:.1f}%")
        mc3.metric(
            "Taux desaccord",
            f"{model_metrics.get('disagree_rate', 0)*100:.1f}%",
            delta="Alerte si > 10%",
            delta_color="off",
        )
    else:
        st.info("Model drift : aucun feedback medecin enregistre pour l'instant.")

    # ── Alertes actives ───────────────────────────────────────────────────────
    alerts = []
    if result.get("data_drift_score", 0) >= 0.30:
        alerts.append("CRITIQUE — Data drift score > 0.30 : investigation obligatoire.")
    elif result.get("data_drift_score", 0) >= 0.20:
        alerts.append("ALERTE — Data drift score > 0.20 : surveillance renforcee.")
    elif result.get("data_drift_score", 0) >= 0.10:
        alerts.append("WARNING — Data drift score > 0.10 : surveiller l'evolution.")

    if result.get("pred_drift_score", 0) >= 0.20:
        alerts.append("ALERTE — Distribution des classes predites a derive.")

    if model_score is not None and model_score >= 0.15:
        alerts.append("ALERTE — Taux de desaccord medecin > 15%.")
    elif model_score is not None and model_score >= 0.10:
        alerts.append("WARNING — Taux de desaccord medecin > 10%.")

    if alerts:
        st.divider()
        for alert in alerts:
            if alert.startswith("CRITIQUE"):
                st.error(alert)
            elif alert.startswith("ALERTE"):
                st.warning(alert)
            else:
                st.info(alert)

    # ── Rapport Evidently complet ─────────────────────────────────────────────
    if result.get("report_html"):
        st.divider()
        st.subheader("Rapport Evidently complet")
        if result.get("created_at"):
            st.caption(f"Genere le : {result['created_at']} | Version modele : {result.get('model_version', 'toutes')}")
        st.components.v1.html(result["report_html"], height=900, scrolling=True)

    # ── Performance du modele (MLflow + Supabase class_metrics) ──────────────
    st.divider()
    st.subheader("Performance du modele (MLflow)")
    st.caption("Evolution de macro_F1 et accuracy par generation — seuil d'alerte IVDR : baisse > 5%")

    load_perf = st.button("Charger les metriques de performance", key="load_perf")
    if load_perf or st.session_state.get("perf_data"):
        if load_perf:
            with st.spinner("Chargement des metriques MLflow + Supabase..."):
                try:
                    from src.evidently.drift_report import load_performance_metrics
                    perf = load_performance_metrics()
                    st.session_state["perf_data"] = perf
                except Exception as e:
                    st.error(f"Impossible de charger les metriques : {e}")
                    perf = None
        else:
            perf = st.session_state.get("perf_data")

        if perf:
            df_global  = perf["df_global"]
            df_classes = perf["df_classes"]
            perf_alerts = perf.get("alerts", [])
            current    = perf.get("current", {})
            baseline   = perf.get("baseline", {})

            # Metriques globales courantes
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Generations evaluees", perf.get("n_generations", 0))
            if current.get("macro_f1") is not None:
                delta_f1 = None
                if baseline.get("macro_f1") is not None and baseline.get("generation") != current.get("generation"):
                    delta_f1 = f"{(current['macro_f1'] - baseline['macro_f1'])*100:+.2f}% vs baseline"
                pc2.metric("macro_F1 (derniere gen)", f"{current['macro_f1']:.4f}", delta=delta_f1)
            if current.get("accuracy") is not None:
                delta_acc = None
                if baseline.get("accuracy") is not None and baseline.get("generation") != current.get("generation"):
                    delta_acc = f"{(current['accuracy'] - baseline['accuracy'])*100:+.2f}% vs baseline"
                pc3.metric("Accuracy (derniere gen)", f"{current['accuracy']:.4f}", delta=delta_acc)
            pc4.metric("Generation actuelle", current.get("generation", "N/A"))

            # Alertes performance
            if perf_alerts:
                st.divider()
                for alert in perf_alerts:
                    if "CRITIQUE" in alert:
                        st.error(alert)
                    elif "ALERTE" in alert:
                        st.warning(alert)
                    else:
                        st.info(alert)

            # Evolution macro_F1 et accuracy par generation
            if not df_global.empty and len(df_global) > 1:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_global["generation"].astype(str),
                    y=df_global["macro_f1"],
                    mode="lines+markers",
                    name="macro_F1",
                    line=dict(color="#1f77b4"),
                ))
                if "accuracy" in df_global.columns and df_global["accuracy"].notna().any():
                    fig.add_trace(go.Scatter(
                        x=df_global["generation"].astype(str),
                        y=df_global["accuracy"],
                        mode="lines+markers",
                        name="Accuracy",
                        line=dict(color="#2ca02c", dash="dash"),
                    ))
                fig.add_hline(
                    y=(df_global["macro_f1"].iloc[0] - 0.05),
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Seuil alerte (-5%)",
                )
                fig.update_layout(
                    title="Evolution des metriques globales par generation",
                    xaxis_title="Generation",
                    yaxis_title="Score",
                    yaxis_range=[0, 1],
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Metriques par classe — focus Erythroblast et IG
            if not df_classes.empty:
                st.divider()
                st.markdown("**Metriques par classe (F1 — focus classes critiques)**")
                critical_cls = ["Erythroblast", "IG"]
                tabs_cls = st.tabs(critical_cls + ["Toutes les classes"])
                for i, cls in enumerate(critical_cls):
                    with tabs_cls[i]:
                        cls_data = df_classes[
                            df_classes["class_name"].str.lower() == cls.lower()
                        ].sort_values("generation")
                        if cls_data.empty:
                            st.info(f"Aucune donnee pour {cls}.")
                        elif len(cls_data) == 1:
                            row = cls_data.iloc[0]
                            st.metric(f"F1 {cls}", f"{row['f1']:.4f}")
                            cc1, cc2, cc3 = st.columns(3)
                            cc1.metric("Precision", f"{row['precision']:.4f}")
                            cc2.metric("Recall", f"{row['recall']:.4f}")
                            cc3.metric("Support", int(row["support"]))
                        else:
                            import plotly.graph_objects as go
                            fig2 = go.Figure()
                            for metric, color in [("f1","#1f77b4"), ("precision","#ff7f0e"), ("recall","#2ca02c")]:
                                fig2.add_trace(go.Scatter(
                                    x=cls_data["generation"].astype(str),
                                    y=cls_data[metric],
                                    mode="lines+markers",
                                    name=metric.capitalize(),
                                    line=dict(color=color),
                                ))
                            fig2.update_layout(
                                title=f"Evolution {cls}",
                                xaxis_title="Generation",
                                yaxis_range=[0, 1],
                                height=300,
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                with tabs_cls[-1]:
                    # Tableau toutes classes — derniere generation
                    if not df_classes.empty:
                        last_gen = df_classes["generation"].max()
                        df_last = df_classes[df_classes["generation"] == last_gen][
                            ["class_name", "precision", "recall", "f1", "support"]
                        ].sort_values("f1", ascending=False)
                        df_last = df_last.rename(columns={
                            "class_name": "Classe",
                            "precision": "Precision",
                            "recall": "Recall",
                            "f1": "F1",
                            "support": "Support",
                        })
                        st.caption(f"Generation {last_gen}")
                        st.dataframe(
                            df_last.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                            use_container_width=True,
                        )


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

    tab_classify, tab_logs, tab_drift = st.tabs(["Classification", "Logs", "Drift"])
    with tab_classify:
        show_classification_tab()
    with tab_logs:
        show_logs_tab()
    with tab_drift:
        show_drift_tab()


if __name__ == "__main__":
    main()
