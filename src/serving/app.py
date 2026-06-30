#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Streamlit : Analyse de frottis sanguin (DenseNet-121)
"""

import base64
import io
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from src.auth.users import verify_user  # noqa: E402

API_URL = os.getenv("API_URL", "http://api:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_MODEL_NAME = "blood-cell-densenet121"

CLASSES = [
    "Basophil", "Eosinophil", "Erythroblast", "IG",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet",
]
CRITICAL = {"Erythroblast", "IG"}

CLASS_COLORS = {
    "Basophil":    "#6366F1",
    "Eosinophil":  "#F59E0B",
    "Erythroblast": "#EF4444",
    "IG":          "#F97316",
    "Lymphocyte":  "#10B981",
    "Monocyte":    "#3B82F6",
    "Neutrophil":  "#8B5CF6",
    "Platelet":    "#EC4899",
}

CLASS_ABBR = {
    "Basophil": "BAS", "Eosinophil": "EOS", "Erythroblast": "ERY",
    "IG": "IG", "Lymphocyte": "LYM", "Monocyte": "MON",
    "Neutrophil": "NEU", "Platelet": "PLT",
}

GRID_COLS = 5


# ── CSS ───────────────────────────────────────────────────────────────────────

def _apply_css() -> None:
    st.markdown("""
<style>
/* ── Masquer la barre Streamlit native ── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar toujours visible — impossible à fermer ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    display: block !important;
    transform: none !important;
    visibility: visible !important;
    min-width: 244px !important;
    width: 244px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: none !important;
    display: block !important;
}

/* ── Global ── */
.stApp { background-color: #F0F4F8; }
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; }

/* ── Texte noir sur fond clair (contenu principal) ── */
[data-testid="stMain"] p,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] li,
[data-testid="stMain"] td,
[data-testid="stMain"] th,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] *,
[data-testid="stMain"] [data-testid="stMetricValue"],
[data-testid="stMain"] [data-testid="stMetricLabel"],
[data-testid="stMain"] [data-testid="stMetricDelta"],
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
[data-testid="stMain"] [data-testid="stText"],
[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] .stMarkdown *,
[data-testid="stMain"] .stCaption,
[data-testid="stMain"] .stCaption *,
[data-testid="stMain"] [class*="caption"],
[data-testid="stMain"] [class*="Caption"] {
    color: #0F172A !important;
}
/* Préserver les couleurs des boutons, badges, et texte blanc sur fond coloré */
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stButton > button[kind="primary"] * {
    color: white !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #1A2B4A 0%, #0D1B2E 100%);
    border-right: 1px solid #2D3F5E;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #CBD5E0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

/* ── Nav radio — fond bleu clair ── */
[data-testid="stSidebar"] [data-baseweb="radio-group"] {
    background: #2563EB;
    border-radius: 10px;
    padding: 5px;
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] {
    padding: 0 !important;
    border-radius: 7px !important;
    margin: 2px 0 !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has([aria-checked="true"]) {
    background: rgba(255,255,255,0.25) !important;
}
/* Cache le cercle radio */
[data-testid="stSidebar"] [role="radio"] { display: none !important; }
/* Label pleine largeur = zone cliquable = toute la ligne */
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    display: block !important;
    width: 100% !important;
    padding: 9px 14px !important;
    color: rgba(255,255,255,0.75) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    line-height: 1.3 !important;
    box-sizing: border-box !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has([aria-checked="true"]) label {
    color: #FFFFFF !important;
}

/* ── Bouton déconnexion — rouge ── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(220,38,38,0.15) !important;
    color: #FCA5A5 !important;
    border: 1px solid rgba(220,38,38,0.35) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(220,38,38,0.28) !important;
    color: #FFFFFF !important;
}

/* ── Buttons main ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.15s;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.4);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    border: 1px solid #E2E8F0;
}

/* ── Expanders — header blanc sur fond sombre ── */
[data-testid="stExpander"] details {
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details summary {
    background: #1A2B4A !important;
    padding: 12px 16px !important;
}
/* Cibler le <p> spécifiquement — surcharge le sélecteur global stMain p */
[data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
/* Contenu de l'expander — fond blanc, texte noir */
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: white !important;
}
</style>
    """, unsafe_allow_html=True)


# ── Helpers DagsHub / DVC ─────────────────────────────────────────────────────

LOTSTIFF_DIR = ROOT / "data" / "lotstiff"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _list_batches() -> list[str]:
    """Batches disponibles (fichiers .dvc dans data/lotstiff/), triés numériquement."""
    return sorted(
        [p.stem for p in LOTSTIFF_DIR.glob("batch*.dvc")],
        key=lambda b: int(b[len("batch"):]),
    )


def _pull_batch(batch_name: str) -> tuple[bool, str]:
    """DVC pull d'un batch. Retourne (succès, message)."""
    dagshub_user = os.getenv("DAGSHUB_USER", "")
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    dvc_file = LOTSTIFF_DIR / f"{batch_name}.dvc"
    for cmd in [
        ["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"],
        ["dvc", "remote", "modify", "dagshub", "--local", "user", dagshub_user],
        ["dvc", "remote", "modify", "dagshub", "--local", "password", dagshub_token],
    ]:
        subprocess.run(cmd, capture_output=True, cwd=ROOT)
    result = subprocess.run(
        ["dvc", "pull", str(dvc_file)],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        return False, result.stderr or result.stdout
    return True, f"{batch_name} synchronisé."


def _list_batch_images(batch_name: str) -> list[Path]:
    """Liste toutes les images d'un batch (avec sous-dossiers par classe)."""
    batch_dir = LOTSTIFF_DIR / batch_name
    if not batch_dir.is_dir():
        return []
    return sorted(
        [p for p in batch_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    )


# ── Helpers image ─────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_headers() -> dict:
    if API_SECRET_KEY:
        return {"X-API-Key": API_SECRET_KEY}
    return {}


def gradcam_predict(
    image: Image.Image, filename: str = "cell.png",
    patient_id: int = None, patient_name: str = None, triggered_by: str = None,
) -> dict:
    """Appelle /gradcam : retourne prédiction + GradCAM base64."""
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        data = {}
        if patient_id is not None:
            data["patient_id"] = patient_id
        if patient_name:
            data["patient_name"] = patient_name
        if triggered_by:
            data["triggered_by"] = triggered_by
        resp = requests.post(
            f"{API_URL}/gradcam",
            files={"file": (filename, buf, "image/png")},
            data=data or None,
            headers=_api_headers(),
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}", "message": resp.text}
    except Exception as e:
        return {"error": str(e), "message": "Connexion API échouée"}


def send_feedback(
    prediction_id: int,
    agrees: bool,
    corrected_class: str = None,
    comment: str = None,
) -> dict:
    try:
        resp = requests.post(
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
        return {"ok": resp.status_code == 200, "message": resp.text}
    except Exception as e:
        return {"ok": False, "message": str(e)}


# ── MLflow / Supabase ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_training_runs() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        port=int(os.getenv("SUPABASE_PORT", 6543)),
        dbname=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        connect_timeout=10,
        sslmode="require",
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
def fetch_mlflow_run_data() -> dict:
    """Métriques + tag git_commit + version Registry pour tous les runs MLflow.

    La "version" MLflow (compteur global, incrémenté à chaque enregistrement
    dans le Registry) est différente de la "génération" (tag métier propre
    au projet) — un même modèle a les deux, ne pas les confondre."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    data = {}
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000):
            data[run.info.run_id] = {
                "macro_f1":   run.data.metrics.get("macro_f1"),
                "accuracy":   run.data.metrics.get("accuracy"),
                "git_commit": run.data.tags.get("git_commit", ""),
            }
    for mv in client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'"):
        if mv.run_id in data:
            data[mv.run_id]["mlflow_version"] = mv.version
    return data


@st.cache_data(ttl=60)
def fetch_production_version() -> dict | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        run = client.get_run(mv.run_id)
        return {
            "version":    mv.version,
            "generation": mv.tags.get("generation"),
            "macro_f1":   run.data.metrics.get("macro_f1"),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_class_confidence(window_days: int = 180) -> pd.DataFrame:
    """Part des predictions de chaque classe corrigees par un medecin
    (table prediction_feedback, agrees=False), sur les derniers window_days jours."""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"), port=int(os.getenv("SUPABASE_PORT", 6543)),
        dbname=os.getenv("SUPABASE_DB"), user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"), connect_timeout=10, sslmode="require",
    )
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.predicted_class,
                   COUNT(*) AS total_predictions,
                   COUNT(*) FILTER (WHERE pf.agrees = FALSE) AS corrections
            FROM predictions p
            LEFT JOIN prediction_feedback pf ON pf.prediction_id = p.id
            WHERE p.created_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY p.predicted_class
        """, (window_days,))
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()
    if not df.empty:
        df["confidence_pct"] = 100 * (1 - df["corrections"] / df["total_predictions"])
    return df


def next_patient_id() -> int:
    """Numero de patient suivant — simule un nouveau patient par lot d'images analyse
    (pas de vrais patients : un lot = un frottis = un patient)."""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"), port=int(os.getenv("SUPABASE_PORT", 6543)),
        dbname=os.getenv("SUPABASE_DB"), user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"), connect_timeout=10, sslmode="require",
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(patient_id), 0) + 1 FROM predictions")
        return cur.fetchone()[0]
    finally:
        conn.close()


def search_predictions(image_name: str = "", patient_name: str = "") -> pd.DataFrame:
    """Recherche dans les predictions par nom d'image (partiel) et/ou par nom de patient (partiel)."""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"), port=int(os.getenv("SUPABASE_PORT", 6543)),
        dbname=os.getenv("SUPABASE_DB"), user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"), connect_timeout=10, sslmode="require",
    )
    try:
        cur = conn.cursor()
        clauses, params = [], []
        if image_name:
            clauses.append("image_name ILIKE %s")
            params.append(f"%{image_name}%")
        if patient_name:
            clauses.append("patient_name ILIKE %s")
            params.append(f"%{patient_name}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(f"""
            SELECT id, patient_id, patient_name, image_name, predicted_class, confidence,
                   model_version, triggered_by, created_at
            FROM predictions
            {where}
            ORDER BY created_at DESC
            LIMIT 200
        """, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


# ── Login ─────────────────────────────────────────────────────────────────────

def login_screen() -> bool:
    if st.session_state.get("authenticated"):
        return True

    col_center = st.columns([1, 1.2, 1])[1]
    with col_center:
        st.markdown("""
        <div style="background:white;border-radius:16px;padding:40px 36px;
                    box-shadow:0 4px 24px rgba(0,0,0,0.08);margin-top:60px;">
            <div style="text-align:center;margin-bottom:28px;">
                <div style="font-size:2.8rem;">🔬</div>
                <h2 style="color:#1E293B;margin:8px 0 4px;font-size:1.5rem;">
                    Blood Cell Analyzer
                </h2>
                <p style="color:#64748B;font-size:0.9rem;margin:0;">
                    Accès réservé au personnel médical
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Identifiant", placeholder="Votre identifiant")
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submitted = st.form_submit_button(
                "Se connecter", use_container_width=True, type="primary"
            )
        if submitted:
            username = username.strip()
            try:
                ok = verify_user(username, password)
            except Exception:
                # Supabase injoignable — fallback dev local (DEV_MODE=1 dans .env)
                if os.getenv("DEV_MODE") == "1":
                    ok = bool(username and password)
                else:
                    st.error("Base d'authentification indisponible. Activez DEV_MODE=1 pour le mode local.")
                    return False
            if ok:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")
    return False


# ── Classification tab ────────────────────────────────────────────────────────

def _class_badge_html(cls: str, conf: float) -> str:
    color = CLASS_COLORS.get(cls, "#6B7280")
    abbr = CLASS_ABBR.get(cls, cls[:3].upper())
    crit = " ⚠" if cls in CRITICAL else ""
    return (
        f'<div style="text-align:center;margin-top:5px;">'
        f'<span style="background:{color};color:white;padding:2px 7px;'
        f'border-radius:4px;font-size:0.68rem;font-weight:700;">{abbr}{crit}</span>'
        f'<div style="font-size:0.7rem;color:#64748B;margin-top:2px;">{conf*100:.0f}%</div>'
        f'</div>'
    )


def _show_cell_detail(res: dict, idx: int) -> None:
    pred_class = res.get("predicted_class", "?")
    confidence = res.get("confidence", 0.0)
    is_critical = res.get("is_critical", False)
    all_probas = res.get("all_probas", {})
    prediction_id = res.get("prediction_id")
    color = CLASS_COLORS.get(pred_class, "#6B7280")

    st.markdown(
        f'<div style="background:white;border-radius:12px;padding:20px 24px;'
        f'border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
        f'<h4 style="margin:0 0 4px;color:#1E293B;">Cellule n°{idx + 1}'
        f'<span style="font-weight:400;color:#94A3B8;font-size:0.85rem;margin-left:10px;">'
        f'{res.get("filename", "")}</span></h4>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if is_critical:
        st.markdown(
            '<div style="background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #EF4444;'
            'border-radius:8px;padding:10px 16px;color:#991B1B;font-weight:600;margin:8px 0;">'
            '⚠️  Classe critique — vérification humaine recommandée</div>',
            unsafe_allow_html=True,
        )

    try:
        conf_row = fetch_class_confidence()
        conf_row = conf_row[conf_row["predicted_class"] == pred_class]
    except Exception:
        conf_row = pd.DataFrame()
    if not conf_row.empty:
        total = int(conf_row["total_predictions"].iloc[0])
        corrections = int(conf_row["corrections"].iloc[0])
        conf_pct = conf_row["confidence_pct"].iloc[0]
        if corrections == 0:
            st.caption(
                f"Indice de confiance pour {pred_class} : {conf_pct:.0f}% "
                f"— jamais corrigée sur {total} prédictions (6 derniers mois)."
            )
        else:
            st.caption(
                f"Indice de confiance pour {pred_class} : {conf_pct:.0f}% "
                f"— corrigée {corrections} fois sur {total} prédictions (6 derniers mois)."
            )

    col_cam, col_orig, col_info = st.columns([2, 2, 3])

    with col_cam:
        st.markdown("**GradCAM**")
        if "gradcam_b64" in res:
            st.image(_b64_to_pil(res["gradcam_b64"]), use_container_width=True,
                     caption="Zones d'attention du modèle")

    with col_orig:
        st.markdown("**Image originale**")
        if "original_img_b64" in res:
            st.image(_b64_to_pil(res["original_img_b64"]), use_container_width=True)

    with col_info:
        st.markdown("**Résultat**")
        st.markdown(
            f'<div style="background:{color}18;border-left:4px solid {color};'
            f'border-radius:8px;padding:14px 18px;margin-bottom:14px;">'
            f'<div style="font-size:1.35rem;font-weight:800;color:{color};">'
            f'{pred_class.upper()}</div>'
            f'<div style="font-size:0.95rem;color:#374151;">Confiance : '
            f'<b>{confidence*100:.1f}%</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Top 5 probabilités**")
        sorted_p = sorted(all_probas.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_p[:5]:
            c = CLASS_COLORS.get(cls, "#6B7280")
            warn = " ⚠" if cls in CRITICAL else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
                f'<span style="width:50px;font-size:0.78rem;font-weight:700;color:{c};">'
                f'{CLASS_ABBR.get(cls, "?")}{warn}</span>'
                f'<div style="flex:1;background:#F1F5F9;border-radius:4px;height:8px;">'
                f'<div style="width:{min(prob*100, 100):.1f}%;background:{c};'
                f'height:8px;border-radius:4px;"></div></div>'
                f'<span style="font-size:0.78rem;color:#374151;width:40px;text-align:right;">'
                f'{prob*100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("**Avis du médecin**")
    feedback_key = f"feedback_done_{prediction_id}_{idx}"

    if st.session_state.get(feedback_key):
        st.success("Votre avis a été enregistré.")
    elif prediction_id is None:
        st.caption("Avis indisponible (Supabase hors ligne)")
    else:
        col_r, col_c = st.columns([1, 1])
        with col_r:
            agrees_label = st.radio(
                "Accord avec la prédiction ?",
                ["Oui", "Non"],
                horizontal=True,
                index=None,
                key=f"agrees_{prediction_id}_{idx}",
            )
        corrected_class = None
        if agrees_label == "Non":
            with col_c:
                other_classes = [c for c in CLASSES if c != pred_class]
                corrected_class = st.selectbox(
                    "Classe correcte",
                    other_classes,
                    key=f"corrected_{prediction_id}_{idx}",
                )
        comment = st.text_area(
            "Commentaire (optionnel)",
            key=f"comment_{prediction_id}_{idx}",
            height=72,
        )
        submit_label = "Correction" if agrees_label == "Non" else "Envoyer mon avis"
        if st.button(submit_label, key=f"submit_{prediction_id}_{idx}", type="primary"):
            if agrees_label is None:
                st.warning("Précisez si vous êtes d'accord ou non.")
            else:
                fb = send_feedback(
                    prediction_id,
                    agrees=(agrees_label == "Oui"),
                    corrected_class=corrected_class,
                    comment=comment or None,
                )
                if fb["ok"]:
                    st.session_state[feedback_key] = True
                    fetch_class_confidence.clear()
                    st.rerun()
                else:
                    st.error(f"Erreur : {fb['message']}")


def show_classification_tab() -> None:
    st.markdown(
        '<p style="color:#1E293B;font-size:1.1rem;font-weight:700;margin:0 0 14px;">'
        '🔬 Analyse de frottis sanguin'
        '<span style="font-weight:400;font-size:0.82rem;color:#94A3B8;margin-left:12px;">'
        'DenseNet-121 · 8 classes · GradCAM++</span></p>',
        unsafe_allow_html=True,
    )

    patient_name_input = st.text_input(
        "Nom du patient (optionnel)",
        placeholder="Ex : GLPG_123456_20250907 — laissé vide, un nom générique sera attribué",
    )

    source = st.radio(
        "Source des images", ["📁 Upload local", "☁️ DagsHub"],
        horizontal=True, label_visibility="collapsed",
    )

    # ── Source : Upload local ──────────────────────────────────────────────────
    if source == "📁 Upload local":
        col_up, col_btn = st.columns([5, 1])
        with col_up:
            uploaded_files = st.file_uploader(
                "Images du frottis",
                type=["jpg", "jpeg", "png", "tiff", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Sélectionnez les images de cellules sanguines du patient",
            )
        with col_btn:
            analyse_btn = st.button(
                "Analyser", type="primary", use_container_width=True,
                disabled=not uploaded_files,
            )

        if not uploaded_files:
            st.info("Chargez les images de cellules du patient pour lancer l'analyse.")
            return

        files_to_analyze = uploaded_files
        st.caption(f"{len(files_to_analyze)} image(s) chargée(s)")

    # ── Source : DagsHub ──────────────────────────────────────────────────────
    else:
        import random as _random
        batches = _list_batches()
        if not batches:
            st.error("Aucun batch .dvc trouvé dans data/lotstiff/")
            return

        col_b, col_n = st.columns([3, 2])
        selected_batch = col_b.selectbox("Batch", batches)
        n_images = col_n.slider("Nombre d'images", 1, 50, 10)

        col_load, col_analyse = st.columns([2, 1])
        load_btn = col_load.button("☁️ Charger depuis DagsHub", use_container_width=True)

        if load_btn:
            st.session_state.pop("dh_images", None)
            with st.spinner(f"DVC pull {selected_batch}…"):
                ok, msg = _pull_batch(selected_batch)
            if not ok:
                st.error(f"DVC pull échoué : {msg}")
                return
            all_imgs = _list_batch_images(selected_batch)
            picked = _random.sample(all_imgs, min(n_images, len(all_imgs)))
            st.session_state["dh_images"] = picked
            st.session_state["dh_batch"] = selected_batch
            st.success(f"{len(picked)} images chargées depuis {selected_batch}")

        dh_images: list[Path] = st.session_state.get("dh_images", [])
        if not dh_images:
            st.info("Sélectionne un batch et clique sur Charger.")
            return

        batch_dir = LOTSTIFF_DIR / st.session_state.get("dh_batch", selected_batch)
        all_names = [str(p.relative_to(batch_dir)) for p in dh_images]
        selected_names = st.multiselect("Images sélectionnées", all_names, default=all_names)
        files_to_analyze = [batch_dir / n for n in selected_names]

        analyse_btn = col_analyse.button(
            "Analyser", type="primary", use_container_width=True,
            disabled=not files_to_analyze,
        )
        if not files_to_analyze:
            return

    # ── Lancer l'analyse (commun upload + DagsHub) ────────────────────────────
    if analyse_btn:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("selected_idx", None)

        try:
            patient_id = next_patient_id()
        except Exception:
            patient_id = None
        patient_name = patient_name_input.strip() or f"Patient {patient_id}"
        triggered_by = st.session_state.get("username")

        results = []
        n_files = len(files_to_analyze)
        progress = st.progress(0, text="Initialisation…")
        status = st.empty()

        for i, f in enumerate(files_to_analyze):
            fname = f.name if hasattr(f, "read") else Path(f).name
            status.caption(f"Analyse de {fname}  ({i + 1}/{n_files})")
            img = Image.open(f).convert("RGB")
            res = gradcam_predict(
                img, fname, patient_id=patient_id, patient_name=patient_name,
                triggered_by=triggered_by,
            )
            res["filename"] = fname
            res["patient_id"] = patient_id
            res["patient_name"] = patient_name
            res["original_img_b64"] = _pil_to_b64(img.resize((224, 224)))
            results.append(res)
            progress.progress((i + 1) / n_files, text=f"{i + 1}/{n_files} analysées")

        progress.empty()
        status.empty()
        st.session_state["batch_results"] = results

    # ── Affichage des résultats ──
    if "batch_results" not in st.session_state:
        return

    results = st.session_state["batch_results"]
    selected_idx = st.session_state.get("selected_idx")
    batch_patient_name = next((r["patient_name"] for r in results if r.get("patient_name")), None)

    st.divider()
    if batch_patient_name is not None:
        st.caption(f"🧑‍⚕️ {batch_patient_name} — lot de {len(results)} images")

    # Résumé
    n_ok = sum(1 for r in results if "predicted_class" in r)
    n_crit = sum(1 for r in results if r.get("is_critical"))
    classes_counts: dict[str, int] = {}
    for r in results:
        cls = r.get("predicted_class", "?")
        classes_counts[cls] = classes_counts.get(cls, 0) + 1
    avg_conf = float(np.mean([r.get("confidence", 0) for r in results if "confidence" in r])) if n_ok else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images analysées", n_ok)
    m2.metric("⚠ Critiques", n_crit)
    m3.metric("Classe dominante", max(classes_counts, key=classes_counts.get) if classes_counts else "—")
    m4.metric("Confiance moy.", f"{avg_conf*100:.1f}%")

    st.markdown("### Grille d'inférence")
    st.caption("Cliquez sur **Détail** pour voir l'analyse complète d'une cellule.")

    # ── Grille ──
    for row_start in range(0, len(results), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col_i in range(GRID_COLS):
            abs_i = row_start + col_i
            if abs_i >= len(results):
                break
            res = results[abs_i]
            with cols[col_i]:
                if "error" in res:
                    st.error("Erreur API")
                    continue
                is_selected = selected_idx == abs_i
                is_crit = res.get("is_critical", False)
                border = (
                    "3px solid #3B82F6" if is_selected
                    else "2px solid #EF4444" if is_crit
                    else "2px solid #E2E8F0"
                )
                st.markdown(
                    f'<div style="border:{border};border-radius:8px;padding:2px;">',
                    unsafe_allow_html=True,
                )
                st.image(_b64_to_pil(res["gradcam_b64"]), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(
                    _class_badge_html(res["predicted_class"], res["confidence"]),
                    unsafe_allow_html=True,
                )
                fname = res.get("filename", "")
                st.caption(fname.rsplit(".", 1)[0] if fname else "")
                if st.button(
                    "Détail",
                    key=f"sel_{abs_i}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state["selected_idx"] = abs_i
                    st.rerun()

    # ── Panneau détail ──
    if selected_idx is not None and selected_idx < len(results):
        st.divider()
        _show_cell_detail(results[selected_idx], selected_idx)


# ── Logs tab ──────────────────────────────────────────────────────────────────

def show_logs_tab() -> None:
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("### Historique des entraînements")
    with col_btn:
        if st.button("↺ Rafraîchir", use_container_width=True):
            fetch_training_runs.clear()
            fetch_mlflow_run_data.clear()
            fetch_production_version.clear()

    try:
        runs = fetch_training_runs()
    except Exception as e:
        st.error(f"Supabase indisponible : {e}")
        return

    if runs.empty:
        st.info("Aucun run d'entraînement loggué.")
        return

    prod = fetch_production_version()
    run_data = fetch_mlflow_run_data()

    runs["macro_f1"]      = runs["mlflow_run_id"].map(lambda r: run_data.get(r, {}).get("macro_f1"))
    runs["accuracy"]      = runs["mlflow_run_id"].map(lambda r: run_data.get(r, {}).get("accuracy"))
    runs["mlflow_version"] = runs["mlflow_run_id"].map(lambda r: run_data.get(r, {}).get("mlflow_version"))
    runs["git_commit"]    = runs["mlflow_run_id"].map(
        lambda r: (run_data.get(r, {}).get("git_commit") or "")[:10]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs loggués", len(runs))
    c2.metric("Taux de succès", f"{(runs['status'] == 'success').mean() * 100:.0f}%")
    c3.metric("Durée moyenne", f"{runs['duration_seconds'].mean():.0f}s")
    if prod:
        c4.metric(
            "@production",
            f"version {prod['version']}",
            f"génération {prod['generation']} · macro_f1={prod['macro_f1']:.4f}"
            if prod["macro_f1"] is not None else f"génération {prod['generation']}",
            delta_color="off",
        )
    else:
        c4.metric("@production", "—")

    st.divider()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        gens = ["Toutes"] + sorted(runs["generation"].dropna().unique().tolist())
        gen_filter = st.selectbox("Génération", gens)
    with col_f2:
        status_filter = st.selectbox("Statut", ["Tous", "success", "failed"])

    filtered = runs.copy()
    if gen_filter != "Toutes":
        filtered = filtered[filtered["generation"] == gen_filter]
    if status_filter != "Tous":
        filtered = filtered[filtered["status"] == status_filter]

    col_order = [
        "mlflow_run_id", "model_name", "generation", "mlflow_version", "git_commit",
        "device", "status", "started_at", "duration_seconds",
        "cpu_percent_avg", "ram_used_mb_avg", "gpu_name",
        "gpu_util_percent_avg", "gpu_mem_used_mb_avg", "macro_f1", "accuracy",
    ]
    col_order = [c for c in col_order if c in filtered.columns]
    st.dataframe(
        filtered[col_order],
        column_config={
            "mlflow_run_id":       "Run ID",
            "model_name":          "Modèle",
            "mlflow_version":      st.column_config.NumberColumn("Version MLflow", format="%d"),
            "generation":          "Génération",
            "git_commit":          st.column_config.TextColumn("Commit Git"),
            "device":              "Device",
            "status":              "Statut",
            "started_at":          st.column_config.DatetimeColumn("Démarré le"),
            "duration_seconds":    st.column_config.NumberColumn("Durée (s)", format="%.0f"),
            "cpu_percent_avg":     st.column_config.NumberColumn("CPU %", format="%.1f"),
            "ram_used_mb_avg":     st.column_config.NumberColumn("RAM (Mo)", format="%.0f"),
            "gpu_name":            "GPU",
            "gpu_util_percent_avg": st.column_config.NumberColumn("GPU %", format="%.1f"),
            "gpu_mem_used_mb_avg": st.column_config.NumberColumn("GPU mem (Mo)", format="%.0f"),
            "macro_f1":            st.column_config.NumberColumn("macro_f1", format="%.4f"),
            "accuracy":            st.column_config.NumberColumn("accuracy", format="%.4f"),
        },
        use_container_width=True,
        hide_index=True,
    )


# ── Monitoring tab ────────────────────────────────────────────────────────────

def show_monitoring_tab() -> None:
    st.markdown("### Monitoring du modèle")

    # ── Placeholder Evidently ── insérer ici les rapports HTML Evidently
    st.markdown("""
    <div style="background:white;border-radius:12px;border:2px dashed #CBD5E0;
                padding:64px 40px;text-align:center;color:#94A3B8;">
        <div style="font-size:3rem;margin-bottom:16px;">📊</div>
        <h2 style="color:#64748B;margin:0 0 8px;">Evidently — Data Drift &amp; Performance</h2>
        <p style="max-width:500px;margin:0 auto 24px;font-size:0.95rem;
                  line-height:1.7;color:#94A3B8;">
            Cet onglet accueillera les rapports de monitoring générés par
            <strong style="color:#64748B;">Evidently AI</strong> : dérive des données,
            dégradation des performances, distributions de prédictions et alertes automatiques.
        </p>
        <div style="background:#F8FAFC;border-radius:8px;display:inline-block;
                    padding:16px 24px;text-align:left;">
            <p style="margin:0 0 8px;font-size:0.85rem;font-weight:700;color:#64748B;">
                🔧 À intégrer par l'équipe monitoring :
            </p>
            <ul style="margin:0;padding-left:20px;font-size:0.85rem;color:#94A3B8;line-height:1.8;">
                <li>Data drift report (<code>DataDriftPreset</code>)</li>
                <li>Classification quality report</li>
                <li>Prediction drift over time</li>
                <li>Feature importance evolution</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_search_tab() -> None:
    """Onglet Recherche : retrouver des predictions par nom d'image ou par nom de patient
    (patient simule — un lot d'images analyse = un patient, cf. show_classification_tab)."""
    st.subheader("Recherche de predictions")
    st.caption(
        "Patients simules : chaque lot analyse dans l'onglet Classification "
        "est associe a un nom de patient (saisi, ou generique si laisse vide)."
    )

    col_name, col_patient, col_search = st.columns([3, 2, 1])
    with col_name:
        image_name = st.text_input("Nom d'image (recherche partielle)", value="")
    with col_patient:
        patient_name_input = st.text_input("Nom du patient (recherche partielle)", value="")
    with col_search:
        st.write("")
        search_btn = st.button("Rechercher", type="primary", use_container_width=True)

    if not search_btn and "search_results" not in st.session_state:
        st.info("Renseigne un nom d'image et/ou un nom de patient, puis clique sur Rechercher.")
        return

    if search_btn:
        try:
            st.session_state["search_results"] = search_predictions(
                image_name=image_name.strip(), patient_name=patient_name_input.strip(),
            )
        except Exception as e:
            st.error(f"Recherche impossible : {e}")
            return

    df = st.session_state.get("search_results")
    if df is None or df.empty:
        st.info("Aucun resultat.")
        return

    st.caption(f"{len(df)} resultat(s) (200 max)")
    df_display = df.rename(columns={
        "id": "ID",
        "patient_id": "ID patient",
        "patient_name": "Patient",
        "image_name": "Image",
        "predicted_class": "Classe predite",
        "confidence": "Confiance",
        "model_version": "Version modele",
        "triggered_by": "Analyse par",
        "created_at": "Date",
    })
    st.dataframe(
        df_display.style.format({"Confiance": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

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
    # CSS global pour forcer les labels en noir
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 style="color:#0f172a;font-weight:700;">Monitoring du drift (IVDR 2017/746)</h2>', unsafe_allow_html=True)  # noqa: E501

    st.markdown("""
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px;
                  background:white;border-radius:10px;overflow:hidden;
                  box-shadow:0 1px 4px rgba(0,0,0,0.07);border:1px solid #e2e8f0;">
      <thead>
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:10px 14px;text-align:left;color:#0f172a;font-weight:700;">Niveau</th>
          <th style="padding:10px 14px;text-align:left;color:#0f172a;font-weight:700;">Score</th>
          <th style="padding:10px 14px;text-align:left;color:#0f172a;font-weight:700;">Signification</th>
          <th style="padding:10px 14px;text-align:left;color:#0f172a;font-weight:700;">Action IVDR</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;"><span
            style="background:#dcfce7;color:#14532d;border:1px solid #16a34a;
            border-radius:6px;padding:3px 10px;font-weight:700;">✅ Normal</span></td>
          <td style="padding:10px 14px;color:#0f172a;font-weight:700;">&lt; 0.10</td>
          <td style="padding:10px 14px;color:#334155;">Aucun drift significatif</td>
          <td style="padding:10px 14px;color:#334155;">Aucune action requise</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;"><span
            style="background:#fef9c3;color:#713f12;border:1px solid #ca8a04;
            border-radius:6px;padding:3px 10px;font-weight:700;">⚠️ Warning</span></td>
          <td style="padding:10px 14px;color:#0f172a;font-weight:700;">0.10 – 0.20</td>
          <td style="padding:10px 14px;color:#334155;">Drift léger détecté</td>
          <td style="padding:10px 14px;color:#334155;">Surveillance renforcée</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;"><span
            style="background:#ffedd5;color:#7c2d12;border:1px solid #ea580c;
            border-radius:6px;padding:3px 10px;font-weight:700;">🟠 Alerte</span></td>
          <td style="padding:10px 14px;color:#0f172a;font-weight:700;">0.20 – 0.30</td>
          <td style="padding:10px 14px;color:#334155;">Drift modéré</td>
          <td style="padding:10px 14px;color:#334155;">Analyse + envisager ré-entraînement (MDCG 2020-1)</td>
        </tr>
        <tr>
          <td style="padding:10px 14px;"><span
            style="background:#fee2e2;color:#7f1d1d;border:1px solid #dc2626;
            border-radius:6px;padding:3px 10px;font-weight:700;">🔴 Critique</span></td>
          <td style="padding:10px 14px;color:#0f172a;font-weight:700;">≥ 0.30</td>
          <td style="padding:10px 14px;color:#334155;">Drift sévère</td>
          <td style="padding:10px 14px;color:#334155;">Investigation immédiate obligatoire (ISO 14971 §9)</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

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

    # ── Compteurs images ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Images de référence", f"{result.get('n_reference', 0):,}")
    c2.metric("Images en production (courantes)", f"{result.get('n_current', 0):,}")
    generated_at = (
        result.get("created_at")
        or result.get("metrics", {}).get("generated_at", "—")
    )
    c3.metric("Rapport généré le", str(generated_at)[:16])

    st.divider()

    # ── Rapport showcase ──────────────────────────────────────────────────────
    try:
        from src.evidently.generate_showcase_report import build_showcase_html
        showcase_html = build_showcase_html(result)
        st.components.v1.html(showcase_html, height=1200, scrolling=True)
    except Exception as e:
        st.error(f"Erreur lors de la generation du rapport showcase : {e}")

    # ── Performance du modele (MLflow + Supabase class_metrics) ──────────────
    st.divider()
    st.markdown('<h3 style="color:#0f172a;font-weight:700;">Performance du modele (MLflow)</h3>', unsafe_allow_html=True)  # noqa: E501
    st.markdown(  # noqa: E501
        '<p style="color:#334155;font-size:14px;">Evolution de macro_F1 et accuracy par generation'
        " — seuil d'alerte IVDR : baisse &gt; 5%</p>",
        unsafe_allow_html=True,
    )

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
                            for metric, color in [("f1", "#1f77b4"), ("precision", "#ff7f0e"), ("recall", "#2ca02c")]:
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
                        valid_gens = df_classes["generation"].dropna()
                        last_gen = max(
                            valid_gens,
                            key=lambda g: int(g[1:]) if isinstance(g, str) and g[1:].isdigit() else 0,
                        ) if not valid_gens.empty else None
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
                        st.markdown(  # noqa: E501
                            f'<p style="color:#334155;font-size:13px;">Génération : '
                            f'<strong style="color:#0f172a;">{last_gen}</strong></p>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            df_last.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                            use_container_width=True,
                        )

    # ── Matrice de confusion ──────────────────────────────────────────────────
    st.divider()
    st.markdown('<h3 style="color:#0f172a;font-weight:700;">Matrice de confusion</h3>', unsafe_allow_html=True)
    try:
        from src.evidently.drift_report import list_confusion_generations, load_confusion_matrix
        generations = list_confusion_generations()
    except Exception as e:
        generations = []
        st.error(f"Impossible de lister les generations : {e}")

    if not generations:
        st.info("Aucune matrice de confusion disponible.")
    else:
        selected_gen = st.selectbox(
            "Génération", generations, index=0, key="cm_generation",
        )
        cm_data = load_confusion_matrix(selected_gen)
        if cm_data is None:
            st.info(f"Aucune matrice pour la generation {selected_gen}.")
        else:
            import plotly.express as px
            class_order = cm_data["class_order"]
            fig_cm = px.imshow(
                cm_data["matrix"],
                x=class_order,
                y=class_order,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Classe predite", y="Classe reelle", color="Nb"),
            )
            fig_cm.update_layout(
                title=f"Generation {cm_data['generation']} — {cm_data['created_at']}",
                height=500,
            )
            st.plotly_chart(fig_cm, use_container_width=True)


def show_context_tab() -> None:
    st.markdown(
        '<h2 style="color:#1E293B;font-weight:700;margin-bottom:4px;">📖 Contexte du projet</h2>'
        '<p style="color:#64748B;font-size:0.9rem;margin-bottom:24px;">'
        'Blood Cell Analyzer — DenseNet-121 · 8 classes · GradCAM++</p>',
        unsafe_allow_html=True,
    )

    # ── 0. Introduction ───────────────────────────────────────────────────────
    with st.expander("🏥 Introduction — CHU Lyon (HCL)", expanded=True):
        col_ctx, col_obj = st.columns(2)
        with col_ctx:
            st.markdown(
                '<div style="background:white;border-radius:12px;padding:20px 22px;'
                'border:1px solid #E2E8F0;margin-bottom:12px;">'
                '<p style="color:#2563EB;font-weight:700;font-size:1rem;margin:0 0 12px;">'
                'Contexte</p>'
                '<ul style="color:#334155;font-size:0.88rem;line-height:1.8;'
                'margin:0;padding-left:18px;">'
                '<li>Le CHU réalise un <strong>volume important d\'analyses sanguines</strong></li>'
                '<li>La lecture de frottis est critique mais '
                '<strong>aujourd\'hui manuelle</strong></li>'
                '<li>Processus actuel : <strong>long, coûteux</strong> et variable selon '
                'les observateurs = besoin de standardisation</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True,
            )
            _img_path = ROOT / "src" / "serving" / "assets" / "blood_test.png"
            if _img_path.exists():
                _, _ic, _ = st.columns([0.2, 0.6, 0.2])
                with _ic:
                    st.image(str(_img_path), use_container_width=True)
        with col_obj:
            st.markdown(
                '<div style="background:white;border-radius:12px;padding:20px 22px;'
                'border:1px solid #E2E8F0;height:100%;">'
                '<p style="color:#2563EB;font-weight:700;font-size:1rem;margin:0 0 12px;">'
                'Objectifs métier</p>'
                '<ul style="color:#334155;font-size:0.88rem;line-height:1.8;'
                'margin:0 0 16px;padding-left:18px;">'
                '<li>Accélérer l\'analyse des frottis sanguins du CHU</li>'
                '<li>Identifier &amp; classer les cellules sanguines</li>'
                '<li><strong>Aide au diagnostic</strong> via les prédictions du modèle</li>'
                '</ul>'
                '<div style="background:#EFF6FF;border-radius:8px;padding:14px 16px;'
                'border:1px solid #BFDBFE;">'
                '<p style="color:#2563EB;font-weight:700;font-size:0.88rem;margin:0 0 8px;">'
                'Users</p>'
                '<ul style="color:#1E40AF;font-size:0.85rem;line-height:1.7;'
                'margin:0;padding-left:16px;">'
                '<li>Biologiste</li><li>Hématologue</li><li>Chercheur</li>'
                '</ul>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── 1. Les 8 classes ──────────────────────────────────────────────────────
    with st.expander("🩸 Les 8 classes de cellules sanguines", expanded=True):
        CLASS_INFO = {
            "Basophil": (
                "BAS",
                "Globule blanc rare, impliqué dans les réactions allergiques et inflammatoires.",
            ),
            "Eosinophil": (
                "EOS",
                "Globule blanc actif dans les réponses immunitaires contre les parasites et les allergies.",
            ),
            "Erythroblast": (
                "ERY",
                "Précurseur des globules rouges. Sa présence dans le sang circulant est anormale"
                " et peut indiquer une anémie sévère ou une leucémie.",
            ),
            "IG": (
                "IG",
                "Immature Granulocyte — granulocyte immature. Marqueur d'infection sévère"
                " ou de pathologie myéloïde.",
            ),
            "Lymphocyte": (
                "LYM",
                "Cellule centrale de l'immunité adaptative (lymphocytes B et T).",
            ),
            "Monocyte": (
                "MON",
                "Grand globule blanc, précurseur des macrophages. Rôle clé dans la phagocytose.",
            ),
            "Neutrophil": (
                "NEU",
                "Globule blanc le plus abondant, première ligne de défense contre les bactéries.",
            ),
            "Platelet": (
                "PLT",
                "Thrombocyte — fragment cellulaire essentiel à la coagulation sanguine.",
            ),
        }
        cols = st.columns(2)
        for i, (cls, (abbr, desc)) in enumerate(CLASS_INFO.items()):
            color = CLASS_COLORS[cls]
            is_crit = cls in CRITICAL
            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:white;border-radius:10px;padding:12px 16px;'
                    f'margin-bottom:10px;border:1px solid #E2E8F0;'
                    f'border-left:4px solid {color};">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.75rem;font-weight:700;">{abbr}</span>'
                    f'<span style="font-weight:700;color:#1E293B;">{cls}</span>'
                    + (
                        '<span style="background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;'
                        'border-radius:4px;padding:1px 6px;font-size:0.7rem;font-weight:700;'
                        'margin-left:auto;">⚠ Critique</span>' if is_crit else ""
                    )
                    + f'</div>'
                    f'<p style="color:#475569;font-size:0.82rem;margin:0;line-height:1.5;">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── 2. Modèle DenseNet-121 ────────────────────────────────────────────────
    with st.expander("🧠 Modèle — DenseNet-121", expanded=True):
        col_txt, col_arch = st.columns([3, 2])
        with col_txt:
            st.markdown("""
<div style="color:#1E293B;">

**DenseNet-121** (Densely Connected Convolutional Network, 2017) est un réseau de neurones
convolutif profond composé de **121 couches**.

Son principe fondamental : chaque couche reçoit en entrée les **feature maps de toutes les
couches précédentes** (connexions denses). Cela permet :
- de réutiliser les caractéristiques apprises à chaque niveau
- de lutter contre le vanishing gradient
- d'être **plus compact** qu'un ResNet pour des performances équivalentes

**Dans ce projet**, le modèle est pré-entraîné sur ImageNet puis **fine-tuné** sur des images
de cellules sanguines (224×224 px). La dernière couche fully-connected est remplacée par un
classifieur à 8 sorties (softmax).

**GradCAM++** génère la carte de chaleur en calculant le gradient du score de la classe prédite
par rapport aux activations de la dernière couche convolutive — ce qui indique visuellement
quelles zones de l'image ont influencé la décision.
</div>
""", unsafe_allow_html=True)
        with col_arch:
            st.markdown(
                '<div style="background:#F8FAFC;border-radius:10px;padding:16px;'
                'border:1px solid #E2E8F0;font-size:0.82rem;color:#334155;line-height:1.8;">'
                '<div style="font-weight:700;color:#1E293B;margin-bottom:10px;">Architecture simplifiée</div>'
                '<div style="font-family:monospace;">'
                '📥 Input (224×224×3)<br>'
                '│<br>'
                '├─ Conv 7×7, stride 2<br>'
                '│<br>'
                '├─ Dense Block 1  (6 layers)<br>'
                '├─ Transition Layer<br>'
                '│<br>'
                '├─ Dense Block 2  (12 layers)<br>'
                '├─ Transition Layer<br>'
                '│<br>'
                '├─ Dense Block 3  (24 layers)<br>'
                '├─ Transition Layer<br>'
                '│<br>'
                '├─ Dense Block 4  (16 layers)<br>'
                '│<br>'
                '├─ Global Avg Pool<br>'
                '│<br>'
                '📤 FC → Softmax (8 classes)'
                '</div></div>',
                unsafe_allow_html=True,
            )

        prod = fetch_production_version()
        if prod:
            st.markdown(
                f'<div style="background:#EFF6FF;border-radius:8px;padding:10px 16px;'
                f'margin-top:12px;border:1px solid #BFDBFE;font-size:0.85rem;color:#1E40AF;">'
                f'🚀 <strong>Version en production</strong> : MLflow v{prod["version"]} '
                f'· Génération {prod["generation"]}'
                + (f' · macro_F1 = <strong>{prod["macro_f1"]:.4f}</strong>' if prod["macro_f1"] else "")
                + '</div>',
                unsafe_allow_html=True,
            )

    # ── 3. Architecture technique ─────────────────────────────────────────────
    with st.expander("⚙️ Architecture technique — MLOps", expanded=False):
        st.graphviz_chart("""
digraph mlops {
    rankdir=LR
    graph [fontname="Helvetica" splines=ortho nodesep=0.4 ranksep=0.7]
    node  [fontname="Helvetica" fontsize=11 style="rounded,filled" margin="0.18,0.1"]
    edge  [fontname="Helvetica" fontsize=9]

    /* ── 1. DATA ── */
    subgraph cluster_data {
        label="① Data" style=filled color="#EFF6FF"
        fontcolor="#1E40AF" fontsize=11 fontname="Helvetica Bold"
        DVC  [label="DVC\\nVersionning" fillcolor="#C7D2FE" color="#4F46E5" fontcolor="#1E1B4B"]
        DAG  [label="DagsHub\\nStorage"  fillcolor="#C7D2FE" color="#4F46E5" fontcolor="#1E1B4B"]
        DVC -> DAG [label="push/pull" dir=both]
    }

    /* ── 2. TRAIN ── */
    subgraph cluster_train {
        label="② Train" style=filled color="#F0FDF4"
        fontcolor="#14532D" fontsize=11 fontname="Helvetica Bold"
        PT   [label="PyTorch\\nDenseNet-121" fillcolor="#BBF7D0" color="#15803D" fontcolor="#14532D"]
        EXP  [label="MLflow\\nExperiment Tracking" fillcolor="#BBF7D0" color="#15803D" fontcolor="#14532D"]
        PT -> EXP [label="params / métriques\\n/ artefacts"]
    }

    /* ── 3. REGISTRY ── */
    subgraph cluster_registry {
        label="③ Model Registry" style=filled color="#FEF9C3"
        fontcolor="#713F12" fontsize=11 fontname="Helvetica Bold"
        REG  [label="MLflow Registry\\nStaging → Production" fillcolor="#FEF08A" color="#CA8A04" fontcolor="#713F12"]
    }

    /* ── 4. SERVE ── */
    subgraph cluster_serve {
        label="④ Serve" style=filled color="#FFF1F2"
        fontcolor="#881337" fontsize=11 fontname="Helvetica Bold"
        API  [label="FastAPI\\nInference + GradCAM" fillcolor="#FECDD3" color="#E11D48" fontcolor="#881337"]
        DOCK [label="Docker\\nContainer" fillcolor="#FECDD3" color="#E11D48" fontcolor="#881337"]
        API -> DOCK [label="image"]
    }

    /* ── 5. MONITOR ── */
    subgraph cluster_monitor {
        label="⑤ Monitor" style=filled color="#F5F3FF"
        fontcolor="#4C1D95" fontsize=11 fontname="Helvetica Bold"
        EVI  [label="Evidently\\nData Drift" fillcolor="#DDD6FE" color="#7C3AED" fontcolor="#4C1D95"]
        SUP  [label="Supabase\\nPrédictions + Feedback" fillcolor="#DDD6FE" color="#7C3AED" fontcolor="#4C1D95"]
        EVI -> SUP [label="rapport"]
    }

    /* ── 6. UI ── */
    subgraph cluster_ui {
        label="⑥ UI" style=filled color="#FFF7ED"
        fontcolor="#7C2D12" fontsize=11 fontname="Helvetica Bold"
        STR  [label="Streamlit\\nDashboard" fillcolor="#FED7AA" color="#EA580C" fontcolor="#7C2D12"]
    }

    /* ── FLUX PRINCIPAL ── */
    DAG  -> PT   [label="images train" color="#4F46E5" fontcolor="#4F46E5"]
    EXP  -> REG  [label="enregistrement modèle" color="#15803D" fontcolor="#15803D"]
    REG  -> API  [label="modèle @production" color="#CA8A04" fontcolor="#CA8A04" style=bold penwidth=2]
    DAG  -> API  [label="images batch" color="#4F46E5" fontcolor="#4F46E5"]
    API  -> SUP  [label="logs prédictions" color="#E11D48" fontcolor="#E11D48"]
    SUP  -> EVI  [label="données courantes" color="#7C3AED" fontcolor="#7C3AED"]
    SUP  -> STR  [label="historique" color="#7C3AED" fontcolor="#7C3AED"]
    API  -> STR  [label="prédiction + GradCAM" color="#E11D48" fontcolor="#E11D48"]
    EVI  -> STR  [label="rapport drift" color="#7C3AED" fontcolor="#7C3AED"]

    /* ── FEEDBACK LOOP ── */
    STR  -> SUP  [label="feedback médecin" style=dashed color="#EA580C" fontcolor="#EA580C"]
    SUP  -> PT   [label="données corrigées\\n(ré-entraînement)"
                  style=dashed color="#EA580C" fontcolor="#EA580C" constraint=false]
}
""", use_container_width=True)

        # Légende avec les vraies couleurs de marque
        STACK = [
            ("DVC",        "#7B61FF", "white"),
            ("DagsHub",    "#4F46E5", "white"),
            ("PyTorch",    "#EE4C2C", "white"),
            ("MLflow",     "#0194E2", "white"),
            ("FastAPI",    "#009688", "white"),
            ("Docker",     "#2496ED", "white"),
            ("Evidently",  "#7C3AED", "white"),
            ("Supabase",   "#3ECF8E", "#1a1a1a"),
            ("Streamlit",  "#FF4B4B", "white"),
        ]
        st.markdown(
            '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">'
            + "".join(
                f'<span style="background:{bg};color:{fg};border-radius:6px;'
                f'padding:3px 11px;font-size:0.78rem;font-weight:700;">{name}</span>'
                for name, bg, fg in STACK
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── 4. Utilisateurs ───────────────────────────────────────────────────────
    with st.expander("👥 Utilisateurs cibles", expanded=False):
        _assets = ROOT / "src" / "serving" / "assets"
        PERSONAS = [
            {
                "name": "Dr Katharina Weill",
                "photo": str(_assets / "Dr Katharina Weil.png"),
                "color": "#3B82F6",
                "age": "34 ans",
                "role": "Biologiste médical",
                "context": "Laboratoire hospitalier",
                "description": (
                    "Biologiste orientée sécurité et qualité. Veut "
                    "<strong>réduire le temps de lecture tout en garantissant "
                    "une détection robuste</strong> des anomalies, avec traçabilité "
                    "et <strong>intégration au flux de validation.</strong>"
                ),
            },
            {
                "name": "Dr Camille Morel",
                "photo": str(_assets / "Dr. Camille Morel.png"),
                "color": "#10B981",
                "age": "42 ans",
                "role": "Hématologue",
                "context": "Service clinique, urgences, hospitalisation, RCP",
                "description": (
                    "Clinicienne expérimentée, habituée aux situations critiques. "
                    "<strong>Recherche des outils qui accélèrent et fiabilisent "
                    "le diagnostic,</strong> sans complexifier, avec preuves "
                    "et robustesse en contexte hospitalier."
                ),
            },
            {
                "name": "Dr Étienne Nagel",
                "photo": str(_assets / "Dr Étienne Nagel.png"),
                "color": "#8B5CF6",
                "age": "33 ans",
                "role": "Post-doctorant en hématologie",
                "context": "Exploitation scientifique des données",
                "description": (
                    "Post-doctorant plein d'ambition, son rôle est "
                    "<strong>l'exploitation scientifique des données</strong> "
                    "des cellules pour la recherche (identifier des corrélations "
                    "entre morphologie et anomalies moléculaires…)."
                ),
            },
        ]

        cols = st.columns(3)
        for col, p in zip(cols, PERSONAS):
            with col:
                photo_path = Path(p["photo"])
                if photo_path.exists():
                    img_b64 = base64.b64encode(photo_path.read_bytes()).decode()
                    img_src = f"data:image/png;base64,{img_b64}"
                else:
                    img_src = ""
                st.markdown(
                    f'<div style="background:white;border-radius:14px;padding:20px;'
                    f'border:1px solid #E2E8F0;'
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.05);height:100%;">'
                    f'<p style="font-style:italic;font-weight:700;font-size:1.05rem;'
                    f'color:#1E293B;text-align:center;margin:0 0 14px;">{p["name"]}</p>'
                    f'<div style="text-align:center;margin-bottom:14px;">'
                    + (
                        f'<img src="{img_src}" '
                        f'style="width:96px;height:96px;border-radius:50%;object-fit:cover;'
                        f'border:3px solid {p["color"]};display:inline-block;"/>'
                        if img_src else
                        f'<div style="width:96px;height:96px;border-radius:50%;'
                        f'background:{p["color"]};display:inline-flex;align-items:center;'
                        f'justify-content:center;font-size:1.5rem;font-weight:800;'
                        f'color:white;">{p["name"][:2]}</div>'
                    )
                    + f'</div>'
                    f'<div style="font-size:0.85rem;color:#475569;line-height:1.75;'
                    f'margin-bottom:16px;">'
                    f'<strong style="color:#1E293B;">{p["age"]}</strong><br>'
                    f'{p["role"]}<br>'
                    f'<span style="color:#94A3B8;">Contexte : {p["context"]}</span>'
                    f'</div>'
                    f'<div style="background:{p["color"]};color:white;text-align:center;'
                    f'border-radius:6px;padding:5px 10px;font-size:0.75rem;font-weight:700;'
                    f'letter-spacing:0.08em;margin-bottom:12px;">DESCRIPTION</div>'
                    f'<p style="font-size:0.85rem;color:#334155;line-height:1.65;margin:0;">'
                    f'{p["description"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── 5. Réglementation & Sécurité ─────────────────────────────────────────
    with st.expander("🛡️ Réglementation & Sécurité", expanded=False):
        col_reg, col_sec = st.columns(2)
        with col_reg:
            st.markdown(
                '<div style="background:white;border-radius:10px;padding:16px;'
                'border:1px solid #E2E8F0;height:100%;">'
                '<div style="font-weight:700;color:#1E293B;margin-bottom:10px;">📋 Cadre réglementaire</div>'
                '<p style="color:#475569;font-size:0.85rem;line-height:1.7;margin:0;">'
                'Ce système est conçu en référence au règlement européen '
                '<strong>IVDR 2017/746</strong> sur les dispositifs médicaux de diagnostic in vitro.<br><br>'
                'Les principaux principes appliqués :'
                '</p>'
                '<ul style="color:#475569;font-size:0.85rem;line-height:1.9;margin:8px 0 0;padding-left:18px;">'
                '<li>Surveillance post-marché via le monitoring de drift (onglet Monitoring)</li>'
                '<li>Seuil d\'alerte de dégradation des performances : <strong>−5% macro_F1</strong></li>'
                '<li>Traçabilité complète des prédictions (Supabase) et des entraînements (MLflow)</li>'
                '<li>Validation humaine obligatoire pour les classes critiques (ERY, IG)</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_sec:
            st.markdown(
                '<div style="background:white;border-radius:10px;padding:16px;'
                'border:1px solid #E2E8F0;height:100%;">'
                '<div style="font-weight:700;color:#1E293B;margin-bottom:10px;">🔒 Sécurité</div>'
                '<ul style="color:#475569;font-size:0.85rem;line-height:1.9;margin:0;padding-left:18px;">'
                '<li><strong>Authentification</strong>'
                ' — accès restreint par identifiants (Supabase Auth)</li>'
                '<li><strong>API sécurisée</strong>'
                ' — clé secrète requise sur chaque appel (<code>X-API-Key</code>)</li>'
                '<li><strong>Traçabilité</strong>'
                " — chaque prédiction enregistre l'utilisateur déclencheur</li>"
                '<li><strong>Feedback médecin</strong>'
                ' — correction humaine tracée pour audit et ré-entraînement</li>'
                '<li><strong>Isolation des services</strong>'
                ' — architecture Docker réseau interne, API non exposée publiquement</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="Blood Cell Analyzer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",  # toujours ouverte
    )
    _apply_css()

    if not login_screen():
        return

    PAGES = {
        "Contexte":       ("📖", show_context_tab),
        "Classification": ("🔬", show_classification_tab),
        "Recherche":      ("🔍", show_search_tab),
        "Logs":           ("📋", show_logs_tab),
        "Monitoring":     ("📊", show_drift_tab),
    }

    if "page" not in st.session_state:
        st.session_state["page"] = "Contexte"

    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 0 16px;text-align:center;">
            <div style="font-size:2.2rem;">🔬</div>
            <div style="font-size:1.05rem;font-weight:700;color:white;margin-top:6px;">
                Blood Cell Analyzer
            </div>
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:2px;">
                DenseNet-121 · 8 classes
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="border-top:1px solid #2D3F5E;margin:4px 0 8px;"></div>',
            unsafe_allow_html=True,
        )

        NAV_OPTIONS = {
            "📖  Contexte":       "Contexte",
            "🔬  Classification": "Classification",
            "🔍  Recherche":      "Recherche",
            "📋  Logs":           "Logs",
            "📊  Monitoring":     "Monitoring",
        }
        current = st.session_state.get("page", "Classification")
        current_key = next(k for k, v in NAV_OPTIONS.items() if v == current)
        selected = st.radio(
            "nav",
            list(NAV_OPTIONS.keys()),
            index=list(NAV_OPTIONS.keys()).index(current_key),
            label_visibility="collapsed",
        )
        if NAV_OPTIONS[selected] != st.session_state.get("page"):
            st.session_state["page"] = NAV_OPTIONS[selected]
            st.rerun()

        st.markdown(
            '<div style="border-top:1px solid #2D3F5E;margin:10px 0 8px;"></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"👤  {st.session_state.get('username', '')}")
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    PAGES[st.session_state["page"]][1]()


if __name__ == "__main__":
    main()
