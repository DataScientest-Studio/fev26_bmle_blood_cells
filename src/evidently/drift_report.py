"""
Génération de rapports de drift Evidently pour le monitoring post-marché.

Drift surveillé (IVDR 2017/746 + MDCG 2020-1) :
  - Data drift    : features image (brightness, contraste, couleurs)
  - Prediction drift : distribution des classes prédites
  - Model drift   : accuracy via feedback médecin

Seuils d'alerte (IVDR / ISO 14971) :
  - data_drift_score > 0.1  → warning
  - data_drift_score > 0.2  → alerte
  - data_drift_score > 0.3  → critique
  - pred_drift sur classes critiques > 15% → critique
  - model macro_f1 baisse > 5%            → alerte
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# ── Seuils IVDR / ISO 14971 ──────────────────────────────────────────────────
DRIFT_WARNING  = 0.10
DRIFT_ALERT    = 0.20
DRIFT_CRITICAL = 0.30
CRITICAL_CLASSES = {"Erythroblast", "IG"}
MODEL_F1_ALERT_DROP = 0.05

FEATURE_COLS = [
    "mean_brightness", "std_brightness",
    "mean_r", "mean_g", "mean_b",
    "image_width", "image_height",
]


def _get_conn():
    from src.auth.db import get_connection
    return get_connection()


def _load_reference() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql(
        "SELECT class_name, mean_brightness, std_brightness, mean_r, mean_g, mean_b,"
        " image_width, image_height FROM reference_features",
        conn,
    )
    conn.close()
    return df


def _load_current(model_version: str | None) -> pd.DataFrame:
    conn = _get_conn()
    if model_version:
        df = pd.read_sql(
            """
            SELECT predicted_class, confidence,
                   mean_brightness, std_brightness, mean_r, mean_g, mean_b,
                   image_width, image_height, created_at
            FROM predictions
            WHERE model_version = %s
              AND mean_brightness IS NOT NULL
            ORDER BY created_at
            """,
            conn,
            params=(model_version,),
        )
    else:
        df = pd.read_sql(
            """
            SELECT predicted_class, confidence,
                   mean_brightness, std_brightness, mean_r, mean_g, mean_b,
                   image_width, image_height, created_at
            FROM predictions
            WHERE mean_brightness IS NOT NULL
            ORDER BY created_at
            """,
            conn,
        )
    conn.close()
    return df


def _load_feedback() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql(
        """
        SELECT p.predicted_class, p.confidence, p.model_version,
               f.agrees, f.corrected_class
        FROM prediction_feedback f
        JOIN predictions p ON p.id = f.prediction_id
        """,
        conn,
    )
    conn.close()
    return df


def _drift_level(score: float) -> str:
    if score is None:
        return "unknown"
    if score >= DRIFT_CRITICAL:
        return "critique"
    if score >= DRIFT_ALERT:
        return "alerte"
    if score >= DRIFT_WARNING:
        return "warning"
    return "normal"


def generate_report(model_version: str | None = None) -> dict:
    """
    Génère un rapport Evidently complet et le sauvegarde dans Supabase.

    Returns
    -------
    dict avec clés : data_drift_detected, data_drift_score, pred_drift_detected,
                     pred_drift_score, model_drift_score, report_html, metrics, level
    """
    from evidently.legacy.report import Report
    from evidently.legacy.metrics import (
        DatasetDriftMetric,
        ColumnDriftMetric,
        DatasetSummaryMetric,
    )

    ref = _load_reference()
    cur = _load_current(model_version)

    if len(cur) < 5:
        return {
            "error": f"Pas assez de prédictions en base ({len(cur)}) — minimum 5 requis.",
            "n_current": len(cur),
        }

    # ── Colonnes communes features image ─────────────────────────────────────
    available = [c for c in FEATURE_COLS if c in cur.columns and c in ref.columns]
    ref_feat = ref[available].dropna()
    cur_feat = cur[available].dropna()

    # ── 1. Data drift ─────────────────────────────────────────────────────────
    data_report = Report(metrics=[
        DatasetDriftMetric(),
        *[ColumnDriftMetric(column_name=c) for c in available],
        DatasetSummaryMetric(),
    ])
    data_report.run(reference_data=ref_feat, current_data=cur_feat)
    data_result = data_report.as_dict()

    dataset_drift = data_result["metrics"][0]["result"]
    data_drift_detected = dataset_drift.get("dataset_drift", False)
    data_drift_share    = dataset_drift.get("share_of_drifted_columns", 0.0)
    n_drifted           = dataset_drift.get("number_of_drifted_columns", 0)

    col_drift_scores = {}
    for i, col in enumerate(available, start=1):
        col_res = data_result["metrics"][i]["result"]
        col_drift_scores[col] = {
            "drift_detected": col_res.get("drift_detected", False),
            "drift_score":    round(col_res.get("drift_score", 0.0), 4),
            "stattest":       col_res.get("stattest_name", ""),
        }

    # ── 2. Prediction drift ───────────────────────────────────────────────────
    pred_report = Report(metrics=[
        ColumnDriftMetric(column_name="predicted_class"),
        ColumnDriftMetric(column_name="confidence"),
    ])
    ref_pred = ref[["class_name"]].rename(columns={"class_name": "predicted_class"})
    cur_pred = cur[["predicted_class", "confidence"]].dropna()
    ref_pred["confidence"] = 0.95  # valeur proxy pour la référence

    pred_report.run(reference_data=ref_pred, current_data=cur_pred)
    pred_result = pred_report.as_dict()

    pred_drift_detected = pred_result["metrics"][0]["result"].get("drift_detected", False)
    pred_drift_score    = round(pred_result["metrics"][0]["result"].get("drift_score", 0.0), 4)
    conf_drift_score    = round(pred_result["metrics"][1]["result"].get("drift_score", 0.0), 4)

    # ── 3. Model drift (feedback médecin) ─────────────────────────────────────
    feedback = _load_feedback()
    model_drift_score = None
    model_metrics = {}
    if not feedback.empty:
        accuracy = feedback["agrees"].mean()
        disagree_rate = 1 - accuracy
        model_metrics = {
            "n_feedback":    len(feedback),
            "accuracy":      round(float(accuracy), 4),
            "disagree_rate": round(float(disagree_rate), 4),
        }
        model_drift_score = round(float(disagree_rate), 4)

    # ── Rapport HTML combiné ──────────────────────────────────────────────────
    from evidently.legacy.report import Report as _Report
    html_report = _Report(metrics=[
        DatasetDriftMetric(),
        *[ColumnDriftMetric(column_name=c) for c in available],
        ColumnDriftMetric(column_name="predicted_class"),
        ColumnDriftMetric(column_name="confidence"),
        DatasetSummaryMetric(),
    ])

    ref_full = ref_feat.copy()
    ref_full["predicted_class"] = ref["class_name"].values[:len(ref_feat)]
    ref_full["confidence"] = 0.95

    cur_full = cur_feat.copy()
    cur_full["predicted_class"] = cur["predicted_class"].values[:len(cur_feat)]
    cur_full["confidence"] = cur["confidence"].values[:len(cur_feat)]

    html_report.run(reference_data=ref_full, current_data=cur_full)
    report_html = html_report.get_html()

    # ── Métriques consolidées ─────────────────────────────────────────────────
    metrics = {
        "data_drift": {
            "detected":         data_drift_detected,
            "share":            round(data_drift_share, 4),
            "n_drifted_features": n_drifted,
            "level":            _drift_level(data_drift_share),
            "per_feature":      col_drift_scores,
        },
        "prediction_drift": {
            "detected":        pred_drift_detected,
            "score":           pred_drift_score,
            "confidence_drift": conf_drift_score,
            "level":           _drift_level(pred_drift_score),
        },
        "model_drift": model_metrics,
        "thresholds": {
            "warning":  DRIFT_WARNING,
            "alert":    DRIFT_ALERT,
            "critical": DRIFT_CRITICAL,
        },
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "n_reference":   len(ref_feat),
        "n_current":     len(cur_feat),
    }

    # ── Sauvegarde dans Supabase ───────────────────────────────────────────────
    conn = _get_conn()
    cur_db = conn.cursor()
    cur_db.execute(
        """
        INSERT INTO drift_reports
            (model_version, n_reference, n_current,
             data_drift_detected, data_drift_score,
             pred_drift_detected, pred_drift_score,
             model_drift_score, n_drifted_features,
             metrics_json, report_html)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (
            model_version, len(ref_feat), len(cur_feat),
            data_drift_detected, data_drift_share,
            pred_drift_detected, pred_drift_score,
            model_drift_score, n_drifted,
            json.dumps(metrics), report_html,
        ),
    )
    report_id = cur_db.fetchone()[0]
    conn.commit()
    cur_db.close()
    conn.close()

    return {
        "report_id":           report_id,
        "data_drift_detected": data_drift_detected,
        "data_drift_score":    data_drift_share,
        "data_drift_level":    _drift_level(data_drift_share),
        "pred_drift_detected": pred_drift_detected,
        "pred_drift_score":    pred_drift_score,
        "pred_drift_level":    _drift_level(pred_drift_score),
        "model_drift_score":   model_drift_score,
        "n_drifted_features":  n_drifted,
        "n_reference":         len(ref_feat),
        "n_current":           len(cur_feat),
        "metrics":             metrics,
        "report_html":         report_html,
    }


def load_performance_metrics() -> dict:
    """
    Charge les métriques de performance du modèle depuis Supabase (class_metrics)
    et MLflow (macro_f1, accuracy) pour le monitoring post-marché.

    Retourne un dict avec :
      - df_global  : DataFrame macro_f1/accuracy par génération (depuis MLflow via training_runs)
      - df_classes : DataFrame precision/recall/f1 par classe et génération
      - baseline   : métriques de la première génération (référence)
      - current    : métriques de la dernière génération
      - alerts     : liste d'alertes IVDR si baisse > seuils
    """
    conn = _get_conn()

    # ── Métriques globales depuis Supabase training_runs + MLflow ────────────
    df_runs = pd.read_sql(
        """
        SELECT mlflow_run_id, generation, fold, started_at, status
        FROM training_runs
        WHERE status = 'success'
        ORDER BY started_at
        """,
        conn,
    )

    # ── Métriques par classe depuis Supabase ──────────────────────────────────
    df_classes = pd.read_sql(
        """
        SELECT generation, class_name, fold,
               AVG(precision) as precision,
               AVG(recall)    as recall,
               AVG(f1)        as f1,
               SUM(support)   as support
        FROM class_metrics
        GROUP BY generation, class_name, fold
        ORDER BY generation, class_name
        """,
        conn,
    )
    conn.close()

    # ── Enrichit avec macro_f1/accuracy depuis MLflow ─────────────────────────
    mlflow_metrics: dict[str, dict] = {}
    if not df_runs.empty:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
            client = MlflowClient()
            for exp in client.search_experiments():
                for run in client.search_runs(
                    experiment_ids=[exp.experiment_id], max_results=2000
                ):
                    mlflow_metrics[run.info.run_id] = {
                        "macro_f1": run.data.metrics.get("macro_f1"),
                        "accuracy": run.data.metrics.get("accuracy"),
                    }
        except Exception:
            pass

    if not df_runs.empty:
        df_runs["macro_f1"] = df_runs["mlflow_run_id"].map(
            lambda r: mlflow_metrics.get(r, {}).get("macro_f1")
        )
        df_runs["accuracy"] = df_runs["mlflow_run_id"].map(
            lambda r: mlflow_metrics.get(r, {}).get("accuracy")
        )
        df_global = (
            df_runs.dropna(subset=["macro_f1"])
            .groupby("generation", as_index=False)
            .agg(macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"))
            .sort_values("generation")
        )
    else:
        df_global = pd.DataFrame(columns=["generation", "macro_f1", "accuracy"])

    # ── Alertes IVDR (baisse > 5% sur macro_f1 ou classes critiques) ─────────
    alerts = []
    if len(df_global) >= 2:
        baseline_f1 = df_global.iloc[0]["macro_f1"]
        current_f1  = df_global.iloc[-1]["macro_f1"]
        drop = baseline_f1 - current_f1 if baseline_f1 else 0
        if drop >= MODEL_F1_ALERT_DROP:
            alerts.append(
                f"ALERTE — macro_f1 a baissé de {drop*100:.1f}% "
                f"({baseline_f1:.4f} → {current_f1:.4f})"
            )

    if not df_classes.empty:
        for cls in CRITICAL_CLASSES:
            cls_data = df_classes[df_classes["class_name"].str.lower() == cls.lower()]
            if len(cls_data) >= 2:
                f1_first = cls_data.iloc[0]["f1"]
                f1_last  = cls_data.iloc[-1]["f1"]
                drop = f1_first - f1_last if f1_first else 0
                if drop >= MODEL_F1_ALERT_DROP:
                    alerts.append(
                        f"CRITIQUE — F1 {cls} a baissé de {drop*100:.1f}% "
                        f"({f1_first:.4f} → {f1_last:.4f})"
                    )

    baseline = df_global.iloc[0].to_dict() if not df_global.empty else {}
    current  = df_global.iloc[-1].to_dict() if not df_global.empty else {}

    return {
        "df_global":  df_global,
        "df_classes": df_classes,
        "baseline":   baseline,
        "current":    current,
        "alerts":     alerts,
        "n_generations": len(df_global),
    }


def list_confusion_generations() -> list[str]:
    """Liste les generations disponibles pour les matrices de confusion, la plus recente en premier."""
    conn = _get_conn()
    try:
        df = pd.read_sql(
            """
            SELECT generation, MAX(created_at) as last_seen
            FROM confusion_matrices
            WHERE generation IS NOT NULL
            GROUP BY generation
            ORDER BY last_seen DESC
            """,
            conn,
        )
    finally:
        conn.close()
    return df["generation"].tolist()


def load_confusion_matrix(generation: str | None = None) -> dict | None:
    """Charge la matrice de confusion d'une generation (la plus recente si non precisee)."""
    conn = _get_conn()
    try:
        if generation:
            df = pd.read_sql(
                """
                SELECT generation, class_order, matrix, created_at
                FROM confusion_matrices
                WHERE generation = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                conn,
                params=(generation,),
            )
        else:
            df = pd.read_sql(
                """
                SELECT generation, class_order, matrix, created_at
                FROM confusion_matrices
                ORDER BY created_at DESC LIMIT 1
                """,
                conn,
            )
    finally:
        conn.close()

    if df.empty:
        return None
    row = df.iloc[0]
    class_order = row["class_order"] if isinstance(row["class_order"], list) else json.loads(row["class_order"])
    matrix = row["matrix"] if isinstance(row["matrix"], list) else json.loads(row["matrix"])
    return {
        "generation":  row["generation"],
        "class_order": class_order,
        "matrix":      matrix,
        "created_at":  str(row["created_at"]),
    }


def load_last_report() -> dict | None:
    """Charge le dernier rapport depuis Supabase."""
    conn = _get_conn()
    try:
        df = pd.read_sql(
            """
            SELECT * FROM drift_reports
            ORDER BY created_at DESC LIMIT 1
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "report_id":           int(row["id"]),
        "created_at":          str(row["created_at"]),
        "model_version":       row["model_version"],
        "n_reference":         int(row["n_reference"]) if row["n_reference"] else 0,
        "n_current":           int(row["n_current"]) if row["n_current"] else 0,
        "data_drift_detected": bool(row["data_drift_detected"]),
        "data_drift_score":    float(row["data_drift_score"]) if row["data_drift_score"] else 0.0,
        "data_drift_level":    _drift_level(float(row["data_drift_score"]) if row["data_drift_score"] else 0.0),
        "pred_drift_detected": bool(row["pred_drift_detected"]),
        "pred_drift_score":    float(row["pred_drift_score"]) if row["pred_drift_score"] else 0.0,
        "pred_drift_level":    _drift_level(float(row["pred_drift_score"]) if row["pred_drift_score"] else 0.0),
        "model_drift_score":   float(row["model_drift_score"]) if row["model_drift_score"] else None,
        "n_drifted_features":  int(row["n_drifted_features"]) if row["n_drifted_features"] else 0,
        "metrics":             (row["metrics_json"] if isinstance(row["metrics_json"], dict)
                                else json.loads(row["metrics_json"] or "{}")),
        "report_html":         row["report_html"],
    }
