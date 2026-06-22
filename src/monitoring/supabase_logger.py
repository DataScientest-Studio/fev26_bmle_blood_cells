"""Logging des métriques d'entraînement (ressources + qualité par classe) vers
Supabase, en plus de MLflow. Échec silencieux si Supabase est indisponible —
ne doit jamais interrompre un entraînement (même pattern que
src/models/predict_model.py)."""

import json
import os

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def _connect():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        port=int(os.getenv("SUPABASE_PORT", 5432)),
        dbname=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        connect_timeout=10,
        sslmode="require",
    )


def log_training_run(
    *, mlflow_run_id, model_name, generation, fold,
    device, started_at, ended_at, resource_summary, status="success",
) -> None:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO training_runs
                (mlflow_run_id, model_name, generation, fold, device, started_at, ended_at,
                 duration_seconds, cpu_percent_avg, ram_used_mb_avg,
                 gpu_name, gpu_util_percent_avg, gpu_mem_used_mb_avg, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                mlflow_run_id, model_name, generation, fold, device, started_at, ended_at,
                resource_summary.get("duration_seconds"),
                resource_summary.get("cpu_percent_avg"),
                resource_summary.get("ram_used_mb_avg"),
                resource_summary.get("gpu_name"),
                resource_summary.get("gpu_util_percent_avg"),
                resource_summary.get("gpu_mem_used_mb_avg"),
                status,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  [warn] Supabase (training_runs) indisponible : {e}")


def log_class_metrics_and_confusion(
    *, mlflow_run_id, model_name, generation, fold, class_names, y_true, y_pred,
) -> None:
    """Calcule précision/recall/f1/support par classe + matrice de confusion,
    et logue les deux dans Supabase (class_metrics, confusion_matrices)."""
    try:
        labels = list(range(len(class_names)))
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

        conn = _connect()
        cur = conn.cursor()
        for i, cls in enumerate(class_names):
            cur.execute(
                """
                INSERT INTO class_metrics
                    (mlflow_run_id, model_name, generation, fold, class_name,
                     precision, recall, f1, support)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    mlflow_run_id, model_name, generation, fold, cls,
                    float(precisions[i]), float(recalls[i]), float(f1s[i]), int(supports[i]),
                ),
            )

        cur.execute(
            """
            INSERT INTO confusion_matrices
                (mlflow_run_id, model_name, generation, fold, class_order, matrix)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                mlflow_run_id, model_name, generation, fold,
                json.dumps(list(class_names)), json.dumps(cm.tolist()),
            ),
        )

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  [warn] Supabase (class_metrics/confusion_matrices) indisponible : {e}")
