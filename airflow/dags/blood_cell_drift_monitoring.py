"""
DAG Airflow — Monitoring drift quotidien (IVDR 2017/746)

Planifié chaque nuit à 7h. Génère un rapport Evidently complet
(data drift, prediction drift, model drift via feedback médecin),
le sauvegarde dans Supabase, et envoie un email d'alerte si le
niveau dépasse le seuil configuré (défaut : warning > 0.10).

Déclenchement manuel possible à tout moment depuis l'UI Airflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "romane",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="blood_cell_drift_monitoring",
    description="Rapport Evidently quotidien + alerte email si drift détecté",
    schedule_interval="0 7 * * *",  # chaque nuit à 7h
    start_date=datetime(2026, 6, 30),
    catchup=False,
    default_args=default_args,
    tags=["blood-cell", "monitoring", "drift", "ivdr"],
)


# ── Task 1 : Générer le rapport Evidently et le sauvegarder en base ────────────
def _generate_drift_report(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")

    from src.evidently.drift_report import generate_report

    result = generate_report(model_version=None)

    if "error" in result:
        raise RuntimeError(f"Rapport impossible : {result['error']}")

    print(
        f"Rapport généré (id={result['report_id']}) — "
        f"data_drift={result['data_drift_score']:.3f} [{result['data_drift_level']}], "
        f"pred_drift={result['pred_drift_score']:.3f} [{result['pred_drift_level']}], "
        f"features driftées={result['n_drifted_features']}"
    )
    context["ti"].xcom_push(key="drift_result", value={
        k: v for k, v in result.items() if k != "report_html"
    })


generate_drift_report = PythonOperator(
    task_id="generate_drift_report",
    python_callable=_generate_drift_report,
    dag=dag,
)


# ── Task 2 : Charger les métriques de performance (alertes IVDR macro_f1) ──────
def _check_performance(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")

    from src.evidently.drift_report import load_performance_metrics

    perf = load_performance_metrics()

    serializable_perf = {
        "alerts":       perf.get("alerts", []),
        "n_generations": perf.get("n_generations", 0),
        "baseline":     perf.get("baseline", {}),
        "current":      perf.get("current", {}),
    }

    if perf.get("alerts"):
        for alert in perf["alerts"]:
            print(f"⚠ {alert}")
    else:
        print(f"Performance OK sur {perf.get('n_generations', 0)} générations.")

    context["ti"].xcom_push(key="perf_result", value=serializable_perf)


check_performance = PythonOperator(
    task_id="check_performance",
    python_callable=_check_performance,
    dag=dag,
)


# ── Task 3 : Envoyer l'email d'alerte si nécessaire ───────────────────────────
def _send_alert_if_needed(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")

    from src.monitoring.email_alert import send_drift_alert

    drift_result = context["ti"].xcom_pull(task_ids="generate_drift_report", key="drift_result")
    perf_result  = context["ti"].xcom_pull(task_ids="check_performance", key="perf_result")

    sent = send_drift_alert(
        drift_result=drift_result,
        perf_result=perf_result,
        min_level="warning",
    )

    if not sent:
        print("Aucune alerte — tous les indicateurs sont dans les seuils normaux.")


send_alert_if_needed = PythonOperator(
    task_id="send_alert_if_needed",
    python_callable=_send_alert_if_needed,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)


# ── Dépendances ────────────────────────────────────────────────────────────────
[generate_drift_report, check_performance] >> send_alert_if_needed
