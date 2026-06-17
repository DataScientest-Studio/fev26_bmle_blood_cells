"""
DAG Airflow — Pipeline d'entraînement Blood Cell Classifier
Planification : chaque dimanche à 2h00
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "blood_cell_densenet121"
PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "romane",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="blood_cell_training_pipeline",
    description="Entraînement planifié + comparaison + promotion du meilleur modèle",
    schedule_interval="0 2 * * 0",  # chaque dimanche à 2h
    start_date=datetime(2026, 6, 17),
    catchup=False,
    default_args=default_args,
    tags=["blood-cell", "ml", "training"],
)


# ── Task 1 : Entraînement ─────────────────────────────────────────────────────
train = BashOperator(
    task_id="train_model",
    bash_command=(
        f"cd {PROJECT_DIR} && "
        "python -m src.train.training "
        "--data-dir data/raw "
        "--epochs-head 5 "
        "--epochs-full 10"
    ),
    dag=dag,
)


# ── Task 2 : Comparer nouvelle version vs champion actuel ─────────────────────
def _compare_and_promote(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Dernière run enregistrée
    runs = mlflow.search_runs(
        experiment_names=["blood_cell_classification"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("Aucun run MLflow trouvé après l'entraînement.")

    new_run = runs.iloc[0]
    new_accuracy = new_run.get("metrics.val_acc", 0)
    new_run_id = new_run["run_id"]

    # Champion actuel dans le Registry
    champion_accuracy = 0.0
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_accuracy = champion_run.data.metrics.get("val_acc", 0)
    except Exception:
        pass  # Pas encore de champion enregistré

    print(f"Nouveau modèle : val_acc={new_accuracy:.4f}")
    print(f"Champion actuel : val_acc={champion_accuracy:.4f}")

    if new_accuracy >= champion_accuracy:
        # Enregistrer dans le Registry et promouvoir
        mv = mlflow.register_model(
            model_uri=f"runs:/{new_run_id}/model",
            name=MODEL_NAME,
        )
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        print(f"Nouveau champion : version {mv.version} (val_acc={new_accuracy:.4f})")
        return "promote_success"
    else:
        print("Le modèle actuel reste champion.")
        return "no_promotion"


compare = BranchPythonOperator(
    task_id="compare_and_promote",
    python_callable=_compare_and_promote,
    dag=dag,
)


# ── Task 3a : Promotion réussie ───────────────────────────────────────────────
promote_success = BashOperator(
    task_id="promote_success",
    bash_command='echo "Nouveau modèle promu champion dans MLflow Registry."',
    dag=dag,
)


# ── Task 3b : Pas de promotion ────────────────────────────────────────────────
no_promotion = BashOperator(
    task_id="no_promotion",
    bash_command='echo "Champion inchangé — nouveau modèle archivé."',
    dag=dag,
)


# ── Task 4 : Fin (toujours exécutée) ─────────────────────────────────────────
done = BashOperator(
    task_id="pipeline_done",
    bash_command='echo "Pipeline blood_cell_training_pipeline terminé."',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)


# ── Dépendances ───────────────────────────────────────────────────────────────
train >> compare >> [promote_success, no_promotion] >> done
