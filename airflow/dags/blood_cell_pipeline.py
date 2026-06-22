"""
DAG Airflow — Pipeline d'entraînement Blood Cell Classifier
Planification : chaque dimanche à 2h00

L'entraînement (cross-validation 5-fold DenseNet-121) tourne à distance sur le
PC Windows (GPU) via SSH/Tailscale — le conteneur Airflow ne fait
qu'orchestrer. Le script distant (src/train/dl_crossval_train.py) enregistre
lui-même la meilleure version dans le MLflow Model Registry et décide de la
promotion @production ; ce DAG se contente ensuite de vérifier le résultat.
"""

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.trigger_rule import TriggerRule

from mlflow.tracking import MlflowClient
import mlflow

MLFLOW_TRACKING_URI = "http://mlflow:5000"  # depuis le conteneur Airflow
MAC_MLFLOW_TAILSCALE_URI = f"http://{os.environ['MAC_TAILSCALE_IP']}:5001"  # depuis le PC Windows
MODEL_NAME = "blood-cell-densenet121"
WINDOWS_SSH_CONN_ID = "ssh_windows_gpu"
WINDOWS_REPO_DIR = os.environ["WINDOWS_REPO_DIR"]
GENERATION = "v1"  # à incrémenter (v2, v3...) à chaque nouveau cycle de données
PROJECT_DIR = "/opt/airflow/project"
INFERENCE_CONTAINERS = ("blood_cell_api", "blood_cell_streamlit")

default_args = {
    "owner": "romane",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="blood_cell_training_pipeline",
    description="Entraînement distant (Windows GPU) + vérification de la promotion du meilleur modèle",
    schedule_interval="0 2 * * 0",  # chaque dimanche à 2h
    start_date=datetime(2026, 6, 17),
    catchup=False,
    default_args=default_args,
    tags=["blood-cell", "ml", "training"],
)


# ── Task 1 : Entraînement distant sur le PC Windows (GPU) via SSH ────────────
train = SSHOperator(
    task_id="train_model",
    ssh_conn_id=WINDOWS_SSH_CONN_ID,
    cmd_timeout=None,
    command=(
        "powershell -NoProfile -Command \""
        f"$env:MLFLOW_TRACKING_URI='{MAC_MLFLOW_TAILSCALE_URI}'; "
        "$env:PYTHONUTF8='1'; "
        f"cd '{WINDOWS_REPO_DIR}'; "
        "& '.venv\\Scripts\\python.exe' -m src.train.dl_crossval_train "
        f"--generation {GENERATION}"
        "\""
    ),
    dag=dag,
)


# ── Task 2 : Vérifier si la nouvelle génération a été promue @production ─────
def _check_promotion(**context):
    """Le script distant a déjà décidé de la promotion (garde-fou macro_f1 +
    recall sur les 8 classes). Cette tâche ne fait qu'observer le résultat
    dans le Registry pour piloter la branche du DAG."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
    except Exception:
        print("Aucune version @production trouvée dans le Registry.")
        return "no_promotion"

    if prod_mv.tags.get("generation") == GENERATION:
        print(f"@production = version {prod_mv.version} (generation={GENERATION}) — promotion confirmée.")
        return "promote_success"

    print(f"@production reste version {prod_mv.version} "
          f"(generation={prod_mv.tags.get('generation')}) — {GENERATION} non promue.")
    return "no_promotion"


check_promotion = BranchPythonOperator(
    task_id="check_promotion",
    python_callable=_check_promotion,
    dag=dag,
)


# ── Task 3a : Promotion confirmée ─────────────────────────────────────────────
promote_success = BashOperator(
    task_id="promote_success",
    bash_command=f'echo "Génération {GENERATION} promue @production dans MLflow Registry."',
    dag=dag,
)


# ── Task 3b : Pas de promotion ────────────────────────────────────────────────
no_promotion = BashOperator(
    task_id="no_promotion",
    bash_command=f'echo "Génération {GENERATION} reste @challenger — @production inchangé."',
    dag=dag,
)


# ── Task 4 : Synchroniser le modèle @production vers le datalake DVC/DagsHub,
#             committer+pousser models.dvc sur GitHub, puis redémarrer les
#             conteneurs d'inférence pour qu'ils chargent la nouvelle version ──
def _sync_to_datalake(**context):
    import torch

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")

    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@production", map_location="cpu")
    torch.save(model.state_dict(), f"{PROJECT_DIR}/models/best_DenseNet_121.pth")
    print(f"Poids v{prod_mv.version} écrits dans models/best_DenseNet_121.pth")

    def run(cmd, redact=None):
        shown = [c if c != redact else "***" for c in cmd] if redact else cmd
        print(f"$ {' '.join(shown)}")
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True)

    run(["dvc", "add", "models"])
    run(["dvc", "push"])
    run(["git", "config", "user.email", "airflow@bloodcells.local"])
    run(["git", "config", "user.name", "Airflow (auto)"])
    run(["git", "add", "models.dvc"])

    generation = prod_mv.tags.get("generation", "?")
    commit_msg = f"Auto : promotion DenseNet-121 v{prod_mv.version} (generation={generation}) vers le datalake"
    commit_result = subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_DIR)
    if commit_result.returncode != 0:
        print("Rien à committer (models.dvc déjà à jour) — pas de push.")
    else:
        token = os.environ["GITHUB_TOKEN"]
        remote_url = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        ).stdout.strip()
        push_url = remote_url.replace("https://", f"https://x-access-token:{token}@")
        run(["git", "push", push_url, "HEAD:main"], redact=push_url)
        print(f"Pushé sur GitHub : {commit_msg}")

    for name in INFERENCE_CONTAINERS:
        result = subprocess.run(["docker", "restart", name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{name} redémarré — charge maintenant v{prod_mv.version}.")
        else:
            print(f"{name} non redémarré (probablement pas lancé) : {result.stderr.strip()}")


sync_to_datalake = PythonOperator(
    task_id="sync_to_datalake",
    python_callable=_sync_to_datalake,
    dag=dag,
)


# ── Task 5 : Fin (toujours exécutée) ──────────────────────────────────────────
done = BashOperator(
    task_id="pipeline_done",
    bash_command='echo "Pipeline blood_cell_training_pipeline terminé."',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)


# ── Dépendances ────────────────────────────────────────────────────────────────
train >> check_promotion >> [promote_success, no_promotion]
promote_success >> sync_to_datalake >> done
no_promotion >> done
