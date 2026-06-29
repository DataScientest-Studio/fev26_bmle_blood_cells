"""
DAG Airflow — Fine-tuning incrémental sur un lot TIFF (générations V2+)

Planifié chaque dimanche 4h (apres le DAG d'entrainement complet de 2h, pour
ne pas se disputer le GPU Windows) : simule l'arrivee reguliere de nouvelles
donnees en consommant un nouveau lot data/lotstiff/batchN a chaque execution
(le prochain index est suivi via l'Airflow Variable
lotstiff_next_batch_idx). Le numero de generation est lui aussi calcule
automatiquement (cf. next_generation() dans _common.py). Si tous les lots
disponibles ont deja ete consommes, le DAG se termine sans rien faire.

Declenchement manuel toujours possible avec batch/generation explicites
(params, "Trigger DAG w/ config") pour rejouer un lot precis.

Le lot (data/lotstiff/batchN, tracke via DVC sur DagsHub — vraies copies,
pas de symlinks) est d'abord recupere sur le PC Windows via git pull +
dvc pull, puis src/train/incremental_finetune.py charge le modèle
@production courant, le fine-tune (avec un buffer de replay Mendeley pour
éviter l'oubli catastrophique), évalue sur un set de référence, et décide
lui-même de la promotion — ce DAG vérifie ensuite le résultat et
synchronise si besoin.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.trigger_rule import TriggerRule

from mlflow.tracking import MlflowClient
import mlflow

from _common import (
    MLFLOW_TRACKING_URI, MODEL_NAME,
    next_generation, next_lotstiff_batch, supabase_env_exports, sync_to_datalake,
)

MAC_MLFLOW_TAILSCALE_URI = f"http://{os.environ['MAC_TAILSCALE_IP']}:5001"  # depuis le PC Windows
WINDOWS_SSH_CONN_ID = "ssh_windows_gpu"
WINDOWS_REPO_DIR = os.environ["WINDOWS_REPO_DIR"]

default_args = {
    "owner": "fred",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="blood_cell_incremental_finetune_pipeline",
    description="Fine-tuning incrémental sur le prochain lot TIFF — planifié chaque dimanche 4h, "
    "ou manuel avec batch/generation en params",
    schedule_interval="0 4 * * 0",  # chaque dimanche à 4h (le DAG complet tourne à 2h)
    start_date=datetime(2026, 6, 22),
    catchup=False,
    default_args=default_args,
    tags=["blood-cell", "ml", "incremental", "tiff"],
    params={"batch": "", "generation": ""},
)


# ── Task 1 : Determiner le lot et la generation a traiter ───────────────────
def _determine_batch_and_generation(**context):
    """Utilise les params si fournis explicitement (declenchement manuel),
    sinon calcule automatiquement le prochain lot/generation (run planifie)."""
    ti = context["ti"]
    batch = context["params"].get("batch") or next_lotstiff_batch()
    if batch is None:
        raise AirflowSkipException("Tous les lots TIFF disponibles ont déjà été consommés.")
    generation = context["params"].get("generation") or next_generation()

    print(f"Lot retenu : {batch} — génération : {generation}")
    ti.xcom_push(key="batch", value=batch)
    ti.xcom_push(key="generation", value=generation)


determine_batch_and_generation = PythonOperator(
    task_id="determine_batch_and_generation",
    python_callable=_determine_batch_and_generation,
    dag=dag,
)


# ── Task 2 : Recuperer le lot depuis DagsHub (dvc pull) sur le PC Windows ────
def _transfer_batch(**context):
    """data/lotstiff/<batch> est tracke via DVC sur DagsHub (vraies copies de
    fichiers, pas de symlinks) -- on met a jour le repo Git distant puis on
    tire les donnees avec dvc pull, directement depuis le PC Windows. Plus
    besoin de resoudre des liens ni de SFTP fichier par fichier."""
    from airflow.providers.ssh.hooks.ssh import SSHHook

    batch_name = context["ti"].xcom_pull(task_ids="determine_batch_and_generation", key="batch")
    hook = SSHHook(ssh_conn_id=WINDOWS_SSH_CONN_ID)
    ssh_client = hook.get_conn()

    def run_remote(cmd):
        _, stdout, stderr = ssh_client.exec_command(cmd)
        status = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        if status != 0:
            err = stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Commande distante échouée ({status}) : {cmd}\n{err}")
        return out

    try:
        run_remote(f"powershell -NoProfile -Command \"cd '{WINDOWS_REPO_DIR}'; git pull\"")
        out = run_remote(
            "powershell -NoProfile -Command "
            f"\"cd '{WINDOWS_REPO_DIR}'; & '.venv\\Scripts\\python.exe' -m dvc pull "
            f"'data/lotstiff/{batch_name}.dvc'\""
        )
    finally:
        ssh_client.close()

    print(out)


transfer_batch = PythonOperator(
    task_id="transfer_batch",
    python_callable=_transfer_batch,
    dag=dag,
)


# ── Task 3 : Fine-tuning incrémental distant sur le PC Windows (GPU) ─────────
_XCOM_BATCH = "{{ ti.xcom_pull(task_ids='determine_batch_and_generation', key='batch') }}"
_XCOM_GENERATION = "{{ ti.xcom_pull(task_ids='determine_batch_and_generation', key='generation') }}"

finetune = SSHOperator(
    task_id="finetune_model",
    ssh_conn_id=WINDOWS_SSH_CONN_ID,
    cmd_timeout=None,
    command=(
        "powershell -NoProfile -Command \""
        f"$env:MLFLOW_TRACKING_URI='{MAC_MLFLOW_TAILSCALE_URI}'; "
        "$env:PYTHONUTF8='1'; "
        f"{supabase_env_exports()}; "
        f"cd '{WINDOWS_REPO_DIR}'; "
        "& '.venv\\Scripts\\python.exe' -m src.train.incremental_finetune "
        f"--batch-dir 'data/lotstiff/{_XCOM_BATCH}' "
        f"--generation '{_XCOM_GENERATION}'"
        "\""
    ),
    dag=dag,
)


# ── Task 4 : Vérifier si cette génération a été promue @production ──────────
def _check_promotion(**context):
    generation = context["ti"].xcom_pull(task_ids="determine_batch_and_generation", key="generation")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
    except Exception:
        print("Aucune version @production trouvée dans le Registry.")
        return "no_promotion"

    if prod_mv.tags.get("generation") == generation:
        print(f"@production = version {prod_mv.version} (generation={generation}) — promotion confirmée.")
        return "promote_success"

    print(f"@production reste version {prod_mv.version} "
          f"(generation={prod_mv.tags.get('generation')}) — {generation} non promue.")
    return "no_promotion"


check_promotion = BranchPythonOperator(
    task_id="check_promotion",
    python_callable=_check_promotion,
    dag=dag,
)


# ── Task 5a/5b : Promotion confirmée / pas de promotion ──────────────────────
promote_success = BashOperator(
    task_id="promote_success",
    bash_command=f'echo "Génération {_XCOM_GENERATION} promue @production dans MLflow Registry."',
    dag=dag,
)

no_promotion = BashOperator(
    task_id="no_promotion",
    bash_command=f'echo "Génération {_XCOM_GENERATION} reste @challenger — @production inchangé."',
    dag=dag,
)


# ── Task 6 : Sync datalake + redémarrage inférence (même logique que le DAG
#             d'entraînement complet) ────────────────────────────────────────
sync = PythonOperator(
    task_id="sync_to_datalake",
    python_callable=sync_to_datalake,
    dag=dag,
)


# ── Task 7 : Fin (toujours exécutée) ──────────────────────────────────────────
done = BashOperator(
    task_id="pipeline_done",
    bash_command='echo "Pipeline blood_cell_incremental_finetune_pipeline terminé."',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)


# ── Dépendances ────────────────────────────────────────────────────────────────
determine_batch_and_generation >> transfer_batch >> finetune >> check_promotion >> [promote_success, no_promotion]
promote_success >> sync >> done
no_promotion >> done
