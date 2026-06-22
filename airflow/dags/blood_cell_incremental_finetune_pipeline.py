"""
DAG Airflow — Fine-tuning incrémental sur un lot TIFF (générations V2+)

Déclenchement manuel uniquement (pas de planification automatique) : chaque
génération correspond à un lot précis (data/tiff_batches/batch_NNN) et un
numéro de génération à choisir au déclenchement (params batch/generation,
modifiables dans l'UI Airflow lors du "Trigger DAG w/ config").

Le lot est d'abord transféré vers le PC Windows (les liens symboliques de
data/tiff_batches ne pointent que sur le filesystem du Mac), puis
src/train/incremental_finetune.py charge le modèle @production courant, le
fine-tune (avec un buffer de replay Mendeley pour éviter l'oubli
catastrophique), évalue sur un set de référence, et décide lui-même de la
promotion — ce DAG vérifie ensuite le résultat et synchronise si besoin.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.trigger_rule import TriggerRule

from mlflow.tracking import MlflowClient
import mlflow

from _common import MLFLOW_TRACKING_URI, MODEL_NAME, PROJECT_DIR, supabase_env_exports, sync_to_datalake

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
    description="Fine-tuning incrémental sur un lot TIFF (V2+) — déclenchement manuel, batch/generation en params",
    schedule_interval=None,
    start_date=datetime(2026, 6, 22),
    catchup=False,
    default_args=default_args,
    tags=["blood-cell", "ml", "incremental", "tiff"],
    params={"batch": "batch_007", "generation": "v9"},
)


# ── Task 1 : Transférer le lot (résout les symlinks) vers le PC Windows ──────
def _transfer_batch(**context):
    """data/tiff_batches/<batch>/<classe>/*.tiff sont des symlinks vers
    l'archive TCIA sur le Mac — inutilisables tels quels depuis le PC
    Windows. On résout chaque lien et on transfère le contenu réel via SFTP."""
    from airflow.providers.ssh.hooks.ssh import SSHHook

    batch_name = context["params"]["batch"]
    local_batch_dir = Path(PROJECT_DIR) / "data" / "tiff_batches" / batch_name
    if not local_batch_dir.is_dir():
        raise FileNotFoundError(f"Lot introuvable : {local_batch_dir}")

    remote_base = f"{WINDOWS_REPO_DIR}\\data\\tiff_batches\\{batch_name}"
    hook = SSHHook(ssh_conn_id=WINDOWS_SSH_CONN_ID)
    ssh_client = hook.get_conn()

    def run_remote(cmd):
        _, stdout, stderr = ssh_client.exec_command(cmd)
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(f"Commande distante échouée ({status}) : {cmd}\n{stderr.read().decode()}")

    run_remote(f"powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '{remote_base}' | Out-Null\"")

    sftp = ssh_client.open_sftp()
    n_files = 0
    try:
        for cls_dir in sorted(local_batch_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            remote_cls_dir = f"{remote_base}\\{cls_dir.name}"
            run_remote(
                f"powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '{remote_cls_dir}' | Out-Null\""
            )
            for img in cls_dir.iterdir():
                if not img.is_file():
                    continue
                sftp.put(str(img.resolve()), f"{remote_cls_dir}\\{img.name}")
                n_files += 1
    finally:
        sftp.close()
        ssh_client.close()

    print(f"{n_files} images transférées vers {remote_base}")


transfer_batch = PythonOperator(
    task_id="transfer_batch",
    python_callable=_transfer_batch,
    dag=dag,
)


# ── Task 2 : Fine-tuning incrémental distant sur le PC Windows (GPU) ─────────
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
        "--batch-dir 'data/tiff_batches/{{ params.batch }}' "
        "--generation '{{ params.generation }}'"
        "\""
    ),
    dag=dag,
)


# ── Task 3 : Vérifier si cette génération a été promue @production ──────────
def _check_promotion(**context):
    generation = context["params"]["generation"]
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


# ── Task 4a/4b : Promotion confirmée / pas de promotion ──────────────────────
promote_success = BashOperator(
    task_id="promote_success",
    bash_command='echo "Génération {{ params.generation }} promue @production dans MLflow Registry."',
    dag=dag,
)

no_promotion = BashOperator(
    task_id="no_promotion",
    bash_command='echo "Génération {{ params.generation }} reste @challenger — @production inchangé."',
    dag=dag,
)


# ── Task 5 : Sync datalake + redémarrage inférence (même logique que le DAG
#             d'entraînement complet) ────────────────────────────────────────
sync = PythonOperator(
    task_id="sync_to_datalake",
    python_callable=sync_to_datalake,
    dag=dag,
)


# ── Task 6 : Fin (toujours exécutée) ──────────────────────────────────────────
done = BashOperator(
    task_id="pipeline_done",
    bash_command='echo "Pipeline blood_cell_incremental_finetune_pipeline terminé."',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)


# ── Dépendances ────────────────────────────────────────────────────────────────
transfer_batch >> finetune >> check_promotion >> [promote_success, no_promotion]
promote_success >> sync >> done
no_promotion >> done
