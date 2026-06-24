"""Fonctions partagées entre les DAGs blood_cell_*_pipeline.

Pas un DAG en soi (pas d'objet DAG() au niveau module) — Airflow scanne ce
fichier sans rien en tirer, les autres DAGs l'importent normalement.
"""

import os
import subprocess

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://mlflow:5000"  # depuis le conteneur Airflow
MODEL_NAME = "blood-cell-densenet121"
PROJECT_DIR = "/opt/airflow/project"
INFERENCE_CONTAINERS = ("blood_cell_api", "blood_cell_streamlit")


def supabase_env_exports() -> str:
    """Construit les exports PowerShell des identifiants Supabase, pour les
    transmettre au process distant (qui n'appelle pas load_dotenv())."""
    return "; ".join(
        f"$env:{var}='{os.environ[var]}'"
        for var in ("SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB", "SUPABASE_USER", "SUPABASE_PASSWORD")
    )


def sync_to_datalake(**context) -> None:
    """Télécharge le modèle @production courant, le pousse vers le datalake
    DVC/DagsHub, committe+push models.dvc sur GitHub, puis redémarre les
    conteneurs d'inférence pour qu'ils chargent la nouvelle version."""
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
