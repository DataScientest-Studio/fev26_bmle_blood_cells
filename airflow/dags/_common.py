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


def next_generation() -> str:
    """Calcule le prochain numero de generation (v1, v2, ...) a partir des
    tags 'generation' deja presents dans le MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    max_n = 0
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        gen = mv.tags.get("generation", "")
        if gen.startswith("v") and gen[1:].isdigit():
            max_n = max(max_n, int(gen[1:]))
    return f"v{max_n + 1}"


def next_lotstiff_batch(variable_name: str = "lotstiff_next_batch_idx") -> str | None:
    """Determine le prochain lot a traiter (data/lotstiff/batchN, tracke via DVC
    sur DagsHub), en se basant sur une Airflow Variable persistee entre les
    runs planifies.

    Retourne None si tous les lots disponibles ont deja ete utilises (fin de
    la simulation d'arrivee de nouvelles donnees)."""
    from pathlib import Path
    from airflow.models import Variable

    batches_dir = Path(PROJECT_DIR) / "data" / "lotstiff"
    available = sorted(
        (p.name for p in batches_dir.iterdir() if p.is_dir() and p.name.startswith("batch")),
        key=lambda n: int(n[len("batch"):]),
    )

    idx = int(Variable.get(variable_name, default_var="1"))
    if idx > len(available):
        return None

    batch_name = available[idx - 1]
    Variable.set(variable_name, str(idx + 1))
    return batch_name


def supabase_env_exports() -> str:
    """Construit les exports PowerShell des identifiants Supabase, pour les
    transmettre au process distant (qui n'appelle pas load_dotenv())."""
    return "; ".join(
        f"$env:{var}='{os.environ[var]}'"
        for var in ("SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB", "SUPABASE_USER", "SUPABASE_PASSWORD")
    )


def _rotate_reference_with_batch(batch_name: str) -> None:
    """Ajoute les images du lot de fine-tuning à reference_features après promotion.
    La référence de drift grandit ainsi avec chaque nouveau cycle de données.
    Échec silencieux — ne doit jamais bloquer la promotion."""
    from pathlib import Path
    try:
        import numpy as np
        from PIL import Image as PILImage
        import psycopg2
    except ImportError as e:
        print(f"[warn] rotation référence ignorée — import manquant : {e}")
        return

    batch_dir = Path(PROJECT_DIR) / "data" / "lotstiff" / batch_name
    if not batch_dir.is_dir():
        print(f"[warn] rotation référence ignorée — dossier introuvable : {batch_dir}")
        return

    try:
        conn = psycopg2.connect(
            host=os.environ.get("SUPABASE_HOST"),
            port=int(os.environ.get("SUPABASE_PORT", 6543)),
            dbname=os.environ.get("SUPABASE_DB"),
            user=os.environ.get("SUPABASE_USER"),
            password=os.environ.get("SUPABASE_PASSWORD"),
            connect_timeout=10,
            sslmode="require",
        )
        cur = conn.cursor()
        rows = []
        for cls_dir in sorted(batch_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tiff", ".tif"}:
                    continue
                try:
                    img = PILImage.open(img_path).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    gray = arr.mean(axis=2)
                    rows.append((
                        cls,
                        float(gray.mean()), float(gray.std()),
                        float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()),
                        float(arr[:, :, 2].mean()),
                        int(img.size[0]), int(img.size[1]),
                    ))
                except Exception:
                    continue

        if rows:
            cur.executemany("""
                INSERT INTO reference_features
                    (class_name, mean_brightness, std_brightness,
                     mean_r, mean_g, mean_b, image_width, image_height)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, rows)
            conn.commit()
            print(f"[OK] référence enrichie : +{len(rows)} images de {batch_name} "
                  f"({', '.join(f'{cls}: {sum(1 for r in rows if r[0]==cls)}' for cls in sorted(set(r[0] for r in rows)))})")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[warn] rotation référence (Supabase) indisponible : {e}")


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

    # Rotation de la référence de drift avec les images du lot qui a déclenché la promotion
    try:
        batch_name = context["ti"].xcom_pull(task_ids="determine_batch_and_generation", key="batch")
        if batch_name:
            _rotate_reference_with_batch(batch_name)
    except Exception as e:
        print(f"[warn] rotation référence ignorée : {e}")
