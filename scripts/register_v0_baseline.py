"""
Enregistre les 5 modèles DenseNet-121 (cross-validation 5-fold, baseline V0)
dans le MLflow Model Registry, sous le même nom que le modèle servi en
production (`blood-cell-densenet121`), sans toucher aux alias
@production/@challenger — chaque version est juste taguée generation=v0.

Sources (OneDrive, déjà synchronisé localement) :
  - poids     : BloodCellCaches/Caches/best_fold{1..5}_DenseNet_121.pth
  - métriques : BloodCellCaches/DL_crossval_ameliorees/crossval_results_per_fold.csv

Usage :
    python -m scripts.register_v0_baseline
"""

import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import timm
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if not os.getenv("ONEDRIVE_CACHE_DIR"):
    raise EnvironmentError("ONEDRIVE_CACHE_DIR doit être défini dans ton .env local.")

ONEDRIVE_CACHE_DIR = Path(os.environ["ONEDRIVE_CACHE_DIR"])
FOLD_WEIGHTS_DIR = ONEDRIVE_CACHE_DIR / "Caches"
CROSSVAL_RESULTS = ONEDRIVE_CACHE_DIR / "DL_crossval_ameliorees" / "crossval_results_per_fold.csv"

MLFLOW_MODEL_NAME = "blood-cell-densenet121"
NUM_CLASSES = 8


def main():
    # Volontairement pas os.getenv("MLFLOW_TRACKING_URI") : .env pointe vers le
    # sqlite local de notebook (src/jupyter/mlflow.db), pas le serveur Docker
    # partagé qu'utilisent Airflow/api.py/le PC Windows. Ce script doit toujours
    # écrire dans le registry partagé.
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("bloodcells-densenet121-baseline")

    df = pd.read_csv(CROSSVAL_RESULTS)
    df = df[df["model"] == "DenseNet-121"]
    if df.empty:
        raise RuntimeError(f"Aucune ligne 'DenseNet-121' dans {CROSSVAL_RESULTS}")

    client = MlflowClient()

    for _, row in df.iterrows():
        fold = int(row["fold"])
        weights_path = FOLD_WEIGHTS_DIR / f"best_fold{fold}_DenseNet_121.pth"
        if not weights_path.exists():
            print(f"  [SKIP] fold {fold} — poids introuvables : {weights_path}")
            continue

        model = timm.create_model("densenet121", pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()

        with mlflow.start_run(run_name=f"DenseNet-121_fold{fold}_v0") as run:
            mlflow.log_params({
                "model": "DenseNet-121",
                "fold": fold,
                "generation": "v0",
                "dataset_source": "mendeley_pbc_18k",
                "n_train": int(row["n_train"]),
                "n_val": int(row["n_val"]),
                "n_test": int(row["n_test"]),
                "epochs": int(row["epochs"]),
            })
            mlflow.set_tags({"generation": "v0", "source": "crossval_ameliorees_onedrive"})
            mlflow.log_metrics({
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
                "auc_roc": float(row["auc_roc"]),
                "elapsed_min": float(row["elapsed_min"]),
            })
            model_info = mlflow.pytorch.log_model(
                model, name="densenet121",
                input_example=torch.zeros(1, 3, 224, 224).numpy(),
                serialization_format="pickle",
            )

            mv = mlflow.register_model(model_uri=model_info.model_uri, name=MLFLOW_MODEL_NAME)
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "generation", "v0")
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "fold", str(fold))
            print(f"  Fold {fold} -> version {mv.version}  "
                  f"(macro_f1={row['macro_f1']:.4f})  tag generation=v0")


if __name__ == "__main__":
    main()
