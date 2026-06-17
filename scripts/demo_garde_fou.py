"""
Demo du garde-fou de promotion MLflow.

Enregistre un run fictif avec des métriques dégradées pour montrer
que le garde-fou bloque la promotion en @production.

Usage :
    python scripts/demo_garde_fou.py
"""

import os
import sys
from pathlib import Path

# Forcer UTF-8 pour les emojis MLflow sur Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MLFLOW_MODEL_NAME = "blood-cell-densenet121"
RECALL_TOLERANCE = 0.02

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
mlflow.set_experiment("bloodcells-densenet121")
client = MlflowClient()


# ── 1. Lire les métriques de @production ──────────────────────────────────────
print("=" * 60)
print("  Demo garde-fou de promotion MLflow")
print("=" * 60)

try:
    prod_mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
    prod_run = client.get_run(prod_mv.run_id)
    pm = prod_run.data.metrics
    prod_f1  = float(pm.get("macro_f1", 0.0))
    prod_ery = float(pm.get("recall_erythroblast", 0.0))
    prod_ig  = float(pm.get("recall_ig", 0.0))
    print(f"\n[production] Version {prod_mv.version}")
    print(f"  macro_f1             = {prod_f1:.4f}")
    print(f"  recall_erythroblast  = {prod_ery:.4f}")
    print(f"  recall_ig            = {prod_ig:.4f}")
except mlflow.exceptions.MlflowException:
    print("[ERREUR] Aucun modele @production trouve. Lancez training.py d'abord.")
    raise SystemExit(1)


# ── 2. Créer un run fictif avec métriques dégradées ───────────────────────────
# On force des métriques PIRES que @production pour déclencher le garde-fou
bad_f1  = max(0.0, prod_f1  - 0.05)   # -5% F1
bad_ery = max(0.0, prod_ery - 0.10)   # -10% recall erythroblast
bad_ig  = max(0.0, prod_ig  - 0.30)   # -30% recall ig  (dépasse la tolérance 2%)

print(f"\n[challenger simulé] Métriques dégradées artificiellement")
print(f"  macro_f1             = {bad_f1:.4f}  (prod - 0.05)")
print(f"  recall_erythroblast  = {bad_ery:.4f}  (prod - 0.10)")
print(f"  recall_ig            = {bad_ig:.4f}  (prod - 0.30)")

with mlflow.start_run(run_name="demo-challenger-degrade") as run:
    mlflow.log_params({
        "batch_size": 8, "epochs_head": 1, "epochs_full": 1,
        "run_type": "demo", "note": "run fictif pour demo garde-fou",
    })
    mlflow.log_metrics({
        "macro_f1": bad_f1,
        "recall_erythroblast": bad_ery,
        "recall_ig": bad_ig,
        "test_acc": 0.05,
        "best_val_acc": 0.05,
    })
    mlflow.set_tag("run_type", "demo")
    run_id = run.info.run_id

print(f"\n  Run fictif cree : {run_id[:8]}...")


# ── 3. Appliquer la logique de promotion ──────────────────────────────────────
print("\n" + "=" * 60)
print("  Application des gardes-fous")
print("=" * 60)

f1_ok  = bad_f1  >= prod_f1
ery_ok = bad_ery >= prod_ery - RECALL_TOLERANCE
ig_ok  = bad_ig  >= prod_ig  - RECALL_TOLERANCE

print(f"\n  macro_f1            : {bad_f1:.4f} >= {prod_f1:.4f}             -> {'OK' if f1_ok else 'KO'}")
print(f"  recall_erythroblast : {bad_ery:.4f} >= {prod_ery:.4f} - {RECALL_TOLERANCE} -> {'OK' if ery_ok else 'KO'}")
print(f"  recall_ig           : {bad_ig:.4f} >= {prod_ig:.4f} - {RECALL_TOLERANCE} -> {'OK' if ig_ok else 'KO'}")

print()
if f1_ok and ery_ok and ig_ok:
    print("  => PROMOTION @production  (tous les gardes-fous passes)")
else:
    reasons = []
    if not f1_ok:
        reasons.append(f"macro_f1 {bad_f1:.4f} < {prod_f1:.4f}")
    if not ery_ok:
        reasons.append(f"recall_erythroblast {bad_ery:.4f} < {prod_ery:.4f} - {RECALL_TOLERANCE}")
    if not ig_ok:
        reasons.append(f"recall_ig {bad_ig:.4f} < {prod_ig:.4f} - {RECALL_TOLERANCE}")
    print(f"  [KO] GARDE-FOU ACTIVE — reste @challenger")
    for r in reasons:
        print(f"       Raison : {r}")

    # Assigner @challenger à la version précédente pour rendre visible dans l'UI
    challenger_version = str(int(prod_mv.version) - 1)
    try:
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "challenger", challenger_version)
        print(f"\n  [MLflow] Alias @challenger assigne -> Version {challenger_version}")
        print(f"  [MLflow] Alias @production conserve -> Version {prod_mv.version}")
    except mlflow.exceptions.MlflowException as e:
        print(f"  [warn] Impossible d'assigner @challenger : {e}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    print(f"\n  Ouvrir : {tracking_uri} -> Model registry -> {MLFLOW_MODEL_NAME}")
    print(f"  @production = Version {prod_mv.version}  |  @challenger = Version {challenger_version}")

print("\n" + "=" * 60)
