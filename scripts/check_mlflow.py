"""Test complet du stack MLflow du projet blood-cells.

Vérifie :
  1. Connexion au serveur MLflow
  2. Création d'expérience et d'un run fictif
  3. Log de params, métriques et tags
  4. Model registry : enregistrement d'un modèle, aliases @production / @challenger
  5. Garde-fous de promotion (logique extraite de training.py)
  6. Nettoyage des données de test

Usage :
    python scripts/test_mlflow.py
"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
TEST_EXPERIMENT = "test-mlflow-blood-cells"
TEST_MODEL_NAME = "test-blood-cell-densenet121"
RECALL_TOLERANCE = 0.02

PASS = True
FAIL = False


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _err(msg: str) -> None:
    print(f"  [KO] {msg}")


# ─────────────────────────────────────────────
# 1. Connexion au serveur MLflow
# ─────────────────────────────────────────────

def test_connection() -> bool:
    print(f"\n[1/5] Connexion MLflow ({TRACKING_URI})")
    try:
        import requests
        r = requests.get(f"{TRACKING_URI}/health", timeout=5)
        r.raise_for_status()
        _ok(f"Serveur répond : {r.text.strip()}")
    except Exception as exc:
        _err(f"Serveur inaccessible : {exc}")
        return FAIL

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        experiments = client.search_experiments()
        _ok(f"{len(experiments)} expérience(s) existante(s) : {[e.name for e in experiments]}")
    except Exception as exc:
        _err(f"MlflowClient échoue : {exc}")
        return FAIL

    return PASS


# ─────────────────────────────────────────────
# 2. Création d'expérience + run fictif
# ─────────────────────────────────────────────

def test_run_logging() -> tuple[bool, str]:
    print("\n[2/5] Création d'un run de test")
    run_id = None
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(TEST_EXPERIMENT)

        with mlflow.start_run(run_name="test-run-auto") as run:
            run_id = run.info.run_id

            mlflow.log_params({
                "batch_size": 32,
                "epochs_head": 5,
                "epochs_full": 10,
                "model": "densenet121",
                "num_classes": 8,
                "device": "cpu",
            })
            _ok("log_params OK")

            for step in range(3):
                mlflow.log_metrics({
                    "train_loss": 1.0 - step * 0.2,
                    "val_loss":   1.1 - step * 0.2,
                    "train_acc":  0.5 + step * 0.1,
                    "val_acc":    0.48 + step * 0.1,
                }, step=step)
            _ok("log_metrics (3 steps) OK")

            test_metrics = {
                "best_val_acc":        0.91,
                "test_acc":            0.90,
                "macro_f1":            0.89,
                "weighted_f1":         0.90,
                "precision_macro":     0.89,
                "recall_macro":        0.89,
                "recall_erythroblast": 0.85,
                "recall_ig":           0.82,
            }
            mlflow.log_metrics(test_metrics)
            _ok(f"log_metrics (test) OK : macro_f1={test_metrics['macro_f1']}")

            mlflow.set_tags({"git_commit": "test_script", "run_type": "test"})
            _ok("set_tags OK")

        _ok(f"Run terminé : {run_id[:8]}...")
        return PASS, run_id

    except Exception as exc:
        _err(f"Erreur : {exc}")
        return FAIL, run_id


# ─────────────────────────────────────────────
# 3. Lecture du run loggué
# ─────────────────────────────────────────────

def test_run_read(run_id: str) -> bool:
    print(f"\n[3/5] Lecture du run {run_id[:8]}...")
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        assert params.get("model") == "densenet121", "param model manquant"
        _ok(f"Params OK (batch_size={params.get('batch_size')})")

        assert "macro_f1" in metrics, "métrique macro_f1 manquante"
        assert "recall_erythroblast" in metrics, "métrique recall_erythroblast manquante"
        _ok(f"Métriques OK (macro_f1={metrics['macro_f1']:.4f}, recall_ig={metrics['recall_ig']:.4f})")

        history = client.get_metric_history(run_id, "val_acc")
        assert len(history) == 3, f"Attendu 3 steps val_acc, reçu {len(history)}"
        _ok(f"Historique val_acc OK ({len(history)} steps)")

        return PASS

    except Exception as exc:
        _err(f"Erreur : {exc}")
        return FAIL


# ─────────────────────────────────────────────
# 4. Model Registry + aliases
# ─────────────────────────────────────────────

def test_registry(run_id: str) -> bool:
    print("\n[4/5] Model Registry")
    client = MlflowClient()

    try:
        try:
            client.create_registered_model(TEST_MODEL_NAME)
            _ok(f"Registered model '{TEST_MODEL_NAME}' créé")
        except mlflow.exceptions.MlflowException:
            _ok(f"Registered model '{TEST_MODEL_NAME}' déjà existant")
    except Exception as exc:
        _err(f"create_registered_model : {exc}")
        return FAIL

    v1 = None
    try:
        mv = mlflow.register_model(
            model_uri=f"runs:/{run_id}/dummy",
            name=TEST_MODEL_NAME,
        )
        v1 = mv.version
        _ok(f"Version {v1} enregistrée")
    except Exception as exc:
        _err(f"register_model échoue (pas d'artefact réel) : {exc}")
        try:
            mv = client.create_model_version(
                name=TEST_MODEL_NAME,
                source=f"runs:/{run_id}",
                run_id=run_id,
            )
            v1 = mv.version
            _ok(f"Version {v1} créée via create_model_version")
        except Exception as exc2:
            _err(f"create_model_version échoue : {exc2}")
            return FAIL

    try:
        client.set_registered_model_alias(TEST_MODEL_NAME, "production", v1)
        _ok(f"Alias @production -> Version {v1}")
    except Exception as exc:
        _err(f"set alias @production : {exc}")
        return FAIL

    try:
        prod_mv = client.get_model_version_by_alias(TEST_MODEL_NAME, "production")
        assert prod_mv.version == v1, f"Alias pointe version {prod_mv.version} ≠ {v1}"
        _ok(f"Lecture @production OK -> Version {prod_mv.version}")
    except Exception as exc:
        _err(f"get_model_version_by_alias : {exc}")
        return FAIL

    return PASS


# ─────────────────────────────────────────────
# 5. Garde-fous de promotion
# ─────────────────────────────────────────────

def test_garde_fous() -> bool:
    print("\n[5/5] Garde-fous de promotion")
    client = MlflowClient()

    try:
        prod_mv = client.get_model_version_by_alias(TEST_MODEL_NAME, "production")
        prod_run = client.get_run(prod_mv.run_id)
        pm = prod_run.data.metrics
        prod_f1  = float(pm.get("macro_f1", 0.89))
        prod_ery = float(pm.get("recall_erythroblast", 0.85))
        prod_ig  = float(pm.get("recall_ig", 0.82))
        _ok(f"Production lue : f1={prod_f1:.4f}, ery={prod_ery:.4f}, ig={prod_ig:.4f}")
    except Exception as exc:
        _err(f"Impossible de lire @production : {exc}")
        return FAIL

    # Scénario A : meilleur modèle -> doit être promu
    good = {"macro_f1": prod_f1 + 0.01, "recall_erythroblast": prod_ery + 0.01, "recall_ig": prod_ig + 0.01}
    if _apply_garde_fous(good, prod_f1, prod_ery, prod_ig):
        _ok("Scénario A (meilleur modèle) -> PROMOTION attendue : OK")
    else:
        _err("Scénario A (meilleur modèle) -> PROMOTION attendue mais bloquée")
        return FAIL

    # Scénario B : recall_ig trop dégradé -> doit être bloqué
    bad = {"macro_f1": prod_f1 + 0.01, "recall_erythroblast": prod_ery - 0.01, "recall_ig": prod_ig - 0.05}
    if not _apply_garde_fous(bad, prod_f1, prod_ery, prod_ig):
        _ok("Scénario B (recall_ig -5%) -> BLOCAGE attendu : OK")
    else:
        _err("Scénario B (recall_ig -5%) -> BLOCAGE attendu mais promu")
        return FAIL

    # Scénario C : macro_f1 inférieur -> doit être bloqué
    degraded = {"macro_f1": prod_f1 - 0.01, "recall_erythroblast": prod_ery, "recall_ig": prod_ig}
    if not _apply_garde_fous(degraded, prod_f1, prod_ery, prod_ig):
        _ok("Scénario C (f1 dégradé) -> BLOCAGE attendu : OK")
    else:
        _err("Scénario C (f1 dégradé) -> BLOCAGE attendu mais promu")
        return FAIL

    return PASS


def _apply_garde_fous(new: dict, prod_f1: float, prod_ery: float, prod_ig: float) -> bool:
    f1_ok  = new["macro_f1"]            >= prod_f1
    ery_ok = new["recall_erythroblast"] >= prod_ery - RECALL_TOLERANCE
    ig_ok  = new["recall_ig"]           >= prod_ig  - RECALL_TOLERANCE
    return f1_ok and ery_ok and ig_ok


# ─────────────────────────────────────────────
# Nettoyage
# ─────────────────────────────────────────────

def cleanup(run_id: str) -> None:
    print("\n[cleanup] Suppression des données de test")
    client = MlflowClient()

    for alias in ["production", "challenger"]:
        try:
            client.delete_registered_model_alias(TEST_MODEL_NAME, alias)
        except Exception:
            pass

    try:
        client.delete_registered_model(TEST_MODEL_NAME)
        _ok(f"Registered model '{TEST_MODEL_NAME}' supprimé")
    except Exception as exc:
        _ok(f"(registered model non supprimé : {exc})")

    try:
        exp = client.get_experiment_by_name(TEST_EXPERIMENT)
        if exp:
            client.delete_experiment(exp.experiment_id)
            _ok(f"Expérience '{TEST_EXPERIMENT}' supprimée")
    except Exception as exc:
        _ok(f"(expérience non supprimée : {exc})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Test MLflow — blood-cells project")
    print(f"  URI : {TRACKING_URI}")
    print("=" * 60)

    run_id = None
    results = {}

    results["1. Connexion"] = test_connection()

    if results["1. Connexion"]:
        ok, run_id = test_run_logging()
        results["2. Log run"] = ok
    else:
        results["2. Log run"] = FAIL
        print("  (Skip : serveur non accessible)")

    if run_id:
        results["3. Lecture run"] = test_run_read(run_id)
        results["4. Registry"]    = test_registry(run_id)
        results["5. Garde-fous"]  = test_garde_fous()
        cleanup(run_id)
    else:
        for k in ["3. Lecture run", "4. Registry", "5. Garde-fous"]:
            results[k] = FAIL

    print("\n" + "=" * 60)
    print("  Résultats")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK  " if ok else "ECHEC"
        print(f"  {name:<22} {status}")
        if not ok:
            all_ok = False

    print("=" * 60)
    print(f"  {'TOUS LES TESTS PASSES' if all_ok else 'CERTAINS TESTS ONT ECHOUE'}")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)
