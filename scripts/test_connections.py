"""Test de connexion à Supabase (PostgreSQL) et DagsHub.

Usage:
    python scripts/test_connections.py

Nécessite un fichier .env à la racine du projet (voir .env.example).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

PASS = True
FAIL = False


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _err(msg: str) -> None:
    print(f"  [KO] {msg}")


# ─────────────────────────────────────────────
# 1. Supabase (PostgreSQL)
# ─────────────────────────────────────────────

def test_supabase() -> bool:
    print("\n[1/2] Supabase (PostgreSQL)")
    try:
        import psycopg2
    except ImportError:
        _err("psycopg2 non installé — pip install psycopg2-binary")
        return FAIL

    required = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB",
                "SUPABASE_USER", "SUPABASE_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        _err(f"Variables manquantes dans .env : {', '.join(missing)}")
        return FAIL

    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("SUPABASE_HOST"),
            port=int(os.getenv("SUPABASE_PORT", 5432)),
            dbname=os.getenv("SUPABASE_DB"),
            user=os.getenv("SUPABASE_USER"),
            password=os.getenv("SUPABASE_PASSWORD"),
            connect_timeout=10,
            sslmode="require",
        )
        _ok("Connexion établie")

        cur = conn.cursor()

        # Insertion d'une ligne de test
        cur.execute("""
            INSERT INTO predictions (image_name, predicted_class, confidence, mlflow_run_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, ("test_connection.jpg", "neutrophil", 0.99, "run_test_connection"))
        row_id = cur.fetchone()[0]
        conn.commit()
        _ok(f"Insertion OK (id={row_id})")

        # Lecture pour vérifier
        cur.execute("SELECT image_name, predicted_class, confidence FROM predictions WHERE id = %s", (row_id,))
        row = cur.fetchone()
        _ok(f"Lecture OK → {row}")

        # Nettoyage de la ligne de test
        cur.execute("DELETE FROM predictions WHERE id = %s", (row_id,))
        conn.commit()
        _ok("Nettoyage de la ligne de test OK")

        cur.close()
        return PASS

    except Exception as exc:
        _err(f"Erreur : {exc}")
        if conn:
            conn.rollback()
        return FAIL
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────
# 2. DagsHub
# ─────────────────────────────────────────────

def test_dagshub() -> bool:
    print("\n[2/2] DagsHub (datalake images + modèles)")
    try:
        import requests
        import yaml
    except ImportError:
        _err("requests ou pyyaml non installé")
        return FAIL

    user = os.getenv("DAGSHUB_USER", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo = os.getenv("DAGSHUB_REPO", "Bloodcells-project")

    if not user or not token:
        _err("DAGSHUB_USER et DAGSHUB_TOKEN manquants dans .env")
        return FAIL

    base = f"https://dagshub.com/{user}/{repo}"
    auth = (user, token)

    # Vérification manifest DVC modèle
    try:
        r = requests.get(f"{base}/raw/main/Models.dvc", auth=auth, timeout=10)
        r.raise_for_status()
        manifest = yaml.safe_load(r.text)
        md5 = manifest["outs"][0]["md5"]
        size_mb = manifest["outs"][0].get("size", 0) / 1024 / 1024
        _ok(f"Manifest Models.dvc OK (md5={md5[:8]}…, taille≈{size_mb:.1f} MB)")
    except Exception as exc:
        _err(f"Manifest Models.dvc échoué : {exc}")
        return FAIL

    # Vérification manifest DVC images
    try:
        r = requests.get(f"{base}/raw/main/Source_100.dvc", auth=auth, timeout=10)
        r.raise_for_status()
        manifest = yaml.safe_load(r.text)
        md5 = manifest["outs"][0]["md5"]
        _ok(f"Manifest Source_100.dvc OK (md5={md5[:8]}…)")
    except Exception as exc:
        _err(f"Manifest Source_100.dvc échoué : {exc}")
        return FAIL

    return PASS


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Test des connexions infrastructure")
    print("=" * 50)

    results = {
        "Supabase": test_supabase(),
        "DagsHub": test_dagshub(),
    }

    print("\n" + "=" * 50)
    all_ok = all(results.values())
    for name, ok in results.items():
        status = "OK" if ok else "ECHEC"
        print(f"  {name:<12} {status}")
    print("=" * 50)

    sys.exit(0 if all_ok else 1)
