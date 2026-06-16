"""
Initialisation de la base Supabase — script one-shot.

Actions :
  1. Crée la table `predictions`  (logs des appels API /predict)
  2. Crée la table `dataset_images` (métadonnées du dataset local)
  3. Peuple `dataset_images` depuis un dossier local (data/Source_100 ou data/raw)

Usage :
    python scripts/init_db.py                          # tables seules
    python scripts/init_db.py --populate data/Source_100
    python scripts/init_db.py --populate data/raw
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

CLASSES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


# ── SQL ───────────────────────────────────────────────────────────────────────

SQL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    image_name      TEXT,
    predicted_class TEXT,
    confidence      FLOAT,
    mlflow_run_id   TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

SQL_DATASET_IMAGES = """
CREATE TABLE IF NOT EXISTS dataset_images (
    id          SERIAL PRIMARY KEY,
    image_name  TEXT    NOT NULL,
    true_class  TEXT    NOT NULL,
    split       TEXT    NOT NULL,  -- 'train', 'val', 'test'
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (image_name)
);
"""


# ── Connexion ─────────────────────────────────────────────────────────────────

def _connect():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        port=int(os.getenv("SUPABASE_PORT", 5432)),
        dbname=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        connect_timeout=10,
        sslmode="require",
    )


# ── Étape 1 & 2 : créer les tables ───────────────────────────────────────────

def create_tables(conn) -> None:
    cur = conn.cursor()
    cur.execute(SQL_PREDICTIONS)
    cur.execute(SQL_DATASET_IMAGES)
    conn.commit()
    cur.close()
    print("  [OK] Tables 'predictions' et 'dataset_images' créées (ou déjà existantes)")


# ── Étape 3 : peupler dataset_images ─────────────────────────────────────────

def _collect_images(data_dir: Path) -> list[dict]:
    """Parcourt data_dir/classe/*.jpg et retourne la liste des métadonnées."""
    rows = []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({"image_name": p.name, "true_class": cls})
    return rows


def _assign_splits(rows: list[dict], seed: int = 42) -> list[dict]:
    """Affecte train/val/test par classe (70/15/15) de façon reproductible."""
    import random
    rng = random.Random(seed)

    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["true_class"], []).append(r)

    result = []
    for cls, items in by_class.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        for i, item in enumerate(shuffled):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            result.append({**item, "split": split})
    return result


def populate_dataset(conn, data_dir: Path) -> None:
    rows = _collect_images(data_dir)
    if not rows:
        print(f"  [KO] Aucune image trouvée dans {data_dir}")
        return

    rows = _assign_splits(rows)

    cur = conn.cursor()
    inserted = skipped = 0
    for r in rows:
        cur.execute("""
            INSERT INTO dataset_images (image_name, true_class, split)
            VALUES (%s, %s, %s)
            ON CONFLICT (image_name) DO NOTHING
        """, (r["image_name"], r["true_class"], r["split"]))
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    cur.close()

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"  [OK] {inserted} lignes insérées, {skipped} ignorées (déjà présentes)")
    print(f"       train={counts['train']}  val={counts['val']}  test={counts['test']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise les tables Supabase")
    parser.add_argument(
        "--populate", metavar="DATA_DIR",
        help="Dossier contenant les sous-dossiers par classe (ex: data/Source_100)"
    )
    args = parser.parse_args()

    required = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB",
                "SUPABASE_USER", "SUPABASE_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[KO] Variables manquantes dans .env : {', '.join(missing)}")
        sys.exit(1)

    try:
        import psycopg2  # noqa: F401
    except ImportError:
        print("[KO] psycopg2 non installé — pip install psycopg2-binary")
        sys.exit(1)

    print("Connexion à Supabase...")
    try:
        conn = _connect()
    except Exception as e:
        print(f"[KO] Connexion échouée : {e}")
        sys.exit(1)
    print("  [OK] Connexion établie")

    print("\nCréation des tables...")
    create_tables(conn)

    if args.populate:
        data_dir = Path(args.populate)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        print(f"\nPopulation de dataset_images depuis {data_dir} ...")
        populate_dataset(conn, data_dir)

    conn.close()
    print("\nTerminé.")


if __name__ == "__main__":
    main()
