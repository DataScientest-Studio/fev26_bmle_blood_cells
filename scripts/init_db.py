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
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version    TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_brightness  FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS std_brightness   FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_r           FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_g           FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS mean_b           FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS image_width      INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS image_height     INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS patient_id       INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS patient_name     TEXT;
"""

# Désaccord médecin sur une prédiction — relié à predictions.id
SQL_PREDICTION_FEEDBACK = """
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id              SERIAL PRIMARY KEY,
    prediction_id   INTEGER REFERENCES predictions(id),
    agrees          BOOLEAN NOT NULL,
    corrected_class TEXT,
    comment         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
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

# Monitoring entraînement : consommation ressources (1 ligne par run/fold)
# La table training_runs existe déjà (créée via l'API : triggered_by, data_dir,
# epochs_head, epochs_full, val_acc, test_acc, status, started_at, mlflow_run_id).
# On l'étend plutôt que d'en créer une autre, pour garder un seul historique
# de runs d'entraînement.
SQL_TRAINING_RUNS = """
CREATE TABLE IF NOT EXISTS training_runs (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'running'
);
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS model_name           TEXT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS generation           TEXT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS fold                 INTEGER;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS device               TEXT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS ended_at             TIMESTAMPTZ;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS duration_seconds     FLOAT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS cpu_percent_avg      FLOAT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS ram_used_mb_avg      FLOAT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS gpu_name             TEXT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS gpu_util_percent_avg FLOAT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS gpu_mem_used_mb_avg  FLOAT;
"""

# Monitoring qualité : % par classe (1 ligne par classe par run/fold)
SQL_CLASS_METRICS = """
CREATE TABLE IF NOT EXISTS class_metrics (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    generation      TEXT,
    fold            INTEGER,
    class_name      TEXT NOT NULL,
    precision       FLOAT,
    recall          FLOAT,
    f1              FLOAT,
    support         INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Données de référence pour le monitoring de drift (stats image par classe)
SQL_REFERENCE_FEATURES = """
CREATE TABLE IF NOT EXISTS reference_features (
    id              SERIAL PRIMARY KEY,
    class_name      TEXT NOT NULL,
    mean_brightness FLOAT,
    std_brightness  FLOAT,
    mean_r          FLOAT,
    mean_g          FLOAT,
    mean_b          FLOAT,
    image_width     INTEGER,
    image_height    INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# Rapports de drift Evidently (IVDR 2017/746)
SQL_DRIFT_REPORTS = """
CREATE TABLE IF NOT EXISTS drift_reports (
    id                   SERIAL PRIMARY KEY,
    model_version        TEXT,
    n_reference          INTEGER,
    n_current            INTEGER,
    data_drift_detected  BOOLEAN,
    data_drift_score     FLOAT,
    pred_drift_detected  BOOLEAN,
    pred_drift_score     FLOAT,
    model_drift_score    FLOAT,
    n_drifted_features   INTEGER,
    metrics_json         JSONB,
    report_html          TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
"""

# Matrice de confusion complète (1 ligne par run/fold, JSON)
SQL_CONFUSION_MATRICES = """
CREATE TABLE IF NOT EXISTS confusion_matrices (
    id              SERIAL PRIMARY KEY,
    mlflow_run_id   TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    generation      TEXT,
    fold            INTEGER,
    class_order     JSONB NOT NULL,
    matrix          JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
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
    cur.execute(SQL_PREDICTION_FEEDBACK)
    cur.execute(SQL_DATASET_IMAGES)
    cur.execute(SQL_TRAINING_RUNS)
    cur.execute(SQL_CLASS_METRICS)
    cur.execute(SQL_CONFUSION_MATRICES)
    cur.execute(SQL_REFERENCE_FEATURES)
    cur.execute(SQL_DRIFT_REPORTS)
    conn.commit()
    cur.close()
    print("  [OK] Tables créées (ou déjà existantes) : predictions, prediction_feedback, "
          "dataset_images, training_runs, class_metrics, confusion_matrices, "
          "reference_features, drift_reports")


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

    print("\nCalcul des stats de référence pour le drift monitoring...")
    _populate_reference_features(conn, data_dir, rows)


def _populate_reference_features(conn, data_dir: Path, rows: list[dict]) -> None:
    """Calcule les stats image (brightness, RGB) par classe et les insère dans reference_features."""
    try:
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        print("  [KO] Pillow/numpy non installés — reference_features non peuplée")
        return

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM reference_features")
    if cur.fetchone()[0] > 0:
        print("  [OK] reference_features déjà peuplée — ignorée")
        cur.close()
        return

    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["true_class"], []).append(r)

    inserted = 0
    for cls, items in by_class.items():
        cls_dir = data_dir / cls
        stats_list = []
        for r in items:
            img_path = cls_dir / r["image_name"]
            if not img_path.exists():
                continue
            try:
                img = PILImage.open(img_path).convert("RGB")
                arr = np.array(img, dtype=np.float32)
                gray = arr.mean(axis=2)
                stats_list.append({
                    "mean_brightness": float(gray.mean()),
                    "std_brightness":  float(gray.std()),
                    "mean_r": float(arr[:, :, 0].mean()),
                    "mean_g": float(arr[:, :, 1].mean()),
                    "mean_b": float(arr[:, :, 2].mean()),
                    "image_width":  img.size[0],
                    "image_height": img.size[1],
                })
            except Exception:
                continue

        if not stats_list:
            continue

        cur.execute("""
            INSERT INTO reference_features
                (class_name, mean_brightness, std_brightness,
                 mean_r, mean_g, mean_b, image_width, image_height)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            cls,
            float(np.mean([s["mean_brightness"] for s in stats_list])),
            float(np.mean([s["std_brightness"]  for s in stats_list])),
            float(np.mean([s["mean_r"]          for s in stats_list])),
            float(np.mean([s["mean_g"]          for s in stats_list])),
            float(np.mean([s["mean_b"]          for s in stats_list])),
            int(np.median([s["image_width"]     for s in stats_list])),
            int(np.median([s["image_height"]    for s in stats_list])),
        ))
        inserted += 1

    conn.commit()
    cur.close()
    print(f"  [OK] reference_features peuplée : {inserted} classe(s) insérée(s)")


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
