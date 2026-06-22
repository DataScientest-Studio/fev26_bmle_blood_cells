"""
Découpe l'archive TCIA (TIFF, "autre instrument") en lots de 200 images
pour simuler des arrivées incrémentales (générations V2, V3...).

Échantillonnage proportionnel/naturel : toutes les images des 7 classes
disponibles (pas de plaquettes dans cette archive) sont mélangées puis
découpées en lots de taille fixe, sans forcer l'équilibre par classe —
certains lots auront peu ou pas de classes rares (basophil, erythroblast).

Usage :
    python -m scripts.split_tiff_batches
"""

import os
import random
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")

if not os.getenv("CANCER_ARCHIVE_DIR"):
    raise EnvironmentError("CANCER_ARCHIVE_DIR doit être défini dans ton .env local.")

SRC = Path(os.environ["CANCER_ARCHIVE_DIR"])
OUT_DIR = ROOT / "data" / "tiff_batches"
BATCH_SIZE = 200
SEED = 42

# Même mapping que CancerImagingArchive/regroup_classes.py (taxonomie Acevedo)
MAPPING = {
    "basophil": ["BAS"],
    "eosinophil": ["EOS"],
    "erythroblast": ["EBO"],
    "lymphocyte": ["LYA", "LYT"],
    "monocyte": ["MOB", "MON"],
    "neutrophil": ["NGS"],
    "ig": ["KSC", "MMZ", "MYB", "MYO", "NGB", "PMB", "PMO"],
}


def main():
    pool = []  # liste de (path, classe)
    per_class_total = {}
    for target_class, src_codes in MAPPING.items():
        files = []
        for code in src_codes:
            src_dir = SRC / code
            if not src_dir.is_dir():
                print(f"  [SKIP] {code} introuvable")
                continue
            files.extend(p for p in src_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
        per_class_total[target_class] = len(files)
        pool.extend((p, target_class) for p in files)

    print("Pool source :")
    for cls, n in per_class_total.items():
        print(f"  {cls:<15} {n:>6}")
    print(f"  {'TOTAL':<15} {len(pool):>6}\n")

    random.Random(SEED).shuffle(pool)

    n_batches = len(pool) // BATCH_SIZE
    leftover = len(pool) - n_batches * BATCH_SIZE
    print(f"{n_batches} lots complets de {BATCH_SIZE} images, {leftover} images restantes (non utilisées).\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(n_batches):
        batch_items = pool[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        batch_dir = OUT_DIR / f"batch_{batch_idx + 1:03d}"

        counts = {}
        for src_path, cls in batch_items:
            cls_dir = batch_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            link = cls_dir / src_path.name
            if not link.exists():
                link.symlink_to(src_path.resolve())
            counts[cls] = counts.get(cls, 0) + 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            detail = ", ".join(f"{c}={n}" for c, n in sorted(counts.items()))
            print(f"  batch_{batch_idx + 1:03d} : {detail}")

    print(f"\n{n_batches} lots créés dans {OUT_DIR}")


if __name__ == "__main__":
    main()
