"""Simule 5 jours d'acquisition d'images : copie 100 images/classe distinctes
par run depuis data/raw/ vers data/runs/run{1..5}/<classe>/.

Usage :
    python scripts/make_acquisition_runs.py
"""

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data" / "raw"
DEST_DIR = ROOT / "data" / "runs"

CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

N_RUNS = 5
N_PER_CLASS_PER_RUN = 100
SEED = 42


def main() -> None:
    rng = random.Random(SEED)

    for cls in CLASSES:
        cls_dir = SOURCE_DIR / cls
        images = sorted(p.name for p in cls_dir.iterdir() if p.is_file())
        needed = N_RUNS * N_PER_CLASS_PER_RUN
        if len(images) < needed:
            raise ValueError(
                f"{cls} : {len(images)} images disponibles, {needed} nécessaires"
            )

        rng.shuffle(images)
        chosen = images[:needed]

        for run_idx in range(N_RUNS):
            run_name = f"run{run_idx + 1}"
            batch = chosen[run_idx * N_PER_CLASS_PER_RUN: (run_idx + 1) * N_PER_CLASS_PER_RUN]
            out_dir = DEST_DIR / run_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in batch:
                shutil.copy2(cls_dir / name, out_dir / name)

        print(f"{cls:15s} : {needed} images réparties sur {N_RUNS} runs")

    print(f"\nTerminé — voir {DEST_DIR}")


if __name__ == "__main__":
    main()
