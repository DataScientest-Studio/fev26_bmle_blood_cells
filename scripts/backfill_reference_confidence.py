"""
Backfill one-shot : ajoute un échantillon frais d'images de data/raw dans
reference_features avec un score de confiance réel (calculé par le modèle
@production), pour donner immédiatement au drift monitoring une vraie
baseline de confidence au lieu du proxy fixe 0.95.

Additif uniquement — n'écrase ni ne supprime les lignes existantes.

Usage :
    python scripts/backfill_reference_confidence.py [--data-dir data/raw] [--per-class 150]
"""

from __future__ import annotations

import argparse
import random
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--per-class", type=int, default=150, metavar="N")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir

    import numpy as np
    import torch
    from PIL import Image as PILImage

    from src.auth.db import get_connection
    from src.serving.api import load_model, transform as api_transform, DEVICE as API_DEVICE

    print("Chargement du modèle @production (MLflow Registry)...")
    model = load_model()

    def infer_confidence(img):
        tensor = api_transform(img).unsqueeze(0).to(API_DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
        return float(probs.max().item())

    conn = get_connection()
    cur = conn.cursor()

    rng = random.Random(42)
    inserted = skipped = 0
    batch = []
    BATCH = 100

    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"  [warn] dossier introuvable : {cls_dir}")
            continue

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        sample = rng.sample(images, min(args.per_class, len(images)))
        print(f"  {cls}: {len(sample)} images échantillonnées")

        for img_path in sample:
            try:
                img = PILImage.open(img_path).convert("RGB")
                arr = np.array(img, dtype=np.float32)
                gray = arr.mean(axis=2)
                confidence = infer_confidence(img)
                batch.append((
                    cls,
                    float(gray.mean()), float(gray.std()),
                    float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean()),
                    int(img.size[0]), int(img.size[1]),
                    confidence,
                ))
            except Exception:
                skipped += 1
                continue

            if len(batch) >= BATCH:
                cur.executemany("""
                    INSERT INTO reference_features
                        (class_name, mean_brightness, std_brightness,
                         mean_r, mean_g, mean_b, image_width, image_height, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, batch)
                inserted += len(batch)
                batch = []
                print(f"    {inserted} images insérées...", flush=True)

    if batch:
        cur.executemany("""
            INSERT INTO reference_features
                (class_name, mean_brightness, std_brightness,
                 mean_r, mean_g, mean_b, image_width, image_height, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, batch)
        inserted += len(batch)

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n[OK] Backfill terminé : {inserted} images insérées (confidence réelle), {skipped} ignorées")


if __name__ == "__main__":
    main()
