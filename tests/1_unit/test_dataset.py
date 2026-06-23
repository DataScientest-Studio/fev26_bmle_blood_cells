"""Tests unitaires — structure et contenu du dataset de fixtures.

Ces tests vérifient que les images de test couvrent bien les 8 classes
et sont lisibles. Ils tournent sans modèle ML ni dépendance externe.
"""
import pytest
from pathlib import Path
from PIL import Image

CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]


@pytest.mark.unit
def test_fixtures_exist(fixtures_dir: Path):
    assert fixtures_dir.exists(), "Le dossier fixtures/ est absent"
    images = list(fixtures_dir.glob("*.jpg"))
    assert len(images) >= 16, f"Attendu ≥16 images, trouvé {len(images)}"


@pytest.mark.unit
def test_fixtures_cover_all_classes(fixtures_dir: Path):
    images = list(fixtures_dir.glob("*.jpg"))
    found = {img.name.split("_")[0] for img in images}
    for cls in CLASSES:
        assert cls in found, f"Classe manquante dans les fixtures : {cls}"


@pytest.mark.unit
def test_images_readable(fixtures_dir: Path):
    for img_path in fixtures_dir.glob("*.jpg"):
        img = Image.open(img_path)
        assert img.size[0] > 0 and img.size[1] > 0
        img.close()


@pytest.mark.unit
def test_images_rgb(fixtures_dir: Path):
    for img_path in fixtures_dir.glob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        assert img.mode == "RGB"
        img.close()
