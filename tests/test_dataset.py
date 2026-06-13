"""Tests de chargement et de structure du dataset."""
import pytest
from pathlib import Path
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]


def test_fixtures_exist():
    assert FIXTURES.exists(), "Le dossier fixtures/ est absent"
    images = list(FIXTURES.glob("*.jpg"))
    assert len(images) >= 16, f"Attendu ≥16 images, trouvé {len(images)}"


def test_fixtures_cover_all_classes():
    images = list(FIXTURES.glob("*.jpg"))
    found = {img.name.split("_")[0] for img in images}
    for cls in CLASSES:
        assert cls in found, f"Classe manquante dans les fixtures : {cls}"


def test_images_readable():
    for img_path in FIXTURES.glob("*.jpg"):
        img = Image.open(img_path)
        assert img.size[0] > 0 and img.size[1] > 0
        img.close()


def test_images_rgb():
    for img_path in FIXTURES.glob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        assert img.mode == "RGB"
        img.close()
