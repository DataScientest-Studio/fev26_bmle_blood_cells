"""
Fixtures partagées entre tous les tests du projet.

conftest.py est automatiquement chargé par pytest pour tous les sous-dossiers
(tests/unit/ et tests/integration/). Les fixtures définies ici sont disponibles
sans import dans chaque fichier de test.
"""
import io
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import densenet121

from src.serving.api import NUM_CLASSES

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Chemin vers les images de test partagées."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def fake_model() -> nn.Module:
    """DenseNet-121 avec poids aléatoires — même architecture que le vrai modèle.

    Session-scoped : instancié une seule fois pour toute la suite de tests.
    Aucune dépendance au modèle entraîné (.pth) ni à MLflow.
    """
    m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture()
def valid_image_bytes() -> bytes:
    """Image RGB 224×224 encodée en JPEG — simule une image de cellule sanguine."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 64, 32)).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()
