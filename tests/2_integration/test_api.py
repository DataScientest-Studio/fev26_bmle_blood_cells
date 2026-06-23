"""
Tests d'intégration — FastAPI Blood Cell Classifier API

Stratégie : le modèle DenseNet-121 n'est pas disponible en CI.
On injecte un faux modèle (poids aléatoires, architecture identique) via
monkeypatch sur les variables globales de l'API, ce qui garantit que les tests
couvrent la logique HTTP et la sérialisation sans dépendre d'artefacts ML.

Les fixtures fake_model et valid_image_bytes sont définies dans tests/conftest.py
et disponibles automatiquement ici.
"""
import io
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from src.serving.api import app, CLASSES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Fixture client — spécifique aux tests API
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(fake_model):
    """TestClient avec modèle mocké et device CPU forcé.

    CPU forcé pour être reproductible en CI (pas de GPU/MPS).
    Module-scoped : une seule instance pour tous les tests du fichier.
    """
    import src.serving.api as api_module
    api_module.model = fake_model
    api_module.model_device = torch.device("cpu")
    api_module.DEVICE = torch.device("cpu")
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests routes Info
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.integration
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "classes" in body
    assert len(body["classes"]) == NUM_CLASSES
    assert body["version"] == "1.0.0"


@pytest.mark.integration
def test_classes(client):
    r = client.get("/classes")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == NUM_CLASSES
    assert body["classes"] == CLASSES


@pytest.mark.integration
def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert "num_classes" in body
    assert body["num_classes"] == NUM_CLASSES
    assert "device" in body


# ---------------------------------------------------------------------------
# Tests route /predict
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_predict_valid_image(client, valid_image_bytes):
    r = client.post(
        "/predict",
        files={"file": ("cell.jpg", valid_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()

    assert "predicted_class" in body
    assert body["predicted_class"] in CLASSES

    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0

    assert "all_probas" in body
    assert set(body["all_probas"].keys()) == set(CLASSES)
    total = sum(body["all_probas"].values())
    assert abs(total - 1.0) < 1e-3

    assert "top3" in body
    assert len(body["top3"]) == 3

    assert "is_critical" in body
    assert isinstance(body["is_critical"], bool)

    assert "inference_ms" in body
    assert body["inference_ms"] >= 0


@pytest.mark.integration
def test_predict_is_critical_flag(client, fake_model, valid_image_bytes):
    """is_critical doit être True si la classe prédite est Erythroblast ou IG."""
    import src.serving.api as api_module

    critical_classes = {"Erythroblast", "IG"}

    for target_class in critical_classes:
        target_idx = CLASSES.index(target_class)

        class ForcedModel(nn.Module):
            def forward(self, x):
                logits = torch.full((x.size(0), NUM_CLASSES), -1e9)
                logits[:, target_idx] = 1e9
                return logits

        api_module.model = ForcedModel()
        r = client.post(
            "/predict",
            files={"file": ("cell.jpg", valid_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_class"] == target_class
        assert body["is_critical"] is True

    api_module.model = fake_model


@pytest.mark.integration
def test_predict_missing_file(client):
    """Sans fichier → FastAPI doit retourner 422 Unprocessable Entity."""
    r = client.post("/predict")
    assert r.status_code == 422


@pytest.mark.integration
def test_predict_invalid_file(client):
    """Fichier non-image → l'API retourne une réponse d'erreur sans crasher."""
    r = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"this is not an image", "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" in body or "predicted_class" in body


@pytest.mark.integration
def test_predict_small_image(client):
    """Image 1×1 — le resize doit absorber sans erreur."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post(
        "/predict",
        files={"file": ("tiny.jpg", buf.read(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "predicted_class" in body or "error" in body


# ---------------------------------------------------------------------------
# Tests authentification API Key
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_predict_rejected_with_wrong_key(valid_image_bytes, monkeypatch):
    """Quand API_SECRET_KEY est définie, une mauvaise clé doit retourner 401."""
    import src.serving.api as api_module
    monkeypatch.setattr(api_module, "_API_SECRET_KEY", "correct-secret")
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            files={"file": ("cell.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "wrong-key"},
        )
    assert r.status_code == 401


@pytest.mark.integration
def test_predict_rejected_without_key(valid_image_bytes, monkeypatch):
    """Quand API_SECRET_KEY est définie, l'absence de clé doit retourner 401."""
    import src.serving.api as api_module
    monkeypatch.setattr(api_module, "_API_SECRET_KEY", "correct-secret")
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            files={"file": ("cell.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert r.status_code == 401


@pytest.mark.integration
def test_predict_accepted_with_correct_key(fake_model, valid_image_bytes, monkeypatch):
    """Quand API_SECRET_KEY est définie, la bonne clé doit être acceptée."""
    import src.serving.api as api_module
    monkeypatch.setattr(api_module, "_API_SECRET_KEY", "correct-secret")
    monkeypatch.setattr(api_module, "model", fake_model)
    monkeypatch.setattr(api_module, "DEVICE", torch.device("cpu"))
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            files={"file": ("cell.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "correct-secret"},
        )
    assert r.status_code == 200
    assert "predicted_class" in r.json()
