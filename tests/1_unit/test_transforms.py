"""Tests unitaires — transformations d'images (resize, ToTensor, normalize).

Ces tests vérifient les fonctions de prétraitement PyTorch utilisées avant
l'inférence DenseNet-121. Aucune dépendance au modèle entraîné.
"""
import pytest
import torch
from pathlib import Path
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _sample_image(fixtures_dir: Path) -> Image.Image:
    images = list(fixtures_dir.glob("*.jpg"))
    assert images, "Aucune image dans fixtures/"
    return Image.open(images[0]).convert("RGB")


@pytest.mark.unit
def test_resize_224(fixtures_dir: Path):
    import torchvision.transforms as T
    img = _sample_image(fixtures_dir)
    out = T.Resize((224, 224))(img)
    assert out.size == (224, 224)


@pytest.mark.unit
def test_resize_300(fixtures_dir: Path):
    import torchvision.transforms as T
    img = _sample_image(fixtures_dir)
    out = T.Resize((300, 300))(img)
    assert out.size == (300, 300)


@pytest.mark.unit
def test_to_tensor_shape(fixtures_dir: Path):
    import torchvision.transforms as T
    img = _sample_image(fixtures_dir)
    tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])(img)
    assert tensor.shape == (3, 224, 224)


@pytest.mark.unit
def test_normalize_output_range(fixtures_dir: Path):
    import torchvision.transforms as T
    img = _sample_image(fixtures_dir)
    tensor = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)
