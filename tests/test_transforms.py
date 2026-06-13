"""Tests des transformations d'images (resize, normalize)."""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_sample_image():
    images = list(FIXTURES.glob("*.jpg"))
    assert images, "Aucune image dans fixtures/"
    return Image.open(images[0]).convert("RGB")


def test_resize_224():
    import torchvision.transforms as T
    img = get_sample_image()
    tf = T.Resize((224, 224))
    out = tf(img)
    assert out.size == (224, 224)


def test_resize_300():
    import torchvision.transforms as T
    img = get_sample_image()
    tf = T.Resize((300, 300))
    out = tf(img)
    assert out.size == (300, 300)


def test_to_tensor_shape():
    import torch
    import torchvision.transforms as T
    img = get_sample_image()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)


def test_normalize_output_range():
    import torch
    import torchvision.transforms as T
    img = get_sample_image()
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)
