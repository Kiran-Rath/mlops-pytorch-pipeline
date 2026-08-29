from __future__ import annotations
import io
import sys
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root and src are in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.dataset import CIFAR10_CLASSES, get_transforms
from src.model import SimpleCNN, get_model
from src.serve import app, load_inference_model


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client with initialized model."""
    load_inference_model()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    """Generate in-memory RGB PNG test image bytes."""
    img = Image.new("RGB", (64, 64), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_resnet18_model_output_shape():
    """Verify ResNet-18 model forward pass output shape for CIFAR-10."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    dummy_input = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_simple_cnn_output_shape():
    """Verify SimpleCNN model forward pass output shape."""
    model = SimpleCNN(num_classes=10)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"


def test_dataset_transforms():
    """Verify transform output tensors are normalized with correct channel dimension."""
    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)

    sample_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    train_tensor = train_tf(sample_img)
    val_tensor = val_tf(sample_img)

    assert train_tensor.shape == (3, 32, 32)
    assert val_tensor.shape == (3, 32, 32)
    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)


def test_health_endpoint(client):
    """Verify GET /health returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 10


def test_predict_endpoint_success(client, sample_image_bytes):
    """Verify POST /predict returns predicted class, confidence, and 10 class probabilities."""
    files = {"image": ("test_image.png", sample_image_bytes, "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert data["predicted_class"] in CIFAR10_CLASSES
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 10

    # Ensure probabilities sum to approx 1.0
    prob_sum = sum(data["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-2) == 1.0


def test_predict_endpoint_invalid_file(client):
    """Verify POST /predict returns HTTP 400 when sent corrupted non-image file."""
    files = {"image": ("corrupted.txt", b"not-an-image-content", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
