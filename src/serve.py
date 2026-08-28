import io
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

try:
    from dataset import CIFAR10_CLASSES, get_transforms
    from model import get_model, load_model_from_checkpoint
except ImportError:
    from src.dataset import CIFAR10_CLASSES, get_transforms
    from src.model import get_model, load_model_from_checkpoint

app = FastAPI(
    title="PyTorch CIFAR-10 Inference Service",
    description="Production-ready FastAPI service for CIFAR-10 image classification",
    version="1.0.0",
)

# Global model holder
model: Optional[nn.Module] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform pipeline for inference (ensures resizing arbitrary input images to 32x32)
inference_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_inference_model() -> Optional[nn.Module]:
    """Load model from checkpoint if available, or initialize default model."""
    global model
    model_path_env = os.getenv("MODEL_PATH")
    candidate_paths = [
        model_path_env,
        "/app/checkpoints/classifier_v1.pt",
        "checkpoints/classifier_v1.pt",
        str(Path(__file__).resolve().parent.parent / "checkpoints" / "classifier_v1.pt"),
    ]

    for p in candidate_paths:
        if p and Path(p).exists():
            try:
                print(f"Loading checkpoint from: {p}", flush=True)
                model = load_model_from_checkpoint(p, architecture="resnet18", num_classes=10, device=device)
                return model
            except Exception as e:
                print(f"Error loading checkpoint from {p}: {e}", flush=True)

    # Fallback to untrained / initialized model for standalone testing if checkpoint is not yet trained
    print("No checkpoint found on disk. Initializing default ResNet-18 model.", flush=True)
    model = get_model(architecture="resnet18", num_classes=10).to(device)
    model.eval()
    return model


@app.on_event("startup")
def startup_event():
    """Load model weights on application startup."""
    load_inference_model()


@app.get("/health", summary="Health Check Endpoint")
def health_check():
    """Return health status and model readiness for Kubernetes probes."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "num_classes": len(CIFAR10_CLASSES),
    }


@app.post("/predict", summary="Predict Image Class Probabilities")
async def predict(image: UploadFile = File(...)):
    """Accept an uploaded image and return class predictions with confidence probabilities."""
    global model
    if model is None:
        model = load_inference_model()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or available for inference",
            )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}",
        )

    try:
        tensor_img = inference_transforms(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().tolist()

        predicted_idx = int(torch.tensor(probabilities).argmax())
        predicted_class = CIFAR10_CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx]

        class_probs: Dict[str, float] = {
            CIFAR10_CLASSES[i]: round(probabilities[i], 4)
            for i in range(len(CIFAR10_CLASSES))
        }

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": round(confidence, 4),
            "probabilities": class_probs,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8080, reload=False)
