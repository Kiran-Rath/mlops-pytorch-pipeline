from __future__ import annotations
import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    from torchvision.models import resnet18
    ResNet18_Weights = None


class SimpleCNN(nn.Module):
    """A lightweight CNN model for image classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """Instantiate and return model architecture for image classification."""
    architecture_lower = architecture.lower()

    if architecture_lower == "resnet18":
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        else:
            model = resnet18(pretrained=pretrained)
        # Adapt final fully connected layer for CIFAR-10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif architecture_lower in ["simple_cnn", "cnn"]:
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. Choose from 'resnet18', 'simple_cnn'."
        )


def load_model_from_checkpoint(
    checkpoint_path: str,
    architecture: str = "resnet18",
    num_classes: int = 10,
    device: torch.device = None,
) -> nn.Module:
    """Load a model state dictionary from a saved checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(architecture=architecture, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()
    return model
