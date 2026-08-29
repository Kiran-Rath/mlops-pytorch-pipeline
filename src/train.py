from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml

# Support both direct script execution and module import
try:
    from dataset import get_dataloaders
    from model import get_model
except ImportError:
    from src.dataset import get_dataloaders
    from src.model import get_model


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on validation data and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    """Main training workflow with early stopping and structured JSON metric logging."""
    # Resolve config path from ENV or well-known locations
    env_config = os.getenv("CONFIG_PATH")
    if env_config and Path(env_config).exists():
        config_path = Path(env_config)
    elif Path("/app/configs/training_config.yaml").exists():
        config_path = Path("/app/configs/training_config.yaml")
    elif Path("configs/training_config.yaml").exists():
        config_path = Path("configs/training_config.yaml")
    else:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"

    print(f"Loading configuration from: {config_path}", flush=True)
    config = load_config(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)

    architecture = config.get("model", {}).get("architecture", "resnet18")
    num_classes = config.get("model", {}).get("num_classes", 10)
    model = get_model(architecture=architecture, num_classes=num_classes).to(device)

    data_dir = config.get("data", {}).get("data_dir", "/app/data")
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    learning_rate = config.get("training", {}).get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config.get("training", {}).get("early_stopping_patience", 3)
    epochs = config.get("training", {}).get("epochs", 10)

    checkpoint_dir = Path(config.get("output", {}).get("checkpoint_dir", "/app/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("output", {}).get("model_name", "classifier_v1.pt")
    save_path = checkpoint_dir / model_name

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(save_path),
                "val_loss": round(val_loss, 4),
            }), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch + 1,
                    "best_val_loss": round(best_val_loss, 4),
                }), flush=True)
                break

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": str(save_path),
    }), flush=True)


if __name__ == "__main__":
    main()
