"""
classifier_utils.py
------------------
Utilities for lightweight image classifiers per dataset.

The classifier is used to verify semantic preservation after degradation
and reconstruction in the semantic communication pipeline.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.image_utils import DATASET_META


class SimpleClassifier(nn.Module):
    def __init__(self, input_channels: int, image_size: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            flat_dim = int(self.features(dummy).view(1, -1).size(1))

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        import torchvision.models as models
        
        # Load MobileNetV2 pre-trained on ImageNet
        # weights argument used per modern torchvision conventions
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        
        # Freeze all but the last layers (mimicking Keras base_model.layers[:-40])
        # MobileNet features has 19 blocks. We freeze 0 through 14, unfreeze 15-18.
        for idx, child in enumerate(self.features.children()):
            if idx < 15:
                for param in child.parameters():
                    param.requires_grad = False

        # In keras: GlobalAveragePooling2D
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace the head with Keras-equivalent topology
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _resolve_classifier_weights(dataset_name: str) -> tuple[Path | None, str | None]:
    import os as _os
    _data_root = Path(_os.environ.get("DATA_ROOT", "/app/data/ml-data"))
    weights_dir = _data_root / "weights"
    latest_path = weights_dir / f"{dataset_name}_classifier.pth"
    core_path = Path(f"app/core/{dataset_name}_classifier.pth")

    if latest_path.exists():
        return latest_path, f"weights/{latest_path.name}"
    if core_path.exists():
        return core_path, f"core/{core_path.name}"
    return None, None


def load_classifier(dataset_name: str) -> tuple[nn.Module | None, bool, str | None]:
    meta = DATASET_META.get(dataset_name)
    if meta is None:
        return None, False, None

    if dataset_name == "cifar10":
        model = MobileNetClassifier(num_classes=meta.get("classes", 10))
    else:
        model = SimpleClassifier(
            input_channels=meta["channels"],
            image_size=meta["height"],
            num_classes=meta.get("classes", 10),
        )
    weights_path, source = _resolve_classifier_weights(dataset_name)
    if weights_path and weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        return model, True, source

    model.eval()
    return model, False, None


def predict_topk(
    model: nn.Module,
    images: torch.Tensor,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, MobileNetClassifier) and images.shape[-1] != 96:
        images = F.interpolate(images, size=(96, 96), mode='bilinear', align_corners=False)
        
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    top_k = max(1, min(int(top_k), probs.size(1)))
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
    return top_indices, top_probs


def format_topk(indices: torch.Tensor, probs: torch.Tensor) -> list[dict]:
    items = []
    for idx, prob in zip(indices.tolist(), probs.tolist()):
        items.append({"label": int(idx), "prob": float(prob)})
    return items
