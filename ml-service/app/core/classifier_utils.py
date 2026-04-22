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
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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


def load_classifier(dataset_name: str) -> tuple[SimpleClassifier | None, bool, str | None]:
    meta = DATASET_META.get(dataset_name)
    if meta is None:
        return None, False, None

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
