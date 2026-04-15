"""
train_local.py
--------------
Standalone training script for AE / CNN-AE / CNN-VAE models on MNIST,
Fashion-MNIST, and CIFAR-10.

Usage:
    python -m app.train_local                              # trains all defaults
    python -m app.train_local --dataset cifar10 --model cnn_vae --epochs 10
    python -m app.train_local --dataset mnist   --model cnn_ae  --epochs 5 --seed 42

Saved weights go to: app/core/<dataset>_<model>.pth
They are automatically picked up by the FastAPI semantic inference endpoints.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.core.model_utils import get_model
from app.core.image_utils import load_dataset, DATASET_META


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    dataset_name: str = "fashion",
    model_type: str = "cnn_vae",
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    latent_dim: int = 32,
    kl_weight: float = 0.005,
    seed: int = 42,
) -> str:
    """
    Train an autoencoder (AE, CNN-AE, or CNN-VAE) on the specified dataset
    and save the weights to disk for later inference.

    Args:
        dataset_name:  One of "mnist", "fashion", "cifar10".
        model_type:    One of "ae", "cnn_ae", "cnn_vae".
        epochs:        Number of training epochs.
        batch_size:    DataLoader batch size.
        learning_rate: Adam optimizer learning rate.
        latent_dim:    Dimensionality of the latent bottleneck.
        kl_weight:     Weight applied to the KL divergence term (VAE only).
        seed:          Random seed for reproducibility.

    Returns:
        Path to the saved weights file.
    """
    set_seed(seed)

    meta = DATASET_META.get(dataset_name)
    if meta is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_META)}")

    channels = meta["channels"]
    img_size = meta["height"]   # assumes square images
    # Number of pixels per image — used to normalise the KL divergence loss
    pixels_per_image = channels * meta["height"] * meta["width"]

    print(f"\n{'='*60}")
    print(f"  Training  : {model_type.upper()}")
    print(f"  Dataset   : {dataset_name}  ({channels}ch, {img_size}x{img_size})")
    print(f"  Epochs    : {epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print(f"  Latent dim: {latent_dim}  |  Seed: {seed}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")

    dataset = load_dataset(dataset_name, train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = get_model(model_type, latent_dim=latent_dim, input_channels=channels, image_size=img_size)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()

            if model_type == "cnn_vae":
                recon, mu, logvar = model(data)
                mse_loss = criterion_mse(recon, data)

                # KL divergence: D_KL(q(z|x) || p(z))
                # Normalise by number of pixels so that kl_weight is
                # dataset-agnostic (works for MNIST 28x28 AND CIFAR-10 32x32x3)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_loss = kld_loss / (data.size(0) * pixels_per_image)

                loss = mse_loss + kl_weight * kld_loss
            else:
                recon = model(data)
                loss = criterion_mse(recon, data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch [{epoch+1}/{epochs}]  Batch [{batch_idx:>4}/{len(loader)}]"
                    f"  Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"  >>> Epoch {epoch+1} complete — avg loss: {avg_loss:.6f}")

    # Save weights
    save_path = f"app/core/{dataset_name}_{model_type}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  Weights saved → {save_path}\n")
    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE/AE models for federated semantic communication."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_META.keys()),
        default=None,
        help="Dataset to train on. If omitted, trains all datasets.",
    )
    parser.add_argument(
        "--model",
        choices=["ae", "cnn_ae", "cnn_vae"],
        default=None,
        help="Model architecture. If omitted, trains all models.",
    )
    parser.add_argument("--epochs",    type=int,   default=5,    help="Number of training epochs.")
    parser.add_argument("--batch",     type=int,   default=128,  help="Batch size.")
    parser.add_argument("--lr",        type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--latent",    type=int,   default=32,   help="Latent dimension size.")
    parser.add_argument("--kl-weight", type=float, default=0.005, help="KL loss weight (VAE only).")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Datasets and models to train — determined by CLI args or defaults
    datasets = [args.dataset] if args.dataset else ["mnist", "fashion", "cifar10"]
    models   = [args.model]   if args.model   else ["cnn_vae", "cnn_ae"]

    # Epoch counts per dataset when using defaults (shorter for larger datasets)
    default_epochs = {"mnist": 5, "fashion": 5, "cifar10": 8}

    for ds in datasets:
        for m in models:
            epochs = args.epochs if (args.dataset or args.model) else default_epochs[ds]
            train_model(
                dataset_name=ds,
                model_type=m,
                epochs=epochs,
                batch_size=args.batch,
                learning_rate=args.lr,
                latent_dim=args.latent,
                kl_weight=args.kl_weight,
                seed=args.seed,
            )

    print("\n✅  All models trained successfully.")
