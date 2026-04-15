"""
image_utils.py
--------------
Utility functions for image processing, dataset loading, and quality metrics.

Metrics implemented:
    - MSE  (Mean Squared Error)
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index) — PyTorch-native, no extra deps required

Quantization helpers:
    - quantize_latent   : float32 → int8/int16 (uniform min-max)
    - dequantize_latent : int8/int16 → float32

Dataset constants (DATASET_META):
    Provides canonical channels, height, width and raw byte size per dataset.
"""

import math

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dataset metadata — canonical sizes for each supported dataset
# ---------------------------------------------------------------------------
DATASET_META = {
    "mnist": {
        "channels": 1,
        "height": 28,
        "width": 28,
        "raw_bytes": 1 * 28 * 28 * 4,  # float32: 3136 bytes
    },
    "fashion": {
        "channels": 1,
        "height": 28,
        "width": 28,
        "raw_bytes": 1 * 28 * 28 * 4,  # float32: 3136 bytes
    },
    "cifar10": {
        "channels": 3,
        "height": 32,
        "width": 32,
        "raw_bytes": 3 * 32 * 32 * 4,  # float32: 12288 bytes
    },
}

DATA_DIR = "/ml-data/datasets"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_transform(dataset_name: str = "mnist") -> transforms.Compose:
    """Return a torchvision transform appropriate for the given dataset."""
    return transforms.Compose([transforms.ToTensor()])


def load_dataset(dataset_name: str = "fashion", train: bool = False):
    """
    Load and return a torchvision dataset.

    Args:
        dataset_name: One of "mnist", "fashion", "cifar10".
        train:        If True, return the training split; else test split.

    Returns:
        A torchvision Dataset object.
    """
    transform = build_transform(dataset_name)
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    if dataset_name == "fashion":
        return torchvision.datasets.FashionMNIST(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=train, download=True, transform=transform
        )
    raise ValueError(f"Unsupported dataset: '{dataset_name}'. Choose from: mnist, fashion, cifar10")


# ---------------------------------------------------------------------------
# Byte-size helpers
# ---------------------------------------------------------------------------

def get_original_bytes(dataset_name: str) -> int:
    """
    Return the raw transmission cost (in bytes) of a single image for the
    given dataset, assuming float32 encoding.

    Args:
        dataset_name: One of "mnist", "fashion", "cifar10".

    Returns:
        Integer number of bytes.
    """
    meta = DATASET_META.get(dataset_name)
    if meta is None:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")
    return meta["raw_bytes"]


def get_latent_bytes(latent: torch.Tensor, bits: int) -> int:
    """
    Compute the byte cost of transmitting a quantized latent vector.

    For bits >= 32 (float32): numel * 4 bytes.
    For bits <  32 (int8/int16): numel * (bits/8) bytes + 4 bytes for the
    scale factor (float32 scalar needed for dequantization).

    Args:
        latent: Latent tensor (any shape).
        bits:   Quantization bit-width.

    Returns:
        Integer number of bytes.
    """
    if bits >= 32:
        return int(latent.numel() * 4)
    scale_overhead = 4  # one float32 for the dequantization scale
    return int(latent.numel() * (bits / 8) + scale_overhead)


# ---------------------------------------------------------------------------
# Quantization / dequantization
# ---------------------------------------------------------------------------

def quantize_latent(latent: torch.Tensor, bits: int = 8):
    """
    Uniform min-max quantization of a latent tensor.

    Args:
        latent: Float32 tensor.
        bits:   Target bit-width (4, 8, 16, or 32).

    Returns:
        (quantized, scale): The integer-typed tensor and the scale factor.
    """
    if bits is None or bits >= 32:
        return latent.clone(), 1.0

    max_abs = torch.max(torch.abs(latent)).item()
    if max_abs == 0:
        max_abs = 1e-5
    max_val = 2 ** (bits - 1) - 1
    scale = max_val / max_abs

    if bits <= 8:
        quantized = torch.round(latent * scale).to(torch.int8)
    else:
        quantized = torch.round(latent * scale).to(torch.int16)

    return quantized, scale


def dequantize_latent(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Dequantize an integer tensor back to float32.

    Args:
        quantized: Integer tensor (int8 or int16).
        scale:     Scale factor returned by quantize_latent.

    Returns:
        Float32 tensor.
    """
    return quantized.to(torch.float32) / scale


def snr_to_noise_std(snr_db: float | None) -> float:
    """Convert SNR in dB to the equivalent AWGN standard deviation."""
    if snr_db is None:
        return 0.0
    return 1.0 / math.sqrt(10 ** (snr_db / 10))


def apply_awgn_noise(image: torch.Tensor, snr_db: float | None) -> torch.Tensor:
    """Apply additive white Gaussian noise and clamp the result to [0, 1]."""
    noise_std = snr_to_noise_std(snr_db)
    if noise_std <= 0:
        return image.clone()
    noisy = image + torch.randn_like(image) * noise_std
    return noisy.clamp(0.0, 1.0)


def apply_random_pixel_mask(
    image: torch.Tensor,
    drop_rate: float | None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Randomly remove pixels by replacing them with a constant fill value."""
    if drop_rate is None or drop_rate <= 0:
        return image.clone()

    drop_rate = max(0.0, min(0.99, float(drop_rate)))
    masked = image.clone()

    if masked.dim() == 4:
        mask_shape = (masked.size(0), 1, masked.size(2), masked.size(3))
    elif masked.dim() == 3 and masked.size(0) in (1, 3):
        mask_shape = (1, masked.size(1), masked.size(2))
    else:
        mask_shape = masked.shape

    keep_mask = (torch.rand(mask_shape, device=masked.device) >= drop_rate).to(masked.dtype)
    if masked.dim() == 4 and keep_mask.size(1) == 1:
        keep_mask = keep_mask.expand(-1, masked.size(1), -1, -1)
    elif masked.dim() == 3 and keep_mask.size(0) == 1:
        keep_mask = keep_mask.expand_as(masked)

    fill_tensor = torch.full_like(masked, float(fill_value))
    return torch.where(keep_mask > 0, masked, fill_tensor).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute the Mean Squared Error between two image tensors.

    Args:
        original:      Reference image tensor, values in [0, 1].
        reconstructed: Reconstructed image tensor, values in [0, 1].

    Returns:
        MSE as a Python float.
    """
    return torch.mean((original - reconstructed) ** 2).item()


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        original:      Reference image tensor.
        reconstructed: Reconstructed image tensor.
        max_val:       Maximum possible pixel value (1.0 for normalized images).

    Returns:
        PSNR in dB. Returns float('inf') if MSE is zero (perfect reconstruction).
    """
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        # Avoid Infinity so JSON serialization does not fail.
        return 120.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two image tensors.

    Implements the full SSIM formula from Wang et al. (2004) using a Gaussian
    window — PyTorch-native, no additional dependencies required.

    Args:
        original:      Reference image tensor, shape [B, C, H, W] or [C, H, W].
        reconstructed: Reconstructed image tensor, same shape as original.
        window_size:   Size of the Gaussian sliding window (default: 11).
        sigma:         Standard deviation of the Gaussian kernel (default: 1.5).
        data_range:    Value range of the input images (default: 1.0).

    Returns:
        SSIM score in [-1, 1]; +1 indicates identical images.
    """
    # Ensure 4D: [B, C, H, W]
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.unsqueeze(0)

    original = original.float()
    reconstructed = reconstructed.float()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    channels = original.shape[1]

    # Build Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.unsqueeze(0)                        # [1, window_size]
    kernel_2d = kernel_1d.t().mm(kernel_1d)               # [window_size, window_size]
    kernel = (
        kernel_2d
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(channels, 1, -1, -1)
        .to(original.device)
    )

    pad = window_size // 2

    mu1 = F.conv2d(original, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(reconstructed, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(original * original, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(reconstructed * reconstructed, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(original * reconstructed, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean().item())


def compute_compression_ratio(dataset_name: str, latent: torch.Tensor, bits: int) -> float:
    """
    Compute the bandwidth compression ratio: original_bytes / latent_bytes.

    A ratio of 24.5 means only 1/24.5 of the original data is transmitted.

    Args:
        dataset_name: Dataset name to determine original image size.
        latent:       Latent tensor (after encoding).
        bits:         Quantization bit-width used for transmission.

    Returns:
        Compression ratio as a float (>1 means compression is beneficial).
    """
    original_bytes = get_original_bytes(dataset_name)
    latent_bytes = get_latent_bytes(latent, bits)
    if latent_bytes == 0:
        return float("inf")
    return original_bytes / latent_bytes


