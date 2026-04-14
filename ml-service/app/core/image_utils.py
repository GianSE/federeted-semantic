import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

DATA_DIR = "/ml-data/datasets"

def quantize_latent(latent, bits=8):
    if bits is None or bits >= 32:
        return latent.clone(), 1.0
        
    max_abs = torch.max(torch.abs(latent)).item()
    if max_abs == 0:
        max_abs = 1e-5
    max_val = 2**(bits - 1) - 1
    scale = max_val / max_abs
    if bits <= 8:
        quantized = torch.round(latent * scale).to(torch.int8)
    else:
        quantized = torch.round(latent * scale).to(torch.int16)
    return quantized, scale

def dequantize_latent(quantized, scale):
    return quantized.to(torch.float32) / scale

def build_transform():
    return transforms.Compose([transforms.ToTensor()])

def load_dataset(dataset_name="fashion", train=False):
    transform = build_transform()
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform)
    if dataset_name == "fashion":
        return torchvision.datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform)
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform)
    raise ValueError(f"Dataset inválido: {dataset_name}")

def compute_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()

def compute_psnr(original, reconstructed, max_val=1.0):
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)

def mask_image_bottom(image, mask_ratio=0.5):
    masked = image.clone()
    if masked.dim() == 3:
        _, h, _ = masked.shape
        cut = int(h * (1 - mask_ratio))
        masked[:, cut:, :] = 0
    elif masked.dim() == 4:
        _, _, h, _ = masked.shape
        cut = int(h * (1 - mask_ratio))
        masked[:, :, cut:, :] = 0
    return masked

def mask_image_right(image, mask_ratio=0.5):
    masked = image.clone()
    if masked.dim() == 3:
        _, _, w = masked.shape
        cut = int(w * (1 - mask_ratio))
        masked[:, :, cut:] = 0
    elif masked.dim() == 4:
        _, _, _, w = masked.shape
        cut = int(w * (1 - mask_ratio))
        masked[:, :, :, cut:] = 0
    return masked

def mask_image_random(image, mask_ratio=0.5):
    mask = (torch.rand_like(image.float()) > mask_ratio).float()
    return image * mask
