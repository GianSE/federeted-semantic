import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

DATA_DIR = "data"

def load_mnist(train=True):
    """Carrega o dataset MNIST (download automático se necessário)"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte para tensor [0, 1]
    ])
    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, 
        train=train, 
        download=True, 
        transform=transform
    )
    return dataset

def load_mnist_filtered(train=True, allowed_labels=None):
    """Carrega MNIST filtrado para apenas certos dígitos (Non-IID)"""
    dataset = load_mnist(train=train)
    if allowed_labels is None:
        return dataset
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_labels]
    return torch.utils.data.Subset(dataset, indices)

def get_random_batch(dataset, batch_size=32):
    """Retorna um batch aleatório de imagens do dataset"""
    indices = torch.randperm(len(dataset))[:batch_size]
    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return images, labels

def mask_image_bottom(image, mask_ratio=0.5):
    """
    Mascara a parte inferior da imagem.
    Simula envio parcial: apenas a metade superior é transmitida.
    """
    masked = image.clone()
    if masked.dim() == 3:
        _, h, w = masked.shape
        cut = int(h * (1 - mask_ratio))
        masked[:, cut:, :] = 0
    elif masked.dim() == 4:
        _, _, h, w = masked.shape
        cut = int(h * (1 - mask_ratio))
        masked[:, :, cut:, :] = 0
    return masked

def mask_image_random(image, mask_ratio=0.5):
    """
    Mascara pixels aleatórios da imagem.
    Simula perda parcial de dados durante transmissão.
    """
    mask = (torch.rand_like(image.float()) > mask_ratio).float()
    masked = image * mask
    return masked

def mask_image_right(image, mask_ratio=0.5):
    """
    Mascara a parte direita da imagem.
    Simula envio parcial: apenas a metade esquerda é transmitida.
    """
    masked = image.clone()
    if masked.dim() == 3:
        _, h, w = masked.shape
        cut = int(w * (1 - mask_ratio))
        masked[:, :, cut:] = 0
    elif masked.dim() == 4:
        _, _, h, w = masked.shape
        cut = int(w * (1 - mask_ratio))
        masked[:, :, :, cut:] = 0
    return masked

def compute_mse(original, reconstructed):
    """Calcula o MSE (Mean Squared Error) entre dois tensores"""
    return torch.mean((original - reconstructed) ** 2).item()

def compute_psnr(original, reconstructed, max_val=1.0):
    """Calcula o PSNR (Peak Signal-to-Noise Ratio) em dB"""
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)
