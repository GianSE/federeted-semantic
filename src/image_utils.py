import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

from config import DATASET, IMAGE_SIZE, CLASS_NAMES

DATA_DIR = "data"


def get_dataset_metadata(dataset_name=None):
    dataset_name = (dataset_name or DATASET).lower()
    if dataset_name == "mnist":
        return {
            "name": "mnist",
            "display_name": "MNIST",
            "channels": 1,
            "image_size": 28,
            "class_names": [str(i) for i in range(10)],
            "label_kind": "dígito",
        }
    if dataset_name == "fashion":
        return {
            "name": "fashion",
            "display_name": "Fashion-MNIST",
            "channels": 1,
            "image_size": 28,
            "class_names": CLASS_NAMES,
            "label_kind": "classe",
        }
    if dataset_name == "cifar10":
        return {
            "name": "cifar10",
            "display_name": "CIFAR-10",
            "channels": 3,
            "image_size": 32,
            "class_names": CLASS_NAMES,
            "label_kind": "classe",
        }
    raise ValueError(f"Dataset não suportado: {dataset_name}")


def build_transform(dataset_name=None):
    metadata = get_dataset_metadata(dataset_name)
    transform_steps = []
    if metadata["image_size"] != IMAGE_SIZE:
        transform_steps.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def load_dataset(train=True, dataset_name=None):
    dataset_name = (dataset_name or DATASET).lower()
    transform = build_transform(dataset_name)

    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform)
    if dataset_name == "fashion":
        return torchvision.datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform)
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform)
    raise ValueError(f"Dataset não suportado: {dataset_name}")


def load_dataset_filtered(train=True, allowed_labels=None, dataset_name=None):
    dataset = load_dataset(train=train, dataset_name=dataset_name)
    if allowed_labels is None:
        return dataset
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_labels]
    return torch.utils.data.Subset(dataset, indices)

def load_mnist(train=True):
    """Carrega o dataset MNIST (download automático se necessário)"""
    return load_dataset(train=train, dataset_name="mnist")

def load_mnist_filtered(train=True, allowed_labels=None):
    """Carrega MNIST filtrado para apenas certos dígitos (Non-IID)"""
    return load_dataset_filtered(train=train, allowed_labels=allowed_labels, dataset_name="mnist")

def get_random_batch(dataset, batch_size=32):
    """Retorna um batch aleatório de imagens do dataset"""
    indices = torch.randperm(len(dataset))[:batch_size]
    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return images, labels


def label_to_name(label, dataset_name=None):
    metadata = get_dataset_metadata(dataset_name)
    label_value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    class_names = metadata["class_names"]
    if 0 <= label_value < len(class_names):
        return class_names[label_value]
    return str(label_value)


def tensor_to_image_array(tensor):
    if isinstance(tensor, np.ndarray):
        array = tensor
    else:
        array = tensor.detach().cpu().float().numpy()

    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[0] == 1:
            return np.clip(array[0], 0.0, 1.0)
        if array.shape[0] in (3, 4):
            return np.clip(np.transpose(array, (1, 2, 0)), 0.0, 1.0)
    return np.clip(array, 0.0, 1.0)

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


def ssim_index(original, reconstructed):
    """Versão diferenciável do SSIM médio por batch/canal."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    if original.dim() == 2:
        img1 = original.unsqueeze(0).unsqueeze(0)
        img2 = reconstructed.unsqueeze(0).unsqueeze(0)
    elif original.dim() == 3:
        img1 = original.unsqueeze(0)
        img2 = reconstructed.unsqueeze(0)
    else:
        img1 = original
        img2 = reconstructed

    img1 = img1.float()
    img2 = img2.float()

    mu1 = img1.mean(dim=(-1, -2), keepdim=True)
    mu2 = img2.mean(dim=(-1, -2), keepdim=True)
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=(-1, -2), keepdim=True)

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return (numerator / denominator).mean()


def reconstruction_loss(original, reconstructed, ssim_weight=0.0):
    mse = torch.mean((original - reconstructed) ** 2)
    if ssim_weight <= 0:
        return mse
    ssim_term = 1.0 - ssim_index(original, reconstructed)
    return mse + ssim_weight * ssim_term

def compute_ssim(original, reconstructed, window_size=7):
    """
    Calcula o SSIM (Structural Similarity Index) entre dois tensores.
    Implementação simplificada para imagens em escala de cinza.
    Retorna valor entre -1 e 1 (1 = idênticas).
    """
    return float(ssim_index(original, reconstructed).item())
