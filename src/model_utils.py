import torch
import torch.nn as nn
import math

from config import IMAGE_CHANNELS, IMAGE_SIZE, MODEL_BACKBONE

def get_device():
    return torch.device("cpu")


def snr_to_noise_std(snr_db):
    """Converte SNR em dB para desvio padrão do ruído AWGN"""
    return 1.0 / math.sqrt(10 ** (snr_db / 10))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.refine = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        return self.refine(self.upsample(x))


def build_simple_backbone(input_channels, output_channels, image_size):
    encoder = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )
    return encoder, decoder, 64, image_size // 4, 256, 64


def build_deep_backbone(input_channels, output_channels, image_size):
    encoder = nn.Sequential(
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
    decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )
    return encoder, decoder, 128, image_size // 8, 512, 128


def build_cifar_backbone(input_channels, output_channels, image_size):
    encoder = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        ResidualBlock(64, 64),
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 128),
        ResidualBlock(128, 256, stride=2),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256, stride=2),
        ResidualBlock(256, 256),
    )
    decoder = nn.Sequential(
        DecoderBlock(256, 256),
        DecoderBlock(256, 128),
        DecoderBlock(128, 64),
        nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )
    return encoder, decoder, 256, image_size // 8, 512, 256


def build_backbone(input_channels, output_channels, image_size):
    if image_size >= 32 and MODEL_BACKBONE == "cifar":
        return build_cifar_backbone(input_channels, output_channels, image_size)
    if image_size >= 32 and MODEL_BACKBONE == "deep":
        return build_deep_backbone(input_channels, output_channels, image_size)
    return build_simple_backbone(input_channels, output_channels, image_size)


class ImageAutoencoder(nn.Module):
    """
    Autoencoder Convolucional para compressão e reconstrução de imagens.
    Encoder: Conv2d → Gargalo (32 dims) → Decoder: ConvTranspose2d
    Projetado para imagens MNIST (1x28x28).
    
    Compressão: 784 pixels → 32 dimensões latentes (96% de compressão)
    """
    def __init__(self, latent_dim=32, input_channels=1, image_size=28):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_conv, self.decoder_conv, self.feature_channels, self.feature_size, hidden_dim, self.decoder_channels = build_backbone(
            input_channels, input_channels, image_size
        )
        
        # Camadas densas do encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_channels * self.feature_size * self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # --- DECODER (Reconstrói a imagem) ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.decoder_channels * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
    
    def encode(self, x):
        """Comprime imagem para vetor latente de 32 dimensões"""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        z = self.encoder_fc(x)
        return z
    
    def decode(self, z):
        """Reconstrói imagem 28x28 a partir do vetor latente"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.decoder_channels, self.feature_size, self.feature_size)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x, snr_db=None):
        z = self.encode(x)
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
        reconstructed = self.decode(z)
        return reconstructed


class ImageVAE(nn.Module):
    """
    Variational Autoencoder Convolucional para comunicação semântica generativa.
    Encoder: Conv2d → μ, log(σ²) → Reparametrização → Decoder: ConvTranspose2d
    
    Diferença do AE: espaço latente é distribuição N(μ, σ²), permitindo geração.
    Loss = MSE + β · KL(q(z|x) || p(z)), onde p(z) = N(0, I)
    """
    def __init__(self, latent_dim=32, input_channels=1, image_size=28):
        super(ImageVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_conv, self.decoder_conv, self.feature_channels, self.feature_size, hidden_dim, self.decoder_channels = build_backbone(
            input_channels, input_channels, image_size
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_channels * self.feature_size * self.feature_size, hidden_dim),
            nn.ReLU(),
        )
        # Duas cabeças: μ e log(σ²)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- DECODER (idêntico ao AE) ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.decoder_channels * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
    
    def encode(self, x):
        """Retorna μ e log(σ²) da distribuição posterior q(z|x)"""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Trick de reparametrização: z = μ + σ · ε, ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Reconstrói imagem 28x28 a partir do vetor latente"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.decoder_channels, self.feature_size, self.feature_size)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x, snr_db=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def generate(self, num_samples=16):
        """Gera imagens novas amostrando z ~ N(0, I)"""
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            return self.decode(z)
    
    @staticmethod
    def kl_divergence(mu, logvar):
        """KL(q(z|x) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def get_model(model_type="autoencoder", latent_dim=32, input_channels=None, image_size=None):
    """Factory: retorna AE ou VAE conforme config"""
    input_channels = IMAGE_CHANNELS if input_channels is None else input_channels
    image_size = IMAGE_SIZE if image_size is None else image_size
    if model_type == "vae":
        return ImageVAE(latent_dim=latent_dim, input_channels=input_channels, image_size=image_size)
    return ImageAutoencoder(latent_dim=latent_dim, input_channels=input_channels, image_size=image_size)
