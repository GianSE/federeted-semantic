import torch
import torch.nn as nn
import math

def get_device():
    return torch.device("cpu")


def snr_to_noise_std(snr_db):
    """Converte SNR em dB para desvio padrão do ruído AWGN"""
    return 1.0 / math.sqrt(10 ** (snr_db / 10))


class ImageAutoencoder(nn.Module):
    """
    Autoencoder Convolucional para compressão e reconstrução de imagens.
    Encoder: Conv2d → Gargalo (32 dims) → Decoder: ConvTranspose2d
    Projetado para imagens MNIST (1x28x28).
    
    Compressão: 784 pixels → 32 dimensões latentes (96% de compressão)
    """
    def __init__(self, latent_dim=32):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # --- ENCODER (Comprime a imagem) ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28 → 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 32x28x28 → 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32x14x14 → 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 64x14x14 → 64x7x7
        )
        
        # Camadas densas do encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # Gargalo: 32 dimensões
        )
        
        # --- DECODER (Reconstrói a imagem) ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x7x7 → 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 32x14x14 → 1x28x28
            nn.Sigmoid(),  # Saída normalizada [0, 1]
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
        x = x.view(x.size(0), 64, 7, 7)
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
    def __init__(self, latent_dim=32):
        super(ImageVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # --- ENCODER (mesmo backbone do AE) ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        # Duas cabeças: μ e log(σ²)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # --- DECODER (idêntico ao AE) ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
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
        x = x.view(x.size(0), 64, 7, 7)
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


def get_model(model_type="autoencoder", latent_dim=32):
    """Factory: retorna AE ou VAE conforme config"""
    if model_type == "vae":
        return ImageVAE(latent_dim=latent_dim)
    return ImageAutoencoder(latent_dim=latent_dim)
