import torch
import torch.nn as nn

def get_device():
    return torch.device("cpu")

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
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
