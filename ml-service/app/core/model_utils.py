import math
import torch
import torch.nn as nn

IMAGE_CHANNELS = 1
IMAGE_SIZE = 28
MODEL_BACKBONE = "simple"

def get_device():
    return torch.device("cpu")


def snr_to_noise_std(snr_db: float | None) -> float:
    if snr_db is None:
        return 0.0
    return 1.0 / math.sqrt(10 ** (snr_db / 10))


def build_simple_backbone(input_channels, output_channels, image_size):
    encoder = nn.Sequential(
        # Block 1: input_channels → 32
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2),  # H/2

        # Block 2: 32 → 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2),  # H/4

        # Block 3: 64 → 128
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # No pool here — keep spatial res for richer features
    )
    decoder = nn.Sequential(
        # Mirror block 3
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        # Mirror block 2
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        # Output projection
        nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )
    fe_size = image_size // 4  # after 2 MaxPool2d
    hidden_dim = 512           # bigger bottleneck FC
    return encoder, decoder, 128, fe_size, hidden_dim, 128

def build_backbone(input_channels, output_channels, image_size):
    return build_simple_backbone(input_channels, output_channels, image_size)

class ImageAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=1, image_size=28):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_conv, self.decoder_conv, self.feature_channels, self.feature_size, hidden_dim, self.decoder_channels = build_backbone(
            input_channels, input_channels, image_size
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_channels * self.feature_size * self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.decoder_channels * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_fc(x)
        return z
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.decoder_channels, self.feature_size, self.feature_size)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x, snr_db: float | None = None):
        z = self.encode(x)
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
        return self.decode(z)

class ImageVAE(nn.Module):
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
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.decoder_channels * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.decoder_channels, self.feature_size, self.feature_size)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x, snr_db: float | None = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
        return self.decode(z), mu, logvar

def get_model(model_type="autoencoder", latent_dim=32, input_channels=1, image_size=28):
    if model_type == "cnn_vae":
        return ImageVAE(latent_dim=latent_dim, input_channels=input_channels, image_size=image_size)
    return ImageAutoencoder(latent_dim=latent_dim, input_channels=input_channels, image_size=image_size)
