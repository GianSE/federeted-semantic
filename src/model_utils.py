import torch
import torch.nn as nn

def get_device():
    return torch.device("cpu")

class SemanticAutoencoder(nn.Module):
    def __init__(self, input_dim=10): # Padrão ajustado para 10
        super(SemanticAutoencoder, self).__init__()
        # Encoder: Comprime 10 letras em 5 números (Conceito)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5), # Latent Space apertado
            nn.ReLU()
        )
        # Decoder: Tenta reconstruir as 10 letras
        self.decoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Saída entre 0 e 1 (para o detokenizer)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded