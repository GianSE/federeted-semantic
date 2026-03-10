import torch
import torch.nn as nn
import math

def get_device():
    return torch.device("cpu")

def snr_to_noise_std(snr_db):
    """Converte SNR em dB para desvio padrão do ruído AWGN"""
    return 1.0 / math.sqrt(10 ** (snr_db / 10))

class TextAutoencoder(nn.Module):
    """
    Autoencoder Semântico para Textos Focado em Compressão Extrema.
    Encoder: Embedding → Simple Recurrent/Linear → Gargalo Latente (Assunto)
    Decoder: Linear → Vocabulário
    
    Atenção: Para testes simples de FedAvg vamos usar uma estrutura MLP pós-embedding 
    e pooling para atingir a compressão desejada de forma rápida.
    """
    def __init__(self, vocab_size, seq_len=50, embed_dim=64, hidden_dim=128, latent_dim=32):
        super(TextAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # --- ENCODER ---
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # Após a view/flatten da sequência (seq_len * embed_dim)
        self.encoder_fc = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Gargalo Compressivo (Assunto)
        )
        
        # --- DECODER ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, seq_len * embed_dim),
            nn.ReLU()
        )
        
        # Projeção final para o tamanho do vocabulário, token a token
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def encode(self, x):
        """Comprime a sequência de tokens no vetor latente"""
        # x.shape = (batch_size, seq_len)
        embedded = self.embedding(x) # (batch_size, seq_len, embed_dim)
        flattened = embedded.view(embedded.size(0), -1) # (batch_size, seq_len * embed_dim)
        z = self.encoder_fc(flattened) # (batch_size, latent_dim)
        return z

    def decode(self, z):
        """Reconstrói a partir do vetor latente"""
        # (batch_size, seq_len * embed_dim)
        decoded_flat = self.decoder_fc(z) 
        
        # Re-shape back para sequência (batch_size, seq_len, embed_dim)
        decoded_seq = decoded_flat.view(decoded_flat.size(0), self.seq_len, -1)
        
        # Projeta cada posição na dimensão do vocabulário (batch_size, seq_len, vocab_size)
        logits = self.output_projection(decoded_seq)
        return logits

    def forward(self, x, snr_db=None):
        z = self.encode(x)
        
        # Injeção de Ruído AWGN no Canal Físico
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
            
        logits = self.decode(z)
        return logits


class TextVAE(nn.Module):
    """
    Variational Autoencoder para Textos Generativos (Semântico).
    """
    def __init__(self, vocab_size, seq_len=50, embed_dim=64, hidden_dim=128, latent_dim=32):
        super(TextVAE, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        # Cabeças do VAE
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, seq_len * embed_dim),
            nn.ReLU()
        )
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        h = self.encoder_fc(flattened)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded_flat = self.decoder_fc(z)
        decoded_seq = decoded_flat.view(decoded_flat.size(0), self.seq_len, -1)
        logits = self.output_projection(decoded_seq)
        return logits

    def forward(self, x, snr_db=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        if snr_db is not None and self.training:
            noise_std = snr_to_noise_std(snr_db)
            z = z + torch.randn_like(z) * noise_std
            
        logits = self.decode(z)
        return logits, mu, logvar

    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def get_model(model_type="autoencoder", vocab_size=10000, seq_len=50, latent_dim=32):
    """Retorna TextAE ou TextVAE"""
    if model_type == "vae":
        return TextVAE(vocab_size=vocab_size, seq_len=seq_len, latent_dim=latent_dim)
    return TextAutoencoder(vocab_size=vocab_size, seq_len=seq_len, latent_dim=latent_dim)
