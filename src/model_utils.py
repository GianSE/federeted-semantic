import torch
import torch.nn as nn

def get_device():
    return torch.device("cpu")

class SemanticAutoencoder(nn.Module):
    def __init__(self, input_dim=30, vocab_size=128, embedding_dim=64, hidden_dim=256):
        super(SemanticAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # --- ENCODER (Lê e Comprime) ---
        # 1. Transforma índices (letras) em vetores ricos (Embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Processa a sequência temporal (LSTM Profunda)
        # Bidirecional = Lê da esquerda pra direita E vice-versa
        self.encoder_lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2,
            bidirectional=True 
        )
        
        # 3. Gargalo Semântico (Onde a mágica da compressão acontece)
        # Comprime 512 (2*256) para apenas 32 números
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32) 
        )

        # --- DECODER (Reconstrói e Gera) ---
        # 4. Expande o conceito de volta
        self.decoder_fc = nn.Linear(32, hidden_dim)
        
        # 5. Reconstrói a frase (LSTM Decoder)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        
        # 6. Escolhe a letra final (Classificador)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len] (Indices inteiros)
        
        # --- ENCODING ---
        embedded = self.embedding(x) # -> [batch, seq, emb_dim]
        _, (hidden, _) = self.encoder_lstm(embedded)
        
        # Pega o último estado das duas direções
        # hidden shape: [num_layers*num_dirs, batch, hidden_dim]
        # Concatenamos as duas direções da última camada
        last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Comprime
        latent = self.bottleneck(last_hidden) # -> [batch, 32]
        
        # --- DECODING ---
        # Prepara para decodificar
        decoder_input = self.decoder_fc(latent).unsqueeze(1) # -> [batch, 1, hidden]
        
        # Repete o conceito para cada posição da frase (para reconstruir)
        decoder_input = decoder_input.repeat(1, self.input_dim, 1) # -> [batch, seq, hidden]
        
        output, _ = self.decoder_lstm(decoder_input)
        logits = self.output_head(output) # -> [batch, seq, vocab_size]
        
        return logits