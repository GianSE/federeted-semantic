import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from datetime import datetime

from config import BATCH_SIZE, COMPRESSION_RATIO, LEARNING_RATE, LOCAL_EPOCHS

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_terminal(msg, node_id="unknown"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(os.path.join(LOG_DIR, f"{node_id}.log"), "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

def calculate_weight_diff(model_before, model_after):
    diff = 0.0
    for p_before, p_after in zip(model_before.parameters(), model_after.parameters()):
        diff += torch.norm(p_after - p_before).item()
    return diff

def compress_weights_top_k(weight_dict, compression_ratio=0.5):
    """
    Mantém apenas os X% pesos mais significativos (maior valor absoluto).
    Zera o restante para economizar banda de comunicação.
    """
    compressed_payload = {}
    
    for key, value in weight_dict.items():
        tensor = value.cpu()
        flattened = tensor.abs().flatten()
        num_params = flattened.numel()
        k = int(num_params * (1 - compression_ratio))
        
        if k < 1:
            compressed_payload[key] = value.numpy().tolist()
            continue

        threshold_value = torch.kthvalue(flattened, num_params - k + 1).values.item()
        mask = tensor.abs() >= threshold_value
        
        arr_numpy = value.numpy()
        compressed_payload[key] = np.where(mask.numpy(), np.round(arr_numpy, 4), None).tolist()
        
    return compressed_payload

def train_and_upload(model, text_batch, server_url, node_id, model_type="autoencoder", vae_beta=1.0, channel_snr_db=None):
    """Treina o modelo (AE ou VAE textual) localmente e envia pesos ao servidor"""
    try:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        model.train()

        # --- TREINO LOCAL ---
        is_vae = model_type == "vae"
        tag = "TextVAE" if is_vae else "TextAutoencoder"
        
        # text_batch é uma tupla (inputs, targets) que para o AE são idênticos: (batch_size, seq_len)
        inputs, targets = text_batch
        
        log_terminal(f"🔥 Treinando {tag} ({inputs.shape[0]} mensagens)...", node_id)
        final_loss = 0
        
        for _ in range(LOCAL_EPOCHS):
            optimizer.zero_grad()
            if is_vae:
                from model_utils import TextVAE
                logits, mu, logvar = model(inputs, snr_db=channel_snr_db)
                
                # Reshape para CrossEntropy: (batch_size * seq_len, vocab_size) e (batch_size * seq_len)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                recon_loss = criterion(logits_flat, targets_flat)
                kl_loss = TextVAE.kl_divergence(mu, logvar)
                loss = recon_loss + vae_beta * kl_loss
            else:
                logits = model(inputs, snr_db=channel_snr_db)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Info de compressão semântica
        with torch.no_grad():
            if is_vae:
                mu, _ = model.encode(inputs)
                z = mu
            else:
                z = model.encode(inputs)
            
            # Número de tokens de entrada vs tamanho do vetor latente
            tokens = inputs.numel()
            latent_size = z.numel()
            compression = tokens / latent_size
        
        log_terminal(f"📊 Loss (CE): {final_loss:.4f} | Compressão semântica: {compression:.0f}x ({latent_size} dims vs {tokens} tokens)", node_id)

        # --- ENVIO DOS PESOS ---
        full_weights = model.state_dict()
        
        if "noisy" in node_id:
            log_terminal(f"⚠️ Aplicando Compressão Top-K ({COMPRESSION_RATIO:.0%})...", node_id)
            final_payload = compress_weights_top_k(full_weights, compression_ratio=COMPRESSION_RATIO)
        else:
            final_payload = {k: v.cpu().numpy().tolist() for k, v in full_weights.items()}

        log_terminal(f"🚀 Enviando pesos ao servidor...", node_id)
        
        try:
            requests.post(f"{server_url}/upload_weights", json={
                "client_id": node_id, 
                "weights": final_payload, 
                "loss": final_loss, 
                "node_id": node_id
            }, timeout=30)
            log_terminal(f"✅ Pesos enviados com sucesso!", node_id)
        except requests.exceptions.Timeout:
            log_terminal(f"⏰ Timeout ao enviar (rede instável)!", node_id)
        except requests.exceptions.ConnectionError:
            log_terminal(f"❌ Erro de conexão com o servidor!", node_id)
            
    except Exception as e:
        log_terminal(f"❌ Erro no treino: {e}", node_id)
