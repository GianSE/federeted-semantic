import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from config import BATCH_SIZE, LOCAL_EPOCHS, LEARNING_RATE, COMPRESSION_RATIO, MAX_BATCHES_PER_EPOCH, COMPRESSED_CLIENTS

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


def should_compress_client(node_id):
    return node_id in COMPRESSED_CLIENTS

def train_and_upload(model, dataset, server_url, node_id, model_type="autoencoder", vae_beta=1.0, channel_snr_db=None):
    """Treina o modelo (AE ou VAE) localmente e envia pesos ao servidor"""
    try:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        model.train()

        # --- TREINO LOCAL ---
        is_vae = model_type == "vae"
        tag = "VAE" if is_vae else "Autoencoder"
        max_batches = MAX_BATCHES_PER_EPOCH if MAX_BATCHES_PER_EPOCH > 0 else None
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        final_loss = 0.0
        processed_images = 0
        last_batch = None

        log_terminal(
            f"🔥 Treinando {tag} ({len(dataset)} imagens locais, batch={BATCH_SIZE}, max_batches/época={max_batches or 'todos'})...",
            node_id,
        )

        for epoch in range(LOCAL_EPOCHS):
            epoch_losses = []
            for batch_idx, (images, _) in enumerate(train_loader, start=1):
                if max_batches is not None and batch_idx > max_batches:
                    break

                optimizer.zero_grad()
                if is_vae:
                    from model_utils import ImageVAE
                    reconstructed, mu, logvar = model(images, snr_db=channel_snr_db)
                    recon_loss = criterion(reconstructed, images)
                    kl_loss = ImageVAE.kl_divergence(mu, logvar)
                    loss = recon_loss + vae_beta * kl_loss
                else:
                    reconstructed = model(images, snr_db=channel_snr_db)
                    loss = criterion(reconstructed, images)

                loss.backward()
                optimizer.step()

                final_loss = loss.item()
                epoch_losses.append(final_loss)
                processed_images += images.size(0)
                last_batch = images

            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                log_terminal(f"   Época {epoch + 1}/{LOCAL_EPOCHS} | loss={avg_epoch_loss:.6f}", node_id)

        if last_batch is None:
            log_terminal("⚠️ Dataset local vazio. Treino ignorado.", node_id)
            return

        # Info de compressão semântica
        with torch.no_grad():
            if is_vae:
                mu, _ = model.encode(last_batch)
                z = mu
            else:
                z = model.encode(last_batch)
            pixels = last_batch.numel()
            latent_size = z.numel()
            compression = pixels / latent_size
        
        log_terminal(
            f"📊 Loss final: {final_loss:.6f} | Amostras vistas: {processed_images} | Compressão semântica: {compression:.0f}x ({latent_size} vs {pixels} valores)",
            node_id,
        )

        # --- ENVIO DOS PESOS ---
        full_weights = model.state_dict()
        
        if should_compress_client(node_id):
            log_terminal(f"⚠️ Aplicando Compressão Top-K ({int(COMPRESSION_RATIO * 100)}%)...", node_id)
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
