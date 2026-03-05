import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_terminal(msg, node_id="unknown"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(os.path.join(LOG_DIR, f"{node_id}.log"), "a") as f:
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

def train_and_upload(model, images, server_url, node_id):
    """Treina o autoencoder de imagens localmente e envia pesos ao servidor"""
    try:
        model_before = copy.deepcopy(model)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Reconstrução de imagens
        model.train()

        # --- TREINO LOCAL ---
        log_terminal(f"🔥 Treinando Autoencoder ({images.shape[0]} imagens)...", node_id)
        epochs = 5
        final_loss = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)  # Autoencoder: entrada = saída desejada
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Info de compressão semântica
        with torch.no_grad():
            z = model.encode(images)
            pixels = images.numel()
            latent_size = z.numel()
            compression = pixels / latent_size
        
        log_terminal(f"📊 Loss: {final_loss:.6f} | Compressão semântica: {compression:.0f}x ({latent_size} vs {pixels} valores)", node_id)

        # --- ENVIO DOS PESOS ---
        full_weights = model.state_dict()
        
        if "full" not in node_id:
            log_terminal("⚠️ Aplicando Compressão Top-K (60%)...", node_id)
            final_payload = compress_weights_top_k(full_weights, compression_ratio=0.6)
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
