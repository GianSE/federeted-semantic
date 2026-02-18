import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from datetime import datetime
from text_utils import detokenize_text

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
    Mantém apenas os X% pesos mais fortes (maior valor absoluto).
    Zera o restante para economizar banda (simulado).
    """
    compressed_payload = {}
    
    for key, value in weight_dict.items():
        tensor = value.cpu() # Garante que está na CPU
        flattened = tensor.abs().flatten()
        num_params = flattened.numel()
        k = int(num_params * (1 - compression_ratio)) # Quantos vamos manter
        
        if k < 1: 
            # Se for muito pequeno, mantém tudo ou nada
            compressed_payload[key] = value.numpy().tolist()
            continue

        # Encontra o valor de corte (threshold) para estar no Top-K
        threshold_value = torch.kthvalue(flattened, num_params - k + 1).values.item()

        # Cria a máscara: Mantém só o que for maior que o threshold
        mask = tensor.abs() >= threshold_value
        
        # Aplica a máscara: O que for False vira None (para o JSON ficar leve/nulo)
        # Nota: Ao enviar 'None' em JSON, economizamos banda real se usarmos compressão gzip no transporte
        arr_numpy = value.numpy()
        compressed_payload[key] = np.where(mask.numpy(), np.round(arr_numpy, 4), None).tolist()
        
    return compressed_payload

def train_and_upload(model, data, targets, server_url, node_id, text=""):
    try:
        model_before = copy.deepcopy(model)
        
        # Otimizador Adam é melhor para LSTMs complexas
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        # Loss para classificação de texto (prevê qual letra é a próxima)
        criterion = nn.CrossEntropyLoss()
        model.train()

        # --- TESTE INICIAL ---
        with torch.no_grad():
            logits = model(data)
            reconstructed = detokenize_text(logits)
        
        log_terminal(f"👁️ Lê: '{text}'", node_id)
        log_terminal(f"🗣️ Diz: '{reconstructed}'", node_id)
        
        # --- TREINO ---
        log_terminal(f"🔥 Treinando Deep LSTM...", node_id)
        epochs = 10 # Modelos grandes aprendem rápido com poucos dados repetidos
        final_loss = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output_logits = model(data) # [1, 30, 128]
            
            # Ajusta formatos para CrossEntropy: (Batch*Seq, Vocab) vs (Batch*Seq)
            loss = criterion(output_logits.view(-1, 128), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        weight_change = calculate_weight_diff(model_before, model)
        log_terminal(f"🧠 Loss: {final_loss:.4f}", node_id)

        # --- ENVIO (Compressão Inteligente) ---
        full_weights = model.state_dict()
        
        if "full" not in node_id:
            log_terminal("⚠️ [GenIA] Aplicando Compressão Semântica (Top-K)...", node_id)
            # Envia apenas os 40% pesos mais importantes (60% de compressão)
            final_payload = compress_weights_top_k(full_weights, compression_ratio=0.6)
        else:
            # Cliente Full envia tudo normal
            final_payload = {k: v.cpu().numpy().tolist() for k, v in full_weights.items()}

        # ... dentro da função train_and_upload ...

        log_terminal(f"🚀 Enviando...", node_id)
        
        try:
            # ADICIONE O TIMEOUT=5 (segundos)
            requests.post(f"{server_url}/upload_weights", json={
                "client_id": node_id, 
                "weights": final_payload, 
                "loss": final_loss, 
                "node_id": node_id
            }, timeout=30) # <--- MUDANÇA AQUI: Se demorar > 30s, aborta.
            
            log_terminal("✅ Enviado.\n", node_id)

        # MUDANÇA AQUI: Agrupe Timeout e ConnectionError
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            log_terminal("⚠️ Rede instável (Timeout ou Falha de Conexão). Pulando...", node_id)
            
        except Exception as e:
            log_terminal(f"❌ Erro Crítico: {str(e)[:20]}...", node_id)

    except Exception as e:
        log_terminal(f"❌ Erro: {e}", node_id)