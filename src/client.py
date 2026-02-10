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

def train_and_upload(model, data, targets, server_url, node_id, text=""):
    try:
        model_before = copy.deepcopy(model)
        
        # Otimizador Adam é melhor para LSTMs complexas
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        # Loss para classificação de texto (prevê qual letra é a próxima)
        criterion = nn.CrossEntropyLoss(ignore_index=0) 
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

        # --- ENVIO (Compressão) ---
        full_weights = model.state_dict()
        final_payload = {}

        if "full" not in node_id:
            log_terminal("⚠️ [GenIA] Comprimindo Matrizes (Dropout)...", node_id)
            for key, value in full_weights.items():
                arr = value.cpu().numpy()
                # Em matrizes gigantes, comprimir 50% é muita economia!
                mask = np.random.choice([True, False], size=arr.shape, p=[0.5, 0.5])
                final_payload[key] = np.where(mask, arr, None).tolist()
        else:
            final_payload = {k: v.cpu().numpy().tolist() for k, v in full_weights.items()}

        log_terminal(f"🚀 Enviando...", node_id)
        
        try:
            requests.post(f"{server_url}/upload_weights", json={
                "client_id": node_id, "weights": final_payload, "loss": final_loss, "node_id": node_id
            })
            log_terminal("✅ Enviado.\n", node_id)
        except:
            log_terminal("❌ Falha envio.\n", node_id)

    except Exception as e:
        log_terminal(f"❌ Erro: {e}", node_id)