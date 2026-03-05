from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

app = Flask(__name__)

# Buffer de pesos: dicionário garante apenas o último envio de cada cliente
round_buffer = {}
REQUIRED_CLIENTS = 2

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

def log_terminal(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def reconstruct_weights(partial_weights, node_id):
    """Reconstrói pesos comprimidos (Top-K) substituindo NaN pela média"""
    reconstructed = {}
    missing_count = 0
    total_count = 0
    
    for key, value_list in partial_weights.items():
        arr = np.array(value_list, dtype=np.float32)
        total_count += arr.size
        
        if arr.dtype == object:
            arr = arr.astype(float)
            
        nans = np.isnan(arr)
        missing = np.sum(nans)
        missing_count += missing
        
        if np.isnan(arr).all():
            col_mean = 0.0
        else:
            col_mean = np.nanmean(arr)
        
        arr[nans] = col_mean
        reconstructed[key] = arr
        
    if missing_count > 0:
        percent = (missing_count / total_count) * 100
        log_terminal(f"🔧 Reconstruindo pesos de {node_id}: {percent:.1f}% faltante preenchido por média")

    return reconstructed

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    try:
        data = request.json
        node_id = data.get('node_id', 'unknown')
        loss = data.get('loss', 0)
        
        if node_id in round_buffer:
            log_terminal(f"♻️  {node_id} reenviou. Atualizando buffer...")
        else:
            log_terminal(f"📥 Recebido de: {node_id} | Aguardando parceiros...")

        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        round_buffer[node_id] = full_weights_reconstructed
        
        log_to_db(node_id, len(str(data['weights'])), loss)
        
        if len(round_buffer) >= REQUIRED_CLIENTS:
            aggregate_weights()
            
        return jsonify({"status": "accepted"}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def aggregate_weights():
    """FedAvg: média dos pesos de todos os clientes"""
    global round_buffer
    log_terminal(f"✨ Todos ({len(round_buffer)}) chegaram! Iniciando FedAvg...")
    
    try:
        new_state_dict = {}
        first_client = next(iter(round_buffer.values()))
        keys = first_client.keys()
        
        for key in keys:
            tensors = [torch.tensor(client_weights[key]) for client_weights in round_buffer.values()]
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        
        torch.save(new_state_dict, "global_model.pth")
        
        log_terminal(f"💾 Modelo Global salvo!")
        
        round_buffer.clear()
        log_terminal("🏁 Rodada finalizada. Buffer limpo.\n")
        
    except Exception as e:
        log_terminal(f"❌ Erro na agregação: {e}")

@app.route('/reconstruct', methods=['POST'])
def reconstruct_image():
    """
    Endpoint de comunicação semântica:
    Recebe vetor latente (32 dims) e retorna imagem reconstruída (28x28).
    Simula: Dispositivo A envia representação comprimida → Dispositivo B reconstrói.
    """
    try:
        from model_utils import ImageAutoencoder
        
        data = request.json
        latent = torch.tensor(data['latent'], dtype=torch.float32).unsqueeze(0)
        
        model = ImageAutoencoder()
        if os.path.exists("global_model.pth"):
            model.load_state_dict(torch.load("global_model.pth", map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            recon = model.decode(latent)
        
        return jsonify({
            "status": "ok",
            "image": recon.squeeze().numpy().tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/complete', methods=['POST'])
def complete_image():
    """
    Endpoint de completação de imagem:
    Recebe imagem parcial (mascarada) → Autoencoder reconstrói imagem completa.
    Simula: envio de pedaços da imagem, modelo completa o restante.
    """
    try:
        from model_utils import ImageAutoencoder
        
        data = request.json
        partial = torch.tensor(data['image'], dtype=torch.float32)
        if partial.dim() == 2:
            partial = partial.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        elif partial.dim() == 3:
            partial = partial.unsqueeze(0)
        
        model = ImageAutoencoder()
        if os.path.exists("global_model.pth"):
            model.load_state_dict(torch.load("global_model.pth", map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            recon = model(partial)  # Encode parcial → Decode completo
        
        return jsonify({
            "status": "ok",
            "image": recon.squeeze().numpy().tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def log_to_db(node_id, bytes_sent, loss):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO training_logs VALUES (?, ?, ?, ?)", 
                     (timestamp, node_id, bytes_sent, loss))
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar no DB: {e}")

if __name__ == "__main__":
    with open(LOG_FILE, "w") as f: f.write("=== SERVIDOR FL INICIADO ===\n")
    app.run(host='0.0.0.0', port=5000)
