from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

app = Flask(__name__)
received_weights = []

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
    reconstructed = {}
    missing_count = 0
    total_count = 0
    
    for key, value_list in partial_weights.items():
        arr = np.array(value_list, dtype=np.float32)
        total_count += arr.size
        
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
        log_terminal(f"🔧 GenIA: {node_id} enviou comprimido. Reconstruindo {percent:.1f}%...")
    else:
        log_terminal(f"📥 Recebido pacote completo de {node_id}.")

    return reconstructed

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    try:
        data = request.json
        node_id = data.get('node_id', 'unknown')
        loss = data.get('loss', 0)
        
        log_terminal(f"📡 Conexão de: {node_id} | Loss: {loss:.4f}")
        
        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        received_weights.append(full_weights_reconstructed)
        
        log_to_db(node_id, len(str(data['weights'])), loss)
        
        if len(received_weights) >= 2:
            aggregate_weights()
            
        return jsonify({"status": "accepted"}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def aggregate_weights():
    global received_weights
    log_terminal("✨ Iniciando Agregação (FedAvg)...")
    try:
        new_state_dict = {}
        keys = received_weights[0].keys()
        for key in keys:
            tensors = [torch.tensor(client[key]) for client in received_weights]
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        
        # --- O PULO DO GATO: SALVAR O MODELO ---
        torch.save(new_state_dict, "global_model.pth")
        
        log_terminal(f"💾 Modelo Global salvo em 'global_model.pth'!")
        log_terminal(f"🧠 Conhecimento Agregado! (2 clientes)")
        log_terminal("♻️  Limpando buffer...\n")
        received_weights = [] 
    except Exception as e:
        log_terminal(f"❌ Erro na agregação: {e}")

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
    with open(LOG_FILE, "w") as f: f.write("=== SERVIDOR INICIADO ===\n")
    app.run(host='0.0.0.0', port=5000)