from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

app = Flask(__name__)

# MUDANÇA 1: Usar Dicionário em vez de Lista
# Isso garante que só guardamos o ULTIMO peso de cada cliente específico
round_buffer = {} 
REQUIRED_CLIENTS = 2 # Quantos clientes únicos precisamos para fechar a rodada

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

def log_terminal(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

# Se você já aplicou a "Agregação Mascarada" (do passo anterior), mantenha aquela lógica.
# Esta função abaixo é a versão compatível com seu código original/atual.
def reconstruct_weights(partial_weights, node_id):
    reconstructed = {}
    missing_count = 0
    total_count = 0
    
    for key, value_list in partial_weights.items():
        arr = np.array(value_list, dtype=np.float32)
        total_count += arr.size
        
        # Se for None (do envio comprimido Top-K ou random), tratamos aqui
        # Se o array vier com None/NaN, substituímos por 0 ou média
        if arr.dtype == object: # Caso tenha Nones misturados
            arr = arr.astype(float) # Converte Nones para Nan
            
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

    return reconstructed

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    try:
        data = request.json
        node_id = data.get('node_id', 'unknown')
        loss = data.get('loss', 0)
        
        # MUDANÇA 2: Lógica de Buffer
        # Se esse cliente já mandou nessa rodada, avisamos que foi atualizado
        if node_id in round_buffer:
            log_terminal(f"♻️  {node_id} enviou de novo (Full é rápido!). Atualizando buffer...")
        else:
            log_terminal(f"cw  Recebido de: {node_id} | Aguardando parceiros...")

        # Processa e guarda no dicionário (sobrescreve se já existir)
        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        round_buffer[node_id] = full_weights_reconstructed
        
        log_to_db(node_id, len(str(data['weights'])), loss)
        
        # Verifica se temos todos os clientes ÚNICOS necessários
        if len(round_buffer) >= REQUIRED_CLIENTS:
            aggregate_weights()
            
        return jsonify({"status": "accepted"}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def aggregate_weights():
    global round_buffer
    log_terminal(f"✨ Todos ({len(round_buffer)}) chegaram! Iniciando FedAvg...")
    
    try:
        new_state_dict = {}
        # Pega as chaves do primeiro cliente do buffer
        first_client = next(iter(round_buffer.values()))
        keys = first_client.keys()
        
        for key in keys:
            # Pega os tensores de TODOS os clientes no buffer
            tensors = [torch.tensor(client_weights[key]) for client_weights in round_buffer.values()]
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        
        torch.save(new_state_dict, "global_model.pth")
        
        log_terminal(f"💾 Modelo Global v{datetime.now().strftime('%H%M%S')} salvo!")
        
        # MUDANÇA 3: Limpar o Dicionário para a próxima rodada
        round_buffer.clear() 
        log_terminal("🏁 Rodada finalizada. Buffer limpo.\n")
        
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
    with open(LOG_FILE, "w") as f: f.write("=== SERVIDOR SINCRONIZADO INICIADO ===\n")
    app.run(host='0.0.0.0', port=5000)