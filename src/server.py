from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
from config import REQUIRED_CLIENTS, CHAOS_SCENARIO, TEST_BATCH_SIZE

app = Flask(__name__)

# Buffer de pesos: dicionário garante apenas o último envio de cada cliente
round_buffer = {}
current_round = 0

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

def log_terminal(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
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
            log_terminal(f"📥 Recebido de: {node_id} | Aguardando parceiros... ({len(round_buffer)+1}/{REQUIRED_CLIENTS})")

        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        round_buffer[node_id] = full_weights_reconstructed
        
        log_to_db(node_id, len(str(data['weights'])), loss, current_round)
        
        if len(round_buffer) >= REQUIRED_CLIENTS:
            aggregate_weights()
            
        return jsonify({"status": "accepted", "round": current_round}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def evaluate_global_model():
    """Avalia o modelo global em imagens de teste e retorna MSE e PSNR"""
    try:
        from model_utils import ImageAutoencoder
        from image_utils import load_mnist, get_random_batch, compute_mse, compute_psnr
        
        model = ImageAutoencoder()
        model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
        model.eval()
        
        dataset = load_mnist(train=False)
        images, _ = get_random_batch(dataset, batch_size=TEST_BATCH_SIZE)
        
        with torch.no_grad():
            reconstructed = model(images)
        
        mse = compute_mse(images, reconstructed)
        psnr = compute_psnr(images, reconstructed)
        return mse, psnr
    except Exception as e:
        log_terminal(f"⚠️ Erro na avaliação: {e}")
        return None, None

def aggregate_weights():
    """FedAvg: média dos pesos de todos os clientes"""
    global round_buffer, current_round
    current_round += 1
    log_terminal(f"✨ Todos ({len(round_buffer)}) chegaram! Iniciando FedAvg... [Rodada {current_round}]")
    
    try:
        new_state_dict = {}
        first_client = next(iter(round_buffer.values()))
        keys = first_client.keys()
        
        for key in keys:
            tensors = [torch.tensor(client_weights[key]) for client_weights in round_buffer.values()]
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        
        torch.save(new_state_dict, "global_model.pth")
        
        log_terminal(f"💾 Modelo Global salvo! [Rodada {current_round}]")
        
        # Avaliação do modelo global
        mse, psnr = evaluate_global_model()
        if mse is not None:
            log_terminal(f"📊 Avaliação Global: MSE={mse:.6f} | PSNR={psnr:.1f} dB")
            log_round_metrics(current_round, mse, psnr)
        
        round_buffer.clear()
        log_terminal(f"🏁 Rodada {current_round} finalizada. Buffer limpo.\n")
        
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

def log_to_db(node_id, bytes_sent, loss, round_number):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO training_logs VALUES (?, ?, ?, ?, ?)", 
                     (timestamp, node_id, bytes_sent, loss, round_number))
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar no DB: {e}")

def log_round_metrics(round_number, global_mse, global_psnr):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scenario = os.environ.get("CHAOS_SCENARIO", CHAOS_SCENARIO)
        cursor.execute("INSERT INTO round_metrics VALUES (?, ?, ?, ?, ?)",
                     (round_number, global_mse, global_psnr, timestamp, scenario))
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar round_metrics: {e}")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write("=== SERVIDOR FL INICIADO ===\n")
    app.run(host='0.0.0.0', port=5000)
