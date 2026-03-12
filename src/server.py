from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader
from config import REQUIRED_CLIENTS, CHAOS_SCENARIO, TEST_BATCH_SIZE, ROUND_TIMEOUT, MODEL_TYPE, LATENT_DIM, MODEL_INIT_SEED

app = Flask(__name__)

GLOBAL_MODEL_PATH = "global_model.pth"

# Buffer de pesos: dicionário garante apenas o último envio de cada cliente
round_buffer = {}
current_round = 0
round_start_time = None

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

def log_terminal(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


def initialize_global_model(force_reset=False):
    from model_utils import get_model

    if os.path.exists(GLOBAL_MODEL_PATH) and not force_reset:
        return False

    previous_state = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    model = get_model(MODEL_TYPE, LATENT_DIM)
    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
    torch.random.set_rng_state(previous_state)

    reason = "resetado" if force_reset else "inicializado"
    log_terminal(
        f"🧠 Modelo global {reason} com seed fixa {MODEL_INIT_SEED} em '{GLOBAL_MODEL_PATH}'."
    )
    return True


def evaluate_model(model, data_loader):
    total_sq_error = 0.0
    total_pixels = 0
    total_ssim = 0.0
    total_images = 0

    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            output = model(images)
            reconstructed = output[0] if isinstance(output, tuple) else output
            diff = images - reconstructed

            total_sq_error += torch.sum(diff * diff).item()
            total_pixels += diff.numel()

            batch_size = images.size(0)
            total_images += batch_size
            total_ssim += compute_ssim(images, reconstructed) * batch_size

    mse = total_sq_error / total_pixels if total_pixels else 0.0
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    ssim = total_ssim / total_images if total_images else 0.0
    return mse, psnr, ssim

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
    global round_start_time
    try:
        data = request.json
        node_id = data.get('node_id', 'unknown')
        loss = data.get('loss', 0)
        
        if node_id in round_buffer:
            log_terminal(f"♻️  {node_id} reenviou. Atualizando buffer...")
        else:
            log_terminal(f"📥 Recebido de: {node_id} | Aguardando parceiros... ({len(round_buffer)+1}/{REQUIRED_CLIENTS})")

        # Marca início da rodada quando primeiro cliente chega
        if not round_buffer:
            round_start_time = time.time()

        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        round_buffer[node_id] = full_weights_reconstructed
        
        log_to_db(node_id, len(str(data['weights'])), loss, current_round)
        
        # Agrega se todos chegaram OU se timeout expirou com pelo menos 1 cliente
        should_aggregate = len(round_buffer) >= REQUIRED_CLIENTS
        if not should_aggregate and round_start_time and len(round_buffer) >= 1:
            elapsed = time.time() - round_start_time
            if elapsed > ROUND_TIMEOUT:
                log_terminal(f"⏱️ Timeout ({elapsed:.0f}s) com {len(round_buffer)}/{REQUIRED_CLIENTS} clientes. Agregando parcial.")
                should_aggregate = True

        if should_aggregate:
            round_start_time = None
            aggregate_weights()
            
        return jsonify({"status": "accepted", "round": current_round}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def evaluate_global_model():
    """Avalia o modelo global em imagens de teste e retorna MSE, PSNR e SSIM"""
    try:
        from model_utils import get_model
        from image_utils import load_mnist, compute_ssim

        initialize_global_model(force_reset=False)
        
        model = get_model(MODEL_TYPE, LATENT_DIM)
        model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location="cpu", weights_only=True))
        
        dataset = load_mnist(train=False)
        test_loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
        mse, psnr, ssim = evaluate_model(model, test_loader)
        return mse, psnr, ssim
    except Exception as e:
        log_terminal(f"⚠️ Erro na avaliação: {e}")
        return None, None, None

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
        
        torch.save(new_state_dict, GLOBAL_MODEL_PATH)
        
        log_terminal(f"💾 Modelo Global salvo! [Rodada {current_round}]")
        
        # Avaliação do modelo global
        mse, psnr, ssim = evaluate_global_model()
        if mse is not None:
            log_terminal(f"📊 Avaliação Global: MSE={mse:.6f} | PSNR={psnr:.1f} dB | SSIM={ssim:.4f}")
            log_round_metrics(current_round, mse, psnr, ssim)
        
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
        from model_utils import get_model
        
        data = request.json
        latent = torch.tensor(data['latent'], dtype=torch.float32).unsqueeze(0)
        
        model = get_model(MODEL_TYPE, LATENT_DIM)
        initialize_global_model(force_reset=False)
        if os.path.exists(GLOBAL_MODEL_PATH):
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location="cpu"))
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
        from model_utils import get_model
        
        data = request.json
        partial = torch.tensor(data['image'], dtype=torch.float32)
        if partial.dim() == 2:
            partial = partial.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        elif partial.dim() == 3:
            partial = partial.unsqueeze(0)
        
        model = get_model(MODEL_TYPE, LATENT_DIM)
        initialize_global_model(force_reset=False)
        if os.path.exists(GLOBAL_MODEL_PATH):
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            output = model(partial)
            if isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output
        
        return jsonify({
            "status": "ok",
            "image": recon.squeeze().numpy().tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset_round', methods=['POST'])
def reset_round():
    global round_buffer, current_round, round_start_time
    round_buffer = {}
    current_round = 0
    round_start_time = None
    initialize_global_model(force_reset=True)
    log_terminal("🔄 Round counter, buffer e modelo global resetados.")
    return jsonify({"status": "reset", "round": 0}), 200

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

def log_round_metrics(round_number, global_mse, global_psnr, global_ssim=None):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scenario = os.environ.get("CHAOS_SCENARIO", CHAOS_SCENARIO)
        cursor.execute("INSERT INTO round_metrics VALUES (?, ?, ?, ?, ?, ?)",
                     (round_number, global_mse, global_psnr, timestamp, scenario, global_ssim or 0.0))
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar round_metrics: {e}")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write("=== SERVIDOR FL INICIADO ===\n")
    initialize_global_model(force_reset=False)
    app.run(host='0.0.0.0', port=5000)
