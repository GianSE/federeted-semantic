import os
import time
import torch
import sqlite3
import json
from server import app as flask_app
from client import train_and_upload
from model_utils import get_model
from image_utils import load_dataset, load_dataset_filtered
from config import NONIID_LABELS, MODEL_TYPE, LATENT_DIM, VAE_BETA, CHANNEL_SNR_DB, DATASET_DISPLAY_NAME, LABEL_KIND

def init_db():
    conn = sqlite3.connect('metrics.db', check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_mse REAL, global_psnr REAL, timestamp TEXT, chaos_scenario TEXT, global_ssim REAL)")
    conn.commit(); conn.close()
    print("✅ DB Pronto!")

def is_paused():
    if not os.path.exists("status.json"): return True
    try:
        with open("status.json") as f: return json.load(f).get("status") == "PAUSED"
    except: return True

if __name__ == "__main__":
    mode = os.environ.get('MODE', 'server')
    node_id = os.environ.get('HOSTNAME', 'node_unknown')
    server_url = os.environ.get('SERVER_URL', 'http://fl-server:5000')

    if mode == 'server':
        print("🚀 Servidor FL Iniciado...")
        print(f"   Modelo: {MODEL_TYPE.upper()} | Latente: {LATENT_DIM}d")
        with open("status.json", "w") as f: json.dump({"status": "PAUSED"}, f)
        init_db()
        flask_app.run(host='0.0.0.0', port=5000)
    else:
        print(f"🤖 [{node_id}] Cliente de Imagens Iniciado.")
        
        # Carrega o modelo (AE ou VAE)
        model = get_model(MODEL_TYPE, LATENT_DIM)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{node_id}] {MODEL_TYPE.upper()} carregado: {total_params:,} parâmetros.")
        
        # Parse canal SNR
        channel_snr = float(CHANNEL_SNR_DB) if CHANNEL_SNR_DB else None
        if channel_snr is not None:
            print(f"[{node_id}] Canal AWGN ativo: SNR={channel_snr} dB")
        
        # Carrega dataset configurado (Non-IID para client-noniid)
        if "noniid" in node_id:
            print(f"[{node_id}] Carregando {DATASET_DISPLAY_NAME} Non-IID ({LABEL_KIND}s {NONIID_LABELS})...")
            dataset = load_dataset_filtered(train=True, allowed_labels=NONIID_LABELS)
            print(f"[{node_id}] {DATASET_DISPLAY_NAME} Non-IID carregado: {len(dataset)} imagens ({LABEL_KIND}s {NONIID_LABELS}).")
        else:
            print(f"[{node_id}] Carregando {DATASET_DISPLAY_NAME}...")
            dataset = load_dataset(train=True)
            print(f"[{node_id}] {DATASET_DISPLAY_NAME} carregado: {len(dataset)} imagens.")
        
        while True:
            if is_paused():
                print(f"[{node_id}] ⏸️ Pausado...", flush=True)
                time.sleep(3)
                continue 
            
            try:
                # Carrega modelo global se existir
                if os.path.exists("global_model.pth"):
                    try:
                        state = torch.load("global_model.pth", map_location="cpu")
                        model.load_state_dict(state)
                    except Exception:
                        pass

                # Treina e envia pesos
                train_and_upload(model, dataset, server_url, node_id,
                                model_type=MODEL_TYPE, vae_beta=VAE_BETA,
                                channel_snr_db=channel_snr)
                
            except Exception as e:
                print(f"❌ {e}", flush=True)
            time.sleep(5)
