import os
import time
import torch
import sqlite3
import json
from server import app as flask_app
from client import train_and_upload
from model_utils import ImageAutoencoder
from image_utils import load_mnist, load_mnist_filtered, get_random_batch
from config import NONIID_LABELS

def init_db():
    conn = sqlite3.connect('metrics.db', check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_mse REAL, global_psnr REAL, timestamp TEXT, chaos_scenario TEXT)")
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
        with open("status.json", "w") as f: json.dump({"status": "PAUSED"}, f)
        init_db()
        flask_app.run(host='0.0.0.0', port=5000)
    else:
        print(f"🤖 [{node_id}] Cliente de Imagens Iniciado.")
        
        # Carrega o autoencoder
        model = ImageAutoencoder()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{node_id}] Autoencoder carregado: {total_params:,} parâmetros.")
        
        # Carrega MNIST (Non-IID para client-noniid)
        if "noniid" in node_id:
            print(f"[{node_id}] Carregando MNIST Non-IID (dígitos {NONIID_LABELS})...")
            dataset = load_mnist_filtered(train=True, allowed_labels=NONIID_LABELS)
            print(f"[{node_id}] MNIST Non-IID carregado: {len(dataset)} imagens (dígitos {NONIID_LABELS}).")
        else:
            print(f"[{node_id}] Carregando MNIST...")
            dataset = load_mnist(train=True)
            print(f"[{node_id}] MNIST carregado: {len(dataset)} imagens.")
        
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
                
                # Pega batch aleatório de imagens
                images, labels = get_random_batch(dataset, batch_size=32)
                
                # Treina e envia pesos
                train_and_upload(model, images, server_url, node_id)
                
            except Exception as e:
                print(f"❌ {e}", flush=True)
            time.sleep(5)
