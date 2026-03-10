import os
import time
import torch
import sqlite3
import json
from server import app as flask_app
from client import train_and_upload
from model_utils import get_model
from text_utils import load_text_dataset, get_random_batch_text, get_label_subset, MAX_VOCAB_SIZE, MAX_SEQ_LEN
from config import MODEL_TYPE, LATENT_DIM, VAE_BETA, CHANNEL_SNR_DB

def init_db():
    conn = sqlite3.connect('metrics.db', check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_celoss REAL, global_accuracy REAL, timestamp TEXT, chaos_scenario TEXT)")
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

    vocab_size = MAX_VOCAB_SIZE
    seq_len = MAX_SEQ_LEN

    if mode == 'server':
        print("🚀 Servidor FL Iniciado...")
        print(f"   Modelo NLP: {MODEL_TYPE.upper()} | Latente: {LATENT_DIM}d | Vocab: {vocab_size}")
        with open("status.json", "w") as f: json.dump({"status": "PAUSED"}, f)
        init_db()
        flask_app.run(host='0.0.0.0', port=5000)
    else:
        print(f"🤖 [{node_id}] Cliente de Texto NLP Iniciado.")
        
        # Carrega o modelo de texto
        model = get_model(MODEL_TYPE, vocab_size=vocab_size, seq_len=seq_len, latent_dim=LATENT_DIM)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{node_id}] {MODEL_TYPE.upper()} Textual carregado: {total_params:,} parâmetros.")
        
        # Parse canal SNR
        channel_snr = float(CHANNEL_SNR_DB) if CHANNEL_SNR_DB else None
        if channel_snr is not None:
            print(f"[{node_id}] Canal AWGN ativo: SNR={channel_snr} dB")
        
        allowed_labels = get_label_subset(node_id)
        print(f"[{node_id}] Carregando corpus textual local...")
        dataset, vocab = load_text_dataset(train=True, allowed_labels=allowed_labels)
        print(f"[{node_id}] Dataset carregado: {len(dataset)} sequências (Vocab: {len(vocab)}).")
        
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
                
                # Pega batch aleatório de textos
                text_batch = get_random_batch_text(dataset, batch_size=32)
                
                # Treina e envia pesos (note que enviamos a tupla text_batch)
                train_and_upload(model, text_batch, server_url, node_id,
                                model_type=MODEL_TYPE, vae_beta=VAE_BETA,
                                channel_snr_db=channel_snr)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ {e}", flush=True)
            time.sleep(5)
