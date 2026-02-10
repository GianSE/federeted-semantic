import os
import time
import torch
import sqlite3
import random
import json

from server import app as flask_app
from client import train_and_upload
from model_utils import SemanticAutoencoder 
from text_utils import tokenize_text

def init_db():
    db_file = 'metrics.db'
    conn = sqlite3.connect(db_file, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS network_metrics (node_id TEXT, bytes_sent INTEGER)")
    conn.commit()
    conn.close()
    print("✅ Banco de dados 'metrics.db' (SQLite) pronto!")

def is_paused():
    # Se o arquivo não existe, considera PAUSADO
    if not os.path.exists("status.json"): return True
    try:
        with open("status.json", "r") as f:
            data = json.load(f)
            return data.get("status") == "PAUSED"
    except: return True

if __name__ == "__main__":
    mode = os.environ.get('MODE', 'server')
    node_id = os.environ.get('HOSTNAME', 'node_unknown')
    server_url = os.environ.get('SERVER_URL', 'http://fl-server:5000')

    if mode == 'server':
        print("🚀 Inicializando Servidor FL (Autoencoder)...")
        
        # --- CORREÇÃO AQUI: FORÇA O PAUSE NO INÍCIO ---
        print("⏸️  Forçando estado inicial: PAUSED")
        with open("status.json", "w") as f:
            json.dump({"status": "PAUSED"}, f)
        # ---------------------------------------------

        init_db()
        flask_app.run(host='0.0.0.0', port=5000)
    else:
        print(f"🤖 [{node_id}] Cliente GenIA iniciado.")
        
        try:
            model = SemanticAutoencoder(input_dim=10)
        except:
            print("⚠️ Erro ao carregar modelo. Usando fallback...")
            # Fallback simples caso dê erro na importação
            from model_utils import SemanticAutoencoder
            model = SemanticAutoencoder(input_dim=10)
            
        print(f"[{node_id}] Aguardando comandos...")
        
        while True:
            # Verifica o pause antes de qualquer coisa
            if is_paused():
                print(f"[{node_id}] ⏸️ Aguardando Play (Adicione frases no Dashboard)...", flush=True)
                time.sleep(3)
                continue 
            
            try:
                if os.path.exists("dataset.txt"):
                    with open("dataset.txt", "r") as f: lines = f.readlines()
                    if lines:
                        text = random.choice(lines).strip()
                        if text:
                            # 1. Entrada
                            data = tokenize_text(text)
                            
                            # 2. Alvo = A própria entrada (Autoencoder)
                            targets = data.clone()
                            
                            train_and_upload(model, data, targets, server_url, node_id, text=text)
                    else:
                        # Se não tiver dataset, não faz nada (mas não trava)
                        pass
                else:
                    with open("dataset.txt", "w") as f: pass
            except Exception as e:
                print(f"❌ [{node_id}] Erro: {e}", flush=True)
            
            time.sleep(5)