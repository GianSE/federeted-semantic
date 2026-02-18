import streamlit as st
import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta # <--- Adicione timedelta
import os
import json
import torch
import numpy as np

# --- CONTROLE DO CAOS ---
st.sidebar.divider()
st.sidebar.subheader("🔥 Controle de Caos (Network)")

# Checkbox para ligar/desligar
chaos_active = st.sidebar.toggle("Ativar Instabilidade", value=True)

# --- LÓGICA DE SINCRONIZAÇÃO (Callback) ---
if 'loss_val' not in st.session_state: st.session_state.loss_val = 0.05
if 'delay_val' not in st.session_state: st.session_state.delay_val = 300

def update_loss_slider(): st.session_state.loss_val = st.session_state.loss_slider
def update_loss_input(): st.session_state.loss_val = st.session_state.loss_input
def update_delay_slider(): st.session_state.delay_val = st.session_state.delay_slider
def update_delay_input(): st.session_state.delay_val = st.session_state.delay_input

# --- 1. PERDA DE PACOTES (LOSS) ---
st.sidebar.write("Perda de Pacotes (%)")
col1, col2 = st.sidebar.columns([3, 1]) # Coluna larga (Slider) e estreita (Input)

with col1:
    st.slider(
        "", min_value=0.00, max_value=5.00, step=0.01,
        key='loss_slider', value=st.session_state.loss_val, 
        on_change=update_loss_slider, format="%.2f"
    )

with col2:
    st.number_input(
        "", min_value=0.00, max_value=5.00, step=0.01,
        key='loss_input', value=st.session_state.loss_val, 
        on_change=update_loss_input, label_visibility="collapsed"
    )

# --- 2. LATÊNCIA (DELAY) ---
st.sidebar.write("Latência (ms)")
col3, col4 = st.sidebar.columns([3, 1])

with col3:
    st.slider(
        "", min_value=0, max_value=2000, step=50,
        key='delay_slider', value=st.session_state.delay_val,
        on_change=update_delay_slider
    )

with col4:
    st.number_input(
        "", min_value=0, max_value=2000, step=50,
        key='delay_input', value=st.session_state.delay_val,
        on_change=update_delay_input, label_visibility="collapsed"
    )

# --- BOTÃO DE APLICAR ---
if st.sidebar.button("⚡ Aplicar Caos"):
    status = "ON" if chaos_active else "OFF"
    # Pega o valor da session_state que está sincronizado
    loss = st.session_state.loss_val
    delay = st.session_state.delay_val
    
    config_str = f"{status} {loss:.2f}% {delay}ms"
    
    with open("chaos_config.txt", "w") as f:
        f.write(config_str)
        
    if chaos_active:
        st.sidebar.error(f"Caos ATIVADO: {loss:.2f}% / {delay}ms")
    else:
        st.sidebar.success("Rede Normalizada")

try:
    from model_utils import SemanticAutoencoder
    from text_utils import tokenize_text, detokenize_text
except ImportError: pass

st.set_page_config(layout="wide", page_title="GenIA Deep", page_icon="🧠")
st.markdown("<style>.terminal-container {height: 400px; overflow-y: scroll; background-color: #0e1117; color: #00ff00; font-family: monospace; padding: 10px; white-space: pre-wrap;}</style>", unsafe_allow_html=True)

STATUS_FILE = "status.json"
DATASET_FILE = "dataset.txt"

def set_status(s): 
    with open(STATUS_FILE, "w") as f: json.dump({"status": s}, f)

st.sidebar.title("🧠 GenIA Deep Control")
status = "PAUSED"
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE) as f: status = json.load(f).get("status", "PAUSED")

if status == "RUNNING":
    st.sidebar.success("🟢 RODANDO")
    if st.sidebar.button("⏸️ PAUSE"): set_status("PAUSED"); st.rerun()
else:
    st.sidebar.warning("⏸️ PAUSADO")
    if st.sidebar.button("▶️ PLAY"): set_status("RUNNING"); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📚 Dataset (Professor)")
new_phrase = st.sidebar.text_input("Frase:", placeholder="GenIA é o futuro")
if st.sidebar.button("💾 Salvar"):
    if new_phrase:
        with open(DATASET_FILE, "a") as f: f.write(new_phrase + "\n")
        st.sidebar.success("Salvo!")

# Na sidebar
if st.sidebar.button("🗑️ Limpar Histórico (Reset DB)"):
    try:
        # 1. Apaga o arquivo antigo se existir
        if os.path.exists("metrics.db"):
            os.remove("metrics.db")
        
        # 2. O PULO DO GATO: Recria a tabela imediatamente
        # Isso garante que o próximo SELECT ou INSERT encontre o lugar certo
        conn = sqlite3.connect('metrics.db')
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                timestamp TEXT, 
                node_id TEXT, 
                bytes_sent INTEGER, 
                loss REAL
            )
        """)
        conn.commit()
        conn.close()
        
        st.sidebar.success("Histórico resetado! Tabela recriada.")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"Erro ao resetar: {e}")

st.sidebar.divider()
st.sidebar.subheader("🧪 Teste Neural")
user_text = st.sidebar.text_input("Erro:", "GenIA e o futuru")
if st.sidebar.button("Reconstruir"):
    if os.path.exists("global_model.pth"):
        try:
            model = SemanticAutoencoder()
            model.load_state_dict(torch.load("global_model.pth"))
            model.eval()
            
            t_in = tokenize_text(user_text)
            with torch.no_grad(): logits = model(t_in)
            rec = detokenize_text(logits)
            
            st.sidebar.write("---")
            c1, c2 = st.sidebar.columns(2)
            c1.text("Entrada:"); c1.code(user_text)
            c2.text("Saída:"); c2.code(rec)
        except Exception as e: st.sidebar.error(f"Erro: {e}")
    else: st.sidebar.warning("Sem modelo global ainda.")

st.title("🛰️ Federated Learning: Deep LSTM Architecture")

def read_log(f):
    p = os.path.join("logs", f)
    if os.path.exists(p): 
        with open(p) as file: 
            # 1. Lê todas as linhas
            lines = file.readlines()
            
            # 2. Pega as últimas 50 (para não ficar pesado)
            last_lines = lines[-50:]
            
            # 3. INVERTE a ordem ([::-1]) para o mais novo ficar no topo
            return "".join(last_lines[::-1])
            
    return "..."

# Em src/dashboard.py

@st.fragment(run_every=1)
def update():
    t1, t2, t3 = st.tabs(["🧠 SERVER", "🔵 FULL", "🟠 NOISY"])
    with t1: st.markdown(f'<div class="terminal-container">{read_log("server.log")}</div>', unsafe_allow_html=True)
    with t2: st.markdown(f'<div class="terminal-container">{read_log("client-full.log")}</div>', unsafe_allow_html=True)
    with t3: st.markdown(f'<div class="terminal-container">{read_log("client-noisy.log")}</div>', unsafe_allow_html=True)

    try:
        # Verifica se o arquivo existe antes de tentar conectar
        if not os.path.exists('metrics.db'):
            st.info("⏳ Aguardando criação do banco de dados...")
            return

        c = sqlite3.connect('metrics.db', check_same_thread=False)
        
        # Tenta ler. Se a tabela não existir, o pandas vai falhar, então tratamos isso.
        try:
            time_threshold = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            query = f"SELECT timestamp, node_id, loss FROM training_logs WHERE timestamp > '{time_threshold}' ORDER BY timestamp ASC"
            df = pd.read_sql(query, c)
        except Exception:
            # Se der erro (tabela sumiu), retornamos DataFrame vazio para não quebrar a tela
            df = pd.DataFrame()
            
        c.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Não precisa mais ordenar aqui pois o SQL já fez (ORDER BY ASC)
            # Pivot table
            chart_data = df.pivot_table(index='timestamp', columns='node_id', values='loss')
            
            # Interpolação para ligar os pontos do Noisy
            chart_data = chart_data.interpolate(method='time')
            
            # Limpa NaNs residuais
            chart_data = chart_data.ffill().bfill()
            
            st.line_chart(chart_data)
        else:
            st.info("⏳ Aguardando dados da sessão atual...")
            
    except Exception as e: 
        st.error(f"Erro no gráfico: {e}")

update()