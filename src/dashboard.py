import streamlit as st
import sqlite3
import pandas as pd
import time
import os
import json
import torch
import numpy as np

# Tenta importar as utils
try:
    from model_utils import SemanticAutoencoder
    from text_utils import tokenize_text, detokenize_text
except ImportError:
    pass

st.set_page_config(layout="wide", page_title="Monitoramento FL", page_icon="📡")

# --- CSS PARA TERMINAL COM SCROLL ---
st.markdown("""
<style>
    .terminal-container {
        height: 400px;
        overflow-y: scroll;
        background-color: #0e1117;
        color: #00ff00;
        font-family: monospace;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- CONTROLE DE ESTADO ---
STATUS_FILE = "status.json"
DATASET_FILE = "dataset.txt"

def get_status():
    if not os.path.exists(STATUS_FILE): return "PAUSED"
    try:
        with open(STATUS_FILE, "r") as f: return json.load(f).get("status", "PAUSED")
    except: return "PAUSED"

def set_status(status):
    with open(STATUS_FILE, "w") as f: json.dump({"status": status}, f)

# --- SIDEBAR ---
st.sidebar.title("🎛️ Painel de Controle")
status = get_status()

if status == "RUNNING":
    st.sidebar.success("Estado: 🟢 TREINANDO")
    if st.sidebar.button("⏸️ PAUSAR"):
        set_status("PAUSED")
        st.rerun()
else:
    st.sidebar.warning("Estado: ⏸️ PAUSADO")
    if st.sidebar.button("▶️ INICIAR"):
        set_status("RUNNING")
        st.rerun()

st.sidebar.divider()

# --- PROFESSOR ---
st.sidebar.subheader("📚 Ensinar a IA (Dataset)")
new_phrase = st.sidebar.text_input("Frase correta para treino:", placeholder="Ex: Federated")

if st.sidebar.button("💾 Adicionar"):
    if new_phrase:
        with open(DATASET_FILE, "a") as f: f.write(new_phrase + "\n")
        st.sidebar.success(f"Adicionado!")
    else:
        st.sidebar.warning("Digite algo!")

if os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "r") as f: lines = f.readlines()
    st.sidebar.caption(f"Frases no cérebro: {len(lines)}")

st.sidebar.divider()

# --- TESTE ---
st.sidebar.subheader("🧪 Teste de Correção")
user_text = st.sidebar.text_input("Tente enganar a IA:", "Federeted")

if st.sidebar.button("Verificar"):
    if os.path.exists("global_model.pth"):
        try:
            model = SemanticAutoencoder(input_dim=10)
            model.load_state_dict(torch.load("global_model.pth"))
            model.eval()
            tensor_in = tokenize_text(user_text)
            with torch.no_grad(): tensor_out = model(tensor_in)
            reconstructed = detokenize_text(tensor_out)
            
            st.sidebar.write("---")
            c1, c2 = st.sidebar.columns(2)
            c1.text("Entrada:"); c1.code(user_text)
            c2.text("IA Diz:"); c2.code(reconstructed)
        except Exception as e: st.sidebar.error(f"Erro: {e}")
    else: st.sidebar.warning("Modelo ainda não salvo.")

# --- DASHBOARD ---
st.title("🛰️ Centro de Comando FL + GenIA")

def read_log_tail(filename, lines=30):
    filepath = os.path.join("logs", filename)
    if not os.path.exists(filepath): return "⏳ Aguardando logs..."
    try:
        with open(filepath, "r") as f: 
            return "".join(f.readlines()[-lines:])
    except: return "⚠️ Lendo..."

# --- FRAGMENTO DE AUTO-ATUALIZAÇÃO (A MÁGICA) ---
# run_every=1 garante que esta função rode a cada 1s sem recarregar a página toda
@st.fragment(run_every=1)
def atualizar_paineis():
    # 1. Logs
    logs_server = read_log_tail("server.log")
    logs_full = read_log_tail("client-full.log")
    logs_noisy = read_log_tail("client-noisy.log")

    tab1, tab2, tab3 = st.tabs(["🧠 SERVER", "🔵 Client Full", "🟠 Client Noisy"])
    
    # Usamos HTML direto para garantir scroll e cor
    with tab1: st.markdown(f'<div class="terminal-container">{logs_server}</div>', unsafe_allow_html=True)
    with tab2: st.markdown(f'<div class="terminal-container">{logs_full}</div>', unsafe_allow_html=True)
    with tab3: st.markdown(f'<div class="terminal-container">{logs_noisy}</div>', unsafe_allow_html=True)

    # 2. Gráficos
    try:
        conn = sqlite3.connect('metrics.db', check_same_thread=False)
        df = pd.read_sql("SELECT timestamp, node_id, loss FROM training_logs ORDER BY timestamp DESC LIMIT 150", conn)
        conn.close()
        
        if not df.empty:
            st.divider()
            st.subheader("📉 Evolução do Erro")
            st.line_chart(df.pivot_table(index='timestamp', columns='node_id', values='loss'))
    except: pass

# Chamada inicial da função auto-atualizável
atualizar_paineis()