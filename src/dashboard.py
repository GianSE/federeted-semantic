import streamlit as st
import sqlite3
import pandas as pd
import time
import os
import json
import torch
import numpy as np

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
        with open(p) as file: return "".join(file.readlines()[-30:])
    return "..."

@st.fragment(run_every=1)
def update():
    t1, t2, t3 = st.tabs(["🧠 SERVER", "🔵 FULL", "🟠 NOISY"])
    with t1: st.markdown(f'<div class="terminal-container">{read_log("server.log")}</div>', unsafe_allow_html=True)
    with t2: st.markdown(f'<div class="terminal-container">{read_log("client-full.log")}</div>', unsafe_allow_html=True)
    with t3: st.markdown(f'<div class="terminal-container">{read_log("client-noisy.log")}</div>', unsafe_allow_html=True)

    try:
        c = sqlite3.connect('metrics.db', check_same_thread=False)
        df = pd.read_sql("SELECT timestamp, node_id, loss FROM training_logs ORDER BY timestamp DESC LIMIT 100", c)
        c.close()
        if not df.empty: 
            st.line_chart(df.pivot_table(index='timestamp', columns='node_id', values='loss'))
    except: pass

update()