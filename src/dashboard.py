import streamlit as st

# ============================================================
# DEVE SER A PRIMEIRA CHAMADA STREAMLIT
# ============================================================
st.set_page_config(layout="wide", page_title="FL Imagens", page_icon="🛰️")

import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
import torch
import numpy as np

# ============================================================
# Imports do projeto
# ============================================================
try:
    from model_utils import ImageAutoencoder
    from image_utils import (load_mnist, get_random_batch,
                             mask_image_bottom, mask_image_random,
                             mask_image_right, compute_mse, compute_psnr)
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False

# ============================================================
# Estilo CSS
# ============================================================
st.markdown("""
<style>
.terminal-container {
    height: 400px; overflow-y: scroll;
    background-color: #0e1117; color: #00ff00;
    font-family: monospace; padding: 10px;
    white-space: pre-wrap; border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

STATUS_FILE = "status.json"

def set_status(s):
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": s}, f)

def get_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                return json.load(f).get("status", "PAUSED")
        except Exception:
            pass
    return "PAUSED"

# ============================================================
# SIDEBAR: Play / Pause
# ============================================================
st.sidebar.title("🛰️ FL Image Control")
status = get_status()

if status == "RUNNING":
    st.sidebar.success("🟢 RODANDO")
    if st.sidebar.button("⏸️ PAUSAR"):
        set_status("PAUSED"); st.rerun()
else:
    st.sidebar.warning("⏸️ PAUSADO")
    if st.sidebar.button("▶️ INICIAR"):
        set_status("RUNNING"); st.rerun()

# ============================================================
# SIDEBAR: Controles de Caos
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("🔥 Controle de Caos (Rede)")

chaos_active = st.sidebar.toggle("Ativar Instabilidade", value=True)

if 'loss_val' not in st.session_state: st.session_state.loss_val = 0.05
if 'delay_val' not in st.session_state: st.session_state.delay_val = 500
if 'corrupt_val' not in st.session_state: st.session_state.corrupt_val = 0.00
if 'dup_val' not in st.session_state: st.session_state.dup_val = 0.00

def update_loss_slider(): st.session_state.loss_val = st.session_state.loss_slider
def update_loss_input(): st.session_state.loss_val = st.session_state.loss_input
def update_delay_slider(): st.session_state.delay_val = st.session_state.delay_slider
def update_delay_input(): st.session_state.delay_val = st.session_state.delay_input
def update_corrupt_slider(): st.session_state.corrupt_val = st.session_state.corrupt_slider
def update_corrupt_input(): st.session_state.corrupt_val = st.session_state.corrupt_input
def update_dup_slider(): st.session_state.dup_val = st.session_state.dup_slider
def update_dup_input(): st.session_state.dup_val = st.session_state.dup_input

st.sidebar.write("Perda de Pacotes (%)")
c1, c2 = st.sidebar.columns([3, 1])
with c1: st.slider("pkt", 0.0, 5.0, key='loss_slider', value=st.session_state.loss_val, on_change=update_loss_slider, format="%.2f", label_visibility="collapsed")
with c2: st.number_input("pkt_n", 0.0, 5.0, key='loss_input', value=st.session_state.loss_val, on_change=update_loss_input, label_visibility="collapsed")

st.sidebar.write("Latência (ms)")
c3, c4 = st.sidebar.columns([3, 1])
with c3: st.slider("lat", 0, 2000, key='delay_slider', value=st.session_state.delay_val, on_change=update_delay_slider, label_visibility="collapsed")
with c4: st.number_input("lat_n", 0, 2000, key='delay_input', value=st.session_state.delay_val, on_change=update_delay_input, label_visibility="collapsed")

st.sidebar.write("Corrupção / Ruído (%)")
c5, c6 = st.sidebar.columns([3, 1])
with c5: st.slider("crp", 0.0, 2.0, step=0.01, key='corrupt_slider', value=st.session_state.corrupt_val, on_change=update_corrupt_slider, format="%.2f", label_visibility="collapsed")
with c6: st.number_input("crp_n", 0.0, 2.0, step=0.01, key='corrupt_input', value=st.session_state.corrupt_val, on_change=update_corrupt_input, label_visibility="collapsed")

st.sidebar.write("Duplicação (%)")
c7, c8 = st.sidebar.columns([3, 1])
with c7: st.slider("dup", 0.0, 5.0, step=0.1, key='dup_slider', value=st.session_state.dup_val, on_change=update_dup_slider, format="%.1f", label_visibility="collapsed")
with c8: st.number_input("dup_n", 0.0, 5.0, step=0.1, key='dup_input', value=st.session_state.dup_val, on_change=update_dup_input, label_visibility="collapsed")

if st.sidebar.button("⚡ Aplicar Caos"):
    s = "ON" if chaos_active else "OFF"
    l = st.session_state.loss_val
    d = st.session_state.delay_val
    cr = st.session_state.corrupt_val
    dp = st.session_state.dup_val
    with open("chaos_config.txt", "w") as f:
        f.write(f"{s} {l:.2f} {d} {cr:.2f} {dp:.2f}")
    if chaos_active:
        st.sidebar.error(f"ATIVO: Loss {l}% | Delay {d}ms | Noise {cr}% | Dup {dp}%")
    else:
        st.sidebar.success("Rede Normalizada")

# Reset DB
st.sidebar.divider()
if st.sidebar.button("🗑️ Limpar Histórico (Reset DB)"):
    try:
        if os.path.exists("metrics.db"):
            os.remove("metrics.db")
        conn = sqlite3.connect('metrics.db')
        conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL)")
        conn.commit(); conn.close()
        st.sidebar.success("Histórico resetado!")
        time.sleep(1); st.rerun()
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

# ============================================================
# MAIN AREA
# ============================================================
st.title("🛰️ Aprendizado Federado: Comunicação Semântica de Imagens")

tab1, tab2, tab3, tab4 = st.tabs(["📡 Comunicação", "🧩 Completação", "🧠 Terminais", "📊 Métricas"])

# ============================================================
# TAB 1: Teste de Comunicação Semântica
# ============================================================
with tab1:
    st.subheader("Teste de Comunicação Semântica")
    st.markdown("""
    **Conceito:** Um dispositivo comprime a imagem para apenas **32 números** (vetor latente)
    e envia ao outro dispositivo, que reconstrói a imagem completa.
    Compressão de **784 → 32 valores** (96% de redução).
    """)

    if not IMPORTS_OK:
        st.error("Erro ao importar módulos do projeto.")
    elif st.button("🎲 Gerar Nova Imagem", key="comm_btn"):
        if os.path.exists("global_model.pth"):
            try:
                model = ImageAutoencoder()
                model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
                model.eval()

                dataset = load_mnist(train=False)
                idx = torch.randint(0, len(dataset), (1,)).item()
                original, label = dataset[idx]
                original = original.unsqueeze(0)  # [1, 1, 28, 28]

                with torch.no_grad():
                    latent = model.encode(original)
                    reconstructed = model.decode(latent)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Original (784 valores)**")
                    st.image(original.squeeze().numpy(), width=200, clamp=True)
                    st.caption(f"Dígito: {label}")

                with col2:
                    st.markdown("**Vetor Latente (32 valores)**")
                    st.bar_chart(latent.squeeze().numpy(), height=200)
                    st.caption(f"Apenas {latent.numel()} números transmitidos")

                with col3:
                    st.markdown("**Reconstruído (784 valores)**")
                    st.image(reconstructed.squeeze().numpy(), width=200, clamp=True)

                    mse = compute_mse(original, reconstructed)
                    psnr = compute_psnr(original, reconstructed)
                    st.caption(f"MSE: {mse:.6f} | PSNR: {psnr:.1f} dB")

                st.success(f"📡 Compressão: 784 → 32 valores = **{784/32:.0f}x** de redução ({(1-32/784)*100:.1f}%)")
            except Exception as e:
                st.error(f"Erro: {e}")
        else:
            st.warning("⏳ Modelo global ainda não disponível. Inicie o treinamento primeiro.")

# ============================================================
# TAB 2: Teste de Completação de Imagem
# ============================================================
with tab2:
    st.subheader("Completação de Imagem Parcial")
    st.markdown("""
    **Conceito:** Um dispositivo envia apenas **parte** da imagem.
    O modelo no outro dispositivo **completa** a informação faltante.
    """)

    mask_type = st.selectbox("Tipo de máscara:", ["Metade Inferior", "Pixels Aleatórios", "Metade Direita"])
    mask_pct = st.slider("% da imagem mascarada:", 10, 90, 50, step=10) / 100.0

    if not IMPORTS_OK:
        st.error("Erro ao importar módulos do projeto.")
    elif st.button("🧩 Testar Completação", key="comp_btn"):
        if os.path.exists("global_model.pth"):
            try:
                model = ImageAutoencoder()
                model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
                model.eval()

                dataset = load_mnist(train=False)
                idx = torch.randint(0, len(dataset), (1,)).item()
                original, label = dataset[idx]
                original = original.unsqueeze(0)

                if mask_type == "Metade Inferior":
                    masked = mask_image_bottom(original, mask_pct)
                elif mask_type == "Pixels Aleatórios":
                    masked = mask_image_random(original, mask_pct)
                else:
                    masked = mask_image_right(original, mask_pct)

                with torch.no_grad():
                    completed = model(masked)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Original**")
                    st.image(original.squeeze().numpy(), width=200, clamp=True)
                    st.caption(f"Dígito: {label}")

                with col2:
                    st.markdown(f"**Enviado ({(1-mask_pct)*100:.0f}% da imagem)**")
                    st.image(masked.squeeze().numpy(), width=200, clamp=True)
                    st.caption(f"Máscara: {mask_type}")

                with col3:
                    st.markdown("**Completado pelo Modelo**")
                    st.image(completed.squeeze().numpy(), width=200, clamp=True)

                    mse = compute_mse(original, completed)
                    psnr = compute_psnr(original, completed)
                    st.caption(f"MSE: {mse:.6f} | PSNR: {psnr:.1f} dB")

                info_sent = (1 - mask_pct) * 100
                st.info(f"📡 Enviado apenas **{info_sent:.0f}%** da imagem. Modelo completou os **{mask_pct*100:.0f}%** faltantes.")
            except Exception as e:
                st.error(f"Erro: {e}")
        else:
            st.warning("⏳ Modelo global ainda não disponível. Inicie o treinamento primeiro.")

# ============================================================
# TAB 3: Terminais (live logs)
# ============================================================
with tab3:
    def read_log(filename):
        path = os.path.join("logs", filename)
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                return "".join(lines[-50:][::-1])
        return "Aguardando logs..."

    @st.fragment(run_every=1)
    def update_terminals():
        t1, t2, t3 = st.tabs(["🧠 SERVER", "🔵 FULL", "🟠 NOISY"])
        with t1:
            st.markdown(f'<div class="terminal-container">{read_log("server.log")}</div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="terminal-container">{read_log("client-full.log")}</div>', unsafe_allow_html=True)
        with t3:
            st.markdown(f'<div class="terminal-container">{read_log("client-noisy.log")}</div>', unsafe_allow_html=True)

    update_terminals()

# ============================================================
# TAB 4: Métricas de Treinamento
# ============================================================
with tab4:
    @st.fragment(run_every=2)
    def update_metrics():
        try:
            if not os.path.exists('metrics.db'):
                st.info("⏳ Aguardando dados de treinamento...")
                return

            conn = sqlite3.connect('metrics.db', check_same_thread=False)

            try:
                threshold = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                df = pd.read_sql(
                    f"SELECT timestamp, node_id, loss FROM training_logs WHERE timestamp > '{threshold}' ORDER BY timestamp ASC",
                    conn
                )
            except Exception:
                st.info("⏳ Aguardando dados...")
                conn.close()
                return

            conn.close()

            if df.empty:
                st.info("📊 Sem dados de treinamento ainda.")
                return

            st.subheader("📉 Curva de Loss por Cliente")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            chart_data = df.pivot_table(index='timestamp', columns='node_id', values='loss', aggfunc='mean')
            chart_data = chart_data.interpolate()
            st.line_chart(chart_data, height=400)

            st.subheader("📊 Estatísticas")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total de Rodadas", len(df))
            last_loss = df.groupby('node_id')['loss'].last()
            with c2:
                if 'client-full' in last_loss.index:
                    st.metric("Loss (Full)", f"{last_loss['client-full']:.6f}")
            with c3:
                if 'client-noisy' in last_loss.index:
                    st.metric("Loss (Noisy)", f"{last_loss['client-noisy']:.6f}")

        except Exception as e:
            st.error(f"Erro: {e}")

    update_metrics()
