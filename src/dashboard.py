import streamlit as st

# ============================================================
# DEVE SER A PRIMEIRA CHAMADA STREAMLIT
# ============================================================
st.set_page_config(layout="wide", page_title="FL Semântico", page_icon="🛰️")

import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    height: 400px; overflow-y: auto;
    background-color: #0e1117; color: #00ff00;
    font-family: 'Courier New', monospace; padding: 12px;
    white-space: pre-wrap; border-radius: 8px;
    font-size: 13px; line-height: 1.4;
    display: flex; flex-direction: column-reverse;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 16px; border-radius: 10px; text-align: center;
    border: 1px solid #0f3460;
}
.metric-card h3 { color: #e94560; margin: 0; font-size: 28px; }
.metric-card p { color: #a0a0a0; margin: 4px 0 0 0; font-size: 13px; }
.topology-box {
    background: #0e1117; border-radius: 10px; padding: 20px;
    border: 1px solid #333; text-align: center;
}
</style>
""", unsafe_allow_html=True)

STATUS_FILE = "status.json"
DB_FILE = "metrics.db"

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

def get_current_round():
    try:
        if not os.path.exists(DB_FILE):
            return 0
        conn = sqlite3.connect(DB_FILE, timeout=5)
        cur = conn.execute("SELECT MAX(round_number) FROM round_metrics")
        row = cur.fetchone()
        conn.close()
        return row[0] if row[0] else 0
    except Exception:
        return 0

# ============================================================
# SIDEBAR: Play / Pause
# ============================================================
st.sidebar.title("🛰️ FL Control Panel")
status = get_status()
current_round = get_current_round()

st.sidebar.markdown(f"**Rodada Atual:** `{current_round}`")

if status == "RUNNING":
    st.sidebar.success("🟢 RODANDO")
    if st.sidebar.button("⏸️ PAUSAR"):
        set_status("PAUSED"); st.rerun()
else:
    st.sidebar.warning("⏸️ PAUSADO")
    if st.sidebar.button("▶️ INICIAR"):
        set_status("RUNNING"); st.rerun()

# ============================================================
# SIDEBAR: Cenários Rápidos
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("⚡ Cenários de Caos")

PRESETS = {
    "Normal (Sem Caos)":  {"loss": 0.0, "delay": 0, "corrupt": 0.0, "dup": 0.0, "active": False},
    "Leve":               {"loss": 1.0, "delay": 200, "corrupt": 0.0, "dup": 0.0, "active": True},
    "Moderado":           {"loss": 3.0, "delay": 500, "corrupt": 0.5, "dup": 1.0, "active": True},
    "Severo":             {"loss": 5.0, "delay": 1000, "corrupt": 2.0, "dup": 5.0, "active": True},
}

preset_choice = st.sidebar.selectbox("Cenário Rápido:", list(PRESETS.keys()))
if st.sidebar.button("🎯 Aplicar Cenário"):
    p = PRESETS[preset_choice]
    st.session_state.loss_val = p["loss"]
    st.session_state.delay_val = p["delay"]
    st.session_state.corrupt_val = p["corrupt"]
    st.session_state.dup_val = p["dup"]
    s = "ON" if p["active"] else "OFF"
    with open("chaos_config.txt", "w") as f:
        f.write(f"{s} {p['loss']:.2f} {p['delay']} {p['corrupt']:.2f} {p['dup']:.2f}")
    st.sidebar.success(f"Cenário '{preset_choice}' aplicado!")

# ============================================================
# SIDEBAR: Controles de Caos Manual
# ============================================================
with st.sidebar.expander("🔧 Controle Manual de Caos"):
    chaos_active = st.toggle("Ativar Instabilidade", value=True)
    
    if 'loss_val' not in st.session_state: st.session_state.loss_val = 0.0
    if 'delay_val' not in st.session_state: st.session_state.delay_val = 0
    if 'corrupt_val' not in st.session_state: st.session_state.corrupt_val = 0.0
    if 'dup_val' not in st.session_state: st.session_state.dup_val = 0.0

    loss_v = st.slider("Perda Pacotes (%)", 0.0, 5.0, st.session_state.loss_val, format="%.2f", key="m_loss")
    delay_v = st.slider("Latência (ms)", 0, 2000, st.session_state.delay_val, key="m_delay")
    corrupt_v = st.slider("Corrupção (%)", 0.0, 2.0, st.session_state.corrupt_val, step=0.01, format="%.2f", key="m_corrupt")
    dup_v = st.slider("Duplicação (%)", 0.0, 5.0, st.session_state.dup_val, step=0.1, format="%.1f", key="m_dup")

    if st.button("⚡ Aplicar"):
        s = "ON" if chaos_active else "OFF"
        with open("chaos_config.txt", "w") as f:
            f.write(f"{s} {loss_v:.2f} {delay_v} {corrupt_v:.2f} {dup_v:.2f}")
        st.session_state.loss_val = loss_v
        st.session_state.delay_val = delay_v
        st.session_state.corrupt_val = corrupt_v
        st.session_state.dup_val = dup_v
        st.success("Caos aplicado!")

# Reset DB
st.sidebar.divider()
if st.sidebar.button("🗑️ Limpar Histórico (Reset DB)"):
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        conn = sqlite3.connect(DB_FILE)
        conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_mse REAL, global_psnr REAL, timestamp TEXT, chaos_scenario TEXT)")
        conn.commit(); conn.close()
        st.sidebar.success("Histórico resetado!")
        time.sleep(1); st.rerun()
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

# ============================================================
# MAIN AREA
# ============================================================
st.title("🛰️ Aprendizado Federado: Comunicação Semântica de Imagens")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Arquitetura", "📡 Comunicação", "🧩 Completação", 
    "🧠 Terminais", "📊 Métricas", "📈 Experimentos"
])

# ============================================================
# TAB 0: Arquitetura / Topologia
# ============================================================
with tab1:
    st.subheader("Topologia do Sistema Federado")
    
    st.markdown("""
    O sistema é composto por **3 clientes** com perfis distintos que treinam localmente 
    e enviam seus pesos ao **servidor central** para agregação via **FedAvg**.
    """)
    
    # Topology diagram using graphviz
    st.graphviz_chart('''
    digraph FL {
        rankdir=TB;
        node [shape=box, style="rounded,filled", fontname="Helvetica"];
        edge [fontname="Helvetica", fontsize=10];
        
        subgraph cluster_clients {
            label="Clientes Locais";
            style="dashed"; color="#666";
            
            full [label="🔵 client-full\\nDados: IID (todos)\\nPesos: Completos\\nRede: Estável", 
                  fillcolor="#E3F2FD", color="#2196F3", penwidth=2];
            noisy [label="🟠 client-noisy\\nDados: IID (todos)\\nPesos: Top-K (40%)\\nRede: Caos (tc netem)", 
                   fillcolor="#FFF3E0", color="#FF9800", penwidth=2];
            noniid [label="🟢 client-noniid\\nDados: Non-IID (0-3)\\nPesos: Completos\\nRede: Estável", 
                    fillcolor="#E8F5E9", color="#4CAF50", penwidth=2];
        }
        
        server [label="🧠 Servidor FL\\nAgregação FedAvg\\nAvaliação Global", 
                fillcolor="#F3E5F5", color="#9C27B0", penwidth=2];
        
        global_model [label="📦 Modelo Global\\nglobal_model.pth\\n(Autoencoder 784→32→784)", 
                      fillcolor="#FFFDE7", color="#FFC107", penwidth=2];
        
        chaos [label="💥 Chaos Injector\\ntc netem\\n(loss/delay/corrupt)", 
               fillcolor="#FFEBEE", color="#F44336", penwidth=2, shape=diamond];
        
        full -> server [label="  Pesos completos  ", color="#2196F3"];
        noisy -> server [label="  Pesos Top-K  ", color="#FF9800", style="dashed"];
        noniid -> server [label="  Pesos completos  ", color="#4CAF50"];
        
        server -> global_model [label="  FedAvg  ", color="#9C27B0", penwidth=2];
        global_model -> full [label="  Download  ", style="dotted"];
        global_model -> noisy [label="  Download  ", style="dotted"];
        global_model -> noniid [label="  Download  ", style="dotted"];
        
        chaos -> noisy [label="  Interfere  ", color="#F44336", style="bold"];
    }
    ''')
    
    # Client profiles
    st.subheader("Perfis dos Clientes")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        #### 🔵 client-full (Estável)
        - **Dados:** IID — todos os 10 dígitos
        - **Pesos:** Enviados completos (100%)
        - **Rede:** Sem perturbação
        - **Papel:** Baseline confiável
        """)
    
    with c2:
        st.markdown("""
        #### 🟠 client-noisy (Instável)
        - **Dados:** IID — todos os 10 dígitos
        - **Pesos:** Compressão Top-K (60% zerados)
        - **Rede:** Caos: loss, latência, corrupção
        - **Papel:** Testa resiliência do FL
        """)
    
    with c3:
        st.markdown("""
        #### 🟢 client-noniid (Heterogêneo)
        - **Dados:** Non-IID — apenas dígitos 0-3
        - **Pesos:** Enviados completos (100%)
        - **Rede:** Sem perturbação
        - **Papel:** Testa robustez a dados enviesados
        """)
    
    # Compression info
    st.subheader("Compressão Semântica")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Pixels por Imagem", "784")
    with m2:
        st.metric("Dimensão Latente", "32")
    with m3:
        st.metric("Fator de Compressão", "24.5×")
    with m4:
        st.metric("Redução", "95.9%")


# ============================================================
# TAB 1: Teste de Comunicação Semântica
# ============================================================
with tab2:
    st.subheader("Teste de Comunicação Semântica")
    st.markdown("""
    **Conceito:** Um dispositivo comprime a imagem para apenas **32 números** (vetor latente)
    e envia ao outro dispositivo, que reconstrói a imagem completa.
    """)

    if not IMPORTS_OK:
        st.error("Erro ao importar módulos do projeto.")
    else:
        comm_mode = st.radio("Modo:", ["Imagem única", "Todos os 10 dígitos"], horizontal=True)
        
        if comm_mode == "Imagem única" and st.button("🎲 Gerar Nova Imagem", key="comm_btn"):
            if os.path.exists("global_model.pth"):
                try:
                    model = ImageAutoencoder()
                    model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
                    model.eval()

                    dataset = load_mnist(train=False)
                    idx = torch.randint(0, len(dataset), (1,)).item()
                    original, label = dataset[idx]
                    original = original.unsqueeze(0)

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
        
        elif comm_mode == "Todos os 10 dígitos" and st.button("🔟 Testar Todos os Dígitos", key="all_digits_btn"):
            if os.path.exists("global_model.pth"):
                try:
                    model = ImageAutoencoder()
                    model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
                    model.eval()
                    
                    dataset = load_mnist(train=False)
                    digit_examples = {}
                    for i in range(len(dataset)):
                        img, lbl = dataset[i]
                        if lbl not in digit_examples:
                            digit_examples[lbl] = img
                        if len(digit_examples) == 10:
                            break
                    
                    # Create 2x10 grid: originals on top, reconstructions below
                    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
                    total_mse = 0
                    total_psnr = 0
                    
                    for d in range(10):
                        img = digit_examples[d].unsqueeze(0)
                        with torch.no_grad():
                            recon = model(img)
                        
                        mse = compute_mse(img, recon)
                        psnr = compute_psnr(img, recon)
                        total_mse += mse
                        total_psnr += psnr
                        
                        axes[0, d].imshow(img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                        axes[0, d].set_title(f"{d}", fontsize=12, fontweight='bold')
                        axes[0, d].axis('off')
                        
                        axes[1, d].imshow(recon.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                        axes[1, d].set_title(f"{psnr:.0f}dB", fontsize=9, color='green')
                        axes[1, d].axis('off')
                    
                    axes[0, 0].set_ylabel("Original", fontsize=11, rotation=0, labelpad=50, va='center')
                    axes[1, 0].set_ylabel("Recon.", fontsize=11, rotation=0, labelpad=50, va='center')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.success(f"📊 Média: MSE={total_mse/10:.4f} | PSNR={total_psnr/10:.1f} dB | Compressão: 24.5×")
                except Exception as e:
                    st.error(f"Erro: {e}")
            else:
                st.warning("⏳ Modelo global ainda não disponível.")

# ============================================================
# TAB 2: Teste de Completação de Imagem
# ============================================================
with tab3:
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
with tab4:
    def read_log(filename):
        path = os.path.join("logs", filename)
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                return "".join(lines[-50:])  # Newest at bottom
        return "Aguardando logs..."

    @st.fragment(run_every=1)
    def update_terminals():
        t1, t2, t3, t4 = st.tabs(["🧠 SERVER", "🔵 FULL", "🟠 NOISY", "🟢 NON-IID"])
        with t1:
            st.markdown(f'<div class="terminal-container">{read_log("server.log")}</div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="terminal-container">{read_log("client-full.log")}</div>', unsafe_allow_html=True)
        with t3:
            st.markdown(f'<div class="terminal-container">{read_log("client-noisy.log")}</div>', unsafe_allow_html=True)
        with t4:
            st.markdown(f'<div class="terminal-container">{read_log("client-noniid.log")}</div>', unsafe_allow_html=True)

    update_terminals()

# ============================================================
# TAB 4: Métricas de Treinamento
# ============================================================
with tab5:
    @st.fragment(run_every=2)
    def update_metrics():
        try:
            if not os.path.exists(DB_FILE):
                st.info("⏳ Aguardando dados de treinamento...")
                return

            conn = sqlite3.connect(DB_FILE, check_same_thread=False)

            # Client loss by round
            try:
                df = pd.read_sql(
                    "SELECT round_number, node_id, loss FROM training_logs WHERE round_number > 0 ORDER BY round_number ASC",
                    conn
                )
            except Exception:
                st.info("⏳ Aguardando dados...")
                conn.close()
                return
            
            # Round metrics (global MSE/PSNR)
            try:
                df_rounds = pd.read_sql(
                    "SELECT round_number, global_mse, global_psnr FROM round_metrics ORDER BY round_number ASC",
                    conn
                )
            except Exception:
                df_rounds = pd.DataFrame()

            conn.close()

            if df.empty:
                st.info("📊 Sem dados de treinamento ainda.")
                return

            # KPI Cards
            st.subheader("📊 Resumo")
            k1, k2, k3, k4 = st.columns(4)
            
            max_round = df['round_number'].max()
            with k1:
                st.metric("Rodada Atual", max_round)
            
            last_loss = df.groupby('node_id')['loss'].last()
            with k2:
                if 'client-full' in last_loss.index:
                    st.metric("🔵 Loss Full", f"{last_loss['client-full']:.6f}")
            with k3:
                if 'client-noisy' in last_loss.index:
                    st.metric("🟠 Loss Noisy", f"{last_loss['client-noisy']:.6f}")
            with k4:
                if 'client-noniid' in last_loss.index:
                    st.metric("🟢 Loss Non-IID", f"{last_loss['client-noniid']:.6f}")

            # Loss curves by round (per client)
            st.subheader("📉 Loss por Cliente (por Rodada)")
            chart_data = df.pivot_table(index='round_number', columns='node_id', values='loss', aggfunc='mean')
            # Rename columns for display
            rename_map = {"client-full": "🔵 Full", "client-noisy": "🟠 Noisy", "client-noniid": "🟢 Non-IID"}
            chart_data = chart_data.rename(columns=rename_map)
            st.line_chart(chart_data, height=350)
            
            # Global model quality
            if not df_rounds.empty:
                st.subheader("🌐 Qualidade do Modelo Global")
                g1, g2 = st.columns(2)
                with g1:
                    st.metric("MSE Global (última)", f"{df_rounds['global_mse'].iloc[-1]:.6f}")
                    st.line_chart(df_rounds.set_index('round_number')['global_mse'], height=250)
                with g2:
                    st.metric("PSNR Global (última)", f"{df_rounds['global_psnr'].iloc[-1]:.1f} dB")
                    st.line_chart(df_rounds.set_index('round_number')['global_psnr'], height=250)

        except Exception as e:
            st.error(f"Erro: {e}")

    update_metrics()

# ============================================================
# TAB 5: Histórico de Experimentos
# ============================================================
with tab6:
    st.subheader("📈 Comparação de Experimentos por Cenário de Caos")
    st.markdown("Carrega resultados exportados pelo `run_experiments.py` da pasta `results/`.")
    
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        st.info("📂 Nenhum resultado de experimento encontrado. Execute `python run_experiments.py` primeiro.")
    else:
        csv_files = [f for f in os.listdir(results_dir) if f.endswith("_round_metrics.csv")]
        
        if not csv_files:
            st.info("📂 Nenhum CSV de resultados encontrado. Execute `python run_experiments.py` primeiro.")
        else:
            # Load all scenarios
            all_data = {}
            for f in csv_files:
                scenario = f.replace("_round_metrics.csv", "")
                try:
                    df = pd.read_csv(os.path.join(results_dir, f))
                    all_data[scenario] = df
                except Exception:
                    pass
            
            if all_data:
                scenarios_found = list(all_data.keys())
                selected = st.multiselect("Cenários:", scenarios_found, default=scenarios_found)
                
                if selected:
                    # MSE comparison
                    st.subheader("MSE Global por Cenário")
                    mse_chart = pd.DataFrame()
                    for s in selected:
                        if s in all_data:
                            df = all_data[s]
                            mse_chart[s] = df.set_index('round_number')['global_mse']
                    if not mse_chart.empty:
                        st.line_chart(mse_chart, height=350)
                    
                    # PSNR comparison  
                    st.subheader("PSNR Global por Cenário")
                    psnr_chart = pd.DataFrame()
                    for s in selected:
                        if s in all_data:
                            df = all_data[s]
                            psnr_chart[s] = df.set_index('round_number')['global_psnr']
                    if not psnr_chart.empty:
                        st.line_chart(psnr_chart, height=350)
                    
                    # Summary table
                    st.subheader("Tabela Comparativa")
                    summary_rows = []
                    for s in selected:
                        if s in all_data:
                            df = all_data[s]
                            summary_rows.append({
                                "Cenário": s,
                                "MSE Final": f"{df['global_mse'].iloc[-1]:.4f}" if len(df) > 0 else "N/A",
                                "PSNR Final (dB)": f"{df['global_psnr'].iloc[-1]:.1f}" if len(df) > 0 else "N/A",
                                "Melhor MSE": f"{df['global_mse'].min():.4f}" if len(df) > 0 else "N/A",
                                "Melhor PSNR (dB)": f"{df['global_psnr'].max():.1f}" if len(df) > 0 else "N/A",
                                "Rodadas": len(df),
                            })
                    if summary_rows:
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            
            # Per-client loss comparison
            log_files = [f for f in os.listdir(results_dir) if f.endswith("_training_logs.csv")]
            if log_files:
                st.divider()
                st.subheader("Loss por Cliente (por Cenário)")
                selected_scenario = st.selectbox("Cenário:", [f.replace("_training_logs.csv", "") for f in log_files])
                path = os.path.join(results_dir, f"{selected_scenario}_training_logs.csv")
                if os.path.exists(path):
                    try:
                        df_logs = pd.read_csv(path)
                        chart = df_logs.pivot_table(index='round_number', columns='node_id', values='loss', aggfunc='mean')
                        rename_map = {"client-full": "🔵 Full", "client-noisy": "🟠 Noisy", "client-noniid": "🟢 Non-IID"}
                        chart = chart.rename(columns=rename_map)
                        st.line_chart(chart, height=350)
                    except Exception as e:
                        st.error(f"Erro ao carregar: {e}")
