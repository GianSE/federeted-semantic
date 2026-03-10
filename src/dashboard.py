import json
import os
import sqlite3
import time

import pandas as pd
import requests
import streamlit as st

from config import NONIID_LABELS, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE, TEXT_RECONSTRUCTION_MODE
MAX_MODEL_CHUNK = int(TEXT_CHUNK_SIZE)


try:
    from text_utils import decode_tokens, get_random_sample_text, load_text_dataset
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False


st.set_page_config(layout="wide", page_title="FL Semântico Textual", page_icon="🛰️")

STATUS_FILE = "status.json"
DB_FILE = "metrics.db"
RESULTS_DIR = "results"
SERVER_URL = os.environ.get("SERVER_URL", "http://fl-server:5000")

st.markdown(
    """
<style>
.terminal-container {
    height: 400px;
    overflow-y: auto;
    background: #121417;
    color: #dce7ea;
    font-family: Consolas, monospace;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #283038;
    white-space: pre-wrap;
}
.panel-box {
    border: 1px solid #283038;
    border-radius: 12px;
    padding: 16px;
    background: linear-gradient(180deg, #faf7ef 0%, #f2eee2 100%);
    min-height: 240px;
}
</style>
""",
    unsafe_allow_html=True,
)


def set_status(status):
    with open(STATUS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"status": status}, handle)


def get_status():
    if not os.path.exists(STATUS_FILE):
        return "PAUSED"
    try:
        with open(STATUS_FILE, encoding="utf-8") as handle:
            return json.load(handle).get("status", "PAUSED")
    except Exception:
        return "PAUSED"


def load_round_metrics_db():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_FILE, timeout=5)
    try:
        df = pd.read_sql(
            "SELECT round_number, global_celoss, global_accuracy, timestamp, chaos_scenario FROM round_metrics ORDER BY round_number ASC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_training_logs_db():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE, timeout=5)
    try:
        df = pd.read_sql(
            "SELECT round_number, node_id, loss FROM training_logs WHERE round_number > 0 ORDER BY round_number ASC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_current_round():
    df = load_round_metrics_db()
    if df.empty:
        return 0
    return int(df["round_number"].max())


@st.cache_resource(show_spinner=False)
def get_test_dataset():
    if not IMPORTS_OK:
        return None, None
    return load_text_dataset(train=False)


def sample_text():
    dataset, vocab = get_test_dataset()
    if dataset is None or vocab is None:
        return "", None
    sample_tensor, _, label = get_random_sample_text(dataset)
    return decode_tokens(sample_tensor.tolist(), vocab), label


def call_api(endpoint, payload):
    response = requests.post(f"{SERVER_URL}{endpoint}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def read_log(filename):
    path = os.path.join("logs", filename)
    if not os.path.exists(path):
        return "Aguardando logs..."
    with open(path, encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return "".join(lines[-60:])


def reset_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_celoss REAL, global_accuracy REAL, timestamp TEXT, chaos_scenario TEXT)")
    conn.commit()
    conn.close()


def load_round_metrics_csv(path):
    return pd.read_csv(path)


st.sidebar.title("🛰️ Painel FL Textual")
status = get_status()
st.sidebar.markdown(f"**Rodada Atual:** {get_current_round()}")

if status == "RUNNING":
    st.sidebar.success("Treinamento em execução")
    if st.sidebar.button("Pausar"):
        set_status("PAUSED")
        st.rerun()
else:
    st.sidebar.warning("Treinamento pausado")
    if st.sidebar.button("Iniciar"):
        set_status("RUNNING")
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Cenários de Caos")
presets = {
    "Normal": {"loss": 0.0, "delay": 0, "corrupt": 0.0, "dup": 0.0, "active": False},
    "Leve": {"loss": 1.0, "delay": 200, "corrupt": 0.0, "dup": 0.0, "active": True},
    "Moderado": {"loss": 3.0, "delay": 500, "corrupt": 0.5, "dup": 1.0, "active": True},
    "Severo": {"loss": 5.0, "delay": 1000, "corrupt": 2.0, "dup": 5.0, "active": True},
}
selected_preset = st.sidebar.selectbox("Preset", list(presets.keys()))
if st.sidebar.button("Aplicar preset"):
    preset = presets[selected_preset]
    status_label = "ON" if preset["active"] else "OFF"
    with open("chaos_config.txt", "w", encoding="utf-8") as handle:
        handle.write(f"{status_label} {preset['loss']:.2f} {preset['delay']} {preset['corrupt']:.2f} {preset['dup']:.2f}")
    st.sidebar.success(f"Preset {selected_preset} aplicado")

st.sidebar.divider()
if st.sidebar.button("Limpar histórico"):
    reset_db()
    st.sidebar.success("Banco reiniciado")
    time.sleep(1)
    st.rerun()

st.title("🛰️ Aprendizado Federado para Comunicação Semântica de Texto")
st.caption("Fluxo 100% textual com treinamento federado, compressão latente por chunks e reconstrução no destino.")

tab_arch, tab_transfer, tab_completion, tab_logs, tab_metrics, tab_exp = st.tabs(
    ["Arquitetura", "Transferência", "Completação", "Terminais", "Métricas", "Experimentos"]
)

with tab_arch:
    st.subheader("Topologia textual")
    st.markdown(
        f"""
O sistema possui três clientes que treinam localmente um autoencoder textual e publicam pesos para o servidor central.
O servidor agrega por FedAvg, expõe endpoints de compressão e geração textual, e o dashboard opera como console de teste.

- `client-full`: cliente estável com dados IID.
- `client-noisy`: cliente com compressão Top-K e caos de rede.
- `client-noniid`: cliente com subconjunto textual Non-IID das classes {NONIID_LABELS}.
- `fl-server`: agrega pesos e disponibiliza `/compress_text`, `/generate_text` e `/complete_text`.
        """
    )
    st.graphviz_chart(
        """
digraph FLText {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Helvetica"];
    client1 [label="client-full\nIID\nrede estável", fillcolor="#f1f5d6"];
    client2 [label="client-noisy\nTop-K\ncaos de rede", fillcolor="#fde2b8"];
    client3 [label="client-noniid\nsubconjunto textual", fillcolor="#dbe8f3"];
    server [label="fl-server\nFedAvg + API textual", fillcolor="#f4dfd0"];
    dash [label="dashboard\nentrada manual\ninspeção de payload", fillcolor="#e6eadf"];
    client1 -> server;
    client2 -> server;
    client3 -> server;
    dash -> server [dir=both];
}
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunk padrão", f"{TEXT_CHUNK_SIZE} tokens")
    col2.metric("Overlap padrão", f"{TEXT_CHUNK_OVERLAP} tokens")
    col3.metric("Modo padrão", TEXT_RECONSTRUCTION_MODE)
    col4.metric("Destino", "texto gerado")

with tab_transfer:
    st.subheader("Compressão e geração de texto")
    if "transfer_text" not in st.session_state:
        sample, _ = sample_text()
        st.session_state.transfer_text = sample or "Cole aqui um texto grande para gerar payload semântico por chunks."

    col_input, col_options = st.columns([2, 1])
    with col_input:
        transfer_text = st.text_area("Texto de origem", key="transfer_text", height=240)
    with col_options:
        mode = st.selectbox("Modo", ["semantic", "faithful"], index=0 if TEXT_RECONSTRUCTION_MODE == "semantic" else 1)
        chunk_size = st.slider("Tokens por chunk", 20, MAX_MODEL_CHUNK, int(TEXT_CHUNK_SIZE), step=5)
        overlap = st.slider("Overlap", 0, min(30, chunk_size - 1), int(min(TEXT_CHUNK_OVERLAP, chunk_size - 1)))
        if st.button("Carregar exemplo"):
            sample, label = sample_text()
            st.session_state.transfer_text = sample
            st.caption(f"Exemplo carregado. Label do corpus: {label}")
            st.rerun()

    if st.button("Comprimir e gerar no destino", type="primary"):
        try:
            payload = call_api("/compress_text", {
                "text": transfer_text,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "mode": mode,
            })
            generated = call_api("/generate_text", {"chunks": payload["chunks"]})
            st.session_state.transfer_result = {"payload": payload, "generated": generated}
        except Exception as exc:
            st.error(f"Falha ao chamar API textual: {exc}")

    result = st.session_state.get("transfer_result")
    if result:
        payload = result["payload"]
        generated = result["generated"]
        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Chunks", payload["num_chunks"])
        meta2.metric("Latentes enviados", sum(len(chunk["latent"]) for chunk in payload["chunks"]))
        meta3.metric("Modo", payload["mode"])
        meta4.metric("Overlap", payload["overlap"])

        left, right = st.columns(2)
        with left:
            st.markdown("**Texto de origem**")
            st.markdown(f"<div class='panel-box'>{transfer_text}</div>", unsafe_allow_html=True)
        with right:
            st.markdown("**Texto gerado no destino**")
            st.markdown(f"<div class='panel-box'>{generated['text']}</div>", unsafe_allow_html=True)

        chunk_rows = []
        for chunk in payload["chunks"]:
            chunk_rows.append({
                "chunk_id": chunk["chunk_id"],
                "source_text": chunk["source_text"],
                "reconstructed_text": chunk["reconstructed_text"],
                "latent_dim": len(chunk["latent"]),
            })
        st.subheader("Payload por chunk")
        st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)

with tab_completion:
    st.subheader("Completação textual")
    if "completion_text" not in st.session_state:
        sample, _ = sample_text()
        st.session_state.completion_text = sample or "Digite um texto para mascarar e completar no destino."

    completion_text = st.text_area("Texto para completação", key="completion_text", height=220)
    col_a, col_b, col_c = st.columns(3)
    strategy = col_a.selectbox("Estratégia", ["truncate", "random"], format_func=lambda value: "Truncar fim" if value == "truncate" else "Mascarar aleatório")
    mask_ratio = col_b.slider("Percentual mascarado", 0.1, 0.9, 0.5, step=0.1)
    completion_chunk = col_c.slider("Chunk size", 20, MAX_MODEL_CHUNK, int(TEXT_CHUNK_SIZE), step=5)

    if st.button("Completar texto"):
        try:
            result = call_api("/complete_text", {
                "text": completion_text,
                "strategy": strategy,
                "mask_ratio": mask_ratio,
                "chunk_size": completion_chunk,
                "overlap": 0,
            })
            st.session_state.completion_result = result
        except Exception as exc:
            st.error(f"Falha ao completar texto: {exc}")

    completion_result = st.session_state.get("completion_result")
    if completion_result:
        orig_col, masked_col, done_col = st.columns(3)
        orig_col.markdown("**Original**")
        orig_col.markdown(f"<div class='panel-box'>{completion_text}</div>", unsafe_allow_html=True)
        masked_col.markdown("**Texto enviado**")
        masked_col.markdown(f"<div class='panel-box'>{completion_result['masked_text']}</div>", unsafe_allow_html=True)
        done_col.markdown("**Texto completado**")
        done_col.markdown(f"<div class='panel-box'>{completion_result['completed_text']}</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(completion_result["chunks"]), use_container_width=True, hide_index=True)

with tab_logs:
    @st.fragment(run_every=1)
    def render_logs():
        t1, t2, t3, t4 = st.tabs(["Servidor", "client-full", "client-noisy", "client-noniid"])
        with t1:
            st.markdown(f"<div class='terminal-container'>{read_log('server.log')}</div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f"<div class='terminal-container'>{read_log('client-full.log')}</div>", unsafe_allow_html=True)
        with t3:
            st.markdown(f"<div class='terminal-container'>{read_log('client-noisy.log')}</div>", unsafe_allow_html=True)
        with t4:
            st.markdown(f"<div class='terminal-container'>{read_log('client-noniid.log')}</div>", unsafe_allow_html=True)

    render_logs()

with tab_metrics:
    df_logs = load_training_logs_db()
    df_rounds = load_round_metrics_db()
    if df_logs.empty:
        st.info("Aguardando métricas de treinamento.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rodada atual", int(df_logs["round_number"].max()))
        last_loss = df_logs.groupby("node_id")["loss"].last()
        k2.metric("Loss client-full", f"{last_loss.get('client-full', 0.0):.4f}")
        k3.metric("Loss client-noisy", f"{last_loss.get('client-noisy', 0.0):.4f}")
        k4.metric("Loss client-noniid", f"{last_loss.get('client-noniid', 0.0):.4f}")

        st.subheader("Loss por cliente")
        chart_data = df_logs.pivot_table(index="round_number", columns="node_id", values="loss", aggfunc="mean")
        st.line_chart(chart_data, height=340)

        if not df_rounds.empty:
            a, b = st.columns(2)
            a.metric("Global CE Loss", f"{df_rounds['global_celoss'].iloc[-1]:.4f}")
            a.line_chart(df_rounds.set_index("round_number")["global_celoss"], height=260)
            b.metric("Global Accuracy", f"{df_rounds['global_accuracy'].iloc[-1]:.1f}%")
            b.line_chart(df_rounds.set_index("round_number")["global_accuracy"], height=260)

with tab_exp:
    st.subheader("Resultados exportados")
    if not os.path.exists(RESULTS_DIR):
        st.info("Nenhum resultado exportado encontrado.")
    else:
        metric_files = sorted(name for name in os.listdir(RESULTS_DIR) if name.endswith("_round_metrics.csv"))
        if not metric_files:
            st.info("Nenhum CSV de métricas encontrado em results/.")
        else:
            all_results = {}
            for name in metric_files:
                path = os.path.join(RESULTS_DIR, name)
                try:
                    all_results[name.replace("_round_metrics.csv", "")] = load_round_metrics_csv(path)
                except Exception:
                    continue

            selected = st.multiselect("Cenários", list(all_results.keys()), default=list(all_results.keys()))
            if selected:
                loss_chart = pd.DataFrame()
                acc_chart = pd.DataFrame()
                summary = []
                for scenario in selected:
                    df = all_results[scenario]
                    loss_chart[scenario] = df.set_index("round_number")["global_celoss"]
                    acc_chart[scenario] = df.set_index("round_number")["global_accuracy"]
                    summary.append({
                        "Cenário": scenario,
                        "CE final": f"{df['global_celoss'].iloc[-1]:.4f}",
                        "Accuracy final": f"{df['global_accuracy'].iloc[-1]:.1f}%",
                        "Melhor CE": f"{df['global_celoss'].min():.4f}",
                        "Melhor accuracy": f"{df['global_accuracy'].max():.1f}%",
                        "Rodadas": len(df),
                    })
                st.line_chart(loss_chart, height=320)
                st.line_chart(acc_chart, height=320)
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
