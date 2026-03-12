"""
Gerador de figuras e tabelas para o paper em modo 100% textual.

Saídas:
  - fig_convergence.png
  - fig_reconstruction.png
  - fig_completion.png
  - fig_chaos.png
  - tab_reconstruction.tex
  - tab_completion.tex
  - tab_chaos_results.tex
"""

import os
import sqlite3
import sys
import textwrap

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import LATENT_DIM, MODEL_TYPE, TEXT_CHUNK_SIZE
from model_utils import get_model
from text_utils import (
    build_masked_document,
    compute_accuracy,
    compute_cross_entropy,
    decode_tokens,
    get_random_sample_text,
    load_text_dataset,
    stitch_text_chunks,
    text_to_chunk_tensors,
    MAX_SEQ_LEN,
    MAX_VOCAB_SIZE,
)

RESULTS_DIR = "results"
FIGURES_DIR = os.environ.get("FIGURES_DIR", os.path.join("..", "paper", "figures"))
TABLES_DIR = os.environ.get("TABLES_DIR", os.path.join("..", "paper", "tables"))
DB_FILE = "metrics.db"
SCENARIOS = ["Normal", "Leve", "Moderado", "Severo"]
SCENARIO_COLORS = {"Normal": "#2f5d62", "Leve": "#5e8b7e", "Moderado": "#d8a15d", "Severo": "#b85c38"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

LATEX_ROW_BREAK = r"\\"


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def load_round_metrics_csv(scenario):
    path = os.path.join(RESULTS_DIR, f"{scenario}_round_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_training_logs_csv(scenario):
    path = os.path.join(RESULTS_DIR, f"{scenario}_training_logs.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_live_round_metrics():
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE, timeout=10)
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


def load_live_training_logs():
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql("SELECT node_id, round_number, loss FROM training_logs ORDER BY round_number ASC", conn)
    finally:
        conn.close()
    return df


def load_model_and_vocab():
    if not os.path.exists("global_model.pth"):
        return None, None, None
    model = get_model(MODEL_TYPE, vocab_size=MAX_VOCAB_SIZE, seq_len=MAX_SEQ_LEN, latent_dim=LATENT_DIM)
    try:
        model.load_state_dict(torch.load("global_model.pth", map_location="cpu", weights_only=True))
    except Exception:
        return None, None, None
    model.eval()
    dataset, vocab = load_text_dataset(train=False)
    return model, dataset, vocab


def run_reconstruction(model, vocab, text):
    chunk_tensors, _ = text_to_chunk_tensors(text, vocab, chunk_size=TEXT_CHUNK_SIZE, overlap=0)
    reconstructed_chunks = []
    chunk_losses = []
    chunk_accuracies = []
    with torch.no_grad():
        for tensor in chunk_tensors:
            output = model(tensor.unsqueeze(0))
            logits = output[0] if isinstance(output, tuple) else output
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            reconstructed_chunks.append(decode_tokens(predictions.tolist(), vocab))
            chunk_losses.append(compute_cross_entropy(tensor.unsqueeze(0), logits))
            chunk_accuracies.append(compute_accuracy(tensor.unsqueeze(0), logits) * 100.0)
    return stitch_text_chunks(reconstructed_chunks), sum(chunk_losses) / len(chunk_losses), sum(chunk_accuracies) / len(chunk_accuracies), len(chunk_tensors)


def build_document_samples(dataset, vocab, sample_count=3, parts_per_document=3):
    documents = []
    for _ in range(sample_count):
        segments = []
        labels = []
        for _ in range(parts_per_document):
            tensor, _, label = get_random_sample_text(dataset)
            segments.append(decode_tokens(tensor.tolist(), vocab))
            labels.append(label)
        documents.append({
            "labels": labels,
            "text": stitch_text_chunks(segments),
        })
    return documents


def wrap_block(text, width=60):
    return textwrap.fill(text or "(vazio)", width=width)


def generate_fig_convergence():
    print("📊 Gerando fig_convergence.png...")
    df = load_training_logs_csv("Normal")
    if df is None:
        df = load_live_training_logs()
    if df is None or df.empty:
        print("  ⚠️ Sem dados de treinamento.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for node_id, group in df.groupby("node_id"):
        grouped = group.groupby("round_number", as_index=False)["loss"].mean()
        ax.plot(grouped["round_number"], grouped["loss"], marker="o", linewidth=2, label=node_id)
    ax.set_xlabel("Rodada")
    ax.set_ylabel("CE Loss")
    ax.set_title("Convergência do treinamento federado textual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_convergence.png"))
    plt.close(fig)


def generate_fig_reconstruction():
    print("📊 Gerando fig_reconstruction.png...")
    model, dataset, vocab = load_model_and_vocab()
    if model is None:
        print("  ⚠️ Modelo global não encontrado.")
        return

    docs = build_document_samples(dataset, vocab, sample_count=3, parts_per_document=3)
    rows = []
    for doc in docs:
        reconstructed, ce_loss, accuracy, chunks = run_reconstruction(model, vocab, doc["text"])
        rows.append((doc["text"], reconstructed, ce_loss, accuracy, chunks))

    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.8 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for index, (original, reconstructed, ce_loss, accuracy, chunks) in enumerate(rows):
        left, right = axes[index]
        left.axis("off")
        right.axis("off")
        left.set_title(f"Documento {index + 1} | {chunks} chunks")
        right.set_title(f"Reconstruído | CE={ce_loss:.3f} | Acc={accuracy:.1f}%")
        left.text(0.0, 1.0, wrap_block(original), va="top", ha="left")
        right.text(0.0, 1.0, wrap_block(reconstructed), va="top", ha="left")
    fig.suptitle("Reconstrução semântica textual por chunks", y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_reconstruction.png"))
    plt.close(fig)
    generate_tab_reconstruction(rows)


def generate_tab_reconstruction(rows):
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Reconstrução semântica textual por documento.}",
        "\\label{tab:reconstruction}",
        "\\begin{tabular}{cccc}",
        "\\hline",
        f"\\textbf{{Documento}} & \\textbf{{Chunks}} & \\textbf{{CE Loss}} & \\textbf{{Accuracy (\\%)}} {LATEX_ROW_BREAK}",
        "\\hline",
    ]
    for index, (_, _, ce_loss, accuracy, chunks) in enumerate(rows, start=1):
        lines.append(f"{index} & {chunks} & {ce_loss:.4f} & {accuracy:.1f} {LATEX_ROW_BREAK}")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(TABLES_DIR, "tab_reconstruction.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_fig_completion():
    print("📊 Gerando fig_completion.png...")
    model, dataset, vocab = load_model_and_vocab()
    if model is None:
        print("  ⚠️ Modelo global não encontrado.")
        return

    document = build_document_samples(dataset, vocab, sample_count=1, parts_per_document=3)[0]["text"]
    configs = [("truncate", 0.3), ("truncate", 0.6), ("random", 0.4)]
    rows = []

    with torch.no_grad():
        for strategy, mask_ratio in configs:
            masked_tensors, masked_text = build_masked_document(document, vocab, strategy=strategy, mask_ratio=mask_ratio, chunk_size=TEXT_CHUNK_SIZE, overlap=0)
            completed_chunks = []
            ce_losses = []
            accuracies = []
            original_tensors, _ = text_to_chunk_tensors(document, vocab, chunk_size=TEXT_CHUNK_SIZE, overlap=0)
            for original_tensor, masked_tensor in zip(original_tensors, masked_tensors):
                output = model(masked_tensor.unsqueeze(0))
                logits = output[0] if isinstance(output, tuple) else output
                predictions = torch.argmax(logits, dim=-1).squeeze(0)
                completed_chunks.append(decode_tokens(predictions.tolist(), vocab))
                ce_losses.append(compute_cross_entropy(original_tensor.unsqueeze(0), logits))
                accuracies.append(compute_accuracy(original_tensor.unsqueeze(0), logits) * 100.0)
            rows.append({
                "strategy": strategy,
                "mask_ratio": mask_ratio,
                "masked_text": masked_text,
                "completed_text": stitch_text_chunks(completed_chunks),
                "ce_loss": sum(ce_losses) / len(ce_losses),
                "accuracy": sum(accuracies) / len(accuracies),
            })

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for index, row in enumerate(rows):
        ax_orig, ax_masked, ax_done = axes[index]
        for axis in (ax_orig, ax_masked, ax_done):
            axis.axis("off")
        ax_orig.set_title("Original")
        ax_masked.set_title(f"Enviado | {row['strategy']} | {row['mask_ratio']:.0%}")
        ax_done.set_title(f"Completo | CE={row['ce_loss']:.3f} | Acc={row['accuracy']:.1f}%")
        ax_orig.text(0.0, 1.0, wrap_block(document, width=48), va="top", ha="left")
        ax_masked.text(0.0, 1.0, wrap_block(row["masked_text"], width=48), va="top", ha="left")
        ax_done.text(0.0, 1.0, wrap_block(row["completed_text"], width=48), va="top", ha="left")
    fig.suptitle("Completação textual a partir de entrada parcial", y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_completion.png"))
    plt.close(fig)
    generate_tab_completion(rows)


def generate_tab_completion(rows):
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Completação textual por estratégia de mascaramento.}",
        "\\label{tab:completion}",
        "\\begin{tabular}{cccc}",
        "\\hline",
        f"\\textbf{{Estratégia}} & \\textbf{{Máscara (\\%)}} & \\textbf{{CE Loss}} & \\textbf{{Accuracy (\\%)}} {LATEX_ROW_BREAK}",
        "\\hline",
    ]
    for row in rows:
        lines.append(f"{row['strategy']} & {row['mask_ratio'] * 100:.0f} & {row['ce_loss']:.4f} & {row['accuracy']:.1f} {LATEX_ROW_BREAK}")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(TABLES_DIR, "tab_completion.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_fig_chaos():
    print("📊 Gerando fig_chaos.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    has_data = False
    for scenario in SCENARIOS:
        df = load_round_metrics_csv(scenario)
        if df is None or df.empty:
            continue
        has_data = True
        color = SCENARIO_COLORS[scenario]
        ax1.plot(df["round_number"], df["global_celoss"], color=color, marker="o", linewidth=2, label=scenario)
        ax2.plot(df["round_number"], df["global_accuracy"], color=color, marker="s", linewidth=2, label=scenario)
    if not has_data:
        live_df = load_live_round_metrics()
        if live_df is not None and not live_df.empty:
            ax1.plot(live_df["round_number"], live_df["global_celoss"], color="#2f5d62", marker="o", linewidth=2, label="Atual")
            ax2.plot(live_df["round_number"], live_df["global_accuracy"], color="#b85c38", marker="s", linewidth=2, label="Atual")
            has_data = True
    if not has_data:
        print("  ⚠️ Sem dados de caos.")
        plt.close(fig)
        return
    ax1.set_xlabel("Rodada")
    ax1.set_ylabel("CE Loss")
    ax1.set_title("CE Loss por cenário")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("Rodada")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy por cenário")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_chaos.png"))
    plt.close(fig)
    generate_tab_chaos()


def generate_tab_chaos():
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Resultados comparativos por cenário de caos no treinamento textual.}",
        "\\label{tab:chaos_results}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        f"\\textbf{{Cenário}} & \\textbf{{CE final}} & \\textbf{{Accuracy final}} & \\textbf{{Rodadas}} & \\textbf{{Melhor CE}} & \\textbf{{Melhor Acc.}} {LATEX_ROW_BREAK}",
        "\\hline",
    ]
    for scenario in SCENARIOS:
        df = load_round_metrics_csv(scenario)
        if df is None or df.empty:
            continue
        lines.append(f"{scenario} & {df['global_celoss'].iloc[-1]:.4f} & {df['global_accuracy'].iloc[-1]:.1f} & {len(df)} & {df['global_celoss'].min():.4f} & {df['global_accuracy'].max():.1f} {LATEX_ROW_BREAK}")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(TABLES_DIR, "tab_chaos_results.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_analysis():
    print("\n📝 ANÁLISE DOS RESULTADOS")
    print("=" * 60)
    for scenario in SCENARIOS:
        df = load_round_metrics_csv(scenario)
        if df is None or df.empty:
            continue
        print(f"\n🔬 {scenario}:")
        print(f"  CE final: {df['global_celoss'].iloc[-1]:.4f}")
        print(f"  Accuracy final: {df['global_accuracy'].iloc[-1]:.1f}%")
        print(f"  Melhor CE: {df['global_celoss'].min():.4f}")
        print(f"  Melhor accuracy: {df['global_accuracy'].max():.1f}%")


def main():
    ensure_dirs()
    print("=" * 60)
    print("📊 GERADOR DE FIGURAS E TABELAS - MODO TEXTO")
    print(f"   Fonte principal: {'metrics.db' if '--live' in sys.argv else 'results/'}")
    print("=" * 60)
    generate_fig_convergence()
    generate_fig_reconstruction()
    generate_fig_completion()
    generate_fig_chaos()
    generate_analysis()
    print(f"\n✅ Figuras salvas em: {os.path.abspath(FIGURES_DIR)}")
    print(f"✅ Tabelas salvas em: {os.path.abspath(TABLES_DIR)}")


if __name__ == "__main__":
    main()
