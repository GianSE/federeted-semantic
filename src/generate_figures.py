"""
Gerador de figuras e tabelas para o paper.
Lê os CSVs de resultados e gera:
  - fig_convergence.png   (curvas de loss por cliente)
  - fig_reconstruction.png (originais vs reconstruídas)
  - fig_completion.png    (original | mascarada | completada)
  - fig_chaos.png         (convergência por cenário de caos)
  - tab_reconstruction.tex (tabela MSE/PSNR por dígito)
  - tab_completion.tex    (tabela completação por tipo/nível)
  - tab_chaos_results.tex (tabela comparativa cenários)

Uso:
    python generate_figures.py          (usa CSVs de results/)
    python generate_figures.py --live   (usa metrics.db diretamente)
"""

import os
import sys
import csv
import sqlite3
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DATASET_DISPLAY_NAME, LATENT_DIM, MODEL_TYPE, PIXELS_PER_IMAGE, CLASS_NAMES, LABEL_KIND, NONIID_LABELS
from image_utils import load_dataset, compute_mse, compute_psnr, compute_ssim, tensor_to_image_array, label_to_name

# ============================================================
# Config
# ============================================================
RESULTS_DIR = "results"
FIGURES_DIR = os.environ.get("FIGURES_DIR", os.path.join("..", "paper", "figures"))
TABLES_DIR = os.environ.get("TABLES_DIR", os.path.join("..", "paper", "tables"))
DB_FILE = "metrics.db"
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "global_model.pth")

SCENARIOS = ["Normal", "Leve", "Moderado", "Severo"]
SCENARIO_COLORS = {"Normal": "#2196F3", "Leve": "#4CAF50", "Moderado": "#FF9800", "Severo": "#F44336"}
CLIENT_COLORS = {"client-full": "#2196F3", "client-noisy": "#FF9800", "client-noniid": "#4CAF50"}
CLIENT_LABELS = {
    "client-full": "Full (Estável)",
    "client-noisy": "Noisy (Top-K)",
    "client-noniid": f"Non-IID ({LABEL_KIND}s {', '.join(str(label) for label in NONIID_LABELS)})",
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️ Checkpoint não encontrado: {checkpoint_path}. Pulando.")
        return False
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        print(f"  ℹ️ Checkpoint: {os.path.abspath(checkpoint_path)}")
        return True
    except RuntimeError as e:
        print(f"  ❌ Erro ao carregar checkpoint: {checkpoint_path}\n    {e}")
        return False


def plot_tensor(ax, tensor_like):
    image = tensor_to_image_array(tensor_like)
    if image.ndim == 2:
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(image)


def collect_class_examples(dataset):
    examples = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        if label not in examples:
            examples[label] = image
        if len(examples) == len(CLASS_NAMES):
            break
    return examples


# ============================================================
# Data Loading
# ============================================================
def load_training_logs_csv(scenario):
    path = os.path.join(RESULTS_DIR, f"{scenario}_training_logs.csv")
    if not os.path.exists(path):
        return None
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row["node_id"]
            if node not in data:
                data[node] = {"rounds": [], "losses": []}
            data[node]["rounds"].append(int(row["round_number"]))
            data[node]["losses"].append(float(row["loss"]))
    return data


def load_round_metrics_csv(scenario):
    path = os.path.join(RESULTS_DIR, f"{scenario}_round_metrics.csv")
    if not os.path.exists(path):
        return None
    rounds, mses, psnrs, ssims = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round_number"]))
            mses.append(float(row["global_mse"]))
            psnrs.append(float(row["global_psnr"]))
            ssims.append(float(row.get("global_ssim", 0)))
    return {"rounds": rounds, "mses": mses, "psnrs": psnrs, "ssims": ssims}


def load_from_db():
    """Load data directly from metrics.db (for --live mode)"""
    if not os.path.exists(DB_FILE):
        return None, None
    conn = sqlite3.connect(DB_FILE, timeout=10)
    
    # Training logs
    training = {}
    try:
        rows = conn.execute("SELECT node_id, round_number, loss FROM training_logs ORDER BY round_number").fetchall()
        for node, rnd, loss in rows:
            if node not in training:
                training[node] = {"rounds": [], "losses": []}
            training[node]["rounds"].append(rnd)
            training[node]["losses"].append(loss)
    except Exception:
        pass

    # Round metrics
    round_data = {"rounds": [], "mses": [], "psnrs": []}
    try:
        rows = conn.execute("SELECT round_number, global_mse, global_psnr FROM round_metrics ORDER BY round_number").fetchall()
        for rnd, mse, psnr in rows:
            round_data["rounds"].append(rnd)
            round_data["mses"].append(mse)
            round_data["psnrs"].append(psnr)
    except Exception:
        pass
    
    conn.close()
    return training, round_data


# ============================================================
# Figure 1: Convergence (Loss per client over rounds)
# ============================================================
def generate_fig_convergence():
    print("📊 Gerando fig_convergence.png...")
    
    # Try Normal scenario first, then fall back to live DB
    training = load_training_logs_csv("Normal")
    if training is None:
        training, _ = load_from_db()
    
    if not training:
        print("  ⚠️ Sem dados de treinamento. Pulando.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for client_id, values in sorted(training.items()):
        color = CLIENT_COLORS.get(client_id, "#999999")
        label = CLIENT_LABELS.get(client_id, client_id)
        rounds = values["rounds"]
        losses = values["losses"]
        
        # Average loss per round for this client
        round_loss = {}
        for r, l in zip(rounds, losses):
            round_loss[r] = l
        
        sorted_rounds = sorted(round_loss.keys())
        sorted_losses = [round_loss[r] for r in sorted_rounds]
        
        ax.plot(sorted_rounds, sorted_losses, color=color, label=label, 
                linewidth=2, marker='o', markersize=3, alpha=0.8)

    ax.set_xlabel("Rodada de Treinamento")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Convergência do Treinamento Federado")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    path = os.path.join(FIGURES_DIR, "fig_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


# ============================================================
# Figure 2: Reconstruction quality (original vs reconstructed)
# ============================================================
def generate_fig_reconstruction(checkpoint_path):
    print("📊 Gerando fig_reconstruction.png...")

    from model_utils import get_model
    
    model = get_model(MODEL_TYPE, LATENT_DIM)
    if not load_checkpoint(model, checkpoint_path):
        return
    model.eval()
    
    dataset = load_dataset(train=False)
    class_examples = collect_class_examples(dataset)
    selected_labels = list(range(min(5, len(CLASS_NAMES))))
    originals = []
    reconstructions = []
    labels = []
    mses = []
    psnrs = []
    
    for label in selected_labels:
        img = class_examples[label].unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            recon = output[0] if isinstance(output, tuple) else output
        originals.append(img)
        reconstructions.append(recon)
        labels.append(label_to_name(label))
        mses.append(compute_mse(img, recon))
        psnrs.append(compute_psnr(img, recon))
    
    fig, axes = plt.subplots(2, len(selected_labels), figsize=(12, 5))
    
    for i in range(len(selected_labels)):
        plot_tensor(axes[0, i], originals[i])
        axes[0, i].set_title(labels[i], fontsize=11)
        axes[0, i].axis('off')
        
        plot_tensor(axes[1, i], reconstructions[i])
        axes[1, i].set_title(f"MSE:{mses[i]:.4f}\nPSNR:{psnrs[i]:.1f}dB", fontsize=9)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Original", fontsize=12, rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel("Reconstruído", fontsize=12, rotation=0, labelpad=60, va='center')
    
    fig.suptitle(
        f"Reconstrução Semântica: {PIXELS_PER_IMAGE} pixels → {LATENT_DIM} valores latentes → {PIXELS_PER_IMAGE} pixels | {DATASET_DISPLAY_NAME}\nCheckpoint: {os.path.basename(checkpoint_path)}",
        fontsize=13,
        y=1.05,
    )
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, "fig_reconstruction.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")
    
    # Generate reconstruction table for ALL 10 digits
    generate_tab_reconstruction(model, dataset)


def generate_tab_reconstruction(model, dataset):
    """Tabela LaTeX: MSE, PSNR e SSIM por classe."""
    class_examples = collect_class_examples(dataset)
    
    rows = []
    for label in range(len(CLASS_NAMES)):
        img = class_examples[label].unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            recon = output[0] if isinstance(output, tuple) else output
        mse = compute_mse(img, recon)
        psnr = compute_psnr(img, recon)
        ssim = compute_ssim(img, recon)
        rows.append((label_to_name(label), mse, psnr, ssim, f"{PIXELS_PER_IMAGE / LATENT_DIM:.1f}$\\times$"))
    
    avg_mse = np.mean([r[1] for r in rows])
    avg_psnr = np.mean([r[2] for r in rows])
    avg_ssim = np.mean([r[3] for r in rows])
    
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Métricas de reconstrução semântica por {LABEL_KIND} em {DATASET_DISPLAY_NAME}.}}\n"
    tex += "\\label{tab:reconstruction}\n"
    tex += "\\begin{tabular}{ccccc}\n\\hline\n"
    tex += f"\\textbf{{{LABEL_KIND.title()}}} & \\textbf{{MSE}} & \\textbf{{PSNR (dB)}} & \\textbf{{SSIM}} & \\textbf{{Compressão}} \\\\\n\\hline\n"
    
    for d, mse, psnr, ssim, comp in rows:
        tex += f"{d} & {mse:.4f} & {psnr:.1f} & {ssim:.4f} & {comp} \\\\\n"
    
    tex += "\\hline\n"
    tex += f"\\textbf{{Média}} & \\textbf{{{avg_mse:.4f}}} & \\textbf{{{avg_psnr:.1f}}} & \\textbf{{{avg_ssim:.4f}}} & {PIXELS_PER_IMAGE / LATENT_DIM:.1f}$\\times$ \\\\\n"
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    path = os.path.join(TABLES_DIR, "tab_reconstruction.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  ✅ {path}")


# ============================================================
# Figure 3: Image completion (original | masked | completed)
# ============================================================
def generate_fig_completion(checkpoint_path):
    print("📊 Gerando fig_completion.png...")

    from model_utils import get_model
    from image_utils import mask_image_bottom, mask_image_random, mask_image_right
    
    model = get_model(MODEL_TYPE, LATENT_DIM)
    if not load_checkpoint(model, checkpoint_path):
        return
    model.eval()
    
    dataset = load_dataset(train=False)
    img, label = dataset[3]
    original = img.unsqueeze(0)
    
    configs = [
        ("Metade Inferior 25%", mask_image_bottom, 0.25),
        ("Pixels Aleatórios 50%", mask_image_random, 0.50),
        ("Metade Inferior 75%", mask_image_bottom, 0.75),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    col_titles = ["Original", "Parcial Enviada", "Completado pelo Modelo"]
    
    completion_data = []  # For table generation
    
    for row_idx, (title, mask_fn, pct) in enumerate(configs):
        masked = mask_fn(original, pct)
        with torch.no_grad():
            output = model(masked)
            completed = output[0] if isinstance(output, tuple) else output
        
        mse = compute_mse(original, completed)
        psnr = compute_psnr(original, completed)
        
        plot_tensor(axes[row_idx, 0], original)
        axes[row_idx, 0].set_ylabel(title, fontsize=10, rotation=0, labelpad=100, va='center')
        axes[row_idx, 0].axis('off')
        
        plot_tensor(axes[row_idx, 1], masked)
        axes[row_idx, 1].set_title(f"{(1-pct)*100:.0f}% enviado" if row_idx == 0 else "")
        axes[row_idx, 1].axis('off')
        
        plot_tensor(axes[row_idx, 2], completed)
        axes[row_idx, 2].set_title(f"MSE:{mse:.4f} PSNR:{psnr:.1f}dB", fontsize=9)
        axes[row_idx, 2].axis('off')
    
    for i, title in enumerate(col_titles):
        axes[0, i].set_title(title, fontsize=12, fontweight='bold')
    
    fig.suptitle(
        f"Completação de Imagem Parcial ({LABEL_KIND.title()}: {label_to_name(label)}) | {DATASET_DISPLAY_NAME}\nCheckpoint: {os.path.basename(checkpoint_path)}",
        fontsize=14,
        y=1.05,
    )
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, "fig_completion.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")
    
    # Generate completion table with full test matrix
    generate_tab_completion(model, original)


def generate_tab_completion(model, original):
    """Tabela LaTeX: completação por tipo de máscara e porcentagem"""
    from image_utils import (mask_image_bottom, mask_image_random,
                             mask_image_right, compute_mse, compute_psnr, compute_ssim)
    
    mask_types = [
        ("Metade Inferior", mask_image_bottom),
        ("Pixels Aleatórios", mask_image_random),
        ("Metade Direita", mask_image_right),
    ]
    percentages = [0.25, 0.50, 0.75]
    
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Qualidade de completação por tipo e nível de mascaramento.}\n"
    tex += "\\label{tab:completion}\n"
    tex += "\\begin{tabular}{lcccc}\n\\hline\n"
    tex += "\\textbf{Tipo de Máscara} & \\textbf{\\% Mascarado} & \\textbf{MSE} & \\textbf{PSNR (dB)} & \\textbf{SSIM} \\\\\n\\hline\n"
    
    for name, fn in mask_types:
        for pct in percentages:
            masked = fn(original, pct)
            with torch.no_grad():
                output = model(masked)
                completed = output[0] if isinstance(output, tuple) else output
            mse = compute_mse(original, completed)
            psnr = compute_psnr(original, completed)
            ssim = compute_ssim(original, completed)
            tex += f"{name} & {int(pct*100)}\\% & {mse:.4f} & {psnr:.1f} & {ssim:.4f} \\\\\n"
        tex += "\\hline\n"
    
    tex += "\\end{tabular}\n\\end{table}\n"
    
    path = os.path.join(TABLES_DIR, "tab_completion.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  ✅ {path}")


# ============================================================
# Figure 4: Chaos convergence (4 scenarios overlaid)
# ============================================================
def generate_fig_chaos():
    print("📊 Gerando fig_chaos.png...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    has_data = False
    
    for scenario in SCENARIOS:
        round_data = load_round_metrics_csv(scenario)
        if round_data is None:
            continue
        has_data = True
        color = SCENARIO_COLORS[scenario]
        
        ax1.plot(round_data["rounds"], round_data["mses"], 
                color=color, label=scenario, linewidth=2, marker='o', markersize=3)
        ax2.plot(round_data["rounds"], round_data["psnrs"],
                color=color, label=scenario, linewidth=2, marker='s', markersize=3)
    
    if not has_data:
        print("  ⚠️ Sem dados de cenários. Pulando.")
        plt.close(fig)
        return
    
    ax1.set_xlabel("Rodada")
    ax1.set_ylabel("MSE Global")
    ax1.set_title("MSE por Cenário de Caos")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.set_xlabel("Rodada")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR por Cenário de Caos")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle("Impacto da Injeção de Caos na Convergência do Treinamento Federado", fontsize=14, y=1.02)
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, "fig_chaos.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")
    
    # Generate chaos results table
    generate_tab_chaos()


def generate_tab_chaos():
    """Tabela LaTeX: resultados comparativos por cenário"""
    
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Resultados comparativos por cenário de injeção de caos.}\n"
    tex += "\\label{tab:chaos_results}\n"
    tex += "\\small\n"
    tex += "\\begin{tabular}{lcccccc}\n\\hline\n"
    tex += "\\textbf{Cenário} & \\textbf{MSE} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{Rod.} & \\textbf{Melhor} & \\textbf{Melhor} \\\\\n"
    tex += " & \\textbf{Final} & \\textbf{(dB)} & \\textbf{Final} & & \\textbf{MSE} & \\textbf{PSNR (dB)} \\\\\n\\hline\n"
    
    for scenario in SCENARIOS:
        round_data = load_round_metrics_csv(scenario)
        if round_data is None:
            continue
        
        final_mse = round_data["mses"][-1] if round_data["mses"] else 0
        final_psnr = round_data["psnrs"][-1] if round_data["psnrs"] else 0
        final_ssim = round_data["ssims"][-1] if round_data.get("ssims") else 0
        best_mse = min(round_data["mses"]) if round_data["mses"] else 0
        best_psnr = max(round_data["psnrs"]) if round_data["psnrs"] else 0
        total_rounds = len(round_data["rounds"])
        
        tex += f"{scenario} & {final_mse:.4f} & {final_psnr:.1f} & {final_ssim:.4f} & {total_rounds} & {best_mse:.4f} & {best_psnr:.1f} \\\\\n"
    
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    path = os.path.join(TABLES_DIR, "tab_chaos_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  ✅ {path}")


# ============================================================
# Analysis text generator
# ============================================================
def generate_analysis():
    print("\n📝 ANÁLISE DOS RESULTADOS")
    print("=" * 60)
    
    for scenario in SCENARIOS:
        data = load_round_metrics_csv(scenario)
        training = load_training_logs_csv(scenario)
        if data is None:
            continue
        
        print(f"\n🔬 {scenario}:")
        print(f"  MSE Final: {data['mses'][-1]:.6f}")
        print(f"  PSNR Final: {data['psnrs'][-1]:.1f} dB")
        print(f"  Melhor MSE: {min(data['mses']):.6f} (rodada {data['rounds'][data['mses'].index(min(data['mses']))]})")
        print(f"  Melhor PSNR: {max(data['psnrs']):.1f} dB")
        
        if training:
            for client, vals in sorted(training.items()):
                if vals["losses"]:
                    print(f"  {client}: Loss final = {vals['losses'][-1]:.6f}")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs()

    args = sys.argv[1:]
    live_mode = "--live" in args
    checkpoint_path = MODEL_CHECKPOINT

    if "--checkpoint" in args:
        idx = args.index("--checkpoint")
        if idx + 1 >= len(args):
            raise SystemExit("Uso: python generate_figures.py [--live] [--checkpoint CAMINHO_DO_MODELO]")
        checkpoint_path = args[idx + 1]

    # Tenta encontrar checkpoint compatível automaticamente se o padrão não existir ou for incompatível
    def find_compatible_checkpoint():
        candidates = [
            os.path.join("results", "Normal_model.pth"),
            os.path.join("results", "Centralizado_AE_model.pth"),
            os.path.join("results", "Centralizado_VAE_model.pth"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return checkpoint_path

    # Testa se o checkpoint padrão é carregável, senão tenta alternativos
    from model_utils import get_model
    test_model = get_model(MODEL_TYPE, LATENT_DIM)
    if not load_checkpoint(test_model, checkpoint_path):
        alt_ckpt = find_compatible_checkpoint()
        if alt_ckpt != checkpoint_path:
            print(f"  ⚠️ Tentando checkpoint alternativo: {alt_ckpt}")
            checkpoint_path = alt_ckpt
        else:
            print("  ❌ Nenhum checkpoint compatível encontrado. Figuras de reconstrução não serão geradas.")

    print("=" * 60)
    print("📊 GERADOR DE FIGURAS E TABELAS")
    print(f"   Modo: {'Live (metrics.db)' if live_mode else 'CSVs (results/)'}")
    print(f"   Checkpoint: {checkpoint_path}")
    print("=" * 60)

    generate_fig_convergence()
    generate_fig_reconstruction(checkpoint_path)
    generate_fig_completion(checkpoint_path)
    generate_fig_chaos()
    generate_analysis()

    print(f"\n✅ Figuras salvas em: {os.path.abspath(FIGURES_DIR)}")
    print(f"✅ Tabelas salvas em: {os.path.abspath(TABLES_DIR)}")


if __name__ == "__main__":
    main()
