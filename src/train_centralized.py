"""
Treinamento centralizado (baseline) para comparação com FL.
Treina o mesmo modelo com todos os dados em um único nó.

Uso:
    python train_centralized.py                    # AE padrão
    python train_centralized.py --model vae        # VAE
    python train_centralized.py --snr 20           # Com canal AWGN 20 dB
"""

import os
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from model_utils import ImageVAE, get_model
from image_utils import load_mnist, compute_ssim
from config import LATENT_DIM, BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS, MAX_BATCHES_PER_EPOCH, TEST_BATCH_SIZE

RESULTS_DIR = "results"
ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))


def evaluate_model(model, data_loader):
    total_sq_error = 0.0
    total_pixels = 0
    total_ssim = 0.0
    total_images = 0

    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            output = model(images)
            reconstructed = output[0] if isinstance(output, tuple) else output
            diff = images - reconstructed

            total_sq_error += torch.sum(diff * diff).item()
            total_pixels += diff.numel()

            batch_size = images.size(0)
            total_images += batch_size
            total_ssim += compute_ssim(images, reconstructed) * batch_size

    mse = total_sq_error / total_pixels if total_pixels else 0.0
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    ssim = total_ssim / total_images if total_images else 0.0
    return mse, psnr, ssim


def train_centralized(model_type="autoencoder", snr_db=None, rounds=ROUNDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = get_model(model_type, LATENT_DIM)
    is_vae = model_type == "vae"
    tag = "VAE" if is_vae else "AE"

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset_train = load_mnist(train=True)
    dataset_test = load_mnist(train=False)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False)
    max_batches = MAX_BATCHES_PER_EPOCH if MAX_BATCHES_PER_EPOCH > 0 else None

    print(f"🏋️ Treinamento Centralizado — {tag}")
    print(f"   Rodadas: {rounds} | Épocas/rodada: {LOCAL_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   Batches/época: {max_batches or 'todos'}")
    if snr_db is not None:
        print(f"   Canal AWGN: SNR={snr_db} dB")

    metrics = []
    start = time.time()

    for r in range(1, rounds + 1):
        model.train()
        train_losses = []

        for _ in range(LOCAL_EPOCHS):
            for batch_idx, (images, _) in enumerate(train_loader, start=1):
                if max_batches is not None and batch_idx > max_batches:
                    break

                optimizer.zero_grad()
                if is_vae:
                    recon, mu, logvar = model(images, snr_db=snr_db)
                    recon_loss = criterion(recon, images)
                    kl_loss = ImageVAE.kl_divergence(mu, logvar)
                    loss = recon_loss + kl_loss
                else:
                    recon = model(images, snr_db=snr_db)
                    loss = criterion(recon, images)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        # Avaliação
        mse, psnr, ssim = evaluate_model(model, test_loader)
        metrics.append({"round": r, "mse": mse, "psnr": psnr, "ssim": ssim})

        elapsed = time.time() - start
        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f"\r  Rodada {r}/{rounds} | Loss={avg_loss:.6f} | MSE={mse:.6f} | PSNR={psnr:.1f} dB | SSIM={ssim:.4f} | {elapsed:.0f}s", end="", flush=True)

    print()

    # Salvar modelo
    suffix = f"_{tag}"
    if snr_db is not None:
        suffix += f"_snr{int(snr_db)}"
    model_path = os.path.join(RESULTS_DIR, f"Centralizado{suffix}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Modelo salvo: {model_path}")

    # Exportar CSV
    csv_path = os.path.join(RESULTS_DIR, f"Centralizado{suffix}_round_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round_number", "global_mse", "global_psnr", "timestamp", "chaos_scenario", "global_ssim"])
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "round_number": m["round"],
                "global_mse": m["mse"],
                "global_psnr": m["psnr"],
                "timestamp": "",
                "chaos_scenario": f"Centralizado{suffix}",
                "global_ssim": m["ssim"],
            })
    print(f"📁 CSV salvo: {csv_path}")

    final = metrics[-1]
    print(f"\n✅ Centralizado {tag} finalizado: MSE={final['mse']:.6f} | PSNR={final['psnr']:.1f} dB | SSIM={final['ssim']:.4f}")
    return metrics


if __name__ == "__main__":
    model_type = "autoencoder"
    snr = None
    for arg in sys.argv[1:]:
        if arg == "--model" or arg == "-m":
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                model_type = sys.argv[idx + 1]
        if arg == "--snr":
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                snr = float(sys.argv[idx + 1])
        if arg == "vae":
            model_type = "vae"

    train_centralized(model_type=model_type, snr_db=snr)
