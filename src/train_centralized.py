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

from model_utils import ImageAutoencoder, ImageVAE, get_model
from image_utils import load_mnist, get_random_batch, compute_mse, compute_psnr, compute_ssim
from config import LATENT_DIM, BATCH_SIZE, LEARNING_RATE

RESULTS_DIR = "results"
ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))
LOCAL_EPOCHS = 5


def train_centralized(model_type="autoencoder", snr_db=None, rounds=ROUNDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = get_model(model_type, LATENT_DIM)
    is_vae = model_type == "vae"
    tag = "VAE" if is_vae else "AE"

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset_train = load_mnist(train=True)
    dataset_test = load_mnist(train=False)

    print(f"🏋️ Treinamento Centralizado — {tag}")
    print(f"   Rodadas: {rounds} | Épocas/rodada: {LOCAL_EPOCHS} | Batch: {BATCH_SIZE}")
    if snr_db is not None:
        print(f"   Canal AWGN: SNR={snr_db} dB")

    metrics = []
    start = time.time()

    for r in range(1, rounds + 1):
        model.train()
        images, _ = get_random_batch(dataset_train, batch_size=BATCH_SIZE)

        for _ in range(LOCAL_EPOCHS):
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

        # Avaliação
        model.eval()
        test_images, _ = get_random_batch(dataset_test, batch_size=100)
        with torch.no_grad():
            output = model(test_images)
            test_recon = output[0] if isinstance(output, tuple) else output

        mse = compute_mse(test_images, test_recon)
        psnr = compute_psnr(test_images, test_recon)
        ssim = compute_ssim(test_images, test_recon)
        metrics.append({"round": r, "mse": mse, "psnr": psnr, "ssim": ssim})

        elapsed = time.time() - start
        print(f"\r  Rodada {r}/{rounds} | MSE={mse:.6f} | PSNR={psnr:.1f} dB | SSIM={ssim:.4f} | {elapsed:.0f}s", end="", flush=True)

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
