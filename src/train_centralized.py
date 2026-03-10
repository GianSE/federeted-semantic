"""
Treinamento centralizado (baseline) para comparação com FL.
Treina o mesmo modelo textual com todos os dados em um único nó.

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

from model_utils import TextVAE, get_model
from text_utils import load_text_dataset, get_random_batch_text, compute_accuracy, compute_cross_entropy, MAX_SEQ_LEN, MAX_VOCAB_SIZE
from config import LATENT_DIM, BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS

RESULTS_DIR = "results"
ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))


def train_centralized(model_type="autoencoder", snr_db=None, rounds=ROUNDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = get_model(model_type, vocab_size=MAX_VOCAB_SIZE, seq_len=MAX_SEQ_LEN, latent_dim=LATENT_DIM)
    is_vae = model_type == "vae"
    tag = "TextVAE" if is_vae else "TextAE"

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    dataset_train, _ = load_text_dataset(train=True)
    dataset_test, _ = load_text_dataset(train=False)

    print(f"🏋️ Treinamento Centralizado — {tag}")
    print(f"   Rodadas: {rounds} | Épocas/rodada: {LOCAL_EPOCHS} | Batch: {BATCH_SIZE}")
    if snr_db is not None:
        print(f"   Canal AWGN: SNR={snr_db} dB")

    metrics = []
    start = time.time()

    for r in range(1, rounds + 1):
        model.train()
        inputs, targets = get_random_batch_text(dataset_train, batch_size=BATCH_SIZE)

        for _ in range(LOCAL_EPOCHS):
            optimizer.zero_grad()
            if is_vae:
                logits, mu, logvar = model(inputs, snr_db=snr_db)
                recon_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                kl_loss = TextVAE.kl_divergence(mu, logvar)
                loss = recon_loss + kl_loss
            else:
                logits = model(inputs, snr_db=snr_db)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

        # Avaliação
        model.eval()
        test_inputs, test_targets = get_random_batch_text(dataset_test, batch_size=100)
        with torch.no_grad():
            output = model(test_inputs)
            test_logits = output[0] if isinstance(output, tuple) else output

        ce_loss = compute_cross_entropy(test_targets, test_logits)
        accuracy = compute_accuracy(test_targets, test_logits) * 100.0
        metrics.append({"round": r, "ce_loss": ce_loss, "accuracy": accuracy})

        elapsed = time.time() - start
        print(f"\r  Rodada {r}/{rounds} | CE Loss={ce_loss:.4f} | Accuracy={accuracy:.1f}% | {elapsed:.0f}s", end="", flush=True)

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
        writer = csv.DictWriter(f, fieldnames=["round_number", "global_celoss", "global_accuracy", "timestamp", "chaos_scenario"])
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "round_number": m["round"],
                "global_celoss": m["ce_loss"],
                "global_accuracy": m["accuracy"],
                "timestamp": "",
                "chaos_scenario": f"Centralizado{suffix}",
            })
    print(f"📁 CSV salvo: {csv_path}")

    final = metrics[-1]
    print(f"\n✅ Centralizado {tag} finalizado: CE Loss={final['ce_loss']:.4f} | Accuracy={final['accuracy']:.1f}%")
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
