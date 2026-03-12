import os

# === Modelo ===
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "autoencoder")  # "autoencoder" ou "vae"
VAE_BETA = float(os.environ.get("VAE_BETA", "1.0"))  # Peso da KL divergence (β-VAE)
MODEL_INIT_SEED = int(os.environ.get("MODEL_INIT_SEED", "42"))

# === Canal AWGN ===
CHANNEL_SNR_DB = os.environ.get("CHANNEL_SNR_DB", "")  # "" = sem canal; ex: "20" para 20 dB

# === Treinamento Federado ===
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
MAX_BATCHES_PER_EPOCH = int(os.environ.get("MAX_BATCHES_PER_EPOCH", "64"))  # 0 = usa todo o dataset local
REQUIRED_CLIENTS = int(os.environ.get("REQUIRED_CLIENTS", "3"))
COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", "0.6"))  # Top-K: zera X% dos pesos
COMPRESSED_CLIENTS = {
	client_id.strip()
	for client_id in os.environ.get("COMPRESSED_CLIENTS", "client-noisy").split(",")
	if client_id.strip()
}

# === FedProx ===
FEDPROX_MU = float(os.environ.get("FEDPROX_MU", "0.0"))  # 0 = FedAvg puro

# === Dataset ===
DATASET = os.environ.get("DATASET", "mnist")  # "mnist" ou "fashion"
DATA_DIST = os.environ.get("DATA_DIST", "iid")  # "iid" ou "noniid"
NONIID_LABELS = [0, 1, 2, 3]  # Dígitos para cliente Non-IID

# === Compressão Adaptativa ===
ADAPTIVE_COMPRESSION = os.environ.get("ADAPTIVE_COMPRESSION", "false").lower() == "true"

# === Cenário de Caos ===
CHAOS_SCENARIO = os.environ.get("CHAOS_SCENARIO", "Normal")

# === Experimentos ===
EXPERIMENT_ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "100"))  # Imagens de teste para avaliação do modelo global
ROUND_TIMEOUT = int(os.environ.get("ROUND_TIMEOUT", "30"))  # Segundos para agregar com clientes disponíveis

# === Servidor ===
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max payload
