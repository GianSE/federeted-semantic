import os


def parse_label_list(raw_value, default_labels):
	if not raw_value:
		return default_labels
	labels = []
	for item in raw_value.split(","):
		item = item.strip()
		if item:
			labels.append(int(item))
	return labels or default_labels


# === Modelo ===
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "autoencoder")
VAE_BETA = float(os.environ.get("VAE_BETA", "1.0"))

# === Canal AWGN ===
CHANNEL_SNR_DB = os.environ.get("CHANNEL_SNR_DB", "")

# === Treinamento Federado ===
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
REQUIRED_CLIENTS = int(os.environ.get("REQUIRED_CLIENTS", "3"))
COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", "0.6"))

# === FedProx ===
FEDPROX_MU = float(os.environ.get("FEDPROX_MU", "0.0"))

# === Dataset textual ===
TEXT_DATASET = os.environ.get("TEXT_DATASET", "local_corpus")
DATA_DIST = os.environ.get("DATA_DIST", "iid")
NONIID_LABELS = parse_label_list(os.environ.get("NONIID_LABELS", "1,2"), [1, 2])

# === Texto longo ===
TEXT_CHUNK_SIZE = int(os.environ.get("TEXT_CHUNK_SIZE", "50"))
TEXT_CHUNK_OVERLAP = int(os.environ.get("TEXT_CHUNK_OVERLAP", "0"))
TEXT_RECONSTRUCTION_MODE = os.environ.get("TEXT_RECONSTRUCTION_MODE", "semantic")

# === Compressão Adaptativa ===
ADAPTIVE_COMPRESSION = os.environ.get("ADAPTIVE_COMPRESSION", "false").lower() == "true"

# === Cenário de Caos ===
CHAOS_SCENARIO = os.environ.get("CHAOS_SCENARIO", "Normal")

# === Experimentos ===
EXPERIMENT_ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "100"))
ROUND_TIMEOUT = int(os.environ.get("ROUND_TIMEOUT", "30"))

# === Servidor ===
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
