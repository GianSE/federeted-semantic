import os


def parse_label_list(raw_value, default_labels):
	if not raw_value:
		return default_labels
	parsed = []
	for item in raw_value.split(","):
		item = item.strip()
		if item:
			parsed.append(int(item))
	return parsed


DATASET = os.environ.get("DATASET", "mnist").lower()

DATASET_CONFIGS = {
	"mnist": {
		"display_name": "MNIST",
		"channels": 1,
		"image_size": 28,
		"class_names": [str(i) for i in range(10)],
		"label_kind": "dígito",
		"default_noniid_labels": [0, 1, 2, 3],
	},
	"fashion": {
		"display_name": "Fashion-MNIST",
		"channels": 1,
		"image_size": 28,
		"class_names": [
			"T-shirt/top",
			"Trouser",
			"Pullover",
			"Dress",
			"Coat",
			"Sandal",
			"Shirt",
			"Sneaker",
			"Bag",
			"Ankle boot",
		],
		"label_kind": "classe",
		"default_noniid_labels": [0, 1, 2, 3],
	},
	"cifar10": {
		"display_name": "CIFAR-10",
		"channels": 3,
		"image_size": 32,
		"class_names": [
			"airplane",
			"automobile",
			"bird",
			"cat",
			"deer",
			"dog",
			"frog",
			"horse",
			"ship",
			"truck",
		],
		"label_kind": "classe",
		"default_noniid_labels": [0, 1, 2, 3],
	},
}

if DATASET not in DATASET_CONFIGS:
	raise ValueError(f"DATASET inválido: {DATASET}. Opções: {', '.join(DATASET_CONFIGS)}")

DATASET_CONFIG = DATASET_CONFIGS[DATASET]
DATASET_DISPLAY_NAME = DATASET_CONFIG["display_name"]
IMAGE_CHANNELS = DATASET_CONFIG["channels"]
IMAGE_SIZE = DATASET_CONFIG["image_size"]
CLASS_NAMES = DATASET_CONFIG["class_names"]
LABEL_KIND = DATASET_CONFIG["label_kind"]
PIXELS_PER_IMAGE = IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
DEFAULT_NONIID_LABELS = DATASET_CONFIG["default_noniid_labels"]
TRAINING_PROFILE = os.environ.get("TRAINING_PROFILE", "standard").lower()

# === Modelo ===
DEFAULT_LATENT_DIM = "128" if DATASET == "cifar10" else "32"
LATENT_DIM = int(os.environ.get("LATENT_DIM", DEFAULT_LATENT_DIM))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "autoencoder")  # "autoencoder" ou "vae"
MODEL_BACKBONE = os.environ.get("MODEL_BACKBONE", "simple")  # "simple", "deep" ou "cifar"
RECON_SSIM_WEIGHT = float(os.environ.get("RECON_SSIM_WEIGHT", "0.0"))
VAE_BETA = float(os.environ.get("VAE_BETA", "1.0"))  # Peso da KL divergence (β-VAE)
MODEL_INIT_SEED = int(os.environ.get("MODEL_INIT_SEED", "42"))

# === Canal AWGN ===
CHANNEL_SNR_DB = os.environ.get("CHANNEL_SNR_DB", "")  # "" = sem canal; ex: "20" para 20 dB

# === Treinamento Federado ===
DEFAULT_LOCAL_EPOCHS = "3" if DATASET == "cifar10" and TRAINING_PROFILE == "quality" else "5"
DEFAULT_MAX_BATCHES = "96" if DATASET == "cifar10" and TRAINING_PROFILE == "quality" else "64"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", DEFAULT_LOCAL_EPOCHS))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
MAX_BATCHES_PER_EPOCH = int(os.environ.get("MAX_BATCHES_PER_EPOCH", DEFAULT_MAX_BATCHES))  # 0 = usa todo o dataset local
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
DATA_DIST = os.environ.get("DATA_DIST", "iid")  # "iid" ou "noniid"
NONIID_LABELS = parse_label_list(os.environ.get("NONIID_LABELS", ""), DEFAULT_NONIID_LABELS)

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
