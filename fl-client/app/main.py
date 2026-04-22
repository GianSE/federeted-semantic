"""
fl_client/app/main.py
---------------------
Federated learning client supporting:
- baseline and compressed transport modes
- byte accounting for uploads
- deterministic seed configuration
"""

import json
import os
import threading
import time
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torch.utils.data import DataLoader, Subset

CLIENT_ID = int(os.environ.get("CLIENT_ID", "1"))
N_CLIENTS = int(os.environ.get("N_CLIENTS", "2"))
SERVER_URL = os.environ.get("FL_SERVER_URL", "http://fl-server:8100")
FL_WEIGHTS_DIR = Path(os.environ.get("FL_WEIGHTS_DIR", "/fl-weights"))
ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "/ml-data"))
DATASETS_DIR = ML_DATA_DIR / "datasets"
FL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=f"FL Client {CLIENT_ID}")
_all_logs: list[str] = []


def _emit(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [client-{CLIENT_ID}] {msg}"
    _all_logs.append(line)
    print(line, flush=True)


def _global_weights_path(rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"global_round_{rnd}.pth"


def _client_weights_path(rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"client_{CLIENT_ID}_round_{rnd}.pth"


def _load_dataset(dataset: str):
    import torchvision
    import torchvision.transforms as T

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.MNIST(str(DATASETS_DIR), train=True, download=True, transform=transform)
    if dataset == "fashion":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.FashionMNIST(str(DATASETS_DIR), train=True, download=True, transform=transform)
    if dataset == "cifar10":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.CIFAR10(str(DATASETS_DIR), train=True, download=True, transform=transform)
    if dataset == "cifar100":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.CIFAR100(str(DATASETS_DIR), train=True, download=True, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset}")


def _wait_for_server(max_wait: int = 120) -> bool:
    _emit(f"Waiting for fl-server at {SERVER_URL}...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.ok:
                _emit("Connected to fl-server")
                return True
        except Exception:
            pass
        time.sleep(3)
    _emit("WARNING: could not reach fl-server; will keep retrying")
    return False


def _quantize_tensor(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    qmax = float((2 ** (bits - 1)) - 1)
    tensor_f = tensor.detach().float().cpu()
    max_abs = float(tensor_f.abs().max().item())
    if max_abs == 0.0:
        return torch.zeros_like(tensor_f, dtype=torch.int8), 1.0
    scale = max_abs / qmax
    q = torch.clamp(torch.round(tensor_f / scale), -qmax, qmax).to(torch.int8)
    return q, float(scale)


def _compress_state_dict(state: dict[str, torch.Tensor], bits: int) -> dict:
    payload: dict[str, object] = {"__compressed__": True, "bits": int(bits), "tensors": {}}
    tensors: dict[str, dict] = {}
    for key, tensor in state.items():
        q, scale = _quantize_tensor(tensor, bits)
        tensors[key] = {
            "shape": list(tensor.shape),
            "scale": scale,
            "data": q,
        }
    payload["tensors"] = tensors
    return payload


def _decompress_state_dict(payload: dict) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    tensors = payload.get("tensors", {})
    for key, meta in tensors.items():
        q = meta["data"].to(torch.float32)
        scale = float(meta["scale"])
        shape = tuple(meta["shape"])
        state[key] = (q * scale).reshape(shape)
    return state


def _load_transport_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj.get("__compressed__"):
        return _decompress_state_dict(obj)
    return obj


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _save_transport_state(state: dict[str, torch.Tensor], path: Path, mode: str, bits: int) -> int:
    if mode == "compressed":
        _atomic_torch_save(_compress_state_dict(state, bits), path)
    else:
        _atomic_torch_save(state, path)
    return int(path.stat().st_size)


def _background_training_loop() -> None:
    import sys

    sys.path.insert(0, "/app")
    from core.image_utils import DATASET_META, apply_awgn_noise, apply_random_pixel_mask
    from core.model_utils import get_model

    startup_delay = (CLIENT_ID - 1) * 5
    _emit(f"Client {CLIENT_ID}/{N_CLIENTS} waiting {startup_delay}s before start")
    time.sleep(startup_delay)
    _wait_for_server()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_rounds: set[int] = set()
    current_ds_name: str | None = None
    shard = None

    _emit(f"Device={device}; polling server")

    while True:
        try:
            status = requests.get(f"{SERVER_URL}/round/status", timeout=5).json()
        except Exception as exc:
            _emit(f"Cannot reach server: {exc}")
            time.sleep(3)
            continue

        state = status.get("state", "idle")
        rnd = status.get("round", 0)

        if state in ("done", "stopped", "error"):
            _emit(f"Server state={state}; exiting")
            break
        if state in ("idle", "starting") or rnd == 0:
            time.sleep(2)
            continue
        if rnd in trained_rounds:
            time.sleep(2)
            continue
        if not status.get("weights_ready"):
            time.sleep(1)
            continue

        config_path = FL_WEIGHTS_DIR / "training_config.json"
        if not config_path.exists():
            time.sleep(2)
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _emit(f"Cannot parse config: {exc}")
            time.sleep(2)
            continue

        dataset = config["dataset"]
        model_type = config["model"]
        epochs = int(config["epochs"])
        awgn_cfg = config.get("awgn", {})
        awgn_enabled = bool(awgn_cfg.get("enabled", False))
        awgn_snr = awgn_cfg.get("snr_db", 10.0)
        masking_cfg = config.get("masking", {})
        masking_enabled = bool(masking_cfg.get("enabled", False))
        masking_drop_rate = float(masking_cfg.get("drop_rate", 0.25))
        masking_fill_value = float(masking_cfg.get("fill_value", 0.0))
        compression_mode = str(config.get("compression_mode", "baseline"))
        compression_bits = int(config.get("compression_bits", 8))
        seed = int(config.get("seed", 42))
        torch.manual_seed(seed + CLIENT_ID)

        if dataset != current_ds_name:
            try:
                full_ds = _load_dataset(dataset)
            except Exception as exc:
                _emit(f"Dataset load failed: {exc}")
                time.sleep(5)
                continue

            total = len(full_ds)
            shard_sz = total // N_CLIENTS
            start_idx = (CLIENT_ID - 1) * shard_sz
            end_idx = start_idx + shard_sz if CLIENT_ID < N_CLIENTS else total
            shard = Subset(full_ds, list(range(start_idx, end_idx)))
            current_ds_name = dataset
            _emit(f"Loaded shard [{start_idx},{end_idx}) size={len(shard)}")

        gpath = _global_weights_path(rnd)
        if not gpath.exists():
            time.sleep(2)
            continue

        meta = DATASET_META.get(dataset, DATASET_META["mnist"])
        channels = meta["channels"]
        img_size = meta["height"]
        pixels_pp = channels * meta["height"] * meta["width"]

        local_model = get_model(model_type, latent_dim=32, input_channels=channels, image_size=img_size)
        local_model.load_state_dict(_load_transport_state(gpath))
        local_model = local_model.to(device)

        loader = DataLoader(shard, batch_size=128, shuffle=True, num_workers=0)
        optimizer = optim.Adam(local_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        kl_weight = 0.005
        local_model.train()

        last_avg = 0.0
        for ep in range(epochs):
            run = cnt = 0
            for bi, (data, _) in enumerate(loader):
                data = data.to(device)
                corrupted = data
                if masking_enabled:
                    corrupted = apply_random_pixel_mask(corrupted, masking_drop_rate, masking_fill_value)
                if awgn_enabled:
                    corrupted = apply_awgn_noise(corrupted, awgn_snr)
                optimizer.zero_grad()
                if model_type == "cnn_vae":
                    recon, mu, logvar = local_model(corrupted)
                    rloss = criterion(recon, data)
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kld /= data.size(0) * pixels_pp
                    loss = rloss + kl_weight * kld
                else:
                    recon = local_model(corrupted)
                    loss = criterion(recon, data)
                loss.backward()
                optimizer.step()
                run += loss.item()
                cnt += 1
                if bi % 60 == 0:
                    _emit(f"[round {rnd} ep {ep+1}/{epochs}] batch {bi}/{len(loader)} loss={loss.item():.5f}")
            last_avg = run / max(cnt, 1)
            _emit(f"[round {rnd} ep {ep+1}/{epochs}] avg_loss={last_avg:.5f}")

        cpath = _client_weights_path(rnd)
        bytes_tx = _save_transport_state(local_model.state_dict(), cpath, compression_mode, compression_bits)
        _emit(f"[round {rnd}] local weights saved bytes={bytes_tx} mode={compression_mode}")

        tries = 0
        while tries < 5:
            try:
                r = requests.post(
                    f"{SERVER_URL}/round/submit/{CLIENT_ID}",
                    json={
                        "loss": last_avg,
                        "client_id": CLIENT_ID,
                        "bytes_transmitted": bytes_tx,
                    },
                    timeout=15,
                )
                if r.ok:
                    _emit(f"[round {rnd}] submission accepted")
                    break
                _emit(f"[round {rnd}] server rejected {r.status_code}: {r.text}")
            except Exception as exc:
                _emit(f"[round {rnd}] submit error: {exc}")
            tries += 1
            time.sleep(3)

        trained_rounds.add(rnd)

    _emit("Training loop exited")


@app.on_event("startup")
def startup_event() -> None:
    t = threading.Thread(target=_background_training_loop, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {"status": "ok", "client_id": CLIENT_ID, "n_clients": N_CLIENTS}


@app.get("/logs")
def get_logs(since: int = 0):
    return {"lines": _all_logs[since:], "total": len(_all_logs)}


@app.get("/logs/stream")
def logs_stream():
    def gen():
        pos = 0
        while True:
            while pos < len(_all_logs):
                yield f"data: {_all_logs[pos]}\n\n"
                pos += 1
            time.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")
