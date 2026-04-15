"""
fl_client/app/main.py
---------------------
Federated Learning Client container.

Each instance trains a local model shard and submits weights to the fl-server.

Environment variables:
    CLIENT_ID      : int  — unique client ID (1, 2, 3 ...)
    N_CLIENTS      : int  — total number of clients (for data partitioning)
    FL_SERVER_URL  : str  — base URL of the fl-server
    FL_WEIGHTS_DIR : str  — path to shared Docker volume for weight files
    ML_DATA_DIR    : str  — path to shared ML data volume (datasets)
"""

import json
import os
import threading
import time
from pathlib import Path
from queue import Queue

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torch.utils.data import DataLoader, Subset

# ── Environment ───────────────────────────────────────────────────────────────
CLIENT_ID      = int(os.environ.get("CLIENT_ID", "1"))
N_CLIENTS      = int(os.environ.get("N_CLIENTS", "2"))
SERVER_URL     = os.environ.get("FL_SERVER_URL", "http://fl-server:8100")
FL_WEIGHTS_DIR = Path(os.environ.get("FL_WEIGHTS_DIR", "/fl-weights"))
ML_DATA_DIR    = Path(os.environ.get("ML_DATA_DIR", "/ml-data"))
DATASETS_DIR   = ML_DATA_DIR / "datasets"
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
    """Download and return the full training dataset (torchvision)."""
    import torchvision
    import torchvision.transforms as T

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.MNIST(str(DATASETS_DIR), train=True, download=True, transform=transform)
    elif dataset == "fashion":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.FashionMNIST(str(DATASETS_DIR), train=True, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = T.Compose([T.ToTensor()])
        return torchvision.datasets.CIFAR10(str(DATASETS_DIR), train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _wait_for_server(max_wait: int = 120) -> bool:
    """Poll fl-server /health until it responds. Returns True if online."""
    _emit(f"Waiting for fl-server at {SERVER_URL}...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.ok:
                _emit("Connected to fl-server ✓")
                return True
        except Exception:
            pass
        time.sleep(3)
    _emit("WARNING: could not reach fl-server — will keep retrying during training")
    return False


def _background_training_loop() -> None:
    """
    Main client loop (runs in a daemon thread from startup).

    State machine:
      IDLE         — server not yet running or training not started
      TRAINING     — actively fetching rounds, training, submitting
      DONE         — server reports 'done', loop exits
    """
    import sys
    sys.path.insert(0, "/app")
    from core.model_utils import get_model
    from core.image_utils import DATASET_META

    # Stagger startup to avoid simultaneous dataset downloads
    startup_delay = (CLIENT_ID - 1) * 5
    _emit(f"Client {CLIENT_ID}/{N_CLIENTS} waiting {startup_delay}s before polling (staggered start)...")
    time.sleep(startup_delay)

    _wait_for_server()

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_rounds: set = set()
    current_ds_name: str | None = None
    shard = None

    _emit(f"Device: {device} | Polling fl-server for rounds...")

    while True:
        # Poll server round status
        try:
            status = requests.get(f"{SERVER_URL}/round/status", timeout=5).json()
        except Exception as exc:
            _emit(f"Cannot reach server: {exc} — retrying in 3s")
            time.sleep(3)
            continue

        state = status.get("state", "idle")
        rnd   = status.get("round", 0)

        if state in ("done", "stopped", "error"):
            _emit(f"Server reports state='{state}' — training complete. Loop finished.")
            break

        if state in ("idle", "starting") or rnd == 0:
            time.sleep(2)
            continue

        if rnd in trained_rounds:
            time.sleep(2)
            continue

        if not status.get("weights_ready"):
            _emit(f"[round {rnd}] Waiting for global weights file...")
            time.sleep(1)
            continue

        # ── Read training config ───────────────────────────────────────────
        config_path = FL_WEIGHTS_DIR / "training_config.json"
        if not config_path.exists():
            _emit("Waiting for training_config.json from server...")
            time.sleep(2)
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _emit(f"Cannot read training config: {exc}")
            time.sleep(2)
            continue

        dataset    = config["dataset"]
        model_type = config["model"]
        epochs     = config["epochs"]

        # ── Load dataset shard (once per dataset) ─────────────────────────
        if dataset != current_ds_name:
            _emit(f"Loading dataset '{dataset}'...")
            try:
                full_ds = _load_dataset(dataset)
            except Exception as exc:
                _emit(f"Dataset load failed: {exc}")
                time.sleep(5)
                continue

            total     = len(full_ds)
            shard_sz  = total // N_CLIENTS
            start_idx = (CLIENT_ID - 1) * shard_sz
            end_idx   = start_idx + shard_sz if CLIENT_ID < N_CLIENTS else total
            shard     = Subset(full_ds, list(range(start_idx, end_idx)))
            current_ds_name = dataset
            _emit(f"Shard ready: indices [{start_idx}, {end_idx}) → {len(shard)} samples")

        # ── Load global weights for this round ────────────────────────────
        gpath = _global_weights_path(rnd)
        if not gpath.exists():
            _emit(f"[round {rnd}] Global weights file not found: {gpath} — waiting...")
            time.sleep(2)
            continue

        meta      = DATASET_META.get(dataset, DATASET_META["mnist"])
        channels  = meta["channels"]
        img_size  = meta["height"]
        pixels_pp = channels * meta["height"] * meta["width"]

        local_model = get_model(model_type, latent_dim=32, input_channels=channels, image_size=img_size)
        local_model.load_state_dict(torch.load(gpath, map_location="cpu", weights_only=True))
        local_model = local_model.to(device)
        _emit(f"[round {rnd}] W_global loaded ✓ | {len(shard)} samples | {epochs} epoch(s)")

        # ── Local training ─────────────────────────────────────────────────
        loader    = DataLoader(shard, batch_size=128, shuffle=True, num_workers=0)
        optimizer = optim.Adam(local_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        kl_weight = 0.005
        local_model.train()

        last_avg = 0.0
        for ep in range(epochs):
            run = cnt = 0
            for bi, (data, _) in enumerate(loader):
                data = data.to(device)
                optimizer.zero_grad()
                if model_type == "cnn_vae":
                    recon, mu, logvar = local_model(data)
                    rloss = criterion(recon, data)
                    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kld  /= data.size(0) * pixels_pp
                    loss  = rloss + kl_weight * kld
                else:
                    recon = local_model(data)
                    loss  = criterion(recon, data)
                loss.backward()
                optimizer.step()
                run += loss.item()
                cnt += 1
                if bi % 60 == 0:
                    _emit(f"[round {rnd} | ep {ep+1}/{epochs}] batch {bi}/{len(loader)} loss={loss.item():.5f}")
            last_avg = run / max(cnt, 1)
            _emit(f"[round {rnd} | ep {ep+1}/{epochs}] ✓ avg_loss={last_avg:.5f}")

        # ── Save local weights to shared volume ────────────────────────────
        cpath = _client_weights_path(rnd)
        torch.save(local_model.state_dict(), cpath)
        _emit(f"[round {rnd}] Local weights saved → {cpath}")

        # ── Submit to fl-server ────────────────────────────────────────────
        tries = 0
        while tries < 5:
            try:
                r = requests.post(
                    f"{SERVER_URL}/round/submit/{CLIENT_ID}",
                    json={"loss": last_avg, "client_id": CLIENT_ID},
                    timeout=15,
                )
                if r.ok:
                    _emit(f"[round {rnd}] ✓ Submission accepted by server")
                    break
                else:
                    _emit(f"[round {rnd}] Server rejected: {r.status_code} {r.text}")
            except Exception as exc:
                _emit(f"[round {rnd}] Submit error: {exc}")
            tries += 1
            time.sleep(3)

        trained_rounds.add(rnd)
        _emit(f"[round {rnd}] Done. Waiting for next round...")

    _emit("Training loop exited.")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    t = threading.Thread(target=_background_training_loop, daemon=True)
    t.start()


# ── API ───────────────────────────────────────────────────────────────────────

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
