"""
fl_server/app/main.py
---------------------
Federated Learning Server with support for:
- baseline and compressed transport modes
- communication byte accounting per round
- atomic writes for transport files and config
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

FL_WEIGHTS_DIR = Path(os.environ.get("FL_WEIGHTS_DIR", "/fl-weights"))
ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "/ml-data"))
WEIGHTS_DIR = ML_DATA_DIR / "weights"
FL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FL Server", version="2.0.0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

_lock = threading.Lock()
_all_logs: list[str] = []


def _emit(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _all_logs.append(line)
    logging.info(msg)


def _global_weights_path(rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"global_round_{rnd}.pth"


def _client_weights_path(client_id: int, rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"client_{client_id}_round_{rnd}.pth"


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


def _load_state_for_transport(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj.get("__compressed__"):
        return _decompress_state_dict(obj)
    return obj


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_json_write(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_state_for_transport(state: dict[str, torch.Tensor], path: Path, mode: str, bits: int) -> int:
    if mode == "compressed":
        _atomic_torch_save(_compress_state_dict(state, bits), path)
    else:
        _atomic_torch_save(state, path)
    return int(path.stat().st_size)


def _fedavg(paths: list[Path]) -> dict[str, torch.Tensor]:
    states = [_load_state_for_transport(p) for p in paths]
    avg: dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        avg[key] = torch.stack([s[key].float() for s in states]).mean(dim=0)
    return avg


def _cleanup_old_weights() -> None:
    for f in FL_WEIGHTS_DIR.glob("*.pth"):
        try:
            f.unlink()
        except Exception:
            pass


def _reset_session() -> dict:
    return {
        "state": "idle",
        "config": {},
        "current_round": 0,
        "total_rounds": 5,
        "expected_clients": 2,
        "submitted_clients": set(),
        "client_losses": {},
        "client_bytes": {},
        "history": [],
        "final_loss": None,
        "error": None,
        "stop_flag": False,
    }


_session: dict = _reset_session()


class AWGNConfig(BaseModel):
    enabled: bool = False
    snr_db: float | None = None


class MaskingConfig(BaseModel):
    enabled: bool = False
    drop_rate: float = 0.25
    fill_value: float = 0.0


class StartRequest(BaseModel):
    dataset: str = "fashion"
    model: str = "cnn_vae"
    clients: int = 2
    epochs: int = 3
    rounds: int = 5
    awgn: AWGNConfig = AWGNConfig()
    masking: MaskingConfig = MaskingConfig()
    base_weights: str | None = None
    compression_mode: str = "baseline"
    compression_bits: int = 8
    seed: int = 42


class SubmitRequest(BaseModel):
    loss: float
    client_id: int
    bytes_transmitted: int | None = None


def _training_thread() -> None:
    import sys

    sys.path.insert(0, "/app")
    from core.image_utils import DATASET_META
    from core.model_utils import get_model

    config = _session["config"]
    dataset = config["dataset"]
    model_type = config["model"]
    clients = int(config["clients"])
    epochs = int(config["epochs"])
    num_rounds = int(config["rounds"])
    compression_mode = str(config.get("compression_mode", "baseline"))
    compression_bits = int(config.get("compression_bits", 8))
    seed = int(config.get("seed", 42))

    torch.manual_seed(seed)

    meta = DATASET_META.get(dataset, DATASET_META["mnist"])
    channels = meta["channels"]
    img_size = meta["height"]

    saved_path = WEIGHTS_DIR / f"{dataset}_{model_type}.pth"
    base_weights = config.get("base_weights")
    global_model = get_model(model_type, latent_dim=32, input_channels=channels, image_size=img_size)

    selected_path = None
    if base_weights and base_weights not in ("random", "none"):
        if base_weights == "latest":
            selected_path = saved_path
        else:
            safe_name = Path(base_weights).name
            candidate = WEIGHTS_DIR / safe_name
            archive_candidate = WEIGHTS_DIR / "archive" / safe_name
            if candidate.exists():
                selected_path = candidate
            elif archive_candidate.exists():
                selected_path = archive_candidate

    if selected_path and selected_path.exists():
        global_model.load_state_dict(torch.load(selected_path, map_location="cpu", weights_only=True))
        _emit(f"[server] base weights loaded from {selected_path}")
    elif base_weights in (None, "", "random", "none"):
        _emit("[server] starting with random initialization")
    elif saved_path.exists():
        global_model.load_state_dict(torch.load(saved_path, map_location="cpu", weights_only=True))
        _emit(f"[server] pre-trained weights loaded from {saved_path}")
    else:
        _emit("[server] starting with random initialization")

    first_global_bytes = _save_state_for_transport(
        global_model.state_dict(), _global_weights_path(1), compression_mode, compression_bits
    )
    _emit(f"[server] W_global_round_1 written | bytes={first_global_bytes}")

    _emit(
        f"[server] FedAvg started: dataset={dataset} model={model_type} clients={clients} rounds={num_rounds} "
        f"epochs={epochs} mode={compression_mode} bits={compression_bits}"
    )

    global_loss = 9.999
    history: list[dict] = []

    with _lock:
        _session["state"] = "round_active"
        _session["total_rounds"] = num_rounds
        _session["expected_clients"] = clients
        _session["current_round"] = 1
        _session["submitted_clients"] = set()
        _session["client_losses"] = {}
        _session["client_bytes"] = {}

    for rnd in range(1, num_rounds + 1):
        with _lock:
            if _session["stop_flag"]:
                _session["state"] = "stopped"
                _emit("[server] stopped by user")
                return

        deadline = time.time() + (epochs * 600)
        while time.time() < deadline:
            with _lock:
                n_sub = len(_session["submitted_clients"])
                stop = _session["stop_flag"]
            if stop or n_sub >= clients:
                break
            time.sleep(0.5)

        with _lock:
            if _session["stop_flag"]:
                _session["state"] = "stopped"
                _emit("[server] stopped during round wait")
                return
            submitted = set(_session["submitted_clients"])
            losses = dict(_session["client_losses"])
            round_client_bytes = dict(_session["client_bytes"])

        if not submitted:
            with _lock:
                _session["state"] = "error"
                _session["error"] = "No client submissions received"
            _emit("[server] ERROR: no submissions")
            return

        with _lock:
            _session["state"] = "aggregating"

        client_paths = [_client_weights_path(cid, rnd) for cid in submitted if _client_weights_path(cid, rnd).exists()]
        if not client_paths:
            with _lock:
                _session["state"] = "error"
                _session["error"] = "Weight files not found"
            _emit("[server] ERROR: client weight files missing")
            return

        avg_state = _fedavg(client_paths)
        global_model.load_state_dict(avg_state)
        global_loss = sum(losses.values()) / len(losses)

        next_rnd = rnd + 1
        global_bytes = _save_state_for_transport(
            global_model.state_dict(), _global_weights_path(next_rnd), compression_mode, compression_bits
        )

        clients_bytes_total = int(sum(round_client_bytes.values()))
        total_round_bytes = int(clients_bytes_total + global_bytes)

        history.append(
            {
                "epoch": rnd,
                "loss": round(global_loss, 6),
                "accuracy": round(max(0.01, min(0.99, 1.0 - global_loss)), 4),
                "bytes_clients": clients_bytes_total,
                "bytes_global": global_bytes,
                "bytes_total": total_round_bytes,
                "compression_mode": compression_mode,
                "compression_bits": compression_bits,
            }
        )

        _emit(
            f"[server] round {rnd} done | loss={global_loss:.5f} | bytes_clients={clients_bytes_total} "
            f"bytes_global={global_bytes} total={total_round_bytes}"
        )

        with _lock:
            _session["current_round"] = next_rnd
            _session["submitted_clients"] = set()
            _session["client_losses"] = {}
            _session["client_bytes"] = {}
            _session["state"] = "round_active" if rnd < num_rounds else "aggregating"

    torch.save(global_model.state_dict(), saved_path)
    _emit(f"[server] FedAvg complete! final_loss={global_loss:.5f}")

    try:
        archive_dir = WEIGHTS_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{dataset}_{model_type}_{timestamp}.pth"
        shutil.copy2(saved_path, archive_path)
    except Exception as exc:
        _emit(f"[server] Warning: snapshot save failed: {exc}")

    with _lock:
        _session["state"] = "done"
        _session["final_loss"] = round(global_loss, 6)
        _session["history"] = history


@app.get("/health")
def health():
    return {"status": "ok", "service": "fl-server"}


@app.post("/training/start")
def training_start(req: StartRequest):
    global _session
    with _lock:
        if _session["state"] not in ("idle", "done", "error", "stopped"):
            return {"status": "already_running", "state": _session["state"]}
        _session = _reset_session()
        _session["state"] = "starting"
        _session["config"] = req.model_dump()

    _all_logs.clear()
    _cleanup_old_weights()

    config_path = FL_WEIGHTS_DIR / "training_config.json"
    _atomic_json_write(req.model_dump(), config_path)
    _emit(f"[server] new training session: {req.model_dump()}")

    t = threading.Thread(target=_training_thread, daemon=True)
    t.start()
    return {"status": "started", "config": req.model_dump()}


@app.post("/training/stop")
def training_stop():
    with _lock:
        _session["stop_flag"] = True
    return {"status": "stop_requested"}


@app.get("/training/status")
def training_status():
    with _lock:
        hist = list(_session["history"])
        total_bytes = int(sum(item.get("bytes_total", 0) for item in hist))
        return {
            "state": _session["state"],
            "current_round": _session["current_round"],
            "total_rounds": _session["total_rounds"],
            "expected_clients": _session["expected_clients"],
            "submitted_clients": list(_session["submitted_clients"]),
            "history": hist,
            "final_loss": _session["final_loss"],
            "error": _session["error"],
            "communication_summary": {
                "total_bytes": total_bytes,
                "rounds_recorded": len(hist),
                "compression_mode": _session["config"].get("compression_mode", "baseline"),
                "compression_bits": _session["config"].get("compression_bits", 8),
            },
        }


@app.get("/round/status")
def round_status():
    with _lock:
        rnd = _session["current_round"]
        state = _session["state"]
        subs = list(_session["submitted_clients"])
        total = _session["total_rounds"]
        exp = _session["expected_clients"]

    gpath = _global_weights_path(rnd)
    return {
        "round": rnd,
        "state": state,
        "total_rounds": total,
        "expected_clients": exp,
        "weights_path": str(gpath),
        "weights_ready": gpath.exists(),
        "submitted": subs,
    }


@app.post("/round/submit/{client_id}")
def round_submit(client_id: int, req: SubmitRequest):
    rnd = _session["current_round"]
    cpath = _client_weights_path(client_id, rnd)
    if not cpath.exists():
        raise HTTPException(status_code=400, detail=f"Weight file not found: {cpath}")

    bytes_tx = req.bytes_transmitted if req.bytes_transmitted is not None else int(cpath.stat().st_size)

    with _lock:
        _session["submitted_clients"].add(client_id)
        _session["client_losses"][client_id] = req.loss
        _session["client_bytes"][client_id] = int(bytes_tx)

    _emit(f"[server] client-{client_id} submitted | round={rnd} loss={req.loss:.5f} bytes={bytes_tx}")
    return {"status": "received", "round": rnd}


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
