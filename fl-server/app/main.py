"""
fl_server/app/main.py
---------------------
Federated Learning Server — coordinates FedAvg training rounds across
multiple fl-client containers.

Weight exchange:  shared Docker volume (/fl-weights/)
Coordination:     HTTP REST API
Log streaming:    Server-Sent Events + polling

Protocol each round:
  1. Server writes /fl-weights/global_round_{N}.pth
  2. Server sets state="round_active"
  3. Each fl-client polls GET /round/status, sees the new round
  4. Client reads weights file, trains locally, writes
       /fl-weights/client_{id}_round_{N}.pth
  5. Client calls POST /round/submit/{client_id} {loss: float}
  6. Server waits for all clients to submit
  7. Server does FedAvg, writes /fl-weights/global_round_{N+1}.pth
  8. Repeat until total_rounds done
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from queue import Queue

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
FL_WEIGHTS_DIR = Path(os.environ.get("FL_WEIGHTS_DIR", "/fl-weights"))
ML_DATA_DIR    = Path(os.environ.get("ML_DATA_DIR", "/ml-data"))
WEIGHTS_DIR    = ML_DATA_DIR / "weights"
FL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FL Server", version="1.0.0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ── Global session state (single training session at a time) ──────────────────
_lock     = threading.Lock()
_all_logs: list[str] = []


def _reset_session() -> dict:
    return {
        "state":             "idle",     # idle|starting|round_active|aggregating|done|error|stopped
        "config":            {},
        "current_round":     0,
        "total_rounds":      5,
        "expected_clients":  2,
        "submitted_clients": set(),
        "client_losses":     {},         # {client_id: float}
        "history":           [],
        "final_loss":        None,
        "error":             None,
        "stop_flag":         False,
    }


_session: dict = _reset_session()


def _emit(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _all_logs.append(line)
    logging.info(msg)


def _global_weights_path(rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"global_round_{rnd}.pth"


def _client_weights_path(client_id: int, rnd: int) -> Path:
    return FL_WEIGHTS_DIR / f"client_{client_id}_round_{rnd}.pth"


def _fedavg(paths: list[Path]) -> dict:
    """Load state_dicts from files and compute parameter-wise mean."""
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    avg: dict = {}
    for key in states[0].keys():
        avg[key] = torch.stack([s[key].float() for s in states]).mean(dim=0)
    return avg


def _cleanup_old_weights() -> None:
    """Remove weight files from previous training sessions."""
    for f in FL_WEIGHTS_DIR.glob("*.pth"):
        try:
            f.unlink()
        except Exception:
            pass


# ── Training background thread ────────────────────────────────────────────────

def _training_thread() -> None:
    """
    Main FedAvg loop running in a daemon thread.
    Uses the shared volume for weight exchange and the _session dict for
    coordination with client poll endpoints.
    """
    import sys
    sys.path.insert(0, "/app")
    from core.model_utils import get_model
    from core.image_utils import DATASET_META

    config     = _session["config"]
    dataset    = config["dataset"]
    model_type = config["model"]
    clients    = config["clients"]
    epochs     = config["epochs"]
    num_rounds = config["rounds"]

    meta      = DATASET_META.get(dataset, DATASET_META["mnist"])
    channels  = meta["channels"]
    img_size  = meta["height"]

    # Load or initialize global model
    saved_path = WEIGHTS_DIR / f"{dataset}_{model_type}.pth"
    global_model = get_model(model_type, latent_dim=32, input_channels=channels, image_size=img_size)
    if saved_path.exists():
        global_model.load_state_dict(torch.load(saved_path, map_location="cpu", weights_only=True))
        _emit(f"[server] pre-trained weights loaded from {saved_path}")
    else:
        _emit("[server] starting with random initialization")

    # Write initial weights for clients to fetch at round 1
    torch.save(global_model.state_dict(), _global_weights_path(1))
    _emit(f"[server] W_global_round_1 written to {_global_weights_path(1)}")

    _emit(f"[server] ══ FedAvg started: dataset={dataset} model={model_type} clients={clients} "
          f"rounds={num_rounds} epochs/round={epochs} ══")

    global_loss = 9.999
    history: list[dict] = []

    with _lock:
        _session["state"]            = "round_active"
        _session["total_rounds"]     = num_rounds
        _session["expected_clients"] = clients
        _session["current_round"]    = 1
        _session["submitted_clients"] = set()
        _session["client_losses"]     = {}

    for rnd in range(1, num_rounds + 1):
        with _lock:
            if _session["stop_flag"]:
                _emit("[server] stopped by user")
                _session["state"] = "stopped"
                return

        _emit(f"[server] ── ROUND {rnd}/{num_rounds} ── waiting for {clients} clients")

        # Wait for all clients to submit their local weights
        timeout  = epochs * 600  # 10 min per epoch, generous
        deadline = time.time() + timeout
        while time.time() < deadline:
            with _lock:
                n_sub = len(_session["submitted_clients"])
                stop  = _session["stop_flag"]
            if stop or n_sub >= clients:
                break
            time.sleep(0.5)

        with _lock:
            if _session["stop_flag"]:
                _emit("[server] stopped during round wait")
                _session["state"] = "stopped"
                return
            submitted = set(_session["submitted_clients"])
            losses    = dict(_session["client_losses"])

        n = len(submitted)
        _emit(f"[server] received {n}/{clients} submissions for round {rnd}")

        if n == 0:
            _emit("[server] ERROR: no submissions — aborting")
            with _lock:
                _session["state"] = "error"
                _session["error"] = "No client submissions received"
            return

        # FedAvg aggregation
        with _lock:
            _session["state"] = "aggregating"

        _emit(f"[server] FedAvg: averaging {n} client models...")
        client_paths = [_client_weights_path(cid, rnd) for cid in submitted if _client_weights_path(cid, rnd).exists()]
        if not client_paths:
            _emit("[server] ERROR: weight files not found on shared volume")
            with _lock:
                _session["state"] = "error"
            return

        avg_state = _fedavg(client_paths)
        global_model.load_state_dict(avg_state)
        global_loss = sum(losses.values()) / len(losses)

        _emit(f"[server] round {rnd} complete | global_loss={global_loss:.5f}")
        history.append({
            "epoch":    rnd,
            "loss":     round(global_loss, 6),
            "accuracy": round(max(0.01, min(0.99, 1.0 - global_loss)), 4),
        })

        # Write global weights for next round (or final)
        next_rnd = rnd + 1
        torch.save(global_model.state_dict(), _global_weights_path(next_rnd))
        _emit(f"[server] W_global_round_{next_rnd} written for next round")

        with _lock:
            _session["current_round"]    = next_rnd
            _session["submitted_clients"] = set()
            _session["client_losses"]     = {}
            _session["state"]            = "round_active" if rnd < num_rounds else "aggregating"

    # Save final weights to shared volume
    torch.save(global_model.state_dict(), saved_path)
    _emit(f"[server] ✓ FedAvg complete! final_loss={global_loss:.5f}")
    _emit(f"[server] Weights saved → {saved_path}")
    _emit(f"[server] Semantic and benchmark endpoints will now use these weights")

    with _lock:
        _session["state"]      = "done"
        _session["final_loss"] = round(global_loss, 6)
        _session["history"]    = history


# ── Pydantic models ───────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    dataset: str = "fashion"
    model:   str = "cnn_vae"
    clients: int = 2
    epochs:  int = 3
    rounds:  int = 5


class SubmitRequest(BaseModel):
    loss:      float
    client_id: int


# ── REST endpoints ────────────────────────────────────────────────────────────

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
        _session["state"]  = "starting"
        _session["config"] = req.model_dump()

    _all_logs.clear()
    _cleanup_old_weights()

    # Write training config so clients can read dataset/model info
    config_path = FL_WEIGHTS_DIR / "training_config.json"
    config_path.write_text(json.dumps(req.model_dump(), indent=2), encoding="utf-8")

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
        return {
            "state":             _session["state"],
            "current_round":     _session["current_round"],
            "total_rounds":      _session["total_rounds"],
            "expected_clients":  _session["expected_clients"],
            "submitted_clients": list(_session["submitted_clients"]),
            "history":           list(_session["history"]),
            "final_loss":        _session["final_loss"],
            "error":             _session["error"],
        }


@app.get("/round/status")
def round_status():
    """Polled by fl-clients to know what round is active and whether to train."""
    with _lock:
        rnd   = _session["current_round"]
        state = _session["state"]
        subs  = list(_session["submitted_clients"])
        total = _session["total_rounds"]
        exp   = _session["expected_clients"]

    gpath = _global_weights_path(rnd)
    return {
        "round":            rnd,
        "state":            state,
        "total_rounds":     total,
        "expected_clients": exp,
        "weights_path":     str(gpath),
        "weights_ready":    gpath.exists(),
        "submitted":        subs,
    }


@app.post("/round/submit/{client_id}")
def round_submit(client_id: int, req: SubmitRequest):
    """
    Called by a client after it has written its local weights file.
    Server verifies the file exists before accepting the submission.
    """
    rnd   = _session["current_round"]
    cpath = _client_weights_path(client_id, rnd)
    if not cpath.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Weight file not found: {cpath}. Client must write weights before submitting."
        )
    with _lock:
        _session["submitted_clients"].add(client_id)
        _session["client_losses"][client_id] = req.loss
    _emit(f"[server] ← client-{client_id} submission accepted | round={rnd} loss={req.loss:.5f}")
    return {"status": "received", "round": rnd}


@app.get("/logs")
def get_logs(since: int = 0):
    """Return log lines since the given offset (for polling)."""
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
