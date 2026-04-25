import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from app.core.config import LOGS_DIR, RUNS_DIR, RESULTADOS_ROOT


class TrainingOrchestrator:
    """
    Orchestrates real federated training via fl-server + fl-client containers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest_experiment_id: str | None = None
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._active_clients = 0
        self._real_training = False  # True when running actual PyTorch training

    def start(
        self,
        dataset: str,
        model: str,
        clients: int,
        awgn: dict | None = None,
        masking: dict | None = None,
        base_weights: str | None = None,
        rounds: int = 5,
        epochs: int = 5,
        compression_mode: str = "baseline",
        compression_bits: int = 8,
        seed: int = 42,
        latent_dim: int = 32,
    ) -> dict:
        """Start a real federated training run (containers)."""
        with self._state_lock:
            if self._running:
                return {
                    "status": "already_running",
                    "dataset": dataset,
                    "model": model,
                }

            self._running = True
            self._paused = False
            self._pause_event.set()
            self._stop_event.clear()
            self._active_clients = clients
            self._real_training = True

        if self._lock.locked():
            return {
                "status": "already_running",
                "dataset": dataset,
                "model": model,
            }

        thread = threading.Thread(
            target=self._run_real_training,
            args=(
                dataset,
                model,
                clients,
                epochs,
                awgn or {},
                masking or {},
                rounds,
                base_weights,
                compression_mode,
                compression_bits,
                seed,
                latent_dim,
            ),
            daemon=True,
        )
        thread.start()
        return {
            "status": "started",
            "mode": "real",
            "dataset": dataset,
            "model": model,
            "clients": clients,
            "epochs": epochs,
            "awgn": awgn or {"enabled": False, "snr_db": None},
            "masking": masking or {"enabled": False, "drop_rate": 0.25, "fill_value": 0.0},
            "rounds": rounds,
            "base_weights": base_weights,
            "compression_mode": compression_mode,
            "compression_bits": compression_bits,
            "seed": seed,
            "latent_dim": latent_dim,
        }

    def status(self) -> dict:
        with self._state_lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "active_clients": self._active_clients,
                "latest_experiment_id": self._latest_experiment_id,
                "real_training": self._real_training,
            }

    def pause(self) -> dict:
        with self._state_lock:
            if not self._running:
                return {"status": "not_running"}
            if self._paused:
                return {"status": "already_paused"}
            self._paused = True
            self._pause_event.clear()
        self._emit("server", "[pause] training paused")
        return {"status": "paused"}

    def resume(self) -> dict:
        with self._state_lock:
            if not self._running:
                return {"status": "not_running"}
            if not self._paused:
                return {"status": "not_paused"}
            self._paused = False
            self._pause_event.set()
        self._emit("server", "[resume] training resumed")
        return {"status": "resumed"}

    def stop(self) -> dict:
        with self._state_lock:
            if not self._running:
                return {"status": "not_running"}
            self._stop_event.set()
            # Unblock training loop if it is currently paused.
            self._pause_event.set()
            self._paused = False
        self._emit("server", "[stop] stop requested by user")
        return {"status": "stop_requested"}

    def clear_logs(self, clients: int | None = None) -> dict:
        with self._state_lock:
            max_clients = clients if clients is not None else self._active_clients

        targets = ["server"] + [f"client-{i}" for i in range(1, max(0, max_clients) + 1)]
        for target in list(set(targets)):
            log_file = LOGS_DIR / f"training_{target}.log"
            log_file.write_text("", encoding="utf-8")

        return {"status": "logs_cleared", "targets": targets}

    def _emit(self, target: str, message: str) -> None:
        log_file = LOGS_DIR / f"training_{target}.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")



    def _run_real_training(
        self,
        dataset: str,
        model: str,
        clients: int,
        epochs: int,
        awgn: dict,
        masking: dict,
        rounds: int,
        base_weights: str | None,
        compression_mode: str,
        compression_bits: int,
        seed: int,
        latent_dim: int = 32,
    ) -> None:
        """
        Container-based FedAvg: delegates training to dedicated fl-server + fl-client containers.

        Architecture:
          fl-server  (container) - coordinates rounds, does FedAvg aggregation
          fl-client-1 ... fl-client-N  (containers) - train locally in true parallel
          ml-service (this container) - proxies logs from all containers to the dashboard SSE

        Communication:
          ml-service  ->  fl-server: POST /training/start
          fl-server   ->  fl-clients: shared Docker volume /fl-weights/ (weight files)
          fl-clients  ->  fl-server:  POST /round/submit/{client_id}
          ml-service  polls /logs from each container and re-emits via self._emit()

        After training, the fl-server saves the aggregated model to the shared
        /ml-data/weights/ volume, which is also mounted by ml-service, so
        /semantic and /benchmark endpoints pick up the new weights automatically.
        """
        import requests as _req

        FL_SERVER = os.environ.get("FL_SERVER_URL", "http://fl-server:8100")
        NUM_ROUNDS = max(1, int(rounds))

        with self._lock:
            awgn_enabled = bool(awgn.get("enabled", False))
            awgn_snr = awgn.get("snr_db")
            if awgn_enabled and awgn_snr is None:
                awgn_snr = 10.0
            masking_enabled = bool(masking.get("enabled", False))
            masking_drop_rate = float(masking.get("drop_rate", 0.25))
            masking_fill_value = float(masking.get("fill_value", 0.0))

            self._emit("server", "=================================================")
            self._emit("server", "  FEDERATED LEARNING — REAL MODE (containers)")
            self._emit("server", "=================================================")
            self._emit("server", f"[init] dataset={dataset} | model={model} | clients={clients}")
            self._emit("server", f"[init] rounds={NUM_ROUNDS} | epochs/round={epochs}")
            self._emit("server", f"[init] mode={compression_mode} | bits={compression_bits} | seed={seed}")
            self._emit("server", f"[init] fl-server: {FL_SERVER}")


            # ── Connect to fl-server ──────────────────────────────────────
            try:
                r = _req.get(f"{FL_SERVER}/health", timeout=5)
                if not r.ok:
                    raise RuntimeError(f"fl-server health check failed: {r.status_code}")
                self._emit("server", "[ok] fl-server is reachable")
            except Exception as exc:
                self._emit("server", f"[error] Cannot reach fl-server: {exc}")
                self._emit("server", "[error] Make sure all containers are running:")
                self._emit("server", "[error]   docker compose up --build -d")
                with self._state_lock:
                    self._running = False
                    self._real_training = False
                return

            # ── Start training on fl-server ───────────────────────────────
            try:
                r = _req.post(
                    f"{FL_SERVER}/training/start",
                    json={"dataset": dataset, "model": model, "clients": clients,
                          "epochs": epochs, "rounds": NUM_ROUNDS,
                          "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
                          "masking": {"enabled": masking_enabled, "drop_rate": masking_drop_rate, "fill_value": masking_fill_value},
                          "base_weights": base_weights,
                          "compression_mode": compression_mode,
                          "compression_bits": compression_bits,
                          "seed": seed,
                          "latent_dim": latent_dim},
                    timeout=10,
                )
                if not r.ok:
                    raise RuntimeError(f"{r.status_code}: {r.text}")
                self._emit("server", "[ok] fl-server accepted training request")
            except Exception as exc:
                self._emit("server", f"[error] Failed to start fl-server training: {exc}")
                with self._state_lock:
                    self._running = False
                    self._real_training = False
                return

            # ── Poll logs from fl-server + each fl-client ─────────────────
            log_offsets   = {"server": 0}
            client_urls   = {}
            for i in range(1, clients + 1):
                log_offsets[f"client-{i}"] = 0
                client_urls[f"client-{i}"] = f"http://fl-client-{i}:8200"

            history     = []
            global_loss = 9.999

            while not self._stop_event.is_set():
                # ── Fetch fl-server status ────────────────────────────────
                try:
                    st = _req.get(f"{FL_SERVER}/training/status", timeout=5).json()
                    state = st.get("state", "idle")
                    if st.get("history"):
                        history = st["history"]
                        global_loss = history[-1]["loss"]
                except Exception:
                    state = "unknown"

                # ── Relay fl-server logs ──────────────────────────────────
                try:
                    logs_r = _req.get(
                        f"{FL_SERVER}/logs?since={log_offsets['server']}", timeout=5
                    ).json()
                    for line in logs_r.get("lines", []):
                        self._emit("server", line)
                    log_offsets["server"] = logs_r.get("total", log_offsets["server"])
                except Exception:
                    pass

                # ── Relay each client's logs ──────────────────────────────
                for i in range(1, clients + 1):
                    key = f"client-{i}"
                    try:
                        logs_r = _req.get(
                            f"{client_urls[key]}/logs?since={log_offsets[key]}", timeout=3
                        ).json()
                        for line in logs_r.get("lines", []):
                            self._emit(key, line)
                        log_offsets[key] = logs_r.get("total", log_offsets[key])
                    except Exception:
                        pass

                if state in ("done", "error", "stopped"):
                    if state == "error":
                        self._emit("server", f"[error] fl-server reported an error: {st.get('error')}")
                    break

                time.sleep(1.5)

            if self._stop_event.is_set():
                try:
                    _req.post(f"{FL_SERVER}/training/stop", timeout=5)
                except Exception:
                    pass
                self._emit("server", "[stopped] treinamento interrompido pelo usuario")

            self._emit("server", f"[done] FedAvg containers finalizado | loss_final={global_loss:.5f}")

            for i in range(1, clients + 1):
                self._emit(f"client-{i}", "[done] loop finalizado")

        with self._state_lock:
            self._running       = False
            self._paused        = False
            self._real_training = False
            self._stop_event.clear()
            self._pause_event.clear()

    def stream(self, target: str):
        log_file = LOGS_DIR / f"training_{target}.log"
        if not log_file.exists():
            log_file.write_text("", encoding="utf-8")
            
        with open(log_file, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    if not self._running:
                        time.sleep(1)
                        yield f"[heartbeat] waiting for new logs on {target}..."
                        continue
                    time.sleep(0.5)
                    continue
                yield line.strip()


orchestrator = TrainingOrchestrator()
