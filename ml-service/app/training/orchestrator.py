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
        rounds: int = 5,
        epochs: int = 5,
    ) -> dict:
        """Start a real federated training run (containers)."""
        with self._state_lock:
            if self._running:
                return {
                    "status": "already_running",
                    "dataset": dataset,
                    "model": model,
                    "distribution": distribution,
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
                "distribution": distribution,
            }

        thread = threading.Thread(
            target=self._run_real_training,
            args=(dataset, model, clients, epochs, awgn or {}, rounds),
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
            "rounds": rounds,
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

    def _new_experiment_dir(self) -> tuple[str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"experimento_{timestamp}"
        experiment_dir = RESULTADOS_ROOT / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["config", "logs", "metrics", "figures", "tables", "modelos"]:
            (experiment_dir / sub).mkdir(parents=True, exist_ok=True)
        return experiment_id, experiment_dir

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        header = list(rows[0].keys())
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join(str(row.get(col, "")) for col in header))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_tex_table(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        header = list(rows[0].keys())
        lines = [
            "\\begin{tabular}{" + "l" * len(header) + "}",
            "\\hline",
            " & ".join(header) + " \\\\",
            "\\hline",
        ]
        for row in rows:
            lines.append(" & ".join(str(row.get(col, "")) for col in header) + " \\\\")
        lines.extend(["\\hline", "\\end{tabular}"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _save_figures(self, experiment_dir: Path, history: list[dict], dataset: str) -> None:
        """
        Save training convergence charts and, if pre-trained weights are
        available, a real reconstruction comparison figure.

        Args:
            experiment_dir: Root directory for this experiment's outputs.
            history:        List of per-round metric dicts ({epoch, loss, accuracy}).
            dataset:        Dataset name (used to load weights and data).
        """
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        accs = [h["accuracy"] for h in history]

        # ── Convergence: Loss ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, losses, color="#ffd166", linewidth=2)
        ax.set_title("Convergencia da Loss (FedAvg Real)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "convergencia_loss.png", dpi=140)
        plt.close(fig)

        # ── Convergence: Accuracy ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, accs, color="#00f6a2", linewidth=2)
        ax.set_title("Convergencia da Acuracia (FedAvg Real)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Acurácia")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "convergencia_accuracy.png", dpi=140)
        plt.close(fig)

        # ── Reconstruction comparison ──────────────────────────────────────
        # Try to load a pre-trained model and produce a real encode→decode
        # comparison.  Fall back to a clearly labelled notice when weights
        # are not yet available.
        weights_path = f"app/core/{dataset}_cnn_vae.pth"
        reconstruction_saved = False

        if os.path.exists(weights_path):
            try:
                from app.core.model_utils import get_model
                from app.core.image_utils import load_dataset

                channels = 3 if dataset == "cifar10" else 1
                img_size = 32 if dataset == "cifar10" else 28
                model = get_model("cnn_vae", input_channels=channels, image_size=img_size)
                model.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=True)
                )
                model.eval()

                torch.manual_seed(42)
                test_ds = load_dataset(dataset, train=False)
                # Sample 4 images for the comparison grid
                indices = torch.randperm(len(test_ds))[:4]
                originals, reconstructions = [], []
                with torch.no_grad():
                    for idx in indices:
                        img, _ = test_ds[int(idx)]
                        img = img.unsqueeze(0)
                        mu, _ = model.encode(img)
                        recon = model.decode(mu)
                        originals.append(img.squeeze().cpu().numpy())
                        reconstructions.append(recon.squeeze().cpu().numpy())

                n = len(originals)
                fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
                fig.suptitle(f"Reconstrução Semântica — {dataset.upper()} (CNN-VAE)", fontsize=12)
                for i in range(n):
                    cmap = None if channels == 3 else "gray"
                    if channels == 3:
                        orig_img = np.transpose(originals[i], (1, 2, 0)).clip(0, 1)
                        recon_img = np.transpose(reconstructions[i], (1, 2, 0)).clip(0, 1)
                    else:
                        orig_img = originals[i].clip(0, 1)
                        recon_img = reconstructions[i].clip(0, 1)
                    axes[0, i].imshow(orig_img, cmap=cmap)
                    axes[0, i].set_title("Original", fontsize=8)
                    axes[0, i].axis("off")
                    axes[1, i].imshow(recon_img, cmap=cmap)
                    axes[1, i].set_title("Reconstruída", fontsize=8)
                    axes[1, i].axis("off")
                fig.tight_layout()
                fig.savefig(
                    experiment_dir / "figures" / "reconstrucao_amostras.png", dpi=140
                )
                plt.close(fig)
                reconstruction_saved = True

            except Exception:  # noqa: BLE001
                pass  # Fall through to placeholder

        if not reconstruction_saved:
            # Placeholder: inform the viewer that real weights are needed
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(
                0.5, 0.5,
                "Pesos do modelo não encontrados.\n"
                "Execute: python -m app.train_local\n"
                "para gerar reconstruções reais.",
                ha="center", va="center", fontsize=11,
                color="#ffd166", transform=ax.transAxes,
                wrap=True,
            )
            ax.set_facecolor("#070d14")
            fig.patch.set_facecolor("#070d14")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                experiment_dir / "figures" / "reconstrucao_amostras.png", dpi=140
            )
            plt.close(fig)

    def _snapshot_logs(self, experiment_dir: Path, clients: int) -> None:
        targets = ["server"] + [f"client-{i}" for i in range(1, clients + 1)]
        for target in targets:
            src = LOGS_DIR / f"training_{target}.log"
            dst = experiment_dir / "logs" / f"{target}.log"
            if src.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def _list_experiments(self) -> list[dict]:
        items = []
        for exp_dir in sorted(RESULTADOS_ROOT.glob("experimento_*"), reverse=True):
            if not exp_dir.is_dir():
                continue
            summary_file = exp_dir / "metrics" / "final_summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": exp_dir.name,
                    "dataset": summary.get("dataset"),
                    "model": summary.get("model"),
                    "distribution": summary.get("distribution", "iid"),
                    "final_loss": summary.get("final_loss"),
                    "final_accuracy": summary.get("final_accuracy"),
                    "timestamp": summary.get("timestamp"),
                }
            )
        return items

    def list_experiments(self) -> list[dict]:
        return self._list_experiments()

    def latest_experiment(self) -> dict | None:
        experiments = self._list_experiments()
        if not experiments:
            return None
        return self.get_experiment(experiments[0]["id"])

    def get_experiment(self, experiment_id: str) -> dict | None:
        exp_dir = RESULTADOS_ROOT / experiment_id
        summary_file = exp_dir / "metrics" / "final_summary.json"
        history_file = exp_dir / "metrics" / "round_metrics.csv"
        if not summary_file.exists() or not history_file.exists():
            return None

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        lines = [line for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        header = lines[0].split(",")
        history = []
        for line in lines[1:]:
            cols = line.split(",")
            row = {header[i]: cols[i] for i in range(min(len(header), len(cols)))}
            history.append(
                {
                    "epoch": int(row.get("epoch", 0)),
                    "loss": float(row.get("loss", 0.0)),
                    "accuracy": float(row.get("accuracy", 0.0)),
                }
            )

        summary["history"] = history
        summary["awgn"] = summary.get("awgn", {"enabled": False, "snr_db": None})
        summary["figures"] = {
            "loss": f"/results/artifact/{experiment_id}/figures/convergencia_loss.png",
            "accuracy": f"/results/artifact/{experiment_id}/figures/convergencia_accuracy.png",
            "reconstruction": f"/results/artifact/{experiment_id}/figures/reconstrucao_amostras.png",
        }
        summary["tables"] = {
            "csv": f"/results/artifact/{experiment_id}/tables/resultados.csv",
            "tex": f"/results/artifact/{experiment_id}/tables/resultados.tex",
        }
        return summary

    def artifact_path(self, experiment_id: str, relative_path: str) -> Path | None:
        base = (RESULTADOS_ROOT / experiment_id).resolve()
        candidate = (base / relative_path).resolve()
        if not str(candidate).startswith(str(base)):
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _run_real_training(self, dataset: str, model: str, clients: int, epochs: int, awgn: dict, rounds: int) -> None:
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
            experiment_id, experiment_dir = self._new_experiment_dir()
            self._latest_experiment_id   = experiment_id
            awgn_enabled = bool(awgn.get("enabled", False))
            awgn_snr = awgn.get("snr_db")
            if awgn_enabled and awgn_snr is None:
                awgn_snr = 10.0

            self._write_json(
                experiment_dir / "config" / "input_config.json",
                {
                    "experiment_id": experiment_id,
                    "dataset":      dataset,
                    "model":        model,
                    "mode":         "real_fedavg_containers",
                    "clients":      clients,
                    "rounds":       NUM_ROUNDS,
                    "epochs_per_round": epochs,
                    "awgn":         {"enabled": awgn_enabled, "snr_db": awgn_snr},
                },
            )

            self._emit("server", "=================================================")
            self._emit("server", "  FEDERATED LEARNING — REAL MODE (containers)")
            self._emit("server", "=================================================")
            self._emit("server", f"[init] dataset={dataset} | model={model} | clients={clients}")
            self._emit("server", f"[init] rounds={NUM_ROUNDS} | epochs/round={epochs}")
            self._emit("server", f"[init] fl-server: {FL_SERVER}")
            self._emit("server", f"[exp] id={experiment_id}")

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
                          "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr}},
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

            # ── Persist experiment artifacts ──────────────────────────────
            metrics = {
                "experiment_id": experiment_id,
                "dataset":      dataset,
                "model":        model,
                "mode":         "real_fedavg_containers",
                "clients":      clients,
                "rounds":       NUM_ROUNDS,
                "epochs_per_round": epochs,
                "awgn":         {"enabled": awgn_enabled, "snr_db": awgn_snr},
                "final_loss":   round(global_loss, 6),
                "final_accuracy": round(max(0.01, min(0.99, 1.0 - global_loss)), 4),
                "timestamp":    int(time.time()),
            }
            latest_file = RUNS_DIR / "latest_metrics.json"
            latest_file.write_text(json.dumps({**metrics, "history": history}, indent=2), encoding="utf-8")
            self._write_json(experiment_dir / "metrics" / "final_summary.json", metrics)
            if history:
                self._write_csv(experiment_dir / "metrics" / "round_metrics.csv", history)
                self._write_csv(experiment_dir / "tables" / "resultados.csv", history)
                self._write_tex_table(experiment_dir / "tables" / "resultados.tex", history)
                self._save_figures(experiment_dir, history, dataset)
            self._snapshot_logs(experiment_dir, clients)
            self._emit("server", f"[done] experimento salvo em {experiment_dir}")
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
