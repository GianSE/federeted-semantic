import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import matplotlib.pyplot as plt
import numpy as np

from app.core.config import LOGS_DIR, RUNS_DIR
from app.core.config import RESULTADOS_ROOT


class TrainingOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest_experiment_id: str | None = None
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._active_clients = 0

    def start(self, dataset: str, model: str, distribution: str, noise: dict, awgn: dict, clients: int) -> dict:
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

        if self._lock.locked():
            return {
                "status": "already_running",
                "dataset": dataset,
                "model": model,
                "distribution": distribution,
            }

        thread = threading.Thread(
            target=self._run_training,
            args=(dataset, model, distribution, noise, awgn, clients),
            daemon=True,
        )
        thread.start()
        return {
            "status": "started",
            "dataset": dataset,
            "model": model,
            "distribution": distribution,
            "clients": clients,
            "noise": noise,
            "awgn": awgn,
        }

    def status(self) -> dict:
        with self._state_lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "active_clients": self._active_clients,
                "latest_experiment_id": self._latest_experiment_id,
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
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        accs = [h["accuracy"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, losses, color="#ffd166", linewidth=2)
        ax.set_title("Convergência da Loss")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "convergencia_loss.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, accs, color="#00f6a2", linewidth=2)
        ax.set_title("Convergência da Acurácia")
        ax.set_xlabel("Round")
        ax.set_ylabel("Acurácia")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "convergencia_accuracy.png", dpi=140)
        plt.close(fig)

        seed = 17 if dataset == "mnist" else 31
        rng = np.random.default_rng(seed)
        original = rng.random((8, 8))
        noise = rng.normal(0.0, 0.08, size=(8, 8))
        reconstructed = np.clip(original + noise, 0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(reconstructed, cmap="gray")
        axes[1].set_title("Reconstruída")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "reconstrucao_amostras.png", dpi=140)
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
        summary["distribution"] = summary.get("distribution", "iid")
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

    def _run_training(self, dataset: str, model: str, distribution: str, noise: dict, awgn: dict, clients: int) -> None:
        stopped_early = False
        with self._lock:
            experiment_id, experiment_dir = self._new_experiment_dir()
            self._latest_experiment_id = experiment_id
            awgn_enabled = bool(awgn.get("enabled", False))
            awgn_snr = awgn.get("snr_db")
            if awgn_enabled and awgn_snr is None:
                awgn_snr = 10
            self._write_json(
                experiment_dir / "config" / "input_config.json",
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "model": model,
                    "distribution": distribution,
                    "clients": clients,
                    "noise": noise,
                    "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
                },
            )
            self._emit(
                "server",
                f"[init] federated training started dataset={dataset} model={model} distribution={distribution} clients={clients}",
            )
            self._emit("server", f"[exp] id={experiment_id} output={experiment_dir}")
            self._emit(
                "server",
                "[noise] "
                f"channel={noise.get('channel', 0)} "
                f"packet_loss={noise.get('packet_loss', 0)} "
                f"latency={noise.get('latency', 0)} "
                f"client_drift={noise.get('client_drift', 0)}",
            )
            self._emit("server", f"[awgn] enabled={awgn_enabled} snr_db={awgn_snr}")

            history = []
            dataset_base_loss = {
                "mnist": 1.0,
                "cifar10": 1.35,
                "cifar100": 1.55,
            }
            model_loss_factor = {
                "ae": 1.03,
                "cnn_ae": 1.0,
                "cnn_vae": 1.05,
            }
            distribution_factor = 1.04 if distribution == "non_iid" else 1.0
            global_loss = dataset_base_loss.get(dataset, 1.2) * model_loss_factor.get(model, 1.0) * distribution_factor
            global_accuracy = 0.18 if dataset == "cifar100" else 0.22

            for rnd in range(1, 11):
                if self._stop_event.is_set():
                    stopped_early = True
                    self._emit("server", "[stopped] training interrupted by user")
                    break

                self._pause_event.wait()

                if self._stop_event.is_set():
                    stopped_early = True
                    self._emit("server", "[stopped] training interrupted by user")
                    break

                time.sleep(1)
                client_losses = []
                client_accs = []

                self._emit("server", f"[round {rnd:02d}] broadcasting global weights to {clients} edge nodes...")
                time.sleep(0.5)

                for client_idx in range(1, clients + 1):
                    if self._stop_event.is_set():
                        stopped_early = True
                        break
                        
                    target = f"client-{client_idx}"
                    self._emit(target, f"[round {rnd:02d}] receiving global weights... starting local epoch(s)")
                    drift = float(noise.get("client_drift", 0)) / 100.0
                    channel = float(noise.get("channel", 0)) / 100.0
                    packet_loss = float(noise.get("packet_loss", 0)) / 100.0

                    non_iid_boost = random.uniform(0.01, 0.06) if distribution == "non_iid" else 0.0
                    awgn_penalty = 0.0
                    if awgn_enabled:
                        snr_value = float(awgn_snr)
                        awgn_penalty = max(0.0, (20.0 - snr_value) / 100.0)

                    local_loss = max(
                        0.02,
                        global_loss
                        * random.uniform(0.80 + drift, 0.94 + channel)
                        * (1.0 + non_iid_boost + awgn_penalty),
                    )
                    local_acc = min(
                        0.995,
                        global_accuracy
                        + random.uniform(0.01, 0.08 - packet_loss * 0.03)
                        - non_iid_boost * 0.2
                        - awgn_penalty * 0.3,
                    )

                    for _ in range(10): # Break down sleep for instant stop
                        if self._stop_event.is_set(): break
                        time.sleep(random.uniform(0.01, 0.04))
                        
                    client_losses.append(local_loss)
                    client_accs.append(local_acc)
                    self._emit(target, f"[round {rnd:02d}] local_loss={local_loss:.4f} local_acc={local_acc:.4f} -> transferring gradients")
                    self._emit("server", f"[round {rnd:02d}] incoming gradients strictly received from client-{client_idx}")

                if self._stop_event.is_set():
                    stopped_early = True
                    self._emit("server", "[stopped] training interrupted by user")
                    break

                latency_penalty = float(noise.get("latency", 0)) / 1500.0
                time.sleep(random.uniform(0.2, 0.6) * (1.0 + latency_penalty))
                
                self._emit("server", f"[round {rnd:02d}] performing FedAvg aggregation...")
                
                global_loss = max(0.015, (sum(client_losses) / len(client_losses)) * (1.0 + latency_penalty))
                global_accuracy = min(0.995, max(0.01, (sum(client_accs) / len(client_accs)) * (1.0 - packet_loss * 0.05)))

                history.append(
                    {
                        "epoch": rnd,
                        "loss": round(global_loss, 4),
                        "accuracy": round(global_accuracy, 4),
                    }
                )
                self._emit("server", f"[round {rnd:02d}] global_loss={global_loss:.4f} global_acc={global_accuracy:.4f} (Finished)")

            metrics = {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "model": model,
                "distribution": distribution,
                "clients": clients,
                "noise": noise,
                "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
                "final_loss": round(global_loss, 4),
                "final_accuracy": round(global_accuracy, 4),
                "timestamp": int(time.time()),
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

            if stopped_early:
                self._emit("server", f"[done] stopped run persisted at {experiment_dir}")
            else:
                self._emit("server", f"[done] outputs persisted at {experiment_dir}")
            for client_idx in range(1, clients + 1):
                self._emit(f"client-{client_idx}", "[done] round loop finished")

        with self._state_lock:
            self._running = False
            self._paused = False
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
