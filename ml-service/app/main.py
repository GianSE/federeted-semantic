from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.training.orchestrator import orchestrator

app = FastAPI(title="semantic-ml-service")


class AWGNConfig(BaseModel):
    enabled: bool = False
    snr_db: float | None = None


class TrainRequest(BaseModel):
    dataset: Literal["mnist", "cifar10", "cifar100"] = "mnist"
    model: Literal["ae", "cnn_ae", "cnn_vae"] = "ae"
    distribution: Literal["iid", "non_iid"] = "iid"
    clients: int = 3
    noise: dict = {
        "channel": 0,
        "packet_loss": 0,
        "latency": 0,
        "client_drift": 0,
    }
    awgn: AWGNConfig = AWGNConfig()


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-service"}


@app.post("/training/start")
def training_start(payload: TrainRequest):
    clients = max(1, min(8, payload.clients))
    return orchestrator.start(
        payload.dataset,
        payload.model,
        payload.distribution,
        payload.noise,
        payload.awgn.model_dump(),
        clients,
    )


@app.get("/training/status")
def training_status():
    return orchestrator.status()


@app.post("/training/pause")
def training_pause():
    return orchestrator.pause()


@app.post("/training/resume")
def training_resume():
    return orchestrator.resume()


@app.post("/training/stop")
def training_stop():
    return orchestrator.stop()


@app.post("/training/logs/clear")
def training_logs_clear(payload: dict | None = None):
    payload = payload or {}
    raw_clients = payload.get("clients")
    clients = int(raw_clients) if raw_clients is not None else None
    return orchestrator.clear_logs(clients)


@app.get("/training/logs/stream")
def training_logs_stream(target: str = "server"):
    def event_gen():
        for message in orchestrator.stream(target):
            yield f"data: {message}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/results/latest")
def results_latest():
    latest = orchestrator.latest_experiment()
    if not latest:
        fallback = {
            "dataset": "-",
            "final_loss": None,
            "final_accuracy": None,
            "history": [],
        }
        return fallback

    return latest


@app.get("/results/experiments")
def results_experiments():
    return {"items": orchestrator.list_experiments()}


@app.get("/results/experiments/{experiment_id}")
def results_experiment(experiment_id: str):
    payload = orchestrator.get_experiment(experiment_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return payload


@app.get("/results/artifact/{experiment_id}/{artifact_path:path}")
def results_artifact(experiment_id: str, artifact_path: str):
    path = orchestrator.artifact_path(experiment_id, artifact_path)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)
