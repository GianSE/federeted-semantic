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
    dataset: Literal["mnist", "fashion", "cifar10", "cifar100"] = "mnist"
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

# ================================
# Módulo: Comunicação Semântica
# ================================
import torch
import numpy as np
from app.core.model_utils import get_model
from app.core.image_utils import load_dataset, quantize_latent, dequantize_latent, mask_image_bottom, mask_image_random, mask_image_right, compute_mse, compute_psnr

class SemanticCompleteRequest(BaseModel):
    mask_type: Literal["Metade Inferior", "Pixels Aleatórios", "Metade Direita"] = "Metade Inferior"
    mask_ratio: float = 0.5
    model_type: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    dataset: Literal["mnist", "fashion", "cifar10"] = "fashion"

class SemanticTradeoffRequest(BaseModel):
    model_type: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    dataset: Literal["mnist", "fashion", "cifar10"] = "cifar10"
    num_samples: int = 5
    snr_levels: list[float] = [20.0, 10.0, 5.0, 0.0]
    bits_levels: list[int] = [32, 16, 8, 4]

class SemanticProcessRequest(BaseModel):
    model_type: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    dataset: Literal["mnist", "fashion", "cifar10"] = "fashion"
    bits: int = 8

def format_tensor(tensor):
    return tensor.squeeze().cpu().float().numpy().tolist()

@app.post("/semantic/process")
def semantic_process(req: SemanticProcessRequest):
    try:
        import os
        channels = 3 if req.dataset == "cifar10" else 1
        img_size = 32 if req.dataset == "cifar10" else 28
        model = get_model(req.model_type, input_channels=channels, image_size=img_size)
        weights_path = f"app/core/{req.dataset}_{req.model_type}.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()
        dataset_obj = load_dataset(req.dataset, train=False)
        idx = torch.randint(0, len(dataset_obj), (1,)).item()
        original, label = dataset_obj[idx]
        original = original.unsqueeze(0)
        
        with torch.no_grad():
            if req.model_type == "cnn_vae":
                latent, _ = model.encode(original)
            else:
                latent = model.encode(original)
                
            q_latent, scale = quantize_latent(latent, bits=req.bits)
            dq_latent = dequantize_latent(q_latent, scale)
            reconstructed = model.decode(dq_latent)
        
        bytes_factor = 4 if req.bits >= 32 else (req.bits/8)
        bytes_scalar = 0 if req.bits >= 32 else 4
        latent_bytes = int(latent.numel() * bytes_factor + bytes_scalar)
        
        return {
            "status": "ok",
            "original": format_tensor(original),
            "reconstructed": format_tensor(reconstructed),
            "label": int(label),
            "mse": compute_mse(original, reconstructed),
            "psnr": compute_psnr(original, reconstructed),
            "latent_size_float": int(latent.numel() * 4),
            "latent_size_int8": latent_bytes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/complete")
def semantic_complete(req: SemanticCompleteRequest):
    try:
        import os
        channels = 3 if req.dataset == "cifar10" else 1
        img_size = 32 if req.dataset == "cifar10" else 28
        model = get_model(req.model_type, input_channels=channels, image_size=img_size)
        weights_path = f"app/core/{req.dataset}_{req.model_type}.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()
        dataset_obj = load_dataset(req.dataset, train=False)
        idx = torch.randint(0, len(dataset_obj), (1,)).item()
        original, label = dataset_obj[idx]
        original = original.unsqueeze(0)
        
        if req.mask_type == "Metade Inferior":
            masked = mask_image_bottom(original, req.mask_ratio)
        elif req.mask_type == "Pixels Aleatórios":
            masked = mask_image_random(original, req.mask_ratio)
        else:
            masked = mask_image_right(original, req.mask_ratio)
            
        with torch.no_grad():
            output = model(masked)
            completed = output[0] if isinstance(output, tuple) else output
            
        return {
            "status": "ok",
            "original": format_tensor(original),
            "masked": format_tensor(masked),
            "completed": format_tensor(completed),
            "mse": compute_mse(original, completed),
            "psnr": compute_psnr(original, completed)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/tradeoff")
def semantic_tradeoff(req: SemanticTradeoffRequest):
    try:
        import os
        channels = 3 if req.dataset == "cifar10" else 1
        img_size = 32 if req.dataset == "cifar10" else 28
        model = get_model(req.model_type, input_channels=channels, image_size=img_size)
        weights_path = f"app/core/{req.dataset}_{req.model_type}.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()
        
        dataset_obj = load_dataset(req.dataset, train=False)
        results = []
        
        with torch.no_grad():
            for _ in range(req.num_samples):
                idx = torch.randint(0, len(dataset_obj), (1,)).item()
                original, label = dataset_obj[idx]
                original = original.unsqueeze(0)
                
                for snr in req.snr_levels:
                    # AWGN on Latent Space
                    if req.model_type == "cnn_vae":
                        latent, _ = model.encode(original)
                    else:
                        latent = model.encode(original)
                        
                    from app.core.model_utils import snr_to_noise_std
                    noise_std = snr_to_noise_std(snr)
                    noisy_latent = latent + torch.randn_like(latent) * noise_std
                    
                    for bits in req.bits_levels:
                        # Quantization
                        q_latent, scale = quantize_latent(noisy_latent, bits)
                        dq_latent = dequantize_latent(q_latent, scale)
                        recon = model.decode(dq_latent)
                        
                        psnr = compute_psnr(original, recon)
                        bytes_factor = 4 if bits >= 32 else (bits/8)
                        bytes_scalar = 0 if bits >= 32 else 4
                        payload = int(latent.numel() * bytes_factor + bytes_scalar)
                        
                        results.append({
                            "snr_db": snr,
                            "bits": bits,
                            "psnr": psnr,
                            "payload_bytes": payload,
                            "dataset": req.dataset
                        })
        
        return {"status": "ok", "items": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
