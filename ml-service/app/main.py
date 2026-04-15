"""
main.py
-------
FastAPI application entry point for the Federated Semantic Communication
ML Service.

Endpoints:
    GET  /health                  Health check
    POST /training/start          Start federated training (real mode)
    GET  /training/status         Current training state
    POST /training/pause          Pause training loop
    POST /training/resume         Resume paused training
    POST /training/stop           Stop training loop
    POST /training/logs/clear     Clear log files
    GET  /training/logs/stream    SSE stream of training logs

    GET  /results/latest          Latest experiment summary
    GET  /results/experiments     List all experiment summaries
    GET  /results/experiments/{id} Single experiment detail
    GET  /results/artifact/{id}/{path} Serve experiment artifacts (images, etc.)

    POST /semantic/process        Encode → quantize → decode a random image

    POST /experiment/benchmark    Cross-dataset benchmark (MSE, PSNR, SSIM,
                                  compression ratio for all datasets and models)

    GET  /info/architecture       System architecture description (JSON)
"""

import os
import time
from typing import Literal

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.training.orchestrator import orchestrator
from app.core.model_utils import get_model, snr_to_noise_std
from app.core.image_utils import (
    load_dataset,
    quantize_latent,
    dequantize_latent,
    compute_mse,
    compute_psnr,
    compute_ssim,
    get_original_bytes,
    get_latent_bytes,
    DATASET_META,
)

app = FastAPI(
    title="Federated Semantic Communication — ML Service",
    description=(
        "Research testbed demonstrating that latent representations (VAE / AE) "
        "reduce data transmission size while preserving semantic information."
    ),
    version="2.0.0",
)


# ===========================================================================
# Request / response schemas
# ===========================================================================

class AWGNConfig(BaseModel):
    enabled: bool = False
    snr_db: float | None = None


class TrainRequest(BaseModel):
    dataset: Literal["mnist", "fashion", "cifar10", "cifar100"] = "mnist"
    model: Literal["ae", "cnn_ae", "cnn_vae"] = "ae"
    clients: int = 3
    awgn: AWGNConfig = AWGNConfig()
    rounds: int = 5
    # Number of local epochs per client round (real training mode only)
    epochs: int = 5


class SemanticProcessRequest(BaseModel):
    model_type: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    dataset: Literal["mnist", "fashion", "cifar10"] = "fashion"
    bits: int = 8
    awgn: AWGNConfig = AWGNConfig()


class BenchmarkRequest(BaseModel):
    """
    Run a full cross-dataset benchmark.

    Evaluates N random test images per (dataset × model) combination
    and returns MSE, PSNR, SSIM, compression_ratio, and byte sizes.
    """
    datasets: list[Literal["mnist", "fashion", "cifar10"]] = ["mnist", "fashion", "cifar10"]
    models: list[Literal["cnn_ae", "cnn_vae"]] = ["cnn_vae", "cnn_ae"]
    bits: int = 8
    num_samples: int = 20
    seed: int = 42


# ===========================================================================
# Helpers
# ===========================================================================

def _format_tensor(tensor: torch.Tensor) -> list:
    """Convert a tensor to a nested Python list suitable for JSON serialisation."""
    return tensor.squeeze().cpu().float().numpy().tolist()


def _load_model(model_type: str, dataset: str) -> torch.nn.Module:
    """
    Instantiate and (if available) load pre-trained weights for a model.

    Args:
        model_type: "cnn_ae" or "cnn_vae".
        dataset:    Dataset name (used to determine image channels / size).

    Returns:
        Model in eval mode.
    """
    meta = DATASET_META.get(dataset, DATASET_META["mnist"])
    channels = meta["channels"]
    img_size = meta["height"]

    model = get_model(model_type, input_channels=channels, image_size=img_size)
    weights_path = f"app/core/{dataset}_{model_type}.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
    else:
        # Weights not yet available — model will produce untrained output.
        # Run `python -m app.train_local` to generate weights.
        pass
    model.eval()
    return model


def _encode(model: torch.nn.Module, model_type: str, x: torch.Tensor) -> torch.Tensor:
    """Encode an image tensor; returns the mean latent (mu) for VAE."""
    with torch.no_grad():
        if model_type == "cnn_vae":
            mu, _ = model.encode(x)
            return mu
        return model.encode(x)


# ===========================================================================
# Health check
# ===========================================================================

@app.get("/health")
def health():
    """Service liveness probe."""
    return {"status": "ok", "service": "ml-service", "version": "2.0.0"}


# ===========================================================================
# Training endpoints (delegate to orchestrator)
# ===========================================================================

@app.post("/training/start")
def training_start(payload: TrainRequest):
    """Start federated training (real mode)."""
    clients = max(1, min(8, payload.clients))
    rounds = max(1, min(50, payload.rounds))
    return orchestrator.start(
        payload.dataset,
        payload.model,
        clients,
        payload.awgn.model_dump(),
        rounds,
        epochs=payload.epochs,
    )


@app.get("/training/status")
def training_status():
    """Return current training state (running, paused, active_clients)."""
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
    """SSE endpoint: streams log lines for a given target (server or client-N)."""
    def event_gen():
        for message in orchestrator.stream(target):
            yield f"data: {message}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ===========================================================================
# Results endpoints (delegate to orchestrator)
# ===========================================================================

@app.get("/results/latest")
def results_latest():
    latest = orchestrator.latest_experiment()
    if not latest:
        return {
            "dataset": "-",
            "final_loss": None,
            "final_accuracy": None,
            "history": [],
        }
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


# ===========================================================================
# Semantic communication endpoints
# ===========================================================================

@app.post("/semantic/process")
def semantic_process(req: SemanticProcessRequest):
    """
    Demonstrate the semantic compression pipeline on a single random image.

    Flow:
        original image → encode → quantize → dequantize → decode → reconstructed

    Returns MSE, PSNR, SSIM, byte sizes, and the compression ratio.
    """
    try:
        model = _load_model(req.model_type, req.dataset)
        dataset_obj = load_dataset(req.dataset, train=False)

        idx = int(torch.randint(0, len(dataset_obj), (1,)).item())
        original, label = dataset_obj[idx]
        original = original.unsqueeze(0)          # [1, C, H, W]

        awgn_enabled = bool(req.awgn.enabled)
        awgn_snr = req.awgn.snr_db
        if awgn_enabled and awgn_snr is None:
            awgn_snr = 10.0

        with torch.no_grad():
            latent = _encode(model, req.model_type, original)
            q_latent, scale = quantize_latent(latent, bits=req.bits)
            dq_latent = dequantize_latent(q_latent, scale)

            reconstructed_clean = model.decode(dq_latent)
            reconstructed_noisy = None

            if awgn_enabled:
                noise_std = snr_to_noise_std(awgn_snr)
                noisy_latent = dq_latent + torch.randn_like(dq_latent) * noise_std
                reconstructed_noisy = model.decode(noisy_latent)

        original_bytes = get_original_bytes(req.dataset)
        latent_bytes_q  = get_latent_bytes(latent, req.bits)
        latent_bytes_f32 = get_latent_bytes(latent, 32)
        ratio = original_bytes / latent_bytes_q if latent_bytes_q > 0 else float("inf")

        mse_clean = compute_mse(original, reconstructed_clean)
        psnr_clean = compute_psnr(original, reconstructed_clean)
        ssim_clean = compute_ssim(original, reconstructed_clean)

        if awgn_enabled and reconstructed_noisy is not None:
            mse_noisy = compute_mse(original, reconstructed_noisy)
            psnr_noisy = compute_psnr(original, reconstructed_noisy)
            ssim_noisy = compute_ssim(original, reconstructed_noisy)
        else:
            mse_noisy = psnr_noisy = ssim_noisy = None

        primary_recon = reconstructed_noisy if awgn_enabled and reconstructed_noisy is not None else reconstructed_clean

        return {
            "status": "ok",
            "original": _format_tensor(original),
            "reconstructed": _format_tensor(primary_recon),
            "reconstructed_clean": _format_tensor(reconstructed_clean),
            "reconstructed_noisy": _format_tensor(reconstructed_noisy) if reconstructed_noisy is not None else None,
            "label": int(label),
            "mse": mse_noisy if mse_noisy is not None else mse_clean,
            "psnr": psnr_noisy if psnr_noisy is not None else psnr_clean,
            "ssim": ssim_noisy if ssim_noisy is not None else ssim_clean,
            "mse_clean": mse_clean,
            "psnr_clean": psnr_clean,
            "ssim_clean": ssim_clean,
            "mse_noisy": mse_noisy,
            "psnr_noisy": psnr_noisy,
            "ssim_noisy": ssim_noisy,
            "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
            "original_bytes": original_bytes,
            "latent_size_float": latent_bytes_f32,
            "latent_size_int8": latent_bytes_q,
            "compression_ratio": round(ratio, 2),
            "bandwidth_reduction_pct": round((1 - latent_bytes_q / original_bytes) * 100, 1),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ===========================================================================
# Cross-dataset benchmark endpoint
# ===========================================================================

@app.post("/experiment/benchmark")
def experiment_benchmark(req: BenchmarkRequest):
    """
    Run a structured cross-dataset benchmark.

    For each (dataset × model) pair, evaluate `num_samples` random test images
    and return aggregate statistics (mean ± std) for:
        - MSE, PSNR, SSIM
        - Compression ratio (original bytes / quantised latent bytes)
        - Bandwidth reduction percentage

    This endpoint provides the scientific evidence for the research hypothesis:
    "Latent representations reduce bandwidth while preserving semantic information."

    Uses a fixed seed for reproducibility across runs.
    """
    try:
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)

        results = []

        for dataset_name in req.datasets:
            # Load dataset once per dataset (expensive op)
            try:
                dataset_obj = load_dataset(dataset_name, train=False)
            except Exception as exc:
                # Dataset not downloaded yet — skip gracefully
                results.append({
                    "dataset": dataset_name,
                    "status": "error",
                    "error": str(exc),
                })
                continue

            original_bytes = get_original_bytes(dataset_name)

            for model_type in req.models:
                try:
                    model = _load_model(model_type, dataset_name)
                except Exception as exc:
                    results.append({
                        "dataset": dataset_name,
                        "model": model_type,
                        "status": "error",
                        "error": str(exc),
                    })
                    continue

                mses, psnrs, ssims, ratios = [], [], [], []
                weights_loaded = os.path.exists(f"app/core/{dataset_name}_{model_type}.pth")

                with torch.no_grad():
                    indices = torch.randperm(len(dataset_obj))[: req.num_samples]
                    for idx in indices:
                        original, _ = dataset_obj[int(idx)]
                        original = original.unsqueeze(0)

                        latent = _encode(model, model_type, original)
                        q_latent, scale = quantize_latent(latent, bits=req.bits)
                        dq_latent = dequantize_latent(q_latent, scale)
                        reconstructed = model.decode(dq_latent)

                        latent_bytes = get_latent_bytes(latent, req.bits)
                        ratio = original_bytes / latent_bytes if latent_bytes > 0 else float("inf")

                        mses.append(compute_mse(original, reconstructed))
                        psnrs.append(compute_psnr(original, reconstructed))
                        ssims.append(compute_ssim(original, reconstructed))
                        ratios.append(ratio)

                results.append({
                    "dataset": dataset_name,
                    "model": model_type,
                    "weights_loaded": weights_loaded,
                    "bits": req.bits,
                    "num_samples": req.num_samples,
                    "latent_dim": 32,
                    "original_bytes": original_bytes,
                    "latent_bytes": get_latent_bytes(
                        torch.zeros(1, 32), req.bits
                    ),
                    "mse_mean": float(np.mean(mses)),
                    "mse_std": float(np.std(mses)),
                    "psnr_mean": float(np.mean(psnrs)),
                    "psnr_std": float(np.std(psnrs)),
                    "ssim_mean": float(np.mean(ssims)),
                    "ssim_std": float(np.std(ssims)),
                    "compression_ratio_mean": float(np.mean(ratios)),
                    "bandwidth_reduction_pct": round(
                        (1 - 1 / float(np.mean(ratios))) * 100, 1
                    ) if float(np.mean(ratios)) > 0 else 0.0,
                    "scalability": {
                        f"{n}_devices": {
                            "total_original_kb": round(n * original_bytes / 1024, 2),
                            "total_latent_kb": round(
                                n * get_latent_bytes(torch.zeros(1, 32), req.bits) / 1024, 2
                            ),
                        }
                        for n in [1, 5, 10, 50, 100]
                    },
                    "status": "ok",
                })

        return {
            "status": "ok",
            "seed": req.seed,
            "bits": req.bits,
            "timestamp": int(time.time()),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ===========================================================================
# System architecture info endpoint
# ===========================================================================

@app.get("/info/architecture")
def info_architecture():
    """
    Return a structured description of the system architecture.

    Useful for the frontend presentation page and academic documentation.
    """
    return {
        "title": "Federated Semantic Communication Testbed",
        "hypothesis": (
            "Latent representations produced by a Variational Autoencoder (VAE) "
            "can significantly reduce data transmission bandwidth while preserving "
            "sufficient semantic information for accurate reconstruction at the receiver."
        ),
        "layers": [
            {
                "name": "Frontend",
                "technology": "React 18 + Vite + Tailwind CSS",
                "description": "Interactive research dashboard with 4 pages: Training, Results, Semantic Comms, and Cross-Dataset Benchmark.",
            },
            {
                "name": "Backend (API Gateway)",
                "technology": "Fastify (Node.js)",
                "description": "Lightweight API gateway that proxies requests from the browser to the Python ML service. Handles CORS, multipart, and SSE streaming.",
            },
            {
                "name": "ML Service",
                "technology": "FastAPI + PyTorch 2.x",
                "description": "Core research engine. Implements VAE/AE training, semantic encoding/decoding, AWGN noise, quantization, and quality metrics.",
            },
        ],
        "models": [
            {
                "id": "cnn_vae",
                "name": "Convolutional Variational Autoencoder (CNN-VAE)",
                "encoder": "Conv2d(in→32) → MaxPool → Conv2d(32→64) → MaxPool → Flatten → FC(256) → μ,σ (32-dim)",
                "decoder": "FC(32→256) → Reshape(64×7×7) → ConvTranspose → ConvTranspose → Sigmoid",
                "latent_dim": 32,
                "loss": "MSE + β·KL(q(z|x) || p(z))",
                "note": "Recommended: probabilistic latent space improves robustness under AWGN noise.",
            },
            {
                "id": "cnn_ae",
                "name": "Convolutional Autoencoder (CNN-AE)",
                "encoder": "Conv2d(in→32) → MaxPool → Conv2d(32→64) → MaxPool → Flatten → FC(256→32)",
                "decoder": "FC(32→256) → Reshape(64×7×7) → ConvTranspose → ConvTranspose → Sigmoid",
                "latent_dim": 32,
                "loss": "MSE",
            },
        ],
        "datasets": [
            {"name": "MNIST",         "key": "mnist",   "classes": 10, "channels": 1, "resolution": "28×28", "raw_bytes": 3136,  "train_size": 60000, "test_size": 10000},
            {"name": "Fashion-MNIST", "key": "fashion", "classes": 10, "channels": 1, "resolution": "28×28", "raw_bytes": 3136,  "train_size": 60000, "test_size": 10000},
            {"name": "CIFAR-10",      "key": "cifar10", "classes": 10, "channels": 3, "resolution": "32×32", "raw_bytes": 12288, "train_size": 50000, "test_size": 10000},
        ],
        "metrics": [
            {"id": "mse",   "name": "Mean Squared Error (MSE)",             "unit": "—",  "better": "lower", "description": "Pixel-level reconstruction fidelity."},
            {"id": "psnr",  "name": "Peak Signal-to-Noise Ratio (PSNR)",    "unit": "dB", "better": "higher", "description": "Logarithmic reconstruction quality; >25 dB is generally considered good."},
            {"id": "ssim",  "name": "Structural Similarity Index (SSIM)",   "unit": "—",  "better": "higher", "description": "Perceptual similarity: accounts for luminance, contrast, and structure. Range [-1, 1]; 1 = identical."},
            {"id": "cr",    "name": "Compression Ratio",                    "unit": "×",  "better": "higher", "description": "original_bytes / latent_bytes. Higher means fewer bytes transmitted."},
            {"id": "bwred", "name": "Bandwidth Reduction",                  "unit": "%",  "better": "higher", "description": "(1 − 1/CR) × 100. Percentage of bandwidth saved vs. raw transmission."},
        ],
        "training": {
            "protocol": "Federated Averaging (FedAvg)",
            "mode": "FedAvg real (containers fl-server + fl-clients)",
            "rounds": 5,
            "note": "Use the training dashboard to start real federated training with optional AWGN; weights are saved to /ml-data/weights.",
        },
    }
