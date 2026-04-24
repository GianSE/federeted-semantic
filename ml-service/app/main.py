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

    POST /classifier/train        Train dataset classifier
    GET  /classifier/status       Classifier training state
    POST /classifier/stop         Stop classifier training
    POST /classifier/logs/clear   Clear classifier logs
    GET  /classifier/logs/stream  SSE stream of classifier logs

    GET  /classifier/results/latest
    GET  /classifier/results/experiments
    GET  /classifier/results/experiments/{id}
    GET  /classifier/results/artifact/{id}/{path}

    GET  /results/latest          Latest experiment summary
    GET  /results/experiments     List all experiment summaries
    GET  /results/experiments/{id} Single experiment detail
    GET  /results/artifact/{id}/{path} Serve experiment artifacts (images, etc.)

    POST /semantic/process        Encode → quantize → decode a random image

    POST /experiment/benchmark    Cross-dataset benchmark (MSE, PSNR, SSIM,
                                  compression ratio for all datasets and models)

    GET  /info/architecture       System architecture description (JSON)
"""

import json
import time
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.training.orchestrator import orchestrator
from app.classifier_orchestrator import classifier_orchestrator
from app.core.config import RESULTADOS_ROOT
from app.core.model_utils import get_model
from app.core.classifier_utils import load_classifier, predict_topk, format_topk
from app.core.image_utils import (
    load_dataset,
    quantize_latent,
    dequantize_latent,
    compute_mse,
    compute_psnr,
    compute_ssim,
    get_original_bytes,
    get_latent_bytes,
    apply_awgn_noise,
    apply_random_pixel_mask,
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


class MaskingConfig(BaseModel):
    enabled: bool = False
    drop_rate: float = 0.25
    fill_value: float = 0.0


class ClassifierConfig(BaseModel):
    enabled: bool = True
    min_confidence: float = 0.5
    top_k: int = 1


class TrainRequest(BaseModel):
    dataset: Literal["mnist", "fashion", "cifar10"] = "mnist"
    model: Literal["ae", "cnn_ae", "cnn_vae"] = "ae"
    latent_dim: int = 32
    clients: int = 3
    awgn: AWGNConfig = AWGNConfig()
    masking: MaskingConfig = MaskingConfig()
    base_weights: str | None = None
    rounds: int = 5
    # Number of local epochs per client round (real training mode only)
    epochs: int = 5
    compression_mode: Literal["baseline", "compressed"] = "baseline"
    compression_bits: int = 8
    seed: int = 42


class ClassifierTrainRequest(BaseModel):
    dataset: Literal["mnist", "fashion", "cifar10"] = "mnist"
    latent_dim: int = 32
    epochs: int = 5
    batch: int = 128
    lr: float = 1e-3
    seed: int = 42
    top_k: int = 1
    min_confidence: float = 0.5
    eval_samples: int = 200
    snr_grid: list[float] = [5, 10, 15, 20, 25]
    masking_grid: list[float] = [0.1, 0.25, 0.4, 0.6]
    bits: int = 8
    semantic_model: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    semantic_weights: str | None = None


class SemanticProcessRequest(BaseModel):
    model_type: Literal["cnn_ae", "cnn_vae"] = "cnn_vae"
    dataset: Literal["mnist", "fashion", "cifar10"] = "fashion"
    latent_dim: int = 32
    bits: int = 8
    awgn: AWGNConfig = AWGNConfig()
    masking: MaskingConfig = MaskingConfig()
    base_weights: str | None = None
    classifier: ClassifierConfig = ClassifierConfig()


class BenchmarkRequest(BaseModel):
    """
    Run a full cross-dataset benchmark.

    Evaluates N random test images per (dataset × model) combination
    and returns MSE, PSNR, SSIM, compression_ratio, and byte sizes.
    """
    datasets: list[Literal["mnist", "fashion", "cifar10"]] = [
        "mnist",
        "fashion",
        "cifar10",
    ]
    models: list[Literal["cnn_ae", "cnn_vae"]] = ["cnn_vae", "cnn_ae"]
    latent_dim: int = 32
    bits: int = 8
    num_samples: int = 20
    seed: int = 42
    awgn: AWGNConfig = AWGNConfig()
    masking: MaskingConfig = MaskingConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    include_samples: bool = False


# ===========================================================================
# Helpers
# ===========================================================================

def _format_tensor(tensor: torch.Tensor) -> list:
    """Convert a tensor to a nested Python list suitable for JSON serialisation."""
    return tensor.squeeze().cpu().float().numpy().tolist()


def _resolve_weights_path(dataset: str, model_type: str, base_weights: str | None, latent_dim: int = 32) -> tuple[Path | None, str | None]:
    # DATA_ROOT is set by docker-compose to /app/data/ml-data
    # fl-server saves weights to ML_DATA_DIR/weights = /app/data/ml-data/weights
    # We must use the same path so ml-service can find the trained weights.
    import os as _os
    _data_root = Path(_os.environ.get("DATA_ROOT", "/app/data/ml-data"))
    weights_dir = _data_root / "weights"
    archive_dir = weights_dir / "archive"
    prefix = f"{dataset}_{model_type}_d{latent_dim}"
    latest_path = weights_dir / f"{prefix}.pth"
    core_path = Path(f"app/core/{prefix}.pth")

    if base_weights is None or base_weights in {"", "random", "none"}:
        return None, None

    selected_path: Path | None = None
    source: str | None = None

    if base_weights:
        if base_weights == "latest":
            selected_path = latest_path
            source = "latest"
        else:
            safe_name = Path(base_weights).name
            candidate = weights_dir / safe_name
            archive_candidate = archive_dir / safe_name
            if candidate.exists():
                selected_path = candidate
                source = f"weights/{safe_name}"
            elif archive_candidate.exists():
                selected_path = archive_candidate
                source = f"archive/{safe_name}"

    if selected_path and selected_path.exists():
        return selected_path, source
    return None, None


def _load_model(model_type: str, dataset: str, base_weights: str | None = None, latent_dim: int = 32) -> tuple[torch.nn.Module, bool, str | None]:
    """
    Instantiate and (if available) load pre-trained weights for a model.

    Args:
        model_type: "cnn_ae" or "cnn_vae".
        dataset:    Dataset name (used to determine image channels / size).
        base_weights: Optional weight key ("latest" or archive filename).

    Returns:
        Model in eval mode, weights_loaded flag, and weights source label.
    """
    meta = DATASET_META.get(dataset, DATASET_META["mnist"])
    channels = meta["channels"]
    img_size = meta["height"]

    model = get_model(model_type, latent_dim=latent_dim, input_channels=channels, image_size=img_size)
    weights_path, source = _resolve_weights_path(dataset, model_type, base_weights, latent_dim)
    if weights_path and weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        weights_loaded = True
    else:
        weights_loaded = False
    model.eval()
    return model, weights_loaded, source


def _encode(model: torch.nn.Module, model_type: str, x: torch.Tensor) -> torch.Tensor:
    """Encode an image tensor; returns the mean latent (mu) for VAE."""
    with torch.no_grad():
        if model_type == "cnn_vae":
            mu, _ = model.encode(x)
            return mu
        return model.encode(x)


def _classify_sample(
    classifier: torch.nn.Module,
    image: torch.Tensor,
    label: int,
    top_k: int,
    min_confidence: float,
) -> dict:
    indices, probs = predict_topk(classifier, image, top_k=top_k)
    topk = format_topk(indices[0], probs[0])
    top1_pred = int(indices[0][0].item())
    top1_conf = float(probs[0][0].item())
    label = int(label)
    in_topk = label in indices[0].tolist()
    correct_top1 = top1_pred == label
    recognized = (in_topk if top_k > 1 else correct_top1) and top1_conf >= min_confidence
    return {
        "pred": top1_pred,
        "confidence": top1_conf,
        "topk": topk,
        "correct_top1": bool(correct_top1),
        "recognized": bool(recognized),
    }


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
        payload.masking.model_dump(),
        payload.base_weights,
        rounds,
        epochs=payload.epochs,
        compression_mode=payload.compression_mode,
        compression_bits=payload.compression_bits,
        seed=payload.seed,
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


# ==========================================================================
# Classifier training endpoints
# ==========================================================================

@app.post("/classifier/train")
def classifier_train(payload: ClassifierTrainRequest):
    return classifier_orchestrator.start(payload.model_dump())


@app.get("/classifier/status")
def classifier_status():
    return classifier_orchestrator.status()


@app.post("/classifier/stop")
def classifier_stop():
    return classifier_orchestrator.stop()


@app.post("/classifier/logs/clear")
def classifier_logs_clear():
    return classifier_orchestrator.clear_logs()


@app.get("/classifier/logs/stream")
def classifier_logs_stream():
    def event_gen():
        for message in classifier_orchestrator.stream():
            yield f"data: {message}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ==========================================================================
# Weights discovery
# ==========================================================================

@app.get("/weights/list")
def weights_list(dataset: str, model: str):
    weights_dir = Path("/ml-data/weights")
    archive_dir = weights_dir / "archive"
    items: list[dict] = []

    prefix = f"{dataset}_{model}"
    latest_path = weights_dir / f"{prefix}.pth"
    if latest_path.exists():
        items.append({
            "key": "latest",
            "label": f"latest ({latest_path.name})",
            "filename": latest_path.name,
            "mtime": latest_path.stat().st_mtime,
        })

    if archive_dir.exists():
        for path in sorted(archive_dir.glob(f"{prefix}_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
            items.append({
                "key": path.name,
                "label": path.name,
                "filename": path.name,
                "mtime": path.stat().st_mtime,
            })

    return {"items": items}


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


@app.post("/results/experiments/{experiment_id}/regenerate-figures")
def results_regenerate_figures(experiment_id: str):
    try:
        return regenerate_figures(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/results/artifact/{experiment_id}/{artifact_path:path}")
def results_artifact(experiment_id: str, artifact_path: str):
    path = orchestrator.artifact_path(experiment_id, artifact_path)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


# ==========================================================================
# Classifier results endpoints
# ==========================================================================

@app.get("/classifier/results/latest")
def classifier_results_latest():
    latest = classifier_orchestrator.latest_experiment()
    if not latest:
        return {
            "dataset": "-",
            "final_accuracy": None,
            "history": [],
        }
    return latest


@app.get("/classifier/results/experiments")
def classifier_results_experiments():
    return {"items": classifier_orchestrator.list_experiments()}


@app.get("/classifier/results/experiments/{experiment_id}")
def classifier_results_experiment(experiment_id: str):
    payload = classifier_orchestrator.get_experiment(experiment_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return payload


@app.get("/classifier/results/artifact/{experiment_id}/{artifact_path:path}")
def classifier_results_artifact(experiment_id: str, artifact_path: str):
    path = classifier_orchestrator.artifact_path(experiment_id, artifact_path)
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
        model, weights_loaded, weights_source = _load_model(
            req.model_type,
            req.dataset,
            req.base_weights,
            req.latent_dim,
        )
        classifier_enabled = bool(req.classifier.enabled)
        classifier, classifier_loaded, classifier_source = load_classifier(req.dataset)
        if not classifier_enabled:
            classifier_loaded = False
            classifier_source = None
        dataset_obj = load_dataset(req.dataset, train=False)

        idx = int(torch.randint(0, len(dataset_obj), (1,)).item())
        original, label = dataset_obj[idx]
        original = original.unsqueeze(0)          # [1, C, H, W]

        awgn_enabled = bool(req.awgn.enabled)
        awgn_snr = req.awgn.snr_db
        if awgn_enabled and awgn_snr is None:
            awgn_snr = 10.0
        masking_enabled = bool(req.masking.enabled)
        masking_drop_rate = float(req.masking.drop_rate) if req.masking.drop_rate is not None else 0.0
        masking_fill_value = float(req.masking.fill_value)

        received = original.clone()
        if masking_enabled:
            received = apply_random_pixel_mask(received, masking_drop_rate, masking_fill_value)
        if awgn_enabled:
            received = apply_awgn_noise(received, awgn_snr)

        with torch.no_grad():
            latent = _encode(model, req.model_type, received)
            q_latent, scale = quantize_latent(latent, bits=req.bits)
            dq_latent = dequantize_latent(q_latent, scale)

            reconstructed = model.decode(dq_latent)

        classifier_payload = {
            "enabled": classifier_enabled,
            "loaded": bool(classifier_loaded),
            "source": classifier_source,
            "top_k": int(req.classifier.top_k),
            "min_confidence": float(req.classifier.min_confidence),
        }
        if classifier_enabled and classifier_loaded and classifier is not None:
            top_k = max(1, int(req.classifier.top_k))
            min_conf = float(req.classifier.min_confidence)
            with torch.no_grad():
                classifier_payload["original"] = _classify_sample(
                    classifier, original, label, top_k, min_conf
                )
                classifier_payload["received"] = _classify_sample(
                    classifier, received, label, top_k, min_conf
                )
                classifier_payload["reconstructed"] = _classify_sample(
                    classifier, reconstructed, label, top_k, min_conf
                )

        original_bytes = get_original_bytes(req.dataset)
        latent_bytes_q  = get_latent_bytes(latent, req.bits)
        latent_bytes_f32 = get_latent_bytes(latent, 32)
        ratio = original_bytes / latent_bytes_q if latent_bytes_q > 0 else float("inf")

        mse_received = compute_mse(original, received)
        psnr_received = compute_psnr(original, received)
        ssim_received = compute_ssim(original, received)

        mse_recon = compute_mse(original, reconstructed)
        psnr_recon = compute_psnr(original, reconstructed)
        ssim_recon = compute_ssim(original, reconstructed)

        return {
            "status": "ok",
            "original": _format_tensor(original),
            "received": _format_tensor(received),
            "reconstructed": _format_tensor(reconstructed),
            "label": int(label),
            "classifier": classifier_payload,
            "weights_loaded": weights_loaded,
            "weights_source": weights_source,
            "mse": mse_recon,
            "psnr": psnr_recon,
            "ssim": ssim_recon,
            "mse_received": mse_received,
            "psnr_received": psnr_received,
            "ssim_received": ssim_received,
            "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
            "masking": {"enabled": masking_enabled, "drop_rate": masking_drop_rate, "fill_value": masking_fill_value},
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

        awgn_enabled = bool(req.awgn.enabled)
        awgn_snr = req.awgn.snr_db
        if awgn_enabled and awgn_snr is None:
            awgn_snr = 10.0
        masking_enabled = bool(req.masking.enabled)
        masking_drop_rate = float(req.masking.drop_rate) if req.masking.drop_rate is not None else 0.0
        masking_fill_value = float(req.masking.fill_value)
        classifier_enabled = bool(req.classifier.enabled)
        top_k = max(1, int(req.classifier.top_k))
        min_conf = float(req.classifier.min_confidence)

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
            classifier, classifier_loaded, classifier_source = load_classifier(dataset_name)
            if not classifier_enabled:
                classifier_loaded = False
                classifier_source = None

            for model_type in req.models:
                try:
                    model, weights_loaded, _ = _load_model(model_type, dataset_name, "latest", req.latent_dim)
                    if not weights_loaded:
                        core_path = Path(f"app/core/{dataset_name}_{model_type}_d{req.latent_dim}.pth")
                        if core_path.exists():
                            model.load_state_dict(
                                torch.load(core_path, map_location="cpu", weights_only=True)
                            )
                            weights_loaded = True
                except Exception as exc:
                    results.append({
                        "dataset": dataset_name,
                        "model": model_type,
                        "status": "error",
                        "error": str(exc),
                    })
                    continue

                mses, psnrs, ssims, ratios = [], [], [], []
                cls_hits_original = 0
                cls_hits_received = 0
                cls_hits_recon = 0
                cls_hits_both = 0
                cls_samples = []
                with torch.no_grad():
                    indices = torch.randperm(len(dataset_obj))[: req.num_samples]
                    for idx in indices:
                        original, label = dataset_obj[int(idx)]
                        original = original.unsqueeze(0)

                        received = original.clone()
                        if masking_enabled:
                            received = apply_random_pixel_mask(received, masking_drop_rate, masking_fill_value)
                        if awgn_enabled:
                            received = apply_awgn_noise(received, awgn_snr)

                        latent = _encode(model, model_type, received)
                        q_latent, scale = quantize_latent(latent, bits=req.bits)
                        dq_latent = dequantize_latent(q_latent, scale)
                        reconstructed = model.decode(dq_latent)

                        latent_bytes = get_latent_bytes(latent, req.bits)
                        ratio = original_bytes / latent_bytes if latent_bytes > 0 else float("inf")

                        mses.append(compute_mse(original, reconstructed))
                        psnrs.append(compute_psnr(original, reconstructed))
                        ssims.append(compute_ssim(original, reconstructed))
                        ratios.append(ratio)

                        if classifier_enabled and classifier_loaded and classifier is not None:
                            label = int(label)
                            pred_original = _classify_sample(
                                classifier, original, label, top_k, min_conf
                            )
                            pred_received = _classify_sample(
                                classifier, received, label, top_k, min_conf
                            )
                            pred_recon = _classify_sample(
                                classifier, reconstructed, label, top_k, min_conf
                            )
                            cls_hits_original += int(pred_original["recognized"])
                            cls_hits_received += int(pred_received["recognized"])
                            cls_hits_recon += int(pred_recon["recognized"])
                            cls_hits_both += int(
                                pred_original["recognized"] and pred_recon["recognized"]
                            )

                            if req.include_samples:
                                cls_samples.append(
                                    {
                                        "label": label,
                                        "original_img": _format_tensor(original),
                                        "received_img": _format_tensor(received),
                                        "reconstructed_img": _format_tensor(reconstructed),
                                        "original": pred_original,
                                        "received": pred_received,
                                        "reconstructed": pred_recon,
                                        # Convenience flags for notebook visualization
                                        "orig_ok": pred_original["recognized"],
                                        "recon_ok": pred_recon["recognized"],
                                        "semantic_lost": pred_original["recognized"] and not pred_recon["recognized"],
                                        "psnr": float(psnrs[-1]),
                                        "ssim": float(ssims[-1]),
                                    }
                                )

                results.append({
                    "dataset": dataset_name,
                    "model": model_type,
                    "weights_loaded": weights_loaded,
                    "bits": req.bits,
                    "num_samples": req.num_samples,
                    "latent_dim": req.latent_dim,
                    "original_bytes": original_bytes,
                    "latent_bytes": get_latent_bytes(
                        torch.zeros(1, req.latent_dim), req.bits
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
                    "classification": {
                        "enabled": classifier_enabled,
                        "loaded": bool(classifier_loaded),
                        "source": classifier_source,
                        "top_k": top_k,
                        "min_confidence": min_conf,
                        "accuracy_original": round(cls_hits_original / max(1, req.num_samples), 4)
                        if classifier_enabled and classifier_loaded
                        else None,
                        "accuracy_received": round(cls_hits_received / max(1, req.num_samples), 4)
                        if classifier_enabled and classifier_loaded
                        else None,
                        "accuracy_reconstructed": round(cls_hits_recon / max(1, req.num_samples), 4)
                        if classifier_enabled and classifier_loaded
                        else None,
                        "semantic_preservation_rate": round(cls_hits_recon / max(1, req.num_samples), 4)
                        if classifier_enabled and classifier_loaded
                        else None,
                        "semantic_preservation_given_original": round(
                            cls_hits_both / max(1, cls_hits_original), 4
                        ) if classifier_enabled and classifier_loaded else None,
                        "samples": cls_samples if req.include_samples and classifier_enabled and classifier_loaded else None,
                    },
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

        timestamp = int(time.time())
        response_payload = {
            "status": "ok",
            "seed": req.seed,
            "bits": req.bits,
            "awgn": {"enabled": awgn_enabled, "snr_db": awgn_snr},
            "masking": {
                "enabled": masking_enabled,
                "drop_rate": masking_drop_rate,
                "fill_value": masking_fill_value,
            },
            "classifier": {
                "enabled": classifier_enabled,
                "top_k": top_k,
                "min_confidence": min_conf,
                "include_samples": bool(req.include_samples),
            },
            "timestamp": timestamp,
            "results": results,
        }
        try:
            bench_dir = RESULTADOS_ROOT / "benchmarks"
            bench_dir.mkdir(parents=True, exist_ok=True)
            bench_path = bench_dir / f"benchmark_{timestamp}.json"
            bench_path.write_text(
                json.dumps(response_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            response_payload["saved_path"] = str(bench_path)
        except Exception:
            response_payload["saved_path"] = None

        return response_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



# ===========================================================================
# Classifier quick-train endpoint
# Trains a lightweight SimpleClassifier inside the container so the benchmark
# can evaluate semantic preservation via classification accuracy.
# POST /classifier/train-quick  { "dataset": "mnist", "epochs": 3, "samples": 2000 }
# GET  /classifier/train-quick/status
# ===========================================================================

class ClassifierQuickTrainRequest(BaseModel):
    dataset: Literal["mnist", "fashion", "cifar10"] = "mnist"
    epochs: int = 3
    samples: int = 2000   # how many training images to use
    seed: int = 42


_clf_train_status: dict = {"running": False, "done": False, "error": None,
                            "epoch": 0, "total_epochs": 0, "dataset": None,
                            "accuracy": None, "saved_to": None}
_clf_train_lock = __import__("threading").Lock()


@app.post("/classifier/train-quick")
def classifier_train_quick(req: ClassifierQuickTrainRequest):
    """
    Train a lightweight image classifier inside the container.
    Required before /experiment/benchmark can evaluate semantic accuracy.
    Takes ~2-5 min for 2000 samples / 3 epochs on CPU.
    """
    import threading as _threading

    with _clf_train_lock:
        if _clf_train_status["running"]:
            return {"status": "already_running", **_clf_train_status}
        _clf_train_status.update({
            "running": True, "done": False, "error": None,
            "epoch": 0, "total_epochs": req.epochs, "dataset": req.dataset,
            "accuracy": None, "saved_to": None,
        })

    def _worker():
        import os as _os
        import torch as _t
        import torch.nn as _nn
        import torch.optim as _optim
        from torch.utils.data import DataLoader, Subset
        from app.core.classifier_utils import SimpleClassifier, MobileNetClassifier
        from app.core.image_utils import load_dataset, DATASET_META

        try:
            _t.manual_seed(req.seed)
            meta  = DATASET_META[req.dataset]
            
            if req.dataset == "cifar10":
                model = MobileNetClassifier(num_classes=meta.get("classes", 10))
            else:
                model = SimpleClassifier(
                    input_channels=meta["channels"],
                    image_size=meta["height"],
                    num_classes=meta["classes"],
                )
            
            ds_full = load_dataset(req.dataset, train=True)
            
            # --- OVERRIDE TRANSFORMS FOR DATA AUGMENTATION (Avoid Overfitting) ---
            from torchvision import transforms as _tf
            if req.dataset == "cifar10":
                # Keras equivalent: Resize(96,96)
                ds_full.transform = _tf.Compose([
                    _tf.Resize((96, 96)),
                    _tf.RandomHorizontalFlip(p=0.5),
                    _tf.RandomRotation(15),
                    _tf.ColorJitter(brightness=0.2, contrast=0.2),
                    _tf.ToTensor()
                ])
            elif req.dataset == "fashion":
                ds_full.transform = _tf.Compose([
                    _tf.RandomHorizontalFlip(p=0.5),
                    _tf.RandomRotation(10),
                    _tf.ToTensor()
                ])
            # ---------------------------------------------------------------------

            n = min(req.samples, len(ds_full))
            indices = list(range(n))
            ds      = Subset(ds_full, indices)
            loader  = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

            # MobileNet Fine-tuning requires ultra low learning rate
            learning_rate = 1e-5 if req.dataset == "cifar10" else 1e-3
            opt      = _optim.Adam(model.parameters(), lr=learning_rate)
            crit     = _nn.CrossEntropyLoss()
            model.train()

            for ep in range(1, req.epochs + 1):
                with _clf_train_lock:
                    _clf_train_status["epoch"] = ep
                correct = total = 0
                for imgs, labels in loader:
                    opt.zero_grad()
                    out  = model(imgs)
                    loss = crit(out, labels)
                    loss.backward()
                    opt.step()
                    preds   = out.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += len(labels)
                acc = correct / total

            # Quick validation on 500 test samples
            ds_test  = load_dataset(req.dataset, train=False)
            if req.dataset == "cifar10":
                ds_test.transform = _tf.Compose([_tf.Resize((96, 96)), _tf.ToTensor()])
            
            n_test   = min(500, len(ds_test))
            ds_t     = Subset(ds_test, list(range(n_test)))
            ldr_test = DataLoader(ds_t, batch_size=64, shuffle=False, num_workers=0)
            model.eval()
            correct = total = 0
            with _t.no_grad():
                for imgs, labels in ldr_test:
                    preds   = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += len(labels)
            val_acc = correct / total

            # Save weights alongside autoencoder weights
            data_root  = Path(_os.environ.get("DATA_ROOT", "/app/data/ml-data"))
            weights_dir = data_root / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            save_path = weights_dir / f"{req.dataset}_classifier.pth"
            _t.save(model.state_dict(), save_path)

            with _clf_train_lock:
                _clf_train_status.update({
                    "running": False, "done": True, "error": None,
                    "accuracy": round(val_acc, 4),
                    "saved_to": str(save_path),
                })

        except Exception as exc:
            with _clf_train_lock:
                _clf_train_status.update({"running": False, "done": False, "error": str(exc)})

    _threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "dataset": req.dataset,
            "epochs": req.epochs, "samples": req.samples}


@app.get("/classifier/train-quick/status")
def classifier_train_quick_status():
    """Return current status of the quick classifier training job."""
    with _clf_train_lock:
        return dict(_clf_train_status)


# ===========================================================================
# Data preparation endpoint (download datasets inside the container)
# ===========================================================================

class DataPrepareRequest(BaseModel):
    datasets: list[Literal["mnist", "fashion", "cifar10"]] = [
        "mnist", "fashion", "cifar10"
    ]


_data_prepare_status: dict = {"running": False, "done": [], "errors": [], "current": None}
_data_prepare_lock = __import__("threading").Lock()


@app.post("/data/prepare")
def data_prepare(req: DataPrepareRequest):
    """
    Download requested datasets inside the container (into the shared volume).

    This endpoint replaces the need to have torchvision installed on the host.
    The notebook calls this endpoint so the containers handle all dataset I/O.
    """
    import threading as _threading

    with _data_prepare_lock:
        if _data_prepare_status["running"]:
            return {"status": "already_running", **_data_prepare_status}
        _data_prepare_status.update({
            "running": True, "done": [], "errors": [], "current": None
        })

    def _worker(datasets_list: list[str]) -> None:
        # Use the same path that image_utils.load_dataset() uses
        # DATA_DIR in image_utils.py is "/ml-data/datasets"
        from app.core.image_utils import DATA_DIR
        import torchvision.datasets as _ds

        datasets_dir = Path(DATA_DIR)
        datasets_dir.mkdir(parents=True, exist_ok=True)

        _cls_map = {
            "mnist":   _ds.MNIST,
            "fashion": _ds.FashionMNIST,
            "cifar10": _ds.CIFAR10,

        }

        for name in datasets_list:
            with _data_prepare_lock:
                _data_prepare_status["current"] = name
            try:
                cls = _cls_map[name]
                cls(root=str(datasets_dir), train=True, download=True)
                cls(root=str(datasets_dir), train=False, download=True)
                with _data_prepare_lock:
                    _data_prepare_status["done"].append(name)
            except Exception as exc:
                with _data_prepare_lock:
                    _data_prepare_status["errors"].append({"dataset": name, "error": str(exc)})

        with _data_prepare_lock:
            _data_prepare_status["running"] = False
            _data_prepare_status["current"] = None

    t = _threading.Thread(target=_worker, args=(req.datasets,), daemon=True)
    t.start()
    return {"status": "started", "datasets": req.datasets}


@app.get("/data/prepare/status")
def data_prepare_status():
    """Return the current status of the dataset preparation job."""
    with _data_prepare_lock:
        return {**_data_prepare_status}


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
