from flask import Flask, request, jsonify
import torch
import numpy as np
import sqlite3
import os
import time
from datetime import datetime
from config import REQUIRED_CLIENTS, CHAOS_SCENARIO, TEST_BATCH_SIZE, ROUND_TIMEOUT, MODEL_TYPE, LATENT_DIM, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE, TEXT_RECONSTRUCTION_MODE
from text_utils import decode_tokens

app = Flask(__name__)

# Buffer de pesos: dicionário garante apenas o último envio de cada cliente
round_buffer = {}
current_round = 0
round_start_time = None

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

def log_terminal(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


def load_checkpoint_if_compatible(model, path="global_model.pth"):
    if not os.path.exists(path):
        return False
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return True
    except Exception as exc:
        log_terminal(f"⚠️ Checkpoint incompatível ignorado: {exc}")
        return False


def get_text_stack():
    from model_utils import get_model
    from text_utils import build_or_load_vocab, MAX_VOCAB_SIZE, MAX_SEQ_LEN

    vocab = build_or_load_vocab()
    model = get_model(MODEL_TYPE, vocab_size=MAX_VOCAB_SIZE, seq_len=MAX_SEQ_LEN, latent_dim=LATENT_DIM)
    load_checkpoint_if_compatible(model)
    model.eval()
    return model, vocab, MAX_SEQ_LEN


def decode_chunk_latent(model, vocab, latent_tensor):
    with torch.no_grad():
        logits = model.decode(latent_tensor)
        predictions = torch.argmax(logits, dim=-1)
    from text_utils import decode_tokens

    return predictions.squeeze(0), decode_tokens(predictions.squeeze(0).tolist(), vocab)


def encode_text_chunks(model, vocab, text, chunk_size=None, overlap=None, mode=None):
    from text_utils import decode_tokens, stitch_text_chunks, text_to_chunk_tensors

    chunk_size = chunk_size or TEXT_CHUNK_SIZE
    overlap = TEXT_CHUNK_OVERLAP if overlap is None else overlap
    mode = mode or TEXT_RECONSTRUCTION_MODE

    chunk_tensors, chunk_ids = text_to_chunk_tensors(text, vocab, chunk_size=chunk_size, overlap=overlap)
    payload_chunks = []
    reconstructed_chunks = []

    with torch.no_grad():
        for chunk_id, (chunk_tensor, token_ids) in enumerate(zip(chunk_tensors, chunk_ids)):
            batch = chunk_tensor.unsqueeze(0)
            if MODEL_TYPE == "vae":
                mu, logvar = model.encode(batch)
                latent = mu if mode == "semantic" else model.reparameterize(mu, logvar)
            else:
                latent = model.encode(batch)

            predictions, reconstructed_text = decode_chunk_latent(model, vocab, latent)
            payload_chunks.append({
                "chunk_id": chunk_id,
                "source_tokens": token_ids,
                "source_text": decode_tokens(token_ids, vocab),
                "latent": latent.squeeze(0).tolist(),
                "reconstructed_tokens": predictions.tolist(),
                "reconstructed_text": reconstructed_text,
            })
            reconstructed_chunks.append(reconstructed_text)

    return payload_chunks, stitch_text_chunks(reconstructed_chunks)


def generate_from_payload(model, vocab, chunks):
    from text_utils import stitch_text_chunks

    reconstructed_chunks = []
    normalized_chunks = []
    for index, chunk in enumerate(chunks):
        latent_values = chunk.get("latent") if isinstance(chunk, dict) else chunk
        latent_tensor = torch.tensor(latent_values, dtype=torch.float32).unsqueeze(0)
        predictions, reconstructed_text = decode_chunk_latent(model, vocab, latent_tensor)
        normalized_chunks.append({
            "chunk_id": chunk.get("chunk_id", index) if isinstance(chunk, dict) else index,
            "latent": latent_values,
            "reconstructed_tokens": predictions.tolist(),
            "reconstructed_text": reconstructed_text,
        })
        reconstructed_chunks.append(reconstructed_text)

    normalized_chunks.sort(key=lambda item: item["chunk_id"])
    return normalized_chunks, stitch_text_chunks([chunk["reconstructed_text"] for chunk in normalized_chunks])

def reconstruct_weights(partial_weights, node_id):
    """Reconstrói pesos comprimidos (Top-K) substituindo NaN pela média"""
    reconstructed = {}
    missing_count = 0
    total_count = 0
    
    for key, value_list in partial_weights.items():
        arr = np.array(value_list, dtype=np.float32)
        total_count += arr.size
        
        if arr.dtype == object:
            arr = arr.astype(float)
            
        nans = np.isnan(arr)
        missing = np.sum(nans)
        missing_count += missing
        
        if np.isnan(arr).all():
            col_mean = 0.0
        else:
            col_mean = np.nanmean(arr)
        
        arr[nans] = col_mean
        reconstructed[key] = arr
        
    if missing_count > 0:
        percent = (missing_count / total_count) * 100
        log_terminal(f"🔧 Reconstruindo pesos de {node_id}: {percent:.1f}% faltante preenchido por média")

    return reconstructed

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    global round_start_time
    try:
        data = request.json
        node_id = data.get('node_id', 'unknown')
        loss = data.get('loss', 0)
        
        if node_id in round_buffer:
            log_terminal(f"♻️  {node_id} reenviou. Atualizando buffer...")
        else:
            log_terminal(f"📥 Recebido de: {node_id} | Aguardando parceiros... ({len(round_buffer)+1}/{REQUIRED_CLIENTS})")

        # Marca início da rodada quando primeiro cliente chega
        if not round_buffer:
            round_start_time = time.time()

        full_weights_reconstructed = reconstruct_weights(data['weights'], node_id)
        round_buffer[node_id] = full_weights_reconstructed
        
        log_to_db(node_id, len(str(data['weights'])), loss, current_round)
        
        # Agrega se todos chegaram OU se timeout expirou com pelo menos 1 cliente
        should_aggregate = len(round_buffer) >= REQUIRED_CLIENTS
        if not should_aggregate and round_start_time and len(round_buffer) >= 1:
            elapsed = time.time() - round_start_time
            if elapsed > ROUND_TIMEOUT:
                log_terminal(f"⏱️ Timeout ({elapsed:.0f}s) com {len(round_buffer)}/{REQUIRED_CLIENTS} clientes. Agregando parcial.")
                should_aggregate = True

        if should_aggregate:
            round_start_time = None
            aggregate_weights()
            
        return jsonify({"status": "accepted", "round": current_round}), 200
    except Exception as e:
        log_terminal(f"❌ Erro no server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def evaluate_global_model():
    """Avalia o modelo global em batches de textos de teste e retorna CE Loss e Acurácia"""
    try:
        from model_utils import get_model
        from text_utils import load_text_dataset, get_random_batch_text, compute_accuracy, compute_cross_entropy, MAX_VOCAB_SIZE, MAX_SEQ_LEN
        
        model = get_model(MODEL_TYPE, vocab_size=MAX_VOCAB_SIZE, seq_len=MAX_SEQ_LEN, latent_dim=LATENT_DIM)
        load_checkpoint_if_compatible(model)
        model.eval()
        
        dataset, _ = load_text_dataset(train=False)
        inputs, targets = get_random_batch_text(dataset, batch_size=TEST_BATCH_SIZE)
        
        with torch.no_grad():
            output = model(inputs)
            if isinstance(output, tuple):  # VAE retorna (logits, mu, logvar)
                logits = output[0]
            else:
                logits = output
        
        accuracy = compute_accuracy(targets, logits)
        ce_loss = compute_cross_entropy(targets, logits)
        
        return ce_loss, accuracy * 100.0
    except Exception as e:
        log_terminal(f"⚠️ Erro na avaliação: {e}")
        return None, None

def aggregate_weights():
    """FedAvg: média dos pesos de todos os clientes"""
    global round_buffer, current_round
    current_round += 1
    log_terminal(f"✨ Todos ({len(round_buffer)}) chegaram! Iniciando FedAvg... [Rodada {current_round}]")
    
    try:
        new_state_dict = {}
        first_client = next(iter(round_buffer.values()))
        keys = first_client.keys()
        
        for key in keys:
            tensors = [torch.tensor(client_weights[key]) for client_weights in round_buffer.values()]
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        
        torch.save(new_state_dict, "global_model.pth")
        
        log_terminal(f"💾 Modelo Global salvo! [Rodada {current_round}]")
        
        # Avaliação do modelo global
        ce_loss, acc = evaluate_global_model()
        if ce_loss is not None:
            log_terminal(f"📊 Avaliação Global: CE Loss={ce_loss:.4f} | Accuracy={acc:.1f}%")
            log_round_metrics(current_round, ce_loss, acc)
        
        round_buffer.clear()
        log_terminal(f"🏁 Rodada {current_round} finalizada. Buffer limpo.\n")
        
    except Exception as e:
        log_terminal(f"❌ Erro na agregação: {e}")

@app.route('/reconstruct', methods=['POST'])
def reconstruct_text():
    """
    Endpoint de comunicação semântica para texto:
    Recebe vetor latente (ex: 32 dims) e retorna a sequência de tokens reconstruída.
    Simula: Dispositivo A envia representação comprimida → Dispositivo B reconstrói.
    """
    try:
        data = request.json
        model, vocab, _ = get_text_stack()
        latent = torch.tensor(data['latent'], dtype=torch.float32).unsqueeze(0)
        predictions, reconstructed_text = decode_chunk_latent(model, vocab, latent)
        
        return jsonify({
            "status": "ok",
            "tokens": predictions.tolist(),
            "text": reconstructed_text,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/complete', methods=['POST'])
def complete_text():
    """
    Endpoint de completação de texto:
    Recebe sequência parcial (mascarada com pads) → Autoencoder reconstrói sequência completa.
    """
    try:
        data = request.json
        model, vocab, _ = get_text_stack()
        partial = torch.tensor(data['tokens'], dtype=torch.long)
        if partial.dim() == 1:
            partial = partial.unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            output = model(partial)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            predictions = torch.argmax(logits, dim=-1)
        
        return jsonify({
            "status": "ok",
            "tokens": predictions.squeeze().numpy().tolist(),
            "text": decode_tokens(predictions.squeeze().tolist(), vocab),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/compress_text', methods=['POST'])
def compress_text():
    try:
        data = request.json or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({"error": "Campo 'text' é obrigatório."}), 400

        model, vocab, max_seq_len = get_text_stack()
        chunk_size = min(int(data.get('chunk_size', TEXT_CHUNK_SIZE)), max_seq_len)
        overlap = min(int(data.get('overlap', TEXT_CHUNK_OVERLAP)), max(0, chunk_size - 1))
        mode = data.get('mode', TEXT_RECONSTRUCTION_MODE)
        chunks, reconstructed_text = encode_text_chunks(model, vocab, text, chunk_size=chunk_size, overlap=overlap, mode=mode)

        return jsonify({
            "status": "ok",
            "mode": mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "reconstructed_text": reconstructed_text,
        }), 200
    except Exception as e:
        log_terminal(f"❌ Erro em /compress_text: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        data = request.json or {}
        chunks = data.get('chunks') or data.get('latents')
        if not chunks:
            return jsonify({"error": "Envie 'chunks' ou 'latents'."}), 400

        model, vocab, _ = get_text_stack()
        normalized_chunks, reconstructed_text = generate_from_payload(model, vocab, chunks)

        return jsonify({
            "status": "ok",
            "num_chunks": len(normalized_chunks),
            "chunks": normalized_chunks,
            "text": reconstructed_text,
        }), 200
    except Exception as e:
        log_terminal(f"❌ Erro em /generate_text: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/complete_text', methods=['POST'])
def complete_document_text():
    try:
        from text_utils import build_masked_document, decode_tokens, stitch_text_chunks

        data = request.json or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({"error": "Campo 'text' é obrigatório."}), 400

        strategy = data.get('strategy', 'truncate')
        mask_ratio = float(data.get('mask_ratio', 0.5))
        model, vocab, max_seq_len = get_text_stack()
        chunk_size = min(int(data.get('chunk_size', TEXT_CHUNK_SIZE)), max_seq_len)
        overlap = min(int(data.get('overlap', TEXT_CHUNK_OVERLAP)), max(0, chunk_size - 1))

        masked_tensors, masked_text = build_masked_document(text, vocab, strategy=strategy, mask_ratio=mask_ratio, chunk_size=chunk_size, overlap=overlap)

        completed_chunks = []
        completed_texts = []
        with torch.no_grad():
            for chunk_id, tensor in enumerate(masked_tensors):
                output = model(tensor.unsqueeze(0))
                logits = output[0] if isinstance(output, tuple) else output
                predictions = torch.argmax(logits, dim=-1).squeeze(0)
                completed_text = decode_tokens(predictions.tolist(), vocab)
                completed_chunks.append({
                    "chunk_id": chunk_id,
                    "masked_tokens": tensor.tolist(),
                    "completed_tokens": predictions.tolist(),
                    "completed_text": completed_text,
                })
                completed_texts.append(completed_text)

        return jsonify({
            "status": "ok",
            "strategy": strategy,
            "mask_ratio": mask_ratio,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "masked_text": masked_text,
            "completed_text": stitch_text_chunks(completed_texts),
            "chunks": completed_chunks,
        }), 200
    except Exception as e:
        log_terminal(f"❌ Erro em /complete_text: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_round', methods=['POST'])
def reset_round():
    global round_buffer, current_round, round_start_time
    round_buffer = {}
    current_round = 0
    round_start_time = None
    log_terminal("🔄 Round counter e buffer resetados.")
    return jsonify({"status": "reset", "round": 0}), 200

def log_to_db(node_id, bytes_sent, loss, round_number):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO training_logs VALUES (?, ?, ?, ?, ?)", 
                     (timestamp, node_id, bytes_sent, loss, round_number))
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar no DB: {e}")

def log_round_metrics(round_number, global_celoss, global_accuracy):
    try:
        conn = sqlite3.connect('metrics.db', timeout=10)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scenario = os.environ.get("CHAOS_SCENARIO", CHAOS_SCENARIO)
        cursor.execute(
            "INSERT INTO round_metrics (round_number, global_celoss, global_accuracy, timestamp, chaos_scenario) VALUES (?, ?, ?, ?, ?)",
            (round_number, global_celoss, global_accuracy, timestamp, scenario),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log_terminal(f"⚠️ Erro ao salvar round_metrics: {e}")

if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f: f.write("=== SERVIDOR FL INICIADO ===\n")
    app.run(host='0.0.0.0', port=5000)
