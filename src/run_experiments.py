"""
Executor automatizado de experimentos para o paper.
Roda o treinamento federado em 4 cenários de caos e exporta resultados.

Uso (dentro do container ou local):
    python run_experiments.py

Pré-requisito: Docker Compose rodando (server + 3 clients + chaos-injector).
Este script controla os experimentos via status.json e chaos_config.txt.
"""

import os
import sys
import time
import json
import sqlite3
import csv
import shutil
import requests
from datetime import datetime

CHAOS_SCENARIOS = {
    "Normal":   {"loss": 0.00, "delay": 0,    "corrupt": 0.00, "duplicate": 0.00},
    "Leve":     {"loss": 1.00, "delay": 200,   "corrupt": 0.00, "duplicate": 0.00},
    "Moderado": {"loss": 3.00, "delay": 500,   "corrupt": 0.50, "duplicate": 1.00},
    "Severo":   {"loss": 5.00, "delay": 1000,  "corrupt": 2.00, "duplicate": 5.00},
}

STATUS_FILE = "status.json"
CHAOS_CONFIG = "chaos_config.txt"
DB_FILE = "metrics.db"
RESULTS_DIR = "results"
TARGET_ROUNDS = int(os.environ.get("EXPERIMENT_ROUNDS", "30"))
POLL_INTERVAL = 3  # seconds
SERVER_URL = os.environ.get("SERVER_URL", "http://fl-server:5000")


def set_status(s):
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": s}, f)


def apply_chaos(scenario_name):
    params = CHAOS_SCENARIOS[scenario_name]
    if scenario_name == "Normal":
        line = f"OFF 0 0 0 0"
    else:
        line = f"ON {params['loss']:.2f} {params['delay']} {params['corrupt']:.2f} {params['duplicate']:.2f}"
    with open(CHAOS_CONFIG, "w") as f:
        f.write(line)


def reset_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS training_logs (timestamp TEXT, node_id TEXT, bytes_sent INTEGER, loss REAL, round_number INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS round_metrics (round_number INTEGER, global_mse REAL, global_psnr REAL, timestamp TEXT, chaos_scenario TEXT)")
    conn.commit()
    conn.close()


def get_current_round():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        cursor = conn.execute("SELECT MAX(round_number) FROM round_metrics")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row[0] is not None else 0
    except Exception:
        return 0


def export_results(scenario_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=10)

    # Export round_metrics
    df = conn.execute("SELECT * FROM round_metrics ORDER BY round_number").fetchall()
    path_rounds = os.path.join(RESULTS_DIR, f"{scenario_name}_round_metrics.csv")
    with open(path_rounds, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round_number", "global_mse", "global_psnr", "timestamp", "chaos_scenario"])
        writer.writerows(df)

    # Export training_logs
    df2 = conn.execute("SELECT * FROM training_logs ORDER BY timestamp").fetchall()
    path_logs = os.path.join(RESULTS_DIR, f"{scenario_name}_training_logs.csv")
    with open(path_logs, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "node_id", "bytes_sent", "loss", "round_number"])
        writer.writerows(df2)

    conn.close()
    print(f"  📁 Exportado: {path_rounds}")
    print(f"  📁 Exportado: {path_logs}")


def remove_global_model():
    if os.path.exists("global_model.pth"):
        os.remove("global_model.pth")


def reset_server():
    """Reseta o round counter do servidor via API"""
    try:
        requests.post(f"{SERVER_URL}/reset_round", timeout=5)
        print("  🔄 Servidor resetado.")
    except Exception:
        print("  ⚠️ Não foi possível resetar servidor (pode não estar acessível).")


def wait_for_rounds(target):
    print(f"  ⏳ Aguardando {target} rodadas...")
    start = time.time()
    while True:
        current = get_current_round()
        elapsed = time.time() - start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\r  Rodada {current}/{target} ({mins}m{secs:02d}s)", end="", flush=True)
        if current >= target:
            print()
            return
        time.sleep(POLL_INTERVAL)


def run_scenario(scenario_name):
    print(f"\n{'='*60}")
    print(f"🔬 CENÁRIO: {scenario_name}")
    print(f"{'='*60}")
    params = CHAOS_SCENARIOS[scenario_name]
    print(f"  Params: Loss={params['loss']}% | Delay={params['delay']}ms | "
          f"Corrupt={params['corrupt']}% | Dup={params['duplicate']}%")

    # 1. Pausar e resetar
    set_status("PAUSED")
    time.sleep(2)
    reset_db()
    remove_global_model()
    reset_server()

    # 2. Aplicar cenário de caos
    apply_chaos(scenario_name)
    print(f"  ⚡ Caos aplicado: {scenario_name}")
    time.sleep(2)

    # 3. Iniciar treinamento
    set_status("RUNNING")
    print(f"  ▶️ Treinamento iniciado...")

    # 4. Aguardar N rodadas
    wait_for_rounds(TARGET_ROUNDS)

    # 5. Pausar
    set_status("PAUSED")
    print(f"  ⏸️ Treinamento pausado.")

    # 6. Exportar resultados
    export_results(scenario_name)

    # 7. Salvar modelo deste cenário
    if os.path.exists("global_model.pth"):
        model_path = os.path.join(RESULTS_DIR, f"{scenario_name}_model.pth")
        shutil.copy("global_model.pth", model_path)
        print(f"  💾 Modelo salvo: {model_path}")

    print(f"  ✅ Cenário {scenario_name} concluído!")


def main():
    print("=" * 60)
    print("🧪 EXECUTOR DE EXPERIMENTOS - Federated Learning")
    print(f"   Cenários: {list(CHAOS_SCENARIOS.keys())}")
    print(f"   Rodadas por cenário: {TARGET_ROUNDS}")
    print("=" * 60)

    start_time = time.time()

    for scenario in CHAOS_SCENARIOS:
        run_scenario(scenario)

    # Restaurar estado normal e melhor modelo
    apply_chaos("Normal")
    set_status("PAUSED")

    # Restaurar modelo do cenário Normal (melhor qualidade)
    normal_model = os.path.join(RESULTS_DIR, "Normal_model.pth")
    if os.path.exists(normal_model):
        shutil.copy(normal_model, "global_model.pth")
        print("\n💾 Modelo Normal restaurado como global_model.pth")

    total_time = time.time() - start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    print(f"\n{'='*60}")
    print(f"🏁 TODOS OS EXPERIMENTOS CONCLUÍDOS! ({mins}m{secs:02d}s)")
    print(f"   Resultados em: {os.path.abspath(RESULTS_DIR)}/")
    print(f"   Próximo passo: python generate_figures.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
