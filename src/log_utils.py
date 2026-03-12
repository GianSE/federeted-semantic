import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def log_terminal(msg: str, node_id: str = "server", log_dir: str = LOG_DIR) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    log_file = os.path.join(log_dir, f"{node_id}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")
