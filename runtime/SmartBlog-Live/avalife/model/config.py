from __future__ import annotations

import os
from pathlib import Path


def runtime_state_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    runtime_dir = Path(os.getenv("WORKER_RUN_STATE_DIR", str(root / "runtime"))).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def model_runtime_socket_path() -> str:
    return str(Path(os.getenv("WORKER_MODEL_SOCKET", str(runtime_state_dir() / "smartblog-live-modeld.sock"))).resolve())
