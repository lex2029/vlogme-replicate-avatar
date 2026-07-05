from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_ROOT_DIR = _THIS_FILE.parents[2]
_LOG_DIR = Path(os.getenv("WORKER_FRONTEND_LOG_DIR", str(_ROOT_DIR / "logs" / "frontend"))).resolve()
_RUN_STATE_DIR = Path(os.getenv("WORKER_RUN_STATE_DIR", str(_ROOT_DIR / "runtime"))).resolve()
_ACTIVE_LOG_FILE = Path(
    os.getenv("WORKER_ACTIVE_FRONTEND_LOG_FILE", str(_RUN_STATE_DIR / "current_frontend_log.txt"))
).resolve()
_ACTIVE_LOG_LINK = Path(
    os.getenv("WORKER_ACTIVE_FRONTEND_LOG_LINK", str(_LOG_DIR / "current.log"))
).resolve()


def install_frontend_file_logging() -> str:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = (_LOG_DIR / f"frontend_{ts}.log").resolve()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.getLogger().level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)

    tmp_file = _ACTIVE_LOG_FILE.with_suffix(".tmp")
    tmp_file.write_text(f"{log_path}\n", encoding="utf-8")
    os.replace(tmp_file, _ACTIVE_LOG_FILE)

    try:
        if _ACTIVE_LOG_LINK.is_symlink() or _ACTIVE_LOG_LINK.exists():
            _ACTIVE_LOG_LINK.unlink()
    except FileNotFoundError:
        pass
    _ACTIVE_LOG_LINK.symlink_to(log_path)
    return str(log_path)
