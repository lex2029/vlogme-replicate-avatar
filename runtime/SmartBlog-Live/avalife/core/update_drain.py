from __future__ import annotations

import json
import os
import time
from typing import Any


_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auto_update_drain_file() -> str:
    path = str(
        os.getenv("SMARTBLOG_AUTO_UPDATE_DRAIN_FILE")
        or os.getenv("WORKER_AUTO_UPDATE_DRAIN_FILE")
        or ""
    ).strip()
    if path:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_WORKSPACE_ROOT, "runtime", "auto_update_drain.json"))


def auto_update_drain_state() -> dict[str, Any]:
    path = auto_update_drain_file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    state = dict(data) if isinstance(data, dict) else {}
    state.setdefault("path", path)
    try:
        state.setdefault("age_sec", max(0.0, time.time() - float(os.path.getmtime(path))))
    except Exception:
        pass
    return state


def auto_update_drain_requested() -> bool:
    return os.path.exists(auto_update_drain_file())
