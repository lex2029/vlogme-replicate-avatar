from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path


DEFAULT_COOLDOWN_SEC = 60 * 60
DEFAULT_LOCK_PATH = "/tmp/vlogme-replicate-avatar-generation-cooldown.lock"
DEFAULT_STATE_PATH = "/tmp/vlogme-replicate-avatar-generation-cooldown.json"


def _cooldown_seconds() -> int:
    raw = os.environ.get("VLOGME_REPLICATE_GENERATION_COOLDOWN_SEC", str(DEFAULT_COOLDOWN_SEC))
    try:
        return max(0, int(float(raw)))
    except Exception:
        return DEFAULT_COOLDOWN_SEC


def _retry_message(remaining_sec: float) -> str:
    minutes = max(1, int((remaining_sec + 59) // 60))
    unit = "minute" if minutes == 1 else "minutes"
    return f"We are temporarily overloaded. Please try again in about {minutes} {unit}."


def _read_last_started_at(path: Path) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    try:
        return float(payload.get("last_started_at") or 0.0)
    except Exception:
        return 0.0


def reserve_generation_slot(*, now: float | None = None) -> None:
    cooldown_sec = _cooldown_seconds()
    if cooldown_sec <= 0:
        return

    now_sec = float(time.time() if now is None else now)
    lock_path = Path(os.environ.get("VLOGME_REPLICATE_GENERATION_COOLDOWN_LOCK", DEFAULT_LOCK_PATH))
    state_path = Path(os.environ.get("VLOGME_REPLICATE_GENERATION_COOLDOWN_STATE", DEFAULT_STATE_PATH))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        last_started_at = _read_last_started_at(state_path)
        remaining_sec = float(cooldown_sec) - (now_sec - last_started_at)
        if last_started_at > 0 and remaining_sec > 0:
            raise RuntimeError(_retry_message(remaining_sec))
        state_path.write_text(
            json.dumps({"last_started_at": now_sec}, separators=(",", ":")),
            encoding="utf-8",
        )
