from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(float(str(os.getenv(name, str(default)) or str(default)).strip()))
    except Exception:
        value = int(default)
    return int(max(int(low), min(int(high), int(value))))


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        value = float(default)
    return float(max(float(low), min(float(high), float(value))))


def _retry_delay(attempt_index: int, *, env_prefix: str) -> float:
    base = _env_float(f"{env_prefix}_BACKOFF_SEC", 2.0, low=0.1, high=120.0)
    cap = _env_float(f"{env_prefix}_BACKOFF_MAX_SEC", 45.0, low=0.1, high=300.0)
    delay = min(float(cap), float(base) * (2 ** max(0, int(attempt_index) - 1)))
    jitter = random.uniform(0.0, min(1.0, delay * 0.2))
    return float(delay + jitter)


class _ProgressReader:
    def __init__(self, file_obj: Any, progress_state: dict[str, Any] | None) -> None:
        self.file_obj = file_obj
        self.progress_state = progress_state

    def read(self, size: int = -1) -> bytes:
        data = self.file_obj.read(size)
        if isinstance(self.progress_state, dict):
            try:
                self.progress_state["uploaded"] = int(self.file_obj.tell())
            except Exception:
                pass
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file_obj, name)


def put_file_to_signed_url(
    *,
    signed_url: str,
    path: str,
    content_type: str,
    connect_timeout: float = 20.0,
    read_timeout: float = 1800.0,
    progress_state: dict[str, Any] | None = None,
    env_prefix: str = "SMARTBLOG_SIGNED_UPLOAD",
    log_prefix: str = "signed-upload",
) -> dict[str, Any]:
    """PUT a local file to a signed URL with retry for transient storage errors.

    Supabase Storage occasionally returns transient 502/503/504 responses for
    large uploads. Retrying the same signed URL is safe for our generated upload
    URLs because they are minted with upsert enabled.
    """

    import requests

    url = str(signed_url or "").strip()
    if not url:
        raise RuntimeError("upload.signed_url is required")
    src = Path(str(path or "")).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"upload source file missing: {src}")

    size = int(src.stat().st_size)
    attempts = _env_int(f"{env_prefix}_RETRIES", 5, low=1, high=20)
    connect = max(1.0, float(connect_timeout))
    read = max(1.0, float(read_timeout))
    last_error = ""
    started = time.perf_counter()

    if isinstance(progress_state, dict):
        progress_state["size"] = int(size)
        progress_state["uploaded"] = 0
        progress_state["attempt"] = 0
        progress_state["attempts"] = int(attempts)

    for attempt in range(1, int(attempts) + 1):
        if isinstance(progress_state, dict):
            progress_state["attempt"] = int(attempt)
            progress_state["uploaded"] = 0
        try:
            with src.open("rb") as f:
                resp = requests.put(
                    url,
                    data=_ProgressReader(f, progress_state) if isinstance(progress_state, dict) else f,
                    headers={
                        "Content-Type": str(content_type or "application/octet-stream"),
                        "Content-Length": str(int(size)),
                    },
                    timeout=(float(connect), float(read)),
                )
            status = int(resp.status_code)
            if 200 <= status < 300:
                if isinstance(progress_state, dict):
                    progress_state["uploaded"] = int(size)
                    progress_state["status_code"] = int(status)
                return {
                    "uploaded": True,
                    "bytes": int(size),
                    "status_code": int(status),
                    "attempts": int(attempt),
                    "upload_sec": float(time.perf_counter() - float(started)),
                }
            body = str(getattr(resp, "text", "") or "")[-2000:]
            last_error = f"HTTP {status}: {body}"
            retryable = status in _RETRYABLE_STATUS
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            retryable = True

        if not retryable or attempt >= attempts:
            break
        delay = _retry_delay(int(attempt), env_prefix=str(env_prefix))
        logging.warning(
            "%s retrying signed upload attempt=%d/%d delay=%.2fs size=%d error=%s",
            str(log_prefix or "signed-upload"),
            int(attempt),
            int(attempts),
            float(delay),
            int(size),
            str(last_error)[-500:],
        )
        time.sleep(float(delay))

    raise RuntimeError(f"signed upload failed after {attempts} attempt(s): {last_error}")
