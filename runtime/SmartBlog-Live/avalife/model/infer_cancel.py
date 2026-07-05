from __future__ import annotations

import json
import os
import time


class InferenceCancelled(RuntimeError):
    pass


def infer_cancel_path() -> str:
    raw = str(os.getenv("MODEL_INFER_CANCEL_PATH", "") or "").strip()
    if raw:
        return os.path.abspath(raw)
    runtime_dir = str(os.getenv("WORKER_RUNTIME_DIR", "") or "").strip()
    if not runtime_dir:
        root_dir = str(os.getenv("ROOT_DIR", os.getcwd()) or os.getcwd()).strip()
        runtime_dir = os.path.join(root_dir, "runtime")
    return os.path.abspath(os.path.join(runtime_dir, "model_infer_cancel.json"))


def clear_infer_cancel() -> None:
    try:
        os.remove(infer_cancel_path())
    except FileNotFoundError:
        pass
    except Exception:
        pass


def request_infer_cancel(*, job_id: str | None = None, reason: str = "") -> None:
    path = infer_cancel_path()
    payload = {
        "job_id": str(job_id or "").strip(),
        "reason": str(reason or "").strip() or "cancelled",
        "requested_at": float(time.time()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    os.replace(tmp, path)


def infer_cancel_requested(*, job_id: str | None = None) -> tuple[bool, str]:
    path = infer_cancel_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return False, ""
    except Exception:
        return False, ""
    if not isinstance(payload, dict):
        return False, ""
    requested_job = str(payload.get("job_id") or "").strip()
    current_job = str(job_id or "").strip()
    if requested_job and current_job and requested_job != current_job:
        return False, ""
    reason = str(payload.get("reason") or "cancelled").strip() or "cancelled"
    return True, reason


def raise_if_infer_cancelled(*, job_id: str | None = None) -> None:
    cancelled, reason = infer_cancel_requested(job_id=job_id)
    if cancelled:
        raise InferenceCancelled(reason)
