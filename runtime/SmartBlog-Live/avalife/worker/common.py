# Copyright 2026
#
# SmartBlog Live frontend runtime: listens for tasks, claims them, connects to
# LiveKit when needed, and streams continuous A/V.
# Uses Wan S2V plus TTS to generate the avatar stream.
#
# IMPORTANT:
# - Frontend and model runtime are now split. This frontend process does not own model weights.
# - Secrets are loaded from `config/worker_secrets.conf` by launch scripts.

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import signal
import select
import time
import urllib.parse
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import torch

from avalife.core.observability import worker_log_level

# Make the repo root importable when launched outside the checkout root.
_THIS_FILE = os.path.abspath(__file__)
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.append(_WORKSPACE_ROOT)
_LIVEKIT_RTC: Any | None = None


def _livekit_rtc() -> Any:
    global _LIVEKIT_RTC
    if _LIVEKIT_RTC is not None:
        return _LIVEKIT_RTC
    try:
        from livekit import rtc as rtc_module
    except Exception as e:
        raise RuntimeError("LiveKit helpers require the livekit package") from e
    _LIVEKIT_RTC = rtc_module
    return rtc_module

# Compile cache defaults for faster cold starts/restarts.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
# Streaming distributed runtime is more stable without inductor cudagraph capture.
# Keep overridable via env, but default to disabled.
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

# Fixed-shape realtime inference benefits from cudnn autotune and TF32 kernels.
# Safe for this worker: model path is deterministic and quality-critical tensors
# remain in bf16 where configured by model/pipeline.
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass
try:
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

def _required_positive_int_env(name: str) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        value = int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from e
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return int(value)


def _required_nonnegative_int_env(name: str) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        value = int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from e
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0, got {value}")
    return int(value)


def _required_positive_float_env(name: str) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        value = float(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid float env {name}={raw!r}") from e
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return float(value)


def _required_nonnegative_float_env(name: str) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        value = float(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid float env {name}={raw!r}") from e
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0, got {value}")
    return float(value)


def _required_str_env(name: str, *, allow_empty: bool = False) -> str:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw and not bool(allow_empty):
        raise RuntimeError(f"Missing required env: {name}")
    return raw


# Single source of truth for FPS in worker runtime.
# This value is propagated to both model (sample_fps) and LiveKit publish rate.
WORKER_FPS = int(_required_positive_int_env("WORKER_FPS"))

# Product denoise quality range. API payloads may choose a value inside this
# range per job; launch defaults are normalized into the same range.
SMARTBLOG_MIN_SAMPLE_STEPS = 4
SMARTBLOG_MAX_SAMPLE_STEPS = 12


def smartblog_clamp_sample_steps(value: Any) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(SMARTBLOG_MIN_SAMPLE_STEPS)
    return int(max(int(SMARTBLOG_MIN_SAMPLE_STEPS), min(int(parsed), int(SMARTBLOG_MAX_SAMPLE_STEPS))))

AVALIFE_BASE_URL = os.getenv("AVALIFE_BASE_URL", "https://api.avalife.ai/v1").rstrip("/")
AVALIFE_SUPABASE_URL = str(os.getenv("AVALIFE_SUPABASE_URL", "") or "").strip()
AVALIFE_SUPABASE_ANON_KEY = str(os.getenv("AVALIFE_SUPABASE_ANON_KEY", "") or "").strip()


def _setup_logging() -> None:
    lvl = int(worker_log_level())
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


_setup_logging()


# Single source for denoise steps (set via config/*.conf).
FORCED_SAMPLE_STEPS = int(smartblog_clamp_sample_steps(_required_positive_int_env("WORKER_SAMPLE_STEPS")))
# Live replies use the same checked-in denoise quality as the rest of the
# product runtime. Keep the alias so the call sites stay explicit.
FORCED_REPLY_SAMPLE_STEPS = int(FORCED_SAMPLE_STEPS)

# Fixed reply/video generation granularity from the checked-in worker config.
# `INFER_FRAMES` is sourced by the shell launchers before Python starts, so this
# stays stable for the whole process while still honoring branch-level A/B tests.
LOCKED_INFER_FRAMES = int(_required_positive_int_env("INFER_FRAMES"))
# liveaudio micro-chunk in samples. Source it from the checked-in runtime config
# so A/B tests can align audio chunk duration with the target infer window while
# keeping WORKER_AUDIO_SAMPLE_RATE as the actual audio clock.
LOCKED_MICRO_CHUNK_SAMPLES = int(_required_positive_int_env("WORKER_LIVEAUDIO_MICRO_CHUNK_SCHEDULE_SAMPLES"))
LOCKED_MAX_PENDING_CLIPS = max(1, min(64, int(_required_positive_int_env("LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS"))))


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def _worker_secret_namespace(worker: Any | None = None) -> str:
    override = str(getattr(worker, "_worker_secret_namespace_override", "") or "").strip().upper() if worker is not None else ""
    if override in {"AVALIFE", "SMARTBLOG"}:
        return override
    source = str(getattr(worker, "_active_source", "") or "").strip().lower() if worker is not None else ""
    if source == "smartblog":
        return "SMARTBLOG"
    if source == "avalife":
        return "AVALIFE"
    control_plane = str(os.getenv("WORKER_CONTROL_PLANE", "avalife") or "avalife").strip().lower()
    if control_plane == "smartblog":
        return "SMARTBLOG"
    return "AVALIFE"


def _worker_secret_env(worker: Any | None, suffix: str) -> str:
    key = f"{_worker_secret_namespace(worker)}_{str(suffix or '').strip()}"
    return str(os.getenv(key, "") or "").strip()


async def run_supervised_loop(
    stop_event: asyncio.Event,
    *,
    name: str,
    factory: Any,
    restart_delay_sec: float = 1.0,
) -> None:
    delay = max(0.1, min(30.0, float(restart_delay_sec or 1.0)))
    loop_name = str(name or "background-loop").strip() or "background-loop"
    while not bool(stop_event.is_set()):
        try:
            await factory()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.warning("%s crashed; restarting in %.1fs: %s", loop_name, delay, e)
        else:
            if bool(stop_event.is_set()):
                break
            logging.warning("%s exited unexpectedly; restarting in %.1fs", loop_name, delay)
        if bool(stop_event.is_set()):
            break
        await asyncio.sleep(delay)


def _safe_json_loads(b: bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads((b or b"").decode("utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _safe_decode_utf8(b: bytes) -> str:
    try:
        return (b or b"").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _string_payload_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    s = str(value).strip()
    return s if s else ""


def _task_payload(task: dict[str, Any] | None) -> dict[str, Any]:
    task_obj = task if isinstance(task, dict) else {}
    payload = task_obj.get("payload")
    if isinstance(payload, dict):
        return payload
    data_obj = task_obj.get("data")
    if isinstance(data_obj, dict):
        nested = data_obj.get("payload")
        if isinstance(nested, dict):
            return nested
    return {}


def _task_id(task: dict[str, Any] | None) -> str:
    task_obj = task if isinstance(task, dict) else {}
    data_obj = task_obj.get("data") if isinstance(task_obj.get("data"), dict) else {}
    return str(task_obj.get("id") or data_obj.get("id") or "").strip()


def _task_type(task: dict[str, Any] | None) -> str:
    task_obj = task if isinstance(task, dict) else {}
    data_obj = task_obj.get("data") if isinstance(task_obj.get("data"), dict) else {}
    return str(task_obj.get("type") or data_obj.get("type") or "").strip()


def _task_status(task: dict[str, Any] | None) -> str:
    task_obj = task if isinstance(task, dict) else {}
    data_obj = task_obj.get("data") if isinstance(task_obj.get("data"), dict) else {}
    return str(task_obj.get("status") or data_obj.get("status") or "").strip().lower()


def _task_session_id(task: dict[str, Any] | None) -> str:
    task_obj = task if isinstance(task, dict) else {}
    data_obj = task_obj.get("data") if isinstance(task_obj.get("data"), dict) else {}
    return str(task_obj.get("session_id") or data_obj.get("session_id") or "").strip()


def _task_avatar_mode(task: dict[str, Any] | None) -> str:
    task_obj = task if isinstance(task, dict) else {}
    data_obj = task_obj.get("data") if isinstance(task_obj.get("data"), dict) else {}
    return str(task_obj.get("avatar_mode") or data_obj.get("avatar_mode") or "").strip()


def _task_is_claimable(task: dict[str, Any] | None) -> bool:
    status = _task_status(task)
    if status in {"assigned", "processing", "done", "completed", "canceled", "cancelled", "error", "failed"}:
        return False
    task_type = _task_type(task)
    if task_type != "publish":
        return False
    return True


def _payload_livekit_url(payload: dict[str, Any]) -> str:
    return _string_payload_value(payload.get("livekit_url"))


def _payload_worker_token(payload: dict[str, Any]) -> str:
    return _string_payload_value(payload.get("worker_token"))


def _normalize_livekit_ws_url(url: str) -> str:
    """
    Normalize URL for SDK room.connect.
    LiveKit expects ws/wss in most SDK paths; convert http/https to ws/wss.
    """
    s = str(url or "").strip()
    if not s:
        return ""
    try:
        u = urllib.parse.urlparse(s)
        scheme = (u.scheme or "").lower()
        if scheme == "https":
            u = u._replace(scheme="wss")
            return urllib.parse.urlunparse(u)
        if scheme == "http":
            u = u._replace(scheme="ws")
            return urllib.parse.urlunparse(u)
    except Exception:
        return s
    return s


def _jwt_claims_unverified(token: str) -> dict[str, Any]:
    """
    Decode JWT payload without signature verification for diagnostics only.
    """
    tok = str(token or "").strip()
    if not tok:
        return {}
    parts = tok.split(".")
    if len(parts) < 2:
        return {}
    payload_b64 = parts[1].strip()
    if not payload_b64:
        return {}
    try:
        pad = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        obj = json.loads(raw.decode("utf-8", errors="ignore"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _livekit_disconnect_reason_name(reason: Any) -> str:
    try:
        reason_int = int(reason)
    except Exception:
        try:
            return str(reason or "").strip() or "UNKNOWN_REASON"
        except Exception:
            return "UNKNOWN_REASON"
    try:
        rtc = _livekit_rtc()
        return str(rtc.DisconnectReason.Name(reason_int) or f"UNKNOWN_REASON_{reason_int}")
    except Exception:
        return f"UNKNOWN_REASON_{reason_int}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return int(default)


_LIVE_SAMPLE_STEP_KEYS = (
    "sample_steps",
    "live_sample_steps",
    "live_denoise_steps",
    "denoise_steps",
    "live_num_inference_steps",
    "num_inference_steps",
)


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_int = int(value)
        return value_int if value_int > 0 else None
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        value_int = int(float(raw))
    except Exception:
        return None
    return value_int if value_int > 0 else None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _claim_payload_dict(claim: dict[str, Any]) -> dict[str, Any]:
    for key in ("payload", "payload_json", "job_payload"):
        payload = claim.get(key)
        if isinstance(payload, dict):
            return payload
    job = _dict_or_empty(claim.get("job"))
    for key in ("payload", "payload_json", "job_payload"):
        payload = job.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _live_sample_step_sources(claim: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    claim_d = _dict_or_empty(claim)
    job = _dict_or_empty(claim_d.get("job"))
    payload = _claim_payload_dict(claim_d)
    live_session = _dict_or_empty(claim_d.get("live_session"))
    live_metadata = _dict_or_empty(live_session.get("metadata_json"))
    consultation = _dict_or_empty(claim_d.get("consultation"))
    sources: list[tuple[str, dict[str, Any]]] = [
        ("payload", payload),
        ("live_session.metadata_json", live_metadata),
        ("live_session", live_session),
        ("consultation", consultation),
        ("job", job),
        ("claim", claim_d),
    ]
    expanded: list[tuple[str, dict[str, Any]]] = []
    for source_name, src in sources:
        if not src:
            continue
        expanded.append((source_name, src))
        for key in ("video", "live", "render", "generation", "inference", "model", "settings", "options"):
            nested = src.get(key)
            if isinstance(nested, dict):
                expanded.append((f"{source_name}.{key}", nested))
    return expanded


def _live_sample_steps_from_claim(claim: dict[str, Any] | None, *, default_value: int) -> tuple[int, str]:
    value = int(smartblog_clamp_sample_steps(default_value))
    source = "env"
    claim_d = _dict_or_empty(claim)
    for source_name, src in _live_sample_step_sources(claim_d):
        for key in _LIVE_SAMPLE_STEP_KEYS:
            candidate = _optional_positive_int(src.get(key))
            if candidate is not None:
                value = int(smartblog_clamp_sample_steps(candidate))
                source = f"{source_name}.{key}"
                break
        if source != "env":
            break
    max_value = int(
        min(
            int(SMARTBLOG_MAX_SAMPLE_STEPS),
            int(
                _safe_int_env(
                    "SMARTBLOG_LIVE_MAX_SAMPLE_STEPS",
                    _safe_int_env(
                        "WORKER_MAX_SAMPLE_STEPS",
                        _safe_int_env("SMARTBLOG_RENDER_MAX_SAMPLE_STEPS", int(SMARTBLOG_MAX_SAMPLE_STEPS)),
                    ),
                )
            ),
        )
    )
    if max_value > 0:
        value = int(min(int(value), int(max_value)))
    return int(smartblog_clamp_sample_steps(value)), str(source)


def reply_sample_steps_for_worker(worker: Any | None = None, *, claim: dict[str, Any] | None = None, trace_id: str | None = None) -> int:
    default_value = int(_safe_int_env("SMARTBLOG_LIVE_SAMPLE_STEPS", int(FORCED_REPLY_SAMPLE_STEPS)))
    active_claim = claim if isinstance(claim, dict) else getattr(worker, "_active_claim", None)
    active_claim = active_claim if isinstance(active_claim, dict) else {}
    value, source = _live_sample_steps_from_claim(active_claim, default_value=int(default_value))
    if source != "env":
        job = _dict_or_empty(active_claim.get("job"))
        job_id = str(job.get("id") or active_claim.get("job_id") or "-")
        signature = (str(job_id), int(value), str(source))
        if worker is None or getattr(worker, "_last_live_sample_steps_log", None) != signature:
            logging.warning(
                "SmartBlog live sample steps override: job=%s trace=%s sample_steps=%d source=%s",
                str(job_id),
                str(trace_id or "-"),
                int(value),
                str(source),
            )
            if worker is not None:
                setattr(worker, "_last_live_sample_steps_log", signature)
    return int(value)


def _async_queue_put_drop_oldest(q: Any, item: Any) -> int:
    """
    Best-effort bounded enqueue for asyncio.Queue.
    Returns number of dropped oldest items (0/1 in current policy).
    Raises the final enqueue error if the item still cannot be queued.
    """
    try:
        q.put_nowait(item)
        return 0
    except asyncio.QueueFull:
        dropped = 0
        try:
            _ = q.get_nowait()
            dropped = 1
        except Exception:
            dropped = 0
        q.put_nowait(item)
        return int(dropped)


# Export everything (including single-underscore helpers) for internal package
# modules that use `from .common import *` during the transitional split.
__all__ = [k for k in globals().keys() if not k.startswith("__")]
