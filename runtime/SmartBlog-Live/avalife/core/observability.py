from __future__ import annotations

import logging
import os
import time
from typing import Any


def env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def runtime_profile_name() -> str:
    raw = str(os.getenv("WORKER_PROFILE_NAME", "prod") or "prod").strip().lower()
    return raw or "prod"


def boot_log_enabled() -> bool:
    return env_flag("WORKER_BOOT_LOG", "0")


def worker_timing_enabled() -> bool:
    return env_flag("WORKER_TIMING_LOG", "0")


def model_timing_enabled() -> bool:
    return env_flag("MODEL_TIMING_LOG", "0")


def media_timing_enabled() -> bool:
    return env_flag("MEDIA_TIMING_LOG", "0") or model_timing_enabled()


def post_vae_timing_enabled() -> bool:
    return env_flag("POST_VAE_TIMING_LOG", "0") or model_timing_enabled()


def deep_timing_enabled() -> bool:
    return env_flag("WORKER_DEEP_TIMING_LOG", "0")


def deep_gpu_sync_timing_enabled() -> bool:
    return deep_timing_enabled() and env_flag("WORKER_DEEP_GPU_SYNC_TIMING", "0")


def _env_log_level(name: str, default: str) -> int:
    raw = str(os.getenv(name, default) or default).strip().upper()
    return int(getattr(logging, raw, getattr(logging, default.upper(), logging.INFO)))


def worker_log_level() -> int:
    return _env_log_level("WORKER_LOG_LEVEL", "INFO")


def model_log_level() -> int:
    return _env_log_level("MODEL_LOG_LEVEL", "INFO")


def model_other_rank_log_level() -> int:
    return _env_log_level("MODEL_OTHER_RANK_LOG_LEVEL", "ERROR")


def maybe_cuda_synchronize(device: Any | None = None) -> None:
    if not deep_gpu_sync_timing_enabled():
        return
    try:
        import torch

        if not bool(torch.cuda.is_available()):
            return
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)
    except Exception:
        return


def log_timing(
    label: str,
    started_at: float,
    *,
    enabled: bool,
    sync_device: Any | None = None,
    level: int = logging.INFO,
    **fields,
) -> None:
    if not enabled:
        return
    maybe_cuda_synchronize(sync_device)
    parts = [f"{key}={value}" for key, value in fields.items()]
    parts.append(f"dt={float(time.perf_counter() - started_at):.3f}s")
    logging.log(level, "%s: %s", str(label), " ".join(parts))


def log_phase_timing(
    component: str,
    phase: str,
    started_at: float,
    *,
    enabled: bool,
    sync_device: Any | None = None,
    level: int = logging.INFO,
    **fields,
) -> None:
    log_timing(
        f"{component} timing",
        started_at,
        enabled=enabled,
        sync_device=sync_device,
        level=level,
        phase=str(phase),
        **fields,
    )
