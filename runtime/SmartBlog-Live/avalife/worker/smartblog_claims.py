from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from .common import _env_flag, _safe_float_env
from .smartblog_jobs import (
    SMARTBLOG_JOB_TYPE_RENDER_VIDEO,
    SMARTBLOG_JOB_TYPE_TEST_VIDEO,
    SMARTBLOG_JOB_TYPE_VIDEO_TEST,
    _smartblog_remote_edge_render_enabled,
    smartblog_render_job_types,
)


SMARTBLOG_JOB_PRIORITY_RENDER = 10
SMARTBLOG_JOB_PRIORITY_GENERIC_BUSY = 50
SMARTBLOG_JOB_PRIORITY_UNKNOWN = 100

_SMARTBLOG_EXTERNAL_JOB_TYPES = (
    SMARTBLOG_JOB_TYPE_RENDER_VIDEO,
    SMARTBLOG_JOB_TYPE_VIDEO_TEST,
    SMARTBLOG_JOB_TYPE_TEST_VIDEO,
)
_SMARTBLOG_DEFAULT_JOB_TYPES = (
    SMARTBLOG_JOB_TYPE_RENDER_VIDEO,
)


def smartblog_control_plane_enabled() -> bool:
    return str(os.getenv("WORKER_CONTROL_PLANE", "avalife") or "avalife").strip().lower() == "smartblog"


def smartblog_poll_interval_sec() -> float:
    return max(1.0, min(30.0, float(_safe_float_env("SMARTBLOG_POLL_SEC", 5.0))))


def smartblog_realtime_safety_poll_sec() -> float:
    return max(30.0, min(600.0, float(_safe_float_env("SMARTBLOG_REALTIME_SAFETY_POLL_SEC", 120.0))))


def smartblog_claim_after_poll_delay_sec() -> float:
    return max(0.0, min(120.0, float(_safe_float_env("SMARTBLOG_CLAIM_AFTER_POLL_DELAY_SEC", 0.0))))


def smartblog_delayed_claim_sec() -> float:
    return max(0.0, min(120.0, float(_safe_float_env("SMARTBLOG_DELAYED_CLAIM_SEC", 0.0))))


def smartblog_delayed_claim_job_types() -> tuple[str, ...]:
    raw = str(os.getenv("SMARTBLOG_DELAYED_CLAIM_JOB_TYPES", "") or "").strip()
    if not raw:
        return ()
    allowed = set(_SMARTBLOG_EXTERNAL_JOB_TYPES)
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        token = str(part or "").strip()
        if token in {"*", "all"}:
            return tuple(_SMARTBLOG_EXTERNAL_JOB_TYPES)
        if token and token in allowed and token not in out:
            out.append(token)
    return tuple(out)


def smartblog_realtime_debug_enabled() -> bool:
    return _env_flag("SMARTBLOG_REALTIME_DEBUG", "0")


def smartblog_progress_interval_sec() -> float:
    return max(2.0, min(60.0, float(_safe_float_env("SMARTBLOG_PROGRESS_SEC", 15.0))))


def smartblog_preempt_delay_sec() -> float:
    return max(0.0, min(120.0, float(_safe_float_env("SMARTBLOG_PREEMPT_DELAY_SEC", 10.0))))


def smartblog_preempt_stop_timeout_sec() -> float:
    return max(1.0, min(60.0, float(_safe_float_env("SMARTBLOG_PREEMPT_STOP_TIMEOUT_SEC", 8.0))))


def smartblog_preempt_enabled() -> bool:
    return _env_flag("SMARTBLOG_PREEMPT_ENABLED", "0")


def smartblog_remote_edge_preclaim_enabled() -> bool:
    return _env_flag("SMARTBLOG_REMOTE_EDGE_PRECLAIM_CHECK", "1")


def smartblog_remote_edge_preclaim_timeout_sec() -> float:
    return max(0.2, min(10.0, float(_safe_float_env("SMARTBLOG_REMOTE_EDGE_PRECLAIM_TIMEOUT_SEC", 2.0))))


def smartblog_remote_edge_preclaim_skip_sec() -> float:
    return max(1.0, min(300.0, float(_safe_float_env("SMARTBLOG_REMOTE_EDGE_PRECLAIM_SKIP_SEC", 10.0))))


def smartblog_job_needs_remote_edge_preclaim(claim_or_job: dict[str, Any] | None) -> bool:
    job_type = str(smartblog_job_type_value(claim_or_job) or "").strip().lower()
    if job_type in set(smartblog_render_job_types()):
        return bool(_smartblog_remote_edge_render_enabled())
    return False


def smartblog_job_needs_delayed_claim(claim_or_job: dict[str, Any] | None) -> bool:
    if smartblog_delayed_claim_sec() <= 0.0:
        return False
    job_type = str(smartblog_job_type_value(claim_or_job) or "").strip()
    return bool(job_type and job_type in set(smartblog_delayed_claim_job_types()))


def smartblog_progress_stale_sec() -> float:
    default_v = max(90.0, float(smartblog_progress_interval_sec()) * 6.0)
    return max(30.0, min(1800.0, float(_safe_float_env("SMARTBLOG_PROGRESS_STALE_SEC", default_v))))


def smartblog_disconnect_timeout_sec() -> float:
    return max(10.0, min(600.0, float(_safe_float_env("SMARTBLOG_DISCONNECT_TIMEOUT_SEC", 60.0))))


def smartblog_supported_job_types() -> tuple[str, ...]:
    raw = str(os.getenv("SMARTBLOG_JOB_TYPES", os.getenv("SMARTBLOG_JOB_TYPE", "")) or "").strip()
    allowed = set(_SMARTBLOG_EXTERNAL_JOB_TYPES)
    if raw:
        out: list[str] = []
        for part in str(raw).split(","):
            token = str(part or "").strip()
            if token and token in allowed and token not in out:
                out.append(token)
        return tuple(out)
    return tuple(_SMARTBLOG_DEFAULT_JOB_TYPES)


def smartblog_job_type_value(claim_or_job: dict[str, Any] | None) -> str:
    src = claim_or_job if isinstance(claim_or_job, dict) else {}
    job = src.get("job") if isinstance(src.get("job"), dict) else None
    if isinstance(job, dict):
        value = str(job.get("job_type") or "").strip()
        if value:
            return value
    return str(src.get("job_type") or "").strip()


def smartblog_job_status_value(claim_or_job: dict[str, Any] | None) -> str:
    src = claim_or_job if isinstance(claim_or_job, dict) else {}
    job = src.get("job") if isinstance(src.get("job"), dict) else None
    if isinstance(job, dict):
        value = str(job.get("status") or "").strip()
        if value:
            return value
    return str(src.get("status") or "").strip()


def _smartblog_time_value_to_mono(
    raw: Any,
    *,
    now_mono: float | None = None,
    now_epoch: float | None = None,
    fallback: float,
) -> float:
    local_now_mono = float(time.monotonic() if now_mono is None else now_mono)
    local_now_epoch = float(time.time() if now_epoch is None else now_epoch)
    if raw is None:
        return float(fallback)
    try:
        if isinstance(raw, (int, float)):
            started_epoch = float(raw)
        else:
            raw_s = str(raw or "").strip()
            if not raw_s:
                return float(fallback)
            if raw_s.endswith("Z"):
                raw_s = raw_s[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw_s)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            started_epoch = float(parsed.timestamp())
    except Exception:
        return float(fallback)
    if started_epoch <= 0.0:
        return float(fallback)
    elapsed_sec = max(0.0, float(local_now_epoch - started_epoch))
    return float(local_now_mono - elapsed_sec)


def smartblog_claim_started_mono(
    claim: dict[str, Any] | None,
    *,
    now_mono: float | None = None,
    now_epoch: float | None = None,
) -> float:
    local_now_mono = float(time.monotonic() if now_mono is None else now_mono)
    src = claim if isinstance(claim, dict) else {}
    job = src.get("job") if isinstance(src.get("job"), dict) else {}
    raw = job.get("started_at") or src.get("started_at")
    return _smartblog_time_value_to_mono(
        raw,
        now_mono=local_now_mono,
        now_epoch=now_epoch,
        fallback=local_now_mono,
    )


def smartblog_progress_stage_fields_for_claim(
    claim_or_job: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": "heartbeat",
        "stage_label": "Worker heartbeat",
        "stage_index": 1,
        "stage_total": 1,
    }


def smartblog_supports_job(claim_or_job: dict[str, Any] | None) -> bool:
    job_type = str(smartblog_job_type_value(claim_or_job)).strip()
    return bool(job_type and job_type in set(smartblog_supported_job_types()))


def smartblog_job_is_queue_candidate(claim_or_job: dict[str, Any] | None) -> bool:
    status = str(smartblog_job_status_value(claim_or_job)).strip().lower()
    if not status:
        return True
    return status == "queued"


def smartblog_job_priority_value(claim_or_job: dict[str, Any] | None) -> int:
    job_type = str(smartblog_job_type_value(claim_or_job)).strip().lower()
    if job_type in set(smartblog_render_job_types()):
        return int(SMARTBLOG_JOB_PRIORITY_RENDER)
    return int(SMARTBLOG_JOB_PRIORITY_UNKNOWN)


def smartblog_is_urgent_job(claim_or_job: dict[str, Any] | None) -> bool:
    return False


def smartblog_sort_jobs_for_claim(jobs: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    ordered: list[tuple[tuple[int, str, int], dict[str, Any]]] = []
    for idx, job in enumerate(list(jobs or [])):
        if not isinstance(job, dict):
            continue
        created_at = str(job.get("created_at") or "~").strip() or "~"
        ordered.append(((int(smartblog_job_priority_value(job)), created_at, int(idx)), dict(job)))
    ordered.sort(key=lambda item: item[0])
    return [dict(job) for _, job in ordered]
