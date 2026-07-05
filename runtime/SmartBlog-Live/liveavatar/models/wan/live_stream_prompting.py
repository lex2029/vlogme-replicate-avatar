from __future__ import annotations

import os


_DEFAULT_IDLE_PROMPT_SUFFIX = (
    "The person is silent with lips closed, calm and steady, with very restrained facial expression, "
    "minimal head motion, minimal body motion, only subtle breathing and occasional natural blinking."
)


def _env_flag(name: str, default: str = "0") -> bool:
    val = str(os.getenv(str(name), str(default)) or str(default)).strip().lower()
    return val in {"1", "true", "yes", "on"}


def stream_prompt_switch_enabled() -> bool:
    return _env_flag("LIVE_AUDIO_STREAM_CLIP_PROMPT_SWITCH", "1")


def normalize_stream_clip_kind(kind: str | None) -> str:
    kk = str(kind or "speech").strip().lower()
    if kk in {"filler", "idle", "silence", "keepalive", "gap_fill", "gap-fill", "gapfill"}:
        return "filler"
    if kk in {"speech_tail", "tail"}:
        return "speech_tail"
    return "speech"


def prompt_switch_clip_kind_for_chunk(
    *,
    kind: str | None,
    source_samples: int | None = None,
    full_chunk_samples: int | None = None,
    turn_done: bool = False,
) -> str:
    kind_norm = normalize_stream_clip_kind(kind)
    if kind_norm != "speech":
        return kind_norm
    try:
        src_n = int(source_samples or 0)
    except Exception:
        src_n = 0
    try:
        full_n = int(full_chunk_samples or 0)
    except Exception:
        full_n = 0
    full_n = max(1, int(full_n))
    # A terminal speech chunk can still contain audible speech for a large part
    # of the clip. Do not switch the whole clip to idle unless explicitly
    # enabled for a tiny residual tail.
    try:
        idle_tail_max = int(os.getenv("LIVE_AUDIO_STREAM_IDLE_TAIL_MAX_SOURCE_SAMPLES", "0") or "0")
    except Exception:
        idle_tail_max = 0
    idle_tail_max = max(0, min(int(full_n), int(idle_tail_max)))
    if bool(turn_done) and 0 < int(src_n) <= int(idle_tail_max):
        return "filler"
    if bool(turn_done) and 0 < int(src_n) < int(full_n):
        return "speech_tail"
    return kind_norm


def merge_stream_clip_kinds(left_kind: str | None, right_kind: str | None) -> str:
    left = normalize_stream_clip_kind(left_kind)
    right = normalize_stream_clip_kind(right_kind)
    if left == "filler" and right == "filler":
        return "filler"
    if left == "speech_tail" and right == "speech_tail":
        return "speech_tail"
    return "speech"


def build_stream_idle_prompt_text(prompt: str, *, idle_prompt: str | None = None) -> str:
    explicit_idle = str(idle_prompt or "").strip()
    if explicit_idle:
        return explicit_idle
    runtime_override = str(os.getenv("WORKER_RUNTIME_IDLE_PROMPT", "") or "").strip()
    if runtime_override:
        return runtime_override
    suffix = str(
        os.getenv("WORKER_IDLE_PROMPT_SUFFIX", _DEFAULT_IDLE_PROMPT_SUFFIX) or _DEFAULT_IDLE_PROMPT_SUFFIX
    ).strip()
    return suffix


def stream_clip_prefers_idle_prompt(*, clip_kind: str | None, is_silence: bool, enabled: bool) -> bool:
    if not bool(enabled):
        return False
    if bool(is_silence):
        return True
    kind_norm = normalize_stream_clip_kind(clip_kind)
    if kind_norm == "filler":
        return True
    if kind_norm == "speech_tail":
        return _env_flag("LIVE_AUDIO_STREAM_SPEECH_TAIL_IDLE_PROMPT", "0")
    return False
