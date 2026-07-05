from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: str) -> bool:
    value = str(os.getenv(name, default) or default).strip().lower()
    return value not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(str(os.getenv(name, str(default)) or default).strip())
    except Exception:
        value = int(default)
    return max(int(low), min(int(high), int(value)))


def _required_env_int(name: str, *, low: int, high: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing required env: {name}")
    try:
        value = int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid integer env {name}={raw!r}") from e
    return max(int(low), min(int(high), int(value)))


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(str(os.getenv(name, str(default)) or default).strip())
    except Exception:
        value = float(default)
    return max(float(low), min(float(high), float(value)))


def liveaudio_allow_long_clips() -> bool:
    return _env_flag("LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS", "0")


def liveaudio_max_clip_frames(infer_frames: int) -> int:
    clip_frames = int(max(1, int(infer_frames)))
    default = int(max(clip_frames, clip_frames * 2))
    return _env_int(
        "LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES",
        default,
        low=clip_frames,
        high=512,
    )


def effective_async_producer_mode(*, requested_async: bool, is_always_on: bool) -> bool:
    """
    Keep async refill whenever the runtime requests it.

    Regular reply streams need enough buffered audio clips to avoid q=0/1
    starvation. The previous reply-only boundary-prefill mode reduced compute
    contention, but in practice it made live speech cadence noticeably worse.
    Preserve the configured async producer for both always-on idle and normal
    reply streams.
    """
    if not bool(requested_async):
        return False
    return True


def reply_boundary_prefill_wait_sec() -> float:
    """
    Small startup/boundary budget for regular reply streams.

    We keep audio encode off the denoise hot path, but allow a very short wait
    right before the next clip begins so a nearly-ready chunk can be picked up
    without falling into q=0/1 starvation.
    """
    return 0.06


def chunk_encode_batch_frames(*, expected_frames: int, infer_frames: int) -> int:
    """
    Pick the audio-encoder batch size for one live speech chunk.

    The model still consumes fixed `infer_frames` clips, but per-chunk audio
    embeddings should be encoded at the chunk's expected video-frame count.
    Otherwise short live chunks are over-encoded to the full clip length and
    then trimmed back down, which adds avoidable encode work and jitter.
    """
    target = int(expected_frames or 0)
    clip_frames = int(max(1, int(infer_frames)))
    if target <= 0:
        return clip_frames
    if liveaudio_allow_long_clips():
        return max(1, min(liveaudio_max_clip_frames(clip_frames), target))
    return max(1, min(clip_frames, target))


def chunk_conditioning_target_frames(*, expected_frames: int, infer_frames: int, clip_kind: str) -> int:
    """
    Pick the final conditioning length for one streamed liveaudio chunk.

    Keep speech chunks at their real frame duration. With 24-frame live blocks
    and 48-frame inference clips, padding each 24-frame speech chunk to 48 frames
    creates a silent second block where the avatar continues moving without
    audio. Residual chunks are carried in `stream_audio_tail` and joined with the
    next chunk; the final tail flush pads only to a block-safe length.
    """
    target = int(expected_frames or 0)
    clip_frames = int(max(1, int(infer_frames)))
    if target <= 0:
        return clip_frames
    if liveaudio_allow_long_clips():
        target = max(1, min(liveaudio_max_clip_frames(clip_frames), target))
    else:
        target = max(1, min(clip_frames, target))
    kind = str(clip_kind or "speech").strip().lower()
    if (
        kind == "speech"
        and int(target) < int(clip_frames)
        and _env_flag("LIVE_AUDIO_STREAM_PAD_SHORT_SPEECH_TO_FULL_CLIP", "0")
    ):
        return int(clip_frames)
    return int(target)


def pending_clip_target(*, max_pending_clips: int, is_always_on: bool) -> int:
    """
    Soft queue target for the async liveaudio producer.

    Always-on idle can keep the full configured queue. Regular reply streams
    should keep a smaller headroom so producer refill does not burst all the
    way to the hard cap on the same GPU while denoise is running.
    """
    hard_cap = int(max(1, int(max_pending_clips)))
    if bool(is_always_on):
        return hard_cap
    configured = _required_env_int("LIVE_AUDIO_STREAM_REPLY_MODEL_QUEUE_TARGET", low=1, high=64)
    if int(configured) > 0:
        return min(hard_cap, int(configured))
    return min(hard_cap, 5)


@dataclass(frozen=True)
class LiveaudioRuntimeConfig:
    tail_frames: int
    poll_sec: float
    immediate_silence: bool
    reply_start_min_clips: int
    timing_log: bool
    phase_sync_debug: bool
    timing_slow_sec: float
    step_trace: bool
    max_pending_clips: int
    refill_during_denoise: bool
    refill_block_interval: int
    refill_max_chunks_per_call: int
    async_producer: bool
    async_start_after_first_clip: bool
    distributed_clip_broadcast: bool
    encode_rank: int

    @classmethod
    def from_env(cls, *, infer_frames: int, world_size: int) -> "LiveaudioRuntimeConfig":
        timing_log = _env_flag("LIVE_AUDIO_TPP_TIMING_LOG", "0")
        phase_sync_default = "1" if bool(timing_log and _env_flag("WORKER_DEEP_GPU_SYNC_TIMING", "0")) else "0"
        distributed_clip_broadcast = int(max(1, int(world_size))) > 1
        encode_rank = 0
        if distributed_clip_broadcast:
            encode_rank = _env_int(
                "LIVE_AUDIO_STREAM_ENCODE_RANK",
                0,
                low=0,
                high=max(0, int(world_size) - 1),
            )
        return cls(
            tail_frames=max(1, int(infer_frames) // 2),
            poll_sec=_env_float("LIVE_AUDIO_STREAM_POLL_SEC", 0.02, low=0.01, high=1.0),
            immediate_silence=_env_flag("LIVE_AUDIO_STREAM_IMMEDIATE_SILENCE", "1"),
            reply_start_min_clips=_required_env_int("LIVE_AUDIO_STREAM_REPLY_START_MIN_CLIPS", low=1, high=8),
            timing_log=timing_log,
            phase_sync_debug=_env_flag("LIVE_AUDIO_TPP_PHASE_SYNC_DEBUG", phase_sync_default),
            timing_slow_sec=_env_float("LIVE_AUDIO_TPP_TIMING_SLOW_SEC", 0.35, low=0.05, high=10.0),
            step_trace=_env_flag("LIVE_AUDIO_TPP_STEP_TRACE", "0"),
            max_pending_clips=_required_env_int("LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS", low=1, high=64),
            refill_during_denoise=_env_flag("LIVE_AUDIO_STREAM_REFILL_DURING_DENOISE", "0"),
            refill_block_interval=_env_int("LIVE_AUDIO_STREAM_REFILL_BLOCK_INTERVAL", 2, low=1, high=16),
            refill_max_chunks_per_call=_env_int(
                "LIVE_AUDIO_STREAM_REFILL_MAX_CHUNKS_PER_CALL",
                1,
                low=1,
                high=8,
            ),
            async_producer=_env_flag("LIVE_AUDIO_STREAM_ASYNC_PRODUCER", "1"),
            async_start_after_first_clip=_env_flag("LIVE_AUDIO_STREAM_ASYNC_START_AFTER_FIRST_CLIP", "1"),
            distributed_clip_broadcast=distributed_clip_broadcast,
            encode_rank=encode_rank,
        )
