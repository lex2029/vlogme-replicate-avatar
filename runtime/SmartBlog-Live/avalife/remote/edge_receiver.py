from __future__ import annotations

import argparse
import array
import asyncio
import importlib
import json
import logging
import os
import shlex
import shutil
import sys
import select
import socket
import subprocess
import tempfile
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from avalife.core.channel_ffmpeg import build_rtmp_ffmpeg_cmd, nvenc_runtime_available, resolve_rtmp_publish_canvas, rtmp_output_fps
from avalife.core.ffmpeg import probe_video_metadata
from avalife.core.upload_retry import put_file_to_signed_url
from avalife.core.watermark import normalize_watermark_text, watermark_drawtext_filter, write_watermark_text_file
from avalife.remote.latent_vae import (
    CudaReadyTensor,
    WanLatentDecoder,
    get_shared_wan_latent_decoder,
    prewarm_shared_wan_latent_decoder_from_env,
)
from avalife.remote.protocol import MAX_PAYLOAD_BYTES, read_message, read_message_timed, write_message
from avalife.remote.subtitles import SubtitleRenderer
from avalife.worker.common import (
    _env_flag,
    _required_nonnegative_float_env,
    _required_nonnegative_int_env,
    _required_positive_float_env,
    _required_positive_int_env,
    _safe_float_env,
    _safe_int_env,
)
from avalife.worker.smartblog_api import SmartBlogClient, smartblog_api_rejection_reason, smartblog_worker_api_key


_EDGE_ACTIVE_SESSIONS: set[str] = set()
_EDGE_ACTIVE_LOCK: asyncio.Lock | None = None
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIVEKIT_RTC: Any | None = None
_CLIP_EXPORTER_MODULE: Any | None = None


def _livekit_rtc() -> Any:
    global _LIVEKIT_RTC
    if _LIVEKIT_RTC is not None:
        return _LIVEKIT_RTC
    try:
        from livekit import rtc as rtc_module
    except Exception as e:
        raise RuntimeError("LiveKit output requires the livekit package") from e
    _LIVEKIT_RTC = rtc_module
    return rtc_module


def _clip_exporter_module() -> Any:
    global _CLIP_EXPORTER_MODULE
    if _CLIP_EXPORTER_MODULE is None:
        _CLIP_EXPORTER_MODULE = importlib.import_module("avalife.remote.clip_exporter")
    return _CLIP_EXPORTER_MODULE


def _edge_clips_enabled() -> bool:
    if not _env_flag("REMOTE_EDGE_CLIPS_ENABLED", "0"):
        return False
    return bool(_clip_exporter_module().edge_clips_enabled())


def _sanitize_clip_component(value: str | None, *, fallback: str = "clip", max_len: int = 96) -> str:
    text = str(value or fallback or "clip").strip() or str(fallback or "clip")
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")
    return (out or str(fallback or "clip"))[: max(1, int(max_len))]


def _edge_state_file() -> str:
    path = str(os.getenv("SMARTBLOG_EDGE_STATE_FILE") or "").strip()
    if path:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_WORKSPACE_ROOT, "runtime", "edge_receiver_state.json"))


def _write_json_atomic(tmp: str, path: str, payload: dict[str, Any]) -> None:
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _exit_after_fatal_cuda_error(*, context: str, session_id: str, job_id: str, err: BaseException) -> None:
    logging.critical(
        "Remote edge fatal CUDA error; exiting process so supervisor restarts with a clean CUDA context: context=%s session=%s job=%s err=%s",
        str(context),
        str(session_id),
        str(job_id),
        err,
    )
    try:
        logging.shutdown()
    finally:
        os._exit(86)


async def _write_edge_state(*, active_sessions: set[str]) -> None:
    path = _edge_state_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "busy": bool(active_sessions),
        "active_sessions": int(len(active_sessions)),
        "sessions": sorted(str(x) for x in active_sessions),
        "pid": int(os.getpid()),
        "updated_at": float(time.time()),
    }
    tmp = f"{path}.tmp"
    await asyncio.to_thread(_write_json_atomic, tmp, path, payload)


async def _mark_edge_session_active(session_id: str, *, active: bool) -> None:
    global _EDGE_ACTIVE_LOCK
    if _EDGE_ACTIVE_LOCK is None:
        _EDGE_ACTIVE_LOCK = asyncio.Lock()
    sid = str(session_id or "").strip()
    async with _EDGE_ACTIVE_LOCK:
        if sid:
            if bool(active):
                _EDGE_ACTIVE_SESSIONS.add(sid)
            else:
                _EDGE_ACTIVE_SESSIONS.discard(sid)
        await _write_edge_state(active_sessions=set(_EDGE_ACTIVE_SESSIONS))


@dataclass
class EdgeSessionConfig:
    session_id: str
    job_id: str
    livekit_url: str
    livekit_token: str
    output: str
    rtmp_url: str
    width: int
    height: int
    fps: int
    sample_rate: int
    mode: str
    live_session_id: str = ""
    workspace_id: str = ""
    persona_id: str = ""
    rtmp_urls: list[str] | None = None
    file_upload_url: str = ""
    file_upload_path: str = ""
    file_public_url: str = ""
    file_content_type: str = "video/mp4"
    file_output_fps: int = 0
    file_target_audio_samples: int = 0
    file_target_duration_sec: float = 0.0
    watermark_text: str = ""
    file_remote_finalizer: bool | None = None


@dataclass
class _LatentDecodeJob:
    payload: bytes
    codec: str
    shape: str
    dtype: str
    keep_last_frames: int | None
    reset_vae: bool
    prime_only: bool
    face_restore: float | None
    background_restore: float | None
    timestamp_us: int | None
    segment_id: str | None
    segment_kind: str
    segment_start_frame: int | None
    segment_frames: int | None
    avatar_ref_path: str = ""
    wire_payload_len: int = 0
    payload_read_sec: float = 0.0
    transport_decode_sec: float = 0.0
    enqueued_at: float = 0.0


@dataclass
class _LatentPostprocessJob:
    frames_tensor: Any
    face_restore: float | None
    background_restore: float | None
    timestamp_us: int | None
    segment_id: str | None
    segment_kind: str
    segment_start_frame: int | None
    segment_frames: int | None
    avatar_ref_path: str = ""
    input_range: str = "01"
    queue_wait_sec: float = 0.0
    decode_sec: float = 0.0
    payload_len: int = 0
    frame_count: int = 0
    enqueued_at: float = 0.0
    defer_stage: bool = False


class _CpuReadyTensor:
    """CPU tensor plus a CUDA event that marks when an async D2H stage is done."""

    __slots__ = ("tensor", "ready_event", "ready_device", "source_ref", "stream_ref")

    def __init__(
        self,
        tensor: Any,
        ready_event: Any | None = None,
        ready_device: str = "",
        source_ref: Any | None = None,
        stream_ref: Any | None = None,
    ) -> None:
        self.tensor = tensor
        self.ready_event = ready_event
        self.ready_device = str(ready_device or "")
        self.source_ref = source_ref
        self.stream_ref = stream_ref

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tensor, name)


@dataclass
class _PendingVideoBlock:
    frames: list[bytes]
    timestamp_us: int | None
    segment_id: str | None = None
    segment_kind: str = ""
    segment_start_frame: int | None = None
    segment_frames: int | None = None
    frames_tensor: Any | None = None
    avatar_ref_path: str = ""


@dataclass
class _SegmentAudio:
    payload: bytes
    sample_rate: int
    segment_id: str = ""
    segment_kind: str = ""
    segment_frames: int | None = None
    audible_samples: int | None = None
    turn_done: bool = False
    subtitle_text: str = ""
    subtitle_start_samples: int | None = None
    subtitle_end_samples: int | None = None
    subtitle_total_samples: int | None = None
    subtitle_alignment: dict[str, Any] | None = None
    subtitle_normalized_alignment: dict[str, Any] | None = None
    subtitle_alignment_base_samples: int | None = None


def _rtmp_candidate_urls(primary: str, candidates: Any) -> list[str]:
    out: list[str] = []
    skip_restream_443 = _env_flag("REMOTE_EDGE_RTMP_SKIP_RESTREAM_443", "1")
    skip_restream_rtmps = _env_flag("REMOTE_EDGE_RTMP_SKIP_RESTREAM_RTMPS", "1")

    def is_skipped(url: str) -> bool:
        try:
            parsed = urlparse(str(url or "").strip())
        except Exception:
            return False
        host = str(parsed.hostname or "").strip().lower()
        if host != "live.restream.io":
            return False
        scheme = str(parsed.scheme or "").strip().lower()
        port = int(parsed.port or 443)
        if scheme == "rtmps" and bool(skip_restream_rtmps):
            return True
        return bool(skip_restream_443) and scheme == "rtmps" and int(port) == 443

    def add(value: Any) -> None:
        url = str(value or "").strip()
        if url and is_skipped(url):
            return
        if url and url not in out:
            out.append(url)

    add(primary)
    if isinstance(candidates, (list, tuple)):
        for item in candidates:
            add(item)
    elif isinstance(candidates, str):
        raw = str(candidates or "").strip()
        if raw.startswith("["):
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    for item in obj:
                        add(item)
                    return out
            except Exception:
                pass
        for part in raw.replace("\n", ",").split(","):
            add(part)
    return out


def _mask_rtmp_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return "<redacted>"
        path_parts = str(parsed.path or "").split("/")
        for idx in range(len(path_parts) - 1, -1, -1):
            if path_parts[idx]:
                path_parts[idx] = "***"
                break
        return urlunparse((parsed.scheme, parsed.netloc, "/".join(path_parts), "", "", ""))
    except Exception:
        return "<redacted>"


class _RtmpEdgeTick:
    __slots__ = ("audio", "audio_off", "video", "video_off")

    def __init__(self, video: bytes, audio: bytes) -> None:
        self.video = memoryview(video).cast("B")
        self.audio = memoryview(audio).cast("B")
        self.video_off = 0
        self.audio_off = 0


class RemoteEdgeSession:
    def __init__(self, cfg: EdgeSessionConfig) -> None:
        self.cfg = cfg
        self.room: Any | None = None
        self.video_source: Any | None = None
        self.audio_source: Any | None = None
        self.av_sync: Any | None = None
        self.latent_decoder: WanLatentDecoder | None = None
        self.latent_decode_queue: asyncio.Queue[_LatentDecodeJob | None] | None = None
        self.latent_decode_task: asyncio.Task[Any] | None = None
        self.latent_postprocess_queue: asyncio.Queue[_LatentPostprocessJob | None] | None = None
        self.latent_postprocess_task: asyncio.Task[Any] | None = None
        self.latent_decode_failed_exc: BaseException | None = None
        self._async_cpu_stage_streams: dict[str, Any] = {}
        self.latent_queue_max = max(1, _required_positive_int_env("REMOTE_EDGE_LATENT_QUEUE_MAX_BLOCKS"))
        self.latent_postprocess_queue_max = max(
            1,
            _safe_int_env("REMOTE_EDGE_POST_VAE_QUEUE_MAX_BLOCKS", 3),
        )
        self.async_post_vae_enabled = _env_flag(
            "REMOTE_EDGE_ASYNC_POST_VAE",
            str(os.getenv("LIVE_RAW_ASYNC_POST_VAE", "0") or "0"),
        )
        self.post_vae_fail_open = _env_flag("REMOTE_EDGE_POST_VAE_FAIL_OPEN", "1")
        self.post_vae_disabled_after_error = False
        self.latent_decode_fail_open = _env_flag("REMOTE_EDGE_LATENT_DECODE_FAIL_OPEN", "1")
        self.latent_decode_fail_open_blocks = 0
        self.latent_decode_count = 0
        self.latent_decode_last_log = 0.0
        self.rtmp_proc: subprocess.Popen[Any] | None = None
        self.rtmp_aux_procs: list[subprocess.Popen[Any]] = []
        self.rtmp_video_fd: int | None = None
        self.rtmp_audio_fd: int | None = None
        self.rtmp_log_f: Any | None = None
        self.rtmp_drain_task: asyncio.Task[Any] | None = None
        self.rtmp_writer_thread: threading.Thread | None = None
        self.rtmp_audio_writer_thread: threading.Thread | None = None
        self.rtmp_video_writer_thread: threading.Thread | None = None
        self.rtmp_writer_stop = threading.Event()
        self.rtmp_writer_cv = threading.Condition()
        self.rtmp_push_lock = asyncio.Lock()
        self.rtmp_writer_exc: BaseException | None = None
        self.rtmp_writer_started_at = 0.0
        self.rtmp_writer_last_tick_at = 0.0
        self.rtmp_writer_ticks = 0
        self.rtmp_queue: deque[_RtmpEdgeTick] = deque()
        self.rtmp_output_fps = rtmp_output_fps(int(self.cfg.fps))
        self.live_rife_backend = self._live_interpolation_backend(
            input_fps=int(self.cfg.fps),
            output_fps=int(self.rtmp_output_fps),
        )
        self.live_rife_enabled = bool(self.live_rife_backend)
        self.rtmp_pipe_fps = int(self.rtmp_output_fps if bool(self.live_rife_enabled) else int(self.cfg.fps))
        self.live_rife_blocks = 0
        self.live_rife_frames_in = 0
        self.live_rife_frames_out = 0
        self.live_rife_last_log = 0.0
        self.live_rife_last_elapsed_sec = 0.0
        self.live_rife_skip_blocks = 0
        self.live_rife_skip_frames_in = 0
        self.live_rife_skip_frames_out = 0
        self.live_rife_skip_last_log = 0.0
        self.file_work_dir = ""
        self.file_raw_video_path = ""
        self.file_raw_audio_path = ""
        self.file_mp4_path = ""
        self.file_video_only_path = ""
        self.watermark_text_file = ""
        self.file_watermark_filter_logged = False
        self.file_video_proc: subprocess.Popen | None = None
        self.file_video_log_f: Any | None = None
        self.file_stream_encode_enabled = False
        self.file_stream_video_frames_written = 0
        self.file_stream_last_frame: bytes | None = None
        self.file_motion_damping_strength = 0.0
        self.file_motion_damping_last_frame: bytes | None = None
        self.file_motion_damping_last_segment_id = ""
        self.file_motion_damping_logged = False
        self.file_motion_damping_frames = 0
        self.file_frame_width = int(self.cfg.width)
        self.file_frame_height = int(self.cfg.height)
        self.file_native_pre_resize = False
        self.file_audio_bytes = 0
        self.file_raw_video_frames = 0
        self.file_raw_video_fps = int(self.cfg.fps)
        self.file_requested_face_restore = 0.0
        self.file_requested_background_restore = 0.0
        self.file_deferred_face_restore = 0.0
        self.file_deferred_background_restore = 0.0
        self.file_deferred_post_vae_logged = False
        self.file_segment_audio_offsets: dict[str, int] = {}
        self.file_segment_frame_offsets: dict[str, int] = {}
        self.file_inline_post_vae_blocks = 0
        self.file_inline_post_vae_elapsed_sec = 0.0
        self.file_interpolation_backend = ""
        self.file_inline_interpolation_enabled = False
        self.file_inline_rife_last_source_frame: bytes | None = None
        self.file_inline_rife_last_source_tensor: Any | None = None
        self.file_inline_rife_last_continuity_key = ""
        self.file_inline_rife_boundary_resets = 0
        self.file_inline_rife_boundary_last_log = 0.0
        self.file_inline_rife_finalized = False
        self.file_post_vae_after_inline_rife_enabled = False
        self.file_rife_blocks = 0
        self.file_rife_frames_in = 0
        self.file_rife_frames_out = 0
        self.file_rife_last_log = 0.0
        self._rtmp_next_tick: float = 0.0
        self.rtmp_queue_max = max(8, _required_positive_int_env("REMOTE_EDGE_RTMP_PAIR_QUEUE"))
        if bool(self.live_rife_enabled):
            queue_default = 64 if str(self.live_rife_backend) in {"torch-rife", "nvidia-fruc"} else 512
            self.rtmp_queue_max = max(
                int(self.rtmp_queue_max),
                _safe_int_env("REMOTE_EDGE_LIVE_RIFE_QUEUE_MIN", int(queue_default)),
            )
        live_rife_ratio = max(1.0, float(self.rtmp_pipe_fps) / float(max(1, int(self.cfg.fps))))
        live_rife_queue_limited_source = max(2, int(float(self.rtmp_queue_max) / live_rife_ratio) - 4)
        self.live_rife_pairwise_enabled = bool(self._live_rife_pairwise_enabled())
        live_rife_batch_default = (
            2
            if bool(self.live_rife_pairwise_enabled)
            else (12 if str(self.live_rife_backend) in {"torch-rife", "nvidia-fruc"} else 240)
        )
        self.live_rife_batch_source_frames = max(
            2,
            min(
                int(live_rife_queue_limited_source),
                _safe_int_env("REMOTE_EDGE_LIVE_RIFE_BATCH_SOURCE_FRAMES", int(live_rife_batch_default)),
            ),
        )
        self.live_gpu_tensor_buffer_blocks = max(
            0,
            _safe_int_env("REMOTE_EDGE_LIVE_GPU_TENSOR_BUFFER_BLOCKS", 96),
        )
        self.live_rife_pre_resize = _env_flag("REMOTE_EDGE_LIVE_RIFE_PRE_RESIZE", "1")
        self.live_rife_require_tensor = _env_flag(
            "REMOTE_EDGE_LIVE_RIFE_REQUIRE_TENSOR",
            "1" if bool(self.live_rife_pre_resize) else "0",
        )
        self.live_rife_skip_on_backlog = _env_flag("REMOTE_EDGE_LIVE_RIFE_SKIP_ON_BACKLOG", "1")
        self.live_rife_skip_pending_blocks = max(
            0,
            _safe_int_env("REMOTE_EDGE_LIVE_RIFE_SKIP_PENDING_BLOCKS", 8),
        )
        self.live_rife_skip_buffered_frames = max(
            0,
            _safe_int_env("REMOTE_EDGE_LIVE_RIFE_SKIP_BUFFERED_FRAMES", max(24, int(self.cfg.fps) * 4)),
        )
        self.live_rife_skip_latent_q = max(
            0,
            _safe_int_env("REMOTE_EDGE_LIVE_RIFE_SKIP_LATENT_Q", 8),
        )
        self.live_rife_skip_rtmp_queue = max(
            0,
            _safe_int_env("REMOTE_EDGE_LIVE_RIFE_SKIP_RTMP_QUEUE", max(32, int(self.rtmp_queue_max) // 2)),
        )
        self.live_rife_skip_after_elapsed_sec = max(
            0.0,
            _safe_float_env("REMOTE_EDGE_LIVE_RIFE_SKIP_AFTER_ELAPSED_SEC", 1.2),
        )
        self.avatar_transition_blur_enabled = _env_flag("REMOTE_EDGE_AVATAR_TRANSITION_BLUR", "1")
        self.avatar_transition_style = str(
            os.getenv("REMOTE_EDGE_AVATAR_TRANSITION_STYLE", "cross_blur_punch") or "cross_blur_punch"
        ).strip().lower()
        self.avatar_transition_pre_sec = max(
            0.0,
            min(2.0, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_BLUR_IN_SEC", 0.18)),
        )
        self.avatar_transition_post_sec = max(
            0.0,
            min(2.0, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_BLUR_OUT_SEC", 0.26)),
        )
        self.avatar_transition_strength = max(
            0.0,
            min(1.0, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_BLUR_STRENGTH", 0.45)),
        )
        self.avatar_transition_sigma = max(
            0.0,
            min(64.0, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_BLUR_SIGMA", 5.5)),
        )
        self.avatar_transition_punch_zoom = max(
            1.0,
            min(1.25, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_PUNCH_ZOOM", 1.025)),
        )
        self.avatar_transition_flash_strength = max(
            0.0,
            min(0.6, _safe_float_env("REMOTE_EDGE_AVATAR_TRANSITION_FLASH_STRENGTH", 0.0)),
        )
        self.avatar_transition_silence_hold_enabled = _env_flag("REMOTE_EDGE_AVATAR_TRANSITION_SILENCE_HOLD", "0")
        self.avatar_transition_events = 0
        self.avatar_transition_last_log = 0.0
        self._last_published_avatar_ref_path = ""
        self._last_avatar_transition_source_frame: bytes | None = None
        self._avatar_transition_gap_base_frame: bytes | None = None
        self._avatar_transition_gap_frames = 0
        self._avatar_transition_silence_base_frame: bytes | None = None
        self._avatar_transition_silence_frames = 0
        self._avatar_transition_silence_ref = ""
        self._avatar_transition_silence_next_ref = ""
        self.avatar_transition_silence_events = 0
        self.avatar_transition_silence_last_log = 0.0
        if str(self.live_rife_backend or "").strip().lower() == "nvidia-fruc":
            self._validate_live_nvidia_fruc_config()
        self.rtmp_queue_drop_keep = max(1, _required_positive_int_env("REMOTE_EDGE_RTMP_DROP_KEEP_TICKS"))
        self.rtmp_video_ahead_ticks = max(1, _required_positive_int_env("REMOTE_EDGE_RTMP_VIDEO_AHEAD_TICKS"))
        self.rtmp_audio_ahead_ticks = max(1, _required_positive_int_env("REMOTE_EDGE_RTMP_AUDIO_AHEAD_TICKS"))
        self.rtmp_write_budget_sec = max(0.005, _required_positive_float_env("REMOTE_EDGE_RTMP_WRITE_BUDGET_SEC"))
        self.rtmp_audio_wait_sec = max(0.0, min(1.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_AUDIO_WAIT_SEC")))
        self.rtmp_resync_queue_sec = max(0.0, min(10.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_RESYNC_QUEUE_SEC")))
        self.rtmp_resync_keep_sec = max(0.0, min(5.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_RESYNC_KEEP_SEC")))
        self.rtmp_backpressure_fail_sec = max(
            0.0, min(120.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_BACKPRESSURE_FAIL_SEC"))
        )
        self.rtmp_candidate_preflight_sec = max(
            0.0, min(10.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_CANDIDATE_PREFLIGHT_SEC"))
        )
        self.rtmp_candidate_preflight_timeout_sec = max(
            1.0, min(30.0, _required_positive_float_env("REMOTE_EDGE_RTMP_CANDIDATE_PREFLIGHT_TIMEOUT_SEC"))
        )
        self.rtmp_preflight_frames = max(0, min(120, _required_nonnegative_int_env("REMOTE_EDGE_RTMP_PREFLIGHT_FRAMES")))
        self.rtmp_preflight_timeout_sec = max(
            1.0, min(120.0, _required_positive_float_env("REMOTE_EDGE_RTMP_PREFLIGHT_TIMEOUT_SEC"))
        )
        self.rtmp_startup_hold_sec = max(
            0.0,
            min(10.0, _safe_float_env("REMOTE_EDGE_RTMP_STARTUP_HOLD_SEC", 0.0)),
        )
        self.rtmp_startup_hold_sent = False
        self.rtmp_gap_fill_fail_sec = max(
            0.0, min(300.0, _required_nonnegative_float_env("REMOTE_EDGE_RTMP_GAP_FILL_FAIL_SEC"))
        )
        self.rtmp_reconnect_on_failure = _env_flag("REMOTE_EDGE_RTMP_RECONNECT_ON_FAILURE", "1")
        self.rtmp_reconnect_count = 0
        self.rtmp_reconnect_max = max(0, _safe_int_env("REMOTE_EDGE_RTMP_MAX_RECONNECTS", 20))
        self.rtmp_reconnect_window_sec = max(
            0.0,
            min(600.0, _safe_float_env("REMOTE_EDGE_RTMP_RECONNECT_WINDOW_SEC", 30.0)),
        )
        self.rtmp_reconnect_window_max = max(0, _safe_int_env("REMOTE_EDGE_RTMP_MAX_RECONNECTS_PER_WINDOW", 3))
        self.rtmp_reconnect_times: deque[float] = deque()
        self.rtmp_backpressure_since = 0.0
        self.rtmp_drop_count = 0
        self.rtmp_stale_drop_count = 0
        self.rtmp_drop_last_log = 0.0
        self.rtmp_resync_drop_count = 0
        self.rtmp_audio_pad_samples = 0
        self.rtmp_audio_pad_events = 0
        self.rtmp_audio_wait_events = 0
        self.rtmp_audio_wait_total_sec = 0.0
        self.rtmp_audio_hold_frames = 0
        self.rtmp_audio_hold_last_log = 0.0
        self.rtmp_gap_fill_enabled = _env_flag("REMOTE_EDGE_RTMP_GAP_FILL", "1")
        self.rtmp_require_real_audio = _env_flag("REMOTE_EDGE_RTMP_REQUIRE_REAL_AUDIO", "1")
        self.rtmp_gap_fill_frames = 0
        self.rtmp_gap_fill_streak_frames = 0
        self.rtmp_gap_fill_last_log = 0.0
        self._last_rtmp_frame: bytes | None = None
        self._last_quiet_rtmp_frame: bytes | None = None
        self._rtmp_startup_frame: bytes | None = None
        self._rtmp_idle_task: asyncio.Task[Any] | None = None
        self._rtmp_idle_stop: asyncio.Event | None = None
        self._rtmp_loud_cooldown_frames = 0
        self.rtmp_quiet_frame_max_abs = max(0, _safe_int_env("REMOTE_EDGE_RTMP_QUIET_FRAME_MAX_ABS", 96))
        self.rtmp_quiet_after_loud_frames = max(
            0,
            _safe_int_env("REMOTE_EDGE_RTMP_QUIET_AFTER_LOUD_FRAMES", max(1, int(self.cfg.fps) // 2)),
        )
        output_kind = str(cfg.output or "").strip().lower()
        subtitle_fps = int(self.rtmp_pipe_fps if output_kind == "rtmp" else int(self.cfg.fps))
        self.subtitle_renderer = SubtitleRenderer(
            width=int(self.cfg.width),
            height=int(self.cfg.height),
            fps=int(subtitle_fps),
            is_live=(str(output_kind) == "rtmp"),
        )
        self.subtitle_audio_frame_anchor = max(
            0.0,
            min(1.0, _safe_float_env("REMOTE_EDGE_SUBTITLE_AUDIO_FRAME_ANCHOR", 0.0)),
        )
        self.subtitle_frame_error_last_log = 0.0
        self.subtitle_frame_map_gap_last_log = 0.0
        self.subtitle_frame_map_gap_events = 0
        clip_timeline_fps = int(cfg.fps)
        if output_kind == "rtmp":
            clip_timeline_fps = int(self._rtmp_pipe_fps())
        elif output_kind == "file":
            clip_timeline_fps = int(self._file_requested_output_fps(input_fps=int(cfg.fps)))
        clip_exporter_mod = _clip_exporter_module() if _edge_clips_enabled() else None
        self.clip_exporter: Any | None = (
            clip_exporter_mod.EdgeClipExporter(
                clip_exporter_mod.EdgeClipContext(
                    session_id=str(cfg.session_id or ""),
                    job_id=str(cfg.job_id or ""),
                    live_session_id=str(cfg.live_session_id or cfg.session_id or ""),
                    workspace_id=str(cfg.workspace_id or ""),
                    persona_id=str(cfg.persona_id or ""),
                    output=str(cfg.output or ""),
                    width=int(cfg.width),
                    height=int(cfg.height),
                    fps=int(clip_timeline_fps),
                    sample_rate=int(cfg.sample_rate),
                )
            )
            if clip_exporter_mod is not None
            else None
        )
        self.frames = 0
        self.audio_frames = 0
        self.audio_chunks_received = 0
        self.audio_chunks_published = 0
        self.audio_seconds_published = 0.0
        self.pending_audio: deque[tuple[bytes, int]] = deque()
        self.segment_audio: dict[str, _SegmentAudio] = {}
        self.pending_blocks: deque[_PendingVideoBlock] = deque()
        self.buffered_video_frames = 0
        self._audio_video_frame_cursor = 0
        self.publish_started = False
        self.publish_stop = asyncio.Event()
        self.publish_cv = asyncio.Condition()
        self.publish_task: asyncio.Task[Any] | None = None
        self.publish_failed_exc: BaseException | None = None
        self.publish_draining = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.start_prebuffer_frames = max(0, _required_nonnegative_int_env("REMOTE_EDGE_START_PREBUFFER_FRAMES"))
        self.start_prebuffer_require_audio = _env_flag("REMOTE_EDGE_START_PREBUFFER_REQUIRE_AUDIO", "1")
        self.audio_frame_ms = max(10, min(100, _required_positive_int_env("REMOTE_EDGE_AUDIO_FRAME_MS")))
        max_buffer_sec = max(0.0, min(300.0, _required_nonnegative_float_env("REMOTE_EDGE_MAX_BUFFER_SEC")))
        self.max_buffered_video_frames = int(round(float(max_buffer_sec) * float(max(1, int(self.cfg.fps)))))
        self.adaptive_restore_enabled = _env_flag("REMOTE_EDGE_ADAPTIVE_RESTORE", "0")
        self.adaptive_background_queue_min_blocks = max(
            1,
            _required_positive_int_env("REMOTE_EDGE_ADAPTIVE_BACKGROUND_QUEUE_MIN_BLOCKS"),
        )
        self.adaptive_background_recover_queue_blocks = max(
            0,
            _required_nonnegative_int_env("REMOTE_EDGE_ADAPTIVE_BACKGROUND_RECOVER_QUEUE_BLOCKS"),
        )
        self.adaptive_background_shed_after_sec = max(
            0.0,
            _required_nonnegative_float_env("REMOTE_EDGE_ADAPTIVE_BACKGROUND_SHED_AFTER_SEC"),
        )
        self._adaptive_background_shed = False
        self._adaptive_background_pressure_since = 0.0
        self._adaptive_restore_last_log = 0.0
        self.stats_interval_sec = max(1.0, min(60.0, _required_positive_float_env("REMOTE_EDGE_STATS_INTERVAL_SEC")))
        self.receive_stats_interval_sec = max(
            1.0,
            min(60.0, _safe_float_env("REMOTE_EDGE_RECEIVE_STATS_INTERVAL_SEC", float(self.stats_interval_sec))),
        )
        self.receive_slow_warn_sec = max(0.0, min(30.0, _safe_float_env("REMOTE_EDGE_RECEIVE_SLOW_WARN_SEC", 0.25)))
        self.latent_decode_slow_warn_sec = max(
            0.0,
            min(120.0, _safe_float_env("REMOTE_EDGE_LATENT_DECODE_SLOW_WARN_SEC", 2.0)),
        )
        if int(self.max_buffered_video_frames) > 0:
            self.max_buffered_video_frames = max(
                int(self.max_buffered_video_frames),
                int(self.start_prebuffer_frames) + max(1, int(self.cfg.fps)),
            )
        self.max_buffer_last_log = 0.0
        self._last_stats_log = 0.0
        self.started_at = time.perf_counter()
        self._receive_stats_last_log = float(self.started_at)
        self._reset_receive_stats_window(self.started_at)

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        if str(self.cfg.output or "").strip().lower() == "file":
            self._start_file_output()
            return
        if str(self.cfg.output or "livekit").strip().lower() == "rtmp":
            if bool(_env_flag("REMOTE_EDGE_RTMP_OPEN_BEFORE_PREBUFFER", "0")):
                logging.warning(
                    "Remote edge RTMP opening idle publish before prebuffer: session=%s job=%s start_prebuffer_frames=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.start_prebuffer_frames),
                )
                raw_pipe_preflight = bool(_env_flag("REMOTE_EDGE_RTMP_RAW_PREFLIGHT_BEFORE_IDLE", "1"))
                await self._start_rtmp_with_preflight_candidates(raw_pipe_preflight=raw_pipe_preflight)
                if self._rtmp_startup_frame is not None or bool(_env_flag("REMOTE_EDGE_RTMP_IDLE_WITHOUT_POSTER", "0")):
                    self._start_rtmp_idle_pump()
                else:
                    logging.warning(
                        "Remote edge RTMP idle pump waiting for startup poster: session=%s job=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                    )
            else:
                logging.warning(
                    "Remote edge RTMP waiting for prebuffer before opening: session=%s job=%s start_prebuffer_frames=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.start_prebuffer_frames),
                )
            return
        rtc = _livekit_rtc()
        self.room = rtc.Room()
        opts = rtc.RoomOptions()
        opts.dynacast = False
        opts.auto_subscribe = True
        connect_timeout = max(3.0, min(60.0, _safe_float_env("REMOTE_EDGE_LIVEKIT_CONNECT_TIMEOUT_SEC", 8.0)))
        try:
            await asyncio.wait_for(
                self.room.connect(self.cfg.livekit_url, self.cfg.livekit_token, opts),
                timeout=float(connect_timeout),
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"LiveKit connect timed out after {connect_timeout:.1f}s") from e

        self.video_source = rtc.VideoSource(
            width=int(self.cfg.width),
            height=int(self.cfg.height),
            is_screencast=bool(_env_flag("REMOTE_EDGE_VIDEO_SCREENCAST", "0")),
        )
        video_track = rtc.LocalVideoTrack.create_video_track("video", self.video_source)
        vopts = rtc.TrackPublishOptions()
        vopts.source = rtc.TrackSource.SOURCE_CAMERA
        vopts.video_codec = rtc.VideoCodec.H264
        vopts.simulcast = False
        max_v_bps = _required_positive_int_env("REMOTE_EDGE_VIDEO_MAX_BITRATE_BPS")
        if max_v_bps > 0:
            vopts.video_encoding.max_bitrate = int(max_v_bps)
        vopts.video_encoding.max_framerate = int(max(1, int(self.cfg.fps)))
        await self.room.local_participant.publish_track(video_track, vopts)

        queue_ms = max(40, min(5000, _required_positive_int_env("REMOTE_EDGE_AUDIO_QUEUE_MS")))
        self.audio_source = rtc.AudioSource(
            sample_rate=int(self.cfg.sample_rate),
            num_channels=1,
            queue_size_ms=int(queue_ms),
        )
        audio_track = rtc.LocalAudioTrack.create_audio_track("audio", self.audio_source)
        aopts = rtc.TrackPublishOptions()
        aopts.source = rtc.TrackSource.SOURCE_MICROPHONE
        aopts.dtx = False
        aopts.red = True
        max_a_bps = _required_positive_int_env("REMOTE_EDGE_AUDIO_MAX_BITRATE_BPS")
        if max_a_bps > 0:
            aopts.audio_encoding.max_bitrate = int(max_a_bps)
        await self.room.local_participant.publish_track(audio_track, aopts)

        if bool(_env_flag("REMOTE_EDGE_USE_AVSYNC", "1")):
            self.av_sync = rtc.AVSynchronizer(
                audio_source=self.audio_source,
                video_source=self.video_source,
                video_fps=float(max(1, int(self.cfg.fps))),
                video_queue_size_ms=float(
                    max(50.0, min(2000.0, _required_positive_float_env("REMOTE_EDGE_AVSYNC_QUEUE_MS")))
                ),
            )
        logging.warning(
            "Remote edge LiveKit started: session=%s job=%s size=%dx%d fps=%d mode=%s avsync=%d start_prebuffer_frames=%d audio_frame_ms=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            self.cfg.mode,
            1 if self.av_sync is not None else 0,
            int(self.start_prebuffer_frames),
            int(self.audio_frame_ms),
        )

    async def close(self, *, progress_cb: Any | None = None) -> dict[str, Any] | None:
        file_result: dict[str, Any] | None = None
        if str(self.cfg.output or "").strip().lower() == "file":
            file_result = await self.finish_file_output(progress_cb=progress_cb)
        else:
            await self.drain_live_output()
            await self._hold_rtmp_end_before_close()
            self._stop_rtmp()
        if self.clip_exporter is not None:
            exporter = self.clip_exporter
            self.clip_exporter = None
            await asyncio.to_thread(exporter.close)
        if self.av_sync is not None:
            try:
                await self.av_sync.aclose()
            except Exception:
                pass
            self.av_sync = None
        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception:
                pass
            self.room = None
        return file_result

    async def _hold_rtmp_end_before_close(self) -> None:
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return
        hold_sec = max(
            0.0,
            min(60.0, _safe_float_env("REMOTE_EDGE_RTMP_END_PAD_SEC", 3.0)),
        )
        if float(hold_sec) <= 0.0:
            return
        if self.rtmp_proc is None:
            return
        frame = self._last_rtmp_frame
        if frame is None:
            return
        fps = max(1, int(self._rtmp_pipe_fps()))
        frames = int(round(float(hold_sec) * float(fps)))
        if frames <= 0:
            return
        logging.warning(
            "Remote edge RTMP end hold started: session=%s job=%s duration=%.1fs frames=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            float(hold_sec),
            int(frames),
        )
        try:
            for _ in range(int(frames)):
                audio = self._silence_audio_bytes_for_output_frames(1, fps=int(fps))
                await self.push_rtmp_frame(frame, audio=audio, gap_fill=True)
            deadline = time.monotonic() + max(15.0, float(hold_sec) + 10.0)
            await self._drain_rtmp_writer_queue(deadline=deadline)
        except Exception:
            logging.exception(
                "Remote edge RTMP end hold failed after live drain: session=%s job=%s",
                self.cfg.session_id,
                self.cfg.job_id,
            )
            return
        logging.warning(
            "Remote edge RTMP end hold complete: session=%s job=%s frames=%d rtmp_queue=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(frames),
            int(self._rtmp_queue_len()),
        )

    def _start_file_output(self) -> None:
        base = str(os.getenv("REMOTE_EDGE_FILE_OUTPUT_DIR", "/tmp/smartblog-remote-edge-files") or "").strip()
        if not base:
            base = "/tmp/smartblog-remote-edge-files"
        os.makedirs(base, exist_ok=True)
        prefix = f"file_{_sanitize_clip_component(self.cfg.session_id or self.cfg.job_id or 'session', max_len=48)}_"
        self.file_work_dir = tempfile.mkdtemp(prefix=prefix, dir=base)
        self.file_raw_video_path = os.path.join(self.file_work_dir, "video.rgb")
        self.file_raw_audio_path = os.path.join(self.file_work_dir, "audio.s16le")
        self.file_mp4_path = os.path.join(self.file_work_dir, "output.mp4")
        self.file_video_only_path = os.path.join(self.file_work_dir, "video_only.mp4")
        input_fps = max(1, int(self.cfg.fps))
        remote_finalizer = bool(self._file_remote_finalizer_enabled())
        output_fps = int(input_fps) if bool(remote_finalizer) else self._file_requested_output_fps(input_fps=int(input_fps))
        self.file_raw_video_frames = 0
        self.file_raw_video_fps = int(input_fps)
        self.file_frame_width = int(self.cfg.width)
        self.file_frame_height = int(self.cfg.height)
        self.file_native_pre_resize = bool(self._file_native_pre_resize_enabled())
        self.file_interpolation_backend = "" if bool(remote_finalizer) else self._file_interpolation_backend(
            input_fps=int(input_fps),
            output_fps=int(output_fps),
        )
        self.file_inline_interpolation_enabled = bool(str(self.file_interpolation_backend) == "torch-rife")
        if bool(self.file_inline_interpolation_enabled):
            self.file_raw_video_fps = int(output_fps)
            self.file_inline_rife_last_source_frame = None
            self.file_inline_rife_last_source_tensor = None
            self.file_inline_rife_finalized = False
        self.file_post_vae_after_inline_rife_enabled = bool(self._file_post_vae_after_inline_rife_enabled())
        self.file_stream_encode_enabled = bool(
            self.file_inline_interpolation_enabled and _env_flag("REMOTE_EDGE_FILE_STREAM_ENCODE", "0")
        )
        self.file_stream_video_frames_written = 0
        self.file_stream_last_frame = None
        self.file_remote_upscale_direct_uploaded = False
        self.file_remote_upscale_result = {}
        self.file_motion_damping_strength = max(0.0, min(0.95, _safe_float_env("REMOTE_EDGE_FILE_MOTION_DAMPING", 0.0)))
        self.file_motion_damping_last_frame = None
        self.file_motion_damping_last_segment_id = ""
        self.file_motion_damping_logged = False
        self.file_motion_damping_frames = 0
        watermark_chars = len(normalize_watermark_text(getattr(self.cfg, "watermark_text", "")))
        if bool(self.file_stream_encode_enabled) and not bool(self.file_native_pre_resize):
            self._start_file_stream_encoder()
        logging.warning(
            "Remote edge FILE output started: session=%s job=%s size=%dx%d fps=%d output_fps=%d interpolation=%s inline_interpolation=%d postvae_after_rife=%d stream_encode=%d native_pre_resize=%d remote_finalizer=%d motion_damping=%.2f watermark_chars=%d target_audio_samples=%d target_duration=%.3fs upload=%d work_dir=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.cfg.width),
            int(self.cfg.height),
            int(input_fps),
            int(output_fps),
            str(self.file_interpolation_backend or "none"),
            1 if bool(self.file_inline_interpolation_enabled) else 0,
            1 if bool(self.file_post_vae_after_inline_rife_enabled) else 0,
            1 if bool(self.file_stream_encode_enabled) else 0,
            1 if bool(self.file_native_pre_resize) else 0,
            1 if bool(remote_finalizer) else 0,
            float(self.file_motion_damping_strength),
            int(watermark_chars),
            int(self.cfg.file_target_audio_samples or 0),
            float(self.cfg.file_target_duration_sec or 0.0),
            1 if str(self.cfg.file_upload_url or "").strip() else 0,
            self.file_work_dir,
        )

    def _file_native_pre_resize_enabled(self) -> bool:
        if str(self.cfg.output or "").strip().lower() != "file":
            return False
        if bool(self._file_worker_finalizer_planned()):
            return True
        return bool(
            not bool(self._file_remote_finalizer_enabled())
            and _env_flag("REMOTE_EDGE_FILE_PRE_RESIZE", "1")
        )

    def _file_worker_finalizer_planned(self) -> bool:
        flag = getattr(self.cfg, "file_remote_finalizer", None)
        return bool(
            str(self.cfg.output or "").strip().lower() == "file"
            and flag is not None
            and not bool(flag)
            and bool(self._file_remote_upscale_url())
        )

    def _file_post_vae_after_inline_rife_enabled(self) -> bool:
        return bool(
            str(self.cfg.output or "").strip().lower() == "file"
            and bool(getattr(self, "file_inline_interpolation_enabled", False))
            and bool(getattr(self, "file_native_pre_resize", False))
            and _env_flag("REMOTE_EDGE_FILE_POST_VAE_AFTER_RIFE", "0")
        )

    def _set_file_frame_size(self, *, width: int, height: int) -> None:
        width_i = max(1, int(width))
        height_i = max(1, int(height))
        old_w = max(1, int(self.file_frame_width or self.cfg.width))
        old_h = max(1, int(self.file_frame_height or self.cfg.height))
        if int(old_w) == int(width_i) and int(old_h) == int(height_i):
            return
        if int(self.file_raw_video_frames) > 0 or int(self.file_stream_video_frames_written) > 0:
            raise RuntimeError(
                "file frame size changed after writing frames: "
                f"{old_w}x{old_h} -> {width_i}x{height_i}"
            )
        self.file_frame_width = int(width_i)
        self.file_frame_height = int(height_i)
        logging.warning(
            "Remote edge FILE native frame size selected: session=%s job=%s input=%dx%d output=%dx%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(width_i),
            int(height_i),
            int(self.cfg.width),
            int(self.cfg.height),
        )

    def _watermark_text_file(self) -> str:
        text = normalize_watermark_text(getattr(self.cfg, "watermark_text", ""))
        if not text:
            return ""
        path = str(getattr(self, "watermark_text_file", "") or "").strip()
        if not path:
            base_dir = str(self.file_work_dir or "/tmp/smartblog-remote-edge").strip()
            os.makedirs(base_dir, exist_ok=True)
            safe_session = _sanitize_clip_component(self.cfg.session_id or self.cfg.job_id or "session", max_len=48)
            path = os.path.join(base_dir, f"watermark_{safe_session}.txt")
            self.watermark_text_file = str(path)
        return write_watermark_text_file(
            path=str(path),
            text=str(text),
            width=int(self.cfg.width),
            height=int(self.cfg.height),
            env_prefixes=("REMOTE_EDGE", "SMARTBLOG"),
        )

    async def _start_rtmp_with_preflight_candidates(self, *, raw_pipe_preflight: bool = True) -> None:
        candidates = _rtmp_candidate_urls(str(self.cfg.rtmp_url or ""), self.cfg.rtmp_urls)
        if not candidates:
            raise RuntimeError("RTMP output requires rtmp_url")
        errors: list[str] = []
        total = int(len(candidates))
        for idx, url in enumerate(candidates, start=1):
            self.cfg.rtmp_url = str(url)
            try:
                if float(self.rtmp_candidate_preflight_sec) > 0.0:
                    await asyncio.to_thread(self._preflight_rtmp_candidate_url, str(url), int(idx), int(total))
                self._start_rtmp(attempt=int(idx), total=int(total))
                # Endpoint preflight only proves that Restream accepts a short
                # standalone lavfi publish. The real rawvideo/audio pipe can
                # still block while ffmpeg opens the RTMP output, so validate
                # the actual pipe before accepting the producer session.
                if bool(raw_pipe_preflight):
                    await self._preflight_rtmp()
                if int(idx) > 1:
                    logging.warning(
                        "Remote edge RTMP fallback selected: session=%s job=%s attempt=%d/%d target=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        int(idx),
                        int(total),
                        _mask_rtmp_url(str(url)),
                    )
                return
            except Exception as e:
                errors.append(f"{_mask_rtmp_url(str(url))}: {e}")
                logging.warning(
                    "Remote edge RTMP candidate failed: session=%s job=%s attempt=%d/%d target=%s err=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(idx),
                    int(total),
                    _mask_rtmp_url(str(url)),
                    e,
                )
                self._stop_rtmp()
                if int(idx) >= int(total):
                    break
                # A failed RTMP pipe preflight can set publish_stop from the
                # writer thread. That failure belongs only to this candidate;
                # the next candidate must be able to start publish/decode loops.
                self.publish_stop.clear()
                self.rtmp_writer_exc = None
        label = "RTMP preflight" if bool(raw_pipe_preflight) else "RTMP open"
        raise RuntimeError(f"{label} failed for all candidates: " + " | ".join(errors))

    async def _ensure_rtmp_started_for_publish(self) -> None:
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return
        proc = self.rtmp_proc
        if proc is not None and proc.poll() is None:
            return
        if proc is not None or self.rtmp_video_fd is not None or self.rtmp_audio_fd is not None:
            self._stop_rtmp()
        logging.warning(
            "Remote edge RTMP opening after prebuffer: session=%s job=%s buffered_frames=%d pending_blocks=%d pending_audio_sec=%.2f",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.buffered_video_frames),
            int(len(self.pending_blocks)),
            float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
        )
        await self._start_rtmp_with_preflight_candidates(raw_pipe_preflight=True)
        await self._send_rtmp_startup_hold_once()

    async def _recover_rtmp_after_failure(self, failed: BaseException | str | None) -> None:
        if not bool(self.rtmp_reconnect_on_failure):
            if isinstance(failed, BaseException):
                raise RuntimeError(f"RTMP writer failed: {failed}") from failed
            raise RuntimeError(f"RTMP writer failed: {failed or 'unknown failure'}")
        self.rtmp_reconnect_count += 1
        now = time.monotonic()
        self.rtmp_reconnect_times.append(float(now))
        window = float(self.rtmp_reconnect_window_sec)
        if window > 0.0:
            cutoff = float(now - window)
            while self.rtmp_reconnect_times and float(self.rtmp_reconnect_times[0]) < cutoff:
                self.rtmp_reconnect_times.popleft()
        window_count = int(len(self.rtmp_reconnect_times))
        if int(self.rtmp_reconnect_max) > 0 and int(self.rtmp_reconnect_count) > int(self.rtmp_reconnect_max):
            raise RuntimeError(
                "remote_edge_unavailable: RTMP reconnect limit exceeded "
                f"count={self.rtmp_reconnect_count} max={self.rtmp_reconnect_max} "
                f"last_error={failed or 'unknown failure'}"
            )
        if (
            int(self.rtmp_reconnect_window_max) > 0
            and window > 0.0
            and int(window_count) > int(self.rtmp_reconnect_window_max)
        ):
            raise RuntimeError(
                "remote_edge_unavailable: RTMP reconnect burst limit exceeded "
                f"count={window_count} max={self.rtmp_reconnect_window_max} window={window:.1f}s "
                f"last_error={failed or 'unknown failure'}"
            )
        logging.warning(
            "Remote edge RTMP reconnecting after writer failure: session=%s job=%s reconnect=%d window_reconnects=%d err=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.rtmp_reconnect_count),
            int(window_count),
            str(failed or "unknown failure"),
        )
        self._stop_rtmp()
        self.publish_stop.clear()
        self.rtmp_writer_exc = None
        self.rtmp_backpressure_since = 0.0
        await self._start_rtmp_with_preflight_candidates(raw_pipe_preflight=True)
        self._rtmp_next_tick = float(time.perf_counter())

    def _raise_if_publish_failed(self) -> None:
        if self.publish_failed_exc is not None:
            raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc

    def set_startup_rgb24(self, payload: bytes) -> None:
        self._raise_if_publish_failed()
        expected = int(self.cfg.width * self.cfg.height * 3)
        if len(payload) != expected:
            raise ValueError(f"invalid poster rgb24 size: got={len(payload)} expected={expected}")
        self._rtmp_startup_frame = bytes(payload)
        self._last_quiet_rtmp_frame = bytes(payload)
        if (
            str(self.cfg.output or "livekit").strip().lower() == "rtmp"
            and not bool(self.publish_started)
            and bool(_env_flag("REMOTE_EDGE_RTMP_OPEN_BEFORE_PREBUFFER", "0"))
        ):
            self._start_rtmp_idle_pump()

    def _rtmp_idle_frame(self) -> bytes:
        frame = self._rtmp_startup_frame or self._last_quiet_rtmp_frame or self._last_rtmp_frame
        if frame is not None:
            return bytes(frame)
        return bytes(int(self.cfg.width * self.cfg.height * 3))

    def _start_rtmp_idle_pump(self) -> None:
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return
        if self._rtmp_idle_task is not None and not self._rtmp_idle_task.done():
            return
        self._rtmp_idle_stop = asyncio.Event()
        self._rtmp_idle_task = asyncio.create_task(
            self._rtmp_idle_loop(),
            name=f"remote-edge-rtmp-idle-{self.cfg.session_id}",
        )

    async def _stop_rtmp_idle_pump(self) -> None:
        stop = self._rtmp_idle_stop
        if stop is not None:
            stop.set()
        task = self._rtmp_idle_task
        self._rtmp_idle_task = None
        self._rtmp_idle_stop = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _rtmp_idle_loop(self) -> None:
        try:
            fps = max(1, int(self._rtmp_pipe_fps()))
            frame_dt = 1.0 / float(fps)
            while not self.publish_stop.is_set():
                stop = self._rtmp_idle_stop
                if stop is not None and stop.is_set():
                    break
                if bool(self.publish_started):
                    break
                frame = self._rtmp_idle_frame()
                audio = self._silence_audio_bytes_for_output_frames(1, fps=int(fps))
                await self.push_rtmp_frame(frame, audio=audio, gap_fill=False)
                await asyncio.sleep(float(frame_dt))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.publish_stop.is_set():
                return
            self.publish_failed_exc = e
            self.publish_stop.set()
            async with self.publish_cv:
                self.publish_cv.notify_all()
            logging.exception("Remote edge RTMP idle pump failed: session=%s job=%s", self.cfg.session_id, self.cfg.job_id)

    async def _send_rtmp_startup_hold_once(self) -> None:
        if bool(self.rtmp_startup_hold_sent):
            return
        self.rtmp_startup_hold_sent = True
        hold_sec = float(getattr(self, "rtmp_startup_hold_sec", 0.0) or 0.0)
        if hold_sec <= 0.0:
            return
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return
        proc = self.rtmp_proc
        if proc is None or proc.poll() is not None:
            return
        fps = max(1, int(self._rtmp_pipe_fps()))
        frames = max(1, int(round(float(hold_sec) * float(fps))))
        logging.warning(
            "Remote edge RTMP startup hold started: session=%s job=%s duration=%.2fs frames=%d pending_blocks=%d buffered_frames=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            float(hold_sec),
            int(frames),
            int(len(self.pending_blocks)),
            int(self.buffered_video_frames),
        )
        for _ in range(int(frames)):
            if self.publish_stop.is_set():
                break
            frame = self._rtmp_idle_frame()
            audio = self._silence_audio_bytes_for_output_frames(1, fps=int(fps))
            await self.push_rtmp_frame(frame, audio=audio, gap_fill=False)
            await asyncio.sleep(1.0 / float(fps))
        logging.warning(
            "Remote edge RTMP startup hold complete: session=%s job=%s frames=%d rtmp_queue=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(frames),
            int(self._rtmp_queue_len()),
        )

    def _preflight_rtmp_candidate_url(self, url: str, attempt: int, total: int) -> None:
        duration = float(self.rtmp_candidate_preflight_sec)
        timeout = max(float(self.rtmp_candidate_preflight_timeout_sec), duration + 3.0)
        output_fps = rtmp_output_fps(int(self.cfg.fps))
        gop = max(1, int(round(float(output_fps) * 2.0)))
        target = str(url or "").strip()
        if not target:
            raise RuntimeError("empty RTMP candidate URL")
        canvas_w, canvas_h, canvas_filters = resolve_rtmp_publish_canvas(
            width=int(self.cfg.width),
            height=int(self.cfg.height),
        )
        video_filters = [f"fps={output_fps}"]
        video_filters.extend(canvas_filters)
        video_filters.extend(["setsar=1", f"setdar=dar={int(canvas_w)}/{int(canvas_h)}"])
        logging.warning(
            "Remote edge RTMP candidate endpoint preflight started: session=%s job=%s attempt=%d/%d target=%s duration=%.1fs timeout=%.1fs",
            self.cfg.session_id,
            self.cfg.job_id,
            int(attempt),
            int(total),
            _mask_rtmp_url(target),
            duration,
            timeout,
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-re",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={int(self.cfg.width)}x{int(self.cfg.height)}:r={max(1, int(self.cfg.fps))}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={max(1, int(self.cfg.sample_rate))}",
            "-t",
            f"{duration:.3f}",
            "-vf",
            ",".join(video_filters),
            "-r",
            str(output_fps),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "750k",
            "-maxrate",
            "750k",
            "-bufsize",
            "1500k",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "flv",
            target,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"RTMP candidate endpoint preflight timed out after {timeout:.1f}s") from e
        if int(proc.returncode) != 0:
            tail = " ".join(str(proc.stderr or "").strip().splitlines()[-3:])
            tail = tail.replace(target, _mask_rtmp_url(target))
            raise RuntimeError(f"RTMP candidate endpoint preflight failed rc={proc.returncode}: {tail or 'no stderr'}")
        logging.warning(
            "Remote edge RTMP candidate endpoint preflight ok: session=%s job=%s attempt=%d/%d target=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(attempt),
            int(total),
            _mask_rtmp_url(target),
        )

    def _start_rtmp(self, *, attempt: int = 1, total: int = 1) -> None:
        if not str(self.cfg.rtmp_url or "").strip():
            raise RuntimeError("RTMP output requires rtmp_url")
        video_r, video_w = os.pipe()
        audio_r, audio_w = os.pipe()
        log_dir = os.path.join("/tmp", "smartblog-remote-edge")
        os.makedirs(log_dir, exist_ok=True)
        safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(self.cfg.session_id or "session"))
        log_path = os.path.join(log_dir, f"ffmpeg_rtmp_{safe_session}.log")
        diag_flv_path = ""
        if bool(_env_flag("REMOTE_EDGE_RTMP_DIAG_FLV", "0")):
            diag_dir = str(os.getenv("REMOTE_EDGE_RTMP_DIAG_DIR", log_dir) or log_dir).strip() or log_dir
            os.makedirs(diag_dir, exist_ok=True)
            diag_flv_path = os.path.join(diag_dir, f"ffmpeg_rtmp_{safe_session}.flv")
        self.rtmp_log_f = open(log_path, "a", encoding="utf-8")
        try:
            clip_segment_pattern = ""
            clip_segment_time_sec = 0.0
            if self.clip_exporter is not None:
                try:
                    clip_segment_pattern = str(self.clip_exporter.rtmp_segment_pattern() or "")
                    clip_segment_time_sec = float(self.clip_exporter.rtmp_segment_time_sec() or 0.0)
                    if clip_segment_pattern:
                        os.makedirs(os.path.dirname(os.path.abspath(clip_segment_pattern)) or ".", exist_ok=True)
                except Exception:
                    logging.exception(
                        "Remote edge clip ring setup failed: session=%s job=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                    )
                    clip_segment_pattern = ""
                    clip_segment_time_sec = 0.0
            cmd = build_rtmp_ffmpeg_cmd(
                width=int(self.cfg.width),
                height=int(self.cfg.height),
                fps=int(self._rtmp_pipe_fps()),
                sample_rate=int(self.cfg.sample_rate),
                video_fifo=f"pipe:{int(video_r)}",
                audio_fifo=f"pipe:{int(audio_r)}",
                rtmp_url=str(self.cfg.rtmp_url),
                segment_pattern=clip_segment_pattern or None,
                segment_time_sec=clip_segment_time_sec or None,
                input_readrate=False,
                diagnostic_flv_path=diag_flv_path or None,
                watermark_text_file=self._watermark_text_file(),
            )
            self.rtmp_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self.rtmp_log_f,
                pass_fds=(int(video_r), int(audio_r)),
                close_fds=True,
            )
        finally:
            for fd in (video_r, audio_r):
                try:
                    os.close(fd)
                except Exception:
                    pass
        self.rtmp_video_fd = int(video_w)
        self.rtmp_audio_fd = int(audio_w)
        os.set_blocking(int(video_w), False)
        os.set_blocking(int(audio_w), False)
        self._rtmp_next_tick = 0.0
        self.rtmp_writer_stop.clear()
        self.rtmp_writer_exc = None
        self.rtmp_writer_started_at = time.monotonic()
        self.rtmp_writer_last_tick_at = self.rtmp_writer_started_at
        self.rtmp_writer_ticks = 0
        self.rtmp_audio_writer_thread = threading.Thread(
            target=self._rtmp_writer_loop,
            args=("audio",),
            name=f"remote-edge-rtmp-audio-writer-{self.cfg.session_id}",
            daemon=True,
        )
        self.rtmp_video_writer_thread = threading.Thread(
            target=self._rtmp_writer_loop,
            args=("video",),
            name=f"remote-edge-rtmp-video-writer-{self.cfg.session_id}",
            daemon=True,
        )
        self.rtmp_writer_thread = self.rtmp_video_writer_thread
        self.rtmp_audio_writer_thread.start()
        self.rtmp_video_writer_thread.start()
        canvas_w, canvas_h, canvas_filters = resolve_rtmp_publish_canvas(width=int(self.cfg.width), height=int(self.cfg.height))
        uses_visual_pad = "pad=" in ",".join(str(item) for item in canvas_filters)
        logging.warning(
            "Remote edge RTMP started: session=%s job=%s attempt=%d/%d target=%s size=%dx%d rtmp_canvas=%dx%d rtmp_pad=%d source_fps=%d pipe_fps=%d live_rife=%d live_rife_backend=%s live_rife_pairwise=%d live_rife_batch=%d ffmpeg_readrate=%d mode=%s start_prebuffer_frames=%d start_require_audio=%d real_audio_lock=%d audio_frame_ms=%d queue_max=%d drop_keep=%d audio_ahead=%d video_ahead=%d audio_wait=%.3fs resync_queue=%.2fs diag_flv=%s log=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(attempt),
            int(total),
            _mask_rtmp_url(str(self.cfg.rtmp_url)),
            self.cfg.width,
            self.cfg.height,
            int(canvas_w),
            int(canvas_h),
            int(bool(uses_visual_pad)),
            int(self.cfg.fps),
            int(self._rtmp_pipe_fps()),
            int(bool(self.live_rife_enabled)),
            str(self.live_rife_backend or "none"),
            int(bool(self._live_rife_pairwise_enabled())),
            int(self.live_rife_batch_source_frames),
            int(bool(self.live_rife_enabled)),
            self.cfg.mode,
            int(self.start_prebuffer_frames),
            int(bool(self.start_prebuffer_require_audio)),
            int(bool(self.rtmp_require_real_audio)),
            int(self.audio_frame_ms),
            int(self.rtmp_queue_max),
            int(self.rtmp_queue_drop_keep),
            int(self.rtmp_audio_ahead_ticks),
            int(self.rtmp_video_ahead_ticks),
            float(self.rtmp_audio_wait_sec),
            float(self.rtmp_resync_queue_sec),
            str(diag_flv_path or "-"),
            str(log_path),
        )
        if self.clip_exporter is not None and str(self.clip_exporter.rtmp_segment_pattern() or "").strip():
            logging.warning(
                "Remote edge RTMP encoded clip ring started: session=%s job=%s pattern=%s segment_sec=%.3f",
                self.cfg.session_id,
                self.cfg.job_id,
                str(self.clip_exporter.rtmp_segment_pattern()),
                float(self.clip_exporter.rtmp_segment_time_sec()),
            )

    def _stop_rtmp(self) -> None:
        stop = self._rtmp_idle_stop
        if stop is not None:
            stop.set()
        self.rtmp_writer_stop.set()
        with self.rtmp_writer_cv:
            self.rtmp_writer_cv.notify_all()
        for attr in ("rtmp_video_fd", "rtmp_audio_fd"):
            fd = getattr(self, attr, None)
            setattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(int(fd))
                except Exception:
                    pass
        proc = self.rtmp_proc
        self.rtmp_proc = None
        aux_procs = list(self.rtmp_aux_procs)
        self.rtmp_aux_procs = []
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        for aux in aux_procs:
            try:
                if aux.poll() is None:
                    aux.terminate()
            except Exception:
                pass
        for aux in aux_procs:
            try:
                aux.wait(timeout=3.0)
            except Exception:
                try:
                    aux.kill()
                except Exception:
                    pass
        threads = [
            self.rtmp_audio_writer_thread,
            self.rtmp_video_writer_thread,
        ]
        self.rtmp_audio_writer_thread = None
        self.rtmp_video_writer_thread = None
        self.rtmp_writer_thread = None
        for thread in threads:
            if thread is None or not thread.is_alive():
                continue
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        log_f = self.rtmp_log_f
        self.rtmp_log_f = None
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        if self.clip_exporter is not None:
            try:
                self.clip_exporter.mark_encoded_ring_finalized()
            except Exception:
                pass
        with self.rtmp_writer_cv:
            self.rtmp_queue.clear()
        self._last_rtmp_frame = None
        self._last_quiet_rtmp_frame = None
        self._rtmp_loud_cooldown_frames = 0

    def _rtmp_writer_failed(self) -> BaseException | None:
        exc = self.rtmp_writer_exc
        if exc is not None:
            return exc
        proc = self.rtmp_proc
        try:
            if proc is not None and proc.poll() is not None:
                return RuntimeError(f"RTMP ffmpeg exited rc={proc.poll()}")
        except Exception:
            pass
        for idx, aux in enumerate(list(self.rtmp_aux_procs), start=1):
            try:
                if aux.poll() is not None:
                    return RuntimeError(f"RTMP auxiliary process {idx} exited rc={aux.poll()}")
            except Exception:
                pass
        return None

    def _notify_publish_stopped_from_thread(self) -> None:
        self.publish_stop.set()

        async def notify() -> None:
            async with self.publish_cv:
                self.publish_cv.notify_all()

        asyncio.create_task(notify())

    def _rtmp_writer_loop(self, kind: str) -> None:
        try:
            while not self.rtmp_writer_stop.is_set():
                with self.rtmp_writer_cv:
                    while not self._rtmp_has_pending_kind_locked(kind) and not self.rtmp_writer_stop.is_set():
                        self.rtmp_writer_cv.wait(timeout=0.100)
                    if self.rtmp_writer_stop.is_set():
                        break
                self._drain_rtmp_queue(kind=kind)
                if self._rtmp_has_pending_kind(kind):
                    time.sleep(0.001)
        except BaseException as e:
            if self.rtmp_writer_stop.is_set():
                return
            self.rtmp_writer_exc = e
            loop = self.loop
            if bool(self.rtmp_reconnect_on_failure):
                if loop is not None and loop.is_running():
                    async def notify() -> None:
                        async with self.publish_cv:
                            self.publish_cv.notify_all()

                    loop.call_soon_threadsafe(lambda: asyncio.create_task(notify()))
            elif loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._notify_publish_stopped_from_thread)
            else:
                self.publish_stop.set()
            logging.exception(
                "Remote edge RTMP %s writer failed: session=%s job=%s",
                kind,
                self.cfg.session_id,
                self.cfg.job_id,
            )

    def _rtmp_queue_len(self) -> int:
        with self.rtmp_writer_cv:
            return int(len(self.rtmp_queue))

    async def _preflight_rtmp(self) -> None:
        frames = int(self.rtmp_preflight_frames)
        if frames <= 0:
            return
        start_ticks = int(self.rtmp_writer_ticks)
        expected = int(self.cfg.width * self.cfg.height * 3)
        frame = self._rtmp_idle_frame()
        if len(frame) != expected:
            frame = bytes(expected)
        samples = max(1, int(round(float(self.cfg.sample_rate) / float(max(1, int(self._rtmp_pipe_fps()))))))
        silence = bytes(int(samples) * 2)
        logging.warning(
            "Remote edge RTMP preflight started: session=%s job=%s frames=%d timeout=%.1fs poster=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(frames),
            float(self.rtmp_preflight_timeout_sec),
            int(self._rtmp_startup_frame is not None),
        )
        for _ in range(int(frames)):
            await self.push_rtmp_frame(frame, audio=silence)
        deadline = time.monotonic() + float(self.rtmp_preflight_timeout_sec)
        while time.monotonic() < deadline:
            failed = self._rtmp_writer_failed()
            if failed is not None:
                raise RuntimeError(f"RTMP preflight failed: {failed}") from failed
            written = int(self.rtmp_writer_ticks) - int(start_ticks)
            if int(written) >= int(frames):
                logging.warning(
                    "Remote edge RTMP preflight ok: session=%s job=%s frames=%d writer_ticks=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(frames),
                    int(self.rtmp_writer_ticks),
                )
                self.frames = 0
                self.audio_frames = 0
                self.audio_seconds_published = 0.0
                self._last_rtmp_frame = None
                self._last_quiet_rtmp_frame = None
                self._rtmp_loud_cooldown_frames = 0
                self._rtmp_next_tick = 0.0
                return
            await asyncio.sleep(0.050)
        raise RuntimeError(
            "RTMP preflight timed out "
            f"after {self.rtmp_preflight_timeout_sec:.1f}s "
            f"writer_ticks={int(self.rtmp_writer_ticks)} queue={self._rtmp_queue_len()}"
        )

    async def _rtmp_drain_loop(self) -> None:
        try:
            while not self.publish_stop.is_set():
                if self.rtmp_queue:
                    self._drain_rtmp_queue()
                    await asyncio.sleep(0.001 if self.rtmp_queue else 0.0)
                else:
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Remote edge RTMP drain loop failed: session=%s job=%s", self.cfg.session_id, self.cfg.job_id)

    @staticmethod
    def _write_some_fd(fd: int, view: memoryview, off: int, deadline: float) -> tuple[int, bool]:
        while off < len(view):
            try:
                n = os.write(int(fd), view[off:])
            except BlockingIOError:
                remaining = float(deadline - time.monotonic())
                if remaining <= 0.0:
                    return off, False
                select.select([], [int(fd)], [], min(0.005, float(remaining)))
                continue
            if n <= 0:
                return off, False
            off += int(n)
            if time.monotonic() >= float(deadline):
                return off, False
        return off, True

    def _rtmp_has_pending_kind_locked(self, kind: str) -> bool:
        if not self.rtmp_queue:
            return False
        attr = "audio_off" if kind == "audio" else "video_off"
        data_attr = "audio" if kind == "audio" else "video"
        ahead = self.rtmp_audio_ahead_ticks if kind == "audio" else self.rtmp_video_ahead_ticks
        for tick in list(self.rtmp_queue)[: int(max(1, int(ahead)))]:
            if int(getattr(tick, attr)) < len(getattr(tick, data_attr)):
                return True
        return False

    def _rtmp_has_pending_kind(self, kind: str) -> bool:
        with self.rtmp_writer_cv:
            return self._rtmp_has_pending_kind_locked(kind)

    def _mark_rtmp_completed_locked(self) -> None:
        completed = 0
        while self.rtmp_queue:
            head = self.rtmp_queue[0]
            if head.audio_off < len(head.audio) or head.video_off < len(head.video):
                break
            self.rtmp_queue.popleft()
            completed += 1
        if completed > 0:
            self.rtmp_writer_ticks += int(completed)
            self.rtmp_writer_last_tick_at = time.monotonic()
            self.rtmp_writer_cv.notify_all()

    def _drain_rtmp_queue(self, *, kind: str | None = None) -> None:
        if kind is None:
            self._drain_rtmp_queue(kind="audio")
            self._drain_rtmp_queue(kind="video")
            return
        with self.rtmp_writer_cv:
            if self.rtmp_video_fd is None or self.rtmp_audio_fd is None:
                return
            fd = int(self.rtmp_audio_fd if kind == "audio" else self.rtmp_video_fd)
            off_attr = "audio_off" if kind == "audio" else "video_off"
            data_attr = "audio" if kind == "audio" else "video"
            ahead = self.rtmp_audio_ahead_ticks if kind == "audio" else self.rtmp_video_ahead_ticks
            fps = max(1, int(self._rtmp_pipe_fps()))
            frame_dt = 1.0 / float(fps)
            deadline = time.monotonic() + min(float(frame_dt), float(self.rtmp_write_budget_sec))
            while self.rtmp_queue and time.monotonic() < deadline:
                # Audio and video are drained by separate writer threads. This
                # lets AAC receive enough samples to open the muxer even when
                # the large raw-video pipe is temporarily backpressured.
                progressed = False
                for tick in list(self.rtmp_queue)[: int(max(1, int(ahead)))]:
                    if time.monotonic() >= deadline:
                        return
                    off = int(getattr(tick, off_attr))
                    data = getattr(tick, data_attr)
                    if off < len(data):
                        new_off, done = self._write_some_fd(
                            fd,
                            data,
                            off,
                            deadline,
                        )
                        setattr(tick, off_attr, int(new_off))
                        progressed = progressed or int(new_off) > int(off)
                        if not done:
                            break
                self._mark_rtmp_completed_locked()
                if progressed:
                    continue
                return

    def _resync_rtmp_queue_if_needed(self) -> None:
        if float(self.rtmp_resync_queue_sec) <= 0.0:
            return
        fps = max(1, int(self.cfg.fps))
        max_ticks = int(max(1, round(float(self.rtmp_resync_queue_sec) * float(fps))))
        if len(self.rtmp_queue) <= max_ticks:
            return
        keep_ticks = int(max(1, round(float(self.rtmp_resync_keep_sec) * float(fps))))
        keep_ticks = int(min(max_ticks, max(1, keep_ticks)))
        dropped = 0
        while len(self.rtmp_queue) > keep_ticks:
            head = self.rtmp_queue[0]
            if int(head.audio_off) <= 0 and int(head.video_off) <= 0:
                self.rtmp_queue.popleft()
                dropped += 1
                continue
            if len(self.rtmp_queue) <= 1:
                break
            del self.rtmp_queue[1]
            dropped += 1
        if dropped > 0:
            self.rtmp_resync_drop_count += int(dropped)
            logging.warning(
                "Remote edge RTMP resync dropped stale A/V ticks: session=%s job=%s dropped=%d queue=%d keep=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(dropped),
                int(len(self.rtmp_queue)),
                int(keep_ticks),
            )

    async def push_rgb24(self, payload: bytes, *, timestamp_us: int | None = None) -> None:
        if self.video_source is None:
            raise RuntimeError("video source is not started")
        expected = int(self.cfg.width * self.cfg.height * 3)
        if len(payload) != expected:
            raise ValueError(f"invalid rgb24 frame size: got={len(payload)} expected={expected}")
        rtc = _livekit_rtc()
        vf = rtc.VideoFrame(self.cfg.width, self.cfg.height, rtc.VideoBufferType.RGB24, payload)
        if self.av_sync is not None:
            ts = None if timestamp_us is None else float(timestamp_us) / 1_000_000.0
            await self.av_sync.push(vf, timestamp=ts)
        else:
            if timestamp_us is None:
                self.video_source.capture_frame(vf)
            else:
                self.video_source.capture_frame(vf, timestamp_us=int(timestamp_us))
        self.frames += 1

    @staticmethod
    def _pcm16_max_abs(payload: bytes) -> int:
        even = (len(payload) // 2) * 2
        if even <= 0:
            return 0
        samples = array.array("h")
        samples.frombytes(payload[:even])
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0
        return max(abs(int(v)) for v in samples)

    def _remember_quiet_rtmp_frame(self, video: bytes, audio: bytes, *, gap_fill: bool) -> None:
        if bool(gap_fill):
            return
        max_abs = self._pcm16_max_abs(bytes(audio or b""))
        if int(max_abs) > int(self.rtmp_quiet_frame_max_abs):
            self._rtmp_loud_cooldown_frames = int(self.rtmp_quiet_after_loud_frames)
            return
        if int(self._rtmp_loud_cooldown_frames) > 0:
            self._rtmp_loud_cooldown_frames -= 1
            return
        self._last_quiet_rtmp_frame = bytes(video)

    def _rtmp_gap_frame(self, reason: str) -> bytes | None:
        _ = reason
        # Once real publishing has started, every gap-fill must freeze the
        # exact last frame. Quiet/startup frames can be from an older idle pose
        # and look like random jumps when audio or video briefly stalls.
        frame = self._last_rtmp_frame
        if frame is None:
            return None
        self._avatar_transition_gap_base_frame = None
        self._avatar_transition_gap_frames = 0
        return bytes(frame)

    def _reset_avatar_transition_silence_hold(self) -> None:
        self._avatar_transition_silence_base_frame = None
        self._avatar_transition_silence_frames = 0
        self._avatar_transition_silence_ref = ""
        self._avatar_transition_silence_next_ref = ""

    def _apply_avatar_transition_silence_hold(
        self,
        frame: bytes,
        audio: bytes,
        *,
        current_avatar_ref_path: str = "",
        next_avatar_ref_path: str = "",
        segment_audio: _SegmentAudio | None = None,
        segment_audio_offset_before: int | None = None,
    ) -> bytes:
        if not bool(getattr(self, "avatar_transition_silence_hold_enabled", False)):
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        if not bool(getattr(self, "avatar_transition_blur_enabled", False)):
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        if float(getattr(self, "avatar_transition_strength", 0.0) or 0.0) <= 0.0:
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        expected = int(self.cfg.width) * int(self.cfg.height) * 3
        if len(frame) != expected:
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        current_ref = str(current_avatar_ref_path or self._last_published_avatar_ref_path or "").strip()
        next_ref = str(next_avatar_ref_path or "").strip()
        if not current_ref or not next_ref or current_ref == next_ref:
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        if int(self._pcm16_max_abs(bytes(audio or b""))) > int(self.rtmp_quiet_frame_max_abs):
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        if not self._is_segment_audio_tail_for_avatar_transition(
            segment_audio,
            segment_audio_offset_before=segment_audio_offset_before,
            audio=audio,
        ):
            self._reset_avatar_transition_silence_hold()
            return bytes(frame)
        if (
            self._avatar_transition_silence_base_frame is None
            or str(self._avatar_transition_silence_ref) != current_ref
            or str(self._avatar_transition_silence_next_ref) != next_ref
        ):
            self._avatar_transition_silence_base_frame = bytes(frame)
            self._avatar_transition_silence_frames = 0
            self._avatar_transition_silence_ref = str(current_ref)
            self._avatar_transition_silence_next_ref = str(next_ref)
            self.avatar_transition_silence_events += 1
            now_log = time.monotonic()
            if now_log - float(self.avatar_transition_silence_last_log) >= 5.0:
                logging.warning(
                    "Remote edge avatar transition silence hold: session=%s job=%s events=%d ref=%s next=%s pre=%.2fs strength=%.2f",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.avatar_transition_silence_events),
                    os.path.basename(str(current_ref)),
                    os.path.basename(str(next_ref)),
                    float(self.avatar_transition_pre_sec),
                    float(self.avatar_transition_strength),
                )
                self.avatar_transition_silence_last_log = float(now_log)
        self._avatar_transition_silence_frames += 1
        fps = int(max(1, int(self._rtmp_pipe_fps())))
        ramp_frames = max(1, int(round(float(max(0.05, float(self.avatar_transition_pre_sec))) * float(fps))))
        progress = min(1.0, float(self._avatar_transition_silence_frames) / float(ramp_frames))
        weight = float(self.avatar_transition_strength) * self._smoothstep(float(progress))
        return self._compose_avatar_transition_frame(
            bytes(self._avatar_transition_silence_base_frame),
            blur_weight=float(weight),
            old_frame=None,
            old_weight=0.0,
            punch_weight=0.0,
            flash_weight=0.0,
        )

    def _is_segment_audio_tail_for_avatar_transition(
        self,
        segment_audio: _SegmentAudio | None,
        *,
        segment_audio_offset_before: int | None,
        audio: bytes,
    ) -> bool:
        if segment_audio is None:
            return False
        try:
            total_samples = int(len(segment_audio.payload or b"") // 2)
        except Exception:
            total_samples = 0
        if total_samples <= 0:
            return False
        try:
            offset_samples = int(max(0, int(segment_audio_offset_before or 0) // 2))
        except Exception:
            offset_samples = 0
        current_samples = int(max(0, len(bytes(audio or b"")) // 2))
        if current_samples <= 0:
            return False
        end_samples = int(min(int(total_samples), int(offset_samples) + int(current_samples)))
        sample_rate = int(getattr(segment_audio, "sample_rate", None) or self.cfg.sample_rate or 0)
        if sample_rate <= 0:
            return False
        tail_sec = max(0.05, float(self.avatar_transition_pre_sec) + (1.0 / float(max(1, int(self._rtmp_pipe_fps())))))
        tail_samples = int(round(float(tail_sec) * float(sample_rate)))
        return bool(int(end_samples) >= max(0, int(total_samples) - int(tail_samples)))

    def _raise_rtmp_media_runway_invariant(self, reason: str, **fields: Any) -> None:
        detail = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        raise RuntimeError(
            "Remote edge RTMP media runway invariant violated: "
            f"reason={str(reason or 'unknown')} session={self.cfg.session_id} job={self.cfg.job_id}"
            + (f" {detail}" if detail else "")
        )

    async def push_rtmp_frame(self, payload: bytes, *, audio: bytes, gap_fill: bool = False) -> None:
        expected = int(self.cfg.width * self.cfg.height * 3)
        if len(payload) != expected:
            raise ValueError(f"invalid rgb24 frame size: got={len(payload)} expected={expected}")
        failed = self._rtmp_writer_failed()
        if failed is not None:
            await self._recover_rtmp_after_failure(failed)
        proc = self.rtmp_proc
        if proc is None or proc.poll() is not None:
            await self._recover_rtmp_after_failure(
                f"RTMP ffmpeg is not running rc={None if proc is None else proc.poll()}"
            )
        if self.rtmp_video_fd is None or self.rtmp_audio_fd is None:
            await self._recover_rtmp_after_failure("RTMP pipe is not open")
        python_pacing = bool(self._rtmp_python_pacing_enabled())
        if bool(python_pacing):
            if float(self._rtmp_next_tick) <= 0.0:
                self._rtmp_next_tick = float(time.perf_counter())
            now = float(time.perf_counter())
            wait = float(self._rtmp_next_tick - now)
            if wait > 0.0:
                await asyncio.sleep(float(wait))
            elif wait < -2.0:
                self._rtmp_next_tick = float(now)
        video = bytes(payload)
        audio_b = bytes(audio or b"")
        recover_after_lock: BaseException | None = None
        if not bool(python_pacing):
            while True:
                failed = self._rtmp_writer_failed()
                if failed is not None:
                    await self._recover_rtmp_after_failure(failed)
                    break
                with self.rtmp_writer_cv:
                    if int(len(self.rtmp_queue)) < int(self.rtmp_queue_max):
                        break
                await asyncio.sleep(0.005)
        with self.rtmp_writer_cv:
            failed = self._rtmp_writer_failed()
            if failed is not None:
                recover_after_lock = failed
            else:
                queue_len = int(len(self.rtmp_queue))
                if queue_len >= int(self.rtmp_queue_max):
                    if not bool(python_pacing):
                        raise RuntimeError(
                            f"RTMP RIFE queue overflow queue={queue_len} max={self.rtmp_queue_max}"
                        )
                    now_log = time.monotonic()
                    if float(self.rtmp_backpressure_since) <= 0.0:
                        self.rtmp_backpressure_since = float(now_log)
                    age = float(now_log - self.rtmp_backpressure_since)
                    writer_stall = float(now_log - max(float(self.rtmp_writer_started_at), float(self.rtmp_writer_last_tick_at)))
                    if float(self.rtmp_backpressure_fail_sec) > 0.0 and (
                        age >= float(self.rtmp_backpressure_fail_sec)
                        or writer_stall >= float(self.rtmp_backpressure_fail_sec)
                    ):
                        raise RuntimeError(
                            "RTMP backpressure persisted "
                            f"{age:.1f}s queue={queue_len} max={self.rtmp_queue_max} writer_stall={writer_stall:.1f}s"
                        )
                    keep = min(max(1, int(self.rtmp_queue_drop_keep)), max(1, int(self.rtmp_queue_max) - 1))
                    dropped = 0
                    while len(self.rtmp_queue) >= int(keep):
                        if self.rtmp_queue and (
                            int(self.rtmp_queue[0].audio_off) > 0 or int(self.rtmp_queue[0].video_off) > 0
                        ):
                            if len(self.rtmp_queue) <= 1:
                                break
                            del self.rtmp_queue[1]
                        else:
                            self.rtmp_queue.popleft()
                        dropped += 1
                    self.rtmp_drop_count += int(dropped)
                    self.rtmp_stale_drop_count += int(dropped)
                    if now_log - float(self.rtmp_drop_last_log) >= 5.0:
                        logging.warning(
                            "Remote edge RTMP backpressure: session=%s job=%s dropped_stale_ticks=%d queue=%d keep=%d max=%d age=%.1fs writer_stall=%.1fs writer_ticks=%d",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            int(self.rtmp_drop_count),
                            int(len(self.rtmp_queue)),
                            int(keep),
                            int(self.rtmp_queue_max),
                            float(age),
                            float(writer_stall),
                            int(self.rtmp_writer_ticks),
                        )
                        self.rtmp_drop_count = 0
                        self.rtmp_drop_last_log = float(now_log)
                elif queue_len <= 0:
                    self.rtmp_backpressure_since = 0.0
                self.rtmp_queue.append(_RtmpEdgeTick(video, audio_b))
                self.rtmp_writer_cv.notify_all()
        if recover_after_lock is not None:
            await self._recover_rtmp_after_failure(recover_after_lock)
            with self.rtmp_writer_cv:
                self.rtmp_queue.append(_RtmpEdgeTick(video, audio_b))
                self.rtmp_writer_cv.notify_all()
        self._last_rtmp_frame = video
        self._remember_quiet_rtmp_frame(video, audio_b, gap_fill=bool(gap_fill))
        if not bool(gap_fill):
            self._avatar_transition_gap_base_frame = None
            self._avatar_transition_gap_frames = 0
        if bool(python_pacing):
            self._rtmp_next_tick += 1.0 / float(max(1, int(self._rtmp_pipe_fps())))
        self.frames += 1
        self.audio_frames += 1
        self.audio_seconds_published += float(len(audio or b"") // 2) / float(max(1, int(self.cfg.sample_rate)))

    async def push_rgb24_many(self, frames: list[bytes], *, timestamp_us: int | None = None) -> None:
        base_ts = None if timestamp_us is None else int(timestamp_us)
        frame_step_us = int(round(1_000_000.0 / float(max(1, int(self.cfg.fps)))))
        for idx, frame in enumerate(frames):
            ts = None if base_ts is None else int(base_ts + idx * frame_step_us)
            await self.push_rgb24(frame, timestamp_us=ts)

    def _file_target_source_frames(self, *, sample_rate: int, fps: int) -> int:
        file_target_samples = int(max(0, int(self.cfg.file_target_audio_samples or 0)))
        if file_target_samples <= 0 and float(self.cfg.file_target_duration_sec or 0.0) > 0.0:
            file_target_samples = int(round(float(self.cfg.file_target_duration_sec) * float(sample_rate)))
        if file_target_samples > 0:
            return int(max(1, (int(file_target_samples) * int(fps) + int(sample_rate) - 1) // int(sample_rate)))
        return 0

    def _file_target_output_frames(self) -> int:
        source_frames = self._file_target_source_frames(
            sample_rate=max(1, int(self.cfg.sample_rate or 16000)),
            fps=max(1, int(self.cfg.fps)),
        )
        if int(source_frames) <= 0:
            return 0
        return int(
            max(
                1,
                round(
                    float(source_frames)
                    * float(max(1, int(self.file_raw_video_fps or self.cfg.fps)))
                    / float(max(1, int(self.cfg.fps)))
                ),
            )
        )

    def _start_file_stream_encoder(self) -> None:
        if not self.file_video_only_path:
            raise RuntimeError("file stream encoder requires file output path")
        if self.file_video_proc is not None:
            return
        width = max(1, int(self.file_frame_width or self.cfg.width))
        height = max(1, int(self.file_frame_height or self.cfg.height))
        fps = max(1, int(self.file_raw_video_fps or self.cfg.fps))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(fps)),
            "-i",
            "pipe:0",
        ]
        video_filters = [] if bool(self.file_native_pre_resize) else self._file_video_filters(width=int(width), height=int(height))
        watermark_filter_enabled = any(str(item).startswith("drawtext=") for item in video_filters)
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
        cmd.extend(["-r", str(int(fps))])
        cmd.extend(self._file_video_encode_args(output_fps=int(fps)))
        cmd.extend(["-an", str(self.file_video_only_path)])
        log_path = str(self.file_video_only_path) + ".ffmpeg.log"
        self.file_video_log_f = open(log_path, "w", encoding="utf-8")
        self.file_video_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.file_video_log_f,
        )
        logging.warning(
            "Remote edge FILE stream encoder started: session=%s job=%s size=%dx%d fps=%d watermark_filter=%d filter_count=%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(width),
            int(height),
            int(fps),
            1 if bool(watermark_filter_enabled) else 0,
            int(len(video_filters)),
            str(self.file_video_only_path),
        )

    def _write_file_stream_frames(self, frames: list[bytes]) -> int:
        proc = self.file_video_proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("file stream encoder is not running")
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"file stream encoder exited rc={proc.poll()}")
        expected = int(max(1, int(self.file_frame_width or self.cfg.width)) * max(1, int(self.file_frame_height or self.cfg.height)) * 3)
        if expected <= 0:
            raise RuntimeError("invalid file stream output dimensions")
        target_frames = int(self._file_target_output_frames())
        source_frames = [bytes(frame) for frame in frames]
        if target_frames > 0:
            remaining = int(target_frames) - int(self.file_stream_video_frames_written)
            if remaining <= 0:
                return 0
            source_frames = source_frames[: int(remaining)]
        if not source_frames:
            return 0
        for frame in source_frames:
            if len(frame) != expected:
                raise ValueError(f"invalid rgb24 stream frame size: got={len(frame)} expected={expected}")
            proc.stdin.write(frame)
            self.file_stream_last_frame = bytes(frame)
        self.file_stream_video_frames_written += int(len(source_frames))
        return int(len(source_frames))

    def _close_file_stream_encoder(self) -> None:
        proc = self.file_video_proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            timeout = max(30.0, _safe_float_env("REMOTE_EDGE_FILE_STREAM_ENCODER_WAIT_SEC", 300.0))
            rc = proc.wait(timeout=float(timeout))
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=15)
        finally:
            self.file_video_proc = None
            if self.file_video_log_f is not None:
                try:
                    self.file_video_log_f.close()
                except Exception:
                    pass
                self.file_video_log_f = None
        if int(rc) != 0:
            raise RuntimeError(f"file stream encoder exited rc={rc}")
        if not os.path.exists(self.file_video_only_path) or os.path.getsize(self.file_video_only_path) <= 0:
            raise RuntimeError("file stream encoder produced no output")

    def _apply_file_motion_damping(self, frames: list[bytes], *, expected: int, segment_id: str | None = None) -> list[bytes]:
        strength = float(getattr(self, "file_motion_damping_strength", 0.0) or 0.0)
        if float(strength) <= 0.0:
            return frames
        if not frames:
            return []

        segment_id_s = str(segment_id or "").strip()
        if segment_id_s and segment_id_s != str(getattr(self, "file_motion_damping_last_segment_id", "") or ""):
            self.file_motion_damping_last_frame = None
            self.file_motion_damping_last_segment_id = segment_id_s

        try:
            import numpy as np
        except Exception as e:
            if not bool(getattr(self, "file_motion_damping_logged", False)):
                self.file_motion_damping_logged = True
                logging.warning(
                    "Remote edge FILE motion damping disabled because numpy is unavailable: session=%s job=%s err=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    e,
                )
            return frames

        previous = self.file_motion_damping_last_frame
        damping_i = max(0, min(950, int(round(float(strength) * 1000.0))))
        current_i = int(1000 - int(damping_i))
        output: list[bytes] = []
        applied = 0
        for frame in frames:
            frame_b = bytes(frame)
            if len(frame_b) != int(expected):
                output.append(frame_b)
                previous = None
                continue
            if previous is None or len(previous) != int(expected):
                output.append(frame_b)
                previous = frame_b
                continue
            cur = np.frombuffer(frame_b, dtype=np.uint8)
            prev = np.frombuffer(previous, dtype=np.uint8)
            blended = (
                (
                    cur.astype(np.uint16) * int(current_i)
                    + prev.astype(np.uint16) * int(damping_i)
                    + 500
                )
                // 1000
            ).astype(np.uint8)
            frame_out = blended.tobytes()
            output.append(frame_out)
            previous = frame_out
            applied += 1

        self.file_motion_damping_last_frame = previous
        self.file_motion_damping_frames += int(applied)
        now = time.monotonic()
        last_log = float(getattr(self, "file_motion_damping_last_log", 0.0) or 0.0)
        if int(applied) > 0 and (
            not bool(getattr(self, "file_motion_damping_logged", False))
            or now - last_log >= 15.0
        ):
            self.file_motion_damping_logged = True
            self.file_motion_damping_last_log = float(now)
            logging.warning(
                "Remote edge FILE motion damping: session=%s job=%s strength=%.2f frames=%d total=%d segment=%s",
                self.cfg.session_id,
                self.cfg.job_id,
                float(strength),
                int(applied),
                int(self.file_motion_damping_frames),
                segment_id_s or "-",
            )
        return output

    def _write_file_frames(
        self,
        frames: list[bytes],
        *,
        segment_id: str | None = None,
        skip_inline_interpolation: bool = False,
        source_frame_count: int | None = None,
    ) -> None:
        if not self.file_raw_video_path:
            raise RuntimeError("file output is not started")
        expected = int(max(1, int(self.file_frame_width or self.cfg.width)) * max(1, int(self.file_frame_height or self.cfg.height)) * 3)
        if expected <= 0:
            raise RuntimeError("invalid file output dimensions")
        source_frames = [bytes(frame) for frame in frames]
        for frame in source_frames:
            if len(frame) != expected:
                raise ValueError(f"invalid rgb24 frame size: got={len(frame)} expected={expected}")
        if bool(self.file_inline_interpolation_enabled) and not bool(skip_inline_interpolation):
            output_frames = self._interpolate_file_frames_inline_torch_rife(source_frames)
        else:
            output_frames = source_frames
        output_frames = self._apply_file_motion_damping(output_frames, expected=int(expected), segment_id=segment_id)
        written_output_frames = int(len(output_frames))
        if output_frames and bool(self.file_stream_encode_enabled):
            if self.file_video_proc is None:
                self._start_file_stream_encoder()
            written_output_frames = int(self._write_file_stream_frames(output_frames))
        elif output_frames:
            with open(self.file_raw_video_path, "ab") as f:
                for frame in output_frames:
                    if len(frame) != expected:
                        raise ValueError(f"invalid rgb24 output frame size: got={len(frame)} expected={expected}")
                    f.write(frame)
        self.frames += int(len(frames) if source_frame_count is None else max(0, int(source_frame_count)))
        self.file_raw_video_frames += int(written_output_frames)

    def _interpolate_file_tensor_inline_torch_rife(self, frames_tensor: Any) -> Any:
        source_frame_count = int(self._tensor_frame_count(frames_tensor))
        if source_frame_count <= 0:
            return frames_tensor
        if bool(self.file_inline_rife_finalized):
            raise RuntimeError("file inline RIFE already finalized")
        import torch

        source_tensor = self._wait_ready_tensor(frames_tensor)
        if not torch.is_tensor(source_tensor):
            raise TypeError("file tensor RIFE expects a torch Tensor")
        had_previous = self.file_inline_rife_last_source_tensor is not None
        if bool(had_previous):
            previous = self._wait_ready_tensor(self.file_inline_rife_last_source_tensor)
            combined = torch.cat([previous.to(device=source_tensor.device, dtype=source_tensor.dtype), source_tensor], dim=0)
        else:
            combined = source_tensor
        if int(self._tensor_frame_count(combined)) < 2:
            self.file_inline_rife_last_source_tensor = source_tensor[-1:].detach().clone().contiguous()
            return source_tensor
        target_frames = max(
            1,
            int(round(float(self._tensor_frame_count(combined)) * float(self.file_raw_video_fps) / float(max(1, int(self.cfg.fps))))),
        )
        out = self._interpolate_tensor_frames_torch_rife_tensor(
            combined,
            target_frames=int(target_frames),
            label="FILE",
        )
        if bool(had_previous):
            out = out[1:]
        if int(self._tensor_frame_count(out)) > 0:
            out = out[:-1]
        self.file_inline_rife_last_source_tensor = source_tensor[-1:].detach().clone().contiguous()
        return out.to(dtype=torch.float32).contiguous()

    def _interpolate_file_frames_inline_torch_rife(self, frames: list[bytes]) -> list[bytes]:
        source_frames = [bytes(frame) for frame in frames]
        if not source_frames:
            return []
        if bool(self.file_inline_rife_finalized):
            raise RuntimeError("file inline RIFE already finalized")
        combined: list[bytes]
        had_previous = self.file_inline_rife_last_source_frame is not None
        if bool(had_previous):
            combined = [bytes(self.file_inline_rife_last_source_frame or b"")] + source_frames
        else:
            combined = source_frames
        if len(combined) < 2:
            self.file_inline_rife_last_source_frame = source_frames[-1]
            return source_frames
        target_frames = max(1, int(round(float(len(combined)) * float(self.file_raw_video_fps) / float(max(1, int(self.cfg.fps))))))
        out = self._interpolate_frames_torch_rife(
            combined,
            width=max(1, int(self.file_frame_width or self.cfg.width)),
            height=max(1, int(self.file_frame_height or self.cfg.height)),
            target_frames=int(target_frames),
            label="FILE",
        )
        if bool(had_previous):
            out = out[1:]
        if out:
            out = out[:-1]
        self.file_inline_rife_last_source_frame = source_frames[-1]
        return [bytes(frame) for frame in out]

    def _write_file_inline_rife_final_frame(self, frame: bytes) -> int:
        expected = int(max(1, int(self.file_frame_width or self.cfg.width)) * max(1, int(self.file_frame_height or self.cfg.height)) * 3)
        if len(frame) != expected:
            raise RuntimeError("invalid final inline RIFE frame size")
        if bool(self.file_stream_encode_enabled):
            return int(self._write_file_stream_frames([bytes(frame)]))
        with open(self.file_raw_video_path, "ab") as f:
            f.write(bytes(frame))
        return 1

    def _flush_file_inline_rife_carry(self, *, reason: str) -> int:
        if not bool(self.file_inline_interpolation_enabled):
            return 0
        if bool(self.file_inline_rife_finalized):
            return 0
        written = 0
        tensor_last = self.file_inline_rife_last_source_tensor
        if tensor_last is not None:
            tensor = tensor_last
            if bool(self.file_post_vae_after_inline_rife_enabled):
                face = float(max(float(self.file_deferred_face_restore), float(self.file_requested_face_restore)))
                background = float(
                    max(float(self.file_deferred_background_restore), float(self.file_requested_background_restore))
                )
                if self._restore_requested(face, background):
                    tensor = self.postprocess_latents_tensor(
                        tensor,
                        face_restore=float(face),
                        background_restore=float(background),
                        resize_output=False,
                        apply_post_vae=True,
                        input_range="01",
                    )
            frames = self._tensor_01_to_file_rgb24_frames(tensor)
            if frames:
                written = int(self._write_file_inline_rife_final_frame(bytes(frames[-1])))
        else:
            last = self.file_inline_rife_last_source_frame
            if last is not None:
                written = int(self._write_file_inline_rife_final_frame(bytes(last)))
        if int(written) > 0:
            self.file_raw_video_frames += int(written)
            now = time.monotonic()
            if now - float(self.file_inline_rife_boundary_last_log) >= 5.0:
                logging.warning(
                    "Remote edge FILE inline RIFE carry flushed: session=%s job=%s reason=%s written=%d key=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    str(reason or "-"),
                    int(written),
                    str(self.file_inline_rife_last_continuity_key or "-"),
                )
                self.file_inline_rife_boundary_last_log = float(now)
        self.file_inline_rife_last_source_frame = None
        self.file_inline_rife_last_source_tensor = None
        return int(written)

    def _prepare_file_inline_rife_continuity(self, block: _PendingVideoBlock) -> None:
        if not bool(self.file_inline_interpolation_enabled):
            return
        if bool(self.file_inline_rife_finalized):
            return
        current_key = str(self._block_file_rife_continuity_key(block) or "").strip()
        previous_key = str(self.file_inline_rife_last_continuity_key or "").strip()
        has_carry = self.file_inline_rife_last_source_frame is not None or self.file_inline_rife_last_source_tensor is not None
        if current_key and previous_key and current_key != previous_key and bool(has_carry):
            self.file_inline_rife_boundary_resets += 1
            written = self._flush_file_inline_rife_carry(reason="continuity_boundary")
            logging.warning(
                "Remote edge FILE inline RIFE continuity reset: session=%s job=%s resets=%d previous=%s current=%s flushed=%d segment=%s kind=%s",
                self.cfg.session_id,
                self.cfg.job_id,
                int(self.file_inline_rife_boundary_resets),
                str(previous_key),
                str(current_key),
                int(written),
                str(getattr(block, "segment_id", "") or "-"),
                str(getattr(block, "segment_kind", "") or "-"),
            )
        if current_key:
            self.file_inline_rife_last_continuity_key = str(current_key)

    def _finalize_file_inline_interpolation(self) -> None:
        if not bool(self.file_inline_interpolation_enabled):
            return
        if bool(self.file_inline_rife_finalized):
            return
        tensor_last = self.file_inline_rife_last_source_tensor
        if tensor_last is not None:
            self._flush_file_inline_rife_carry(reason="finalize_tensor")
            self.file_inline_rife_finalized = True
            return
        last = self.file_inline_rife_last_source_frame
        if last is not None:
            self._flush_file_inline_rife_carry(reason="finalize_frame")
        self.file_inline_rife_finalized = True

    def _write_file_audio(self, payload: bytes) -> None:
        if not payload:
            return
        if not self.file_raw_audio_path:
            raise RuntimeError("file output is not started")
        even = (len(payload) // 2) * 2
        if even <= 0:
            return
        with open(self.file_raw_audio_path, "ab") as f:
            f.write(payload[:even])
        self.file_audio_bytes += int(even)
        self.audio_frames += 1
        self.audio_seconds_published += float(even // 2) / float(max(1, int(self.cfg.sample_rate)))

    def _target_audio_samples_for_video(self, frame_count: int) -> int:
        target_samples = self._peek_target_audio_samples_for_video(frame_count)
        self._audio_video_frame_cursor += int(max(0, int(frame_count)))
        return int(target_samples)

    def _peek_target_audio_samples_for_video(self, frame_count: int) -> int:
        frames = int(max(0, int(frame_count)))
        if frames <= 0:
            return 0
        sample_rate = int(max(1, int(self.cfg.sample_rate)))
        fps = int(max(1, int(self.cfg.fps)))
        start = int((int(self._audio_video_frame_cursor) * sample_rate) // fps)
        end = int(((int(self._audio_video_frame_cursor) + frames) * sample_rate) // fps)
        return int(max(0, int(end - start)))

    def _silence_audio_bytes_for_output_frames(self, frame_count: int, *, fps: int | None = None) -> bytes:
        frames = int(max(0, int(frame_count)))
        if frames <= 0:
            return b""
        sample_rate = int(max(1, int(self.cfg.sample_rate)))
        fps_i = int(max(1, int(fps if fps is not None else self.cfg.fps)))
        start = int((int(self.frames) * sample_rate) // fps_i)
        end = int(((int(self.frames) + frames) * sample_rate) // fps_i)
        samples = int(max(0, int(end - start)))
        return b"\x00" * int(samples * 2)

    def _pending_audio_samples(self) -> int:
        total = 0
        for payload, _sample_rate in self.pending_audio:
            total += int(len(payload) // 2)
        return int(total)

    def _pending_segment_audio_samples(self) -> int:
        total = 0
        for item in self.segment_audio.values():
            total += int(len(item.payload) // 2)
        return int(total)

    def _block_has_audio(self, block: _PendingVideoBlock) -> bool:
        segment_id = str(block.segment_id or "").strip()
        if segment_id:
            return segment_id in self.segment_audio
        return self._has_audio_for_video_frames(int(self._block_frame_count(block)))

    @staticmethod
    def _unwrap_ready_tensor(frames_tensor: Any | None) -> Any | None:
        return getattr(frames_tensor, "tensor", frames_tensor)

    @staticmethod
    def _wait_ready_tensor(frames_tensor: Any) -> Any:
        import torch

        tensor = getattr(frames_tensor, "tensor", frames_tensor)
        event = getattr(frames_tensor, "ready_event", None)
        if event is None:
            return tensor
        if not torch.is_tensor(tensor):
            return tensor
        if bool(getattr(tensor, "is_cuda", False)):
            device = getattr(tensor, "device", None)
            with torch.cuda.device(device):
                torch.cuda.current_stream(device=device).wait_event(event)
        else:
            event.synchronize()
        return tensor

    @staticmethod
    def _record_ready_tensor(frames_tensor: Any) -> Any:
        import torch

        if not torch.is_tensor(frames_tensor) or not bool(getattr(frames_tensor, "is_cuda", False)):
            return frames_tensor
        try:
            device = getattr(frames_tensor, "device", None)
            with torch.cuda.device(device):
                event = torch.cuda.Event(blocking=False)
                event.record(torch.cuda.current_stream(device=device))
            return CudaReadyTensor(frames_tensor, event, str(device or ""))
        except Exception:
            logging.exception("Remote edge failed to record CUDA ready event for stabilized tensor")
            return frames_tensor

    def _async_cpu_stage_stream(self, device: Any) -> Any:
        import torch

        key = str(device or "")
        stream = self._async_cpu_stage_streams.get(key)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            self._async_cpu_stage_streams[key] = stream
        return stream

    @staticmethod
    def _tensor_frame_count(frames_tensor: Any | None) -> int:
        frames_tensor = RemoteEdgeSession._unwrap_ready_tensor(frames_tensor)
        if frames_tensor is None:
            return 0
        try:
            return max(0, int(frames_tensor.shape[0]))
        except Exception:
            return 0

    @staticmethod
    def _visible_frame_count(frame_count: int, segment_frames: int | None) -> int:
        frames = int(max(0, int(frame_count)))
        if frames <= 0:
            return 0
        if segment_frames is None:
            return int(frames)
        try:
            visible = int(segment_frames)
        except Exception:
            return int(frames)
        return int(max(0, min(int(frames), int(visible))))

    @staticmethod
    def _visible_frame_window(
        frame_count: int,
        segment_start_frame: int | None,
        segment_frames: int | None,
    ) -> tuple[int, int]:
        frames = int(max(0, int(frame_count)))
        if frames <= 0:
            return 0, 0
        try:
            start = int(segment_start_frame if segment_start_frame is not None else 0)
        except Exception:
            start = 0
        start = int(max(0, min(int(frames), int(start))))
        visible = RemoteEdgeSession._visible_frame_count(
            int(max(0, int(frames) - int(start))),
            segment_frames,
        )
        return int(start), int(max(0, min(int(visible), int(frames) - int(start))))

    @staticmethod
    def _slice_tensor_frames(frames_tensor: Any, frame_count: int, *, start_frame: int = 0) -> Any:
        import torch

        visible = int(max(0, int(frame_count)))
        start = int(max(0, int(start_frame)))
        tensor = RemoteEdgeSession._wait_ready_tensor(frames_tensor)
        if not torch.is_tensor(tensor):
            return frames_tensor
        current = int(RemoteEdgeSession._tensor_frame_count(tensor))
        if start <= 0 and visible >= int(current):
            return tensor
        if visible <= 0:
            return tensor[:0].contiguous()
        end = int(max(int(start), min(int(current), int(start) + int(visible))))
        return tensor[int(start):end].contiguous()

    @staticmethod
    def _slice_frame_bytes(frames: list[bytes], frame_count: int, *, start_frame: int = 0) -> list[bytes]:
        visible = int(max(0, int(frame_count)))
        start = int(max(0, int(start_frame)))
        if start <= 0 and visible >= len(frames):
            return [bytes(frame) for frame in frames]
        if visible <= 0:
            return []
        return [bytes(frame) for frame in frames[int(start): int(start) + int(visible)]]

    def _block_frame_count(self, block: _PendingVideoBlock) -> int:
        if block.frames:
            return int(len(block.frames))
        return int(self._tensor_frame_count(block.frames_tensor))

    @staticmethod
    def _tensor_01_to_rgb24_frames(frames_tensor: Any) -> list[bytes]:
        from avalife.remote.torch_rife import tensor_01_to_rgb24_frames

        return tensor_01_to_rgb24_frames(frames_tensor)

    @staticmethod
    def _tensor_hw(frames_tensor: Any | None) -> tuple[int, int]:
        frames_tensor = RemoteEdgeSession._unwrap_ready_tensor(frames_tensor)
        if frames_tensor is None:
            return (0, 0)
        try:
            if int(frames_tensor.ndim) != 4:
                return (0, 0)
            return (max(0, int(frames_tensor.shape[2])), max(0, int(frames_tensor.shape[3])))
        except Exception:
            return (0, 0)

    @staticmethod
    def _resize_tensor_01_cover_crop(frames_tensor: Any, *, target_h: int, target_w: int) -> Any:
        import torch
        import torch.nn.functional as F

        frames_tensor = RemoteEdgeSession._wait_ready_tensor(frames_tensor)
        if not torch.is_tensor(frames_tensor):
            raise TypeError("_resize_tensor_01_cover_crop expects a torch Tensor")
        if int(frames_tensor.ndim) != 4 or int(frames_tensor.shape[1]) != 3:
            raise ValueError(f"_resize_tensor_01_cover_crop expects T,3,H,W tensor, got shape={tuple(frames_tensor.shape)}")
        target_h = int(target_h)
        target_w = int(target_w)
        if target_h <= 0 or target_w <= 0 or int(frames_tensor.shape[0]) <= 0:
            return frames_tensor.contiguous()
        source_h = int(frames_tensor.shape[2])
        source_w = int(frames_tensor.shape[3])
        if source_h == target_h and source_w == target_w:
            return frames_tensor.contiguous()
        if source_h <= 0 or source_w <= 0:
            return frames_tensor.contiguous()
        scale = max(float(target_h) / float(source_h), float(target_w) / float(source_w))
        resize_h = max(int(target_h), int(round(float(source_h) * float(scale))))
        resize_w = max(int(target_w), int(round(float(source_w) * float(scale))))
        resized = F.interpolate(frames_tensor, size=(int(resize_h), int(resize_w)), mode="bicubic", align_corners=False)
        top = max(0, (int(resize_h) - int(target_h)) // 2)
        left = max(0, (int(resize_w) - int(target_w)) // 2)
        return resized[:, :, int(top): int(top) + int(target_h), int(left): int(left) + int(target_w)].contiguous().clamp(0.0, 1.0)

    def _resize_tensor_01_to_output(self, frames_tensor: Any) -> Any:
        return self._resize_tensor_01_cover_crop(
            frames_tensor,
            target_h=int(self.cfg.height),
            target_w=int(self.cfg.width),
        )

    def _tensor_01_to_output_rgb24_frames(self, frames_tensor: Any) -> list[bytes]:
        return self._tensor_01_to_rgb24_frames(self._resize_tensor_01_to_output(frames_tensor))

    def _tensor_01_to_file_rgb24_frames(self, frames_tensor: Any) -> list[bytes]:
        if bool(self.file_native_pre_resize):
            source_h, source_w = self._tensor_hw(frames_tensor)
            if int(source_h) > 0 and int(source_w) > 0:
                self._set_file_frame_size(width=int(source_w), height=int(source_h))
            return self._tensor_01_to_rgb24_frames(frames_tensor)
        return self._tensor_01_to_output_rgb24_frames(frames_tensor)

    def _stabilize_tensor_for_async_post_vae(self, frames_tensor: Any) -> Any:
        import os
        import torch

        frames_tensor = self._wait_ready_tensor(frames_tensor)
        if not torch.is_tensor(frames_tensor):
            return frames_tensor
        stage_mode = str(os.getenv("REMOTE_EDGE_ASYNC_POST_VAE_STAGE", "source") or "source").strip().lower()
        direct_p2p = _env_flag("REMOTE_EDGE_ASYNC_POST_VAE_DIRECT_P2P", "0")
        sync_stage = _env_flag("REMOTE_EDGE_ASYNC_POST_VAE_STAGE_SYNC", "0")
        target_device = str(
            os.getenv("REMOTE_EDGE_POST_VAE_DEVICE")
            or os.getenv("LIVE_RAW_POST_VAE_ENHANCER_DEVICE")
            or ""
        ).strip()
        if not target_device or target_device.lower() in {"auto", "default", "same"}:
            target_device = str(frames_tensor.device)
        with torch.inference_mode():
            source = frames_tensor.detach()
            if bool(sync_stage) and bool(getattr(source, "is_cuda", False)):
                torch.cuda.synchronize(device=source.device)
            if stage_mode in {"target", "gpu", "cuda"}:
                source_device = torch.device(source.device) if bool(getattr(source, "is_cuda", False)) else None
                target = torch.device(target_device)
                if bool(source_device is not None and source_device != target and not bool(direct_p2p)):
                    host = source.to(device="cpu", dtype=torch.float32, copy=True, non_blocking=False).contiguous()
                    stable = host.to(device=target_device, dtype=torch.float32, copy=True, non_blocking=False).contiguous()
                else:
                    stable = source.to(device=target_device, dtype=torch.float32, copy=True, non_blocking=True).contiguous()
                if bool(sync_stage) and bool(getattr(stable, "is_cuda", False)):
                    torch.cuda.synchronize(device=stable.device)
                return self._record_ready_tensor(stable)
            if stage_mode in {"source", "clone"}:
                stable = source.to(dtype=torch.float32, copy=True, non_blocking=True).contiguous()
                if bool(sync_stage) and bool(getattr(stable, "is_cuda", False)):
                    torch.cuda.synchronize(device=stable.device)
                return self._record_ready_tensor(stable)
            cpu_dtype_name = str(os.getenv("REMOTE_EDGE_ASYNC_POST_VAE_CPU_DTYPE", "source") or "source").strip().lower()
            cpu_dtype = torch.float32
            if cpu_dtype_name in {"source", "native", "input"}:
                if source.dtype in {torch.float16, torch.bfloat16, torch.float32}:
                    cpu_dtype = source.dtype
            elif cpu_dtype_name in {"fp16", "float16", "half"}:
                cpu_dtype = torch.float16
            elif cpu_dtype_name in {"bf16", "bfloat16"}:
                cpu_dtype = torch.bfloat16
            if (
                bool(_env_flag("REMOTE_EDGE_ASYNC_POST_VAE_CPU_STAGE_ASYNC", "0"))
                and bool(getattr(source, "is_cuda", False))
            ):
                try:
                    device = getattr(source, "device", None)
                    with torch.cuda.device(device):
                        current_stream = torch.cuda.current_stream(device=device)
                        copy_stream = self._async_cpu_stage_stream(device)
                        host = torch.empty_like(source, device="cpu", dtype=cpu_dtype, pin_memory=True)
                        copy_stream.wait_stream(current_stream)
                        with torch.cuda.stream(copy_stream):
                            host.copy_(source, non_blocking=True)
                            ready_event = torch.cuda.Event(blocking=False)
                            ready_event.record(copy_stream)
                    return _CpuReadyTensor(
                        host,
                        ready_event,
                        str(device or ""),
                        source_ref=source,
                        stream_ref=copy_stream,
                    )
                except Exception as e:
                    logging.warning(
                        "Remote edge async CPU stage unavailable; falling back to blocking CPU stage: %s",
                        e,
                    )
            stable = (
                source
                .to(device="cpu", dtype=cpu_dtype, copy=True, non_blocking=False)
                .contiguous()
            )
            return stable

    def _materialize_block_frames(self, block: _PendingVideoBlock) -> list[bytes]:
        if block.frames:
            return [bytes(frame) for frame in block.frames]
        if block.frames_tensor is None:
            return []
        frames = self._tensor_01_to_output_rgb24_frames(block.frames_tensor)
        block.frames = list(frames)
        block.frames_tensor = None
        return frames

    def _block_frames_bytes(self, block: _PendingVideoBlock) -> list[bytes]:
        return self._materialize_block_frames(block)

    def _materialize_file_block_frames(self, block: _PendingVideoBlock) -> list[bytes]:
        if block.frames:
            return [bytes(frame) for frame in block.frames]
        if block.frames_tensor is None:
            return []
        frames = self._tensor_01_to_file_rgb24_frames(block.frames_tensor)
        block.frames = list(frames)
        block.frames_tensor = None
        return frames

    def _block_file_frames_bytes(self, block: _PendingVideoBlock) -> list[bytes]:
        return self._materialize_file_block_frames(block)

    @staticmethod
    def _block_avatar_ref_path(block: _PendingVideoBlock | None) -> str:
        if block is None:
            return ""
        return str(getattr(block, "avatar_ref_path", "") or "").strip()

    @staticmethod
    def _last_nonempty_avatar_ref_path(blocks: list[_PendingVideoBlock]) -> str:
        for block in reversed(list(blocks or [])):
            ref = RemoteEdgeSession._block_avatar_ref_path(block)
            if ref:
                return ref
        return ""

    @staticmethod
    def _first_nonempty_avatar_ref_path(blocks: list[_PendingVideoBlock]) -> str:
        for block in list(blocks or []):
            ref = RemoteEdgeSession._block_avatar_ref_path(block)
            if ref:
                return ref
        return ""

    @staticmethod
    def _block_file_rife_continuity_key(block: _PendingVideoBlock | None) -> str:
        if block is None:
            return ""
        ref = RemoteEdgeSession._block_avatar_ref_path(block)
        if ref:
            return f"avatar:{ref}"
        kind = str(getattr(block, "segment_kind", "") or "").strip().lower()
        if kind:
            return f"kind:{kind}"
        return ""

    def _peek_pending_avatar_ref_path(self) -> str:
        for block in list(self.pending_blocks):
            ref = self._block_avatar_ref_path(block)
            if ref:
                return ref
        return ""

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = max(0.0, min(1.0, float(value)))
        return float(x * x * (3.0 - 2.0 * x))

    @staticmethod
    def _avatar_transition_style_has_crossfade(style: str) -> bool:
        return str(style or "").strip().lower() in {
            "cross",
            "cross_blur",
            "cross_blur_punch",
            "blur_dissolve",
            "dissolve",
        }

    @staticmethod
    def _avatar_transition_style_has_punch(style: str) -> bool:
        return str(style or "").strip().lower() in {
            "punch",
            "punch_in",
            "cross_blur_punch",
            "blur_punch",
        }

    @staticmethod
    def _zoom_rgb24_array(arr: Any, *, zoom: float) -> Any:
        zoom_f = float(max(1.0, float(zoom)))
        if zoom_f <= 1.0005:
            return arr
        import cv2

        height, width = int(arr.shape[0]), int(arr.shape[1])
        crop_w = max(2, min(width, int(round(float(width) / zoom_f))))
        crop_h = max(2, min(height, int(round(float(height) / zoom_f))))
        x0 = max(0, (width - crop_w) // 2)
        y0 = max(0, (height - crop_h) // 2)
        cropped = arr[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def _blur_rgb24_frame(self, frame: bytes, *, weight: float) -> bytes:
        weight_f = float(max(0.0, min(1.0, float(weight))))
        if weight_f <= 0.001:
            return bytes(frame)
        expected = int(self.cfg.width) * int(self.cfg.height) * 3
        if len(frame) != expected:
            return bytes(frame)
        import cv2
        import numpy as np

        arr = np.frombuffer(frame, dtype=np.uint8).reshape((int(self.cfg.height), int(self.cfg.width), 3))
        sigma = max(0.1, float(self.avatar_transition_sigma) * weight_f / max(0.001, float(self.avatar_transition_strength)))
        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
        out = cv2.addWeighted(arr, 1.0 - float(weight_f), blurred, float(weight_f), 0.0)
        return out.astype(np.uint8, copy=False).tobytes()

    def _compose_avatar_transition_frame(
        self,
        frame: bytes,
        *,
        blur_weight: float,
        old_frame: bytes | None = None,
        old_weight: float = 0.0,
        punch_weight: float = 0.0,
        flash_weight: float = 0.0,
    ) -> bytes:
        blur_weight_f = float(max(0.0, min(1.0, float(blur_weight))))
        old_weight_f = float(max(0.0, min(0.85, float(old_weight))))
        punch_weight_f = self._smoothstep(float(punch_weight))
        flash_weight_f = float(max(0.0, min(0.6, float(flash_weight))))
        if (
            blur_weight_f <= 0.001
            and old_weight_f <= 0.001
            and punch_weight_f <= 0.001
            and flash_weight_f <= 0.001
        ):
            return bytes(frame)
        expected = int(self.cfg.width) * int(self.cfg.height) * 3
        if len(frame) != expected:
            return bytes(frame)
        import cv2
        import numpy as np

        arr = np.frombuffer(frame, dtype=np.uint8).reshape((int(self.cfg.height), int(self.cfg.width), 3))
        out = arr
        if punch_weight_f > 0.001 and float(self.avatar_transition_punch_zoom) > 1.0005:
            zoom = 1.0 + (float(self.avatar_transition_punch_zoom) - 1.0) * float(punch_weight_f)
            out = self._zoom_rgb24_array(out, zoom=float(zoom))
        if blur_weight_f > 0.001:
            sigma = max(
                0.1,
                float(self.avatar_transition_sigma)
                * blur_weight_f
                / max(0.001, float(self.avatar_transition_strength)),
            )
            blurred = cv2.GaussianBlur(
                out,
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_REPLICATE,
            )
            out = cv2.addWeighted(out, 1.0 - blur_weight_f, blurred, blur_weight_f, 0.0)
        if old_frame is not None and old_weight_f > 0.001 and len(old_frame) == expected:
            old_arr = np.frombuffer(old_frame, dtype=np.uint8).reshape((int(self.cfg.height), int(self.cfg.width), 3))
            if blur_weight_f > 0.001:
                sigma = max(
                    0.1,
                    float(self.avatar_transition_sigma)
                    * max(blur_weight_f, old_weight_f)
                    / max(0.001, float(self.avatar_transition_strength)),
                )
                old_arr = cv2.GaussianBlur(
                    old_arr,
                    (0, 0),
                    sigmaX=float(sigma),
                    sigmaY=float(sigma),
                    borderType=cv2.BORDER_REPLICATE,
                )
            out = cv2.addWeighted(out, 1.0 - old_weight_f, old_arr, old_weight_f, 0.0)
        if flash_weight_f > 0.001:
            white = np.full_like(out, 255, dtype=np.uint8)
            out = cv2.addWeighted(out, 1.0 - flash_weight_f, white, flash_weight_f, 0.0)
        return out.astype(np.uint8, copy=False).tobytes()

    def _avatar_transition_weight(self, frame_idx: int, *, boundary_idx: int, output_fps: int) -> float:
        max_strength = float(self.avatar_transition_strength)
        if max_strength <= 0.0:
            return 0.0
        pre_frames = int(round(float(self.avatar_transition_pre_sec) * float(max(1, int(output_fps)))))
        post_frames = int(round(float(self.avatar_transition_post_sec) * float(max(1, int(output_fps)))))
        idx = int(frame_idx)
        boundary = int(boundary_idx)
        weight = 0.0
        if pre_frames > 0 and boundary - pre_frames <= idx < boundary:
            weight = max(weight, max_strength * float(idx - (boundary - pre_frames) + 1) / float(pre_frames))
        if post_frames > 0 and boundary <= idx < boundary + post_frames:
            weight = max(weight, max_strength * (1.0 - float(idx - boundary) / float(post_frames)))
        return float(max(0.0, min(max_strength, weight)))

    def _apply_avatar_transition_blur(
        self,
        frames: list[bytes],
        *,
        batch_blocks: list[_PendingVideoBlock],
        output_fps: int,
        next_avatar_ref_path: str = "",
    ) -> list[bytes]:
        def remember_tail(source_frames: list[bytes]) -> None:
            if source_frames:
                self._last_avatar_transition_source_frame = bytes(source_frames[-1])

        if not bool(self.avatar_transition_blur_enabled):
            last_ref = self._last_nonempty_avatar_ref_path(list(batch_blocks))
            if last_ref:
                self._last_published_avatar_ref_path = str(last_ref)
            remember_tail(list(frames or []))
            return frames
        if not frames or len(frames) < 2:
            last_ref = self._last_nonempty_avatar_ref_path(list(batch_blocks))
            if last_ref:
                self._last_published_avatar_ref_path = str(last_ref)
            remember_tail(list(frames or []))
            return frames
        source_total = int(sum(max(0, self._block_frame_count(block)) for block in list(batch_blocks or [])))
        if source_total <= 0:
            remember_tail(list(frames or []))
            return frames
        refs = [self._block_avatar_ref_path(block) for block in list(batch_blocks or [])]
        counts = [max(0, self._block_frame_count(block)) for block in list(batch_blocks or [])]
        boundaries: list[tuple[int, str, str]] = []
        current_ref = str(self._last_published_avatar_ref_path or "").strip()
        first_ref = self._first_nonempty_avatar_ref_path(list(batch_blocks))
        if current_ref and first_ref and first_ref != current_ref:
            boundaries.append((0, current_ref, first_ref))
        if first_ref:
            current_ref = str(first_ref)
        cursor = 0
        for ref, count in zip(refs, counts):
            ref_s = str(ref or "").strip()
            if ref_s and current_ref and ref_s != current_ref:
                out_boundary = int(round(float(cursor) * float(len(frames)) / float(max(1, source_total))))
                boundaries.append((max(0, min(len(frames), out_boundary)), current_ref, ref_s))
            if ref_s:
                current_ref = str(ref_s)
            cursor += int(count)
        if not boundaries:
            last_ref = self._last_nonempty_avatar_ref_path(list(batch_blocks))
            if last_ref:
                self._last_published_avatar_ref_path = str(last_ref)
            remember_tail(list(frames or []))
            return frames
        style = str(self.avatar_transition_style or "blur").strip().lower()
        crossfade_enabled = self._avatar_transition_style_has_crossfade(style)
        punch_enabled = self._avatar_transition_style_has_punch(style)
        max_strength = max(0.0, min(1.0, float(self.avatar_transition_strength)))
        pre_frames = int(round(float(self.avatar_transition_pre_sec) * float(max(1, int(output_fps)))))
        post_frames = int(round(float(self.avatar_transition_post_sec) * float(max(1, int(output_fps)))))
        effects: list[dict[str, Any]] = [
            {
                "blur": 0.0,
                "old_weight": 0.0,
                "old_frame": None,
                "punch": 0.0,
                "flash": 0.0,
            }
            for _ in frames
        ]
        for boundary, _old_ref, _new_ref in boundaries:
            boundary_i = max(0, min(len(frames), int(boundary)))
            old_frame: bytes | None = None
            if boundary_i > 0 and boundary_i - 1 < len(frames):
                old_frame = bytes(frames[boundary_i - 1])
            elif boundary_i == 0 and (
                self._avatar_transition_silence_base_frame is not None
                or self._avatar_transition_gap_base_frame is not None
                or self._last_avatar_transition_source_frame is not None
            ):
                old_frame = bytes(
                    self._avatar_transition_silence_base_frame
                    or self._avatar_transition_gap_base_frame
                    or self._last_avatar_transition_source_frame
                    or b""
                )
            start = max(0, boundary_i - int(pre_frames))
            end = min(len(frames), boundary_i + int(post_frames))
            for idx in range(int(start), int(end)):
                weight = self._avatar_transition_weight(idx, boundary_idx=int(boundary_i), output_fps=int(output_fps))
                effects[idx]["blur"] = max(float(effects[idx]["blur"]), float(weight))
                normalized = float(weight) / max(0.001, float(max_strength))
                effects[idx]["flash"] = max(
                    float(effects[idx]["flash"]),
                    float(self.avatar_transition_flash_strength) * self._smoothstep(float(normalized)),
                )
                if idx >= boundary_i and int(post_frames) > 0:
                    post_progress = max(0.0, min(1.0, float(idx - boundary_i) / float(max(1, int(post_frames)))))
                    fade = self._smoothstep(1.0 - float(post_progress))
                    if bool(crossfade_enabled) and old_frame is not None:
                        old_weight = min(0.75, float(max_strength) * float(fade))
                        if old_weight > float(effects[idx]["old_weight"]):
                            effects[idx]["old_weight"] = float(old_weight)
                            effects[idx]["old_frame"] = old_frame
                    if bool(punch_enabled):
                        effects[idx]["punch"] = max(float(effects[idx]["punch"]), float(fade))
        out_frames = []
        for frame, effect in zip(frames, effects):
            out_frames.append(
                self._compose_avatar_transition_frame(
                    bytes(frame),
                    blur_weight=float(effect.get("blur") or 0.0),
                    old_frame=effect.get("old_frame"),
                    old_weight=float(effect.get("old_weight") or 0.0),
                    punch_weight=float(effect.get("punch") or 0.0),
                    flash_weight=float(effect.get("flash") or 0.0),
                )
            )
        self.avatar_transition_events += int(len(boundaries))
        now = time.monotonic()
        if now - float(self.avatar_transition_last_log) >= 5.0:
            logging.warning(
                "Remote edge avatar transition: session=%s job=%s style=%s events=%d boundaries=%d frames=%d fps=%d pre=%.2fs post=%.2fs strength=%.2f sigma=%.1f punch=%.3f flash=%.2f",
                self.cfg.session_id,
                self.cfg.job_id,
                str(style or "blur"),
                int(self.avatar_transition_events),
                int(len(boundaries)),
                int(len(frames)),
                int(output_fps),
                float(self.avatar_transition_pre_sec),
                float(self.avatar_transition_post_sec),
                float(self.avatar_transition_strength),
                float(self.avatar_transition_sigma),
                float(self.avatar_transition_punch_zoom),
                float(self.avatar_transition_flash_strength),
            )
            self.avatar_transition_last_log = float(now)
        last_ref = self._last_nonempty_avatar_ref_path(list(batch_blocks))
        if last_ref:
            self._last_published_avatar_ref_path = str(last_ref)
        remember_tail(list(frames or []))
        return out_frames

    @staticmethod
    def _concat_block_tensors(blocks: list[_PendingVideoBlock]) -> Any | None:
        tensors = [block.frames_tensor for block in blocks]
        if not tensors or any(tensor is None for tensor in tensors):
            return None
        shapes = [tuple(int(v) for v in tensor.shape[1:]) for tensor in tensors]
        if len(set(shapes)) > 1:
            return None
        if len(tensors) == 1:
            return tensors[0]
        import torch

        return torch.cat(tensors, dim=0).contiguous()

    @staticmethod
    def _pending_tensor_blocks_count(blocks: deque[_PendingVideoBlock]) -> int:
        return sum(1 for block in blocks if block.frames_tensor is not None)

    def _spill_pending_tensor_blocks(self) -> int:
        max_tensor_blocks = int(getattr(self, "live_gpu_tensor_buffer_blocks", 0))
        tensor_blocks = int(self._pending_tensor_blocks_count(self.pending_blocks))
        spilled = 0
        if int(max_tensor_blocks) <= 0:
            target = 0
        else:
            target = int(max_tensor_blocks)
        if tensor_blocks <= target:
            return 0
        for block in self.pending_blocks:
            if int(tensor_blocks) <= int(target):
                break
            if block.frames_tensor is None:
                continue
            self._materialize_block_frames(block)
            tensor_blocks -= 1
            spilled += 1
        if spilled > 0:
            logging.warning(
                "Remote edge spilled live GPU frame tensors: session=%s job=%s spilled_blocks=%d kept_tensor_blocks=%d pending_blocks=%d buffered_frames=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(spilled),
                int(self._pending_tensor_blocks_count(self.pending_blocks)),
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
            )
        return int(spilled)

    def _take_audio_bytes_for_video(self, frame_count: int, *, pad_missing: bool) -> bytes:
        target_samples = int(self._target_audio_samples_for_video(frame_count))
        if target_samples <= 0:
            return b""
        parts: list[bytes] = []
        remaining = int(target_samples)
        while self.pending_audio and remaining > 0:
            payload, sample_rate = self.pending_audio[0]
            chunk_samples = int(len(payload) // 2)
            if chunk_samples <= 0:
                self.pending_audio.popleft()
                continue
            take_samples = int(min(int(chunk_samples), int(remaining)))
            take_bytes = int(take_samples * 2)
            if take_bytes <= 0:
                break
            parts.append(bytes(payload[:take_bytes]))
            remaining -= int(take_samples)
            if take_samples >= chunk_samples:
                self.pending_audio.popleft()
            else:
                self.pending_audio[0] = (bytes(payload[take_bytes:]), int(sample_rate or self.cfg.sample_rate))
        if bool(pad_missing) and remaining > 0:
            self.rtmp_audio_pad_events += 1
            self.rtmp_audio_pad_samples += int(remaining)
            parts.append(b"\x00" * int(remaining * 2))
        return b"".join(parts)

    def _take_segment_audio_bytes_for_video(
        self,
        segment_audio: _SegmentAudio,
        *,
        offset_bytes: int,
        frame_count: int,
        pad_missing: bool,
    ) -> tuple[bytes, int]:
        target_samples = int(self._target_audio_samples_for_video(frame_count))
        if target_samples <= 0:
            return b"", int(offset_bytes)
        target_bytes = int(target_samples * 2)
        start = int(max(0, int(offset_bytes)))
        end = int(start + target_bytes)
        chunk = bytes(segment_audio.payload[start:end])
        if len(chunk) < target_bytes and bool(pad_missing):
            missing = int((target_bytes - len(chunk)) // 2)
            if missing > 0:
                self.rtmp_audio_pad_events += 1
                self.rtmp_audio_pad_samples += int(missing)
                chunk += b"\x00" * int(missing * 2)
        return chunk, int(end)

    def _take_file_block_audio_for_video(
        self,
        block: _PendingVideoBlock,
        *,
        frame_count: int,
        pad_missing: bool,
    ) -> tuple[bytes, _SegmentAudio | None, int]:
        segment_id = str(block.segment_id or "").strip()
        if segment_id:
            segment_audio = self.segment_audio.get(segment_id)
            if segment_audio is not None:
                offset = int(max(0, int(self.file_segment_audio_offsets.get(segment_id, 0))))
                audio, new_offset = self._take_segment_audio_bytes_for_video(
                    segment_audio,
                    offset_bytes=int(offset),
                    frame_count=int(frame_count),
                    pad_missing=bool(pad_missing),
                )
                self.file_segment_audio_offsets[segment_id] = int(new_offset)
                consumed_frames = int(self.file_segment_frame_offsets.get(segment_id, 0)) + int(max(0, int(frame_count)))
                self.file_segment_frame_offsets[segment_id] = int(consumed_frames)
                segment_frames = getattr(segment_audio, "segment_frames", None)
                done_by_frames = bool(segment_frames is not None and int(consumed_frames) >= int(segment_frames))
                done_by_audio = bool(int(new_offset) >= int(len(segment_audio.payload or b"")))
                if bool(done_by_frames or done_by_audio):
                    self.segment_audio.pop(segment_id, None)
                    self.file_segment_audio_offsets.pop(segment_id, None)
                    self.file_segment_frame_offsets.pop(segment_id, None)
                return bytes(audio), segment_audio, int(offset)
        audio = self._take_audio_bytes_for_video(int(frame_count), pad_missing=bool(pad_missing))
        return bytes(audio), None, 0

    def _rtmp_subtitle_frame_map_from_audio_parts(
        self,
        ranges: list[tuple[_SegmentAudio, int, int]],
        audio_parts: list[bytes],
        offsets: list[int],
    ) -> list[tuple[_SegmentAudio, int] | None]:
        normalized_ranges: list[tuple[_SegmentAudio, int, int]] = []
        for segment_audio, range_start, range_end in list(ranges or []):
            start = int(max(0, int(range_start)))
            end = int(max(start, int(range_end)))
            if end <= start:
                continue
            normalized_ranges.append((segment_audio, start, end))
        if not normalized_ranges:
            return [None for _ in list(audio_parts or [])]

        subtitle_map: list[tuple[_SegmentAudio, int] | None] = []
        gaps = 0
        for part_idx, part in enumerate(list(audio_parts or [])):
            part_start = int(offsets[int(part_idx)]) if int(part_idx) < len(offsets) else 0
            part_len = int(len(part or b""))
            part_end = int(part_start + max(0, part_len))
            best_audio: _SegmentAudio | None = None
            best_range_start = 0
            best_overlap = -1
            best_center_distance = float("inf")
            part_center = float(part_start + part_end) / 2.0
            for subtitle_audio, range_start, range_end in normalized_ranges:
                overlap_start = max(int(part_start), int(range_start))
                overlap_end = min(int(part_end), int(range_end))
                overlap = int(max(0, int(overlap_end) - int(overlap_start)))
                if overlap <= 0 and part_len <= 0 and int(range_start) <= int(part_start) < int(range_end):
                    overlap = 1
                if overlap <= 0:
                    continue
                range_center = float(range_start + range_end) / 2.0
                center_distance = abs(float(part_center) - float(range_center))
                if int(overlap) > int(best_overlap) or (
                    int(overlap) == int(best_overlap) and float(center_distance) < float(best_center_distance)
                ):
                    best_audio = subtitle_audio
                    best_range_start = int(range_start)
                    best_overlap = int(overlap)
                    best_center_distance = float(center_distance)
            if best_audio is None:
                subtitle_map.append(None)
                gaps += 1
                continue
            local_offset = int(max(0, int(part_start) - int(best_range_start)))
            subtitle_map.append((best_audio, int(local_offset)))

        if int(gaps) > 0:
            now = time.monotonic()
            self.subtitle_frame_map_gap_events += int(gaps)
            if now - float(getattr(self, "subtitle_frame_map_gap_last_log", 0.0) or 0.0) >= 10.0:
                self.subtitle_frame_map_gap_last_log = float(now)
                first_gap = next((idx for idx, entry in enumerate(subtitle_map) if entry is None), -1)
                try:
                    first_range = normalized_ranges[0]
                    last_range = normalized_ranges[-1]
                    logging.warning(
                        "Remote edge subtitle frame map gaps: session=%s job=%s gaps=%d total_gap_events=%d parts=%d first_gap=%d ranges=%d first_range=%d-%d last_range=%d-%d",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        int(gaps),
                        int(self.subtitle_frame_map_gap_events),
                        int(len(audio_parts or [])),
                        int(first_gap),
                        int(len(normalized_ranges)),
                        int(first_range[1]),
                        int(first_range[2]),
                        int(last_range[1]),
                        int(last_range[2]),
                    )
                except Exception:
                    logging.warning(
                        "Remote edge subtitle frame map gaps: session=%s job=%s gaps=%d total_gap_events=%d parts=%d ranges=%d",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        int(gaps),
                        int(self.subtitle_frame_map_gap_events),
                        int(len(audio_parts or [])),
                        int(len(normalized_ranges)),
                    )
        return subtitle_map

    def _subtitle_frame(
        self,
        frame: bytes,
        *,
        segment_audio: _SegmentAudio | None,
        segment_audio_offset_before: int,
        audio: bytes,
    ) -> bytes:
        renderer = self.subtitle_renderer
        if renderer is None or not bool(getattr(renderer, "enabled", False)):
            return frame
        if segment_audio is None:
            return frame
        text = str(getattr(segment_audio, "subtitle_text", "") or "").strip()
        if not text:
            return frame
        try:
            frame_samples = int(max(0, len(audio) // 2))
            played_samples = int(max(0, int(segment_audio_offset_before) // 2)) + int(
                round(float(frame_samples) * float(getattr(self, "subtitle_audio_frame_anchor", 0.0) or 0.0))
            )
            base_start = int(segment_audio.subtitle_start_samples or 0)
            absolute_samples = int(base_start + int(played_samples))
            total_samples = int(segment_audio.subtitle_total_samples or 0)
            if total_samples <= 0:
                total_samples = int(segment_audio.subtitle_end_samples or 0)
            if total_samples <= 0:
                audible = int(segment_audio.audible_samples or (len(segment_audio.payload) // 2))
                total_samples = int(base_start + max(1, int(audible)))
            alignment = (
                segment_audio.subtitle_normalized_alignment
                if isinstance(segment_audio.subtitle_normalized_alignment, dict) and segment_audio.subtitle_normalized_alignment
                else segment_audio.subtitle_alignment
            )
            alignment_base = int(
                segment_audio.subtitle_alignment_base_samples
                if isinstance(segment_audio.subtitle_alignment_base_samples, int)
                else segment_audio.subtitle_start_samples
                if isinstance(segment_audio.subtitle_start_samples, int)
                else 0
            )
            if int(total_samples) > 0 and int(base_start) >= int(total_samples):
                return frame
            local_offset_samples = int(max(0, int(absolute_samples) - int(alignment_base)))
            if int(total_samples) > int(alignment_base):
                local_offset_samples = int(
                    min(int(local_offset_samples), max(0, int(total_samples) - int(alignment_base)))
                )
            if int(total_samples) > int(alignment_base):
                progress = float(local_offset_samples) / float(max(1, int(total_samples) - int(alignment_base)))
            else:
                progress = float(absolute_samples) / float(max(1, int(total_samples)))
            segment_end_local_samples = int(max(0, int(total_samples) - int(alignment_base)))
            return renderer.render(
                frame,
                text=text,
                progress=float(progress),
                alignment=alignment if isinstance(alignment, dict) else None,
                sample_offset_samples=int(local_offset_samples),
                sample_rate=int(segment_audio.sample_rate or self.cfg.sample_rate),
                segment_end_samples=int(segment_end_local_samples),
            )
        except Exception:
            now = time.monotonic()
            if now - float(getattr(self, "subtitle_frame_error_last_log", 0.0) or 0.0) >= 10.0:
                self.subtitle_frame_error_last_log = float(now)
                logging.exception(
                    "Remote edge subtitle frame failed: session=%s job=%s segment=%s offset_bytes=%s audio_bytes=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    str(getattr(segment_audio, "segment_id", "") or ""),
                    str(segment_audio_offset_before),
                    int(len(bytes(audio or b""))),
                )
            return frame

    def _take_audio_for_video(self, frame_count: int) -> list[tuple[bytes, int]]:
        target_samples = int(self._target_audio_samples_for_video(frame_count))
        if target_samples <= 0:
            return []
        taken: list[tuple[bytes, int]] = []
        remaining = int(target_samples)
        while self.pending_audio and remaining > 0:
            payload, sample_rate = self.pending_audio[0]
            chunk_samples = int(len(payload) // 2)
            if chunk_samples <= 0:
                self.pending_audio.popleft()
                continue
            take_samples = int(min(int(chunk_samples), int(remaining)))
            take_bytes = int(take_samples * 2)
            if take_bytes <= 0:
                break
            taken.append((bytes(payload[:take_bytes]), int(sample_rate or self.cfg.sample_rate)))
            remaining -= int(take_samples)
            if take_samples >= chunk_samples:
                self.pending_audio.popleft()
            else:
                self.pending_audio[0] = (bytes(payload[take_bytes:]), int(sample_rate or self.cfg.sample_rate))
        return taken

    def _ready_to_start_publish(self) -> bool:
        if self.publish_draining:
            return True
        if self.publish_started:
            return True
        if str(self.cfg.output or "").strip().lower() == "file":
            return True
        if int(self.buffered_video_frames) < int(self.start_prebuffer_frames):
            return False
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return True
        if not bool(self.start_prebuffer_require_audio):
            return True
        if self.pending_blocks and str(self.pending_blocks[0].segment_id or "").strip():
            return self._block_has_audio(self.pending_blocks[0])
        target_frames = int(min(int(self.buffered_video_frames), max(0, int(self.start_prebuffer_frames))))
        target_samples = int(self._peek_target_audio_samples_for_video(target_frames))
        return int(self._pending_audio_samples()) >= int(target_samples)

    def _has_audio_for_video_frames(self, frame_count: int) -> bool:
        target_samples = int(self._peek_target_audio_samples_for_video(frame_count))
        if target_samples <= 0:
            return True
        return int(self._pending_audio_samples()) >= int(target_samples)

    def _adaptive_restore_strengths(
        self,
        *,
        face_restore: float | None,
        background_restore: float | None,
    ) -> tuple[float | None, float | None]:
        if bool(self.post_vae_disabled_after_error):
            return 0.0, 0.0
        output_kind = str(self.cfg.output or "").strip().lower()
        if output_kind == "file":
            face_cap = float(
                _safe_float_env(
                    "REMOTE_EDGE_FILE_FACE_RESTORE_MAX",
                    _safe_float_env("REMOTE_EDGE_FACE_RESTORE_MAX", 1.0),
                )
            )
            background_cap = float(
                _safe_float_env(
                    "REMOTE_EDGE_FILE_BACKGROUND_RESTORE_MAX",
                    _safe_float_env("REMOTE_EDGE_BACKGROUND_RESTORE_MAX", 0.0),
                )
            )
        else:
            face_cap = float(
                _safe_float_env(
                    "REMOTE_EDGE_LIVE_FACE_RESTORE_MAX",
                    _safe_float_env("REMOTE_EDGE_FACE_RESTORE_MAX", 1.0),
                )
            )
            background_cap = float(
                _safe_float_env(
                    "REMOTE_EDGE_LIVE_BACKGROUND_RESTORE_MAX",
                    _safe_float_env("REMOTE_EDGE_BACKGROUND_RESTORE_MAX", 0.0),
                )
            )
        face_cap = max(0.0, min(1.0, face_cap))
        background_cap = max(0.0, min(1.0, background_cap))
        if face_restore is not None:
            face_restore = min(max(0.0, float(face_restore)), float(face_cap))
        if background_restore is not None:
            background_restore = min(max(0.0, float(background_restore)), float(background_cap))
        if not bool(self.adaptive_restore_enabled):
            return face_restore, background_restore

        buffered = int(max(0, int(self.buffered_video_frames)))
        pending_block_count = int(max(0, len(self.pending_blocks)))
        prev_bg = bool(self._adaptive_background_shed)
        latent_q = 0
        try:
            latent_q = int(self.latent_decode_queue.qsize())
        except Exception:
            latent_q = 0

        background_requested = background_restore is None or float(background_restore) > 0.0
        edge_decode_active = bool(self.publish_started) and int(self.latent_decode_count) >= 2
        publish_buffer_capped = (
            int(self.max_buffered_video_frames) > 0
            and int(buffered) >= int(self.max_buffered_video_frames)
        )
        if bool(background_requested) and bool(edge_decode_active) and not bool(publish_buffer_capped):
            buffer_low = int(self.adaptive_background_queue_min_blocks)
            buffer_recover = max(int(buffer_low) + 1, int(self.adaptive_background_recover_queue_blocks))
            now = time.monotonic()
            decode_backlog_pressure = int(latent_q) >= int(buffer_low)
            publish_buffer_pressure = (
                int(pending_block_count) <= int(buffer_low)
                or int(self.rtmp_gap_fill_streak_frames) > 0
            )
            pressure_active = bool(decode_backlog_pressure or publish_buffer_pressure)
            if bool(self._adaptive_background_shed):
                recovered = (
                    int(pending_block_count) >= int(buffer_recover)
                    and int(self.rtmp_gap_fill_streak_frames) <= 0
                    and int(latent_q) < int(buffer_low)
                )
                self._adaptive_background_shed = not bool(recovered)
                if not bool(self._adaptive_background_shed):
                    self._adaptive_background_pressure_since = 0.0
            else:
                if bool(pressure_active):
                    if float(self._adaptive_background_pressure_since) <= 0.0:
                        self._adaptive_background_pressure_since = float(now)
                    pressure_age = float(now) - float(self._adaptive_background_pressure_since)
                    self._adaptive_background_shed = (
                        pressure_age >= float(self.adaptive_background_shed_after_sec)
                    )
                else:
                    self._adaptive_background_pressure_since = 0.0
        else:
            self._adaptive_background_shed = False
            self._adaptive_background_pressure_since = 0.0

        out_face = face_restore
        out_background = background_restore
        if bool(self._adaptive_background_shed):
            out_background = 0.0

        if prev_bg != bool(self._adaptive_background_shed):
            now = time.monotonic()
            if now - float(self._adaptive_restore_last_log) >= 1.0:
                logging.warning(
                    "Remote edge adaptive restore: session=%s job=%s buffered_frames=%d buffer_max=%d buffer_capped=%d latent_q=%d queue_min=%d queue_recover=%d shed_after=%.2fs pressure_age=%.2fs shed_background=%d face_locked=1 face=%s->%s background=%s->%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(buffered),
                    int(self.max_buffered_video_frames),
                    1 if bool(publish_buffer_capped) else 0,
                    int(latent_q),
                    int(self.adaptive_background_queue_min_blocks),
                    max(int(self.adaptive_background_queue_min_blocks) + 1, int(self.adaptive_background_recover_queue_blocks)),
                    float(self.adaptive_background_shed_after_sec),
                    (
                        0.0
                        if float(self._adaptive_background_pressure_since) <= 0.0
                        else max(0.0, time.monotonic() - float(self._adaptive_background_pressure_since))
                    ),
                    1 if bool(self._adaptive_background_shed) else 0,
                    "-" if face_restore is None else f"{float(face_restore):.2f}",
                    "-" if out_face is None else f"{float(out_face):.2f}",
                    "-" if background_restore is None else f"{float(background_restore):.2f}",
                    "-" if out_background is None else f"{float(out_background):.2f}",
                )
                self._adaptive_restore_last_log = float(now)

        return out_face, out_background

    async def enqueue_video_frames(
        self,
        frames: list[bytes],
        *,
        timestamp_us: int | None = None,
        segment_id: str | None = None,
        segment_kind: str = "",
        segment_start_frame: int | None = None,
        segment_frames: int | None = None,
        avatar_ref_path: str = "",
    ) -> None:
        if not frames:
            return
        visible_start_frame, visible_frame_count = self._visible_frame_window(
            len(frames),
            segment_start_frame,
            segment_frames,
        )
        if visible_frame_count <= 0:
            return
        if int(visible_start_frame) > 0 or int(visible_frame_count) < int(len(frames)):
            frames = self._slice_frame_bytes(
                list(frames),
                int(visible_frame_count),
                start_frame=int(visible_start_frame),
            )
        if self.publish_failed_exc is not None:
            raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc
        async with self.publish_cv:
            if self.publish_task is not None and self.publish_task.done():
                try:
                    exc = self.publish_task.exception()
                except asyncio.CancelledError:
                    exc = None
                if exc is not None:
                    self.publish_failed_exc = exc
                    raise RuntimeError(f"remote edge publish failed: {exc}") from exc
            if self.publish_task is None or self.publish_task.done():
                self.publish_task = asyncio.create_task(
                    self._publish_loop(),
                    name=f"remote-edge-publish-{self.cfg.session_id}",
                )
            while (
                int(self.max_buffered_video_frames) > 0
                and int(self.buffered_video_frames) >= int(self.max_buffered_video_frames)
                and not self.publish_stop.is_set()
            ):
                now_log = time.monotonic()
                if now_log - float(self.max_buffer_last_log) >= 5.0:
                    logging.warning(
                        "Remote edge publish buffer full: session=%s job=%s buffered_frames=%d max=%d pending_blocks=%d pending_audio_sec=%.2f",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        int(self.buffered_video_frames),
                        int(self.max_buffered_video_frames),
                        int(len(self.pending_blocks)),
                        float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
                    )
                    self.max_buffer_last_log = float(now_log)
                await self.publish_cv.wait()
            if self.publish_stop.is_set():
                return
            self.pending_blocks.append(
                _PendingVideoBlock(
                    frames=list(frames),
                    timestamp_us=timestamp_us,
                    segment_id=None if not str(segment_id or "").strip() else str(segment_id),
                    segment_kind=str(segment_kind or ""),
                    segment_start_frame=None if segment_start_frame is None else int(segment_start_frame),
                    segment_frames=None if segment_frames is None else int(segment_frames),
                    avatar_ref_path=str(avatar_ref_path or ""),
                )
            )
            self.buffered_video_frames += int(len(frames))
            self.publish_cv.notify_all()

    async def enqueue_video_tensor(
        self,
        frames_tensor: Any,
        *,
        timestamp_us: int | None = None,
        segment_id: str | None = None,
        segment_kind: str = "",
        segment_start_frame: int | None = None,
        segment_frames: int | None = None,
        avatar_ref_path: str = "",
    ) -> None:
        frame_count = int(self._tensor_frame_count(frames_tensor))
        if frame_count <= 0:
            return
        visible_start_frame, visible_frame_count = self._visible_frame_window(
            int(frame_count),
            segment_start_frame,
            segment_frames,
        )
        if visible_frame_count <= 0:
            return
        if int(visible_start_frame) > 0 or int(visible_frame_count) < int(frame_count):
            frames_tensor = self._slice_tensor_frames(
                frames_tensor,
                int(visible_frame_count),
                start_frame=int(visible_start_frame),
            )
            frame_count = int(visible_frame_count)
        if self.publish_failed_exc is not None:
            raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc
        frames: list[bytes] = []
        async with self.publish_cv:
            if self.publish_task is not None and self.publish_task.done():
                try:
                    exc = self.publish_task.exception()
                except asyncio.CancelledError:
                    exc = None
                if exc is not None:
                    self.publish_failed_exc = exc
                    raise RuntimeError(f"remote edge publish failed: {exc}") from exc
            if self.publish_task is None or self.publish_task.done():
                self.publish_task = asyncio.create_task(
                    self._publish_loop(),
                    name=f"remote-edge-publish-{self.cfg.session_id}",
                )
            while (
                int(self.max_buffered_video_frames) > 0
                and int(self.buffered_video_frames) >= int(self.max_buffered_video_frames)
                and not self.publish_stop.is_set()
            ):
                now_log = time.monotonic()
                if now_log - float(self.max_buffer_last_log) >= 5.0:
                    logging.warning(
                        "Remote edge publish buffer full: session=%s job=%s buffered_frames=%d max=%d pending_blocks=%d pending_audio_sec=%.2f",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        int(self.buffered_video_frames),
                        int(self.max_buffered_video_frames),
                        int(len(self.pending_blocks)),
                        float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
                    )
                    self.max_buffer_last_log = float(now_log)
                await self.publish_cv.wait()
            if self.publish_stop.is_set():
                return
            keep_tensor = int(self.live_gpu_tensor_buffer_blocks) > 0 and (
                int(self._pending_tensor_blocks_count(self.pending_blocks)) < int(self.live_gpu_tensor_buffer_blocks)
            )
            if not bool(keep_tensor):
                frames = self._tensor_01_to_output_rgb24_frames(frames_tensor)
                frames_tensor = None
            self.pending_blocks.append(
                _PendingVideoBlock(
                    frames=list(frames),
                    frames_tensor=frames_tensor,
                    timestamp_us=timestamp_us,
                    segment_id=None if not str(segment_id or "").strip() else str(segment_id),
                    segment_kind=str(segment_kind or ""),
                    segment_start_frame=None if segment_start_frame is None else int(segment_start_frame),
                    segment_frames=None if segment_frames is None else int(segment_frames),
                    avatar_ref_path=str(avatar_ref_path or ""),
                )
            )
            self.buffered_video_frames += int(frame_count)
            self._spill_pending_tensor_blocks()
            self.publish_cv.notify_all()

    def _record_clip_segment(
        self,
        block: _PendingVideoBlock,
        *,
        frames: list[bytes],
        segment_audio: _SegmentAudio | None,
        start_frame: int,
        end_frame: int,
    ) -> None:
        exporter = self.clip_exporter
        if exporter is None:
            return
        segment_id = str(block.segment_id or "").strip()
        if not segment_id:
            return
        expected_frame_bytes = int(max(1, int(self.cfg.width)) * max(1, int(self.cfg.height)) * 3)
        bad_frame_bytes = 0
        for frame in list(frames or []):
            if len(frame) != int(expected_frame_bytes):
                bad_frame_bytes = int(len(frame))
                break
        if bad_frame_bytes > 0:
            now = time.monotonic()
            last_log = float(getattr(self, "_clip_segment_size_mismatch_last_log", 0.0) or 0.0)
            if now - last_log >= 10.0:
                logging.warning(
                    "Remote edge clip segment record skipped: session=%s job=%s segment=%s got_frame_bytes=%d expected_frame_bytes=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    segment_id,
                    int(bad_frame_bytes),
                    int(expected_frame_bytes),
                )
                setattr(self, "_clip_segment_size_mismatch_last_log", float(now))
            return
        try:
            exporter.record_segment(
                _clip_exporter_module().EdgeClipSegment(
                    segment_id=segment_id,
                    segment_kind=str(block.segment_kind or getattr(segment_audio, "segment_kind", "") or ""),
                    frames=list(frames),
                    audio=b"" if segment_audio is None else bytes(segment_audio.payload),
                    subtitle_text="" if segment_audio is None else str(segment_audio.subtitle_text or ""),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    audible_samples=None if segment_audio is None else segment_audio.audible_samples,
                    subtitle_start_samples=None if segment_audio is None else segment_audio.subtitle_start_samples,
                    subtitle_end_samples=None if segment_audio is None else segment_audio.subtitle_end_samples,
                    subtitle_total_samples=None if segment_audio is None else segment_audio.subtitle_total_samples,
                    turn_done=False if segment_audio is None else bool(segment_audio.turn_done),
                )
            )
        except Exception:
            logging.exception(
                "Remote edge clip segment record failed: session=%s job=%s segment=%s",
                self.cfg.session_id,
                self.cfg.job_id,
                segment_id,
            )

    async def _publish_loop(self) -> None:
        try:
            while not self.publish_stop.is_set():
                block: _PendingVideoBlock | None = None
                gap_fill = False
                gap_fill_reason = "none"
                async with self.publish_cv:
                    while not self.publish_stop.is_set():
                        if bool(self.publish_draining) and not self.pending_blocks:
                            self.publish_stop.set()
                            self.publish_cv.notify_all()
                            return
                        ready_to_start = self._ready_to_start_publish()
                        if self.pending_blocks and ready_to_start:
                            candidate = self.pending_blocks[0]
                            candidate_frames = int(self._block_frame_count(candidate))
                            if (
                                bool(self.publish_started)
                                and not bool(self.publish_draining)
                                and bool(self.rtmp_require_real_audio)
                                and str(self.cfg.output or "livekit").strip().lower() == "rtmp"
                                and int(candidate_frames) > 0
                                and not self._block_has_audio(candidate)
                            ):
                                self._raise_rtmp_media_runway_invariant(
                                    "block_without_matching_audio",
                                    buffered_frames=int(self.buffered_video_frames),
                                    candidate_frames=int(candidate_frames),
                                    pending_audio_sec=f"{float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))):.2f}",
                                    pending_blocks=int(len(self.pending_blocks)),
                                )
                            else:
                                if not self.publish_started:
                                    self.publish_started = True
                                    logging.warning(
                                        "Remote edge publish prebuffer ready: session=%s job=%s buffered_frames=%d pending_blocks=%d pending_audio=%d pending_audio_samples=%d require_audio=%d",
                                        self.cfg.session_id,
                                        self.cfg.job_id,
                                        int(self.buffered_video_frames),
                                        int(len(self.pending_blocks)),
                                        int(len(self.pending_audio)),
                                        int(self._pending_audio_samples()),
                                        int(bool(self.start_prebuffer_require_audio)),
                                    )
                                block = self.pending_blocks.popleft()
                                self.buffered_video_frames = max(0, self.buffered_video_frames - int(self._block_frame_count(block)))
                                self.publish_cv.notify_all()
                                break
                        if (
                            bool(self.publish_started)
                            and not bool(self.publish_draining)
                            and bool(self.rtmp_gap_fill_enabled)
                            and str(self.cfg.output or "livekit").strip().lower() == "rtmp"
                            and self._last_rtmp_frame is not None
                        ):
                            gap_fill_reason = "no_video"
                            gap_frame = self._rtmp_gap_frame(str(gap_fill_reason))
                            if gap_frame is None:
                                await self.publish_cv.wait()
                                continue
                            block = _PendingVideoBlock(frames=[bytes(gap_frame)], timestamp_us=None)
                            gap_fill = True
                            break
                        await self.publish_cv.wait()
                if block is None:
                    continue
                frames = block.frames
                timestamp_us = block.timestamp_us
                if str(self.cfg.output or "").strip().lower() == "file":
                    _ = timestamp_us
                    source_frame_count = int(self._block_frame_count(block))
                    audio, segment_audio, segment_audio_offset = self._take_file_block_audio_for_video(
                        block,
                        frame_count=int(source_frame_count),
                        pad_missing=True,
                    )
                    frames = []
                    file_frames_preprocessed_output_fps = False
                    self._prepare_file_inline_rife_continuity(block)
                    if (
                        int(source_frame_count) > 0
                        and block.frames_tensor is not None
                        and bool(self.file_post_vae_after_inline_rife_enabled)
                    ):
                        post_rife_started = time.perf_counter()
                        rife_sec = 0.0
                        post_vae_sec = 0.0
                        frames_tensor = block.frames_tensor
                        try:
                            if bool(self.file_inline_interpolation_enabled):
                                rife_started = time.perf_counter()
                                frames_tensor = self._interpolate_file_tensor_inline_torch_rife(frames_tensor)
                                rife_sec = max(0.0, time.perf_counter() - float(rife_started))
                            post_face_restore = float(
                                max(float(self.file_deferred_face_restore), float(self.file_requested_face_restore))
                            )
                            post_background_restore = float(
                                max(float(self.file_deferred_background_restore), float(self.file_requested_background_restore))
                            )
                            if self._restore_requested(post_face_restore, post_background_restore):
                                post_started = time.perf_counter()
                                frames_tensor = await asyncio.to_thread(
                                    self.postprocess_latents_tensor,
                                    frames_tensor,
                                    face_restore=float(post_face_restore),
                                    background_restore=float(post_background_restore),
                                    resize_output=False,
                                    apply_post_vae=True,
                                    input_range="01",
                                )
                                post_vae_sec = max(0.0, time.perf_counter() - float(post_started))
                                self.file_inline_post_vae_blocks += 1
                                self.file_inline_post_vae_elapsed_sec += float(post_vae_sec)
                            frames = self._tensor_01_to_file_rgb24_frames(frames_tensor)
                            block.frames = list(frames)
                            block.frames_tensor = None
                            file_frames_preprocessed_output_fps = True
                            logging.warning(
                                "Remote edge FILE RIFE-before-PostVAE block timing: session=%s job=%s segment=%s source_frames=%d output_frames=%d rife=%.3fs postvae_native=%.3fs total=%.3fs face=%.2f background=%.2f",
                                self.cfg.session_id,
                                self.cfg.job_id,
                                str(block.segment_id or ""),
                                int(source_frame_count),
                                int(len(frames)),
                                float(rife_sec),
                                float(post_vae_sec),
                                float(max(0.0, time.perf_counter() - float(post_rife_started))),
                                float(post_face_restore),
                                float(post_background_restore),
                            )
                        except Exception as e:
                            logging.exception(
                                "Remote edge FILE RIFE-before-PostVAE failed: session=%s job=%s segment=%s err=%s",
                                self.cfg.session_id,
                                self.cfg.job_id,
                                str(block.segment_id or ""),
                                e,
                            )
                            raise
                    if not frames:
                        frames = self._block_file_frames_bytes(block)
                    if segment_audio is not None and frames:
                        audio_parts = self._split_pcm16le_for_frames(bytes(audio), int(len(frames)))
                        if len(audio_parts) == len(frames):
                            subtitled_frames: list[bytes] = []
                            audio_offset = int(segment_audio_offset)
                            for frame, audio_part in zip(frames, audio_parts):
                                subtitled_frames.append(
                                    self._subtitle_frame(
                                        bytes(frame),
                                        segment_audio=segment_audio,
                                        segment_audio_offset_before=int(audio_offset),
                                        audio=bytes(audio_part),
                                    )
                                )
                                audio_offset += int(len(audio_part))
                            frames = subtitled_frames
                    self._write_file_frames(
                        frames,
                        segment_id=block.segment_id,
                        skip_inline_interpolation=bool(file_frames_preprocessed_output_fps),
                        source_frame_count=int(source_frame_count)
                        if bool(file_frames_preprocessed_output_fps)
                        else None,
                    )
                    self._write_file_audio(audio)
                    if str(block.segment_id or "").strip():
                        self._record_clip_segment(
                            block,
                            frames=frames,
                            segment_audio=segment_audio,
                            start_frame=int(self.frames) - int(len(frames)),
                            end_frame=int(self.frames),
                        )
                    self.audio_chunks_published += 1
                    self.log_stats()
                    continue
                if str(self.cfg.output or "livekit").strip().lower() == "rtmp":
                    await self._stop_rtmp_idle_pump()
                    await self._ensure_rtmp_started_for_publish()
                    _ = timestamp_us
                    batch_blocks = [block]
                    live_rife_pairwise = bool(self._live_rife_pairwise_enabled())
                    if bool(self.live_rife_enabled) and not bool(gap_fill) and not bool(live_rife_pairwise):
                        target_source_frames = max(int(self._block_frame_count(block)), int(self.live_rife_batch_source_frames))
                        batch_frame_count = int(self._block_frame_count(block))
                        async with self.publish_cv:
                            while (
                                int(batch_frame_count) < int(target_source_frames)
                                and self.pending_blocks
                                and not self.publish_stop.is_set()
                            ):
                                next_block = self.pending_blocks[0]
                                if (
                                    not bool(self.publish_draining)
                                    and bool(self.rtmp_require_real_audio)
                                    and not bool(self._block_has_audio(next_block))
                                ):
                                    break
                                if (
                                    int(batch_frame_count) > 0
                                    and int(batch_frame_count) + int(self._block_frame_count(next_block)) > int(target_source_frames)
                                ):
                                    break
                                next_block = self.pending_blocks.popleft()
                                self.buffered_video_frames = max(0, self.buffered_video_frames - int(self._block_frame_count(next_block)))
                                batch_blocks.append(next_block)
                                batch_frame_count += int(self._block_frame_count(next_block))
                            self.publish_cv.notify_all()
                        if len(batch_blocks) > 1:
                            frames = [frame for item in batch_blocks for frame in item.frames]
                    source_frame_count_initial = int(sum(self._block_frame_count(item) for item in batch_blocks))
                    if (
                        not frames
                        and (
                            bool(gap_fill)
                            or not bool(self.live_rife_enabled)
                            or bool(live_rife_pairwise)
                            or int(source_frame_count_initial) < 2
                        )
                    ):
                        frames = [frame for item in batch_blocks for frame in self._block_frames_bytes(item)]
                    segment_audio = None
                    if (not bool(gap_fill)) and str(block.segment_id or "").strip():
                        segment_audio = self.segment_audio.pop(str(block.segment_id), None)
                    segment_start_frame = int(self.frames)
                    frame_idx = 0
                    segment_audio_offset = 0
                    rtmp_frames = frames
                    rtmp_audio_parts: list[bytes] | None = None
                    rtmp_segment_audio_offsets: list[int] | None = None
                    rtmp_pairwise_source_frames: int | None = None
                    rtmp_clip_records: list[tuple[_PendingVideoBlock, _SegmentAudio | None, int, int]] = []
                    rtmp_subtitle_audio_ranges: list[tuple[_SegmentAudio, int, int]] = []
                    rtmp_subtitle_frame_map: list[tuple[_SegmentAudio, int] | None] | None = None
                    rtmp_source_total_frames = int(source_frame_count_initial)
                    if bool(self.live_rife_enabled) and not bool(gap_fill):
                        source_frame_count = int(rtmp_source_total_frames)
                        if source_frame_count >= 2:
                            source_audio_parts: list[bytes] = []
                            source_cursor = 0
                            source_audio_cursor = 0
                            for batch_block in batch_blocks:
                                batch_segment_audio = None
                                if batch_block is block:
                                    batch_segment_audio = segment_audio
                                elif str(batch_block.segment_id or "").strip():
                                    batch_segment_audio = self.segment_audio.pop(str(batch_block.segment_id), None)
                                batch_frame_count = int(self._block_frame_count(batch_block))
                                if batch_segment_audio is not None:
                                    audio_part, _batch_audio_offset = self._take_segment_audio_bytes_for_video(
                                        batch_segment_audio,
                                        offset_bytes=0,
                                        frame_count=int(batch_frame_count),
                                        pad_missing=bool(self.publish_draining) or not bool(self.rtmp_require_real_audio),
                                    )
                                else:
                                    target_samples = int(self._peek_target_audio_samples_for_video(int(batch_frame_count)))
                                    if (
                                        not bool(self.publish_draining)
                                        and target_samples > 0
                                        and float(self.rtmp_audio_wait_sec) > 0.0
                                    ):
                                        wait_start = time.perf_counter()
                                        deadline = time.perf_counter() + float(self.rtmp_audio_wait_sec)
                                        while (
                                            int(self._pending_audio_samples()) < int(target_samples)
                                            and time.perf_counter() < deadline
                                            and not self.publish_stop.is_set()
                                        ):
                                            await asyncio.sleep(0.005)
                                        waited = max(0.0, time.perf_counter() - wait_start)
                                        if waited > 0.001:
                                            self.rtmp_audio_wait_events += 1
                                            self.rtmp_audio_wait_total_sec += float(waited)
                                    if (
                                        not bool(self.publish_draining)
                                        and bool(self.rtmp_require_real_audio)
                                        and int(target_samples) > 0
                                        and int(self._pending_audio_samples()) < int(target_samples)
                                    ):
                                        self._raise_rtmp_media_runway_invariant(
                                            "audio_hold",
                                            buffered_frames=int(self.buffered_video_frames),
                                            pending_audio_sec=f"{float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))):.2f}",
                                            pending_blocks=int(len(self.pending_blocks)),
                                            target_samples=int(target_samples),
                                        )
                                    audio_part = self._take_audio_bytes_for_video(
                                        int(batch_frame_count),
                                        pad_missing=bool(self.publish_draining) or not bool(self.rtmp_require_real_audio),
                                    )
                                source_audio_parts.append(bytes(audio_part))
                                audio_start = int(source_audio_cursor)
                                source_audio_cursor += int(len(audio_part))
                                if batch_segment_audio is not None:
                                    rtmp_subtitle_audio_ranges.append(
                                        (
                                            batch_segment_audio,
                                            int(audio_start),
                                            int(source_audio_cursor),
                                        )
                                    )
                                if str(batch_block.segment_id or "").strip():
                                    rtmp_clip_records.append(
                                        (
                                            batch_block,
                                            batch_segment_audio,
                                            int(source_cursor),
                                            int(batch_frame_count),
                                        )
                                    )
                                source_cursor += int(batch_frame_count)
                            audio_block = b"".join(source_audio_parts)
                            if len(batch_blocks) != 1:
                                segment_audio = None
                            else:
                                segment_audio_offset = 0
                            if bool(live_rife_pairwise):
                                target_output_frames = max(
                                    int(source_frame_count),
                                    int(
                                        round(
                                            float(source_frame_count)
                                            * float(self._rtmp_pipe_fps())
                                            / float(max(1, int(self.cfg.fps)))
                                        )
                                    ),
                                )
                                rtmp_audio_parts = self._split_pcm16le_for_frames(bytes(audio_block), int(target_output_frames))
                                rtmp_pairwise_source_frames = int(source_frame_count)
                                rtmp_frames = []
                            else:
                                rtmp_frames = await asyncio.to_thread(
                                    self._interpolate_live_blocks_rife,
                                    list(batch_blocks),
                                    input_fps=int(self.cfg.fps),
                                    output_fps=int(self._rtmp_pipe_fps()),
                                )
                                rtmp_audio_parts = self._split_pcm16le_for_frames(bytes(audio_block), int(len(rtmp_frames)))
                            offsets: list[int] = []
                            running = 0
                            for part in rtmp_audio_parts or []:
                                offsets.append(int(running))
                                running += int(len(part))
                            if segment_audio is not None:
                                rtmp_segment_audio_offsets = offsets
                            if rtmp_subtitle_audio_ranges and rtmp_audio_parts is not None:
                                rtmp_subtitle_frame_map = self._rtmp_subtitle_frame_map_from_audio_parts(
                                    list(rtmp_subtitle_audio_ranges),
                                    list(rtmp_audio_parts or []),
                                    list(offsets),
                                )
                    if (not bool(gap_fill)) and rtmp_frames:
                        rtmp_frames = self._apply_avatar_transition_blur(
                            [bytes(frame) for frame in rtmp_frames],
                            batch_blocks=list(batch_blocks),
                            output_fps=int(self._rtmp_pipe_fps()),
                            next_avatar_ref_path=str(self._peek_pending_avatar_ref_path()),
                        )
                    if rtmp_pairwise_source_frames is not None:
                        target_output_frames = int(len(rtmp_audio_parts or []))
                        emitted_frames = 0

                        async def push_pairwise_frame(frame_bytes: bytes) -> None:
                            nonlocal emitted_frames, frame_idx
                            if int(emitted_frames) >= int(target_output_frames):
                                return
                            part_idx = int(emitted_frames)
                            emitted_frames += 1
                            segment_audio_offset_before = (
                                int(rtmp_segment_audio_offsets[int(part_idx)])
                                if rtmp_segment_audio_offsets is not None and int(part_idx) < int(len(rtmp_segment_audio_offsets))
                                else 0
                            )
                            audio = (
                                bytes(rtmp_audio_parts[int(part_idx)])
                                if rtmp_audio_parts is not None and int(part_idx) < int(len(rtmp_audio_parts))
                                else self._silence_audio_bytes_for_output_frames(1, fps=int(self._rtmp_pipe_fps()))
                            )
                            frame = bytes(frame_bytes)
                            subtitle_segment_audio = segment_audio
                            subtitle_segment_audio_offset_before = int(segment_audio_offset_before)
                            if rtmp_subtitle_frame_map is not None and int(part_idx) < int(len(rtmp_subtitle_frame_map)):
                                subtitle_entry = rtmp_subtitle_frame_map[int(part_idx)]
                                subtitle_segment_audio = None
                                subtitle_segment_audio_offset_before = 0
                                if subtitle_entry is not None:
                                    subtitle_segment_audio, subtitle_segment_audio_offset_before = subtitle_entry
                            frame = self._apply_avatar_transition_silence_hold(
                                frame,
                                audio,
                                current_avatar_ref_path=str(self._last_nonempty_avatar_ref_path(list(batch_blocks))),
                                next_avatar_ref_path=str(self._peek_pending_avatar_ref_path()),
                                segment_audio=subtitle_segment_audio,
                                segment_audio_offset_before=int(subtitle_segment_audio_offset_before),
                            )
                            if subtitle_segment_audio is not None:
                                frame = self._subtitle_frame(
                                    frame,
                                    segment_audio=subtitle_segment_audio,
                                    segment_audio_offset_before=int(subtitle_segment_audio_offset_before),
                                    audio=audio,
                                )
                            await self.push_rtmp_frame(frame, audio=audio, gap_fill=False)
                            frame_idx += 1
                            self.rtmp_gap_fill_streak_frames = 0

                        source_frame_count = int(rtmp_pairwise_source_frames)
                        if source_frame_count <= 1:
                            for frame in self._fit_live_rife_frame_count(list(frames), int(target_output_frames)):
                                await push_pairwise_frame(frame)
                        else:
                            for source_idx in range(0, int(source_frame_count) - 1):
                                if int(emitted_frames) >= int(target_output_frames):
                                    break
                                pair_frames = await asyncio.to_thread(
                                    self._interpolate_live_frames_rife,
                                    [bytes(frames[int(source_idx)]), bytes(frames[int(source_idx) + 1])],
                                    input_fps=int(self.cfg.fps),
                                    output_fps=int(self._rtmp_pipe_fps()),
                                )
                                first_frame, mid_frame, second_frame = self._live_rife_pair_output_frames(
                                    [bytes(frames[int(source_idx)]), bytes(frames[int(source_idx) + 1])],
                                    pair_frames,
                                )
                                if int(source_idx) == 0:
                                    await push_pairwise_frame(first_frame)
                                await push_pairwise_frame(mid_frame)
                                await push_pairwise_frame(second_frame)
                            while int(emitted_frames) < int(target_output_frames) and frames:
                                await push_pairwise_frame(bytes(frames[-1]))
                    else:
                        while int(frame_idx) < int(len(rtmp_frames)):
                            frame = rtmp_frames[int(frame_idx)]
                            target_samples = int(self._peek_target_audio_samples_for_video(1))
                            if bool(gap_fill):
                                # Hold video with silence only. Consuming queued
                                # speech audio here makes audio run ahead of the
                                # real generated frames and causes cumulative A/V
                                # drift once generation catches up.
                                audio = self._silence_audio_bytes_for_output_frames(1, fps=int(self._rtmp_pipe_fps()))
                            elif rtmp_audio_parts is not None:
                                segment_audio_offset_before = (
                                    int(rtmp_segment_audio_offsets[int(frame_idx)])
                                    if rtmp_segment_audio_offsets is not None
                                    else 0
                                )
                                audio = bytes(rtmp_audio_parts[int(frame_idx)])
                            elif segment_audio is not None:
                                segment_audio_offset_before = int(segment_audio_offset)
                                audio, segment_audio_offset = self._take_segment_audio_bytes_for_video(
                                    segment_audio,
                                    offset_bytes=int(segment_audio_offset),
                                    frame_count=1,
                                    pad_missing=bool(self.publish_draining) or not bool(self.rtmp_require_real_audio),
                                )
                            else:
                                segment_audio_offset_before = 0
                                if (
                                    not bool(self.publish_draining)
                                    and target_samples > 0
                                    and float(self.rtmp_audio_wait_sec) > 0.0
                                ):
                                    wait_start = time.perf_counter()
                                    deadline = time.perf_counter() + float(self.rtmp_audio_wait_sec)
                                    while (
                                        int(self._pending_audio_samples()) < int(target_samples)
                                        and time.perf_counter() < deadline
                                        and not self.publish_stop.is_set()
                                    ):
                                        await asyncio.sleep(0.005)
                                    waited = max(0.0, time.perf_counter() - wait_start)
                                    if waited > 0.001:
                                        self.rtmp_audio_wait_events += 1
                                        self.rtmp_audio_wait_total_sec += float(waited)
                                if (
                                    not bool(self.publish_draining)
                                    and bool(self.rtmp_require_real_audio)
                                    and int(target_samples) > 0
                                    and int(self._pending_audio_samples()) < int(target_samples)
                                ):
                                    self._raise_rtmp_media_runway_invariant(
                                        "audio_hold",
                                        buffered_frames=int(self.buffered_video_frames),
                                        pending_audio_sec=f"{float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))):.2f}",
                                        pending_blocks=int(len(self.pending_blocks)),
                                        target_samples=int(target_samples),
                                    )
                                audio = self._take_audio_bytes_for_video(
                                    1,
                                    pad_missing=bool(self.publish_draining) or not bool(self.rtmp_require_real_audio),
                                )
                            subtitle_segment_audio = segment_audio
                            subtitle_segment_audio_offset_before = int(segment_audio_offset_before)
                            if rtmp_subtitle_frame_map is not None and int(frame_idx) < int(len(rtmp_subtitle_frame_map)):
                                subtitle_entry = rtmp_subtitle_frame_map[int(frame_idx)]
                                subtitle_segment_audio = None
                                subtitle_segment_audio_offset_before = 0
                                if subtitle_entry is not None:
                                    subtitle_segment_audio, subtitle_segment_audio_offset_before = subtitle_entry
                            if not bool(gap_fill):
                                frame = self._apply_avatar_transition_silence_hold(
                                    bytes(frame),
                                    audio,
                                    current_avatar_ref_path=str(self._last_nonempty_avatar_ref_path(list(batch_blocks))),
                                    next_avatar_ref_path=str(self._peek_pending_avatar_ref_path()),
                                    segment_audio=subtitle_segment_audio,
                                    segment_audio_offset_before=int(subtitle_segment_audio_offset_before),
                                )
                            if (not bool(gap_fill)) and subtitle_segment_audio is not None:
                                frame = self._subtitle_frame(
                                    frame,
                                    segment_audio=subtitle_segment_audio,
                                    segment_audio_offset_before=int(subtitle_segment_audio_offset_before),
                                    audio=audio,
                                )
                            await self.push_rtmp_frame(frame, audio=audio, gap_fill=bool(gap_fill))
                            frame_idx += 1
                            if bool(gap_fill):
                                self.rtmp_gap_fill_frames += 1
                                self.rtmp_gap_fill_streak_frames += 1
                                if str(gap_fill_reason) == "audio_hold":
                                    self.rtmp_audio_hold_frames += 1
                                    now_hold_log = time.monotonic()
                                    if now_hold_log - float(self.rtmp_audio_hold_last_log) >= 10.0:
                                        logging.warning(
                                            "Remote edge RTMP waiting for real audio: session=%s job=%s hold_frames=%d pending_blocks=%d buffered_frames=%d pending_audio_sec=%.2f",
                                            self.cfg.session_id,
                                            self.cfg.job_id,
                                            int(self.rtmp_audio_hold_frames),
                                            int(len(self.pending_blocks)),
                                            int(self.buffered_video_frames),
                                            float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
                                        )
                                        self.rtmp_audio_hold_last_log = float(now_hold_log)
                                max_gap_frames = int(
                                    round(float(self.rtmp_gap_fill_fail_sec) * float(max(1, int(self._rtmp_pipe_fps()))))
                                )
                                if int(max_gap_frames) > 0 and int(self.rtmp_gap_fill_streak_frames) > int(max_gap_frames):
                                    raise RuntimeError(
                                        "RTMP gap-fill persisted "
                                        f"{self.rtmp_gap_fill_streak_frames / float(max(1, int(self._rtmp_pipe_fps()))):.1f}s "
                                        f"without real frames"
                                    )
                                now_log = time.monotonic()
                                if now_log - float(self.rtmp_gap_fill_last_log) >= 10.0:
                                    logging.warning(
                                        "Remote edge RTMP gap-fill active: session=%s job=%s reason=%s gap_frames=%d pending_blocks=%d buffered_frames=%d pending_audio_sec=%.2f",
                                        self.cfg.session_id,
                                        self.cfg.job_id,
                                        str(gap_fill_reason),
                                        int(self.rtmp_gap_fill_frames),
                                        int(len(self.pending_blocks)),
                                        int(self.buffered_video_frames),
                                        float(self._pending_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
                                    )
                                    self.rtmp_gap_fill_last_log = float(now_log)
                            else:
                                self.rtmp_gap_fill_streak_frames = 0
                    segment_end_frame = int(self.frames)
                    if (not bool(gap_fill)) and rtmp_clip_records:
                        output_total = max(1, int(segment_end_frame) - int(segment_start_frame))
                        source_total = max(1, int(rtmp_source_total_frames))
                        for record_block, record_audio, source_start, source_count in rtmp_clip_records:
                            record_start = int(segment_start_frame) + int(
                                round(float(source_start) * float(output_total) / float(source_total))
                            )
                            record_end = int(segment_start_frame) + int(
                                round(float(int(source_start) + int(source_count)) * float(output_total) / float(source_total))
                            )
                            record_rel_start = max(0, int(record_start) - int(segment_start_frame))
                            record_rel_end = max(int(record_rel_start), int(record_end) - int(segment_start_frame))
                            record_frames = (
                                list(rtmp_frames[int(record_rel_start): int(record_rel_end)])
                                if rtmp_frames
                                else self._block_frames_bytes(record_block)
                            )
                            self._record_clip_segment(
                                record_block,
                                frames=record_frames,
                                segment_audio=record_audio,
                                start_frame=int(record_start),
                                end_frame=int(max(record_start, record_end)),
                            )
                    elif (not bool(gap_fill)) and str(block.segment_id or "").strip():
                        self._record_clip_segment(
                            block,
                            frames=rtmp_frames if rtmp_frames else (frames if frames else self._block_frames_bytes(block)),
                            segment_audio=segment_audio,
                            start_frame=int(segment_start_frame),
                            end_frame=int(segment_end_frame),
                        )
                    self.audio_chunks_published += 1
                    self.log_stats()
                    continue
                segment_audio = None
                segment_start_frame = int(self.frames)
                frames = frames if frames else self._block_frames_bytes(block)
                if str(block.segment_id or "").strip():
                    segment_audio = self.segment_audio.pop(str(block.segment_id), None)
                    audio_chunks = (
                        [(segment_audio.payload, int(segment_audio.sample_rate))]
                        if segment_audio is not None
                        else []
                    )
                else:
                    audio_chunks = self._take_audio_for_video(len(frames))
                frames = self._apply_avatar_transition_blur(
                    [bytes(frame) for frame in frames],
                    batch_blocks=[block],
                    output_fps=int(self.cfg.fps),
                    next_avatar_ref_path=str(self._peek_pending_avatar_ref_path()),
                )
                await self.push_rgb24_many(frames, timestamp_us=timestamp_us)
                for payload, sample_rate in audio_chunks:
                    await self.push_pcm16le(payload, sample_rate=int(sample_rate))
                segment_end_frame = int(self.frames)
                if str(block.segment_id or "").strip():
                    self._record_clip_segment(
                        block,
                        frames=frames,
                        segment_audio=segment_audio,
                        start_frame=int(segment_start_frame),
                        end_frame=int(segment_end_frame),
                    )
                self.log_stats()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.publish_failed_exc = e
            self.publish_stop.set()
            self._stop_rtmp()
            async with self.publish_cv:
                self.publish_cv.notify_all()
            logging.exception("Remote edge publish loop failed: session=%s job=%s", self.cfg.session_id, self.cfg.job_id)
            raise

    async def stop_publish_loop(self) -> None:
        self.publish_stop.set()
        await self._stop_rtmp_idle_pump()
        async with self.publish_cv:
            self.publish_cv.notify_all()
        if self.rtmp_drain_task is not None:
            self.rtmp_drain_task.cancel()
            await asyncio.gather(self.rtmp_drain_task, return_exceptions=True)
            self.rtmp_drain_task = None
        if self.publish_task is not None:
            self.publish_task.cancel()
            await asyncio.gather(self.publish_task, return_exceptions=True)
            self.publish_task = None

    async def drain_live_output(self) -> None:
        timeout = max(
            1.0,
            min(600.0, _safe_float_env("REMOTE_EDGE_LIVE_DRAIN_TIMEOUT_SEC", 180.0)),
        )
        deadline = time.monotonic() + float(timeout)
        logging.warning(
            "Remote edge live drain started: session=%s job=%s latent_q=%d pending_blocks=%d buffered_frames=%d rtmp_queue=%d timeout=%.1fs",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0,
            int(len(self.pending_blocks)),
            int(self.buffered_video_frames),
            int(self._rtmp_queue_len()),
            float(timeout),
        )
        try:
            await asyncio.wait_for(
                self._drain_latent_decode_loop(),
                timeout=max(0.1, float(deadline - time.monotonic())),
            )
            self.publish_draining = True
            await self._drain_live_publish_loop(deadline=deadline)
            await self._drain_rtmp_writer_queue(deadline=deadline)
        except BaseException:
            logging.exception(
                "Remote edge live drain failed: session=%s job=%s pending_blocks=%d buffered_frames=%d rtmp_queue=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
                int(self._rtmp_queue_len()),
            )
            await self.stop_latent_decode_loop()
            await self.stop_publish_loop()
            raise
        logging.warning(
            "Remote edge live drain complete: session=%s job=%s frames=%d audio_frames=%d pending_blocks=%d buffered_frames=%d rtmp_queue=%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.frames),
            int(self.audio_frames),
            int(len(self.pending_blocks)),
            int(self.buffered_video_frames),
            int(self._rtmp_queue_len()),
        )

    async def _drain_latent_decode_loop(self) -> None:
        queue = self.latent_decode_queue
        task = self.latent_decode_task
        if queue is None or task is None:
            return
        if task.done():
            self.latent_decode_task = None
            try:
                exc = task.exception()
            except asyncio.CancelledError as e:
                exc = e
            if exc is not None:
                raise RuntimeError(f"remote edge latent decode failed: {exc}") from exc
            if self.latent_decode_failed_exc is not None:
                raise RuntimeError(f"remote edge latent decode failed: {self.latent_decode_failed_exc}") from self.latent_decode_failed_exc
            return
        await queue.join()
        await queue.put(None)
        await task
        self.latent_decode_task = None
        await self._drain_latent_postprocess_loop()
        if self.latent_decode_failed_exc is not None:
            raise RuntimeError(f"remote edge latent decode failed: {self.latent_decode_failed_exc}") from self.latent_decode_failed_exc

    async def _drain_latent_postprocess_loop(self) -> None:
        queue = self.latent_postprocess_queue
        task = self.latent_postprocess_task
        if queue is None or task is None:
            return
        if task.done():
            self.latent_postprocess_task = None
            try:
                exc = task.exception()
            except asyncio.CancelledError as e:
                exc = e
            if exc is not None:
                raise RuntimeError(f"remote edge latent postprocess failed: {exc}") from exc
            return
        await queue.join()
        await queue.put(None)
        await task
        self.latent_postprocess_task = None

    async def _drain_live_publish_loop(self, *, deadline: float) -> None:
        task = self.publish_task
        if task is None:
            return
        last_log = 0.0
        async with self.publish_cv:
            self.publish_cv.notify_all()
        while True:
            if self.publish_failed_exc is not None:
                raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc
            if task.done():
                try:
                    exc = task.exception()
                except asyncio.CancelledError as e:
                    exc = e
                if exc is not None:
                    raise RuntimeError(f"remote edge publish failed: {exc}") from exc
                self.publish_task = None
                return
            if time.monotonic() >= float(deadline):
                raise TimeoutError(
                    "remote edge live publish drain timed out "
                    f"buffered_frames={self.buffered_video_frames} pending_blocks={len(self.pending_blocks)}"
                )
            now = time.monotonic()
            if now - float(last_log) >= 5.0:
                logging.warning(
                    "Remote edge live drain waiting: session=%s job=%s pending_blocks=%d buffered_frames=%d pending_audio=%d rtmp_queue=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(len(self.pending_blocks)),
                    int(self.buffered_video_frames),
                    int(len(self.pending_audio)),
                    int(self._rtmp_queue_len()),
                )
                last_log = float(now)
            async with self.publish_cv:
                self.publish_cv.notify_all()
            await asyncio.sleep(0.05)

    async def _drain_rtmp_writer_queue(self, *, deadline: float) -> None:
        if str(self.cfg.output or "livekit").strip().lower() != "rtmp":
            return
        last_log = 0.0
        while self._rtmp_queue_len() > 0:
            failed = self._rtmp_writer_failed()
            if failed is not None:
                raise RuntimeError(f"RTMP writer failed during live drain: {failed}") from failed
            if time.monotonic() >= float(deadline):
                raise TimeoutError(f"remote edge RTMP queue drain timed out queue={self._rtmp_queue_len()}")
            now = time.monotonic()
            if now - float(last_log) >= 5.0:
                logging.warning(
                    "Remote edge RTMP queue drain waiting: session=%s job=%s queue=%d writer_ticks=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self._rtmp_queue_len()),
                    int(self.rtmp_writer_ticks),
                )
                last_log = float(now)
            with self.rtmp_writer_cv:
                self.rtmp_writer_cv.notify_all()
            await asyncio.sleep(0.02)

    def _file_output_source_progress(self) -> float:
        target_frames = 0
        if float(self.cfg.file_target_duration_sec or 0.0) > 0.0:
            target_frames = int(round(float(self.cfg.file_target_duration_sec) * float(max(1, int(self.cfg.fps)))))
        if int(target_frames) <= 0 and int(self.cfg.file_target_audio_samples or 0) > 0:
            target_frames = int(
                round(
                    (float(self.cfg.file_target_audio_samples) / float(max(1, int(self.cfg.sample_rate))))
                    * float(max(1, int(self.cfg.fps)))
                )
            )
        if int(target_frames) <= 0:
            return 0.0
        return float(max(0.0, min(1.0, float(self.frames) / float(max(1, int(target_frames))))))

    async def _drain_file_latent_decode_loop(self, *, progress_cb: Any | None = None) -> None:
        queue = self.latent_decode_queue
        task = self.latent_decode_task
        if queue is None or task is None:
            return
        if task.done():
            self.latent_decode_task = None
            try:
                exc = task.exception()
            except asyncio.CancelledError as e:
                exc = e
            if exc is not None:
                raise RuntimeError(f"remote edge latent decode failed: {exc}") from exc
            if self.latent_decode_failed_exc is not None:
                raise RuntimeError(f"remote edge latent decode failed: {self.latent_decode_failed_exc}") from self.latent_decode_failed_exc
            return
        join_task = asyncio.create_task(queue.join())
        try:
            while not join_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(join_task), timeout=1.0)
                except asyncio.TimeoutError:
                    if progress_cb is not None:
                        await progress_cb(
                            "drain",
                            self._file_output_source_progress(),
                            frames=int(self.frames),
                            latent_q=int(queue.qsize()),
                            pending_blocks=int(len(self.pending_blocks)),
                        )
        finally:
            if not join_task.done():
                join_task.cancel()
                await asyncio.gather(join_task, return_exceptions=True)
        await queue.put(None)
        await task
        self.latent_decode_task = None
        if self.latent_decode_failed_exc is not None:
            raise RuntimeError(f"remote edge latent decode failed: {self.latent_decode_failed_exc}") from self.latent_decode_failed_exc

    async def _drain_file_publish_loop(self, *, progress_cb: Any | None = None) -> None:
        task = self.publish_task
        if task is None:
            return
        deadline = time.monotonic() + max(30.0, _safe_float_env("REMOTE_EDGE_FILE_DRAIN_TIMEOUT_SEC", 600.0))
        last_progress = 0.0
        while True:
            if self.publish_failed_exc is not None:
                raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc
            async with self.publish_cv:
                if int(self.buffered_video_frames) <= 0 and not self.pending_blocks:
                    break
                self.publish_cv.notify_all()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "remote edge file publish drain timed out "
                    f"buffered_frames={self.buffered_video_frames} pending_blocks={len(self.pending_blocks)}"
                )
            now = time.monotonic()
            if progress_cb is not None and now - float(last_progress) >= 1.0:
                await progress_cb(
                    "drain",
                    self._file_output_source_progress(),
                    frames=int(self.frames),
                    buffered_frames=int(self.buffered_video_frames),
                    pending_blocks=int(len(self.pending_blocks)),
                )
                last_progress = float(now)
            await asyncio.sleep(0.05)
        self.publish_stop.set()
        async with self.publish_cv:
            self.publish_cv.notify_all()
        await task
        self.publish_task = None
        if self.publish_failed_exc is not None:
            raise RuntimeError(f"remote edge publish failed: {self.publish_failed_exc}") from self.publish_failed_exc

    def _flush_file_pending_audio(self) -> None:
        while self.pending_audio:
            payload, _sample_rate = self.pending_audio.popleft()
            self._write_file_audio(bytes(payload))
        for segment_audio in list(self.segment_audio.values()):
            self._write_file_audio(bytes(segment_audio.payload))
        self.segment_audio.clear()

    async def finish_file_output(self, *, progress_cb: Any | None = None) -> dict[str, Any]:
        loop = asyncio.get_running_loop()

        async def emit_async(phase: str, progress: float, **extra: Any) -> None:
            if progress_cb is None:
                return
            payload = {
                "phase": str(phase or ""),
                "progress": float(max(0.0, min(1.0, float(progress)))),
                "session_id": str(self.cfg.session_id or ""),
                "job_id": str(self.cfg.job_id or ""),
                **dict(extra or {}),
            }
            try:
                res = progress_cb(payload)
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                logging.exception(
                    "Remote edge file progress send failed: session=%s job=%s phase=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    phase,
                )

        def emit_thread(phase: str, progress: float, **extra: Any) -> None:
            if progress_cb is None:
                return
            payload = {
                "phase": str(phase or ""),
                "progress": float(max(0.0, min(1.0, float(progress)))),
                "session_id": str(self.cfg.session_id or ""),
                "job_id": str(self.cfg.job_id or ""),
                **dict(extra or {}),
            }
            try:
                fut = asyncio.run_coroutine_threadsafe(progress_cb(payload), loop)
                fut.result(timeout=5.0)
            except Exception:
                logging.exception(
                    "Remote edge file progress send failed: session=%s job=%s phase=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    phase,
                )

        await emit_async("drain", 0.0)
        await self._drain_file_latent_decode_loop(progress_cb=emit_async)
        await self._drain_file_publish_loop(progress_cb=emit_async)
        self._flush_file_pending_audio()
        await emit_async("encode", 0.0, frames=int(self.frames))
        result = await asyncio.to_thread(self._encode_and_upload_file_output, emit_thread)
        keep = bool(_env_flag("REMOTE_EDGE_FILE_KEEP_LOCAL", "0"))
        if not keep and self.file_work_dir:
            shutil.rmtree(self.file_work_dir, ignore_errors=True)
        await emit_async("done", 1.0, **dict(result or {}))
        return result

    def _record_file_requested_restore(self, *, face_restore: float | None, background_restore: float | None) -> None:
        if str(self.cfg.output or "").strip().lower() != "file":
            return
        try:
            face = max(0.0, float(0.0 if face_restore is None else face_restore))
        except Exception:
            face = 0.0
        try:
            background = max(0.0, float(0.0 if background_restore is None else background_restore))
        except Exception:
            background = 0.0
        self.file_requested_face_restore = max(float(self.file_requested_face_restore), float(face))
        self.file_requested_background_restore = max(float(self.file_requested_background_restore), float(background))

    def _file_post_vae_encode_cmd(
        self,
        *,
        input_video: str,
        output_video: str,
        width: int,
        height: int,
        fps: int,
    ) -> list[str]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(fps)),
            "-i",
            "pipe:0",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
        ]
        cmd.extend(["-r", str(int(fps))])
        cmd.extend(self._file_video_encode_args(output_fps=int(fps)))
        cmd.extend(self._file_audio_encode_args())
        cmd.extend(["-shortest", str(output_video)])
        return cmd

    def _bgr_frame_buffers_to_tensor01(self, buffers: list[bytearray], *, width: int, height: int) -> Any:
        if not buffers:
            raise RuntimeError("post-vae file batch is empty")
        if self.latent_decoder is None:
            self.latent_decoder = get_shared_wan_latent_decoder()
        torch_mod = getattr(self.latent_decoder, "torch", None)
        if torch_mod is None:
            import torch as torch_mod  # type: ignore[no-redef]
        device = getattr(self.latent_decoder, "device", "cuda:0")
        frames = [
            torch_mod.frombuffer(buf, dtype=torch_mod.uint8).view(int(height), int(width), 3)
            for buf in buffers
        ]
        tensor = torch_mod.stack(frames, dim=0).permute(0, 3, 1, 2).contiguous()
        tensor = tensor.to(device=device)
        tensor = tensor[:, [2, 1, 0], :, :].to(dtype=torch_mod.float32).div_(255.0)
        return tensor

    def _tensor01_to_bgr_bytes_list(self, frames_01: Any) -> list[bytes]:
        if self.latent_decoder is None:
            self.latent_decoder = get_shared_wan_latent_decoder()
        torch_mod = getattr(self.latent_decoder, "torch", None)
        if torch_mod is None:
            import torch as torch_mod  # type: ignore[no-redef]
        with torch_mod.inference_mode():
            bgr_u8 = (
                frames_01
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch_mod.uint8)[:, [2, 1, 0], :, :]
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
        return [bgr_u8[idx].tobytes() for idx in range(int(bgr_u8.shape[0]))]

    def _start_file_progress_heartbeat(
        self,
        progress_cb: Any | None,
        *,
        phase: str,
        estimate_sec: float,
        start_progress: float = 0.05,
        max_progress: float = 0.95,
        interval_sec: float = 3.0,
        **fields: Any,
    ) -> tuple[threading.Event | None, threading.Thread | None]:
        if progress_cb is None:
            return None, None
        phase_s = str(phase or "").strip()
        if not phase_s:
            return None, None
        estimate = max(1.0, float(estimate_sec or 1.0))
        start_p = max(0.0, min(1.0, float(start_progress)))
        max_p = max(start_p, min(0.99, float(max_progress)))
        interval = max(0.5, float(interval_sec or 3.0))
        stop = threading.Event()
        started = time.perf_counter()

        def run() -> None:
            while not stop.wait(float(interval)):
                elapsed = max(0.0, time.perf_counter() - float(started))
                frac = min(1.0, float(elapsed) / float(estimate))
                progress = min(float(max_p), float(start_p) + (float(max_p) - float(start_p)) * float(frac))
                try:
                    progress_cb(str(phase_s), float(progress), elapsed_sec=float(elapsed), **dict(fields or {}))
                except Exception:
                    logging.exception(
                        "Remote edge file progress heartbeat failed: session=%s job=%s phase=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        str(phase_s),
                    )

        thread = threading.Thread(
            target=run,
            name=f"remote-edge-file-progress-{phase_s}-{self.cfg.session_id}",
            daemon=True,
        )
        thread.start()
        return stop, thread

    def _resize_file_video_to_output(
        self,
        *,
        input_video: str,
        output_video: str,
        fps: int,
        progress_cb: Any | None = None,
    ) -> str:
        input_abs = os.path.abspath(str(input_video))
        output_abs = os.path.abspath(str(output_video))
        os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)
        target_w = max(1, int(self.cfg.width))
        target_h = max(1, int(self.cfg.height))
        probe = probe_video_metadata(input_abs)
        source_w = max(1, int(probe.width or self.file_frame_width or target_w))
        source_h = max(1, int(probe.height or self.file_frame_height or target_h))
        final_filters = self._file_video_filters(width=int(target_w), height=int(target_h))
        if int(source_w) == int(target_w) and int(source_h) == int(target_h) and not final_filters:
            return str(input_abs)
        fps_i = max(1, int(fps or round(float(probe.fps or self.file_raw_video_fps or self.cfg.fps))))
        filters = []
        if int(source_w) != int(target_w) or int(source_h) != int(target_h):
            filters.extend(
                [
                    f"scale={int(target_w)}:{int(target_h)}:force_original_aspect_ratio=increase",
                    f"crop={int(target_w)}:{int(target_h)}",
                ]
            )
        filters.append("setsar=1")
        filters.extend(final_filters)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(input_abs),
            "-vf",
            ",".join(filters),
            "-r",
            str(int(fps_i)),
        ]
        cmd.extend(self._file_video_encode_args(output_fps=int(fps_i)))
        cmd.extend(self._file_audio_encode_args())
        cmd.append(str(output_abs))
        if progress_cb is not None:
            progress_cb("final_resize", 0.05, source=f"{source_w}x{source_h}", target=f"{target_w}x{target_h}")
        duration_sec = float(probe.duration_sec or 0.0)
        timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_FINAL_RESIZE_TIMEOUT_SEC", 60.0 + duration_sec * 8.0))
        started = time.perf_counter()
        self._run_file_subprocess(cmd, timeout=float(timeout), label="ffmpeg file final resize")
        if progress_cb is not None:
            progress_cb("final_resize", 1.0, source=f"{source_w}x{source_h}", target=f"{target_w}x{target_h}")
        size = int(os.path.getsize(output_abs)) if os.path.exists(output_abs) else 0
        logging.warning(
            "Remote edge FILE final resize done: session=%s job=%s %dx%d->%dx%d fps=%d elapsed=%.3fs bytes=%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(source_w),
            int(source_h),
            int(target_w),
            int(target_h),
            int(fps_i),
            float(time.perf_counter() - float(started)),
            int(size),
            str(output_abs),
        )
        return str(output_abs)

    def _file_nvvfx_enabled(self) -> bool:
        return bool(
            _env_flag("REMOTE_EDGE_FILE_NVVFX", "0")
            or _env_flag("REMOTE_EDGE_FILE_NVIDIA_VFX", "0")
        )

    def _file_remote_upscale_url(self) -> str:
        return str(
            os.getenv("REMOTE_EDGE_FILE_UPSCALE_SERVICE_URL", os.getenv("SMARTBLOG_FILE_UPSCALE_SERVICE_URL", ""))
            or ""
        ).strip()

    def _file_remote_finalizer_enabled(self) -> bool:
        override = getattr(self.cfg, "file_remote_finalizer", None)
        if override is not None:
            return bool(override)
        return bool(
            str(self.cfg.output or "").strip().lower() == "file"
            and bool(self._file_remote_upscale_url())
            and _env_flag("REMOTE_EDGE_FILE_REMOTE_FINALIZER", "0")
        )

    def _apply_file_remote_upscale(
        self,
        *,
        input_video: str,
        output_video: str,
        progress_cb: Any | None = None,
    ) -> str:
        import requests

        url = self._file_remote_upscale_url()
        if not url:
            return str(input_video)
        input_abs = os.path.abspath(str(input_video))
        output_abs = os.path.abspath(str(output_video))
        os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)
        probe = probe_video_metadata(input_abs)
        duration_sec = float(probe.duration_sec or 0.0)
        estimate_sec = max(
            15.0,
            _safe_float_env(
                "REMOTE_EDGE_FILE_UPSCALE_EXPECTED_SEC",
                20.0 + max(0.0, float(duration_sec)) * 2.0,
            ),
        )
        stop, thread = self._start_file_progress_heartbeat(
            progress_cb,
            phase="upscale",
            estimate_sec=float(estimate_sec),
            start_progress=0.02,
            max_progress=0.98,
            interval_sec=max(1.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_HEARTBEAT_SEC", 3.0)),
            service=str(url),
        )
        started = time.perf_counter()
        tmp_path = f"{output_abs}.tmp"
        secret = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_SHARED_SECRET", "") or "").strip()
        headers = {"Authorization": f"Bearer {secret}"} if secret else {}
        direct_upload = bool(
            self._file_remote_finalizer_enabled()
            and _env_flag("REMOTE_EDGE_FILE_UPSCALE_DIRECT_UPLOAD", "0")
            and str(self.cfg.file_upload_url or "").strip()
        )
        params = {
            "quality": str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_QUALITY", "DEBLUR_HIGH") or "DEBLUR_HIGH"),
            "scale": str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_SCALE", "1") or "1"),
            "upscale": "1" if _env_flag("REMOTE_EDGE_FILE_UPSCALE_ENABLED", "0") else "0",
        }
        remote_rife = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE", "") or "").strip()
        if remote_rife:
            params["rife"] = str(remote_rife)
        remote_target_fps = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_FPS", "") or "").strip()
        if remote_target_fps:
            params["target_fps"] = str(remote_target_fps)
        remote_target_width = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_WIDTH", "") or "").strip()
        remote_target_height = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_HEIGHT", "") or "").strip()
        if bool(self._file_remote_finalizer_enabled()):
            if not remote_target_width:
                remote_target_width = str(max(1, int(self.cfg.width or 0)))
            if not remote_target_height:
                remote_target_height = str(max(1, int(self.cfg.height or 0)))
        if remote_target_width:
            params["target_width"] = str(remote_target_width)
        if remote_target_height:
            params["target_height"] = str(remote_target_height)
        remote_rife_batch = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE_BATCH_SOURCE_FRAMES", "") or "").strip()
        if remote_rife_batch:
            params["rife_batch_source_frames"] = str(remote_rife_batch)
        remote_rife_stage = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE_STAGE", "") or "").strip()
        if remote_rife_stage:
            params["rife_stage"] = str(remote_rife_stage)
        connect_timeout = max(3.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_CONNECT_TIMEOUT_SEC", 15.0))
        read_timeout = max(30.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_READ_TIMEOUT_SEC", 1800.0))
        try:
            if progress_cb is not None:
                progress_cb(
                    "upscale",
                    0.01,
                    service=str(url),
                    quality=str(params["quality"]),
                    scale=str(params["scale"]),
                    rife=str(params.get("rife", "")),
                    target_fps=str(params.get("target_fps", "")),
                    target_width=str(params.get("target_width", "")),
                    target_height=str(params.get("target_height", "")),
                    direct_upload=1 if bool(direct_upload) else 0,
                )
            with open(input_abs, "rb") as f:
                files = {"file": (os.path.basename(input_abs) or "input.mp4", f, "video/mp4")}
                data = {}
                if bool(direct_upload):
                    data["upload_url"] = str(self.cfg.file_upload_url or "")
                    data["upload_content_type"] = str(self.cfg.file_content_type or "video/mp4")
                resp = requests.post(
                    url,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    stream=True,
                    timeout=(float(connect_timeout), float(read_timeout)),
                )
            if int(resp.status_code) != 200:
                body = ""
                try:
                    body = str(resp.text or "")[-4000:]
                except Exception:
                    body = ""
                raise RuntimeError(f"upscale service failed HTTP {resp.status_code}: {body}")
            if bool(direct_upload):
                try:
                    payload = dict(resp.json() or {})
                except Exception as e:
                    raise RuntimeError("upscale service direct upload returned non-JSON response") from e
                if not bool(payload.get("uploaded")):
                    raise RuntimeError(f"upscale service direct upload did not confirm upload: keys={sorted(payload.keys())}")
                self.file_remote_upscale_direct_uploaded = True
                self.file_remote_upscale_result = dict(payload)
                logging.warning(
                    "Remote edge FILE remote upscale direct upload confirmed: session=%s job=%s bytes=%s frames=%s fps=%s size=%sx%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    str(payload.get("bytes", "")),
                    str(payload.get("frames", "")),
                    str(payload.get("output_fps", "")),
                    str(payload.get("output_width", "")),
                    str(payload.get("output_height", "")),
                )
                return str(input_abs)
            bytes_written = 0
            with open(tmp_path, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    out.write(chunk)
                    bytes_written += len(chunk)
            if int(bytes_written) <= 0:
                raise RuntimeError("upscale service returned an empty response")
            os.replace(str(tmp_path), str(output_abs))
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=2.0)
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
        elapsed = float(time.perf_counter() - float(started))
        size = int(os.path.getsize(output_abs)) if os.path.exists(output_abs) else 0
        logging.warning(
            "Remote edge FILE remote upscale done: session=%s job=%s url=%s quality=%s scale=%s target=%sx%s target_fps=%s elapsed=%.3fs bytes=%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            str(url),
            str(params["quality"]),
            str(params["scale"]),
            str(params.get("target_width", "")),
            str(params.get("target_height", "")),
            str(params.get("target_fps", "")),
            float(elapsed),
            int(size),
            str(output_abs),
        )
        if progress_cb is not None:
            progress_cb(
                "upscale",
                1.0,
                service=str(url),
                bytes=int(size),
                elapsed_sec=float(elapsed),
            )
        return str(output_abs)

    def _apply_file_nvvfx(
        self,
        *,
        input_video: str,
        output_video: str,
        progress_cb: Any | None = None,
    ) -> str:
        input_abs = os.path.abspath(str(input_video))
        output_abs = os.path.abspath(str(output_video))
        os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)
        scale = max(1, min(4, _safe_int_env("REMOTE_EDGE_FILE_NVVFX_SCALE", 1)))
        quality_name = str(os.getenv("REMOTE_EDGE_FILE_NVVFX_QUALITY", "DEBLUR_HIGH") or "DEBLUR_HIGH").strip().upper()
        gpu = max(0, _safe_int_env("REMOTE_EDGE_FILE_NVVFX_GPU", 0))
        started = time.perf_counter()

        try:
            import av
            import numpy as np
            import torch
            from nvvfx import VideoSuperRes
        except Exception as e:
            raise RuntimeError("NVIDIA VFX file postprocess requires av, torch, and nvidia-vfx") from e

        if quality_name not in VideoSuperRes.QualityLevel.__members__:
            raise RuntimeError(f"unknown NVIDIA VFX quality: {quality_name}")
        quality = VideoSuperRes.QualityLevel[quality_name]

        probe = probe_video_metadata(input_abs)
        input_container = av.open(str(input_abs))
        try:
            input_stream = input_container.streams.video[0]
            input_stream.thread_type = "AUTO"
            source_w = max(1, int(input_stream.codec_context.width or probe.width or self.cfg.width))
            source_h = max(1, int(input_stream.codec_context.height or probe.height or self.cfg.height))
            output_w = int(source_w * int(scale))
            output_h = int(source_h * int(scale))
            fps_f = float(probe.fps or input_stream.average_rate or self.file_raw_video_fps or self.cfg.fps or 30.0)
            fps_i = max(1, int(round(fps_f)))
            total_frames = int(getattr(input_stream, "frames", 0) or probe.frames or 0)

            torch.cuda.set_device(int(gpu))
            stream_ptr = torch.cuda.current_stream().cuda_stream
            sr = VideoSuperRes(device=int(gpu), quality=quality)
            sr.input_width = int(source_w)
            sr.input_height = int(source_h)
            sr.output_width = int(output_w)
            sr.output_height = int(output_h)
            load_started = time.perf_counter()
            sr.load()
            torch.cuda.synchronize()
            load_sec = float(time.perf_counter() - float(load_started))

            video_tmp = f"{os.path.splitext(output_abs)[0]}.nvvfx.video.mp4"
            mux_tmp = f"{os.path.splitext(output_abs)[0]}.nvvfx.mux.mp4"
            for tmp in (video_tmp, mux_tmp, output_abs):
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{int(output_w)}x{int(output_h)}",
                "-r",
                str(int(fps_i)),
                "-i",
                "pipe:0",
                "-an",
            ]
            cmd.extend(self._file_video_encode_args(output_fps=int(fps_i)))
            cmd.append(str(video_tmp))
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.stdin is None:
                raise RuntimeError("failed to open NVIDIA VFX ffmpeg stdin")

            processed = 0
            if progress_cb is not None:
                progress_cb(
                    "nvvfx",
                    0.02,
                    source=f"{source_w}x{source_h}",
                    output=f"{output_w}x{output_h}",
                    quality=str(quality_name),
                    scale=int(scale),
                )
            try:
                for frame in input_container.decode(input_stream):
                    arr = frame.to_ndarray(format="rgb24")
                    tensor = torch.from_numpy(np.asarray(arr)).to(f"cuda:{int(gpu)}")
                    tensor = tensor.permute(2, 0, 1).float().div_(255.0).contiguous()
                    vfx_output = sr.run(tensor, stream_ptr=stream_ptr)
                    rgb_output = torch.from_dlpack(vfx_output.image)
                    frame_np = (
                        rgb_output.clamp(0.0, 1.0)
                        .mul(255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    proc.stdin.write(frame_np.tobytes())
                    processed += 1
                    if progress_cb is not None and (processed == 1 or processed % max(1, int(fps_i)) == 0):
                        frac = (
                            min(0.98, float(processed) / float(max(1, int(total_frames))))
                            if int(total_frames) > 0
                            else 0.5
                        )
                        progress_cb(
                            "nvvfx",
                            float(frac),
                            frames=int(processed),
                            total_frames=int(total_frames),
                            quality=str(quality_name),
                            scale=int(scale),
                        )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            try:
                proc.wait(timeout=max(60.0, _safe_float_env("REMOTE_EDGE_FILE_NVVFX_ENCODE_WAIT_SEC", 600.0)))
            except subprocess.TimeoutExpired as e:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise RuntimeError("NVIDIA VFX ffmpeg encode timed out") from e
            stdout_b = proc.stdout.read() if proc.stdout is not None else b""
            stderr_b = proc.stderr.read() if proc.stderr is not None else b""
            if int(proc.returncode or 0) != 0:
                stderr_s = (stderr_b or b"").decode("utf-8", errors="replace")[-4000:]
                stdout_s = (stdout_b or b"").decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(f"NVIDIA VFX ffmpeg encode failed ({proc.returncode}): {stderr_s or stdout_s}")

            mux_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-i",
                str(video_tmp),
                "-i",
                str(input_abs),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                "-shortest",
                str(mux_tmp),
            ]
            timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_NVVFX_MUX_TIMEOUT_SEC", 60.0 + float(probe.duration_sec or 0.0) * 4.0))
            self._run_file_subprocess(mux_cmd, timeout=float(timeout), label="ffmpeg file NVIDIA VFX mux")
            os.replace(str(mux_tmp), str(output_abs))
            try:
                os.unlink(str(video_tmp))
            except Exception:
                pass
        finally:
            input_container.close()

        size = int(os.path.getsize(output_abs)) if os.path.exists(output_abs) else 0
        elapsed = float(time.perf_counter() - float(started))
        logging.warning(
            "Remote edge FILE NVIDIA VFX done: session=%s job=%s %dx%d->%dx%d quality=%s scale=%d load=%.3fs elapsed=%.3fs bytes=%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(source_w),
            int(source_h),
            int(output_w),
            int(output_h),
            str(quality_name),
            int(scale),
            float(load_sec),
            float(elapsed),
            int(size),
            str(output_abs),
        )
        if progress_cb is not None:
            progress_cb("nvvfx", 1.0, frames=int(processed), quality=str(quality_name), scale=int(scale))
        return str(output_abs)

    def _encode_and_upload_file_output(self, progress_cb: Any | None = None) -> dict[str, Any]:
        if progress_cb is not None:
            progress_cb("encode", 0.05, frames=int(self.frames))
        mp4_path = self._encode_file_output()
        if progress_cb is not None:
            progress_cb("encode", 1.0, frames=int(self.frames))
        if bool(self.file_native_pre_resize) and not bool(self._file_worker_finalizer_planned()):
            final_resize_path = os.path.join(
                os.path.dirname(str(mp4_path)) or ".",
                f"{os.path.splitext(os.path.basename(str(mp4_path)))[0]}.final.mp4",
            )
            mp4_path = self._resize_file_video_to_output(
                input_video=str(mp4_path),
                output_video=str(final_resize_path),
                fps=max(1, int(self.file_raw_video_fps or self.cfg.fps)),
                progress_cb=progress_cb,
            )
        elif bool(self.file_native_pre_resize) and bool(self._file_worker_finalizer_planned()):
            logging.warning(
                "Remote edge FILE final resize skipped: session=%s job=%s reason=worker_finalizer_planned native_size=%dx%d delivery_size=%dx%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(self.file_frame_width or 0),
                int(self.file_frame_height or 0),
                int(self.cfg.width or 0),
                int(self.cfg.height or 0),
            )
        remote_upscale_url = self._file_remote_upscale_url()
        if bool(self._file_worker_finalizer_planned()):
            # The worker will run the authoritative media finalizer after this
            # intermediate file is uploaded. Do not run the edge-side upscale/
            # RIFE pass too, or the finalizer will interpolate a second time and
            # stretch/cut the timeline.
            if remote_upscale_url:
                logging.warning(
                    "Remote edge FILE remote upscale skipped: session=%s job=%s reason=worker_finalizer_planned",
                    self.cfg.session_id,
                    self.cfg.job_id,
                )
            remote_upscale_url = ""
        if remote_upscale_url:
            remote_upscale_path = os.path.join(
                os.path.dirname(str(mp4_path)) or ".",
                f"{os.path.splitext(os.path.basename(str(mp4_path)))[0]}.upscaled.mp4",
            )
            try:
                mp4_path = self._apply_file_remote_upscale(
                    input_video=str(mp4_path),
                    output_video=str(remote_upscale_path),
                    progress_cb=progress_cb,
                )
            except Exception:
                if bool(_env_flag("REMOTE_EDGE_FILE_UPSCALE_FAIL_OPEN", "1")):
                    logging.exception(
                        "Remote edge FILE remote upscale failed open: session=%s job=%s url=%s input=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        str(remote_upscale_url),
                        str(mp4_path),
                    )
                else:
                    raise
        elif self._file_nvvfx_enabled():
            nvvfx_path = os.path.join(
                os.path.dirname(str(mp4_path)) or ".",
                f"{os.path.splitext(os.path.basename(str(mp4_path)))[0]}.nvvfx.mp4",
            )
            try:
                mp4_path = self._apply_file_nvvfx(
                    input_video=str(mp4_path),
                    output_video=str(nvvfx_path),
                    progress_cb=progress_cb,
                )
            except Exception:
                if bool(_env_flag("REMOTE_EDGE_FILE_NVVFX_FAIL_OPEN", "1")):
                    logging.exception(
                        "Remote edge FILE NVIDIA VFX failed open: session=%s job=%s input=%s",
                        self.cfg.session_id,
                        self.cfg.job_id,
                        str(mp4_path),
                    )
                else:
                    raise
        upload_url = str(self.cfg.file_upload_url or "").strip()
        remote_direct_uploaded = bool(getattr(self, "file_remote_upscale_direct_uploaded", False))
        remote_upscale_result = dict(getattr(self, "file_remote_upscale_result", {}) or {})
        if upload_url and not bool(remote_direct_uploaded):
            if progress_cb is not None:
                progress_cb("upload", 0.05, path=str(self.cfg.file_upload_path or ""))
            self._put_file_output(upload_url, mp4_path)
            if progress_cb is not None:
                progress_cb("upload", 1.0, path=str(self.cfg.file_upload_path or ""))
        elif upload_url and bool(remote_direct_uploaded) and progress_cb is not None:
            progress_cb(
                "upload",
                1.0,
                path=str(self.cfg.file_upload_path or ""),
                remote_direct=1,
                bytes=int(remote_upscale_result.get("bytes") or 0),
            )
        size = int(remote_upscale_result.get("bytes") or 0) if bool(remote_direct_uploaded) else (
            int(os.path.getsize(mp4_path)) if os.path.exists(mp4_path) else 0
        )
        output_frames = int(remote_upscale_result.get("frames") or (self.file_raw_video_frames or self.frames))
        output_fps = int(remote_upscale_result.get("output_fps") or (self.file_raw_video_fps or self.cfg.fps))
        result = {
            "session_id": str(self.cfg.session_id or ""),
            "job_id": str(self.cfg.job_id or ""),
            "uploaded": bool(upload_url),
            "remote_direct_upload": bool(remote_direct_uploaded),
            "path": str(self.cfg.file_upload_path or ""),
            "public_url": str(self.cfg.file_public_url or ""),
            "content_type": str(self.cfg.file_content_type or "video/mp4"),
            "bytes": int(size),
            "frames": int(self.frames),
            "output_frames": int(output_frames),
            "output_fps": int(output_fps),
            "duration_sec": float(self.frames) / float(max(1, int(self.cfg.fps))),
            "post_vae_inline_blocks": int(self.file_inline_post_vae_blocks),
            "post_vae_inline_elapsed_sec": float(self.file_inline_post_vae_elapsed_sec),
        }
        logging.warning(
            "Remote edge FILE output ready: session=%s job=%s uploaded=%d remote_direct=%d frames=%d output_frames=%d output_fps=%d post_vae_inline_blocks=%d bytes=%d path=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            1 if upload_url else 0,
            1 if bool(remote_direct_uploaded) else 0,
            int(self.frames),
            int(output_frames),
            int(output_fps),
            int(self.file_inline_post_vae_blocks),
            int(size),
            str(self.cfg.file_upload_path or "-"),
        )
        return result

    def _normalize_file_audio_for_mux(self, *, sample_rate: int, fps: int) -> tuple[int, int]:
        raw_audio = str(self.file_raw_audio_path)
        actual_audio_bytes = int(os.path.getsize(raw_audio)) if os.path.exists(raw_audio) else 0
        file_target_samples = int(max(0, int(self.cfg.file_target_audio_samples or 0)))
        if file_target_samples <= 0 and float(self.cfg.file_target_duration_sec or 0.0) > 0.0:
            file_target_samples = int(round(float(self.cfg.file_target_duration_sec) * float(sample_rate)))
        if int(file_target_samples) > 0:
            target_source_frames = int(
                max(1, (int(file_target_samples) * int(fps) + int(sample_rate) - 1) // int(sample_rate))
            )
            target_audio_bytes = int(file_target_samples) * 2
        else:
            target_source_frames = int(max(1, int(self.frames)))
            target_audio_bytes = int(round(float(target_source_frames) * float(sample_rate) / float(fps))) * 2
            if actual_audio_bytes > target_audio_bytes and target_audio_bytes > 0:
                samples = int(actual_audio_bytes // 2)
                target_source_frames = int(max(1, (int(samples) * int(fps) + int(sample_rate) - 1) // int(sample_rate)))
                target_audio_bytes = int(samples) * 2
        if actual_audio_bytes < target_audio_bytes:
            with open(raw_audio, "ab") as f:
                f.write(b"\x00" * int(target_audio_bytes - actual_audio_bytes))
        elif actual_audio_bytes > target_audio_bytes and target_audio_bytes > 0:
            with open(raw_audio, "rb+") as f:
                f.truncate(int(target_audio_bytes))
        self.frames = int(target_source_frames)
        return int(target_source_frames), int(target_audio_bytes)

    def _encode_streamed_file_output(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        raw_video_fps: int,
        sample_rate: int,
    ) -> str:
        if not bool(self.file_stream_encode_enabled):
            raise RuntimeError("file stream encode is not enabled")
        target_source_frames, _target_audio_bytes = self._normalize_file_audio_for_mux(
            sample_rate=int(sample_rate),
            fps=int(fps),
        )
        target_output_frames = int(
            max(
                1,
                round(float(target_source_frames) * float(max(1, int(raw_video_fps))) / float(max(1, int(fps)))),
            )
        )
        if int(self.file_raw_video_frames) < int(target_output_frames):
            last = self.file_stream_last_frame
            if last is None:
                raise RuntimeError("file stream encode cannot pad video without a last frame")
            missing = int(target_output_frames) - int(self.file_raw_video_frames)
            while missing > 0:
                chunk = min(60, int(missing))
                written = int(self._write_file_stream_frames([bytes(last)] * int(chunk)))
                if written <= 0:
                    break
                missing -= int(written)
        self._close_file_stream_encoder()
        if int(self.file_raw_video_frames) <= 0:
            raise RuntimeError("file stream encoder received no video frames")
        cmd = self._file_ffmpeg_mux_streamed_video_cmd(
            sample_rate=int(sample_rate),
            video_path=str(self.file_video_only_path),
            raw_audio=str(self.file_raw_audio_path),
            mp4_path=str(self.file_mp4_path),
        )
        duration_sec = float(target_source_frames) / float(max(1, int(fps)))
        timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_ENCODE_TIMEOUT_SEC", 60.0 + duration_sec * 4.0))
        self._run_file_subprocess(cmd, timeout=float(timeout), label="ffmpeg file mux")
        logging.warning(
            "Remote edge FILE stream encode done: session=%s job=%s frames=%d output_frames=%d output_fps=%d video_bytes=%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(target_source_frames),
            int(self.file_raw_video_frames),
            int(raw_video_fps),
            int(os.path.getsize(self.file_video_only_path)) if os.path.exists(self.file_video_only_path) else 0,
            str(self.file_mp4_path),
        )
        return str(self.file_mp4_path)

    def _encode_file_output(self) -> str:
        if not self.file_mp4_path:
            raise RuntimeError("file output is not started")
        width = max(1, int(self.file_frame_width or self.cfg.width))
        height = max(1, int(self.file_frame_height or self.cfg.height))
        fps = max(1, int(self.cfg.fps))
        sample_rate = max(1, int(self.cfg.sample_rate or 16000))
        frame_bytes = int(width * height * 3)
        raw_video = str(self.file_raw_video_path)
        raw_audio = str(self.file_raw_audio_path)
        if int(self.frames) <= 0:
            raise RuntimeError("remote edge file output has no video frames")
        self._finalize_file_inline_interpolation()
        raw_video_fps = max(1, int(self.file_raw_video_fps or fps))
        raw_video_frames = max(0, int(self.file_raw_video_frames or self.frames))
        if bool(self.file_stream_encode_enabled):
            return self._encode_streamed_file_output(
                width=int(width),
                height=int(height),
                fps=int(fps),
                raw_video_fps=int(raw_video_fps),
                sample_rate=int(sample_rate),
            )
        expected_video_bytes = int(frame_bytes * int(raw_video_frames))
        if not os.path.exists(raw_video) or os.path.getsize(raw_video) != expected_video_bytes:
            raise RuntimeError(
                f"file raw video size mismatch: got={0 if not os.path.exists(raw_video) else os.path.getsize(raw_video)} expected={expected_video_bytes}"
            )
        actual_audio_bytes = int(os.path.getsize(raw_audio)) if os.path.exists(raw_audio) else 0

        def set_source_frame_count(target_source_frames_raw: int) -> None:
            target_source_frames = int(max(1, int(target_source_frames_raw)))
            target_raw_frames = int(
                max(
                    1,
                    round(float(target_source_frames) * float(raw_video_fps) / float(max(1, int(fps)))),
                )
            )
            current_raw_frames = max(0, int(self.file_raw_video_frames or self.frames))
            if int(target_raw_frames) < int(current_raw_frames):
                with open(raw_video, "rb+") as f:
                    f.truncate(int(target_raw_frames) * int(frame_bytes))
            elif int(target_raw_frames) > int(current_raw_frames):
                with open(raw_video, "rb") as f:
                    f.seek(-frame_bytes, os.SEEK_END)
                    last_frame = f.read(frame_bytes)
                if len(last_frame) != frame_bytes:
                    raise RuntimeError("file raw video has no complete last frame")
                with open(raw_video, "ab") as f:
                    for _ in range(int(target_raw_frames) - int(current_raw_frames)):
                        f.write(last_frame)
            self.frames = int(target_source_frames)
            self.file_raw_video_frames = int(target_raw_frames)

        file_target_samples = int(max(0, int(self.cfg.file_target_audio_samples or 0)))
        if file_target_samples <= 0 and float(self.cfg.file_target_duration_sec or 0.0) > 0.0:
            file_target_samples = int(round(float(self.cfg.file_target_duration_sec) * float(sample_rate)))
        if file_target_samples > 0:
            frames_before = int(self.frames)
            raw_frames_before = int(self.file_raw_video_frames or self.frames)
            audio_before = int(actual_audio_bytes)
            target_frames = int(
                (int(file_target_samples) * int(fps) + int(sample_rate) - 1) // int(sample_rate)
            )
            target_frames = int(max(1, int(target_frames)))
            if int(target_frames) != int(self.frames):
                set_source_frame_count(int(target_frames))
            target_audio_bytes = int(file_target_samples) * 2
            if actual_audio_bytes < target_audio_bytes:
                with open(raw_audio, "ab") as f:
                    f.write(b"\x00" * int(target_audio_bytes - actual_audio_bytes))
                actual_audio_bytes = int(target_audio_bytes)
            elif actual_audio_bytes > target_audio_bytes:
                with open(raw_audio, "rb+") as f:
                    f.truncate(int(target_audio_bytes))
                actual_audio_bytes = int(target_audio_bytes)
            if int(frames_before) != int(self.frames) or int(audio_before) != int(actual_audio_bytes):
                logging.warning(
                    "Remote edge FILE output trimmed: session=%s job=%s frames=%d->%d raw_frames=%d->%d raw_fps=%d audio_bytes=%d->%d target_samples=%d target_duration=%.3fs",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(frames_before),
                    int(self.frames),
                    int(raw_frames_before),
                    int(self.file_raw_video_frames or self.frames),
                    int(raw_video_fps),
                    int(audio_before),
                    int(actual_audio_bytes),
                    int(file_target_samples),
                    float(file_target_samples) / float(max(1, int(sample_rate))),
                )
        else:
            target_audio_bytes = int(round(float(self.frames) * float(sample_rate) / float(fps))) * 2
            if actual_audio_bytes > target_audio_bytes and target_audio_bytes > 0:
                extra_samples = int((actual_audio_bytes - target_audio_bytes) // 2)
                extra_frames = int((extra_samples * fps + sample_rate - 1) // sample_rate)
                if extra_frames > 0:
                    set_source_frame_count(int(self.frames) + int(extra_frames))
                    target_audio_bytes = int(round(float(self.frames) * float(sample_rate) / float(fps))) * 2
            if actual_audio_bytes < target_audio_bytes:
                with open(raw_audio, "ab") as f:
                    f.write(b"\x00" * int(target_audio_bytes - actual_audio_bytes))
            elif actual_audio_bytes > target_audio_bytes and target_audio_bytes > 0:
                with open(raw_audio, "rb+") as f:
                    f.truncate(int(target_audio_bytes))
        output_fps = self._file_requested_output_fps(input_fps=int(fps))
        duration_sec = float(self.frames) / float(max(1, int(fps)))
        if bool(self.file_inline_interpolation_enabled):
            cmd = self._file_ffmpeg_encode_cmd(
                width=width,
                height=height,
                fps=int(raw_video_fps),
                output_fps=int(raw_video_fps),
                sample_rate=sample_rate,
                raw_video=raw_video,
                raw_audio=raw_audio,
                mp4_path=str(self.file_mp4_path),
                apply_video_filters=not bool(self.file_native_pre_resize),
            )
            timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_ENCODE_TIMEOUT_SEC", 60.0 + duration_sec * 8.0))
            self._run_file_subprocess(cmd, timeout=float(timeout), label="ffmpeg file encode")
        elif self._file_interpolation_enabled(input_fps=int(fps), output_fps=int(output_fps)):
            self._interpolate_file_output_rife(
                width=int(width),
                height=int(height),
                input_fps=int(fps),
                output_fps=int(output_fps),
                sample_rate=int(sample_rate),
                raw_video=str(raw_video),
                raw_audio=str(raw_audio),
                mp4_path=str(self.file_mp4_path),
                duration_sec=float(duration_sec),
            )
        else:
            cmd = self._file_ffmpeg_encode_cmd(
                width=width,
                height=height,
                fps=fps,
                output_fps=int(output_fps),
                sample_rate=sample_rate,
                raw_video=raw_video,
                raw_audio=raw_audio,
                mp4_path=str(self.file_mp4_path),
                apply_video_filters=not bool(self.file_native_pre_resize),
            )
            timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_ENCODE_TIMEOUT_SEC", 60.0 + duration_sec * 8.0))
            self._run_file_subprocess(cmd, timeout=float(timeout), label="ffmpeg file encode")
        if not os.path.exists(self.file_mp4_path) or os.path.getsize(self.file_mp4_path) <= 0:
            raise RuntimeError("ffmpeg file encode produced no output")
        return str(self.file_mp4_path)

    def _file_requested_output_fps(self, *, input_fps: int) -> int:
        return max(
            1,
            int(
                self.cfg.file_output_fps
                or _safe_int_env("REMOTE_EDGE_FILE_OUTPUT_FPS", _safe_int_env("LIVE_CHANNEL_RTMP_OUTPUT_FPS", max(1, int(input_fps))))
            ),
        )

    def _file_interpolation_enabled(self, *, input_fps: int, output_fps: int) -> bool:
        return bool(self._file_interpolation_backend(input_fps=int(input_fps), output_fps=int(output_fps)))

    def _file_interpolation_backend(self, *, input_fps: int, output_fps: int) -> str:
        if int(output_fps) <= int(input_fps):
            return ""
        raw = str(os.getenv("REMOTE_EDGE_FILE_INTERPOLATION", "") or "").strip().lower()
        if raw in {"torch-rife", "rife-torch", "torch", "pytorch", "inmemory", "in-memory"}:
            return "torch-rife"
        if raw in {"rife", "rife-ncnn", "rife-ncnn-vulkan", "2x", "x2"}:
            return "rife-ncnn"
        if _env_flag("REMOTE_EDGE_FILE_RIFE_TORCH_ENABLED", "0"):
            return "torch-rife"
        if _env_flag("REMOTE_EDGE_FILE_INTERPOLATION", "0") or _env_flag("REMOTE_EDGE_FILE_RIFE_ENABLED", "0"):
            return "rife-ncnn"
        return ""

    def _live_interpolation_enabled(self, *, input_fps: int, output_fps: int) -> bool:
        return bool(self._live_interpolation_backend(input_fps=int(input_fps), output_fps=int(output_fps)))

    def _live_interpolation_backend(self, *, input_fps: int, output_fps: int) -> str:
        if int(output_fps) <= int(input_fps):
            return ""
        raw = str(os.getenv("REMOTE_EDGE_LIVE_INTERPOLATION", "") or "").strip().lower()
        if raw in {"nvidia-fruc", "nvfruc", "nvof-fruc", "ofa-fruc", "fruc"}:
            return "nvidia-fruc"
        if raw in {"torch-rife", "rife-torch", "torch", "pytorch", "inmemory", "in-memory"}:
            return "torch-rife"
        if raw in {"rife", "rife-ncnn", "rife-ncnn-vulkan", "2x", "x2"}:
            return "rife-ncnn"
        if _env_flag("REMOTE_EDGE_LIVE_RIFE_TORCH_ENABLED", "0"):
            return "torch-rife"
        if _env_flag("REMOTE_EDGE_LIVE_INTERPOLATION", "0") or _env_flag("REMOTE_EDGE_LIVE_RIFE_ENABLED", "0"):
            return "rife-ncnn"
        return ""

    def _nvidia_fruc_binary_path(self) -> str:
        configured = str(os.getenv("REMOTE_EDGE_NVFRUC_BIN", "") or "").strip()
        if configured:
            return configured
        found = shutil.which("NvOFFRUCSample")
        if found:
            return str(found)
        found = shutil.which("NvFRUCSample")
        if found:
            return str(found)
        return ""

    def _validate_live_nvidia_fruc_config(self) -> None:
        binary = self._nvidia_fruc_binary_path()
        if not binary:
            raise RuntimeError(
                "REMOTE_EDGE_LIVE_INTERPOLATION=nvidia-fruc requires NVIDIA Optical Flow SDK "
                "NvOFFRUCSample/NvFRUCSample binary; set REMOTE_EDGE_NVFRUC_BIN"
            )
        if not os.path.exists(binary) and shutil.which(binary) is None:
            raise RuntimeError(f"NVIDIA FRUC binary not found: {binary}")

    def _rtmp_pipe_fps(self) -> int:
        value = getattr(self, "rtmp_pipe_fps", None)
        if value is not None:
            return max(1, int(value))
        output_fps = rtmp_output_fps(int(self.cfg.fps))
        if self._live_interpolation_backend(input_fps=int(self.cfg.fps), output_fps=int(output_fps)):
            return max(1, int(output_fps))
        return max(1, int(self.cfg.fps))

    def _rtmp_python_pacing_enabled(self) -> bool:
        # Keep one RTMP clock for both normal and interpolated live output.
        # Feeding interpolated blocks to ffmpeg in bursts lets its pipe queue
        # drift ahead of the real audio/video cadence.
        return True

    def _live_rife_pairwise_enabled(self) -> bool:
        value = getattr(self, "live_rife_pairwise_enabled", None)
        if value is not None:
            return bool(value)
        backend = str(getattr(self, "live_rife_backend", "") or "").strip().lower()
        return bool(backend == "torch-rife" and _env_flag("REMOTE_EDGE_LIVE_RIFE_PAIRWISE", "0"))

    @staticmethod
    def _split_pcm16le_for_frames(payload: bytes, frame_count: int) -> list[bytes]:
        frames = int(max(0, int(frame_count)))
        if frames <= 0:
            return []
        even = int((len(payload or b"") // 2) * 2)
        payload_b = bytes(payload or b"")[:even]
        total_samples = int(len(payload_b) // 2)
        parts: list[bytes] = []
        for idx in range(frames):
            start_samples = int((int(idx) * int(total_samples)) // int(frames))
            end_samples = int(((int(idx) + 1) * int(total_samples)) // int(frames))
            parts.append(payload_b[int(start_samples * 2): int(end_samples * 2)])
        return parts

    @staticmethod
    def _fit_live_rife_frame_count(frames: list[bytes], target_frames: int) -> list[bytes]:
        target = max(0, int(target_frames))
        source = [bytes(frame) for frame in frames]
        if target <= 0 or not source:
            return []
        if len(source) == target:
            return source
        if len(source) < target:
            return source + [source[-1]] * int(target - len(source))
        if target == 1:
            return [source[0]]
        last = int(len(source) - 1)
        return [source[min(last, int(round(float(idx) * float(last) / float(target - 1))))] for idx in range(target)]

    @classmethod
    def _live_rife_pair_output_frames(cls, source_pair: list[bytes], interpolated_frames: list[bytes]) -> tuple[bytes, bytes, bytes]:
        interpolated = [bytes(frame) for frame in interpolated_frames]
        if len(interpolated) >= 3:
            return bytes(interpolated[0]), bytes(interpolated[1]), bytes(interpolated[2])
        fallback = cls._fit_live_rife_frame_count([bytes(frame) for frame in source_pair], 3)
        if len(fallback) < 3:
            raise RuntimeError("live RIFE pairwise fallback produced fewer than 3 frames")
        return bytes(fallback[0]), bytes(fallback[1]), bytes(fallback[2])

    def _live_rife_skip_reason(self) -> str:
        if not bool(getattr(self, "live_rife_skip_on_backlog", False)):
            return ""
        pending_blocks = int(len(self.pending_blocks))
        buffered_frames = int(self.buffered_video_frames)
        rtmp_queue = int(self._rtmp_queue_len())
        latent_q = 0
        try:
            latent_q = int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0
        except Exception:
            latent_q = 0
        if int(self.live_rife_skip_pending_blocks) > 0 and pending_blocks >= int(self.live_rife_skip_pending_blocks):
            return f"pending_blocks={pending_blocks}"
        if int(self.live_rife_skip_buffered_frames) > 0 and buffered_frames >= int(self.live_rife_skip_buffered_frames):
            return f"buffered_frames={buffered_frames}"
        if int(self.live_rife_skip_latent_q) > 0 and latent_q >= int(self.live_rife_skip_latent_q):
            return f"latent_q={latent_q}"
        if int(self.live_rife_skip_rtmp_queue) > 0 and rtmp_queue >= int(self.live_rife_skip_rtmp_queue):
            return f"rtmp_queue={rtmp_queue}"
        last_elapsed = float(getattr(self, "live_rife_last_elapsed_sec", 0.0) or 0.0)
        if (
            float(self.live_rife_skip_after_elapsed_sec) > 0.0
            and last_elapsed >= float(self.live_rife_skip_after_elapsed_sec)
        ):
            return f"last_elapsed={last_elapsed:.3f}s"
        return ""

    def _skip_live_rife_frames(
        self,
        blocks: list[_PendingVideoBlock],
        *,
        input_fps: int,
        output_fps: int,
        reason: str,
    ) -> list[bytes]:
        frames = [frame for block in blocks for frame in self._block_frames_bytes(block)]
        source_count = int(len(frames))
        target_frames = max(
            int(source_count),
            int(round(float(source_count) * float(output_fps) / float(max(1, int(input_fps))))),
        )
        out_frames = self._fit_live_rife_frame_count(frames, int(target_frames))
        self.live_rife_skip_blocks += 1
        self.live_rife_skip_frames_in += int(source_count)
        self.live_rife_skip_frames_out += int(len(out_frames))
        if str(reason or "").startswith("last_elapsed="):
            # Cool down for one block after a slow RIFE call, then probe again.
            # Keeping the old elapsed value would permanently disable RIFE.
            self.live_rife_last_elapsed_sec = 0.0
        now = time.monotonic()
        if now - float(self.live_rife_skip_last_log) >= 5.0:
            logging.warning(
                "Remote edge LIVE RIFE skipped: session=%s job=%s reason=%s skip_blocks=%d frames=%d->%d block=%d->%d pending_blocks=%d buffered_frames=%d latent_q=%d rtmp_queue=%d last_elapsed=%.3fs",
                self.cfg.session_id,
                self.cfg.job_id,
                str(reason or "backlog"),
                int(self.live_rife_skip_blocks),
                int(self.live_rife_skip_frames_in),
                int(self.live_rife_skip_frames_out),
                int(source_count),
                int(len(out_frames)),
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
                int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0,
                int(self._rtmp_queue_len()),
                float(getattr(self, "live_rife_last_elapsed_sec", 0.0) or 0.0),
            )
            self.live_rife_skip_last_log = float(now)
        return out_frames

    def _interpolate_live_blocks_rife(
        self,
        blocks: list[_PendingVideoBlock],
        *,
        input_fps: int,
        output_fps: int,
    ) -> list[bytes]:
        skip_reason = self._live_rife_skip_reason()
        if skip_reason:
            return self._skip_live_rife_frames(
                list(blocks),
                input_fps=int(input_fps),
                output_fps=int(output_fps),
                reason=str(skip_reason),
            )
        backend = str(getattr(self, "live_rife_backend", "") or "").strip().lower()
        frames_tensor = self._concat_block_tensors(list(blocks))
        if frames_tensor is not None:
            if backend == "torch-rife":
                return self._interpolate_live_tensor_rife(
                    frames_tensor,
                    input_fps=int(input_fps),
                    output_fps=int(output_fps),
                )
            if backend == "nvidia-fruc" and bool(self.live_rife_pre_resize):
                return self._interpolate_live_tensor_nvidia_fruc(
                    frames_tensor,
                    input_fps=int(input_fps),
                    output_fps=int(output_fps),
                )
        if bool(getattr(self, "live_rife_require_tensor", False)):
            return self._skip_live_rife_frames(
                list(blocks),
                input_fps=int(input_fps),
                output_fps=int(output_fps),
                reason="missing_tensor",
            )
        frames = [frame for block in blocks for frame in self._block_frames_bytes(block)]
        return self._interpolate_live_frames_rife(
            frames,
            input_fps=int(input_fps),
            output_fps=int(output_fps),
        )

    def _interpolate_live_tensor_rife(
        self,
        frames_tensor: Any,
        *,
        input_fps: int,
        output_fps: int,
    ) -> list[bytes]:
        source_frame_count = int(self._tensor_frame_count(frames_tensor))
        if source_frame_count <= 0:
            return []
        if source_frame_count < 2 or int(output_fps) <= int(input_fps):
            return self._tensor_01_to_output_rgb24_frames(frames_tensor)
        target_frames = max(
            int(source_frame_count),
            int(round(float(source_frame_count) * float(output_fps) / float(max(1, int(input_fps))))),
        )
        return self._interpolate_tensor_frames_torch_rife(
            frames_tensor,
            target_frames=int(target_frames),
            label="LIVE",
        )

    def _interpolate_live_tensor_nvidia_fruc(
        self,
        frames_tensor: Any,
        *,
        input_fps: int,
        output_fps: int,
    ) -> list[bytes]:
        source_frame_count = int(self._tensor_frame_count(frames_tensor))
        if source_frame_count <= 0:
            return []
        if source_frame_count < 2 or int(output_fps) <= int(input_fps):
            return self._tensor_01_to_output_rgb24_frames(frames_tensor)
        source_h, source_w = self._tensor_hw(frames_tensor)
        if int(source_h) <= 0 or int(source_w) <= 0:
            return self._tensor_01_to_output_rgb24_frames(frames_tensor)
        target_frames = max(
            int(source_frame_count),
            int(round(float(source_frame_count) * float(output_fps) / float(max(1, int(input_fps))))),
        )
        return self._interpolate_frames_nvidia_fruc(
            self._tensor_01_to_rgb24_frames(frames_tensor),
            input_fps=int(input_fps),
            output_fps=int(output_fps),
            target_frames=int(target_frames),
            width=int(source_w),
            height=int(source_h),
            output_width=int(self.cfg.width),
            output_height=int(self.cfg.height),
        )

    def _interpolate_live_frames_rife(
        self,
        frames: list[bytes],
        *,
        input_fps: int,
        output_fps: int,
    ) -> list[bytes]:
        source_frames = [bytes(frame) for frame in frames]
        if len(source_frames) < 2 or int(output_fps) <= int(input_fps):
            return source_frames
        expected = int(self.cfg.width * self.cfg.height * 3)
        for frame in source_frames:
            if len(frame) != expected:
                raise ValueError(f"invalid rgb24 frame size for live RIFE: got={len(frame)} expected={expected}")
        target_frames = max(
            len(source_frames),
            int(round(float(len(source_frames)) * float(output_fps) / float(max(1, int(input_fps))))),
        )
        backend = str(getattr(self, "live_rife_backend", "") or "").strip().lower()
        if backend == "torch-rife":
            return self._interpolate_frames_torch_rife(
                source_frames,
                width=int(self.cfg.width),
                height=int(self.cfg.height),
                target_frames=int(target_frames),
                label="LIVE",
            )
        if backend == "nvidia-fruc":
            return self._interpolate_frames_nvidia_fruc(
                source_frames,
                input_fps=int(input_fps),
                output_fps=int(output_fps),
                target_frames=int(target_frames),
            )
        start = time.perf_counter()
        base_dir = str(os.getenv("REMOTE_EDGE_LIVE_RIFE_WORK_DIR", "/tmp/smartblog-live-rife") or "/tmp/smartblog-live-rife")
        os.makedirs(base_dir, exist_ok=True)
        prefix = f"{_sanitize_clip_component(self.cfg.session_id, fallback='session')}_{_sanitize_clip_component(self.cfg.job_id, fallback='job')}_"
        with tempfile.TemporaryDirectory(prefix=prefix, dir=base_dir) as work_dir:
            raw_video = os.path.join(work_dir, "input.rgb")
            input_dir = os.path.join(work_dir, "input")
            output_dir = os.path.join(work_dir, "output")
            output_raw = os.path.join(work_dir, "output.rgb")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            with open(raw_video, "wb") as f:
                for frame in source_frames:
                    f.write(frame)
            input_pattern = os.path.join(input_dir, "%08d.png")
            output_pattern = os.path.join(output_dir, "%08d.png")
            decode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{int(self.cfg.width)}x{int(self.cfg.height)}",
                "-r",
                str(int(input_fps)),
                "-i",
                raw_video,
                "-frames:v",
                str(int(len(source_frames))),
                input_pattern,
            ]
            duration_sec = float(len(source_frames)) / float(max(1, int(input_fps)))
            timeout = max(5.0, _safe_float_env("REMOTE_EDGE_LIVE_RIFE_TIMEOUT_SEC", 8.0 + float(duration_sec) * 20.0))
            self._run_file_subprocess(decode_cmd, timeout=float(timeout), label="ffmpeg live RIFE frame decode")
            rife_cmd = [
                self._rife_binary_path(),
                "-i",
                input_dir,
                "-o",
                output_dir,
                "-n",
                str(int(target_frames)),
            ]
            model_path = str(os.getenv("REMOTE_EDGE_RIFE_MODEL", "") or "").strip()
            if model_path:
                rife_cmd.extend(["-m", model_path])
            gpu_id = str(os.getenv("REMOTE_EDGE_RIFE_GPU_ID", "") or "").strip()
            if gpu_id:
                rife_cmd.extend(["-g", gpu_id])
            threads = str(os.getenv("REMOTE_EDGE_RIFE_THREADS", "") or "").strip()
            if threads:
                rife_cmd.extend(["-j", threads])
            pattern = str(os.getenv("REMOTE_EDGE_RIFE_OUTPUT_PATTERN", "") or "").strip()
            if pattern:
                rife_cmd.extend(["-f", pattern])
            self._run_file_subprocess(rife_cmd, timeout=float(timeout), label="rife-ncnn-vulkan live interpolation")
            actual_frames = len([name for name in os.listdir(output_dir) if name.lower().endswith((".png", ".jpg", ".webp"))])
            if actual_frames <= 0:
                raise RuntimeError("rife-ncnn-vulkan live interpolation produced no output frames")
            encode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-framerate",
                str(int(output_fps)),
                "-i",
                output_pattern,
                "-frames:v",
                str(int(actual_frames)),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                output_raw,
            ]
            self._run_file_subprocess(encode_cmd, timeout=float(timeout), label="ffmpeg live RIFE raw encode")
            with open(output_raw, "rb") as f:
                data = f.read()
        frame_bytes = int(expected)
        out_frames = [
            bytes(data[offset: offset + frame_bytes])
            for offset in range(0, len(data), frame_bytes)
            if len(data[offset: offset + frame_bytes]) == frame_bytes
        ]
        if not out_frames:
            raise RuntimeError("live RIFE output raw video contains no complete frames")
        self.live_rife_blocks += 1
        self.live_rife_frames_in += int(len(source_frames))
        self.live_rife_frames_out += int(len(out_frames))
        self.live_rife_last_elapsed_sec = max(0.0, time.perf_counter() - start) if "start" in locals() else 0.0
        now = time.monotonic()
        if now - float(self.live_rife_last_log) >= 10.0:
            logging.warning(
                "Remote edge LIVE RIFE interpolation: session=%s job=%s blocks=%d frames=%d->%d block=%d->%d fps=%d->%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(self.live_rife_blocks),
                int(self.live_rife_frames_in),
                int(self.live_rife_frames_out),
                int(len(source_frames)),
                int(len(out_frames)),
                int(input_fps),
                int(output_fps),
            )
            self.live_rife_last_log = float(now)
        return out_frames

    def _interpolate_frames_nvidia_fruc(
        self,
        frames: list[bytes],
        *,
        input_fps: int,
        output_fps: int,
        target_frames: int,
        width: int | None = None,
        height: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> list[bytes]:
        source_frames = [bytes(frame) for frame in frames]
        if len(source_frames) < 2 or int(output_fps) <= int(input_fps):
            return source_frames
        width_i = max(1, int(width if width is not None else self.cfg.width))
        height_i = max(1, int(height if height is not None else self.cfg.height))
        output_width_i = max(1, int(output_width if output_width is not None else self.cfg.width))
        output_height_i = max(1, int(output_height if output_height is not None else self.cfg.height))
        expected = int(width_i * height_i * 3)
        for frame in source_frames:
            if len(frame) != expected:
                raise ValueError(f"invalid rgb24 frame size for NVIDIA FRUC: got={len(frame)} expected={expected}")
        binary = self._nvidia_fruc_binary_path()
        if not binary:
            raise RuntimeError("NVIDIA FRUC binary is not configured; set REMOTE_EDGE_NVFRUC_BIN")
        start = time.perf_counter()
        base_dir = str(os.getenv("REMOTE_EDGE_NVFRUC_WORK_DIR", os.getenv("REMOTE_EDGE_LIVE_RIFE_WORK_DIR", "/tmp/smartblog-live-fruc")) or "/tmp/smartblog-live-fruc")
        os.makedirs(base_dir, exist_ok=True)
        prefix = f"{_sanitize_clip_component(self.cfg.session_id, fallback='session')}_{_sanitize_clip_component(self.cfg.job_id, fallback='job')}_"
        yuv_frame_bytes = int(width_i * height_i * 3 // 2)
        timeout = max(
            5.0,
            _safe_float_env(
                "REMOTE_EDGE_NVFRUC_TIMEOUT_SEC",
                8.0 + (float(len(source_frames)) / float(max(1, int(input_fps)))) * 20.0,
            ),
        )
        with tempfile.TemporaryDirectory(prefix=prefix, dir=base_dir) as work_dir:
            raw_rgb = os.path.join(work_dir, "input.rgb")
            input_yuv = os.path.join(work_dir, "input.yuv")
            output_dir = os.path.join(work_dir, "output")
            output_rgb = os.path.join(work_dir, "output.rgb")
            os.makedirs(output_dir, exist_ok=True)
            with open(raw_rgb, "wb") as f:
                for frame in source_frames:
                    f.write(frame)
            to_yuv_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width_i}x{height_i}",
                "-r",
                str(int(input_fps)),
                "-i",
                raw_rgb,
                "-frames:v",
                str(int(len(source_frames))),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                input_yuv,
            ]
            self._run_file_subprocess(to_yuv_cmd, timeout=float(timeout), label="ffmpeg NVIDIA FRUC rgb-to-yuv")
            fruc_cmd = [
                binary,
                f"--input={input_yuv}",
                f"--width={width_i}",
                f"--height={height_i}",
                f"--output={output_dir}",
                "--surfaceformat=0",
                "--allocationtype=0",
                "--cudasurfacetype=0",
            ]
            extra = str(os.getenv("REMOTE_EDGE_NVFRUC_EXTRA_ARGS", "") or "").strip()
            if extra:
                fruc_cmd.extend(shlex.split(extra))
            fruc_env = None
            fruc_visible = str(os.getenv("REMOTE_EDGE_NVFRUC_CUDA_VISIBLE_DEVICES", "") or "").strip()
            if fruc_visible:
                fruc_env = dict(os.environ)
                fruc_env["CUDA_VISIBLE_DEVICES"] = fruc_visible
            self._run_file_subprocess(
                fruc_cmd,
                timeout=float(timeout),
                label="NVIDIA FRUC live interpolation",
                env=fruc_env,
            )
            candidates = [
                os.path.join(output_dir, name)
                for name in os.listdir(output_dir)
                if name.lower().endswith(".yuv")
            ]
            if not candidates:
                raise RuntimeError("NVIDIA FRUC produced no YUV output")
            output_yuv = max(candidates, key=lambda path: os.path.getsize(path))
            actual_frames = int(os.path.getsize(output_yuv) // max(1, int(yuv_frame_bytes)))
            if actual_frames <= 0:
                raise RuntimeError("NVIDIA FRUC output contains no complete YUV420 frames")
            to_rgb_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width_i}x{height_i}",
                "-r",
                str(int(output_fps)),
                "-i",
                output_yuv,
            ]
            if int(output_width_i) != int(width_i) or int(output_height_i) != int(height_i):
                scale_flags = str(os.getenv("LIVE_CHANNEL_RTMP_SCALE_FLAGS", "bicubic") or "bicubic").strip()
                to_rgb_cmd.extend([
                    "-vf",
                    f"scale={int(output_width_i)}:{int(output_height_i)}:flags={scale_flags}",
                ])
            to_rgb_cmd.extend([
                "-frames:v",
                str(int(actual_frames)),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                output_rgb,
            ])
            self._run_file_subprocess(to_rgb_cmd, timeout=float(timeout), label="ffmpeg NVIDIA FRUC yuv-to-rgb")
            with open(output_rgb, "rb") as f:
                data = f.read()
        output_expected = int(output_width_i * output_height_i * 3)
        out_frames = [
            bytes(data[offset: offset + output_expected])
            for offset in range(0, len(data), output_expected)
            if len(data[offset: offset + output_expected]) == output_expected
        ]
        out_frames = self._fit_live_rife_frame_count(out_frames, int(target_frames))
        if not out_frames:
            raise RuntimeError("NVIDIA FRUC output raw video contains no complete RGB frames")
        elapsed = max(0.0, time.perf_counter() - start)
        self.live_rife_last_elapsed_sec = float(elapsed)
        self.live_rife_blocks += 1
        self.live_rife_frames_in += int(len(source_frames))
        self.live_rife_frames_out += int(len(out_frames))
        now = time.monotonic()
        if now - float(self.live_rife_last_log) >= 10.0:
            logging.warning(
                "Remote edge LIVE NVIDIA FRUC interpolation: session=%s job=%s binary=%s blocks=%d frames=%d->%d block=%d->%d fps=%d->%d size=%dx%d->%dx%d elapsed=%.3fs",
                self.cfg.session_id,
                self.cfg.job_id,
                str(binary),
                int(self.live_rife_blocks),
                int(self.live_rife_frames_in),
                int(self.live_rife_frames_out),
                int(len(source_frames)),
                int(len(out_frames)),
                int(input_fps),
                int(output_fps),
                int(width_i),
                int(height_i),
                int(output_width_i),
                int(output_height_i),
                float(elapsed),
            )
            self.live_rife_last_log = float(now)
        return out_frames

    def _interpolate_tensor_frames_torch_rife(
        self,
        frames_tensor: Any,
        *,
        target_frames: int,
        label: str,
    ) -> list[bytes]:
        from avalife.remote.torch_rife import get_shared_torch_rife_interpolator, tensor_01_to_rgb24_frames

        source_frame_count = int(self._tensor_frame_count(frames_tensor))
        source_h, source_w = self._tensor_hw(frames_tensor)
        if source_frame_count < 2:
            return self._tensor_01_to_output_rgb24_frames(frames_tensor)
        model_dir = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
        weights_path = str(
            os.getenv("REMOTE_EDGE_TORCH_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors"))
            or os.path.join(model_dir, "flownet.safetensors")
        )
        device = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DEVICE", "cuda:0") or "cuda:0")
        dtype_name = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DTYPE", "float16") or "float16")
        batch_pairs = max(1, _safe_int_env("REMOTE_EDGE_TORCH_RIFE_BATCH_PAIRS", 4))
        start = time.perf_counter()
        interpolator = get_shared_torch_rife_interpolator(
            model_dir=model_dir,
            weights_path=weights_path,
            device=device,
            dtype_name=dtype_name,
            batch_pairs=int(batch_pairs),
        )
        out_tensor = interpolator.interpolate_tensor_x2(
            frames_tensor,
            target_frames=int(target_frames),
        )
        out_h, out_w = self._tensor_hw(out_tensor)
        out_tensor = self._resize_tensor_01_to_output(out_tensor)
        resized_h, resized_w = self._tensor_hw(out_tensor)
        out_frames = tensor_01_to_rgb24_frames(out_tensor)
        elapsed = max(0.0, time.perf_counter() - start)
        self.live_rife_last_elapsed_sec = float(elapsed)
        if not out_frames:
            raise RuntimeError(f"torch-rife {str(label).lower()} interpolation produced no output frames")
        label_s = str(label or "RIFE").strip().upper()
        if label_s == "LIVE":
            self.live_rife_blocks += 1
            self.live_rife_frames_in += int(source_frame_count)
            self.live_rife_frames_out += int(len(out_frames))
            now = time.monotonic()
            if now - float(self.live_rife_last_log) >= 10.0:
                logging.warning(
                    "Remote edge LIVE RIFE interpolation: session=%s job=%s backend=torch-rife-gpu blocks=%d frames=%d->%d block=%d->%d fps=%d->%d size=%dx%d->%dx%d->%dx%d elapsed=%.3fs batch_pairs=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.live_rife_blocks),
                    int(self.live_rife_frames_in),
                    int(self.live_rife_frames_out),
                    int(source_frame_count),
                    int(len(out_frames)),
                    int(self.cfg.fps),
                    int(self._rtmp_pipe_fps()),
                    int(source_w),
                    int(source_h),
                    int(out_w),
                    int(out_h),
                    int(resized_w),
                    int(resized_h),
                    float(elapsed),
                    int(batch_pairs),
                )
                self.live_rife_last_log = float(now)
        return out_frames

    def _interpolate_tensor_frames_torch_rife_tensor(
        self,
        frames_tensor: Any,
        *,
        target_frames: int,
        label: str,
    ) -> Any:
        from avalife.remote.torch_rife import get_shared_torch_rife_interpolator

        source_frame_count = int(self._tensor_frame_count(frames_tensor))
        if source_frame_count < 2:
            return self._wait_ready_tensor(frames_tensor)
        model_dir = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
        weights_path = str(
            os.getenv("REMOTE_EDGE_TORCH_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors"))
            or os.path.join(model_dir, "flownet.safetensors")
        )
        device = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DEVICE", "cuda:0") or "cuda:0")
        dtype_name = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DTYPE", "float16") or "float16")
        batch_pairs = max(1, _safe_int_env("REMOTE_EDGE_TORCH_RIFE_BATCH_PAIRS", 4))
        start = time.perf_counter()
        interpolator = get_shared_torch_rife_interpolator(
            model_dir=model_dir,
            weights_path=weights_path,
            device=device,
            dtype_name=dtype_name,
            batch_pairs=int(batch_pairs),
        )
        out_tensor = interpolator.interpolate_tensor_x2(
            frames_tensor,
            target_frames=int(target_frames),
        )
        elapsed = max(0.0, time.perf_counter() - start)
        label_s = str(label or "RIFE").strip().upper()
        if label_s == "FILE":
            self.file_rife_blocks += 1
            self.file_rife_frames_in += int(source_frame_count)
            self.file_rife_frames_out += int(self._tensor_frame_count(out_tensor))
            now = time.monotonic()
            if now - float(self.file_rife_last_log) >= 10.0:
                logging.warning(
                    "Remote edge FILE RIFE interpolation tensor: session=%s job=%s backend=torch-rife blocks=%d frames=%d->%d block=%d->%d elapsed=%.3fs batch_pairs=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.file_rife_blocks),
                    int(self.file_rife_frames_in),
                    int(self.file_rife_frames_out),
                    int(source_frame_count),
                    int(self._tensor_frame_count(out_tensor)),
                    float(elapsed),
                    int(batch_pairs),
                )
                self.file_rife_last_log = float(now)
        return out_tensor

    def _interpolate_frames_torch_rife(
        self,
        frames: list[bytes],
        *,
        width: int,
        height: int,
        target_frames: int,
        label: str,
    ) -> list[bytes]:
        from avalife.remote.torch_rife import get_shared_torch_rife_interpolator

        source_frames = [bytes(frame) for frame in frames]
        if len(source_frames) < 2:
            return source_frames
        model_dir = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
        weights_path = str(
            os.getenv("REMOTE_EDGE_TORCH_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors"))
            or os.path.join(model_dir, "flownet.safetensors")
        )
        device = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DEVICE", "cuda:0") or "cuda:0")
        dtype_name = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DTYPE", "float16") or "float16")
        batch_pairs = max(1, _safe_int_env("REMOTE_EDGE_TORCH_RIFE_BATCH_PAIRS", 4))
        start = time.perf_counter()
        interpolator = get_shared_torch_rife_interpolator(
            model_dir=model_dir,
            weights_path=weights_path,
            device=device,
            dtype_name=dtype_name,
            batch_pairs=int(batch_pairs),
        )
        out_frames = interpolator.interpolate_x2(
            source_frames,
            width=int(width),
            height=int(height),
            target_frames=int(target_frames),
        )
        elapsed = max(0.0, time.perf_counter() - start)
        if str(label or "RIFE").strip().upper() != "FILE":
            self.live_rife_last_elapsed_sec = float(elapsed)
        if not out_frames:
            raise RuntimeError(f"torch-rife {str(label).lower()} interpolation produced no output frames")
        label_s = str(label or "RIFE").strip().upper()
        if label_s == "FILE":
            self.file_rife_blocks += 1
            self.file_rife_frames_in += int(len(source_frames))
            self.file_rife_frames_out += int(len(out_frames))
            now = time.monotonic()
            if now - float(self.file_rife_last_log) >= 10.0:
                logging.warning(
                    "Remote edge FILE RIFE interpolation: session=%s job=%s backend=torch-rife blocks=%d frames=%d->%d block=%d->%d elapsed=%.3fs batch_pairs=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.file_rife_blocks),
                    int(self.file_rife_frames_in),
                    int(self.file_rife_frames_out),
                    int(len(source_frames)),
                    int(len(out_frames)),
                    float(elapsed),
                    int(batch_pairs),
                )
                self.file_rife_last_log = float(now)
        else:
            self.live_rife_blocks += 1
            self.live_rife_frames_in += int(len(source_frames))
            self.live_rife_frames_out += int(len(out_frames))
            now = time.monotonic()
            if now - float(self.live_rife_last_log) >= 10.0:
                logging.warning(
                    "Remote edge LIVE RIFE interpolation: session=%s job=%s backend=torch-rife blocks=%d frames=%d->%d block=%d->%d fps=%d->%d elapsed=%.3fs batch_pairs=%d",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    int(self.live_rife_blocks),
                    int(self.live_rife_frames_in),
                    int(self.live_rife_frames_out),
                    int(len(source_frames)),
                    int(len(out_frames)),
                    int(self.cfg.fps),
                    int(self._rtmp_pipe_fps()),
                    float(elapsed),
                    int(batch_pairs),
                )
                self.live_rife_last_log = float(now)
        return out_frames

    def _run_file_subprocess(
        self,
        cmd: list[str],
        *,
        timeout: float,
        label: str,
        env: dict[str, str] | None = None,
    ) -> None:
        proc = subprocess.run(
            list(cmd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout),
            check=False,
            env=env,
        )
        if int(proc.returncode) != 0:
            tail = " ".join(str(proc.stderr or "").strip().splitlines()[-8:])
            raise RuntimeError(f"{label} failed rc={proc.returncode}: {tail or 'no stderr'}")

    def _rife_binary_path(self) -> str:
        configured = str(os.getenv("REMOTE_EDGE_RIFE_BIN", "") or "").strip()
        if configured:
            return configured
        found = shutil.which("rife-ncnn-vulkan")
        if found:
            return str(found)
        for candidate in (
            "/usr/local/bin/rife-ncnn-vulkan",
            "/opt/rife-ncnn-vulkan/rife-ncnn-vulkan",
            "/opt/rife-ncnn-vulkan/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan",
        ):
            if os.path.exists(candidate):
                return str(candidate)
        raise RuntimeError(
            "REMOTE_EDGE_FILE_INTERPOLATION is enabled but rife-ncnn-vulkan was not found; "
            "set REMOTE_EDGE_RIFE_BIN or install rife-ncnn-vulkan"
        )

    def _interpolate_file_output_rife(
        self,
        *,
        width: int,
        height: int,
        input_fps: int,
        output_fps: int,
        sample_rate: int,
        raw_video: str,
        raw_audio: str,
        mp4_path: str,
        duration_sec: float,
    ) -> None:
        work_dir = str(self.file_work_dir or "").strip()
        if not work_dir:
            raise RuntimeError("file interpolation requires file work dir")
        input_dir = os.path.join(work_dir, "rife_input")
        output_dir = os.path.join(work_dir, "rife_output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_pattern = os.path.join(input_dir, "%08d.png")
        output_pattern = os.path.join(output_dir, "%08d.png")
        target_frames = max(1, int(round(float(self.frames) * float(output_fps) / float(max(1, int(input_fps))))))
        logging.warning(
            "Remote edge FILE RIFE interpolation start: session=%s job=%s frames=%d fps=%d->%d target_frames=%d size=%dx%d",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.frames),
            int(input_fps),
            int(output_fps),
            int(target_frames),
            int(width),
            int(height),
        )
        decode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(input_fps)),
            "-i",
            raw_video,
            "-frames:v",
            str(int(self.frames)),
            input_pattern,
        ]
        timeout = max(60.0, _safe_float_env("REMOTE_EDGE_FILE_RIFE_TIMEOUT_SEC", 180.0 + float(duration_sec) * 40.0))
        self._run_file_subprocess(decode_cmd, timeout=float(timeout), label="ffmpeg RIFE frame decode")

        rife_cmd = [
            self._rife_binary_path(),
            "-i",
            input_dir,
            "-o",
            output_dir,
            "-n",
            str(int(target_frames)),
        ]
        model_path = str(os.getenv("REMOTE_EDGE_RIFE_MODEL", "") or "").strip()
        if model_path:
            rife_cmd.extend(["-m", model_path])
        gpu_id = str(os.getenv("REMOTE_EDGE_RIFE_GPU_ID", "") or "").strip()
        if gpu_id:
            rife_cmd.extend(["-g", gpu_id])
        threads = str(os.getenv("REMOTE_EDGE_RIFE_THREADS", "") or "").strip()
        if threads:
            rife_cmd.extend(["-j", threads])
        pattern = str(os.getenv("REMOTE_EDGE_RIFE_OUTPUT_PATTERN", "") or "").strip()
        if pattern:
            rife_cmd.extend(["-f", pattern])
        self._run_file_subprocess(rife_cmd, timeout=float(timeout), label="rife-ncnn-vulkan interpolation")

        actual_frames = len([name for name in os.listdir(output_dir) if name.lower().endswith((".png", ".jpg", ".webp"))])
        if int(actual_frames) <= 0:
            raise RuntimeError("rife-ncnn-vulkan produced no output frames")
        if int(actual_frames) != int(target_frames):
            logging.warning(
                "Remote edge FILE RIFE frame count differs: session=%s job=%s actual=%d target=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(actual_frames),
                int(target_frames),
            )
        encode_cmd = self._file_ffmpeg_encode_frames_cmd(
            output_fps=int(output_fps),
            sample_rate=int(sample_rate),
            frame_pattern=str(output_pattern),
            raw_audio=str(raw_audio),
            mp4_path=str(mp4_path),
        )
        self._run_file_subprocess(encode_cmd, timeout=float(timeout), label="ffmpeg RIFE output encode")
        logging.warning(
            "Remote edge FILE RIFE interpolation done: session=%s job=%s frames=%d->%d fps=%d->%d output=%s",
            self.cfg.session_id,
            self.cfg.job_id,
            int(self.frames),
            int(actual_frames),
            int(input_fps),
            int(output_fps),
            str(mp4_path),
        )

    def _file_ffmpeg_mux_streamed_video_cmd(
        self,
        *,
        sample_rate: int,
        video_path: str,
        raw_audio: str,
        mp4_path: str,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(video_path),
            "-f",
            "s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-i",
            str(raw_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *self._file_audio_encode_args(),
            "-shortest",
            str(mp4_path),
        ]

    def _file_ffmpeg_encode_cmd(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        output_fps: int | None = None,
        sample_rate: int,
        raw_video: str,
        raw_audio: str,
        mp4_path: str,
        apply_video_filters: bool = True,
    ) -> list[str]:
        output_fps_i = max(1, int(output_fps if output_fps is not None else self._file_requested_output_fps(input_fps=int(fps))))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(fps)),
            "-i",
            raw_video,
            "-f",
            "s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-i",
            raw_audio,
        ]
        video_filters = self._file_video_filters(width=int(width), height=int(height)) if bool(apply_video_filters) else []
        if int(output_fps_i) != int(fps):
            video_filters.insert(0, f"fps={int(output_fps_i)}")
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
        cmd.extend(["-r", str(int(output_fps_i))])
        cmd.extend(self._file_video_encode_args(output_fps=int(output_fps_i)))
        cmd.extend(self._file_audio_encode_args())
        cmd.append(mp4_path)
        return cmd

    def _file_ffmpeg_encode_frames_cmd(
        self,
        *,
        output_fps: int,
        sample_rate: int,
        frame_pattern: str,
        raw_audio: str,
        mp4_path: str,
    ) -> list[str]:
        output_fps_i = max(1, int(output_fps))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-framerate",
            str(int(output_fps_i)),
            "-i",
            frame_pattern,
            "-f",
            "s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-i",
            raw_audio,
        ]
        video_filters = self._file_video_filters()
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
        cmd.extend(["-r", str(int(output_fps_i))])
        cmd.extend(self._file_video_encode_args(output_fps=int(output_fps_i)))
        cmd.extend(self._file_audio_encode_args())
        cmd.append(mp4_path)
        return cmd

    def _file_video_filters(self, *, width: int | None = None, height: int | None = None) -> list[str]:
        filters: list[str] = []
        raw = str(os.getenv("REMOTE_EDGE_FILE_UNSHARP", "") or "").strip()
        if raw and raw.lower() not in {"0", "false", "no", "off", "none"}:
            if raw.lower() in {"1", "true", "yes", "on"}:
                raw = "3:3:0.30:3:3:0.0"
            filters.append(raw if raw.startswith("unsharp=") else f"unsharp={raw}")
        watermark_filter = watermark_drawtext_filter(
            text_file=self._watermark_text_file(),
            width=int(width if width is not None else self.cfg.width),
            height=int(height if height is not None else self.cfg.height),
            env_prefixes=("REMOTE_EDGE", "SMARTBLOG"),
        )
        if watermark_filter:
            filters.append(str(watermark_filter))
        watermark_chars = len(normalize_watermark_text(getattr(self.cfg, "watermark_text", "")))
        if not bool(getattr(self, "file_watermark_filter_logged", False)):
            self.file_watermark_filter_logged = True
            logging.warning(
                "Remote edge FILE watermark filter: session=%s job=%s chars=%d enabled=%d text_file=%s",
                self.cfg.session_id,
                self.cfg.job_id,
                int(watermark_chars),
                1 if bool(watermark_filter) else 0,
                os.path.basename(str(self.watermark_text_file or "")) if str(self.watermark_text_file or "") else "-",
            )
        return filters

    def _file_video_encode_args(self, *, output_fps: int) -> list[str]:
        gop = max(1, int(round(float(output_fps) * _safe_float_env("REMOTE_EDGE_FILE_KEYFRAME_SEC", 2.0))))
        encoder = str(
            os.getenv("REMOTE_EDGE_FILE_VIDEO_ENCODER", os.getenv("REMOTE_EDGE_CLIP_VIDEO_ENCODER", os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "h264_nvenc")))
            or "h264_nvenc"
        ).strip().lower()
        if encoder in {"nvenc", "h264_nvenc"}:
            if not nvenc_runtime_available():
                logging.warning(
                    "Remote edge FILE h264_nvenc unavailable; falling back to libx264 for source file encode: session=%s job=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                )
                encoder = "libx264"
            else:
                bitrate = str(os.getenv("REMOTE_EDGE_FILE_VIDEO_BITRATE", "5000k") or "5000k")
                return [
                    "-c:v",
                    "h264_nvenc",
                    "-profile:v",
                    "high",
                    "-preset",
                    str(os.getenv("REMOTE_EDGE_FILE_NVENC_PRESET", "p4") or "p4"),
                    "-rc",
                    "vbr",
                    "-cq",
                    str(os.getenv("REMOTE_EDGE_FILE_NVENC_CQ", "21") or "21"),
                    "-b:v",
                    bitrate,
                    "-maxrate",
                    str(os.getenv("REMOTE_EDGE_FILE_VIDEO_MAXRATE", bitrate) or bitrate),
                    "-bufsize",
                    str(os.getenv("REMOTE_EDGE_FILE_VIDEO_BUFSIZE", "10000k") or "10000k"),
                    "-g",
                    str(gop),
                    "-keyint_min",
                    str(gop),
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ]
        if encoder in {"libx264", "x264", "h264"}:
            return [
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                str(os.getenv("REMOTE_EDGE_FILE_X264_PRESET", "veryfast") or "veryfast"),
                "-crf",
                str(os.getenv("REMOTE_EDGE_FILE_X264_CRF", "20") or "20"),
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
            ]
        return [
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            str(os.getenv("REMOTE_EDGE_FILE_X264_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("REMOTE_EDGE_FILE_X264_CRF", "20") or "20"),
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]

    def _file_audio_encode_args(self) -> list[str]:
        return [
                "-c:a",
                "aac",
                "-b:a",
                str(os.getenv("REMOTE_EDGE_FILE_AUDIO_BITRATE", "160k") or "160k"),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
        ]

    def _put_file_output(self, signed_url: str, mp4_path: str) -> None:
        put_file_to_signed_url(
            signed_url=str(signed_url),
            path=str(mp4_path),
            content_type=str(self.cfg.file_content_type or "video/mp4"),
            connect_timeout=20.0,
            read_timeout=1800.0,
            env_prefix="REMOTE_EDGE_FILE_UPLOAD",
            log_prefix="remote-edge-file-upload",
        )

    def queue_pcm16le(
        self,
        payload: bytes,
        *,
        sample_rate: int | None = None,
        segment_id: str | None = None,
        segment_kind: str = "",
        segment_frames: int | None = None,
        segment_audible_samples: int | None = None,
        segment_turn_done: bool = False,
        subtitle_text: str = "",
        subtitle_start_samples: int | None = None,
        subtitle_end_samples: int | None = None,
        subtitle_total_samples: int | None = None,
        subtitle_alignment: dict[str, Any] | None = None,
        subtitle_normalized_alignment: dict[str, Any] | None = None,
        subtitle_alignment_base_samples: int | None = None,
    ) -> None:
        self._raise_if_publish_failed()
        samples = int(len(payload) // 2)
        if samples <= 0:
            return
        self.audio_chunks_received += 1
        segment_id_s = str(segment_id or "").strip()
        if segment_id_s:
            segment_audio_obj = _SegmentAudio(
                payload=bytes(payload),
                sample_rate=int(sample_rate or self.cfg.sample_rate),
                segment_id=str(segment_id_s),
                segment_kind=str(segment_kind or ""),
                segment_frames=None if segment_frames is None else int(segment_frames),
                audible_samples=None if segment_audible_samples is None else int(segment_audible_samples),
                turn_done=bool(segment_turn_done),
                subtitle_text=str(subtitle_text or ""),
                subtitle_start_samples=None if subtitle_start_samples is None else int(subtitle_start_samples),
                subtitle_end_samples=None if subtitle_end_samples is None else int(subtitle_end_samples),
                subtitle_total_samples=None if subtitle_total_samples is None else int(subtitle_total_samples),
                subtitle_alignment=dict(subtitle_alignment) if isinstance(subtitle_alignment, dict) else None,
                subtitle_normalized_alignment=(
                    dict(subtitle_normalized_alignment) if isinstance(subtitle_normalized_alignment, dict) else None
                ),
                subtitle_alignment_base_samples=(
                    None if subtitle_alignment_base_samples is None else int(subtitle_alignment_base_samples)
                ),
            )
            self.segment_audio[segment_id_s] = segment_audio_obj
            renderer = self.subtitle_renderer
            if renderer is not None and bool(getattr(renderer, "enabled", False)):
                alignment = (
                    segment_audio_obj.subtitle_normalized_alignment
                    if isinstance(segment_audio_obj.subtitle_normalized_alignment, dict)
                    and segment_audio_obj.subtitle_normalized_alignment
                    else segment_audio_obj.subtitle_alignment
                )
                if str(segment_audio_obj.subtitle_text or "").strip() and isinstance(alignment, dict) and alignment:
                    renderer.prepare(text=str(segment_audio_obj.subtitle_text or ""), alignment=alignment)
        else:
            self.pending_audio.append((bytes(payload), int(sample_rate or self.cfg.sample_rate)))
        try:
            asyncio.get_running_loop().create_task(self._notify_publish_ready())
        except RuntimeError:
            pass

    async def _notify_publish_ready(self) -> None:
        async with self.publish_cv:
            self.publish_cv.notify_all()

    async def flush_pending_audio(self) -> None:
        while self.pending_audio:
            payload, sample_rate = self.pending_audio.popleft()
            await self.push_pcm16le(payload, sample_rate=int(sample_rate))

    async def push_pcm16le(self, payload: bytes, *, sample_rate: int | None = None) -> None:
        if self.audio_source is None:
            raise RuntimeError("audio source is not started")
        sample_rate_i = int(sample_rate or self.cfg.sample_rate)
        sample_rate_i = max(1, sample_rate_i)
        total_samples = int(len(payload) // 2)
        if total_samples <= 0:
            return
        frame_samples = max(1, int(round(float(sample_rate_i) * float(self.audio_frame_ms) / 1000.0)))
        frame_bytes = int(frame_samples * 2)
        view = memoryview(payload)
        for offset in range(0, int(total_samples * 2), frame_bytes):
            chunk = bytes(view[offset : min(offset + frame_bytes, int(total_samples * 2))])
            samples = int(len(chunk) // 2)
            if samples <= 0:
                continue
            rtc = _livekit_rtc()
            af = rtc.AudioFrame(
                chunk,
                sample_rate=int(sample_rate_i),
                num_channels=1,
                samples_per_channel=int(samples),
            )
            if self.av_sync is not None:
                await self.av_sync.push(af)
            else:
                await self.audio_source.capture_frame(af)
            self.audio_frames += 1
            self.audio_seconds_published += float(samples) / float(sample_rate_i)
        self.audio_chunks_published += 1

    def _reset_receive_stats_window(self, now: float) -> None:
        self._receive_stats_window_started = float(now)
        self._receive_msg_count = 0
        self._receive_latent_count = 0
        self._receive_audio_count = 0
        self._receive_wire_bytes = 0
        self._receive_payload_bytes = 0
        self._receive_message_wait_total_sec = 0.0
        self._receive_message_wait_max_sec = 0.0
        self._receive_read_total_sec = 0.0
        self._receive_read_max_sec = 0.0
        self._receive_transport_decode_total_sec = 0.0
        self._receive_transport_decode_max_sec = 0.0
        self._receive_enqueue_wait_total_sec = 0.0
        self._receive_enqueue_wait_max_sec = 0.0
        self._receive_decode_queue_wait_total_sec = 0.0
        self._receive_decode_queue_wait_max_sec = 0.0
        self._receive_latent_decode_total_sec = 0.0
        self._receive_latent_decode_max_sec = 0.0
        self._receive_video_enqueue_total_sec = 0.0
        self._receive_video_enqueue_max_sec = 0.0
        self._receive_decoded_block_count = 0
        self._receive_decoded_frames = 0

    def _record_transport_message(
        self,
        *,
        typ: str,
        wire_bytes: int,
        payload_read_sec: float,
        message_wait_sec: float,
    ) -> None:
        self._receive_msg_count += 1
        self._receive_wire_bytes += int(max(0, int(wire_bytes)))
        wait = max(0.0, float(message_wait_sec))
        read = max(0.0, float(payload_read_sec))
        self._receive_message_wait_total_sec += float(wait)
        self._receive_message_wait_max_sec = max(float(self._receive_message_wait_max_sec), float(wait))
        self._receive_read_total_sec += float(read)
        self._receive_read_max_sec = max(float(self._receive_read_max_sec), float(read))
        if str(typ or "").strip().lower() == "audio.pcm16le":
            self._receive_audio_count += 1
            self._receive_payload_bytes += int(max(0, int(wire_bytes)))
        if float(self.receive_slow_warn_sec) > 0.0 and read >= float(self.receive_slow_warn_sec):
            logging.warning(
                "Remote edge slow socket read: session=%s job=%s type=%s wire=%.2fMiB message_wait=%.3fs payload_read=%.3fs latent_q=%d pending_blocks=%d buffered_frames=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                str(typ or ""),
                float(max(0, int(wire_bytes))) / 1048576.0,
                float(wait),
                float(read),
                int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0,
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
            )

    def _record_transport_decode(self, *, label: str, raw_bytes: int, decode_sec: float) -> None:
        raw = int(max(0, int(raw_bytes)))
        self._receive_payload_bytes += int(raw)
        decode = max(0.0, float(decode_sec))
        self._receive_transport_decode_total_sec += float(decode)
        self._receive_transport_decode_max_sec = max(float(self._receive_transport_decode_max_sec), float(decode))
        if float(self.receive_slow_warn_sec) > 0.0 and decode >= float(self.receive_slow_warn_sec):
            logging.warning(
                "Remote edge slow payload decode: session=%s job=%s label=%s raw=%.2fMiB decode=%.3fs",
                self.cfg.session_id,
                self.cfg.job_id,
                str(label or ""),
                float(raw) / 1048576.0,
                float(decode),
            )

    def _record_latent_enqueue(
        self,
        *,
        put_wait_sec: float,
        queue_size: int,
        wire_payload_len: int,
        raw_payload_len: int,
        payload_read_sec: float,
        transport_decode_sec: float,
    ) -> None:
        self._receive_latent_count += 1
        wait = max(0.0, float(put_wait_sec))
        self._receive_enqueue_wait_total_sec += float(wait)
        self._receive_enqueue_wait_max_sec = max(float(self._receive_enqueue_wait_max_sec), float(wait))
        if float(self.receive_slow_warn_sec) > 0.0 and wait >= float(self.receive_slow_warn_sec):
            logging.warning(
                "Remote edge latent queue put waited: session=%s job=%s wait=%.3fs queued=%d max=%d wire=%.2fMiB raw=%.2fMiB payload_read=%.3fs transport_decode=%.3fs pending_blocks=%d buffered_frames=%d rtmp_queue=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                float(wait),
                int(queue_size),
                int(self.latent_queue_max),
                float(max(0, int(wire_payload_len))) / 1048576.0,
                float(max(0, int(raw_payload_len))) / 1048576.0,
                max(0.0, float(payload_read_sec)),
                max(0.0, float(transport_decode_sec)),
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
                int(self._rtmp_queue_len()),
            )
        self._maybe_log_receive_stats()

    def _record_latent_decode_result(
        self,
        *,
        queue_wait_sec: float,
        decode_sec: float,
        video_enqueue_sec: float,
        frames: int,
        payload_len: int,
    ) -> None:
        queue_wait = max(0.0, float(queue_wait_sec))
        decode = max(0.0, float(decode_sec))
        video_enqueue = max(0.0, float(video_enqueue_sec))
        self._receive_decode_queue_wait_total_sec += float(queue_wait)
        self._receive_decode_queue_wait_max_sec = max(float(self._receive_decode_queue_wait_max_sec), float(queue_wait))
        self._receive_latent_decode_total_sec += float(decode)
        self._receive_latent_decode_max_sec = max(float(self._receive_latent_decode_max_sec), float(decode))
        self._receive_video_enqueue_total_sec += float(video_enqueue)
        self._receive_video_enqueue_max_sec = max(float(self._receive_video_enqueue_max_sec), float(video_enqueue))
        self._receive_decoded_block_count += 1
        self._receive_decoded_frames += int(max(0, int(frames)))
        decode_warn = float(self.latent_decode_slow_warn_sec)
        enqueue_warn = float(self.receive_slow_warn_sec)
        if (decode_warn > 0.0 and decode >= decode_warn) or (enqueue_warn > 0.0 and video_enqueue >= enqueue_warn):
            logging.warning(
                "Remote edge latent decode timing: session=%s job=%s queue_wait=%.3fs decode=%.3fs video_enqueue=%.3fs frames=%d payload=%.2fMiB latent_q=%d pending_blocks=%d buffered_frames=%d rtmp_queue=%d gap_fill=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                float(queue_wait),
                float(decode),
                float(video_enqueue),
                int(frames),
                float(max(0, int(payload_len))) / 1048576.0,
                int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0,
                int(len(self.pending_blocks)),
                int(self.buffered_video_frames),
                int(self._rtmp_queue_len()),
                int(self.rtmp_gap_fill_frames),
            )
        self._maybe_log_receive_stats()

    def _maybe_log_receive_stats(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not bool(force) and now - float(self._receive_stats_last_log) < float(self.receive_stats_interval_sec):
            return
        total_events = int(self._receive_msg_count) + int(self._receive_latent_count) + int(self._receive_decoded_frames)
        if not bool(force) and total_events <= 0:
            return
        msg_count = max(1, int(self._receive_msg_count))
        latent_count = max(1, int(self._receive_latent_count))
        decoded_blocks = max(1, int(self._receive_decoded_block_count))
        window_sec = max(0.001, float(now - float(self._receive_stats_window_started)))
        try:
            latent_q = int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0
        except Exception:
            latent_q = 0
        logging.warning(
            "Remote edge receive stats: output=%s session=%s age=%.1fs window=%.1fs messages=%d latents=%d audio=%d wire=%.2fMiB raw=%.2fMiB message_wait_avg=%.3fs message_wait_max=%.3fs payload_read_avg=%.3fs payload_read_max=%.3fs transport_decode_avg=%.3fs transport_decode_max=%.3fs queue_put_avg=%.3fs queue_put_max=%.3fs decode_queue_avg=%.3fs decode_queue_max=%.3fs latent_decode_avg=%.3fs latent_decode_max=%.3fs video_enqueue_avg=%.3fs video_enqueue_max=%.3fs decoded_frames=%d latent_q=%d pending_blocks=%d buffered_frames=%d rtmp_queue=%d rtmp_gap_fill=%d",
            str(self.cfg.output or "livekit"),
            self.cfg.session_id,
            float(now - self.started_at),
            float(window_sec),
            int(self._receive_msg_count),
            int(self._receive_latent_count),
            int(self._receive_audio_count),
            float(self._receive_wire_bytes) / 1048576.0,
            float(self._receive_payload_bytes) / 1048576.0,
            float(self._receive_message_wait_total_sec) / float(msg_count),
            float(self._receive_message_wait_max_sec),
            float(self._receive_read_total_sec) / float(msg_count),
            float(self._receive_read_max_sec),
            float(self._receive_transport_decode_total_sec) / float(latent_count),
            float(self._receive_transport_decode_max_sec),
            float(self._receive_enqueue_wait_total_sec) / float(latent_count),
            float(self._receive_enqueue_wait_max_sec),
            float(self._receive_decode_queue_wait_total_sec) / float(decoded_blocks),
            float(self._receive_decode_queue_wait_max_sec),
            float(self._receive_latent_decode_total_sec) / float(decoded_blocks),
            float(self._receive_latent_decode_max_sec),
            float(self._receive_video_enqueue_total_sec) / float(decoded_blocks),
            float(self._receive_video_enqueue_max_sec),
            int(self._receive_decoded_frames),
            int(latent_q),
            int(len(self.pending_blocks)),
            int(self.buffered_video_frames),
            int(self._rtmp_queue_len()),
            int(self.rtmp_gap_fill_frames),
        )
        self._receive_stats_last_log = float(now)
        self._reset_receive_stats_window(float(now))

    def log_stats(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not bool(force) and now - self._last_stats_log < float(self.stats_interval_sec):
            return
        self._last_stats_log = now
        audio_q = 0.0
        try:
            audio_q = (
                float(getattr(self.audio_source, "queued_duration", 0.0) or 0.0)
                if self.audio_source is not None
                else 0.0
            )
        except Exception:
            audio_q = 0.0
        stats_fps = int(self._rtmp_pipe_fps()) if str(self.cfg.output or "").strip().lower() == "rtmp" else int(self.cfg.fps)
        video_sec = float(self.frames) / float(max(1, int(stats_fps)))
        pending_audio_sec = 0.0
        for payload, sample_rate in self.pending_audio:
            pending_audio_sec += float(len(payload) // 2) / float(max(1, int(sample_rate or self.cfg.sample_rate)))
        rtmp_audio_pad_sec = float(self.rtmp_audio_pad_samples) / float(max(1, int(self.cfg.sample_rate)))
        latent_q = 0
        try:
            latent_q = int(self.latent_decode_queue.qsize()) if self.latent_decode_queue is not None else 0
        except Exception:
            latent_q = 0
        logging.warning(
            "Remote edge publish stats: output=%s session=%s frames=%d video_sec=%.2f audio_chunks=%d/%d audio_frames=%d audio_sec=%.2f av_drift=%.3f latent_q=%d pending_blocks=%d buffered_frames=%d pending_audio=%d pending_audio_sec=%.2f pending_segment_audio=%d/%.2fs audio_q=%.2f rtmp_queue=%d rtmp_writer_ticks=%d rtmp_stale_drops=%d rtmp_resync_drops=%d rtmp_gap_fill=%d rtmp_audio_hold=%d rtmp_audio_pad=%.3fs/%d rtmp_audio_wait=%.3fs/%d",
            str(self.cfg.output or "livekit"),
            self.cfg.session_id,
            int(self.frames),
            float(video_sec),
            int(self.audio_chunks_published),
            int(self.audio_chunks_received),
            int(self.audio_frames),
            float(self.audio_seconds_published),
            float(self.audio_seconds_published - video_sec),
            int(latent_q),
            int(len(self.pending_blocks)),
            int(self.buffered_video_frames),
            int(len(self.pending_audio)),
            float(pending_audio_sec),
            int(len(self.segment_audio)),
            float(self._pending_segment_audio_samples()) / float(max(1, int(self.cfg.sample_rate))),
            float(audio_q),
            int(self._rtmp_queue_len()),
            int(self.rtmp_writer_ticks),
            int(self.rtmp_stale_drop_count),
            int(self.rtmp_resync_drop_count),
            int(self.rtmp_gap_fill_frames),
            int(self.rtmp_audio_hold_frames),
            float(rtmp_audio_pad_sec),
            int(self.rtmp_audio_pad_events),
            float(self.rtmp_audio_wait_total_sec),
            int(self.rtmp_audio_wait_events),
        )

    def decode_latents(
        self,
        payload: bytes,
        *,
        codec: str,
        shape: str,
        dtype: str,
        keep_last_frames: int | None,
        reset_vae: bool,
        prime_only: bool,
        face_restore: float | None,
        background_restore: float | None,
    ) -> list[bytes]:
        if not bool(_env_flag("REMOTE_EDGE_LATENT_DECODE", "1")):
            raise RuntimeError("remote latent decode is disabled; set REMOTE_EDGE_LATENT_DECODE=1 on edge")
        if self.latent_decoder is None:
            self.latent_decoder = get_shared_wan_latent_decoder()
        return self.latent_decoder.decode_payload_threadsafe(
            payload,
            codec=codec,
            shape=shape,
            dtype=dtype,
            keep_last_frames=keep_last_frames,
            reset=reset_vae,
            prime_only=prime_only,
            face_restore=face_restore,
            background_restore=background_restore,
            output_width=int(self.cfg.width),
            output_height=int(self.cfg.height),
        )

    def decode_latents_tensor(
        self,
        payload: bytes,
        *,
        codec: str,
        shape: str,
        dtype: str,
        keep_last_frames: int | None,
        reset_vae: bool,
        prime_only: bool,
        face_restore: float | None,
        background_restore: float | None,
        resize_output: bool = True,
        apply_post_vae: bool = True,
        return_m11: bool = False,
    ) -> Any:
        if not bool(_env_flag("REMOTE_EDGE_LATENT_DECODE", "1")):
            raise RuntimeError("remote latent decode is disabled; set REMOTE_EDGE_LATENT_DECODE=1 on edge")
        if self.latent_decoder is None:
            self.latent_decoder = get_shared_wan_latent_decoder()
        return self.latent_decoder.decode_payload_tensor_01_threadsafe(
            payload,
            codec=codec,
            shape=shape,
            dtype=dtype,
            keep_last_frames=keep_last_frames,
            reset=reset_vae,
            prime_only=prime_only,
            face_restore=face_restore,
            background_restore=background_restore,
            output_width=int(self.cfg.width),
            output_height=int(self.cfg.height),
            resize_output=bool(resize_output),
            apply_post_vae=bool(apply_post_vae),
            return_m11=bool(return_m11),
        )

    def postprocess_latents_tensor(
        self,
        frames_tensor: Any,
        *,
        face_restore: float | None,
        background_restore: float | None,
        resize_output: bool = True,
        apply_post_vae: bool = True,
        input_range: str = "01",
    ) -> Any:
        if self.latent_decoder is None:
            self.latent_decoder = get_shared_wan_latent_decoder()
        return self.latent_decoder.postprocess_frames_tensor_01_threadsafe(
            frames_tensor,
            face_restore=face_restore,
            background_restore=background_restore,
            output_width=int(self.cfg.width),
            output_height=int(self.cfg.height),
            resize_output=bool(resize_output),
            apply_post_vae=bool(apply_post_vae),
            input_range=str(input_range or "01"),
        )

    def _decode_live_latents_as_gpu_tensor(self) -> bool:
        backend = str(self.live_rife_backend or "").strip().lower()
        return bool(
            str(self.cfg.output or "").strip().lower() == "rtmp"
            and bool(self.live_rife_enabled)
            and (
                backend == "torch-rife"
                or (backend == "nvidia-fruc" and bool(self.live_rife_pre_resize))
            )
            and not bool(self._live_rife_pairwise_enabled())
        )

    @staticmethod
    def _restore_requested(face_restore: float | None, background_restore: float | None) -> bool:
        try:
            face = float(0.0 if face_restore is None else face_restore)
        except Exception:
            face = 0.0
        try:
            background = float(0.0 if background_restore is None else background_restore)
        except Exception:
            background = 0.0
        return bool(max(0.0, face) > 0.0 or max(0.0, background) > 0.0)

    def _async_post_vae_for_job(self, *, face_restore: float | None, background_restore: float | None) -> bool:
        return bool(
            self.async_post_vae_enabled
            and not bool(self.post_vae_disabled_after_error)
            and str(self.cfg.output or "").strip().lower() == "rtmp"
            and not bool(self.live_rife_enabled)
            and self._restore_requested(face_restore, background_restore)
        )

    def _ensure_latent_decode_loop(self) -> asyncio.Queue[_LatentDecodeJob | None]:
        if self.latent_decode_queue is None:
            self.latent_decode_queue = asyncio.Queue(maxsize=int(self.latent_queue_max))
        if self.latent_decode_task is None or self.latent_decode_task.done():
            self.latent_decode_task = asyncio.create_task(
                self._latent_decode_loop(),
                name=f"remote-edge-latent-decode-{self.cfg.session_id}",
            )
        return self.latent_decode_queue

    def _ensure_latent_postprocess_loop(self) -> asyncio.Queue[_LatentPostprocessJob | None]:
        if self.latent_postprocess_queue is None:
            self.latent_postprocess_queue = asyncio.Queue(maxsize=int(self.latent_postprocess_queue_max))
        if self.latent_postprocess_task is None or self.latent_postprocess_task.done():
            self.latent_postprocess_task = asyncio.create_task(
                self._latent_postprocess_loop(),
                name=f"remote-edge-post-vae-{self.cfg.session_id}",
            )
        return self.latent_postprocess_queue

    async def enqueue_latent_postprocess(self, job: _LatentPostprocessJob) -> None:
        queue = self._ensure_latent_postprocess_loop()
        await queue.put(job)

    async def enqueue_latents(
        self,
        payload: bytes,
        *,
        codec: str,
        shape: str,
        dtype: str,
        keep_last_frames: int | None,
        reset_vae: bool,
        prime_only: bool,
        face_restore: float | None,
        background_restore: float | None,
        timestamp_us: int | None,
        segment_id: str | None = None,
        segment_kind: str = "",
        segment_start_frame: int | None = None,
        segment_frames: int | None = None,
        avatar_ref_path: str = "",
        wire_payload_len: int | None = None,
        payload_read_sec: float = 0.0,
        transport_decode_sec: float = 0.0,
    ) -> None:
        if self.latent_decode_failed_exc is not None:
            raise RuntimeError(f"remote edge latent decode failed: {self.latent_decode_failed_exc}") from self.latent_decode_failed_exc
        self._raise_if_publish_failed()
        queue = self._ensure_latent_decode_loop()
        enqueued_at = time.perf_counter()
        job = _LatentDecodeJob(
            payload=bytes(payload),
            codec=str(codec),
            shape=str(shape),
            dtype=str(dtype),
            keep_last_frames=keep_last_frames,
            reset_vae=bool(reset_vae),
            prime_only=bool(prime_only),
            face_restore=face_restore,
            background_restore=background_restore,
            timestamp_us=timestamp_us,
            segment_id=None if not str(segment_id or "").strip() else str(segment_id),
            segment_kind=str(segment_kind or ""),
            segment_start_frame=None if segment_start_frame is None else int(segment_start_frame),
            segment_frames=None if segment_frames is None else int(segment_frames),
            avatar_ref_path=str(avatar_ref_path or ""),
            wire_payload_len=int(len(payload) if wire_payload_len is None else max(0, int(wire_payload_len))),
            payload_read_sec=max(0.0, float(payload_read_sec)),
            transport_decode_sec=max(0.0, float(transport_decode_sec)),
            enqueued_at=float(enqueued_at),
        )
        put_started = time.perf_counter()
        await queue.put(
            job
        )
        qsize = int(queue.qsize())
        put_wait = max(0.0, time.perf_counter() - float(put_started))
        self._record_latent_enqueue(
            put_wait_sec=float(put_wait),
            queue_size=int(qsize),
            wire_payload_len=int(job.wire_payload_len),
            raw_payload_len=int(len(job.payload)),
            payload_read_sec=float(job.payload_read_sec),
            transport_decode_sec=float(job.transport_decode_sec),
        )
        now = time.monotonic()
        if qsize >= max(2, int(self.latent_queue_max) // 2) and now - float(self.latent_decode_last_log) >= 5.0:
            logging.warning(
                "Remote edge latent queue filling: session=%s job=%s queued=%d max=%d decoded=%d",
                self.cfg.session_id,
                self.cfg.job_id,
                int(qsize),
                int(self.latent_queue_max),
                int(self.latent_decode_count),
            )
            self.latent_decode_last_log = float(now)

    async def _latent_decode_loop(self) -> None:
        queue = self.latent_decode_queue
        if queue is None:
            return
        try:
            while not self.publish_stop.is_set():
                job = await queue.get()
                try:
                    if job is None:
                        return
                    decode_started = time.perf_counter()
                    queue_wait = max(0.0, float(decode_started) - float(job.enqueued_at or decode_started))
                    decode_core_sec = 0.0
                    stabilize_sec = 0.0
                    post_enqueue_sec = 0.0
                    face_restore, background_restore = self._adaptive_restore_strengths(
                        face_restore=job.face_restore,
                        background_restore=job.background_restore,
                    )
                    self._record_file_requested_restore(
                        face_restore=face_restore,
                        background_restore=background_restore,
                    )
                    file_post_vae_after_rife = bool(self._file_post_vae_after_inline_rife_enabled())
                    decode_face_restore = face_restore
                    decode_background_restore = background_restore
                    if (
                        bool(file_post_vae_after_rife)
                        and self._restore_requested(face_restore, background_restore)
                    ):
                        decode_face_restore = 0.0
                        decode_background_restore = 0.0
                    file_native_pre_resize = bool(self._file_native_pre_resize_enabled())
                    use_tensor_frames = bool(
                        self._decode_live_latents_as_gpu_tensor()
                        or file_native_pre_resize
                    )
                    async_post_vae = self._async_post_vae_for_job(
                        face_restore=decode_face_restore,
                        background_restore=decode_background_restore,
                    )
                    if bool(use_tensor_frames):
                        frames_tensor = await asyncio.to_thread(
                            self.decode_latents_tensor,
                            job.payload,
                            codec=job.codec,
                            shape=job.shape,
                            dtype=job.dtype,
                            keep_last_frames=job.keep_last_frames,
                            reset_vae=job.reset_vae,
                            prime_only=job.prime_only,
                            face_restore=decode_face_restore,
                            background_restore=decode_background_restore,
                            resize_output=not bool(
                                file_native_pre_resize
                                or self.live_rife_pre_resize
                            ),
                            apply_post_vae=True,
                        )
                        frames = []
                        frame_count = int(self._tensor_frame_count(frames_tensor))
                    elif bool(async_post_vae):
                        native_post_vae_range = bool(_env_flag("REMOTE_EDGE_ASYNC_POST_VAE_NATIVE_RANGE", "1"))
                        frames_tensor = await asyncio.to_thread(
                            self.decode_latents_tensor,
                            job.payload,
                            codec=job.codec,
                            shape=job.shape,
                            dtype=job.dtype,
                            keep_last_frames=job.keep_last_frames,
                            reset_vae=job.reset_vae,
                            prime_only=job.prime_only,
                            face_restore=0.0,
                            background_restore=0.0,
                            resize_output=False,
                            apply_post_vae=False,
                            return_m11=bool(native_post_vae_range),
                        )
                        frames = []
                        frame_count = int(self._tensor_frame_count(frames_tensor))
                        decode_core_sec = max(0.0, time.perf_counter() - float(decode_started))
                    else:
                        frames = await asyncio.to_thread(
                            self.decode_latents,
                            job.payload,
                            codec=job.codec,
                            shape=job.shape,
                            dtype=job.dtype,
                            keep_last_frames=job.keep_last_frames,
                            reset_vae=job.reset_vae,
                            prime_only=job.prime_only,
                            face_restore=decode_face_restore,
                            background_restore=decode_background_restore,
                        )
                        frames_tensor = None
                        frame_count = int(len(frames or []))
                    decode_sec = max(0.0, time.perf_counter() - float(decode_started))
                    decoded_frame_count = int(frame_count)
                    visible_start_frame, visible_frame_count = self._visible_frame_window(
                        int(frame_count),
                        job.segment_start_frame,
                        job.segment_frames,
                    )
                    if int(frame_count) > 0 and int(visible_frame_count) <= 0:
                        self.latent_decode_count += 1
                        self._record_latent_decode_result(
                            queue_wait_sec=float(queue_wait),
                            decode_sec=float(decode_sec),
                            video_enqueue_sec=0.0,
                            frames=0,
                            payload_len=int(len(job.payload)),
                        )
                        continue
                    if (
                        int(visible_frame_count) > 0
                        and (int(visible_start_frame) > 0 or int(visible_frame_count) < int(frame_count))
                    ):
                        if frames_tensor is not None:
                            frames_tensor = self._slice_tensor_frames(
                                frames_tensor,
                                int(visible_frame_count),
                                start_frame=int(visible_start_frame),
                            )
                        if frames:
                            frames = self._slice_frame_bytes(
                                list(frames),
                                int(visible_frame_count),
                                start_frame=int(visible_start_frame),
                            )
                        frame_count = int(visible_frame_count)
                        logging.warning(
                            "Remote edge latent visible trim: session=%s job=%s segment=%s decoded=%d start=%d visible=%d",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            str(job.segment_id or ""),
                            int(decoded_frame_count),
                            int(visible_start_frame),
                            int(visible_frame_count),
                        )
                    self.latent_decode_count += 1
                    video_enqueue_sec = 0.0
                    if int(frame_count) > 0 and bool(async_post_vae):
                        defer_stage = bool(_env_flag("REMOTE_EDGE_ASYNC_POST_VAE_DEFER_STAGE", "0"))
                        if not bool(defer_stage):
                            stabilize_started = time.perf_counter()
                            frames_tensor = self._stabilize_tensor_for_async_post_vae(frames_tensor)
                            stabilize_sec = max(0.0, time.perf_counter() - float(stabilize_started))
                        enqueue_started = time.perf_counter()
                        await self.enqueue_latent_postprocess(
                            _LatentPostprocessJob(
                                frames_tensor=frames_tensor,
                                face_restore=face_restore,
                                background_restore=background_restore,
                                timestamp_us=job.timestamp_us,
                                segment_id=job.segment_id,
                                segment_kind=job.segment_kind,
                                segment_start_frame=None,
                                segment_frames=job.segment_frames,
                                avatar_ref_path=str(job.avatar_ref_path or ""),
                                input_range="m11" if bool(native_post_vae_range) else "01",
                                queue_wait_sec=float(queue_wait),
                                decode_sec=float(decode_sec),
                                payload_len=int(len(job.payload)),
                                frame_count=int(frame_count),
                                enqueued_at=float(time.perf_counter()),
                                defer_stage=bool(defer_stage),
                            )
                        )
                        post_enqueue_sec = max(0.0, time.perf_counter() - float(enqueue_started))
                        if (
                            bool(_env_flag("POST_VAE_TIMING_LOG", "0"))
                            or float(decode_sec) >= 0.75
                            or float(stabilize_sec) >= 0.10
                            or bool(defer_stage)
                        ):
                            logging.warning(
                                "Remote edge latent phase timing: session=%s job=%s frames=%d queue_wait=%.3fs decode_core=%.3fs stabilize=%.3fs post_enqueue=%.3fs total_before_post=%.3fs stage=%s deferred_stage=%d latent_q=%d post_q=%d",
                                self.cfg.session_id,
                                self.cfg.job_id,
                                int(frame_count),
                                float(queue_wait),
                                float(decode_core_sec or decode_sec),
                                float(stabilize_sec),
                                float(post_enqueue_sec),
                                float(time.perf_counter() - float(decode_started)),
                                str(os.getenv("REMOTE_EDGE_ASYNC_POST_VAE_STAGE", "source") or "source"),
                                1 if bool(defer_stage) else 0,
                                int(queue.qsize()),
                                int(self.latent_postprocess_queue.qsize()) if self.latent_postprocess_queue is not None else 0,
                            )
                        continue
                    if int(frame_count) > 0 and bool(use_tensor_frames):
                        video_enqueue_started = time.perf_counter()
                        await self.enqueue_video_tensor(
                            frames_tensor,
                            timestamp_us=job.timestamp_us,
                            segment_id=job.segment_id,
                            segment_kind=job.segment_kind,
                            segment_start_frame=None,
                            segment_frames=job.segment_frames,
                            avatar_ref_path=str(job.avatar_ref_path or ""),
                        )
                        video_enqueue_sec = max(0.0, time.perf_counter() - float(video_enqueue_started))
                    elif frames:
                        video_enqueue_started = time.perf_counter()
                        await self.enqueue_video_frames(
                            frames,
                            timestamp_us=job.timestamp_us,
                            segment_id=job.segment_id,
                            segment_kind=job.segment_kind,
                            segment_start_frame=None,
                            segment_frames=job.segment_frames,
                            avatar_ref_path=str(job.avatar_ref_path or ""),
                        )
                        video_enqueue_sec = max(0.0, time.perf_counter() - float(video_enqueue_started))
                    self._record_latent_decode_result(
                        queue_wait_sec=float(queue_wait),
                        decode_sec=float(decode_sec),
                        video_enqueue_sec=float(video_enqueue_sec),
                        frames=int(frame_count),
                        payload_len=int(len(job.payload)),
                    )
                except Exception as e:
                    if _is_cuda_context_poisoned_error(e):
                        self.latent_decode_failed_exc = e
                        self.publish_failed_exc = e
                        self.publish_stop.set()
                        async with self.publish_cv:
                            self.publish_cv.notify_all()
                        logging.exception(
                            "Remote edge latent decode hit fatal CUDA error: session=%s job=%s codec=%s shape=%s dtype=%s reset=%d prime=%d keep_last_frames=%s",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            getattr(job, "codec", ""),
                            getattr(job, "shape", ""),
                            getattr(job, "dtype", ""),
                            1 if bool(getattr(job, "reset_vae", False)) else 0,
                            1 if bool(getattr(job, "prime_only", False)) else 0,
                            getattr(job, "keep_last_frames", None),
                        )
                        _exit_after_fatal_cuda_error(
                            context="latent_decode",
                            session_id=self.cfg.session_id,
                            job_id=self.cfg.job_id,
                            err=e,
                        )
                    if bool(self.latent_decode_fail_open) and str(self.cfg.output or "").strip().lower() == "rtmp":
                        self.latent_decode_fail_open_blocks += 1
                        logging.exception(
                            "Remote edge latent decode failed open; skipped block and continuing live session: session=%s job=%s skipped=%d codec=%s shape=%s dtype=%s reset=%d prime=%d keep_last_frames=%s err=%s",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            int(self.latent_decode_fail_open_blocks),
                            getattr(job, "codec", ""),
                            getattr(job, "shape", ""),
                            getattr(job, "dtype", ""),
                            1 if bool(getattr(job, "reset_vae", False)) else 0,
                            1 if bool(getattr(job, "prime_only", False)) else 0,
                            getattr(job, "keep_last_frames", None),
                            e,
                        )
                        continue
                    raise
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.latent_decode_failed_exc = e
            self.publish_failed_exc = e
            self.publish_stop.set()
            async with self.publish_cv:
                self.publish_cv.notify_all()
            logging.exception("Remote edge latent decode loop failed: session=%s job=%s", self.cfg.session_id, self.cfg.job_id)
            if _is_cuda_context_poisoned_error(e):
                _exit_after_fatal_cuda_error(
                    context="latent_decode_outer",
                    session_id=self.cfg.session_id,
                    job_id=self.cfg.job_id,
                    err=e,
                )

    async def _latent_postprocess_loop(self) -> None:
        queue = self.latent_postprocess_queue
        if queue is None:
            return
        try:
            while not self.publish_stop.is_set():
                job = await queue.get()
                try:
                    if job is None:
                        return
                    post_started = time.perf_counter()
                    post_queue_wait = max(0.0, float(post_started) - float(job.enqueued_at or post_started))
                    post_stage_sec = 0.0
                    post_failed = False
                    apply_restore = not bool(self.post_vae_disabled_after_error)
                    try:
                        frames_input = job.frames_tensor
                        if bool(job.defer_stage):
                            stage_started = time.perf_counter()
                            frames_input = await asyncio.to_thread(
                                self._stabilize_tensor_for_async_post_vae,
                                frames_input,
                            )
                            post_stage_sec = max(0.0, time.perf_counter() - float(stage_started))
                        frames_tensor = await asyncio.to_thread(
                            self.postprocess_latents_tensor,
                            frames_input,
                            face_restore=job.face_restore if bool(apply_restore) else 0.0,
                            background_restore=job.background_restore if bool(apply_restore) else 0.0,
                            resize_output=True,
                            apply_post_vae=bool(apply_restore),
                            input_range=str(job.input_range or "01"),
                        )
                    except Exception as e:
                        cuda_poisoned = bool(_is_cuda_context_poisoned_error(e))
                        can_fail_open = bool(self.post_vae_fail_open) and str(self.cfg.output or "").strip().lower() == "rtmp"
                        if bool(cuda_poisoned) and not bool(can_fail_open):
                            self.latent_decode_failed_exc = e
                            self.publish_failed_exc = e
                            self.publish_stop.set()
                            async with self.publish_cv:
                                self.publish_cv.notify_all()
                            logging.exception(
                                "Remote edge async PostVAE hit fatal CUDA error: session=%s job=%s frames=%d face_restore=%.2f background_restore=%.2f",
                                self.cfg.session_id,
                                self.cfg.job_id,
                                int(job.frame_count),
                                float(job.face_restore or 0.0),
                                float(job.background_restore or 0.0),
                            )
                            _exit_after_fatal_cuda_error(
                                context="async_post_vae",
                                session_id=self.cfg.session_id,
                                job_id=self.cfg.job_id,
                                err=e,
                            )
                        if not bool(can_fail_open):
                            raise
                        post_failed = True
                        self.post_vae_disabled_after_error = True
                        logging.exception(
                            "Remote edge PostVAE failed open; disabling restore for this session and continuing raw: session=%s job=%s cuda_poisoned=%d err=%s",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            1 if bool(cuda_poisoned) else 0,
                            e,
                        )
                        try:
                            frames_input = job.frames_tensor
                            if bool(job.defer_stage):
                                stage_started = time.perf_counter()
                                frames_input = await asyncio.to_thread(
                                    self._stabilize_tensor_for_async_post_vae,
                                    frames_input,
                                )
                                post_stage_sec = max(0.0, time.perf_counter() - float(stage_started))
                            frames_tensor = await asyncio.to_thread(
                                self.postprocess_latents_tensor,
                                frames_input,
                                face_restore=0.0,
                                background_restore=0.0,
                                resize_output=True,
                                apply_post_vae=False,
                                input_range=str(job.input_range or "01"),
                            )
                        except Exception as fallback_e:
                            self.latent_decode_failed_exc = fallback_e
                            self.publish_failed_exc = fallback_e
                            self.publish_stop.set()
                            async with self.publish_cv:
                                self.publish_cv.notify_all()
                            logging.exception(
                                "Remote edge raw PostVAE fallback failed: session=%s job=%s cuda_poisoned=%d",
                                self.cfg.session_id,
                                self.cfg.job_id,
                                1 if _is_cuda_context_poisoned_error(fallback_e) else 0,
                            )
                            if _is_cuda_context_poisoned_error(fallback_e):
                                _exit_after_fatal_cuda_error(
                                    context="async_post_vae_raw_fallback",
                                    session_id=self.cfg.session_id,
                                    job_id=self.cfg.job_id,
                                    err=fallback_e,
                                )
                            raise
                    post_sec = max(0.0, time.perf_counter() - float(post_started))
                    frame_count = int(self._tensor_frame_count(frames_tensor))
                    video_enqueue_sec = 0.0
                    if int(frame_count) > 0:
                        video_enqueue_started = time.perf_counter()
                        frames = await asyncio.to_thread(self._tensor_01_to_output_rgb24_frames, frames_tensor)
                        await self.enqueue_video_frames(
                            frames,
                            timestamp_us=job.timestamp_us,
                            segment_id=job.segment_id,
                            segment_kind=job.segment_kind,
                            segment_frames=job.segment_frames,
                            avatar_ref_path=str(job.avatar_ref_path or ""),
                        )
                        video_enqueue_sec = max(0.0, time.perf_counter() - float(video_enqueue_started))
                    self._record_latent_decode_result(
                        queue_wait_sec=float(job.queue_wait_sec) + float(post_queue_wait),
                        decode_sec=float(job.decode_sec) + float(post_sec),
                        video_enqueue_sec=float(video_enqueue_sec),
                        frames=int(frame_count),
                        payload_len=int(job.payload_len),
                    )
                    if bool(_env_flag("POST_VAE_TIMING_LOG", "0")) or post_sec >= 1.0:
                        logging.warning(
                            "Remote edge async PostVAE timing: session=%s job=%s frames=%d post=%.3fs stage=%.3fs queue_wait=%.3fs decode_queue_wait=%.3fs decode_core=%.3fs video_enqueue=%.3fs fail_open=%d deferred_stage=%d post_q=%d",
                            self.cfg.session_id,
                            self.cfg.job_id,
                            int(frame_count),
                            float(post_sec),
                            float(post_stage_sec),
                            float(post_queue_wait),
                            float(job.queue_wait_sec),
                            float(job.decode_sec),
                            float(video_enqueue_sec),
                            1 if bool(post_failed) else 0,
                            1 if bool(job.defer_stage) else 0,
                            int(queue.qsize()),
                        )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if bool(self.post_vae_fail_open) and str(self.cfg.output or "").strip().lower() == "rtmp":
                self.post_vae_disabled_after_error = True
                logging.exception(
                    "Remote edge PostVAE loop failed open; disabling restore for this session and keeping live session alive: session=%s job=%s cuda_poisoned=%d err=%s",
                    self.cfg.session_id,
                    self.cfg.job_id,
                    1 if _is_cuda_context_poisoned_error(e) else 0,
                    e,
                )
                return
            self.latent_decode_failed_exc = e
            self.publish_failed_exc = e
            self.publish_stop.set()
            async with self.publish_cv:
                self.publish_cv.notify_all()
            logging.exception("Remote edge latent postprocess loop failed: session=%s job=%s", self.cfg.session_id, self.cfg.job_id)
            if _is_cuda_context_poisoned_error(e):
                _exit_after_fatal_cuda_error(
                    context="async_post_vae_outer",
                    session_id=self.cfg.session_id,
                    job_id=self.cfg.job_id,
                    err=e,
                )

    async def stop_latent_decode_loop(self) -> None:
        task = self.latent_decode_task
        self.latent_decode_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        post_task = self.latent_postprocess_task
        self.latent_postprocess_task = None
        if post_task is not None:
            post_task.cancel()
            await asyncio.gather(post_task, return_exceptions=True)


def _remote_edge_progress_rejection_is_terminal(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return False
    terminal_markers = (
        "inferencecancelled",
        "inference_cancelled",
        "remote_edge_unavailable",
        "remote_edge_send_failed",
        "reply_cancelled",
        "reply_canceled",
    )
    if any(marker in normalized for marker in terminal_markers):
        return True
    terminal_statuses = (
        "job_status:failed",
        "job_status:cancelled",
        "job_status:canceled",
    )
    if any(status in normalized for status in terminal_statuses):
        return True
    return normalized in {"failed", "cancelled", "canceled"}


def _is_cuda_context_poisoned_error(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "illegal memory access",
            "device-side assert",
            "cuda context is destroyed",
            "cublas_status_not_initialized",
        )
    )


async def _progress_loop(
    session: RemoteEdgeSession,
    stop: asyncio.Event,
    *,
    writer: asyncio.StreamWriter | None = None,
) -> None:
    if not bool(_env_flag("REMOTE_EDGE_API_PROGRESS", "1")):
        return
    if str(session.cfg.output or "").strip().lower() == "file":
        return
    job_id = str(session.cfg.job_id or "").strip()
    if not job_id:
        return
    try:
        client = SmartBlogClient(smartblog_worker_api_key())
    except Exception as e:
        logging.warning("Remote edge SmartBlog progress disabled: %s", e)
        return
    interval = max(0.5, min(60.0, float(_required_positive_float_env("REMOTE_EDGE_API_PROGRESS_SEC"))))
    try:
        while not stop.is_set():
            progress = 1 if int(session.frames) <= 0 else min(95, 5 + int(session.frames // max(1, session.cfg.fps)))
            try:
                ack = await client.progress(
                    job_id=job_id,
                    progress=int(progress),
                    stage="streaming",
                    stage_label=f"Remote edge {str(session.cfg.output or 'livekit').upper()} publish",
                )
                if isinstance(ack, dict) and (ack.get("success") is False or ack.get("ok") is False):
                    reason = smartblog_api_rejection_reason(ack, default="Remote edge progress rejected by API")
                    logging.warning(
                        "Remote edge progress rejected by API: job=%s reason=%s",
                        job_id,
                        reason,
                    )
                    if _remote_edge_progress_rejection_is_terminal(reason):
                        exc = RuntimeError(f"remote edge stopped by SmartBlog API progress rejection: {reason}")
                        session.publish_failed_exc = exc
                        session.publish_stop.set()
                        stop.set()
                        if writer is not None:
                            try:
                                writer.close()
                            except Exception:
                                pass
                        logging.warning(
                            "Remote edge stopping after terminal progress rejection: session=%s job=%s reason=%s",
                            session.cfg.session_id,
                            job_id,
                            reason,
                        )
                        break
            except Exception as e:
                logging.warning("Remote edge progress failed: job=%s err=%s", job_id, e)
            try:
                await asyncio.wait_for(stop.wait(), timeout=float(interval))
            except asyncio.TimeoutError:
                pass
    finally:
        await client.aclose()


def _decode_transport_payload(msg: dict[str, Any], payload: bytes, *, label: str) -> bytes:
    payload_codec = str(msg.get("payload_codec") or msg.get("payload_encoding") or "").strip().lower()
    if payload_codec in {"", "none", "identity", "raw"}:
        return payload
    if payload_codec != "zlib":
        raise RuntimeError(f"unsupported {label} payload_codec: {payload_codec}")
    raw_len = int(msg.get("payload_raw_len") or msg.get("uncompressed_payload_len") or 0)
    if raw_len < 0 or raw_len > int(MAX_PAYLOAD_BYTES):
        raise RuntimeError(f"invalid {label} payload_raw_len={raw_len}")
    try:
        decoded = zlib.decompress(payload)
    except Exception as e:
        raise RuntimeError(f"failed to decompress {label} zlib payload: {e}") from e
    if raw_len and int(len(decoded)) != int(raw_len):
        raise RuntimeError(f"{label} zlib size mismatch: got={len(decoded)} expected={raw_len}")
    return decoded


def _tune_transport_socket(sock_obj: Any, *, role: str) -> None:
    if sock_obj is None:
        return
    try:
        sock_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    sndbuf = max(0, min(256 * 1024 * 1024, _safe_int_env("REMOTE_EDGE_SOCKET_SNDBUF_BYTES", 8 * 1024 * 1024)))
    rcvbuf = max(0, min(256 * 1024 * 1024, _safe_int_env("REMOTE_EDGE_SOCKET_RCVBUF_BYTES", 8 * 1024 * 1024)))
    if int(sndbuf) > 0:
        try:
            sock_obj.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(sndbuf))
        except Exception:
            pass
    if int(rcvbuf) > 0:
        try:
            sock_obj.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        except Exception:
            pass
    try:
        actual_snd = int(sock_obj.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        actual_rcv = int(sock_obj.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        logging.warning(
            "Remote edge socket tuned: role=%s nodelay=1 sndbuf=%d rcvbuf=%d",
            str(role),
            int(actual_snd),
            int(actual_rcv),
        )
    except Exception:
        pass


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    _tune_transport_socket(writer.get_extra_info("socket"), role="edge")
    session: RemoteEdgeSession | None = None
    edge_session_marked = False
    stop_progress = asyncio.Event()
    progress_task: asyncio.Task[Any] | None = None
    try:
        header, _ = await read_message(reader)
        if str(header.get("type") or "") != "hello":
            raise RuntimeError(f"expected hello, got {header}")
        shared_secret = str(os.getenv("REMOTE_EDGE_SHARED_SECRET", "") or "").strip()
        if shared_secret and str(header.get("auth_token") or "") != shared_secret:
            raise RuntimeError("remote edge auth failed")
        rtmp_urls = _rtmp_candidate_urls(str(header.get("rtmp_url") or ""), header.get("rtmp_urls"))
        cfg = EdgeSessionConfig(
            session_id=str(header.get("session_id") or f"remote-{int(time.time() * 1000)}"),
            job_id=str(header.get("job_id") or ""),
            livekit_url=str(header.get("livekit_url") or ""),
            livekit_token=str(header.get("livekit_token") or ""),
            output=str(header.get("output") or ("rtmp" if rtmp_urls else "livekit")).strip().lower(),
            rtmp_url=str(rtmp_urls[0] if rtmp_urls else ""),
            width=max(1, int(header.get("width") or 0)),
            height=max(1, int(header.get("height") or 0)),
            fps=max(1, int(header.get("fps") or 12)),
            sample_rate=max(8000, int(header.get("sample_rate") or 16000)),
            mode=str(header.get("mode") or "rgb24").strip().lower(),
            live_session_id=str(header.get("live_session_id") or header.get("session_id") or ""),
            workspace_id=str(header.get("workspace_id") or ""),
            persona_id=str(header.get("persona_id") or ""),
            rtmp_urls=list(rtmp_urls),
            file_upload_url=str(header.get("file_upload_url") or ""),
            file_upload_path=str(header.get("file_upload_path") or ""),
            file_public_url=str(header.get("file_public_url") or ""),
            file_content_type=str(header.get("file_content_type") or "video/mp4"),
            file_output_fps=max(0, int(header.get("file_output_fps") or 0)),
            file_target_audio_samples=max(0, int(header.get("file_target_audio_samples") or 0)),
            file_target_duration_sec=max(0.0, float(header.get("file_target_duration_sec") or 0.0)),
            watermark_text=normalize_watermark_text(header.get("watermark_text")),
            file_remote_finalizer=(
                None
                if "file_remote_finalizer" not in header
                else str(header.get("file_remote_finalizer") or "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        )
        if cfg.output not in {"livekit", "rtmp", "file"}:
            cfg.output = "livekit"
        if cfg.output == "livekit" and (not cfg.livekit_url or not cfg.livekit_token):
            raise RuntimeError("hello requires livekit_url and livekit_token")
        if cfg.output == "rtmp" and not _rtmp_candidate_urls(str(cfg.rtmp_url or ""), cfg.rtmp_urls):
            raise RuntimeError("hello output=rtmp requires rtmp_url")
        if cfg.output == "file" and not str(cfg.file_upload_url or "").strip():
            raise RuntimeError("hello output=file requires file_upload_url")
        session = RemoteEdgeSession(cfg)
        await _mark_edge_session_active(cfg.session_id, active=True)
        edge_session_marked = True
        await session.start()
        await write_message(writer, "hello.ok", session_id=cfg.session_id)
        if cfg.output != "file":
            progress_task = asyncio.create_task(
                _progress_loop(session, stop_progress, writer=writer),
                name=f"remote-edge-progress-{cfg.session_id}",
            )

        while True:
            msg, payload, read_timing = await read_message_timed(reader)
            message_wait_sec = float(read_timing.get("header_read_sec") or 0.0)
            payload_read_sec = float(read_timing.get("payload_read_sec") or 0.0)
            typ = str(msg.get("type") or "").strip().lower()
            session._record_transport_message(
                typ=typ,
                wire_bytes=int(len(payload)),
                payload_read_sec=float(payload_read_sec),
                message_wait_sec=float(message_wait_sec),
            )
            if typ == "eos":
                break
            if typ == "video.rgb24":
                ts = msg.get("timestamp_us")
                await session.enqueue_video_frames([payload], timestamp_us=(None if ts is None else int(ts)))
                continue
            if typ in {"video.poster.rgb24", "poster.rgb24"}:
                session.set_startup_rgb24(payload)
                logging.warning(
                    "Remote edge startup poster received: session=%s job=%s bytes=%d",
                    cfg.session_id,
                    cfg.job_id,
                    int(len(payload)),
                )
                continue
            if typ == "audio.pcm16le":
                session.queue_pcm16le(
                    payload,
                    sample_rate=int(msg.get("sample_rate") or cfg.sample_rate),
                    segment_id=(None if msg.get("segment_id") is None else str(msg.get("segment_id") or "")),
                    segment_kind=str(msg.get("segment_kind") or ""),
                    segment_frames=(None if msg.get("segment_frames") is None else int(msg.get("segment_frames") or 0)),
                    segment_audible_samples=(
                        None
                        if msg.get("segment_audible_samples") is None
                        else int(msg.get("segment_audible_samples") or 0)
                    ),
                    segment_turn_done=bool(msg.get("segment_turn_done") or False),
                    subtitle_text=str(msg.get("subtitle_text") or ""),
                    subtitle_start_samples=(
                        None
                        if msg.get("subtitle_start_samples") is None
                        else int(msg.get("subtitle_start_samples") or 0)
                    ),
                    subtitle_end_samples=(
                        None
                        if msg.get("subtitle_end_samples") is None
                        else int(msg.get("subtitle_end_samples") or 0)
                    ),
                    subtitle_total_samples=(
                        None
                        if msg.get("subtitle_total_samples") is None
                        else int(msg.get("subtitle_total_samples") or 0)
                    ),
                    subtitle_alignment=(
                        dict(msg.get("subtitle_alignment"))
                        if isinstance(msg.get("subtitle_alignment"), dict)
                        else None
                    ),
                    subtitle_normalized_alignment=(
                        dict(msg.get("subtitle_normalized_alignment"))
                        if isinstance(msg.get("subtitle_normalized_alignment"), dict)
                        else None
                    ),
                    subtitle_alignment_base_samples=(
                        None
                        if msg.get("subtitle_alignment_base_samples") is None
                        else int(msg.get("subtitle_alignment_base_samples") or 0)
                    ),
                )
                continue
            if typ == "video.latents":
                wire_payload_len = int(len(payload))
                transport_decode_started = time.perf_counter()
                payload = _decode_transport_payload(msg, payload, label="video.latents")
                transport_decode_sec = max(0.0, time.perf_counter() - float(transport_decode_started))
                session._record_transport_decode(
                    label="video.latents",
                    raw_bytes=int(len(payload)),
                    decode_sec=float(transport_decode_sec),
                )
                keep_raw = msg.get("keep_last_frames")
                ts = msg.get("timestamp_us")
                await session.enqueue_latents(
                    payload,
                    codec=str(msg.get("codec") or "torch.save"),
                    shape=str(msg.get("shape") or ""),
                    dtype=str(msg.get("dtype") or ""),
                    keep_last_frames=(None if keep_raw is None else int(keep_raw)),
                    reset_vae=bool(msg.get("reset_vae") or msg.get("reset") or False),
                    prime_only=bool(msg.get("prime_only") or False),
                    face_restore=(None if msg.get("face_restore") is None else float(msg.get("face_restore") or 0.0)),
                    background_restore=(
                        None if msg.get("background_restore") is None else float(msg.get("background_restore") or 0.0)
                    ),
                    timestamp_us=(None if ts is None else int(ts)),
                    segment_id=(None if msg.get("segment_id") is None else str(msg.get("segment_id") or "")),
                    segment_kind=str(msg.get("segment_kind") or ""),
                    segment_start_frame=(
                        None if msg.get("segment_start_frame") is None else int(msg.get("segment_start_frame") or 0)
                    ),
                    segment_frames=(None if msg.get("segment_frames") is None else int(msg.get("segment_frames") or 0)),
                    avatar_ref_path=str(msg.get("avatar_ref_path") or msg.get("avatarRefPath") or ""),
                    wire_payload_len=int(wire_payload_len),
                    payload_read_sec=float(payload_read_sec),
                    transport_decode_sec=float(transport_decode_sec),
                )
                continue
            raise RuntimeError(f"unsupported remote message type: {typ}")
    except EOFError:
        pass
    except Exception as e:
        logging.exception("Remote edge session failed: peer=%s err=%s", peer, e)
        try:
            await write_message(writer, "error", error=str(e))
        except Exception:
            pass
    finally:
        stop_progress.set()
        if progress_task is not None:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
        if session is not None:
            session._maybe_log_receive_stats(force=True)
            logging.warning(
                "Remote edge session finishing: session=%s job=%s frames=%d audio_frames=%d pending_blocks=%d buffered_frames=%d pending_audio=%d age=%.1fs",
                session.cfg.session_id,
                session.cfg.job_id,
                int(session.frames),
                int(session.audio_frames),
                int(len(session.pending_blocks)),
                int(session.buffered_video_frames),
                int(len(session.pending_audio)),
                float(time.perf_counter() - session.started_at),
            )
            file_result: dict[str, Any] | None = None
            close_error: BaseException | None = None
            async def _file_progress(payload: dict[str, Any]) -> None:
                await write_message(writer, "file.progress", **dict(payload or {}))

            try:
                output_kind = str(session.cfg.output or "").strip().lower()
                close_timeout = max(
                    5.0,
                    min(600.0, _safe_float_env("REMOTE_EDGE_SESSION_CLOSE_TIMEOUT_SEC", 240.0)),
                )
                if output_kind == "file":
                    configured_file_close = _safe_float_env(
                        "REMOTE_EDGE_FILE_SESSION_CLOSE_TIMEOUT_SEC",
                        0.0,
                    )
                    if float(configured_file_close) > 0.0:
                        close_timeout = max(30.0, min(24 * 3600.0, float(configured_file_close)))
                    else:
                        target_duration = float(session.cfg.file_target_duration_sec or 0.0)
                        if target_duration <= 0.0 and int(session.cfg.file_target_audio_samples or 0) > 0:
                            target_duration = float(session.cfg.file_target_audio_samples) / float(
                                max(1, int(session.cfg.sample_rate or 16000))
                            )
                        if target_duration <= 0.0 and int(session.frames) > 0:
                            target_duration = float(session.frames) / float(max(1, int(session.cfg.fps or 1)))
                        file_drain_timeout = _safe_float_env("REMOTE_EDGE_FILE_DRAIN_TIMEOUT_SEC", 600.0)
                        file_encode_timeout = _safe_float_env(
                            "REMOTE_EDGE_FILE_ENCODE_TIMEOUT_SEC",
                            60.0 + float(max(0.0, target_duration)) * 8.0,
                        )
                        close_timeout = max(
                            float(close_timeout),
                            float(file_drain_timeout)
                            + float(file_encode_timeout)
                            + 60.0,
                        )
                        close_timeout = max(30.0, min(24 * 3600.0, float(close_timeout)))
                file_result = await asyncio.wait_for(
                    session.close(
                        progress_cb=_file_progress if output_kind == "file" else None
                    ),
                    timeout=float(close_timeout),
                )
            except asyncio.TimeoutError as e:
                close_error = e
                logging.exception(
                    "Remote edge session close timed out: session=%s job=%s",
                    session.cfg.session_id,
                    session.cfg.job_id,
                )
                try:
                    session._stop_rtmp()
                except Exception:
                    pass
            except BaseException as e:
                close_error = e
                logging.exception(
                    "Remote edge session close failed: session=%s job=%s",
                    session.cfg.session_id,
                    session.cfg.job_id,
                )
                try:
                    await write_message(writer, "error", error=str(e))
                except Exception:
                    pass
            else:
                if isinstance(file_result, dict):
                    try:
                        await write_message(writer, "file.uploaded", **dict(file_result))
                    except Exception:
                        logging.exception(
                            "Remote edge file result send failed: session=%s job=%s",
                            session.cfg.session_id,
                            session.cfg.job_id,
                        )
            finally:
                if edge_session_marked:
                    await _mark_edge_session_active(session.cfg.session_id, active=False)
                    edge_session_marked = False
            logging.warning(
                "Remote edge session closed: session=%s job=%s",
                session.cfg.session_id,
                session.cfg.job_id,
            )
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def _configured_video_encoders() -> set[str]:
    raw_values = [
        os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "libx264"),
        os.getenv("REMOTE_EDGE_FILE_VIDEO_ENCODER", ""),
        os.getenv("REMOTE_EDGE_CLIP_VIDEO_ENCODER", ""),
    ]
    return {str(value or "").strip().lower() for value in raw_values if str(value or "").strip()}


def _validate_required_video_encoders() -> None:
    if not _env_flag("REMOTE_EDGE_REQUIRE_VIDEO_ENCODER_READY", "1"):
        return
    encoders = _configured_video_encoders()
    if encoders.intersection({"nvenc", "h264_nvenc"}) and not nvenc_runtime_available():
        raise RuntimeError(
            "h264_nvenc runtime unavailable; refusing to start remote edge receiver. "
            "Fix NVIDIA encode runtime or configure the edge profile explicitly."
        )


def _torch_rife_live_env_enabled() -> bool:
    raw = str(os.getenv("REMOTE_EDGE_LIVE_INTERPOLATION", "") or "").strip().lower()
    if raw in {"torch-rife", "rife-torch", "torch", "pytorch", "inmemory", "in-memory"}:
        return True
    return bool(_env_flag("REMOTE_EDGE_LIVE_RIFE_TORCH_ENABLED", "0"))


def _parse_rife_prewarm_size(raw: str) -> tuple[int, int]:
    value = str(raw or "").strip().lower().replace("*", "x").replace(",", "x")
    parts = [part for part in value.split("x") if part]
    if len(parts) != 2:
        return (704, 384)
    try:
        width = max(16, int(parts[0]))
        height = max(16, int(parts[1]))
    except ValueError:
        return (704, 384)
    return (int(width), int(height))


def _prewarm_live_torch_rife_from_env() -> None:
    import torch

    from avalife.remote.torch_rife import get_shared_torch_rife_interpolator

    model_dir = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
    weights_path = str(
        os.getenv("REMOTE_EDGE_TORCH_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors"))
        or os.path.join(model_dir, "flownet.safetensors")
    )
    device = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DEVICE", "cuda:0") or "cuda:0")
    dtype_name = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DTYPE", "float16") or "float16")
    batch_pairs = max(1, _safe_int_env("REMOTE_EDGE_TORCH_RIFE_BATCH_PAIRS", 4))
    width, height = _parse_rife_prewarm_size(str(os.getenv("REMOTE_EDGE_TORCH_RIFE_PREWARM_SIZE", "704x384") or "704x384"))
    dtype = torch.float16 if str(dtype_name).lower() in {"fp16", "float16", "half"} else torch.float32
    torch_device = torch.device(device)
    interpolator = get_shared_torch_rife_interpolator(
        model_dir=model_dir,
        weights_path=weights_path,
        device=device,
        dtype_name=dtype_name,
        batch_pairs=int(batch_pairs),
    )
    dummy = torch.zeros((2, 3, int(height), int(width)), device=torch_device, dtype=dtype)
    _ = interpolator.interpolate_tensor_x2(dummy, target_frames=4)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(device=torch_device)
    logging.warning(
        "Remote edge live torch-RIFE prewarmed: size=%dx%d batch_pairs=%d device=%s",
        int(width),
        int(height),
        int(batch_pairs),
        str(device),
    )


async def amain() -> None:
    parser = argparse.ArgumentParser(description="SmartBlog remote LiveKit edge receiver")
    parser.add_argument("--host", default=os.getenv("REMOTE_EDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("REMOTE_EDGE_PORT", "8787")))
    args = parser.parse_args()

    logging.basicConfig(
        level=str(os.getenv("REMOTE_EDGE_LOG_LEVEL", os.getenv("WORKER_LOG_LEVEL", "INFO"))).upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _validate_required_video_encoders()
    if _env_flag("REMOTE_EDGE_PRELOAD_LATENT_DECODER", "1") and _env_flag("REMOTE_EDGE_LATENT_DECODE", "1"):
        logging.warning("Remote edge preloading shared latent decoder")
        await asyncio.to_thread(prewarm_shared_wan_latent_decoder_from_env)
    if _env_flag("REMOTE_EDGE_PREWARM_LIVE_RIFE", "1") and _torch_rife_live_env_enabled():
        logging.warning("Remote edge prewarming live torch-RIFE")
        await asyncio.to_thread(_prewarm_live_torch_rife_from_env)
    await _write_edge_state(active_sessions=set())
    server = await asyncio.start_server(handle_client, host=str(args.host), port=int(args.port))
    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    logging.warning("Remote edge receiver listening: %s", sockets)
    async with server:
        await server.serve_forever()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
