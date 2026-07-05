from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from avalife.core.upload_retry import put_file_to_signed_url
from avalife.worker.smartblog_api import smartblog_api_base_url, smartblog_worker_api_key


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return float(default)


def edge_clips_enabled() -> bool:
    return bool(_env_flag("REMOTE_EDGE_CLIPS_ENABLED", "0"))


def clip_group_id(segment_id: str | None) -> str:
    value = str(segment_id or "").strip()
    if not value:
        return ""
    return re.sub(r":b\d+$", "", value)


def sanitize_clip_component(value: str | None, *, fallback: str = "clip", max_len: int = 96) -> str:
    text = str(value or "").strip()
    if not text:
        text = str(fallback or "clip")
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    out = re.sub(r"_+", "_", out).strip("._-")
    if not out:
        out = str(fallback or "clip")
    return out[: max(1, int(max_len))]


def _spoken_text_fingerprint(value: str | None) -> str:
    text = str(value or "")
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def _looks_like_updated_cumulative_text(current: str, incoming: str) -> bool:
    current_fp = _spoken_text_fingerprint(current)
    incoming_fp = _spoken_text_fingerprint(incoming)
    if not current_fp or not incoming_fp:
        return False
    if current_fp in incoming_fp:
        return True
    if len(incoming_fp) < len(current_fp):
        return False
    prefix = incoming_fp[: len(current_fp)]
    if prefix == current_fp:
        return True
    mismatches = sum(1 for a, b in zip(current_fp, prefix) if a != b)
    return float(mismatches) / float(max(1, len(current_fp))) <= 0.08


def merge_spoken_text(current: str | None, incoming: str | None) -> str:
    current_s = str(current or "").strip()
    incoming_s = str(incoming or "").strip()
    if not current_s:
        return incoming_s
    if not incoming_s:
        return current_s
    if incoming_s == current_s or incoming_s in current_s:
        return current_s
    if current_s in incoming_s or _looks_like_updated_cumulative_text(current_s, incoming_s):
        return incoming_s

    current_fp = _spoken_text_fingerprint(current_s)
    incoming_fp = _spoken_text_fingerprint(incoming_s)
    if incoming_fp and incoming_fp in current_fp:
        return current_s
    return f"{current_s} {incoming_s}".strip()


@dataclass(frozen=True)
class _EncodedRingSegment:
    index: int
    path: str
    start_sec: float
    end_sec: float


class EncodedClipRing:
    def __init__(self, *, session_id: str, root_dir: str, segment_sec: float, retention_sec: float) -> None:
        self.session_id = sanitize_clip_component(session_id or "session", fallback="session", max_len=96)
        self.root_dir = os.path.abspath(str(root_dir or "/tmp/smartblog-edge-clip-ring"))
        self.segment_sec = max(0.5, min(30.0, float(segment_sec)))
        self.retention_sec = max(float(self.segment_sec) * 3.0, float(retention_sec))
        self.session_dir = os.path.join(self.root_dir, self.session_id)
        self.segment_pattern = os.path.join(self.session_dir, "seg_%09d.ts")
        self._finalized = False
        os.makedirs(self.session_dir, exist_ok=True)

    def mark_finalized(self) -> None:
        self._finalized = True

    def cleanup(self) -> None:
        segments = self._segments()
        if not segments:
            return
        latest = max(int(seg.index) for seg in segments)
        retain_count = max(3, int(float(self.retention_sec) / float(self.segment_sec)) + 3)
        min_keep = int(latest) - int(retain_count)
        if min_keep <= 0:
            return
        for seg in segments:
            if int(seg.index) < int(min_keep):
                try:
                    os.remove(seg.path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logging.debug("Encoded clip ring cleanup failed: path=%s err=%s", seg.path, e)

    def remux_range(
        self,
        *,
        start_sec: float,
        end_sec: float,
        out_mp4: str,
        wait_timeout_sec: float,
    ) -> tuple[float, float, int]:
        segments = self._wait_for_segments(start_sec=start_sec, end_sec=end_sec, timeout_sec=float(wait_timeout_sec))
        if not segments:
            raise RuntimeError("encoded clip ring has no segments for requested range")
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
        list_path = os.path.join(os.path.dirname(os.path.abspath(out_mp4)), "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"file '{self._ffmpeg_concat_quote(seg.path)}'\n")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-movflags",
            "+faststart",
            out_mp4,
        ]
        timeout = max(30.0, _safe_float_env("REMOTE_EDGE_CLIP_RING_REMUX_TIMEOUT_SEC", 120.0))
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout),
            check=False,
        )
        if int(proc.returncode) != 0:
            tail = " ".join(str(proc.stderr or "").strip().splitlines()[-5:])
            raise RuntimeError(f"ffmpeg ring clip remux failed rc={proc.returncode}: {tail or 'no stderr'}")
        if not os.path.exists(out_mp4) or os.path.getsize(out_mp4) <= 0:
            raise RuntimeError("ffmpeg ring clip remux produced no output")
        return float(segments[0].start_sec), float(segments[-1].end_sec), int(len(segments))

    def _wait_for_segments(self, *, start_sec: float, end_sec: float, timeout_sec: float) -> list[_EncodedRingSegment]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        last_error = "encoded clip ring segment not ready"
        while True:
            try:
                return self._select_segments(start_sec=float(start_sec), end_sec=float(end_sec))
            except RuntimeError as e:
                last_error = str(e)
            if time.monotonic() >= deadline:
                raise RuntimeError(last_error)
            time.sleep(0.25)

    def _select_segments(self, *, start_sec: float, end_sec: float) -> list[_EncodedRingSegment]:
        start = max(0.0, float(start_sec))
        end = max(start + 0.05, float(end_sec))
        segments = self._segments()
        if not segments:
            raise RuntimeError("encoded clip ring has no completed segments yet")
        start_idx = int(max(0, int(start // float(self.segment_sec))))
        end_idx = int(max(start_idx, int((end - 0.001) // float(self.segment_sec))))
        max_idx = max(int(seg.index) for seg in segments)
        required_max_idx = int(end_idx)
        if (not self._finalized) and int(max_idx) <= int(required_max_idx):
            raise RuntimeError(
                f"encoded clip ring waiting for segment after end: have_max={max_idx} need_gt={required_max_idx}"
            )
        by_idx = {int(seg.index): seg for seg in segments}
        missing = [idx for idx in range(int(start_idx), int(end_idx) + 1) if idx not in by_idx]
        if missing:
            raise RuntimeError(f"encoded clip ring missing segments: first_missing={missing[0]} count={len(missing)}")
        return [by_idx[idx] for idx in range(int(start_idx), int(end_idx) + 1)]

    def _segments(self) -> list[_EncodedRingSegment]:
        out: list[_EncodedRingSegment] = []
        try:
            names = os.listdir(self.session_dir)
        except FileNotFoundError:
            return []
        for name in names:
            match = re.fullmatch(r"seg_(\d{9})\.ts", str(name or ""))
            if not match:
                continue
            idx = int(match.group(1))
            path = os.path.join(self.session_dir, name)
            if not os.path.isfile(path):
                continue
            start = float(idx) * float(self.segment_sec)
            out.append(_EncodedRingSegment(index=idx, path=path, start_sec=start, end_sec=start + float(self.segment_sec)))
        out.sort(key=lambda seg: int(seg.index))
        return out

    @staticmethod
    def _ffmpeg_concat_quote(value: str) -> str:
        return str(value or "").replace("'", "'\\''")


@dataclass(frozen=True)
class EdgeClipContext:
    session_id: str
    job_id: str
    live_session_id: str = ""
    workspace_id: str = ""
    persona_id: str = ""
    output: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0
    sample_rate: int = 0


@dataclass(frozen=True)
class EdgeClipSegment:
    segment_id: str
    segment_kind: str
    frames: list[bytes]
    audio: bytes = b""
    subtitle_text: str = ""
    start_frame: int | None = None
    end_frame: int | None = None
    audible_samples: int | None = None
    subtitle_start_samples: int | None = None
    subtitle_end_samples: int | None = None
    subtitle_total_samples: int | None = None
    turn_done: bool = False


@dataclass
class _ClipGroup:
    group_id: str
    sequence: int
    segment_kind: str = ""
    subtitle_text: str = ""
    segment_ids: list[str] = field(default_factory=list)
    work_dir: str = ""
    raw_video_path: str = ""
    raw_audio_path: str = ""
    frame_count: int = 0
    audio_bytes: int = 0
    start_frame: int | None = None
    end_frame: int | None = None
    audible_samples: int = 0
    subtitle_start_samples: int | None = None
    subtitle_end_samples: int | None = None
    subtitle_total_samples: int | None = None
    truncated: bool = False

    @property
    def block_index(self) -> int:
        return max(0, int(self.sequence) - 1)


class EdgeClipExporter:
    def __init__(self, context: EdgeClipContext) -> None:
        self.context = context
        self.queue_max = max(1, min(64, _safe_int_env("REMOTE_EDGE_CLIP_QUEUE_MAX", 4)))
        max_sec = max(1.0, _safe_float_env("REMOTE_EDGE_CLIP_MAX_SEC", 600.0))
        self.timeline_fps = max(1, int(context.fps or 1))
        default_max_frames = int(round(float(self.timeline_fps) * float(max_sec)))
        self.max_frames = max(1, _safe_int_env("REMOTE_EDGE_CLIP_MAX_FRAMES", int(default_max_frames)))
        self.min_duration_sec = max(0.0, _safe_float_env("REMOTE_EDGE_CLIP_MIN_SEC", 0.35))
        self.require_audio = bool(_env_flag("REMOTE_EDGE_CLIP_REQUIRE_AUDIO", "1"))
        self.upload_enabled = bool(_env_flag("REMOTE_EDGE_CLIP_UPLOAD", "1"))
        self.keep_local = bool(_env_flag("REMOTE_EDGE_CLIP_KEEP_LOCAL", "0"))
        self.keep_failed = bool(_env_flag("REMOTE_EDGE_CLIP_KEEP_FAILED", "1"))
        self.group_by_turn = bool(_env_flag("REMOTE_EDGE_CLIP_GROUP_BY_TURN", "1"))
        self.output_fps = max(1, _safe_int_env("REMOTE_EDGE_CLIP_OUTPUT_FPS", _safe_int_env("LIVE_CHANNEL_RTMP_OUTPUT_FPS", 30)))
        self.output_dir = os.path.abspath(
            str(os.getenv("REMOTE_EDGE_CLIP_DIR", "/tmp/smartblog-edge-clips") or "/tmp/smartblog-edge-clips")
        )
        source = str(os.getenv("REMOTE_EDGE_CLIP_SOURCE", "raw") or "raw").strip().lower()
        self.clip_source = source if source in {"raw", "rtmp_ring", "encoded_ring"} else "raw"
        self.encoded_ring: EncodedClipRing | None = None
        if self.clip_source in {"rtmp_ring", "encoded_ring"} and str(context.output or "").strip().lower() == "rtmp":
            self.encoded_ring = EncodedClipRing(
                session_id=str(context.session_id or context.job_id or "session"),
                root_dir=str(os.getenv("REMOTE_EDGE_CLIP_RING_DIR", "/tmp/smartblog-edge-clip-ring") or "/tmp/smartblog-edge-clip-ring"),
                segment_sec=_safe_float_env("REMOTE_EDGE_CLIP_RING_SEGMENT_SEC", 2.0),
                retention_sec=_safe_float_env("REMOTE_EDGE_CLIP_RING_RETENTION_SEC", 900.0),
            )
            # Encoded ring timestamps are based on the actual edge output
            # timeline. The edge receiver passes RTMP pipe fps here, not the
            # original producer fps, so clip ranges match the segment ring.
        self.allowed_kinds = {
            item.strip().lower()
            for item in str(os.getenv("REMOTE_EDGE_CLIP_KINDS", "speech") or "speech").split(",")
            if item.strip()
        }
        self._queue: queue.Queue[_ClipGroup | None] = queue.Queue(maxsize=int(self.queue_max))
        self._current: _ClipGroup | None = None
        self._sequence = 0
        self._closed = False
        self._active_exports = 0
        self._active_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"remote-edge-clip-export-{sanitize_clip_component(context.session_id)}",
            daemon=True,
        )
        self._thread.start()
        logging.warning(
            "Remote edge clip exporter enabled: session=%s job=%s output_dir=%s queue_max=%d max_frames=%d group_by_turn=%d upload=%d source=%s ring=%s",
            str(context.session_id or "-"),
            str(context.job_id or "-"),
            str(self.output_dir),
            int(self.queue_max),
            int(self.max_frames),
            int(bool(self.group_by_turn)),
            int(bool(self.upload_enabled)),
            str(self.clip_source),
            "-" if self.encoded_ring is None else str(self.encoded_ring.segment_pattern),
        )

    def rtmp_segment_pattern(self) -> str:
        ring = getattr(self, "encoded_ring", None)
        return "" if ring is None else str(ring.segment_pattern)

    def rtmp_segment_time_sec(self) -> float:
        ring = getattr(self, "encoded_ring", None)
        return 0.0 if ring is None else float(ring.segment_sec)

    def mark_encoded_ring_finalized(self) -> None:
        ring = getattr(self, "encoded_ring", None)
        if ring is not None:
            ring.mark_finalized()

    def record_segment(self, segment: EdgeClipSegment) -> None:
        if self._closed:
            return
        ring = getattr(self, "encoded_ring", None)
        if ring is not None:
            ring.cleanup()
        group_id = clip_group_id(segment.segment_id)
        if not group_id:
            return
        current = self._current
        if current is not None and (not bool(self.group_by_turn)) and current.group_id != group_id:
            self._submit_current()
            current = None
        kind = str(segment.segment_kind or "").strip().lower()
        if self.allowed_kinds and kind and (not self._kind_allowed(kind)):
            return
        if self.require_audio and not bytes(segment.audio or b""):
            return
        if not segment.frames:
            return
        if current is None:
            self._sequence += 1
            current = _ClipGroup(group_id=str(group_id), sequence=int(self._sequence), segment_kind=str(kind or "speech"))
            if getattr(self, "encoded_ring", None) is None:
                self._ensure_group_storage(current)
            self._current = current
        self._append_to_group(current, segment)
        if bool(self.group_by_turn) and (bool(segment.turn_done) or bool(current.truncated)):
            self._submit_current()

    def _kind_allowed(self, kind: str) -> bool:
        kind_s = str(kind or "").strip().lower()
        if not self.allowed_kinds:
            return True
        if kind_s in self.allowed_kinds:
            return True
        return bool("speech" in self.allowed_kinds and kind_s.startswith("speech"))

    def close(self, *, timeout_sec: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._submit_current()
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass
        timeout = _safe_float_env("REMOTE_EDGE_CLIP_CLOSE_TIMEOUT_SEC", 300.0) if timeout_sec is None else float(timeout_sec)
        if timeout > 0.0 and self._thread.is_alive():
            self._thread.join(timeout=float(timeout))
            if self._thread.is_alive():
                logging.warning(
                    "Remote edge clip exporter did not stop before timeout: session=%s job=%s queued=%d",
                    self.context.session_id,
                    self.context.job_id,
                    int(self._queue.qsize()),
                )

    def _append_to_group(self, group: _ClipGroup, segment: EdgeClipSegment) -> None:
        remaining = int(self.max_frames) - int(group.frame_count)
        frames = list(segment.frames)
        if remaining <= 0:
            group.truncated = True
            return
        if len(frames) > remaining:
            frames = frames[:remaining]
            group.truncated = True
        if getattr(self, "encoded_ring", None) is None:
            self._ensure_group_storage(group)
            expected = int(max(1, int(self.context.width)) * max(1, int(self.context.height)) * 3)
            with open(group.raw_video_path, "ab") as f:
                for frame in frames:
                    frame_b = bytes(frame)
                    if len(frame_b) != expected:
                        raise RuntimeError(f"clip frame size mismatch: got={len(frame_b)} expected={expected}")
                    f.write(frame_b)
        group.frame_count += int(len(frames))
        audio_b = bytes(segment.audio or b"")
        if audio_b:
            if len(frames) < len(segment.frames):
                sample_rate = max(1, int(self.context.sample_rate or 16000))
                fps = max(1, int(self.context.fps or 1))
                keep_samples = int(round(float(len(frames)) * float(sample_rate) / float(fps)))
                audio_b = audio_b[: int(max(0, keep_samples) * 2)]
            if audio_b:
                if getattr(self, "encoded_ring", None) is None:
                    self._ensure_group_storage(group)
                    with open(group.raw_audio_path, "ab") as f:
                        f.write(audio_b)
                group.audio_bytes += int(len(audio_b))
        if str(segment.segment_id or "") and str(segment.segment_id) not in group.segment_ids:
            group.segment_ids.append(str(segment.segment_id))
        if str(segment.segment_kind or "").strip():
            group.segment_kind = str(segment.segment_kind).strip().lower()
        subtitle = str(segment.subtitle_text or "").strip()
        if subtitle:
            group.subtitle_text = merge_spoken_text(group.subtitle_text, subtitle)
        if segment.start_frame is not None:
            start = int(segment.start_frame)
            group.start_frame = start if group.start_frame is None else min(int(group.start_frame), start)
        if segment.end_frame is not None:
            end = int(segment.end_frame)
            group.end_frame = end if group.end_frame is None else max(int(group.end_frame), end)
        if segment.audible_samples is not None:
            group.audible_samples += max(0, int(segment.audible_samples))
        for attr in ("subtitle_start_samples", "subtitle_end_samples", "subtitle_total_samples"):
            value = getattr(segment, attr)
            if value is None:
                continue
            value_i = max(0, int(value))
            old = getattr(group, attr)
            if old is None:
                setattr(group, attr, value_i)
            elif attr == "subtitle_start_samples":
                setattr(group, attr, min(int(old), value_i))
            else:
                setattr(group, attr, max(int(old), value_i))

    def _submit_current(self) -> None:
        group = self._current
        self._current = None
        if group is None or group.frame_count <= 0:
            self._discard_group(group)
            return
        fps = max(1, int(self.timeline_fps or self.context.fps or 1))
        if (float(group.frame_count) / float(fps)) < float(self.min_duration_sec):
            self._discard_group(group)
            return
        try:
            self._queue.put_nowait(group)
        except queue.Full:
            dropped: _ClipGroup | None = None
            try:
                dropped = self._queue.get_nowait()
                self._discard_group(dropped)
                try:
                    self._queue.task_done()
                except Exception:
                    pass
                self._queue.put_nowait(group)
                logging.warning(
                    "Remote edge clip export queue full; dropped older clip and queued latest: session=%s job=%s dropped_group=%s queued_group=%s queued_frames=%d",
                    self.context.session_id,
                    self.context.job_id,
                    "-" if dropped is None else str(dropped.group_id),
                    group.group_id,
                    int(group.frame_count),
                )
                return
            except Exception:
                if dropped is group:
                    dropped = None
            logging.warning(
                "Remote edge clip export queue full; dropping latest clip: session=%s job=%s group=%s frames=%d",
                self.context.session_id,
                self.context.job_id,
                group.group_id,
                int(group.frame_count),
            )
            self._discard_group(group)

    def _worker_loop(self) -> None:
        while True:
            group = self._queue.get()
            try:
                if group is None:
                    return
                self._set_active_export_delta(1)
                self._process_group(group)
            except Exception:
                logging.exception(
                    "Remote edge clip export failed: session=%s job=%s group=%s",
                    self.context.session_id,
                    self.context.job_id,
                    "-" if group is None else group.group_id,
                )
            finally:
                if group is not None:
                    self._set_active_export_delta(-1)
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def _set_active_export_delta(self, delta: int) -> None:
        with self._active_lock:
            self._active_exports = max(0, int(self._active_exports) + int(delta))

    def _active_export_count(self) -> int:
        with self._active_lock:
            return max(0, int(self._active_exports))

    def _process_group(self, group: _ClipGroup) -> None:
        if getattr(self, "encoded_ring", None) is None:
            self._ensure_group_storage(group)
        else:
            self._ensure_group_export_dir(group)
        work_dir = str(group.work_dir)
        mp4_path = os.path.join(work_dir, "clip.mp4")
        failed = False
        upload_result: dict[str, Any] = {}
        clip_id = ""
        try:
            if self.upload_enabled:
                upload_result = self._request_clip_upload_url(group)
                clip_id = str(upload_result.get("clip_id") or "").strip()
            ring_segment_count = 0
            ring_start_sec = None
            ring_end_sec = None
            if getattr(self, "encoded_ring", None) is not None:
                ring_start_sec, ring_end_sec, ring_segment_count = self._remux_group_from_ring(group, mp4_path)
            else:
                self._encode_group(group, mp4_path)
            if self.upload_enabled:
                self._put_clip(upload_result, mp4_path)
            if upload_result:
                clip_body = self._register_clip(group, upload_result)
                self._add_clip_memory(group, clip_body=clip_body)
            logging.warning(
                "Remote edge clip exported: session=%s job=%s group=%s block_index=%d clip_id=%s frames=%d path=%s storage=%s source=%s ring_segments=%d ring_range=%s",
                self.context.session_id,
                self.context.job_id,
                group.group_id,
                int(group.block_index),
                str(clip_id or "-"),
                int(group.frame_count),
                str(mp4_path),
                str(((upload_result.get("upload") if isinstance(upload_result.get("upload"), dict) else {}) or {}).get("path") or "-"),
                str(self.clip_source),
                int(ring_segment_count),
                "-" if ring_start_sec is None or ring_end_sec is None else f"{float(ring_start_sec):.3f}-{float(ring_end_sec):.3f}",
            )
        except Exception as e:
            failed = True
            if clip_id:
                self._register_clip_failure(clip_id=clip_id, block_index=int(group.block_index), error=str(e))
            raise
        finally:
            keep = bool(self.keep_local or (failed and self.keep_failed))
            if not keep:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _ensure_group_export_dir(self, group: _ClipGroup) -> None:
        if str(group.work_dir or "").strip():
            return
        group.work_dir = tempfile.mkdtemp(
            prefix=f"clip_{sanitize_clip_component(group.group_id, max_len=48)}_",
            dir=self._session_output_dir(),
        )

    def _ensure_group_storage(self, group: _ClipGroup) -> None:
        if str(group.work_dir or "").strip():
            return
        group.work_dir = tempfile.mkdtemp(
            prefix=f"clip_{sanitize_clip_component(group.group_id, max_len=48)}_",
            dir=self._session_output_dir(),
        )
        group.raw_video_path = os.path.join(group.work_dir, "video.rgb")
        group.raw_audio_path = os.path.join(group.work_dir, "audio.s16le")

    @staticmethod
    def _discard_group(group: _ClipGroup | None) -> None:
        if group is None:
            return
        work_dir = str(getattr(group, "work_dir", "") or "").strip()
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _session_output_dir(self) -> str:
        session_dir = os.path.join(
            self.output_dir,
            sanitize_clip_component(self.context.session_id or self.context.job_id or "session"),
        )
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _encode_group(self, group: _ClipGroup, mp4_path: str) -> None:
        width = max(1, int(self.context.width))
        height = max(1, int(self.context.height))
        fps = max(1, int(self.context.fps))
        sample_rate = max(1, int(self.context.sample_rate or 16000))
        raw_video = str(group.raw_video_path or os.path.join(os.path.dirname(mp4_path), "video.rgb"))
        raw_audio = str(group.raw_audio_path or os.path.join(os.path.dirname(mp4_path), "audio.s16le"))
        expected_video_bytes = int(width * height * 3 * int(group.frame_count))
        if not os.path.exists(raw_video) or os.path.getsize(raw_video) != expected_video_bytes:
            raise RuntimeError(
                f"clip raw video size mismatch: got={0 if not os.path.exists(raw_video) else os.path.getsize(raw_video)} expected={expected_video_bytes}"
            )
        samples = int(round(float(group.frame_count) * float(sample_rate) / float(fps)))
        expected_audio_bytes = int(max(0, samples) * 2)
        if not os.path.exists(raw_audio) or os.path.getsize(raw_audio) <= 0:
            with open(raw_audio, "wb") as f:
                f.write(b"\x00" * int(expected_audio_bytes))
        else:
            actual_audio_bytes = int(os.path.getsize(raw_audio))
            if actual_audio_bytes < int(expected_audio_bytes):
                with open(raw_audio, "ab") as f:
                    f.write(b"\x00" * int(expected_audio_bytes - actual_audio_bytes))
            elif actual_audio_bytes > int(expected_audio_bytes):
                with open(raw_audio, "rb+") as f:
                    f.truncate(int(expected_audio_bytes))
        cmd = self._ffmpeg_encode_cmd(
            width=width,
            height=height,
            fps=fps,
            sample_rate=sample_rate,
            raw_video=raw_video,
            raw_audio=raw_audio,
            mp4_path=mp4_path,
        )
        duration_sec = float(group.frame_count) / float(max(1, int(fps)))
        default_timeout = max(180.0, 60.0 + float(duration_sec) * 6.0)
        timeout = max(10.0, _safe_float_env("REMOTE_EDGE_CLIP_ENCODE_TIMEOUT_SEC", float(default_timeout)))
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout),
            check=False,
        )
        if int(proc.returncode) != 0:
            tail = " ".join(str(proc.stderr or "").strip().splitlines()[-5:])
            raise RuntimeError(f"ffmpeg clip encode failed rc={proc.returncode}: {tail or 'no stderr'}")
        if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) <= 0:
            raise RuntimeError("ffmpeg clip encode produced no output")

    def _remux_group_from_ring(self, group: _ClipGroup, mp4_path: str) -> tuple[float, float, int]:
        ring = getattr(self, "encoded_ring", None)
        if ring is None:
            raise RuntimeError("encoded ring is not configured")
        start_sec, end_sec = self._clip_seconds(group)
        wait_timeout = _safe_float_env("REMOTE_EDGE_CLIP_RING_WAIT_TIMEOUT_SEC", 120.0)
        ring_start, ring_end, segment_count = ring.remux_range(
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            out_mp4=str(mp4_path),
            wait_timeout_sec=float(wait_timeout),
        )
        ring.cleanup()
        return float(ring_start), float(ring_end), int(segment_count)

    def _ffmpeg_encode_cmd(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        sample_rate: int,
        raw_video: str,
        raw_audio: str,
        mp4_path: str,
    ) -> list[str]:
        encoder = str(
            os.getenv("REMOTE_EDGE_CLIP_VIDEO_ENCODER", os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "h264_nvenc"))
            or "h264_nvenc"
        ).strip().lower()
        output_fps = max(1, int(self.output_fps))
        gop = max(1, int(round(float(output_fps) * _safe_float_env("REMOTE_EDGE_CLIP_KEYFRAME_SEC", 2.0))))
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
        if int(output_fps) != int(fps):
            cmd.extend(["-vf", f"fps={int(output_fps)}"])
        cmd.extend(["-r", str(int(output_fps))])
        if encoder in {"nvenc", "h264_nvenc"}:
            bitrate = str(os.getenv("REMOTE_EDGE_CLIP_VIDEO_BITRATE", "3500k") or "3500k")
            cmd.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-profile:v",
                    "high",
                    "-preset",
                    str(os.getenv("REMOTE_EDGE_CLIP_NVENC_PRESET", "p4") or "p4"),
                    "-rc",
                    "vbr",
                    "-cq",
                    str(os.getenv("REMOTE_EDGE_CLIP_NVENC_CQ", "22") or "22"),
                    "-b:v",
                    bitrate,
                    "-maxrate",
                    str(os.getenv("REMOTE_EDGE_CLIP_VIDEO_MAXRATE", bitrate) or bitrate),
                    "-bufsize",
                    str(os.getenv("REMOTE_EDGE_CLIP_VIDEO_BUFSIZE", "7000k") or "7000k"),
                    "-g",
                    str(gop),
                    "-keyint_min",
                    str(gop),
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "high",
                    "-preset",
                    str(os.getenv("REMOTE_EDGE_CLIP_X264_PRESET", "veryfast") or "veryfast"),
                    "-crf",
                    str(os.getenv("REMOTE_EDGE_CLIP_X264_CRF", "21") or "21"),
                    "-g",
                    str(gop),
                    "-keyint_min",
                    str(gop),
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                str(os.getenv("REMOTE_EDGE_CLIP_AUDIO_BITRATE", "128k") or "128k"),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                mp4_path,
            ]
        )
        return cmd

    def _request_clip_upload_url(self, group: _ClipGroup) -> dict[str, Any]:
        import requests

        api = smartblog_api_base_url()
        token = smartblog_worker_api_key()
        live_session_id = str(self.context.live_session_id or self.context.session_id or "").strip()
        if not live_session_id:
            raise RuntimeError("live_session_id is required for clip_upload_url")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        resp = requests.post(
            api,
            headers=headers,
            json={
                "action": "clip_upload_url",
                "live_session_id": live_session_id,
                "block_index": int(group.block_index),
            },
            timeout=(10.0, 60.0),
        )
        resp.raise_for_status()
        obj = resp.json()
        if not isinstance(obj, dict):
            raise RuntimeError("clip_upload_url returned non-object")
        if obj.get("success") is False or obj.get("ok") is False:
            raise RuntimeError(str(obj.get("reason") or obj.get("error") or "clip_upload_url rejected"))
        clip_id = str(obj.get("clip_id") or "").strip()
        upload = obj.get("upload") if isinstance(obj.get("upload"), dict) else {}
        signed_url = str(upload.get("signed_url") or "").strip()
        if not clip_id:
            raise RuntimeError("clip_upload_url returned no clip_id")
        if not signed_url:
            raise RuntimeError("clip_upload_url returned no upload.signed_url")
        return dict(obj)

    def _put_clip(self, upload_result: dict[str, Any], mp4_path: str) -> None:
        upload = upload_result.get("upload") if isinstance(upload_result.get("upload"), dict) else {}
        signed_url = str(upload.get("signed_url") or "").strip()
        if not signed_url:
            raise RuntimeError("clip_upload_url result has no upload.signed_url")
        put_file_to_signed_url(
            signed_url=str(signed_url),
            path=str(mp4_path),
            content_type="video/mp4",
            connect_timeout=20.0,
            read_timeout=1800.0,
            env_prefix="SMARTBLOG_CLIP_UPLOAD",
            log_prefix="clip-signed-upload",
        )

    def _clip_registration_body(self, group: _ClipGroup, upload_result: dict[str, Any]) -> dict[str, Any]:
        start_seconds, end_seconds = self._clip_seconds(group)
        return {
            "action": "clip",
            "clip_id": str(upload_result.get("clip_id") or ""),
            "block_index": int(group.block_index),
            "clip_start": float(start_seconds),
            "clip_end": float(end_seconds),
            "block_start_at": float(start_seconds),
            "block_end_at": float(end_seconds),
            "spoken_text": str(group.subtitle_text or ""),
            "comments": [],
            "job_id": str(self.context.job_id or ""),
        }

    def _register_clip(self, group: _ClipGroup, upload_result: dict[str, Any]) -> dict[str, Any]:
        import requests

        body = self._clip_registration_body(group, upload_result)
        resp = requests.post(
            smartblog_api_base_url(),
            headers={"Authorization": f"Bearer {smartblog_worker_api_key()}", "Content-Type": "application/json"},
            json=body,
            timeout=(10.0, 60.0),
        )
        resp.raise_for_status()
        obj = resp.json()
        if isinstance(obj, dict) and (obj.get("success") is False or obj.get("ok") is False):
            raise RuntimeError(str(obj.get("reason") or obj.get("error") or "clip rejected"))
        return dict(body)

    def _add_clip_memory(self, group: _ClipGroup, *, clip_body: dict[str, Any]) -> None:
        import requests

        persona_id = str(self.context.persona_id or "").strip()
        summary = str(clip_body.get("spoken_text") or "").strip()
        if not persona_id or not summary:
            return
        clip_id = str(clip_body.get("clip_id") or "").strip()
        source_id = clip_id or f"{self.context.live_session_id or self.context.session_id or self.context.job_id}:{int(group.block_index)}"
        try:
            resp = requests.post(
                smartblog_api_base_url(),
                headers={"Authorization": f"Bearer {smartblog_worker_api_key()}", "Content-Type": "application/json"},
                json={
                    "action": "add_memory",
                    "persona_id": persona_id,
                    "summary": summary,
                    "source_type": "live",
                    "source_id": str(source_id),
                },
                timeout=(10.0, 60.0),
            )
            resp.raise_for_status()
            obj = resp.json()
            if isinstance(obj, dict) and (obj.get("success") is False or obj.get("ok") is False):
                raise RuntimeError(str(obj.get("reason") or obj.get("error") or "add_memory rejected"))
        except Exception as e:
            logging.warning(
                "Remote edge clip memory add failed: session=%s job=%s clip_id=%s block_index=%d persona=%s err=%s",
                self.context.session_id,
                self.context.job_id,
                str(clip_id or "-"),
                int(group.block_index),
                persona_id,
                e,
            )

    def _register_clip_failure(self, *, clip_id: str, block_index: int, error: str) -> None:
        import requests

        try:
            resp = requests.post(
                smartblog_api_base_url(),
                headers={"Authorization": f"Bearer {smartblog_worker_api_key()}", "Content-Type": "application/json"},
                json={
                    "action": "clip",
                    "clip_id": str(clip_id),
                    "block_index": int(block_index),
                    "error": str(error or "clip export failed")[:1000],
                    "job_id": str(self.context.job_id or ""),
                },
                timeout=(10.0, 60.0),
            )
            resp.raise_for_status()
        except Exception:
            logging.exception(
                "Remote edge clip failure registration failed: session=%s job=%s clip_id=%s block_index=%d",
                self.context.session_id,
                self.context.job_id,
                str(clip_id),
                int(block_index),
            )

    def _clip_seconds(self, group: _ClipGroup) -> tuple[float, float]:
        fps = max(1, int(self.timeline_fps or self.context.fps or 1))
        start_frame = 0 if group.start_frame is None else int(group.start_frame)
        end_frame = int(start_frame) + int(group.frame_count) if group.end_frame is None else int(group.end_frame)
        return float(start_frame) / float(fps), float(end_frame) / float(fps)

    def debug_snapshot(self) -> dict[str, Any]:
        current = self._current
        queued = int(self._queue.qsize())
        active = int(self._active_export_count())
        return {
            "enabled": True,
            "busy": bool(current is not None or queued > 0 or active > 0),
            "queued": int(queued),
            "active_exports": int(active),
            "closed": bool(self._closed),
            "current_group_id": "" if current is None else str(current.group_id),
            "current_frames": 0 if current is None else int(current.frame_count),
            "sequence": int(self._sequence),
        }
