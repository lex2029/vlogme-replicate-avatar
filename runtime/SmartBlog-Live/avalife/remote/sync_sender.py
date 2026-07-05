from __future__ import annotations

import io
import json
import logging
import os
import queue
import socket
import threading
import time
import wave
import zlib
from dataclasses import dataclass, field
from typing import Any

from .protocol import MAX_HEADER_BYTES


class SyncRemoteStreamSender:
    """Blocking sender for model hot paths that cannot run an asyncio loop."""

    def __init__(self, *, host: str, port: int, timeout_sec: float = 5.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(max(0.25, float(timeout_sec)))
        self.sock: socket.socket | None = None
        self.output = ""
        self.file_progress_path = ""
        self._transport_log_last = 0.0
        self._transport_raw_bytes = 0
        self._transport_wire_bytes = 0
        self._latent_zlib_disabled = False
        self._latent_zlib_trials = 0
        self._latent_zlib_hits = 0
        self._write_stats_last = 0.0
        self._write_stats_messages = 0
        self._write_stats_payload_messages = 0
        self._write_stats_wire_bytes = 0
        self._write_stats_header_sec = 0.0
        self._write_stats_header_max_sec = 0.0
        self._write_stats_payload_sec = 0.0
        self._write_stats_payload_max_sec = 0.0
        self._write_stats_total_sec = 0.0
        self._write_stats_total_max_sec = 0.0

    def connect(self) -> None:
        sock = socket.create_connection((self.host, int(self.port)), timeout=float(self.timeout_sec))
        _tune_socket(sock, role="producer")
        sock.settimeout(float(self.timeout_sec))
        self.sock = sock

    def set_timeout(self, timeout_sec: float) -> None:
        sock = self._sock()
        sock.settimeout(float(max(0.25, float(timeout_sec))))

    def close(self, *, send_eos: bool = True) -> dict[str, Any] | None:
        sock = self.sock
        if sock is None:
            return None
        result: dict[str, Any] | None = None
        close_error: BaseException | None = None
        if bool(send_eos):
            try:
                self.write_message("eos")
                if str(self.output or "").strip().lower() == "file":
                    sock.settimeout(
                        _float_env("REMOTE_EDGE_FILE_RESULT_TIMEOUT_SEC", 1800.0, low=1.0, high=7200.0)
                    )
                    result = self._read_file_result()
            except BaseException as e:
                close_error = e
        self.sock = None
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        if close_error is not None and str(self.output or "").strip().lower() == "file":
            raise close_error
        return result

    def abort(self) -> None:
        self.close(send_eos=False)

    def hello(
        self,
        *,
        session_id: str,
        job_id: str,
        livekit_url: str,
        livekit_token: str,
        width: int,
        height: int,
        fps: int,
        sample_rate: int,
        mode: str,
        output: str = "livekit",
        rtmp_url: str = "",
        rtmp_urls: list[str] | tuple[str, ...] | None = None,
        live_session_id: str = "",
        workspace_id: str = "",
        persona_id: str = "",
        auth_token: str = "",
        file_upload_url: str = "",
        file_upload_path: str = "",
        file_public_url: str = "",
        file_content_type: str = "video/mp4",
        file_progress_path: str = "",
        file_output_fps: int = 0,
        file_target_audio_samples: int = 0,
        file_target_duration_sec: float = 0.0,
        watermark_text: str = "",
        file_remote_finalizer: bool | None = None,
    ) -> dict[str, Any]:
        rtmp_url_list = [str(item or "").strip() for item in (rtmp_urls or []) if str(item or "").strip()]
        if rtmp_url and str(rtmp_url).strip() not in rtmp_url_list:
            rtmp_url_list.insert(0, str(rtmp_url).strip())
        self.output = str(output or "livekit").strip().lower()
        self.file_progress_path = str(file_progress_path or "").strip()
        fields: dict[str, Any] = {
            "session_id": str(session_id),
            "job_id": str(job_id),
            "livekit_url": str(livekit_url),
            "livekit_token": str(livekit_token),
            "output": str(output or "livekit"),
            "rtmp_url": str(rtmp_url or ""),
            "rtmp_urls": list(rtmp_url_list),
            "live_session_id": str(live_session_id or ""),
            "workspace_id": str(workspace_id or ""),
            "persona_id": str(persona_id or ""),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "sample_rate": int(sample_rate),
            "mode": str(mode),
            "auth_token": str(auth_token or ""),
            "file_upload_url": str(file_upload_url or ""),
            "file_upload_path": str(file_upload_path or ""),
            "file_public_url": str(file_public_url or ""),
            "file_content_type": str(file_content_type or "video/mp4"),
            "file_output_fps": int(file_output_fps or 0),
            "file_target_audio_samples": int(max(0, int(file_target_audio_samples or 0))),
            "file_target_duration_sec": float(max(0.0, float(file_target_duration_sec or 0.0))),
            "watermark_text": str(watermark_text or ""),
        }
        if file_remote_finalizer is not None:
            fields["file_remote_finalizer"] = bool(file_remote_finalizer)
        self.write_message("hello", **fields)
        resp, _ = self.read_message()
        if str(resp.get("type") or "") != "hello.ok":
            raise RuntimeError(f"remote edge rejected hello: {resp}")
        return resp

    def _read_file_result(self) -> dict[str, Any]:
        while True:
            header, _payload = self.read_message()
            typ = str(header.get("type") or "").strip().lower()
            if typ in {"file.uploaded", "file.result"}:
                self._write_file_progress({"phase": "done", "progress": 1.0, **dict(header)})
                return dict(header)
            if typ == "file.progress":
                self._write_file_progress(dict(header))
                continue
            if typ == "error":
                raise RuntimeError(str(header.get("error") or "remote edge file output failed"))

    def _write_file_progress(self, payload: dict[str, Any]) -> None:
        path = str(self.file_progress_path or "").strip()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            obj = dict(payload or {})
            obj["updated_at_ms"] = int(time.time() * 1000.0)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            logging.exception("Remote edge file progress write failed: %s", path)

    def note_file_progress(self, *, phase: str, progress: float, **fields: Any) -> None:
        self._write_file_progress(
            {
                "phase": str(phase or ""),
                "progress": float(max(0.0, min(1.0, float(progress)))),
                **dict(fields or {}),
            }
        )

    def send_latents(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        codec: str = "torch.save",
        shape: str = "",
        dtype: str = "",
        timestamp_us: int | None = None,
        keep_last_frames: int | None = None,
        reset_vae: bool = False,
        prime_only: bool = False,
        face_restore: float | None = None,
        background_restore: float | None = None,
        segment_id: str | None = None,
        segment_kind: str | None = None,
        segment_start_frame: int | None = None,
        segment_frames: int | None = None,
        avatar_ref_path: str | None = None,
    ) -> None:
        payload_to_send, transport_fields = self._prepare_latent_transport_payload(payload)
        self.write_message(
            "video.latents",
            payload_to_send,
            codec=str(codec),
            shape=str(shape),
            dtype=str(dtype),
            timestamp_us=None if timestamp_us is None else int(timestamp_us),
            keep_last_frames=None if keep_last_frames is None else int(keep_last_frames),
            reset_vae=bool(reset_vae),
            prime_only=bool(prime_only),
            face_restore=None if face_restore is None else float(face_restore),
            background_restore=None if background_restore is None else float(background_restore),
            segment_id=None if segment_id is None else str(segment_id),
            segment_kind=None if segment_kind is None else str(segment_kind),
            segment_start_frame=None if segment_start_frame is None else int(segment_start_frame),
            segment_frames=None if segment_frames is None else int(segment_frames),
            avatar_ref_path=None if avatar_ref_path is None else str(avatar_ref_path),
            **transport_fields,
        )

    def _prepare_latent_transport_payload(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> tuple[bytes | bytearray | memoryview, dict[str, Any]]:
        payload_view = memoryview(payload).cast("B")
        raw_len = int(payload_view.nbytes)
        if raw_len <= 0:
            return payload, {}
        enabled = _flag_env("REMOTE_EDGE_LATENT_ZLIB", "0") or _flag_env(
            "REMOTE_EDGE_LATENT_TRANSPORT_ZLIB",
            "0",
        )
        if not bool(enabled) or bool(self._latent_zlib_disabled):
            self._note_latent_transport(raw_len=raw_len, wire_len=raw_len, codec="raw")
            return payload, {}
        min_bytes = _int_env("REMOTE_EDGE_LATENT_ZLIB_MIN_BYTES", 64 * 1024, low=0, high=512 * 1024 * 1024)
        if raw_len < int(min_bytes):
            self._note_latent_transport(raw_len=raw_len, wire_len=raw_len, codec="raw.small")
            return payload, {}
        level = _int_env("REMOTE_EDGE_LATENT_ZLIB_LEVEL", 1, low=1, high=9)
        min_savings = _float_env("REMOTE_EDGE_LATENT_ZLIB_MIN_SAVINGS", 0.05, low=0.0, high=0.95)
        try:
            compressed = zlib.compress(payload_view, level=int(level))
        except Exception as e:
            logging.warning("Remote edge latent zlib compression failed; sending raw: %s", e)
            self._note_latent_transport(raw_len=raw_len, wire_len=raw_len, codec="raw.zlib_failed")
            return payload, {}
        self._latent_zlib_trials += 1
        wire_len = int(len(compressed))
        if wire_len >= int(float(raw_len) * (1.0 - float(min_savings))):
            self._maybe_disable_ineffective_latent_zlib()
            self._note_latent_transport(raw_len=raw_len, wire_len=raw_len, codec="raw.zlib_skipped")
            return payload, {}
        self._latent_zlib_hits += 1
        self._note_latent_transport(raw_len=raw_len, wire_len=wire_len, codec="zlib")
        return compressed, {
            "payload_codec": "zlib",
            "payload_raw_len": int(raw_len),
        }

    def _maybe_disable_ineffective_latent_zlib(self) -> None:
        probe_blocks = _int_env("REMOTE_EDGE_LATENT_ZLIB_PROBE_BLOCKS", 8, low=0, high=1000)
        if int(probe_blocks) <= 0:
            return
        if int(self._latent_zlib_trials) < int(probe_blocks):
            return
        if int(self._latent_zlib_hits) > 0:
            return
        self._latent_zlib_disabled = True
        logging.warning(
            "Remote edge latent zlib disabled for this session: no payload met savings threshold after %d trials",
            int(self._latent_zlib_trials),
        )

    def _note_latent_transport(self, *, raw_len: int, wire_len: int, codec: str) -> None:
        self._transport_raw_bytes += int(max(0, int(raw_len)))
        self._transport_wire_bytes += int(max(0, int(wire_len)))
        interval = _float_env("REMOTE_EDGE_LATENT_TRANSPORT_LOG_SEC", 10.0, low=0.0, high=300.0)
        if float(interval) <= 0.0:
            return
        now = time.monotonic()
        if now - float(self._transport_log_last) < float(interval):
            return
        self._transport_log_last = float(now)
        raw_total = int(max(1, int(self._transport_raw_bytes)))
        wire_total = int(max(0, int(self._transport_wire_bytes)))
        logging.warning(
            "Remote edge latent transport: codec=%s last_raw=%.2fMiB last_wire=%.2fMiB total_raw=%.1fMiB total_wire=%.1fMiB ratio=%.3f",
            str(codec),
            float(raw_len) / 1048576.0,
            float(wire_len) / 1048576.0,
            float(raw_total) / 1048576.0,
            float(wire_total) / 1048576.0,
            float(wire_total) / float(raw_total),
        )

    def send_rgb24(self, payload: bytes | bytearray | memoryview, *, timestamp_us: int | None = None) -> None:
        self.write_message(
            "video.rgb24",
            payload,
            timestamp_us=None if timestamp_us is None else int(timestamp_us),
        )

    def send_poster_rgb24(self, payload: bytes | bytearray | memoryview) -> None:
        self.write_message("video.poster.rgb24", payload)

    def send_pcm16le(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        sample_rate: int = 16000,
        segment_id: str | None = None,
        segment_kind: str | None = None,
        segment_frames: int | None = None,
        segment_audible_samples: int | None = None,
        segment_turn_done: bool | None = None,
        subtitle_text: str | None = None,
        subtitle_start_samples: int | None = None,
        subtitle_end_samples: int | None = None,
        subtitle_total_samples: int | None = None,
        subtitle_alignment: dict[str, Any] | None = None,
        subtitle_normalized_alignment: dict[str, Any] | None = None,
        subtitle_alignment_base_samples: int | None = None,
    ) -> None:
        self.write_message(
            "audio.pcm16le",
            payload,
            sample_rate=int(sample_rate),
            segment_id=None if segment_id is None else str(segment_id),
            segment_kind=None if segment_kind is None else str(segment_kind),
            segment_frames=None if segment_frames is None else int(segment_frames),
            segment_audible_samples=None if segment_audible_samples is None else int(segment_audible_samples),
            segment_turn_done=None if segment_turn_done is None else bool(segment_turn_done),
            subtitle_text=None if subtitle_text is None else str(subtitle_text),
            subtitle_start_samples=None if subtitle_start_samples is None else int(subtitle_start_samples),
            subtitle_end_samples=None if subtitle_end_samples is None else int(subtitle_end_samples),
            subtitle_total_samples=None if subtitle_total_samples is None else int(subtitle_total_samples),
            subtitle_alignment=subtitle_alignment if isinstance(subtitle_alignment, dict) else None,
            subtitle_normalized_alignment=(
                subtitle_normalized_alignment if isinstance(subtitle_normalized_alignment, dict) else None
            ),
            subtitle_alignment_base_samples=(
                None if subtitle_alignment_base_samples is None else int(subtitle_alignment_base_samples)
            ),
        )

    def send_wav_pcm16le(self, path: str, *, source_chunk_idx: int = 0) -> None:
        pcm, sample_rate = _read_mono_pcm16_wav(str(path))
        self.send_pcm16le(pcm, sample_rate=int(sample_rate))

    def write_message(
        self,
        message_type: str,
        payload: bytes | bytearray | memoryview = b"",
        **fields: Any,
    ) -> None:
        sock = self._sock()
        payload_view = memoryview(payload).cast("B")
        header = {"type": str(message_type), "payload_len": int(payload_view.nbytes)}
        header.update(fields)
        header_bytes = (json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        started = time.perf_counter()
        header_started = time.perf_counter()
        sock.sendall(header_bytes)
        header_sec = float(time.perf_counter() - header_started)
        payload_sec = 0.0
        if int(payload_view.nbytes) > 0:
            payload_started = time.perf_counter()
            sock.sendall(payload_view)
            payload_sec = float(time.perf_counter() - payload_started)
        total_sec = float(time.perf_counter() - started)
        self._note_write_timing(
            message_type=str(message_type),
            header_bytes=int(len(header_bytes)),
            payload_bytes=int(payload_view.nbytes),
            header_sec=float(header_sec),
            payload_sec=float(payload_sec),
            total_sec=float(total_sec),
        )

    def _note_write_timing(
        self,
        *,
        message_type: str,
        header_bytes: int,
        payload_bytes: int,
        header_sec: float,
        payload_sec: float,
        total_sec: float,
    ) -> None:
        wire_bytes = int(max(0, int(header_bytes))) + int(max(0, int(payload_bytes)))
        self._write_stats_messages += 1
        if int(payload_bytes) > 0:
            self._write_stats_payload_messages += 1
        self._write_stats_wire_bytes += int(wire_bytes)
        self._write_stats_header_sec += float(header_sec)
        self._write_stats_header_max_sec = max(float(self._write_stats_header_max_sec), float(header_sec))
        self._write_stats_payload_sec += float(payload_sec)
        self._write_stats_payload_max_sec = max(float(self._write_stats_payload_max_sec), float(payload_sec))
        self._write_stats_total_sec += float(total_sec)
        self._write_stats_total_max_sec = max(float(self._write_stats_total_max_sec), float(total_sec))

        slow_warn_sec = _float_env("REMOTE_EDGE_PRODUCER_SEND_SLOW_WARN_SEC", 0.25, low=0.0, high=30.0)
        if float(slow_warn_sec) > 0.0 and float(total_sec) >= float(slow_warn_sec):
            logging.warning(
                "Remote edge producer slow send: type=%s wire=%.2fMiB header_send=%.3fs payload_send=%.3fs total_send=%.3fs",
                str(message_type),
                float(wire_bytes) / 1048576.0,
                float(header_sec),
                float(payload_sec),
                float(total_sec),
            )

        try:
            interval_default = float(
                os.getenv(
                    "REMOTE_EDGE_PRODUCER_SEND_STATS_INTERVAL_SEC",
                    os.getenv("REMOTE_EDGE_PRODUCER_STATS_INTERVAL_SEC", "0"),
                )
                or 0
            )
        except Exception:
            interval_default = 0.0
        interval = max(0.0, min(300.0, float(interval_default)))
        if float(interval) <= 0.0:
            return
        now = time.monotonic()
        if now - float(self._write_stats_last) < float(interval):
            return
        messages = int(max(1, int(self._write_stats_messages)))
        payload_messages = int(max(1, int(self._write_stats_payload_messages)))
        logging.warning(
            "Remote edge producer send stats: messages=%d payload_messages=%d wire=%.2fMiB header_avg=%.3fs header_max=%.3fs payload_avg=%.3fs payload_max=%.3fs total_avg=%.3fs total_max=%.3fs",
            int(self._write_stats_messages),
            int(self._write_stats_payload_messages),
            float(self._write_stats_wire_bytes) / 1048576.0,
            float(self._write_stats_header_sec) / float(messages),
            float(self._write_stats_header_max_sec),
            float(self._write_stats_payload_sec) / float(payload_messages),
            float(self._write_stats_payload_max_sec),
            float(self._write_stats_total_sec) / float(messages),
            float(self._write_stats_total_max_sec),
        )
        self._write_stats_last = float(now)
        self._write_stats_messages = 0
        self._write_stats_payload_messages = 0
        self._write_stats_wire_bytes = 0
        self._write_stats_header_sec = 0.0
        self._write_stats_header_max_sec = 0.0
        self._write_stats_payload_sec = 0.0
        self._write_stats_payload_max_sec = 0.0
        self._write_stats_total_sec = 0.0
        self._write_stats_total_max_sec = 0.0

    def read_message(self) -> tuple[dict[str, Any], bytes]:
        sock = self._sock()
        line = bytearray()
        while True:
            b = sock.recv(1)
            if not b:
                raise EOFError("remote stream closed")
            line.extend(b)
            if b == b"\n":
                break
            if len(line) > MAX_HEADER_BYTES:
                raise ValueError("remote stream header too large")
        header = json.loads(bytes(line).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("remote stream header must be object")
        payload_len = int(header.get("payload_len") or 0)
        payload = self._recvall(payload_len) if payload_len > 0 else b""
        return dict(header), payload

    def _recvall(self, nbytes: int) -> bytes:
        sock = self._sock()
        chunks: list[bytes] = []
        remaining = int(nbytes)
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise EOFError("remote stream closed while reading payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _sock(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("remote sender is not connected")
        return self.sock


@dataclass
class _AsyncSendItem:
    kind: str
    payload: Any = None
    fields: dict[str, Any] = field(default_factory=dict)


class AsyncRemoteStreamSender:
    """Ordered background sender for the model hot path.

    The producer must not serialize tensors or block on socket writes inside the
    denoise loop. This wrapper keeps message order, stages CUDA tensors to pinned
    CPU memory, and moves latent serialization/socket IO to a single sender
    thread.
    """

    _STOP = object()

    def __init__(
        self,
        sender: SyncRemoteStreamSender,
        *,
        max_queue: int = 16,
        put_timeout_sec: float = 0.25,
        close_timeout_sec: float = 30.0,
    ) -> None:
        self.sender = sender
        self.max_queue = int(max(1, min(256, int(max_queue))))
        self.put_timeout_sec = float(max(0.05, float(put_timeout_sec)))
        self.close_timeout_sec = float(max(0.0, float(close_timeout_sec)))
        self.queue: queue.Queue[_AsyncSendItem | object] = queue.Queue(maxsize=int(self.max_queue))
        self._closed = False
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._result: dict[str, Any] | None = None
        self._drop_audio_before = 0
        self._drop_audio_lock = threading.Lock()
        self._last_full_log = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="remote-edge-async-sender",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_env(cls, sender: SyncRemoteStreamSender) -> "AsyncRemoteStreamSender":
        return cls(
            sender,
            max_queue=_int_env("REMOTE_EDGE_ASYNC_QUEUE_MAX_BLOCKS", 16, low=1, high=256),
            put_timeout_sec=_float_env("REMOTE_EDGE_ASYNC_QUEUE_PUT_TIMEOUT_SEC", 0.25, low=0.05, high=10.0),
            close_timeout_sec=_float_env("REMOTE_EDGE_ASYNC_CLOSE_DRAIN_SEC", 30.0, low=0.0, high=300.0),
        )

    def check_error(self) -> None:
        with self._error_lock:
            err = self._error
        if err is not None:
            raise RuntimeError(f"remote edge async sender failed: {err}") from err

    def close(self, *, drain: bool = True) -> dict[str, Any] | None:
        if self._closed:
            return None if self._result is None else dict(self._result)
        self._closed = True
        if not bool(drain):
            try:
                self.sender.abort()
            except AttributeError:
                try:
                    self.sender.close(send_eos=False)
                except TypeError:
                    self.sender.close()
            except Exception:
                pass
            self._drain_queued_items()
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
            return None
        close_timeout_sec = float(self.close_timeout_sec)
        if str(getattr(self.sender, "output", "") or "").strip().lower() == "file":
            close_timeout_sec = max(
                close_timeout_sec,
                _float_env("REMOTE_EDGE_FILE_CLOSE_TIMEOUT_SEC", 1800.0, low=1.0, high=7200.0),
            )
        deadline = time.monotonic() + float(close_timeout_sec)
        while True:
            remaining = float(deadline - time.monotonic())
            if float(close_timeout_sec) <= 0.0 or remaining <= 0.0:
                break
            try:
                self.queue.put(self._STOP, timeout=min(0.25, remaining))
                break
            except queue.Full:
                self.check_error()
                continue
        else:
            pass
        if not self._thread.is_alive():
            self.check_error()
            return None if self._result is None else dict(self._result)
        self._thread.join(timeout=max(0.0, float(deadline - time.monotonic())))
        if self._thread.is_alive():
            logging.warning(
                "Remote edge async sender did not drain before close: queued=%d max=%d",
                int(self.queue.qsize()),
                int(self.max_queue),
            )
            try:
                self.sender.abort()
            except AttributeError:
                try:
                    self.sender.close(send_eos=False)
                except TypeError:
                    self.sender.close()
            except Exception:
                pass
            return None
        self.check_error()
        return None if self._result is None else dict(self._result)

    def abort(self) -> None:
        self.close(drain=False)

    def drop_audio_before(self, source_chunk_idx: int) -> None:
        idx = int(max(0, int(source_chunk_idx or 0)))
        if idx <= 0:
            return
        with self._drop_audio_lock:
            self._drop_audio_before = max(int(self._drop_audio_before), int(idx))

    def send_latents_tensor(
        self,
        tensor: Any,
        *,
        codec: str = "torch.save",
        shape: str = "",
        dtype: str = "",
        timestamp_us: int | None = None,
        keep_last_frames: int | None = None,
        reset_vae: bool = False,
        prime_only: bool = False,
        face_restore: float | None = None,
        background_restore: float | None = None,
        segment_id: str | None = None,
        segment_kind: str | None = None,
        segment_start_frame: int | None = None,
        segment_frames: int | None = None,
        avatar_ref_path: str | None = None,
    ) -> None:
        cpu_tensor, event = self._stage_tensor_for_worker(tensor)
        self._put(
            _AsyncSendItem(
                kind="latents",
                payload={"tensor": cpu_tensor, "event": event},
                fields={
                    "codec": str(codec),
                    "shape": str(shape),
                    "dtype": str(dtype),
                    "timestamp_us": None if timestamp_us is None else int(timestamp_us),
                    "keep_last_frames": None if keep_last_frames is None else int(keep_last_frames),
                    "reset_vae": bool(reset_vae),
                    "prime_only": bool(prime_only),
                    "face_restore": None if face_restore is None else float(face_restore),
                    "background_restore": None if background_restore is None else float(background_restore),
                    "segment_id": None if segment_id is None else str(segment_id),
                    "segment_kind": None if segment_kind is None else str(segment_kind),
                    "segment_start_frame": None if segment_start_frame is None else int(segment_start_frame),
                    "segment_frames": None if segment_frames is None else int(segment_frames),
                    "avatar_ref_path": None if avatar_ref_path is None else str(avatar_ref_path),
                },
            )
        )

    def send_pcm16le(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        sample_rate: int = 16000,
        segment_id: str | None = None,
        segment_kind: str | None = None,
        segment_frames: int | None = None,
        segment_audible_samples: int | None = None,
        segment_turn_done: bool | None = None,
        subtitle_text: str | None = None,
        subtitle_start_samples: int | None = None,
        subtitle_end_samples: int | None = None,
        subtitle_total_samples: int | None = None,
        subtitle_alignment: dict[str, Any] | None = None,
        subtitle_normalized_alignment: dict[str, Any] | None = None,
        subtitle_alignment_base_samples: int | None = None,
    ) -> None:
        self._put(
            _AsyncSendItem(
                kind="audio.pcm16le",
                payload=bytes(payload),
                fields={
                    "sample_rate": int(sample_rate),
                    "source_chunk_idx": 0,
                    "segment_id": None if segment_id is None else str(segment_id),
                    "segment_kind": None if segment_kind is None else str(segment_kind),
                    "segment_frames": None if segment_frames is None else int(segment_frames),
                    "segment_audible_samples": (
                        None if segment_audible_samples is None else int(segment_audible_samples)
                    ),
                    "segment_turn_done": None if segment_turn_done is None else bool(segment_turn_done),
                    "subtitle_text": None if subtitle_text is None else str(subtitle_text),
                    "subtitle_start_samples": None if subtitle_start_samples is None else int(subtitle_start_samples),
                    "subtitle_end_samples": None if subtitle_end_samples is None else int(subtitle_end_samples),
                    "subtitle_total_samples": None if subtitle_total_samples is None else int(subtitle_total_samples),
                    "subtitle_alignment": subtitle_alignment if isinstance(subtitle_alignment, dict) else None,
                    "subtitle_normalized_alignment": (
                        subtitle_normalized_alignment if isinstance(subtitle_normalized_alignment, dict) else None
                    ),
                    "subtitle_alignment_base_samples": (
                        None if subtitle_alignment_base_samples is None else int(subtitle_alignment_base_samples)
                    ),
                },
            )
        )

    def send_wav_pcm16le(self, path: str, *, source_chunk_idx: int) -> None:
        self._put(
            _AsyncSendItem(
                kind="audio.wav",
                payload=str(path),
                fields={"source_chunk_idx": int(max(0, int(source_chunk_idx or 0)))},
            )
        )

    def send_rgb24(self, payload: bytes | bytearray | memoryview, *, timestamp_us: int | None = None) -> None:
        self._put(
            _AsyncSendItem(
                kind="video.rgb24",
                payload=bytes(payload),
                fields={"timestamp_us": None if timestamp_us is None else int(timestamp_us)},
            )
        )

    def send_poster_rgb24(self, payload: bytes | bytearray | memoryview) -> None:
        self._put(_AsyncSendItem(kind="poster.rgb24", payload=bytes(payload)))

    def note_file_progress(self, *, phase: str, progress: float, **fields: Any) -> None:
        self.sender.note_file_progress(phase=str(phase or ""), progress=float(progress), **dict(fields or {}))

    def _put(self, item: _AsyncSendItem) -> None:
        if self._closed:
            raise RuntimeError("remote edge async sender is closed")
        while True:
            self.check_error()
            try:
                self.queue.put(item, timeout=float(self.put_timeout_sec))
                return
            except queue.Full:
                now = time.monotonic()
                if now - float(self._last_full_log) >= 5.0:
                    logging.warning(
                        "Remote edge async sender queue full: queued=%d max=%d",
                        int(self.queue.qsize()),
                        int(self.max_queue),
                    )
                    self._last_full_log = float(now)

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self._STOP:
                        return
                    assert isinstance(item, _AsyncSendItem)
                    self._send_item(item)
                finally:
                    self.queue.task_done()
        except BaseException as e:
            with self._error_lock:
                self._error = e
            logging.exception("Remote edge async sender failed")
        finally:
            try:
                result = self.sender.close()
                if isinstance(result, dict):
                    self._result = dict(result)
            except BaseException as e:
                with self._error_lock:
                    if self._error is None:
                        self._error = e
                logging.exception("Remote edge async sender close failed")

    def _drain_queued_items(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.queue.task_done()
            except Exception:
                pass

    def _send_item(self, item: _AsyncSendItem) -> None:
        if item.kind == "latents":
            data = dict(item.payload or {})
            event = data.get("event")
            if event is not None:
                event.synchronize()
            tensor = data.get("tensor")
            codec = str(item.fields.get("codec") or "torch.save")
            payload = _tensor_payload(tensor, codec=codec)
            self.sender.send_latents(payload, **dict(item.fields))
            return
        if item.kind == "audio.wav":
            source_chunk_idx = int(item.fields.get("source_chunk_idx") or 0)
            if self._should_drop_audio(source_chunk_idx):
                return
            pcm, sample_rate = _read_mono_pcm16_wav(str(item.payload or ""))
            fields = dict(item.fields)
            fields.pop("source_chunk_idx", None)
            fields["sample_rate"] = int(sample_rate)
            self.sender.send_pcm16le(pcm, **fields)
            return
        if item.kind == "audio.pcm16le":
            source_chunk_idx = int(item.fields.get("source_chunk_idx") or 0)
            if self._should_drop_audio(source_chunk_idx):
                return
            fields = dict(item.fields)
            fields.pop("source_chunk_idx", None)
            fields["sample_rate"] = int(fields.get("sample_rate") or 16000)
            self.sender.send_pcm16le(bytes(item.payload or b""), **fields)
            return
        if item.kind == "video.rgb24":
            self.sender.send_rgb24(bytes(item.payload or b""), timestamp_us=item.fields.get("timestamp_us"))
            return
        if item.kind == "poster.rgb24":
            self.sender.send_poster_rgb24(bytes(item.payload or b""))
            return
        raise RuntimeError(f"unsupported async remote message: {item.kind}")

    def _should_drop_audio(self, source_chunk_idx: int) -> bool:
        idx = int(max(0, int(source_chunk_idx or 0)))
        if idx <= 0:
            return False
        with self._drop_audio_lock:
            threshold = int(self._drop_audio_before)
        return bool(int(threshold) > 0 and int(idx) < int(threshold))

    def _stage_tensor_for_worker(self, tensor: Any) -> tuple[Any, Any | None]:
        try:
            import torch

            detached = tensor.detach()
            if bool(getattr(detached, "is_cuda", False)) and _flag_env("REMOTE_EDGE_ASYNC_SYNC_BEFORE_CPU_COPY", "1"):
                torch.cuda.synchronize(device=detached.device)
            if bool(getattr(detached, "is_cuda", False)) and _flag_env("REMOTE_EDGE_ASYNC_PINNED_COPY", "0"):
                try:
                    with torch.cuda.device(detached.device):
                        cpu_tensor = torch.empty_like(detached, device="cpu", pin_memory=True)
                        cpu_tensor.copy_(detached, non_blocking=True)
                        event = torch.cuda.Event(blocking=False)
                        event.record(torch.cuda.current_stream(detached.device))
                    return cpu_tensor, event
                except Exception as e:
                    logging.warning("Remote edge async pinned copy unavailable; falling back to sync CPU copy: %s", e)
            return detached.to("cpu"), None
        except Exception:
            raise


def _tensor_payload(tensor: Any, *, codec: str = "torch.save") -> bytes | memoryview:
    import torch

    codec_s = str(codec or "torch.save").strip().lower()
    if codec_s in {"raw", "raw_tensor", "tensor.raw"}:
        raw = tensor.detach().contiguous()
        if bool(getattr(raw, "is_cuda", False)):
            raw = raw.to("cpu")
        return memoryview(raw.view(torch.uint8).numpy()).cast("B")

    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _read_mono_pcm16_wav(path: str) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        pcm = wf.readframes(wf.getnframes())
    if channels != 1 or sampwidth != 2:
        raise RuntimeError(f"unsupported wav for remote edge audio: path={path} channels={channels} sampwidth={sampwidth}")
    return bytes(pcm), int(sample_rate)


def _flag_env(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(str(name), str(default)) or str(default)).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        return max(int(low), min(int(high), int(os.getenv(str(name), str(default)) or default)))
    except Exception:
        return int(default)


def _float_env(name: str, default: float, *, low: float, high: float) -> float:
    try:
        return max(float(low), min(float(high), float(os.getenv(str(name), str(default)) or default)))
    except Exception:
        return float(default)


def _tune_socket(sock: socket.socket, *, role: str) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    sndbuf = _int_env("REMOTE_EDGE_SOCKET_SNDBUF_BYTES", 8 * 1024 * 1024, low=0, high=256 * 1024 * 1024)
    rcvbuf = _int_env("REMOTE_EDGE_SOCKET_RCVBUF_BYTES", 8 * 1024 * 1024, low=0, high=256 * 1024 * 1024)
    if int(sndbuf) > 0:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(sndbuf))
        except Exception:
            pass
    if int(rcvbuf) > 0:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        except Exception:
            pass
    try:
        actual_snd = int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        actual_rcv = int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        logging.warning(
            "Remote edge socket tuned: role=%s nodelay=1 sndbuf=%d rcvbuf=%d",
            str(role),
            int(actual_snd),
            int(actual_rcv),
        )
    except Exception:
        pass
