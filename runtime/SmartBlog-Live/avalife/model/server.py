from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import signal
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import torch

from avalife.core import engine as la
from avalife.core.heartbeat import ProcessHeartbeat
from avalife.core.observability import media_timing_enabled, model_timing_enabled
from .config import model_runtime_socket_path
from .infer_cancel import InferenceCancelled, clear_infer_cancel, request_infer_cancel
from .media import process_media_sync
from .protocol import (
    CancelMediaRequest,
    CancelMediaResponse,
    CancelInferRequest,
    CancelInferResponse,
    InferRequest,
    InferResponse,
    MediaProcessRequest,
    MediaProcessResponse,
    error_response,
    ok_response,
)
from avalife.worker.live_audio_shm import live_audio_shm_write_header


_MEDIA_PROCESS_SESSION_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.getenv(name, default) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _cuda_memory_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return -1.0, -1.0
    try:
        device = torch.cuda.current_device()
        allocated = float(torch.cuda.memory_allocated(device)) / (1024.0 ** 3)
        reserved = float(torch.cuda.memory_reserved(device)) / (1024.0 ** 3)
        return allocated, reserved
    except Exception:
        return -1.0, -1.0


def _cleanup_cuda_allocator_after_infer(*, job_id: str, ok: bool) -> None:
    if not _env_flag("SMARTBLOG_MODELD_CLEAR_CUDA_AFTER_INFER", "0"):
        return
    before_alloc, before_reserved = _cuda_memory_gib()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        pass
    after_alloc, after_reserved = _cuda_memory_gib()
    logging.info(
        "Model runtime CUDA cleanup after infer: job=%s ok=%d "
        "cuda_alloc %.2f->%.2f GiB cuda_reserved %.2f->%.2f GiB",
        str(job_id or "-"),
        1 if bool(ok) else 0,
        float(before_alloc),
        float(after_alloc),
        float(before_reserved),
        float(after_reserved),
    )


class ModelRuntimeServer:
    def __init__(self, *, heartbeat: ProcessHeartbeat | None = None) -> None:
        self.socket_path = model_runtime_socket_path()
        self._server: asyncio.AbstractServer | None = None
        self._stop = asyncio.Event()
        self._heartbeat = heartbeat
        self._active_infer_lock = threading.Lock()
        self._active_infer_req: InferRequest | None = None
        self._active_media_lock = threading.Lock()
        self._active_media_req: MediaProcessRequest | None = None
        self._active_media_cancel: threading.Event | None = None

    def request_stop(self) -> None:
        self._stop.set()

    async def serve_forever(self) -> None:
        sock_path = Path(self.socket_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if sock_path.exists() or sock_path.is_socket():
                sock_path.unlink()
        except FileNotFoundError:
            pass

        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)
        try:
            os.chmod(self.socket_path, 0o660)
        except Exception:
            pass
        logging.info("Model runtime listening: socket=%s", self.socket_path)
        if self._heartbeat is not None:
            self._heartbeat.set_state("serving", socket=self.socket_path)

        async with self._server:
            stop_task = asyncio.create_task(self._stop.wait(), name="model-runtime-stop")
            serve_task = asyncio.create_task(self._server.serve_forever(), name="model-runtime-serve")
            done, pending = await asyncio.wait(
                {stop_task, serve_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for task in done:
                if task is serve_task:
                    await asyncio.gather(task, return_exceptions=True)

        try:
            if sock_path.exists():
                sock_path.unlink()
        except Exception:
            pass

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = "local"
        try:
            line = await reader.readline()
            if not line:
                return
            payload = json.loads(line.decode("utf-8"))
            op = str(payload.get("op") or "").strip()
            if op == "ping":
                if self._heartbeat is not None:
                    self._heartbeat.set_state("serving", last_ping_at=float(time.time()))
                resp = ok_response(status="ready")
            elif op == "infer":
                req = InferRequest.from_payload(dict(payload.get("request") or {}))
                resp = (await asyncio.to_thread(self._handle_infer_sync, req)).to_payload()
            elif op == "media_process":
                req = MediaProcessRequest.from_payload(dict(payload.get("request") or {}))
                resp = (await asyncio.to_thread(self._handle_media_process_sync, req)).to_payload()
            elif op == "cancel_active_infer":
                req = CancelInferRequest.from_payload(dict(payload.get("request") or {}))
                resp = (await asyncio.to_thread(self._handle_cancel_active_infer_sync, req)).to_payload()
            elif op == "cancel_active_media":
                req = CancelMediaRequest.from_payload(dict(payload.get("request") or {}))
                resp = (await asyncio.to_thread(self._handle_cancel_active_media_sync, req)).to_payload()
            else:
                resp = error_response(f"unsupported op: {op}")
        except Exception as e:
            logging.exception("Model runtime request failed: peer=%s err=%s", peer, e)
            resp = error_response(f"{type(e).__name__}: {e}")
        try:
            writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _handle_infer_sync(self, req: InferRequest) -> InferResponse:
        t0 = time.perf_counter()
        lock_wait_s = 0.0
        run_single_s = 0.0
        infer_ok = False
        if model_timing_enabled():
            logging.info(
                "Model runtime infer request: job=%s clips=%d steps=%d infer_frames=%d size=%s live_raw=%d face=%.2f bg=%.2f",
                str(req.job_id or "-"),
                int(req.num_clip),
                int(req.sample_steps),
                int(req.infer_frames),
                str(req.size or "-"),
                1 if str(req.live_raw_dir or "").strip() else 0,
                float(req.face_restore),
                float(req.background_restore),
            )
        if str(req.live_raw_dir or "").strip():
            logging.warning(
                "Model runtime live infer request: job=%s size=%s clips=%d ref=%s live_raw=%s prompt=%d video_prompt=%d negative_prompt=%d idle_prompt=%d tpp_cfg=%s",
                str(req.job_id or "-"),
                str(req.size or "-"),
                int(req.num_clip),
                os.path.basename(str(req.image_path or "")) or "-",
                os.path.basename(str(req.live_raw_dir or "")) or "-",
                int(len(str(req.prompt or ""))),
                int(len(str(req.video_prompt or ""))),
                int(len(str(req.negative_prompt or ""))),
                int(len(str(req.idle_prompt or ""))),
                str(req.tpp_cfg_mode or "-"),
            )
        if self._heartbeat is not None:
            self._heartbeat.set_state(
                "busy",
                infer_job_id=str(req.job_id or ""),
                infer_active_since=float(time.time()),
            )
        try:
            t_lock = time.perf_counter()
            with la.GENERATION_LOCK:
                clear_infer_cancel()
                with self._active_infer_lock:
                    self._active_infer_req = req
                try:
                    lock_wait_s = float(time.perf_counter() - t_lock)
                    t_run = time.perf_counter()
                    video_path = la.run_single_sample(
                        req.prompt,
                        req.image_path,
                        req.audio_path,
                        int(req.num_clip),
                        int(req.sample_steps),
                        float(req.sample_guide_scale),
                        int(req.infer_frames),
                        req.size,
                        int(req.base_seed),
                        req.sample_solver,
                        float(req.face_restore),
                        float(req.background_restore),
                        job_id=req.job_id,
                        enable_live_hls=bool(req.enable_live_hls),
                        live_raw_dir=req.live_raw_dir,
                        save_live_raw_mp4=bool(req.save_live_raw_mp4),
                        video_prompt=str(req.video_prompt or ""),
                        negative_prompt=str(req.negative_prompt or ""),
                        idle_prompt=str(req.idle_prompt or ""),
                        stream_file_output_path=str(req.stream_file_output_path or ""),
                        stream_file_output_width=int(req.stream_file_output_width or 0),
                        stream_file_output_height=int(req.stream_file_output_height or 0),
                        stream_file_output_fps=float(req.stream_file_output_fps or 0.0),
                        stream_file_trim_duration_sec=float(req.stream_file_trim_duration_sec or 0.0),
                        stream_file_interpolation=str(req.stream_file_interpolation or ""),
                        tpp_cfg_mode=str(req.tpp_cfg_mode or ""),
                        lipsync_audio_path=str(req.lipsync_audio_path or ""),
                    )
                    run_single_s = float(time.perf_counter() - t_run)
                    live_raw_without_mp4 = bool(str(req.live_raw_dir or "").strip()) and (not bool(req.save_live_raw_mp4))
                    if (not bool(req.enable_live_hls)) and (not bool(live_raw_without_mp4)) and (not str(video_path or "").strip()):
                        raise RuntimeError("model runtime returned no saved video output path")
                finally:
                    clear_infer_cancel()
                    with self._active_infer_lock:
                        if self._active_infer_req is req:
                            self._active_infer_req = None
            infer_ok = True
            return InferResponse(
                ok=True,
                video_path=(str(video_path) if video_path else None),
                lock_wait_s=float(lock_wait_s),
                run_single_s=float(run_single_s),
                total_s=float(time.perf_counter() - t0),
            )
        except InferenceCancelled as e:
            logging.warning(
                "Model runtime infer cancelled: job=%s reason=%s",
                str(req.job_id or "-"),
                str(e or "cancelled"),
            )
            return InferResponse(
                ok=False,
                error=f"InferenceCancelled: {e}",
                lock_wait_s=float(lock_wait_s),
                run_single_s=float(run_single_s),
                total_s=float(time.perf_counter() - t0),
            )
        except Exception as e:
            logging.exception("Model runtime infer failed: job=%s err=%s", str(req.job_id or "-"), e)
            return InferResponse(
                ok=False,
                error=f"{type(e).__name__}: {e}",
                lock_wait_s=float(lock_wait_s),
                run_single_s=float(run_single_s),
                total_s=float(time.perf_counter() - t0),
            )
        finally:
            if model_timing_enabled():
                logging.info(
                    "Model runtime infer timing: job=%s ok=%d lock_wait=%.3fs run_single=%.3fs total=%.3fs",
                    str(req.job_id or "-"),
                    1 if infer_ok else 0,
                    float(lock_wait_s),
                    float(run_single_s),
                    float(time.perf_counter() - t0),
                )
            with self._active_infer_lock:
                if self._active_infer_req is req:
                    self._active_infer_req = None
            if self._heartbeat is not None:
                self._heartbeat.set_state(
                    "serving",
                    last_infer_finished_at=float(time.time()),
                    last_infer_total_s=float(time.perf_counter() - t0),
                )
                self._heartbeat.clear_fields("infer_job_id", "infer_active_since")
            _cleanup_cuda_allocator_after_infer(job_id=str(req.job_id or ""), ok=bool(infer_ok))

    @staticmethod
    def _estimate_liveaudio_chunks(queue_dir: str) -> int:
        qdir = os.path.abspath(str(queue_dir or "").strip())
        if not qdir:
            return 0
        progress_path = os.path.join(qdir, "progress.json")
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            if not isinstance(progress, dict):
                progress = {}
        except Exception:
            progress = {}
        for key in ("seen_chunks",):
            try:
                value = int(progress.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                return int(value)
        try:
            next_idx = int(progress.get("next_idx") or 0)
        except Exception:
            next_idx = 0
        if next_idx > 1:
            return int(next_idx - 1)
        max_idx = 0
        try:
            for name in os.listdir(qdir):
                stem, ext = os.path.splitext(str(name))
                if ext.lower() != ".wav":
                    continue
                try:
                    idx = int(stem)
                except Exception:
                    idx = 0
                max_idx = max(int(max_idx), int(idx))
        except Exception:
            pass
        return int(max_idx)

    @staticmethod
    def _write_liveaudio_done_marker(queue_dir: str, *, chunks_total: int, sample_rate: int, status: str) -> None:
        qdir = os.path.abspath(str(queue_dir or "").strip())
        os.makedirs(qdir, exist_ok=True)
        payload = {
            "done": True,
            "status": str(status or "cancelled"),
            "chunks_total": int(max(0, int(chunks_total))),
            "ts_ms": int(time.time() * 1000.0),
            "sample_rate": int(max(1, int(sample_rate or 16000))),
        }
        tmp = os.path.join(qdir, f"done.json.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}")
        final = os.path.join(qdir, "done.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(tmp, final)
        done_tmp = os.path.join(qdir, f".done.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}")
        done_final = os.path.join(qdir, ".done")
        with open(done_tmp, "w", encoding="utf-8") as f:
            f.write("done\n")
        os.replace(done_tmp, done_final)

    @staticmethod
    def _mark_liveaudio_shm_done(queue_dir: str, *, chunks_total: int, sample_rate: int) -> None:
        qdir = os.path.abspath(str(queue_dir or "").strip())
        meta_path = os.path.join(qdir, "audio_shm.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        shm_name = str(meta.get("shm_name") or "").strip()
        if not shm_name:
            return
        try:
            chunk_capacity = int(meta.get("chunk_capacity") or 0)
        except Exception:
            chunk_capacity = 0
        try:
            sample_capacity = int(meta.get("sample_capacity") or 0)
        except Exception:
            sample_capacity = 0
        try:
            shm_rate = int(meta.get("sample_rate") or 0)
        except Exception:
            shm_rate = 0
        if chunk_capacity <= 0 or sample_capacity <= 0:
            return
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=str(shm_name), create=False)
            live_audio_shm_write_header(
                shm.buf,
                written_samples=0,
                written_chunks=int(max(0, int(chunks_total))),
                total_samples=0,
                speech_end_samples=0,
                sample_capacity=int(sample_capacity),
                chunk_capacity=int(chunk_capacity),
                sample_rate=int(max(1, int(shm_rate or sample_rate or 16000))),
                done=True,
            )
        except FileNotFoundError:
            return
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass

    def _handle_cancel_active_infer_sync(self, req: CancelInferRequest) -> CancelInferResponse:
        t0 = time.perf_counter()
        with self._active_infer_lock:
            active_req = self._active_infer_req
        if active_req is None:
            clear_infer_cancel()
            return CancelInferResponse(ok=True, cancelled=False, total_s=float(time.perf_counter() - t0))
        audio_path = str(getattr(active_req, "audio_path", "") or "").strip()
        job_id = str(getattr(active_req, "job_id", "") or "").strip() or None
        try:
            request_infer_cancel(job_id=job_id, reason=str(req.reason or "cancelled"))
            if not audio_path.startswith("liveaudio://"):
                logging.warning(
                    "Model runtime cancel_active_infer: job=%s reason=%s non_liveaudio=1",
                    job_id or "-",
                    str(req.reason or "-"),
                )
                return CancelInferResponse(
                    ok=True,
                    cancelled=True,
                    active_job_id=job_id,
                    total_s=float(time.perf_counter() - t0),
                )
            queue_dir = os.path.abspath(str(audio_path[len("liveaudio://") :]).strip())
            chunks_total = int(self._estimate_liveaudio_chunks(queue_dir))
            self._mark_liveaudio_shm_done(
                queue_dir,
                chunks_total=int(chunks_total),
                sample_rate=16000,
            )
            self._write_liveaudio_done_marker(
                queue_dir,
                chunks_total=int(chunks_total),
                sample_rate=16000,
                status="cancelled",
            )
            logging.warning(
                "Model runtime cancel_active_infer: job=%s reason=%s queue=%s chunks=%d",
                job_id or "-",
                str(req.reason or "-"),
                os.path.basename(queue_dir) or queue_dir,
                int(chunks_total),
            )
            return CancelInferResponse(
                ok=True,
                cancelled=True,
                active_job_id=job_id,
                total_s=float(time.perf_counter() - t0),
            )
        except Exception as e:
            logging.exception("Model runtime cancel_active_infer failed: job=%s err=%s", job_id or "-", e)
            return CancelInferResponse(
                ok=False,
                cancelled=False,
                active_job_id=job_id,
                error=f"{type(e).__name__}: {e}",
                total_s=float(time.perf_counter() - t0),
            )

    def _handle_media_process_sync(self, req: MediaProcessRequest) -> MediaProcessResponse:
        t0 = time.perf_counter()
        cancel_event = threading.Event()
        if media_timing_enabled():
            logging.info(
                "Model runtime media request: source_kind=%s upscale=%d face=%.2f bg=%.2f output=%sx%s preserve_audio=%d",
                str(req.source_kind or "-"),
                1 if bool(req.upscale) else 0,
                float(req.face_restore),
                float(req.background_restore),
                int(req.output_width or 0),
                int(req.output_height or 0),
                1 if bool(req.preserve_audio) else 0,
            )
        with self._active_media_lock:
            self._active_media_req = req
            self._active_media_cancel = cancel_event
        if self._heartbeat is not None:
            self._heartbeat.set_state(
                "busy",
                media_job_source=str(req.source_path or ""),
                media_job_kind=str(req.source_kind or ""),
                media_active_since=float(time.time()),
            )
        try:
            with _MEDIA_PROCESS_SESSION_LOCK:
                resp = process_media_sync(req, gpu_lock=la.GENERATION_LOCK, cancel_event=cancel_event)
            return MediaProcessResponse(
                ok=bool(resp.ok),
                output_path=(str(resp.output_path) if resp.output_path else None),
                error=(str(resp.error) if resp.error else None),
                frames_written=int(resp.frames_written or 0),
                total_s=float(time.perf_counter() - t0),
            )
        except Exception as e:
            logging.exception("Model runtime media_process failed: source=%s err=%s", str(req.source_path or "-"), e)
            return MediaProcessResponse(
                ok=False,
                error=f"{type(e).__name__}: {e}",
                frames_written=0,
                total_s=float(time.perf_counter() - t0),
            )
        finally:
            if media_timing_enabled():
                logging.info(
                    "Model runtime media timing: source_kind=%s total=%.3fs",
                    str(req.source_kind or "-"),
                    float(time.perf_counter() - t0),
                )
            with self._active_media_lock:
                if self._active_media_req is req:
                    self._active_media_req = None
                    self._active_media_cancel = None
            if self._heartbeat is not None:
                self._heartbeat.set_state(
                    "serving",
                    last_media_finished_at=float(time.time()),
                    last_media_total_s=float(time.perf_counter() - t0),
                )
                self._heartbeat.clear_fields("media_job_source", "media_job_kind", "media_active_since")

    def _handle_cancel_active_media_sync(self, req: CancelMediaRequest) -> CancelMediaResponse:
        t0 = time.perf_counter()
        with self._active_media_lock:
            active_req = self._active_media_req
            cancel_event = self._active_media_cancel
        if active_req is None or cancel_event is None:
            return CancelMediaResponse(ok=True, cancelled=False, total_s=float(time.perf_counter() - t0))
        source_path = str(getattr(active_req, "source_path", "") or "").strip() or None
        try:
            cancel_event.set()
            logging.warning(
                "Model runtime cancel_active_media: source=%s reason=%s",
                str(source_path or "-"),
                str(req.reason or "-"),
            )
            return CancelMediaResponse(
                ok=True,
                cancelled=True,
                active_source_path=source_path,
                total_s=float(time.perf_counter() - t0),
            )
        except Exception as e:
            logging.exception("Model runtime cancel_active_media failed: source=%s err=%s", source_path or "-", e)
            return CancelMediaResponse(
                ok=False,
                cancelled=False,
                active_source_path=source_path,
                error=f"{type(e).__name__}: {e}",
                total_s=float(time.perf_counter() - t0),
            )


def install_model_runtime_signal_handlers(server: ModelRuntimeServer) -> None:
    def _handle_stop(sig: int, _frame: Any) -> None:
        logging.warning("Model runtime signal %s received; stopping...", sig)
        server.request_stop()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle_stop)
        except Exception:
            pass
