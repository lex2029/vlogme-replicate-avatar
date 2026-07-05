from __future__ import annotations

from .common import *
from avalife.core.heartbeat import frontend_clear_startup_progress, frontend_mark_startup_progress
from avalife.core.path import sanitize_job_id

from .model_client import ModelRuntimeClient
from .smartblog_api import (
    LocalSmartBlogMockClient,
    SmartBlogClient,
    smartblog_mock_claim_file,
    smartblog_worker_api_key,
    smartblog_worker_api_single_endpoint,
)
from .smartblog_claims import smartblog_job_type_value
from .smartblog_control_plane import SmartBlogControlPlaneMixin
from .smartblog_jobs import SmartBlogRenderJobsMixin, smartblog_is_render_job
from .render_tts_helpers import SmartBlogRenderTTSMixin


class SmartBlogRenderOnlyWorker(SmartBlogControlPlaneMixin, SmartBlogRenderJobsMixin, SmartBlogRenderTTSMixin):
    """SmartBlog worker shell for B300 render-only pods.

    This keeps the same render implementation as the legacy SmartBlog worker,
    but avoids starting live/reply/chat loops. B300 pods should only claim
    render_video jobs; live sessions remain on the 3xB200 live worker.
    """

    def __init__(self, *, args: Any) -> None:
        self._frontend_heartbeat = None
        self.args = args
        self.sample_rate = max(8000, min(48000, _safe_int_env("WORKER_AUDIO_SAMPLE_RATE", 16000)))
        self.api_key = smartblog_worker_api_key()
        self.avatar_mode = os.getenv("AVATAR_MODE", "2dlive").strip() or "2dlive"
        self._mock_claim_file = smartblog_mock_claim_file()
        if self._mock_claim_file:
            self._smartblog_api = LocalSmartBlogMockClient(self._mock_claim_file)
            logging.warning("SmartBlog render-only mock mode enabled: claim_file=%s", self._mock_claim_file)
        else:
            self._smartblog_api = SmartBlogClient(self.api_key)
        self._model_client = ModelRuntimeClient()

        self._stop = asyncio.Event()
        self.current_session_id: str | None = None
        self.current_publish_task_id: str | None = None
        self._active_source: str = ""
        self._active_session_started_mono: float = 0.0
        self._smartblog_session_transport: str = "render"
        self._session_stop_sent_by_server: bool = False
        self._worker_secret_namespace_override: str = ""

        self._claim_lock = asyncio.Lock()
        self._smartblog_loop_ref: asyncio.AbstractEventLoop | None = None
        self._smartblog_job_q: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue(maxsize=512)
        self._active_smartblog_job_task: asyncio.Task | None = None
        self._smartblog_preempt_task: asyncio.Task | None = None
        self._smartblog_preempt_job_id: str = ""
        self._smartblog_preempt_started_mono: float = 0.0
        self._smartblog_post_drain_poll_task: asyncio.Task | None = None
        self._smartblog_remote_edge_unavailable_until_mono: float = 0.0

        self._smartblog_finalize_tasks: set[asyncio.Task] = set()
        self._smartblog_finalize_completed_count: int = 0
        self._smartblog_finalize_failed_count: int = 0
        self._smartblog_finalize_last_error: str = ""
        self._smartblog_finalize_last_error_mono: float = 0.0
        self._smartblog_finalize_last_complete_mono: float = 0.0
        self._smartblog_finalizing_job_id: str = ""
        self._render_tts_clients_lock = asyncio.Lock()
        self._render_eleven_http_session = None

        self._smartblog_realtime_connected: bool = False
        self._smartblog_realtime_status: str = "idle"
        self._smartblog_realtime_last_subscribed_mono: float = 0.0
        self._smartblog_realtime_last_raw_event_mono: float = 0.0
        self._smartblog_realtime_last_event_mono: float = 0.0
        self._smartblog_realtime_last_poll_mono: float = 0.0
        self._smartblog_realtime_last_poll_source: str = ""
        self._smartblog_realtime_last_error: str = ""
        self._smartblog_realtime_last_error_mono: float = 0.0
        self._smartblog_realtime_last_disconnect_mono: float = 0.0
        self._smartblog_realtime_last_recovery_mono: float = 0.0
        self._smartblog_realtime_last_recovery_reason: str = ""
        self._smartblog_realtime_reconnect_count: int = 0
        self._smartblog_realtime_recovery_count: int = 0
        self._smartblog_realtime_raw_event_count: int = 0
        self._smartblog_realtime_job_event_count: int = 0
        self._smartblog_realtime_debug_event_count: int = 0
        self._smartblog_realtime_last_event_kind: str = ""
        self._smartblog_realtime_last_parsed_from: str = ""
        self._smartblog_realtime_last_payload_keys: str = ""
        self._smartblog_realtime_last_data_keys: str = ""

        self._active_job: dict[str, Any] | None = None
        self._active_claim: dict[str, Any] | None = None
        self._last_progress_ok_mono: float = 0.0

        # Kept for shared heartbeat/control helpers. Render-only workers do not
        # create these live transports.
        self.lk_streamer = None
        self.lk_room = None
        self._channel_streamer = None
        self._active_session_stop = None
        self._render_cancel = asyncio.Event()

    async def aclose(self) -> None:
        await self._close_render_tts_clients()
        await self._smartblog_api.aclose()

    def hot_reload_non_model_logic(self) -> tuple[bool, str]:
        return False, "render-only worker does not support legacy hot reload"

    def request_stop(self) -> None:
        self._stop.set()
        try:
            self._smartblog_job_q.put_nowait({"__stop__": True})
        except Exception:
            pass
        preempt_task = getattr(self, "_smartblog_preempt_task", None)
        if isinstance(preempt_task, asyncio.Task) and (not preempt_task.done()):
            preempt_task.cancel()

    def _worker_busy(self) -> bool:
        active_task = getattr(self, "_active_smartblog_job_task", None)
        return bool(
            str(self.current_publish_task_id or "").strip()
            or (isinstance(active_task, asyncio.Task) and not active_task.done())
        )

    @staticmethod
    def _liveaudio_uri_from_dir(queue_dir: str) -> str:
        return f"liveaudio://{os.path.abspath(str(queue_dir or '').strip())}"

    @staticmethod
    def _prepare_liveaudio_queue_dir(base_dir: str, job_id: str) -> str:
        qdir = os.path.join(str(base_dir), "live_audio_queue", sanitize_job_id(str(job_id)))
        os.makedirs(qdir, exist_ok=True)
        for name in os.listdir(qdir):
            p = os.path.join(qdir, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass
        return qdir

    @staticmethod
    def _write_liveaudio_done_marker(
        queue_dir: str,
        *,
        chunks_total: int,
        status: str = "ok",
        total_samples: int | None = None,
        speech_end_samples: int | None = None,
        speech_end_sec: float | None = None,
        sample_rate: int = 16000,
    ) -> None:
        payload = {
            "done": True,
            "status": str(status or "ok"),
            "chunks_total": int(max(0, int(chunks_total))),
            "ts_ms": int(_now_ms()),
        }
        if isinstance(total_samples, (int, float)):
            payload["total_samples"] = int(max(0, int(round(float(total_samples)))))
        if isinstance(sample_rate, (int, float)):
            payload["sample_rate"] = int(max(1, int(round(float(sample_rate)))))
        if isinstance(speech_end_samples, (int, float)):
            payload["speech_end_samples"] = int(max(0, int(round(float(speech_end_samples)))))
        if isinstance(speech_end_sec, (int, float)):
            sec = float(speech_end_sec)
            if math.isfinite(sec) and sec >= 0.0:
                payload["speech_end_sec"] = float(sec)

        try:
            os.makedirs(str(queue_dir), exist_ok=True)
            tmp = os.path.join(queue_dir, f"done.json.tmp.{os.getpid()}.{time.time_ns()}")
            final = os.path.join(queue_dir, "done.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            os.replace(tmp, final)
        except Exception as e:
            logging.error("Liveaudio done marker failed: dir=%s err=%s", str(queue_dir), e)

        try:
            done_tmp = os.path.join(queue_dir, f".done.tmp.{os.getpid()}.{time.time_ns()}")
            done_final = os.path.join(queue_dir, ".done")
            with open(done_tmp, "w", encoding="utf-8") as f:
                f.write("done\n")
            os.replace(done_tmp, done_final)
        except Exception as e:
            logging.error("Liveaudio .done marker failed: dir=%s err=%s", str(queue_dir), e)

    @staticmethod
    def _write_liveaudio_chunk_meta(
        queue_dir: str,
        *,
        chunk_idx: int,
        kind: str,
        audible: bool,
        source_samples: int | None = None,
        source_frames: int | None = None,
        conditioning_frames: int | None = None,
        visible_start_frames: int | None = None,
        visible_frames: int | None = None,
        turn_done: bool = False,
        subtitle_text: str | None = None,
        subtitle_start_samples: int | None = None,
        subtitle_end_samples: int | None = None,
        subtitle_total_samples: int | None = None,
        subtitle_alignment: dict[str, Any] | None = None,
        subtitle_normalized_alignment: dict[str, Any] | None = None,
        subtitle_alignment_base_samples: int | None = None,
        lipsync_audio_path: str | None = None,
        embedded_visible_start_frames: bool | None = None,
        avatar_ref_path: str | None = None,
        visual_prompt: str | None = None,
        negative_prompt: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chunk_idx": int(max(1, int(chunk_idx))),
            "kind": str(kind or "speech").strip().lower() or "speech",
            "audible": bool(audible),
            "turn_done": bool(turn_done),
            "ts_ms": int(_now_ms()),
        }
        if isinstance(source_samples, (int, float)):
            payload["source_samples"] = int(max(0, int(round(float(source_samples)))))
        if isinstance(source_frames, (int, float)):
            payload["source_frames"] = int(max(0, int(round(float(source_frames)))))
        if isinstance(conditioning_frames, (int, float)):
            payload["conditioning_frames"] = int(max(0, int(round(float(conditioning_frames)))))
        if isinstance(visible_start_frames, (int, float)):
            payload["visible_start_frames"] = int(max(0, int(round(float(visible_start_frames)))))
        if isinstance(visible_frames, (int, float)):
            payload["visible_frames"] = int(max(0, int(round(float(visible_frames)))))
        if subtitle_text is not None:
            payload["subtitle_text"] = str(subtitle_text or "")
        if isinstance(subtitle_start_samples, (int, float)):
            payload["subtitle_start_samples"] = int(max(0, int(round(float(subtitle_start_samples)))))
        if isinstance(subtitle_end_samples, (int, float)):
            payload["subtitle_end_samples"] = int(max(0, int(round(float(subtitle_end_samples)))))
        if isinstance(subtitle_total_samples, (int, float)):
            payload["subtitle_total_samples"] = int(max(0, int(round(float(subtitle_total_samples)))))
        if isinstance(subtitle_alignment, dict) and subtitle_alignment:
            payload["subtitle_alignment"] = dict(subtitle_alignment)
        if isinstance(subtitle_normalized_alignment, dict) and subtitle_normalized_alignment:
            payload["subtitle_normalized_alignment"] = dict(subtitle_normalized_alignment)
        if isinstance(subtitle_alignment_base_samples, (int, float)):
            payload["subtitle_alignment_base_samples"] = int(max(0, int(round(float(subtitle_alignment_base_samples)))))
        if embedded_visible_start_frames is not None:
            payload["embedded_visible_start_frames"] = bool(embedded_visible_start_frames)
        for key, value in (
            ("lipsync_audio_path", lipsync_audio_path),
            ("avatar_ref_path", avatar_ref_path),
            ("visual_prompt", visual_prompt),
            ("negative_prompt", negative_prompt),
        ):
            value_s = str(value or "").strip()
            if value_s:
                payload[key] = value_s
        try:
            os.makedirs(str(queue_dir), exist_ok=True)
            stem = f"{int(max(1, int(chunk_idx))):06d}"
            tmp = os.path.join(queue_dir, f"{stem}.meta.json.tmp.{os.getpid()}.{time.time_ns()}")
            final = os.path.join(queue_dir, f"{stem}.meta.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            os.replace(tmp, final)
        except Exception as e:
            logging.warning(
                "Liveaudio chunk meta write failed: dir=%s idx=%s err=%s",
                str(queue_dir),
                str(chunk_idx),
                str(e),
            )

    async def _cancel_frontend_background_tasks(self) -> None:
        tasks: list[asyncio.Task] = []
        for attr in (
            "_smartblog_preempt_task",
            "_smartblog_post_drain_poll_task",
        ):
            task = getattr(self, attr, None)
            if isinstance(task, asyncio.Task) and (not task.done()):
                task.cancel()
                tasks.append(task)
            setattr(self, attr, None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._smartblog_wait_for_finalize_tasks()
        except Exception as e:
            logging.warning("Waiting for SmartBlog render finalize tasks failed: %s", e)

    async def _run_claimed_smartblog_job(self, claim: dict[str, Any], *, source: str) -> None:
        if not smartblog_is_render_job(claim):
            job_type = str(smartblog_job_type_value(claim) or "").strip().lower()
            raise RuntimeError(f"render-only worker cannot run SmartBlog job_type={job_type!r}")

        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_id = str(job.get("id") or "").strip() or "-"
        job_type = str(smartblog_job_type_value(claim) or "-")
        logging.info(
            "SmartBlog render-only claim accepted: id=%s type=%s source=%s",
            job_id,
            job_type or "-",
            str(source),
        )
        dropped = int(self._drain_smartblog_job_queue())
        if dropped > 0:
            logging.info("Dropped queued SmartBlog jobs after render-only claim: dropped=%d", dropped)

        job_task = asyncio.create_task(
            self._run_smartblog_render_job(claim),
            name=f"smartblog-render-job-{sanitize_job_id(job_id)}",
        )
        self._active_smartblog_job_task = job_task
        try:
            await job_task
        except asyncio.CancelledError:
            logging.warning("SmartBlog render-only job task cancelled: id=%s type=%s", job_id, job_type or "-")
        finally:
            if getattr(self, "_active_smartblog_job_task", None) is job_task:
                self._active_smartblog_job_task = None
            if (not self._stop.is_set()) and (not self._worker_busy()):
                self._signal_smartblog_worker_idle()

    async def run_forever(self) -> None:
        startup_steps = 2
        frontend_mark_startup_progress(
            self,
            phase="frontend_startup",
            stage="model_runtime_wait",
            step=1,
            total_steps=int(startup_steps),
        )
        model_ready_t0 = time.perf_counter()
        await self._model_client.wait_ready(
            timeout_sec=max(60.0, float(_safe_int_env("MODEL_RUNTIME_READY_TIMEOUT_SEC", 900)))
        )
        logging.info("Model runtime ready in %.2fs", float(time.perf_counter() - model_ready_t0))
        frontend_mark_startup_progress(
            self,
            phase="frontend_startup",
            stage="render_control_plane",
            step=2,
            total_steps=int(startup_steps),
        )
        frontend_clear_startup_progress(self)
        logging.info("Starting SmartBlog render-only control-plane")
        realtime_enabled_raw = str(os.getenv("SMARTBLOG_REALTIME_ENABLED") or "").strip()
        if realtime_enabled_raw:
            realtime_enabled = _env_flag("SMARTBLOG_REALTIME_ENABLED", "0")
        else:
            realtime_enabled = not bool(smartblog_worker_api_single_endpoint())
        rt: asyncio.Task | None = None
        if realtime_enabled:
            rt = asyncio.create_task(
                run_supervised_loop(
                    self._stop,
                    name="smartblog-render-realtime",
                    factory=self._smartblog_realtime_loop,
                ),
                name="smartblog-render-realtime",
            )
        else:
            self._smartblog_realtime_status = "disabled"
            logging.info("SmartBlog realtime disabled; worker API polling is authoritative")
            rt = asyncio.create_task(
                run_supervised_loop(
                    self._stop,
                    name="smartblog-render-polling",
                    factory=self._smartblog_polling_loop,
                ),
                name="smartblog-render-polling",
            )
        try:
            await self._stop.wait()
        finally:
            if rt is not None:
                rt.cancel()
                await asyncio.gather(rt, return_exceptions=True)
            await self._cancel_frontend_background_tasks()
            await self.aclose()
