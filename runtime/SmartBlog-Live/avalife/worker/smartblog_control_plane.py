from __future__ import annotations

from .common import *
from avalife.core.path import sanitize_job_id
from avalife.core.update_drain import auto_update_drain_requested
from pathlib import Path

from .smartblog_api import smartblog_supabase_service_role_key, smartblog_supabase_url
from .smartblog_claims import (
    SMARTBLOG_JOB_PRIORITY_GENERIC_BUSY,
    SMARTBLOG_JOB_PRIORITY_UNKNOWN,
    smartblog_claim_after_poll_delay_sec,
    smartblog_delayed_claim_sec,
    smartblog_is_urgent_job,
    smartblog_job_is_queue_candidate,
    smartblog_job_needs_delayed_claim,
    smartblog_job_needs_remote_edge_preclaim,
    smartblog_job_priority_value,
    smartblog_job_status_value,
    smartblog_job_type_value,
    smartblog_poll_interval_sec,
    smartblog_preempt_delay_sec,
    smartblog_preempt_enabled,
    smartblog_preempt_stop_timeout_sec,
    smartblog_realtime_debug_enabled,
    smartblog_realtime_safety_poll_sec,
    smartblog_remote_edge_preclaim_enabled,
    smartblog_remote_edge_preclaim_skip_sec,
    smartblog_remote_edge_preclaim_timeout_sec,
    smartblog_sort_jobs_for_claim,
    smartblog_supported_job_types,
    smartblog_supports_job,
)
from .smartblog_jobs import smartblog_render_job_types


def _smartblog_run_state_dir() -> Path:
    return Path(os.getenv("WORKER_RUN_STATE_DIR", str(Path.cwd() / "runtime"))).resolve()


def _smartblog_claim_quarantine_path() -> Path:
    return Path(
        os.getenv(
            "WORKER_CLAIM_QUARANTINE_PATH",
            str(_smartblog_run_state_dir() / "worker_claim_quarantine.json"),
        )
    ).resolve()


def _smartblog_modeld_heartbeat_path() -> Path:
    return Path(
        os.getenv(
            "WORKER_MODELD_HEARTBEAT_PATH",
            str(_smartblog_run_state_dir() / "modeld_heartbeat.json"),
        )
    ).resolve()


def _smartblog_claim_quarantine_enabled() -> bool:
    return _env_flag("SMARTBLOG_CLAIM_QUARANTINE_ENABLED", "1")


def _smartblog_preclaim_modeld_health_enabled() -> bool:
    return _env_flag("SMARTBLOG_PRECLAIM_MODELD_HEALTHCHECK", "1")


def _smartblog_modeld_heartbeat_stale_sec() -> float:
    return max(5.0, min(300.0, float(_safe_float_env("WORKER_MODELD_HEARTBEAT_STALE_SEC", 90.0))))


def _smartblog_compact_error_text(error: Any, *, limit: int = 1500) -> str:
    text = str(error or "").strip()
    if len(text) > int(limit):
        return text[: int(limit)]
    return text


def _smartblog_render_route_mode() -> str:
    raw = str(
        os.getenv(
            "SMARTBLOG_RENDER_ROUTE",
            os.getenv("SMARTBLOG_RENDER_JOB_ROUTE", os.getenv("SMARTBLOG_RENDER_PAYLOAD_ROUTE", "all")),
        )
        or "all"
    ).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"", "*", "all", "any", "default"}:
        return "all"
    if normalized in {"avatar", "avatar_only", "avataronly", "no_hunyuan", "no_ltx", "without_hunyuan", "without_ltx"}:
        return "avatar_only"
    if normalized in {"hunyuan", "hunyuan_only", "hunyuanonly", "ltx", "ltx_only", "ltxonly", "video_insert", "video_inserts"}:
        return "hunyuan_only"
    if normalized in {
        "hunyuan_primary_avatar_backup",
        "hunyuan_with_avatar_backup",
        "hunyuan_plus_avatar_backup",
        "ltx_primary_avatar_backup",
        "ltx_with_avatar_backup",
    }:
        return "hunyuan_primary_avatar_backup"
    return "all"


def _smartblog_render_route_payload_sources(claim_or_job: dict[str, Any] | None) -> list[dict[str, Any]]:
    src = claim_or_job if isinstance(claim_or_job, dict) else {}
    job = src.get("job") if isinstance(src.get("job"), dict) else {}
    sources: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, dict) and value:
            sources.append(dict(value))

    for candidate in (
        src.get("payload_json"),
        src.get("payload"),
        job.get("payload_json"),
        job.get("payload"),
    ):
        add(candidate)
        if isinstance(candidate, dict):
            add(candidate.get("assets"))
            add(candidate.get("video"))
            add(candidate.get("ltx"))
            add(candidate.get("hunyuan"))
    add(src.get("assets"))
    add(src.get("video"))
    add(src.get("ltx"))
    add(src.get("hunyuan"))
    return sources


def _smartblog_render_source_uses_hunyuan(value: Any, *, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if isinstance(value, dict):
        for key, item in value.items():
            key_s = str(key or "").strip().lower()
            if key_s in {
                "ltx",
                "ltx_video",
                "ltxvideo",
                "hunyuan",
                "hunyuan_video",
                "hunyuanvideo",
                "video2",
                "video_model",
                "videomodel",
            } and item not in (None, "", [], {}):
                return True
            if key_s in {"inserts", "insert", "video_inserts", "videoinserts"} and isinstance(item, (list, tuple)) and item:
                return True
            if key_s in {
                "kind",
                "type",
                "render_engine",
                "renderengine",
                "video_engine",
                "videoengine",
                "engine",
                "backend",
                "model",
                "model_id",
                "modelid",
                "generator",
                "pipeline",
                "provider",
            }:
                text = str(item or "").strip().lower()
                if any(token in text for token in ("ltx", "hunyuan", "text_to_video", "text-to-video")):
                    return True
            if key_s in {"frames", "frame_assets", "frameassets", "video_frames", "videoframes"} and isinstance(item, (list, tuple)):
                if any(_smartblog_render_source_uses_hunyuan(part, depth=depth + 1) for part in item):
                    return True
            elif isinstance(item, (dict, list, tuple)) and _smartblog_render_source_uses_hunyuan(item, depth=depth + 1):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_smartblog_render_source_uses_hunyuan(item, depth=depth + 1) for item in value)
    return False


def _smartblog_render_uses_hunyuan(claim_or_job: dict[str, Any] | None) -> bool | None:
    sources = _smartblog_render_route_payload_sources(claim_or_job)
    if not sources:
        return None
    return any(_smartblog_render_source_uses_hunyuan(src) for src in sources)


def _smartblog_render_route_mismatch_reason(claim_or_job: dict[str, Any] | None) -> str:
    mode = _smartblog_render_route_mode()
    if mode in {"all", "hunyuan_primary_avatar_backup"}:
        return ""
    job_type = str(smartblog_job_type_value(claim_or_job) or "").strip().lower()
    if job_type not in set(smartblog_render_job_types()):
        return ""
    uses_hunyuan = _smartblog_render_uses_hunyuan(claim_or_job)
    if uses_hunyuan is None:
        return f"render route needs payload_json to decide route={mode}"
    if mode == "hunyuan_only" and not bool(uses_hunyuan):
        return "render route mismatch route=hunyuan_only job=avatar_only"
    if mode == "avatar_only" and bool(uses_hunyuan):
        return "render route mismatch route=avatar_only job=hunyuan"
    return ""


def _smartblog_render_avatar_backup_delay_sec() -> float:
    return max(
        0.0,
        min(
            120.0,
            float(
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_BACKUP_DELAY_SEC",
                    _safe_float_env("SMARTBLOG_AVATAR_BACKUP_DELAY_SEC", 0.0),
                )
            ),
        ),
    )


def _smartblog_render_is_avatar_backup_job(claim_or_job: dict[str, Any] | None) -> bool:
    if _smartblog_render_route_mode() != "hunyuan_primary_avatar_backup":
        return False
    job_type = str(smartblog_job_type_value(claim_or_job) or "").strip().lower()
    if job_type not in set(smartblog_render_job_types()):
        return False
    uses_hunyuan = _smartblog_render_uses_hunyuan(claim_or_job)
    return uses_hunyuan is False


class SmartBlogControlPlaneMixin:
    def _smartblog_claim_quarantine_reason(self) -> str:
        if not bool(_smartblog_claim_quarantine_enabled()):
            return ""
        path = _smartblog_claim_quarantine_path()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return f"claim_quarantine:{path}"
        reason = str(data.get("reason") or "claim_quarantine").strip() or "claim_quarantine"
        job_id = str(data.get("job_id") or "").strip()
        return f"{reason}:job={job_id}" if job_id else reason

    def _smartblog_mark_claim_quarantine(
        self,
        *,
        reason: str,
        error: Any = None,
        job_id: str = "",
        job_type: str = "",
    ) -> None:
        if not bool(_smartblog_claim_quarantine_enabled()):
            return
        path = _smartblog_claim_quarantine_path()
        payload = {
            "reason": str(reason or "runtime_failure").strip() or "runtime_failure",
            "error": _smartblog_compact_error_text(error),
            "job_id": str(job_id or "").strip(),
            "job_type": str(job_type or "").strip(),
            "pid": int(os.getpid()),
            "created_at_unix": float(time.time()),
            "created_at_mono": float(time.monotonic()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
            logging.error(
                "SmartBlog worker claim quarantine enabled: reason=%s job=%s type=%s path=%s",
                payload["reason"],
                payload["job_id"] or "-",
                payload["job_type"] or "-",
                str(path),
            )
        except Exception as write_err:
            logging.warning("SmartBlog worker claim quarantine write failed: path=%s err=%s", str(path), write_err)

    def _smartblog_modeld_ready_for_claim_reason(self) -> str:
        if not bool(_smartblog_preclaim_modeld_health_enabled()):
            return ""
        path = _smartblog_modeld_heartbeat_path()
        if not path.exists():
            return f"modeld_heartbeat_missing:{path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"modeld_heartbeat_invalid:{e}"
        now = float(time.time())
        ts = float(data.get("timestamp", 0.0) or 0.0)
        stale_sec = float(_smartblog_modeld_heartbeat_stale_sec())
        if ts <= 0.0 or (now - ts) > stale_sec:
            return f"modeld_heartbeat_stale:age={now - ts:.1f}s"
        state = str(data.get("state") or "").strip().lower()
        if state != "serving":
            return f"modeld_not_serving:{state or '-'}"
        return ""

    def _smartblog_claim_gate_reason(self, *, source: str) -> str:
        reason = self._smartblog_claim_quarantine_reason()
        if reason:
            return reason
        reason = self._smartblog_modeld_ready_for_claim_reason()
        if reason:
            return reason
        return ""

    def _smartblog_claim_gate_ready(self, *, source: str) -> bool:
        reason = self._smartblog_claim_gate_reason(source=source)
        if not reason:
            return True
        logging.warning(
            "Skip SmartBlog claim while worker is not fully healthy: source=%s reason=%s",
            str(source or "-"),
            str(reason),
        )
        return False

    async def _smartblog_poll_supported_jobs(self, *, source: str) -> list[dict[str, Any]]:
        job_types = tuple(smartblog_supported_job_types())
        jobs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for job_type in job_types:
            job_type_s = str(job_type or "").strip()
            if not job_type_s:
                continue
            polled = await self._smartblog_api.poll(job_type=job_type_s)
            for job in list(polled or []):
                if not isinstance(job, dict):
                    continue
                job_id = str(job.get("id") or "").strip()
                dedupe_key = job_id or f"{job_type_s}:{len(jobs)}"
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)
                jobs.append(dict(job))
        logging.info(
            "SmartBlog poll supported job types: source=%s job_types=%s jobs=%d",
            str(source or "poll"),
            ",".join(job_types) or "-",
            len(jobs),
        )
        return jobs

    def _smartblog_supported_live_orientations(self) -> tuple[str, ...]:
        return ()

    def _smartblog_runtime_mismatch_reason(self, claim_or_job: dict[str, Any] | None) -> str:
        return str(_smartblog_render_route_mismatch_reason(claim_or_job) or "")

    def _smartblog_job_runtime_supported(self, claim_or_job: dict[str, Any] | None, *, source: str) -> bool:
        reason = self._smartblog_runtime_mismatch_reason(claim_or_job)
        if not reason:
            return True
        src = claim_or_job if isinstance(claim_or_job, dict) else {}
        job = src.get("job") if isinstance(src.get("job"), dict) else {}
        logging.info(
            "Ignore SmartBlog job unsupported by this worker runtime: id=%s type=%s status=%s source=%s reason=%s",
            str(job.get("id") or src.get("id") or "-"),
            str(smartblog_job_type_value(src) or "-"),
            str(smartblog_job_status_value(src) or "-"),
            str(source or "-"),
            str(reason),
        )
        return False

    async def _smartblog_job_with_preclaim_runtime_hints(
        self,
        job: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any] | None:
        if not self._smartblog_job_runtime_supported(job, source=source):
            return None
        return dict(job)

    def _active_worker_priority(self) -> int:
        active_source = str(getattr(self, "_active_source", "") or "").strip().lower()
        if active_source == "smartblog":
            return int(smartblog_job_priority_value(self._active_claim or self._active_job))
        if self._worker_busy():
            return int(SMARTBLOG_JOB_PRIORITY_GENERIC_BUSY)
        return int(SMARTBLOG_JOB_PRIORITY_UNKNOWN)

    def _active_worker_job_type(self) -> str:
        active_source = str(getattr(self, "_active_source", "") or "").strip().lower()
        if active_source == "smartblog":
            return str(smartblog_job_type_value(self._active_claim or self._active_job) or "").strip() or "-"
        return "-"

    def _smartblog_job_event_gate_reason(self) -> str:
        if bool(auto_update_drain_requested()):
            return "auto_update_pending"
        claim_gate_reason = self._smartblog_claim_gate_reason(source="realtime-gate")
        if claim_gate_reason:
            return f"worker_unhealthy:{claim_gate_reason}"
        if bool(self._worker_busy()):
            return "busy"
        try:
            if bool(self._claim_lock.locked()):
                return "claim_lock"
        except Exception:
            pass
        return "ready"

    def _drain_smartblog_job_queue(self) -> int:
        dropped = 0
        while True:
            try:
                _ = self._smartblog_job_q.get_nowait()
                dropped += 1
            except Exception:
                break
        return int(dropped)

    def _smartblog_mark_realtime_subscribed(self) -> None:
        now = float(time.monotonic())
        if float(getattr(self, "_smartblog_realtime_last_subscribed_mono", 0.0) or 0.0) > 0.0:
            self._smartblog_realtime_reconnect_count = int(getattr(self, "_smartblog_realtime_reconnect_count", 0) or 0) + 1
        self._smartblog_realtime_connected = True
        self._smartblog_realtime_status = "subscribed"
        self._smartblog_realtime_last_subscribed_mono = now
        self._smartblog_realtime_last_error = ""
        self._smartblog_realtime_last_error_mono = 0.0

    def _smartblog_mark_realtime_event(self) -> None:
        self._smartblog_realtime_last_event_mono = float(time.monotonic())
        self._smartblog_realtime_job_event_count = int(getattr(self, "_smartblog_realtime_job_event_count", 0) or 0) + 1

    def _smartblog_mark_realtime_raw_event(
        self,
        *,
        event_kind: str = "",
        parsed_from: str = "",
        payload_keys: str = "",
        data_keys: str = "",
    ) -> None:
        self._smartblog_realtime_last_raw_event_mono = float(time.monotonic())
        self._smartblog_realtime_raw_event_count = int(getattr(self, "_smartblog_realtime_raw_event_count", 0) or 0) + 1
        self._smartblog_realtime_last_event_kind = str(event_kind or "").strip()
        self._smartblog_realtime_last_parsed_from = str(parsed_from or "").strip()
        self._smartblog_realtime_last_payload_keys = str(payload_keys or "").strip()
        self._smartblog_realtime_last_data_keys = str(data_keys or "").strip()

    def _smartblog_mark_realtime_poll(self, *, source: str) -> None:
        self._smartblog_realtime_last_poll_mono = float(time.monotonic())
        self._smartblog_realtime_last_poll_source = str(source or "").strip()

    def _smartblog_mark_realtime_failure(self, *, state: str, error_text: str = "") -> None:
        now = float(time.monotonic())
        self._smartblog_realtime_connected = False
        self._smartblog_realtime_status = str(state or "error").strip().lower() or "error"
        self._smartblog_realtime_last_disconnect_mono = now
        if str(error_text or "").strip():
            self._smartblog_realtime_last_error = str(error_text).strip()
            self._smartblog_realtime_last_error_mono = now

    def _signal_smartblog_realtime_state(self, state: str, *, error_text: str = "") -> None:
        marker = {
            "__realtime_state__": str(state or "unknown").strip().lower() or "unknown",
            "__error__": str(error_text or "").strip(),
        }

        def _push() -> None:
            try:
                self._smartblog_job_q.put_nowait(marker)
                return
            except asyncio.QueueFull:
                try:
                    _ = self._smartblog_job_q.get_nowait()
                except Exception:
                    return
                try:
                    self._smartblog_job_q.put_nowait(marker)
                except Exception:
                    return
            except Exception:
                return

        if self._smartblog_loop_ref is None:
            _push()
            return
        try:
            self._smartblog_loop_ref.call_soon_threadsafe(_push)
        except Exception:
            _push()

    def _signal_smartblog_worker_idle(self) -> None:
        marker = {"__worker_idle__": True}

        def _push() -> None:
            try:
                self._smartblog_job_q.put_nowait(marker)
                return
            except asyncio.QueueFull:
                try:
                    _ = self._smartblog_job_q.get_nowait()
                except Exception:
                    return
                try:
                    self._smartblog_job_q.put_nowait(marker)
                except Exception:
                    return
            except Exception:
                return

        if self._smartblog_loop_ref is None:
            _push()
            return
        try:
            self._smartblog_loop_ref.call_soon_threadsafe(_push)
        except Exception:
            _push()

    def _schedule_smartblog_post_drain_poll(self, *, source: str) -> None:
        current = getattr(self, "_smartblog_post_drain_poll_task", None)
        if isinstance(current, asyncio.Task) and (not current.done()):
            return
        source_s = str(source or "drain").strip() or "drain"

        async def _runner() -> None:
            started = time.monotonic()
            try:
                while (not self._stop.is_set()) and bool(auto_update_drain_requested()):
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    if time.monotonic() - started >= 600.0:
                        logging.warning(
                            "SmartBlog post-drain poll gave up after %.1fs: source=%s",
                            float(time.monotonic() - started),
                            source_s,
                        )
                        return
                if self._stop.is_set():
                    return
                if self._worker_busy():
                    logging.info("SmartBlog post-drain poll skipped while worker busy: source=%s", source_s)
                    return
                waited = max(0.0, time.monotonic() - started)
                logging.info(
                    "SmartBlog post-drain bootstrap poll triggered: source=%s waited=%.1fs",
                    source_s,
                    float(waited),
                )
                await self._smartblog_bootstrap_scan(source=f"post-drain:{source_s}")
            finally:
                if getattr(self, "_smartblog_post_drain_poll_task", None) is asyncio.current_task():
                    self._smartblog_post_drain_poll_task = None

        def _start() -> None:
            current_inner = getattr(self, "_smartblog_post_drain_poll_task", None)
            if isinstance(current_inner, asyncio.Task) and (not current_inner.done()):
                return
            self._smartblog_post_drain_poll_task = asyncio.create_task(
                _runner(),
                name="smartblog-post-drain-poll",
            )

        loop = getattr(self, "_smartblog_loop_ref", None)
        if loop is not None:
            try:
                loop.call_soon_threadsafe(_start)
                return
            except Exception:
                pass
        try:
            _start()
        except RuntimeError:
            logging.info("SmartBlog post-drain poll not scheduled without running loop: source=%s", source_s)

    def _handle_smartblog_realtime_subscribe_state(self, state: Any, error: Exception | None = None) -> None:
        state_name = str(getattr(state, "value", state) or "").strip().lower() or "unknown"
        error_text = str(error or "").strip()
        if state_name == "subscribed":
            self._smartblog_mark_realtime_subscribed()
            logging.info(
                "SmartBlog realtime subscribe state: state=%s raw_count=%d job_count=%d debug_count=%d busy=%d",
                state_name,
                int(getattr(self, "_smartblog_realtime_raw_event_count", 0) or 0),
                int(getattr(self, "_smartblog_realtime_job_event_count", 0) or 0),
                int(getattr(self, "_smartblog_realtime_debug_event_count", 0) or 0),
                1 if self._worker_busy() else 0,
            )
            return
        self._smartblog_mark_realtime_failure(state=state_name, error_text=error_text)
        logging.warning(
            "SmartBlog realtime subscribe state: state=%s err=%s raw_count=%d job_count=%d debug_count=%d",
            state_name,
            error_text or "-",
            int(getattr(self, "_smartblog_realtime_raw_event_count", 0) or 0),
            int(getattr(self, "_smartblog_realtime_job_event_count", 0) or 0),
            int(getattr(self, "_smartblog_realtime_debug_event_count", 0) or 0),
        )
        self._signal_smartblog_realtime_state(state_name, error_text=error_text)

    def _smartblog_extract_realtime_job_row(
        self, payload: Any
    ) -> tuple[dict[str, Any] | None, str, str, str, str, str]:
        job_row: dict[str, Any] | None = None
        parsed_from = ""
        payload_type = type(payload).__name__
        payload_keys = ""
        data_keys = ""
        event_kind = ""
        try:
            if isinstance(payload, dict):
                payload_keys = ",".join(sorted([str(k) for k in payload.keys()]))
                event_kind = str(payload.get("eventType") or payload.get("type") or payload.get("event") or "").strip()
                cand = payload.get("new")
                if isinstance(cand, dict):
                    job_row = cand
                    parsed_from = "dict:new"
                if job_row is None:
                    cand = payload.get("record")
                    if isinstance(cand, dict):
                        job_row = cand
                        parsed_from = "dict:record"
                data_obj = payload.get("data")
                if isinstance(data_obj, dict):
                    data_keys = ",".join(sorted([str(k) for k in data_obj.keys()]))
                    if job_row is None:
                        cand = data_obj.get("record")
                        if isinstance(cand, dict):
                            job_row = cand
                            parsed_from = "dict:data.record"
            if not event_kind and hasattr(payload, "event_type"):
                event_kind = str(getattr(payload, "event_type", "") or "").strip()
            if not event_kind and hasattr(payload, "type"):
                event_kind = str(getattr(payload, "type", "") or "").strip()
            if job_row is None and hasattr(payload, "data"):
                data_obj = getattr(payload, "data", None)
                if isinstance(data_obj, dict):
                    if not data_keys:
                        data_keys = ",".join(sorted([str(k) for k in data_obj.keys()]))
                    cand = data_obj.get("record")
                    if isinstance(cand, dict):
                        job_row = cand
                        parsed_from = "attr:data.record"
            if job_row is None and hasattr(payload, "new"):
                cand = getattr(payload, "new", None)
                if isinstance(cand, dict):
                    job_row = cand
                    parsed_from = "attr:new"
            if job_row is None and hasattr(payload, "record"):
                cand = getattr(payload, "record", None)
                if isinstance(cand, dict):
                    job_row = cand
                    parsed_from = "attr:record"
        except Exception:
            pass
        return job_row, parsed_from, payload_type, payload_keys, data_keys, event_kind

    def _on_smartblog_realtime_debug_event(self, payload: Any) -> None:
        try:
            self._smartblog_realtime_debug_event_count = int(getattr(self, "_smartblog_realtime_debug_event_count", 0) or 0) + 1
            job_row, parsed_from, payload_type, payload_keys, data_keys, event_kind = self._smartblog_extract_realtime_job_row(payload)
            self._smartblog_mark_realtime_raw_event(
                event_kind=event_kind,
                parsed_from=parsed_from,
                payload_keys=payload_keys,
                data_keys=data_keys,
            )
            logging.info(
                "SmartBlog realtime sniff: payload_type=%s event=%s parsed_from=%s payload_keys=%s data_keys=%s id=%s type=%s status=%s",
                str(payload_type or "-"),
                str(event_kind or "-"),
                str(parsed_from or "-"),
                str(payload_keys or "-"),
                str(data_keys or "-"),
                str((job_row or {}).get("id") or "-"),
                str(smartblog_job_type_value(job_row or {}) or "-"),
                str(smartblog_job_status_value(job_row or {}) or "-"),
            )
        except Exception as e:
            logging.warning("SmartBlog realtime sniff callback failed: %s", e)

    async def _smartblog_realtime_recover_once(self, *, reason: str) -> None:
        self._smartblog_realtime_recovery_count = int(getattr(self, "_smartblog_realtime_recovery_count", 0) or 0) + 1
        self._smartblog_realtime_last_recovery_mono = float(time.monotonic())
        self._smartblog_realtime_last_recovery_reason = str(reason or "").strip()
        if self._worker_busy():
            if bool(smartblog_preempt_enabled()):
                logging.warning("SmartBlog realtime recovery: urgent poll while busy reason=%s", reason or "-")
                await self._smartblog_poll_for_urgent_preempt_candidate(source=f"realtime-recover:{reason}")
                return
            logging.info("SmartBlog realtime recovery skipped while worker busy: reason=%s", reason or "-")
            return
        logging.warning("SmartBlog realtime recovery: bootstrap poll reason=%s", reason or "-")
        _ = await self._smartblog_bootstrap_scan(source=f"realtime-recover:{reason}")

    def _enqueue_smartblog_job(self, job: dict[str, Any]) -> None:
        if not smartblog_supports_job(job):
            return
        if str(self._smartblog_job_event_gate_reason() or "ready").strip().lower() == "busy":
            return
        job_copy = dict(job or {})

        def _push() -> None:
            try:
                self._smartblog_job_q.put_nowait(job_copy)
                return
            except asyncio.QueueFull:
                pass
            dropped_job_id = "-"
            dropped_job_type = "-"
            try:
                dropped = self._smartblog_job_q.get_nowait()
                if isinstance(dropped, dict):
                    dropped_job_id = str(dropped.get("id") or "-")
                    dropped_job_type = str(smartblog_job_type_value(dropped) or "-")
            except Exception:
                pass
            try:
                self._smartblog_job_q.put_nowait(job_copy)
                logging.warning(
                    "SmartBlog realtime queue full: dropped_oldest id=%s type=%s replaced_by=%s qmax=%d",
                    dropped_job_id,
                    dropped_job_type,
                    str(job_copy.get("id") or "-"),
                    int(getattr(self._smartblog_job_q, "maxsize", 0) or 0),
                )
            except Exception:
                pass

        if self._smartblog_loop_ref is None:
            _push()
            return
        try:
            self._smartblog_loop_ref.call_soon_threadsafe(_push)
        except Exception:
            _push()

    def _current_smartblog_job_id(self) -> str:
        return str((((self._active_claim or {}).get("job") or {}).get("id") or ((self._active_job or {}).get("id") or ""))).strip()

    def _current_smartblog_job_type(self) -> str:
        return str(smartblog_job_type_value(self._active_claim or self._active_job) or "").strip().lower()

    def _current_smartblog_job_is_preemptible(self) -> bool:
        if not bool(smartblog_preempt_enabled()):
            return False
        if str(getattr(self, "_active_source", "") or "").strip().lower() != "smartblog":
            return False
        current_job_id = str(self._current_smartblog_job_id() or "").strip()
        finalizing_job_id = str(getattr(self, "_smartblog_finalizing_job_id", "") or "").strip()
        if current_job_id and finalizing_job_id and current_job_id == finalizing_job_id:
            return False
        current_type = str(self._current_smartblog_job_type() or "").strip().lower()
        if current_type not in set(smartblog_render_job_types()):
            return False
        return not bool(smartblog_is_urgent_job(self._active_claim or self._active_job))

    def _clear_smartblog_preempt_tracking(self) -> None:
        self._smartblog_preempt_task = None
        self._smartblog_preempt_job_id = ""
        self._smartblog_preempt_started_mono = 0.0

    def _schedule_smartblog_preempt(self, job: dict[str, Any], *, source: str) -> None:
        if not bool(smartblog_preempt_enabled()):
            return
        if not isinstance(job, dict):
            return
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            return
        if not smartblog_is_urgent_job(job):
            return
        if not bool(self._current_smartblog_job_is_preemptible()):
            return

        def _push() -> None:
            current_task = getattr(self, "_smartblog_preempt_task", None)
            current_job_id = str(getattr(self, "_smartblog_preempt_job_id", "") or "").strip()
            if isinstance(current_task, asyncio.Task) and (not current_task.done()) and current_job_id == job_id:
                return
            if isinstance(current_task, asyncio.Task) and (not current_task.done()):
                current_task.cancel()
            self._smartblog_preempt_job_id = job_id
            self._smartblog_preempt_started_mono = float(time.monotonic())
            self._smartblog_preempt_task = asyncio.create_task(
                self._smartblog_preempt_after_delay(dict(job), source=str(source or "realtime")),
                name=f"smartblog-preempt-{sanitize_job_id(job_id)}",
            )

        if self._smartblog_loop_ref is None:
            _push()
            return
        try:
            self._smartblog_loop_ref.call_soon_threadsafe(_push)
        except Exception:
            _push()

    async def _claim_smartblog_job(self, job: dict[str, Any], *, source: str, allow_while_busy: bool = False) -> dict[str, Any] | None:
        if not smartblog_supports_job(job):
            return None
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            return None
        checked_job = await self._smartblog_job_with_preclaim_runtime_hints(job, source=str(source))
        if checked_job is None:
            return None
        job = checked_job
        if not self._smartblog_claim_gate_ready(source=str(source)):
            return None
        if not await self._smartblog_remote_edge_available_before_claim(job, source=str(source)):
            return None
        async with self._claim_lock:
            if not self._smartblog_claim_gate_ready(source=f"{source}:locked"):
                return None
            if bool(auto_update_drain_requested()):
                logging.info("Skip SmartBlog job while auto-update drain is pending: id=%s source=%s", job_id, str(source))
                return None
            if (not bool(allow_while_busy)) and self._worker_busy():
                logging.info("Skip SmartBlog job while worker busy: id=%s source=%s", job_id, str(source))
                return None
            claim_t0 = time.perf_counter()
            claim = await self._smartblog_api.claim(job_id=job_id)
            claim_elapsed = float(time.perf_counter() - claim_t0)
            if claim_elapsed >= 2.0:
                logging.warning(
                    "SmartBlog claim API slow: id=%s source=%s elapsed=%.2fs claimed=%d",
                    job_id,
                    str(source),
                    float(claim_elapsed),
                    int(bool(claim.get("claimed"))),
                )
            else:
                logging.info(
                    "SmartBlog claim API timing: id=%s source=%s elapsed=%.2fs claimed=%d",
                    job_id,
                    str(source),
                    float(claim_elapsed),
                    int(bool(claim.get("claimed"))),
                )
            if not bool(claim.get("claimed")):
                logging.info("SmartBlog claim skipped: id=%s source=%s", job_id, str(source))
                return None
            preclaim_payload = job.get("payload_json") if isinstance(job.get("payload_json"), dict) else {}
            if preclaim_payload and not isinstance(claim.get("payload_json"), dict):
                claim["payload_json"] = dict(preclaim_payload)
                logging.info(
                    "SmartBlog claim preserved preclaim payload_json: id=%s source=%s keys=%s",
                    job_id,
                    str(source),
                    ",".join(sorted(str(k) for k in preclaim_payload.keys())) or "-",
                )
            claim_job = claim.get("job") if isinstance(claim.get("job"), dict) else None
            if preclaim_payload and isinstance(claim_job, dict) and not isinstance(claim_job.get("payload_json"), dict):
                claim_job["payload_json"] = dict(preclaim_payload)
            checked_claim = dict(claim)
            if not self._smartblog_job_runtime_supported(checked_claim, source=f"{source}:claimed"):
                released_id = str((((checked_claim or {}).get("job") or {}).get("id") or job_id)).strip()
                try:
                    await self._smartblog_api.release(job_id=released_id)
                    logging.warning(
                        "SmartBlog claim released due to unsupported runtime route/profile: id=%s source=%s",
                        released_id or job_id,
                        str(source),
                    )
                except Exception as e:
                    logging.warning(
                        "SmartBlog claim release failed after unsupported runtime route/profile: id=%s source=%s err=%s",
                        released_id or job_id,
                        str(source),
                        e,
                    )
                return None
            return dict(checked_claim)

    async def _smartblog_remote_edge_available_before_claim(self, job: dict[str, Any], *, source: str) -> bool:
        if not bool(smartblog_remote_edge_preclaim_enabled()):
            return True
        if not bool(smartblog_job_needs_remote_edge_preclaim(job)):
            return True
        job_id = str(job.get("id") or "").strip() or "-"
        job_type = str(smartblog_job_type_value(job) or "-").strip() or "-"
        now_mono = float(time.monotonic())
        unavailable_until = float(getattr(self, "_smartblog_remote_edge_unavailable_until_mono", 0.0) or 0.0)
        if unavailable_until > now_mono:
            logging.info(
                "Skip SmartBlog edge job before claim: remote edge recently unavailable wait=%.1fs id=%s type=%s source=%s",
                float(unavailable_until - now_mono),
                job_id,
                job_type,
                str(source or "-"),
            )
            return False

        host = str(os.getenv("REMOTE_EDGE_HOST", "") or "").strip()
        port = int(_safe_int_env("REMOTE_EDGE_PORT", 0))
        if not host or port <= 0:
            self._smartblog_remote_edge_unavailable_until_mono = now_mono + float(smartblog_remote_edge_preclaim_skip_sec())
            logging.warning(
                "Skip SmartBlog edge job before claim: missing remote edge config host=%s port=%s id=%s type=%s source=%s",
                host or "-",
                str(port or "-"),
                job_id,
                job_type,
                str(source or "-"),
            )
            return False

        timeout_sec = float(smartblog_remote_edge_preclaim_timeout_sec())
        writer: Any | None = None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_sec)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception as e:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            self._smartblog_remote_edge_unavailable_until_mono = now_mono + float(smartblog_remote_edge_preclaim_skip_sec())
            logging.warning(
                "Skip SmartBlog edge job before claim: remote edge unavailable host=%s port=%s timeout=%.1fs id=%s type=%s source=%s err=%s",
                host,
                int(port),
                timeout_sec,
                job_id,
                job_type,
                str(source or "-"),
                e,
            )
            return False

    async def _stop_current_smartblog_job_for_preempt(self, *, reason: str) -> bool:
        job_task = getattr(self, "_active_smartblog_job_task", None)
        if not isinstance(job_task, asyncio.Task):
            return False
        if job_task.done():
            return True
        session_stop = getattr(self, "_active_session_stop", None)
        if isinstance(session_stop, asyncio.Event):
            session_stop.set()
        try:
            self._render_cancel.set()
        except Exception:
            pass
        try:
            _ = await self._model_client.cancel_active_media(reason=str(reason or "smartblog_preempt"))
        except Exception as e:
            logging.warning("SmartBlog preempt cancel_active_media failed: %s", e)
        try:
            _ = await self._model_client.cancel_active_infer(reason=str(reason or "smartblog_preempt"))
        except Exception as e:
            logging.warning("SmartBlog preempt cancel_active_infer failed: %s", e)
        job_task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(job_task, return_exceptions=True), timeout=float(smartblog_preempt_stop_timeout_sec()))
            return True
        except asyncio.TimeoutError:
            logging.warning(
                "SmartBlog preempt stop timeout: job=%s type=%s timeout=%ss",
                str(self._current_smartblog_job_id() or "-"),
                str(self._current_smartblog_job_type() or "-"),
                int(smartblog_preempt_stop_timeout_sec()),
            )
            return False

    async def _smartblog_preempt_after_delay(self, job: dict[str, Any], *, source: str) -> None:
        job_id = str(job.get("id") or "").strip()
        try:
            if not bool(smartblog_preempt_enabled()):
                return
            delay_sec = float(smartblog_preempt_delay_sec())
            if delay_sec > 0.0:
                logging.info(
                    "SmartBlog urgent preempt scheduled: wait=%ss incoming_id=%s incoming_type=%s source=%s",
                    int(delay_sec),
                    job_id or "-",
                    str(smartblog_job_type_value(job) or "-"),
                    str(source),
                )
                await asyncio.sleep(delay_sec)
            if self._stop.is_set():
                return
            if (not self._worker_busy()) or (not self._current_smartblog_job_is_preemptible()):
                return
            incoming_priority = int(smartblog_job_priority_value(job))
            active_priority = int(self._active_worker_priority())
            if incoming_priority >= active_priority:
                return
            claim = await self._claim_smartblog_job(job, source=f"{source}:preempt_claim", allow_while_busy=True)
            if claim is None:
                return
            current_job_id = str(self._current_smartblog_job_id() or "").strip()
            current_job_type = str(self._current_smartblog_job_type() or "").strip()
            stop_ok = await self._stop_current_smartblog_job_for_preempt(
                reason=f"preempt_for_{smartblog_job_type_value(claim) or job_id or 'urgent'}"
            )
            if not stop_ok:
                try:
                    await self._smartblog_api.release(job_id=str((((claim or {}).get('job') or {}).get('id') or job_id)))
                except Exception as e:
                    logging.warning("SmartBlog urgent release-back failed after stop timeout: err=%s", e)
                return
            if current_job_id:
                try:
                    await self._smartblog_api.release(job_id=current_job_id)
                    logging.warning(
                        "SmartBlog current job released for urgent preempt: released_id=%s released_type=%s incoming_id=%s incoming_type=%s",
                        current_job_id,
                        current_job_type or "-",
                        str((((claim or {}).get('job') or {}).get('id') or "-")),
                        str(smartblog_job_type_value(claim) or "-"),
                    )
                except Exception as e:
                    logging.warning(
                        "SmartBlog release current job failed after preempt stop: job=%s err=%s",
                        current_job_id,
                        e,
                    )
            await self._run_claimed_smartblog_job(claim, source=f"{source}:preempt_run")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.warning("SmartBlog urgent preempt flow failed: job=%s err=%s", job_id or "-", e)
        finally:
            current_task = asyncio.current_task()
            if getattr(self, "_smartblog_preempt_task", None) is current_task:
                self._clear_smartblog_preempt_tracking()

    async def _smartblog_poll_for_urgent_preempt_candidate(self, *, source: str) -> None:
        if not bool(smartblog_preempt_enabled()):
            return
        if bool(auto_update_drain_requested()):
            return
        if not bool(self._current_smartblog_job_is_preemptible()):
            return
        self._smartblog_mark_realtime_poll(source=str(source or "urgent-poll"))
        try:
            jobs = await self._smartblog_poll_supported_jobs(source=str(source or "urgent-poll"))
        except Exception as e:
            logging.warning("SmartBlog urgent preempt poll failed: %s", e)
            return
        urgent_jobs = smartblog_sort_jobs_for_claim(
            [dict(job) for job in list(jobs or []) if smartblog_supports_job(job) and smartblog_job_is_queue_candidate(job) and smartblog_is_urgent_job(job)]
        )
        if not urgent_jobs:
            return
        self._schedule_smartblog_preempt(urgent_jobs[0], source=str(source or "poll"))

    async def _smartblog_wait_before_claim(
        self,
        jobs: Sequence[dict[str, Any]] | None,
        *,
        source: str,
    ) -> bool:
        ordered_jobs = [dict(job) for job in list(jobs or []) if isinstance(job, dict)]
        source_s = str(source or "bootstrap").strip().lower()
        if source_s.startswith("realtime"):
            return True
        claim_delay_sec = float(smartblog_claim_after_poll_delay_sec())
        if not ordered_jobs or claim_delay_sec <= 0.0:
            return True
        logging.info(
            "SmartBlog claim delay: source=%s delay=%.1fs ordered=%s",
            str(source or "bootstrap"),
            float(claim_delay_sec),
            ",".join([str(job.get("id") or "-") for job in ordered_jobs]) or "-",
        )
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=float(claim_delay_sec))
            return False
        except asyncio.TimeoutError:
            return True

    def _on_smartblog_job_event(self, payload: Any) -> None:
        try:
            job_row, parsed_from, payload_type, payload_keys, data_keys, event_kind = self._smartblog_extract_realtime_job_row(payload)
            self._smartblog_mark_realtime_raw_event(
                event_kind=event_kind,
                parsed_from=parsed_from,
                payload_keys=payload_keys,
                data_keys=data_keys,
            )
            if isinstance(job_row, dict):
                logging.info(
                    "SmartBlog realtime raw event: payload_type=%s event=%s parsed_from=%s payload_keys=%s data_keys=%s id=%s type=%s status=%s",
                    str(payload_type or "-"),
                    str(event_kind or "-"),
                    str(parsed_from or "-"),
                    str(payload_keys or "-"),
                    str(data_keys or "-"),
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(smartblog_job_status_value(job_row) or "-"),
                )
            else:
                logging.warning(
                    "SmartBlog realtime raw event dropped: payload_type=%s event=%s payload_keys=%s data_keys=%s",
                    str(payload_type or "-"),
                    str(event_kind or "-"),
                    str(payload_keys or "-"),
                    str(data_keys or "-"),
                )
            if not isinstance(job_row, dict):
                return
            if not smartblog_supports_job(job_row):
                logging.info(
                    "Ignore SmartBlog realtime unsupported job: id=%s type=%s status=%s",
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(smartblog_job_status_value(job_row) or "-"),
                )
                return
            if not self._smartblog_job_runtime_supported(job_row, source="realtime"):
                return
            if not smartblog_job_is_queue_candidate(job_row):
                logging.info(
                    "Ignore SmartBlog realtime job with non-queued status: id=%s type=%s status=%s",
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(smartblog_job_status_value(job_row) or "-"),
                )
                return
            gate_reason = str(self._smartblog_job_event_gate_reason() or "ready").strip().lower()
            if gate_reason == "busy":
                incoming_priority = int(smartblog_job_priority_value(job_row))
                active_priority = int(self._active_worker_priority())
                if incoming_priority < active_priority:
                    if bool(smartblog_preempt_enabled()):
                        logging.warning(
                            "Deferred higher-priority SmartBlog job while worker busy: incoming_id=%s incoming_type=%s incoming_priority=%d active_source=%s active_type=%s active_priority=%d",
                            str(job_row.get("id") or "-"),
                            str(smartblog_job_type_value(job_row) or "-"),
                            incoming_priority,
                            str(getattr(self, "_active_source", "") or "-"),
                            str(self._active_worker_job_type() or "-"),
                            active_priority,
                        )
                        self._schedule_smartblog_preempt(job_row, source="realtime")
                    else:
                        logging.info(
                            "SmartBlog realtime job waits for idle worker; preempt disabled: incoming_id=%s incoming_type=%s active_source=%s active_type=%s",
                            str(job_row.get("id") or "-"),
                            str(smartblog_job_type_value(job_row) or "-"),
                            str(getattr(self, "_active_source", "") or "-"),
                            str(self._active_worker_job_type() or "-"),
                        )
                logging.info(
                    "Ignore SmartBlog realtime job while worker busy: id=%s type=%s status=%s",
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(job_row.get("status") or "-"),
                )
                return
            if gate_reason == "auto_update_pending":
                logging.info(
                    "Ignore SmartBlog realtime job while auto-update drain is pending: id=%s type=%s status=%s",
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(job_row.get("status") or "-"),
                )
                self._schedule_smartblog_post_drain_poll(source="realtime-event")
                return
            if gate_reason == "claim_lock":
                logging.info(
                    "Queue SmartBlog realtime job during claim handoff: id=%s type=%s status=%s",
                    str(job_row.get("id") or "-"),
                    str(smartblog_job_type_value(job_row) or "-"),
                    str(job_row.get("status") or "-"),
                )
                self._smartblog_mark_realtime_event()
                self._enqueue_smartblog_job(job_row)
                return
            logging.info(
                "SmartBlog realtime job event: id=%s type=%s status=%s",
                str(job_row.get("id") or "-"),
                str(smartblog_job_type_value(job_row) or "-"),
                str(job_row.get("status") or "-"),
            )
            self._smartblog_mark_realtime_event()
            self._enqueue_smartblog_job(job_row)
        except Exception as e:
            logging.warning("SmartBlog realtime callback failed: %s", e)

    async def _smartblog_bootstrap_scan(self, *, urgent_only: bool = False, source: str = "bootstrap") -> bool:
        if bool(auto_update_drain_requested()):
            logging.info("Skip SmartBlog bootstrap poll while auto-update drain is pending: source=%s", str(source or "bootstrap"))
            self._schedule_smartblog_post_drain_poll(source=str(source or "bootstrap"))
            return False
        if not self._smartblog_claim_gate_ready(source=str(source or "bootstrap")):
            return False
        self._smartblog_mark_realtime_poll(source=str(source or "bootstrap"))
        try:
            jobs = await self._smartblog_poll_supported_jobs(source=str(source or "bootstrap"))
        except Exception as e:
            logging.warning("SmartBlog bootstrap poll failed: %s", e)
            return False
        supported_jobs: list[dict[str, Any]] = []
        for job in list(jobs or []):
            if not isinstance(job, dict):
                continue
            if not smartblog_supports_job(job):
                continue
            if not smartblog_job_is_queue_candidate(job):
                continue
            checked_job = await self._smartblog_job_with_preclaim_runtime_hints(job, source=str(source or "bootstrap"))
            if checked_job is None:
                continue
            supported_jobs.append(dict(checked_job))
        if urgent_only:
            supported_jobs = [dict(job) for job in supported_jobs if smartblog_is_urgent_job(job)]
        ordered_jobs = smartblog_sort_jobs_for_claim(supported_jobs)
        logging.info(
            "SmartBlog bootstrap poll: jobs=%s supported=%s urgent_only=%d ordered=%s",
            len(list(jobs or [])),
            len(ordered_jobs),
            int(bool(urgent_only)),
            ",".join([str(smartblog_job_type_value(job) or "-") for job in ordered_jobs]) or "-",
        )
        if not await self._smartblog_wait_before_claim(ordered_jobs, source=str(source or "bootstrap")):
            return False
        for job in ordered_jobs:
            claimed = await self._claim_and_run_smartblog_job(job, source=str(source or "bootstrap"))
            if claimed:
                return True
        return False

    async def _claim_and_run_smartblog_job(self, job: dict[str, Any], *, source: str) -> bool:
        if not smartblog_supports_job(job):
            return False
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            return False
        if not await self._smartblog_wait_before_job_claim(job, source=str(source)):
            return False
        if not self._smartblog_claim_gate_ready(source=f"{source}:after-wait"):
            return False
        if bool(auto_update_drain_requested()):
            logging.info("Skip SmartBlog claim after delayed wait while auto-update drain is pending: id=%s", job_id)
            return False
        if self._worker_busy():
            logging.info("Skip SmartBlog claim after delayed wait while worker became busy: id=%s", job_id)
            return False
        if not smartblog_supports_job(job) or not smartblog_job_is_queue_candidate(job):
            return False
        claim = await self._claim_smartblog_job(job, source=str(source), allow_while_busy=False)
        if claim is None:
            return False
        await self._run_claimed_smartblog_job(claim, source=str(source))
        return True

    async def _smartblog_wait_before_job_claim(self, job: dict[str, Any], *, source: str) -> bool:
        if _smartblog_render_is_avatar_backup_job(job):
            delay_sec = float(_smartblog_render_avatar_backup_delay_sec())
            if delay_sec > 0.0:
                logging.info(
                    "SmartBlog render avatar backup standby claim: id=%s type=%s source=%s wait=%.1fs",
                    str(job.get("id") or "-"),
                    str(smartblog_job_type_value(job) or "-"),
                    str(source or "-"),
                    delay_sec,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay_sec)
                    return False
                except asyncio.TimeoutError:
                    if not smartblog_job_is_queue_candidate(job):
                        return False
                    return True
        if not smartblog_job_needs_delayed_claim(job):
            return True
        delay_sec = float(smartblog_delayed_claim_sec())
        if delay_sec <= 0.0:
            return True
        logging.info(
            "SmartBlog delayed standby claim: id=%s type=%s source=%s wait=%.1fs",
            str(job.get("id") or "-"),
            str(smartblog_job_type_value(job) or "-"),
            str(source or "-"),
            delay_sec,
        )
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay_sec)
            return False
        except asyncio.TimeoutError:
            return True

    async def _handle_smartblog_realtime_job_trigger(self, job_row: dict[str, Any]) -> bool:
        logging.info(
            "SmartBlog realtime event trigger: id=%s type=%s status=%s",
            str(job_row.get("id") or "-"),
            str(smartblog_job_type_value(job_row) or "-"),
            str(smartblog_job_status_value(job_row) or "-"),
        )
        direct_source = "realtime-urgent-direct" if smartblog_is_urgent_job(job_row) else "realtime-direct"
        claimed = await self._claim_and_run_smartblog_job(job_row, source=direct_source)
        if claimed:
            return True
        logging.info(
            "SmartBlog realtime direct claim missed; falling back to poll: id=%s type=%s status=%s source=%s",
            str(job_row.get("id") or "-"),
            str(smartblog_job_type_value(job_row) or "-"),
            str(smartblog_job_status_value(job_row) or "-"),
            direct_source,
        )
        return bool(await self._smartblog_bootstrap_scan(source="realtime-event"))

    async def _smartblog_polling_loop(self) -> None:
        interval_sec = float(smartblog_poll_interval_sec())
        logging.info(
            "SmartBlog polling loop started: interval=%.1fs job_types=%s",
            interval_sec,
            ",".join(smartblog_supported_job_types()) or "-",
        )
        while not self._stop.is_set():
            try:
                if self._worker_busy():
                    if bool(smartblog_preempt_enabled()):
                        await self._smartblog_poll_for_urgent_preempt_candidate(source="poll-loop-busy")
                else:
                    _ = await self._smartblog_bootstrap_scan(source="poll-loop")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.warning("SmartBlog polling loop failed: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                continue

    async def _smartblog_realtime_loop(self) -> None:
        try:
            from supabase import acreate_client
        except Exception as e:
            raise RuntimeError("supabase package is required for SmartBlog realtime.") from e

        supabase_url = smartblog_supabase_url()
        supabase_realtime_key = smartblog_supabase_service_role_key()
        safety_poll_sec = float(smartblog_realtime_safety_poll_sec())
        backoff_s = 1.0
        while not self._stop.is_set():
            supabase = None
            channel = None
            recover_reason = ""
            try:
                self._smartblog_loop_ref = asyncio.get_running_loop()
                self._smartblog_realtime_connected = False
                self._smartblog_realtime_status = "connecting"
                while not self._smartblog_job_q.empty():
                    try:
                        _ = self._smartblog_job_q.get_nowait()
                    except Exception:
                        break
                supabase = await acreate_client(supabase_url, supabase_realtime_key)
                channel = supabase.channel("smartblog-worker-jobs")
                if smartblog_realtime_debug_enabled():
                    channel = channel.on_postgres_changes(
                        event="*",
                        schema="public",
                        table="jobs",
                        callback=self._on_smartblog_realtime_debug_event,
                    )
                channel = channel.on_postgres_changes(
                    event="INSERT",
                    schema="public",
                    table="jobs",
                    callback=self._on_smartblog_job_event,
                ).on_postgres_changes(
                    event="UPDATE",
                    schema="public",
                    table="jobs",
                    callback=self._on_smartblog_job_event,
                )
                await channel.subscribe(self._handle_smartblog_realtime_subscribe_state)
                logging.info(
                    "SmartBlog realtime subscribed: table=jobs events=INSERT,UPDATE debug_all=%d safety_poll_sec=%s job_types=%s",
                    1 if smartblog_realtime_debug_enabled() else 0,
                    int(safety_poll_sec),
                    ",".join(smartblog_supported_job_types()),
                )
                backoff_s = 1.0
                _ = await self._smartblog_bootstrap_scan(source="bootstrap")

                while not self._stop.is_set():
                    try:
                        job_row = await asyncio.wait_for(
                            self._smartblog_job_q.get(),
                            timeout=float(safety_poll_sec),
                        )
                    except asyncio.TimeoutError:
                        if self._worker_busy():
                            if bool(smartblog_preempt_enabled()):
                                await self._smartblog_poll_for_urgent_preempt_candidate(source="safety-poll")
                            continue
                        logging.info("SmartBlog realtime safety poll triggered after %ss idle", int(safety_poll_sec))
                        _ = await self._smartblog_bootstrap_scan(source="safety-poll")
                        continue
                    if not isinstance(job_row, dict):
                        continue
                    if bool(job_row.get("__realtime_state__")):
                        recover_reason = str(job_row.get("__realtime_state__") or "channel_state").strip().lower()
                        error_text = str(job_row.get("__error__") or "").strip()
                        raise RuntimeError(
                            f"SmartBlog realtime channel state={recover_reason or 'unknown'} err={error_text or '-'}"
                        )
                    if bool(job_row.get("__worker_idle__")):
                        if not self._worker_busy():
                            logging.info(
                                "SmartBlog worker idle trigger: immediate bootstrap poll job_types=%s",
                                ",".join(smartblog_supported_job_types()),
                            )
                            _ = await self._smartblog_bootstrap_scan(source="post-job")
                        continue
                    if bool(job_row.get("__stop__")) and self._stop.is_set():
                        break
                    _ = await self._handle_smartblog_realtime_job_trigger(job_row)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as e:
                status = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                self._smartblog_mark_realtime_failure(state=f"http_{status}", error_text=str(e))
                recover_reason = recover_reason or f"http_{status}"
                if status in {401, 403}:
                    logging.warning(
                        "SmartBlog realtime unauthorized: set WORKER_API_KEY and SUPABASE_SERVICE_ROLE_KEY"
                    )
                    if not self._stop.is_set():
                        await self._smartblog_realtime_recover_once(reason=recover_reason)
                    await asyncio.sleep(max(backoff_s, 30.0))
                    backoff_s = min(30.0, max(1.0, backoff_s * 2.0))
                    continue
                logging.warning("SmartBlog realtime loop HTTP error: status=%s err=%s", status, e)
                if not self._stop.is_set():
                    await self._smartblog_realtime_recover_once(reason=recover_reason)
                await asyncio.sleep(backoff_s)
                backoff_s = min(30.0, max(1.0, backoff_s * 2.0))
            except Exception as e:
                self._smartblog_mark_realtime_failure(state="loop_error", error_text=str(e))
                recover_reason = recover_reason or "loop_error"
                logging.warning("SmartBlog realtime loop failed: %s", e)
                if not self._stop.is_set():
                    await self._smartblog_realtime_recover_once(reason=recover_reason)
                await asyncio.sleep(backoff_s)
                backoff_s = min(30.0, max(1.0, backoff_s * 2.0))
            finally:
                self._smartblog_loop_ref = None
                if not self._stop.is_set() and bool(self._smartblog_realtime_connected):
                    self._smartblog_mark_realtime_failure(state="disconnected")
                try:
                    if channel is not None:
                        await channel.unsubscribe()
                except Exception:
                    pass
                try:
                    if supabase is not None:
                        close_fn = getattr(supabase, "aclose", None)
                        if callable(close_fn):
                            maybe_awaitable = close_fn()
                            if asyncio.iscoroutine(maybe_awaitable):
                                await maybe_awaitable
                except Exception:
                    pass
