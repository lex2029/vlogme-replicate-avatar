from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from avalife.core.update_drain import auto_update_drain_requested, auto_update_drain_state


def _default_runtime_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime"


def _heartbeat_interval_sec() -> float:
    raw = str(os.getenv("WORKER_HEARTBEAT_INTERVAL_SEC", "5") or "5").strip()
    try:
        value = float(raw)
    except Exception:
        value = 5.0
    return max(0.5, min(60.0, float(value)))


def frontend_heartbeat_path() -> str:
    return str(
        Path(
            os.getenv(
                "WORKER_FRONTEND_HEARTBEAT_PATH",
                str(_default_runtime_dir() / "frontend_heartbeat.json"),
            )
        ).resolve()
    )


def modeld_heartbeat_path() -> str:
    return str(
        Path(
            os.getenv(
                "WORKER_MODELD_HEARTBEAT_PATH",
                str(_default_runtime_dir() / "modeld_heartbeat.json"),
            )
        ).resolve()
    )


def _env_bounded_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(float(low), min(float(high), float(value)))


def frontend_startup_phase_max_sec() -> float:
    return _env_bounded_float("WORKER_FRONTEND_STARTUP_PHASE_MAX_SEC", 900.0, low=30.0, high=7200.0)


def modeld_startup_phase_max_sec() -> float:
    return _env_bounded_float("WORKER_MODELD_STARTUP_PHASE_MAX_SEC", 900.0, low=30.0, high=7200.0)


class ProcessHeartbeat:
    def __init__(self, *, path: str, component: str, interval_sec: float | None = None) -> None:
        self.path = Path(path).resolve()
        self.component = str(component or "process").strip() or "process"
        self.interval_sec = float(interval_sec or _heartbeat_interval_sec())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        now = float(time.time())
        self._fields: dict[str, Any] = {
            "component": self.component,
            "pid": int(os.getpid()),
            "state": "starting",
            "started_at": now,
        }

    def set_state(self, state: str, **fields: Any) -> None:
        with self._lock:
            self._fields["state"] = str(state or "running").strip() or "running"
            for key, value in fields.items():
                if value is None:
                    self._fields.pop(str(key), None)
                else:
                    self._fields[str(key)] = value

    def clear_fields(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self._fields.pop(str(key), None)

    def mark_startup_progress(
        self,
        *,
        state: str,
        phase: str,
        stage: str,
        step: int | None = None,
        total_steps: int | None = None,
        phase_started_at: float | None = None,
        phase_timeout_sec: float | None = None,
        **fields: Any,
    ) -> None:
        now = float(time.time())
        phase_name = str(phase or "startup").strip() or "startup"
        stage_name = str(stage or "unknown").strip() or "unknown"
        with self._lock:
            prev_phase = str(self._fields.get("startup_phase") or "").strip()
            self._fields["state"] = str(state or "starting").strip() or "starting"
            self._fields["startup_phase"] = phase_name
            self._fields["startup_stage"] = stage_name
            if phase_started_at is not None:
                self._fields["startup_started_at"] = float(phase_started_at)
            elif prev_phase != phase_name or float(self._fields.get("startup_started_at") or 0.0) <= 0.0:
                self._fields["startup_started_at"] = now
            self._fields["startup_progress_at"] = now
            self._fields["startup_progress_seq"] = int(self._fields.get("startup_progress_seq") or 0) + 1
            if step is None:
                self._fields.pop("startup_step", None)
            else:
                self._fields["startup_step"] = int(step)
            if total_steps is None:
                self._fields.pop("startup_total_steps", None)
            else:
                self._fields["startup_total_steps"] = int(total_steps)
            if phase_timeout_sec is None:
                self._fields.pop("startup_phase_timeout_sec", None)
            else:
                self._fields["startup_phase_timeout_sec"] = float(phase_timeout_sec)
            for key, value in fields.items():
                if value is None:
                    self._fields.pop(str(key), None)
                else:
                    self._fields[str(key)] = value

    def clear_startup_progress(self) -> None:
        self.clear_fields(
            "startup_phase",
            "startup_stage",
            "startup_step",
            "startup_total_steps",
            "startup_started_at",
            "startup_progress_at",
            "startup_progress_seq",
            "startup_phase_timeout_sec",
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._fields)
        now = float(time.time())
        data["timestamp"] = now
        data["monotonic"] = float(time.monotonic())
        return data

    def write_once(self) -> None:
        payload = self.snapshot()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self.write_once()
            except Exception as e:
                logging.warning("Heartbeat write failed: component=%s err=%s", self.component, e)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.write_once()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.component}-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def close(self, final_state: str = "stopped") -> None:
        self.set_state(final_state)
        try:
            self.write_once()
        except Exception:
            pass
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


def _safe_queue_size(obj: Any) -> int:
    try:
        return int(obj.qsize())
    except Exception:
        return 0


def _safe_event_is_set(obj: Any) -> bool:
    try:
        return bool(obj is not None and obj.is_set())
    except Exception:
        return False


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_bool_call(obj: Any, method_name: str) -> bool | None:
    try:
        fn = getattr(obj, method_name, None)
        if callable(fn):
            return bool(fn())
    except Exception:
        return None
    return None


def _safe_float_call(obj: Any, method_name: str) -> float | None:
    try:
        fn = getattr(obj, method_name, None)
        if callable(fn):
            out = float(fn() or 0.0)
            return out if out > 0.0 else None
    except Exception:
        return None
    return None


def _safe_room_state(room: Any) -> str:
    try:
        state = getattr(room, "connection_state", None)
        if state is None:
            return ""
        name = getattr(state, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip().lower()
        text = str(state).strip().lower()
        return text
    except Exception:
        return ""


def _startup_seq_next(worker: Any) -> int:
    seq = int(getattr(worker, "_frontend_startup_progress_seq", 0) or 0) + 1
    setattr(worker, "_frontend_startup_progress_seq", int(seq))
    return int(seq)


def frontend_mark_startup_progress(
    worker: Any,
    *,
    phase: str,
    stage: str,
    step: int | None = None,
    total_steps: int | None = None,
    detail: str | None = None,
) -> None:
    now = float(time.time())
    phase_name = str(phase or "frontend_startup").strip() or "frontend_startup"
    stage_name = str(stage or "unknown").strip() or "unknown"
    setattr(worker, "_frontend_startup_active", True)
    prev_phase = str(getattr(worker, "_frontend_startup_phase", "") or "").strip()
    setattr(worker, "_frontend_startup_phase", phase_name)
    setattr(worker, "_frontend_startup_stage", stage_name)
    if prev_phase != phase_name or float(getattr(worker, "_frontend_startup_started_at", 0.0) or 0.0) <= 0.0:
        setattr(worker, "_frontend_startup_started_at", float(now))
    setattr(worker, "_frontend_startup_progress_at", float(now))
    setattr(worker, "_frontend_startup_progress_seq", _startup_seq_next(worker))
    setattr(worker, "_frontend_startup_step", (None if step is None else int(step)))
    setattr(worker, "_frontend_startup_total_steps", (None if total_steps is None else int(total_steps)))
    setattr(worker, "_frontend_startup_detail", (str(detail).strip() if detail else ""))
    heartbeat = getattr(worker, "_frontend_heartbeat", None)
    if isinstance(heartbeat, ProcessHeartbeat):
        heartbeat.mark_startup_progress(
            state="warming_up",
            phase=phase_name,
            stage=stage_name,
            step=step,
            total_steps=total_steps,
            phase_started_at=float(getattr(worker, "_frontend_startup_started_at", now) or now),
            phase_timeout_sec=float(frontend_startup_phase_max_sec()),
            startup_detail=(str(detail).strip() if detail else None),
        )


def frontend_clear_startup_progress(worker: Any) -> None:
    setattr(worker, "_frontend_startup_active", False)
    setattr(worker, "_frontend_startup_phase", "")
    setattr(worker, "_frontend_startup_stage", "")
    setattr(worker, "_frontend_startup_started_at", 0.0)
    setattr(worker, "_frontend_startup_progress_at", 0.0)
    setattr(worker, "_frontend_startup_progress_seq", 0)
    setattr(worker, "_frontend_startup_step", None)
    setattr(worker, "_frontend_startup_total_steps", None)
    setattr(worker, "_frontend_startup_detail", "")
    heartbeat = getattr(worker, "_frontend_heartbeat", None)
    if isinstance(heartbeat, ProcessHeartbeat):
        heartbeat.clear_startup_progress()


def frontend_worker_state_fields(worker: Any) -> tuple[str, dict[str, Any]]:
    now_mono = float(time.monotonic())
    control_plane = str(os.getenv("WORKER_CONTROL_PLANE", "avalife") or "avalife").strip().lower() or "avalife"
    lk_streamer = getattr(worker, "lk_streamer", None)
    channel_streamer = getattr(worker, "_channel_streamer", None)
    lk_room = getattr(worker, "lk_room", None)
    reply_busy = _safe_event_is_set(getattr(worker, "_reply_busy", None))
    busy = bool(
        str(getattr(worker, "current_publish_task_id", "") or "").strip()
        or lk_streamer is not None
        or channel_streamer is not None
        or reply_busy
    )
    transport = "livekit" if lk_streamer is not None else ("channel" if channel_streamer is not None else "idle")
    session_started_mono = _safe_float(getattr(worker, "_active_session_started_mono", 0.0))
    session_age_sec = (now_mono - session_started_mono) if session_started_mono > 0.0 else None
    livekit_state = _safe_room_state(lk_room) if lk_room is not None else ""
    livekit_remote_count = None
    if lk_room is not None:
        try:
            livekit_remote_count = int(getattr(worker, "_livekit_remote_count", 0) or 0)
        except Exception:
            livekit_remote_count = 0
    livekit_empty_age_sec = None
    livekit_last_remote_absent_mono = _safe_float(getattr(worker, "_livekit_last_remote_absent_mono", 0.0))
    if lk_room is not None and livekit_remote_count is not None and livekit_remote_count <= 0 and livekit_last_remote_absent_mono > 0.0:
        livekit_empty_age_sec = max(0.0, now_mono - livekit_last_remote_absent_mono)
    rtmp_enabled = _safe_bool_call(channel_streamer, "has_rtmp_sink")
    rtmp_alive = _safe_bool_call(channel_streamer, "rtmp_sink_alive")
    rtmp_last_alive_mono = _safe_float_call(channel_streamer, "rtmp_last_alive_mono")
    rtmp_last_alive_age_sec = None
    if rtmp_last_alive_mono is not None and rtmp_last_alive_mono > 0.0:
        rtmp_last_alive_age_sec = max(0.0, now_mono - float(rtmp_last_alive_mono))

    fields: dict[str, Any] = {
        "control_plane": control_plane,
        "busy": bool(busy),
        "auto_update_drain_requested": bool(auto_update_drain_requested()),
        "source": str(getattr(worker, "_active_source", "") or "").strip() or None,
        "transport": transport,
        "session_id": str(getattr(worker, "current_session_id", "") or "").strip() or None,
        "publish_task_id": str(getattr(worker, "current_publish_task_id", "") or "").strip() or None,
        "session_age_sec": None if session_age_sec is None else round(float(session_age_sec), 3),
        "reply_busy": bool(reply_busy),
        "comment_qsize": int(_safe_queue_size(getattr(worker, "_comment_q", None))),
        "realtime_qsize": int(_safe_queue_size(getattr(worker, "_realtime_task_q", None))),
        "livekit_state": livekit_state or None,
        "livekit_remote_count": livekit_remote_count,
        "livekit_empty_age_sec": None
        if livekit_empty_age_sec is None
        else round(float(livekit_empty_age_sec), 3),
        "rtmp_enabled": None if rtmp_enabled is None else bool(rtmp_enabled),
        "rtmp_alive": None if rtmp_alive is None else bool(rtmp_alive),
        "rtmp_last_alive_age_sec": None
        if rtmp_last_alive_age_sec is None
        else round(float(rtmp_last_alive_age_sec), 3),
        "chat_poll_enabled": bool(getattr(worker, "_chat_poll_enabled", False))
        if hasattr(worker, "_chat_poll_enabled")
        else None,
        "chat_last_ok_age_sec": None,
        "progress_last_ok_age_sec": None,
        "session_failed": bool(getattr(worker, "_session_failed", False))
        if hasattr(worker, "_session_failed")
        else None,
        "session_fail_reason": str(getattr(worker, "_session_fail_reason", "") or "").strip() or None,
    }
    if bool(fields["auto_update_drain_requested"]):
        drain = auto_update_drain_state()
        fields["auto_update_drain_target"] = str(drain.get("target") or "").strip() or None
        fields["auto_update_drain_reason"] = str(drain.get("reason") or "").strip() or None
        if isinstance(drain.get("age_sec"), (int, float)):
            fields["auto_update_drain_age_sec"] = round(float(drain.get("age_sec")), 3)
    if lk_streamer is not None:
        try:
            heartbeat_payload = getattr(lk_streamer, "heartbeat_payload", None)
            if callable(heartbeat_payload):
                fields.update(dict(heartbeat_payload()))
        except Exception:
            pass

    last_chat_ok_mono = _safe_float(getattr(worker, "_last_chat_ok_mono", 0.0))
    if last_chat_ok_mono > 0.0:
        fields["chat_last_ok_age_sec"] = round(max(0.0, now_mono - last_chat_ok_mono), 3)
    last_progress_ok_mono = _safe_float(getattr(worker, "_last_progress_ok_mono", 0.0))
    if last_progress_ok_mono > 0.0:
        fields["progress_last_ok_age_sec"] = round(max(0.0, now_mono - last_progress_ok_mono), 3)

    smartblog_rt_status = str(getattr(worker, "_smartblog_realtime_status", "") or "").strip()
    if smartblog_rt_status:
        fields["smartblog_finalize_pending_count"] = len(getattr(worker, "_smartblog_finalize_tasks", set()) or set())
        fields["smartblog_finalize_completed_count"] = int(
            getattr(worker, "_smartblog_finalize_completed_count", 0) or 0
        )
        fields["smartblog_finalize_failed_count"] = int(
            getattr(worker, "_smartblog_finalize_failed_count", 0) or 0
        )
        fields["smartblog_finalize_last_error"] = str(getattr(worker, "_smartblog_finalize_last_error", "") or "").strip() or None
        finalize_last_error_mono = _safe_float(getattr(worker, "_smartblog_finalize_last_error_mono", 0.0))
        if finalize_last_error_mono > 0.0:
            fields["smartblog_finalize_last_error_age_sec"] = round(max(0.0, now_mono - finalize_last_error_mono), 3)
        finalize_last_complete_mono = _safe_float(getattr(worker, "_smartblog_finalize_last_complete_mono", 0.0))
        if finalize_last_complete_mono > 0.0:
            fields["smartblog_finalize_last_complete_age_sec"] = round(
                max(0.0, now_mono - finalize_last_complete_mono),
                3,
            )
        fields["smartblog_realtime_status"] = smartblog_rt_status
        fields["smartblog_realtime_connected"] = bool(getattr(worker, "_smartblog_realtime_connected", False))
        fields["smartblog_realtime_reconnect_count"] = int(getattr(worker, "_smartblog_realtime_reconnect_count", 0) or 0)
        fields["smartblog_realtime_recovery_count"] = int(getattr(worker, "_smartblog_realtime_recovery_count", 0) or 0)
        fields["smartblog_realtime_last_poll_source"] = str(
            getattr(worker, "_smartblog_realtime_last_poll_source", "") or ""
        ).strip() or None
        rt_last_subscribed_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_subscribed_mono", 0.0))
        if rt_last_subscribed_mono > 0.0:
            fields["smartblog_realtime_last_subscribed_age_sec"] = round(max(0.0, now_mono - rt_last_subscribed_mono), 3)
        fields["smartblog_realtime_raw_event_count"] = int(getattr(worker, "_smartblog_realtime_raw_event_count", 0) or 0)
        fields["smartblog_realtime_job_event_count"] = int(getattr(worker, "_smartblog_realtime_job_event_count", 0) or 0)
        fields["smartblog_realtime_debug_event_count"] = int(getattr(worker, "_smartblog_realtime_debug_event_count", 0) or 0)
        fields["smartblog_realtime_last_event_kind"] = str(
            getattr(worker, "_smartblog_realtime_last_event_kind", "") or ""
        ).strip() or None
        fields["smartblog_realtime_last_parsed_from"] = str(
            getattr(worker, "_smartblog_realtime_last_parsed_from", "") or ""
        ).strip() or None
        fields["smartblog_realtime_last_payload_keys"] = str(
            getattr(worker, "_smartblog_realtime_last_payload_keys", "") or ""
        ).strip() or None
        fields["smartblog_realtime_last_data_keys"] = str(
            getattr(worker, "_smartblog_realtime_last_data_keys", "") or ""
        ).strip() or None
        rt_last_raw_event_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_raw_event_mono", 0.0))
        if rt_last_raw_event_mono > 0.0:
            fields["smartblog_realtime_last_raw_event_age_sec"] = round(max(0.0, now_mono - rt_last_raw_event_mono), 3)
        rt_last_event_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_event_mono", 0.0))
        if rt_last_event_mono > 0.0:
            fields["smartblog_realtime_last_event_age_sec"] = round(max(0.0, now_mono - rt_last_event_mono), 3)
        rt_last_poll_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_poll_mono", 0.0))
        if rt_last_poll_mono > 0.0:
            fields["smartblog_realtime_last_poll_age_sec"] = round(max(0.0, now_mono - rt_last_poll_mono), 3)
        rt_last_error = str(getattr(worker, "_smartblog_realtime_last_error", "") or "").strip()
        fields["smartblog_realtime_last_error"] = rt_last_error or None
        rt_last_error_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_error_mono", 0.0))
        if rt_last_error_mono > 0.0:
            fields["smartblog_realtime_last_error_age_sec"] = round(max(0.0, now_mono - rt_last_error_mono), 3)
        rt_last_disconnect_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_disconnect_mono", 0.0))
        if rt_last_disconnect_mono > 0.0:
            fields["smartblog_realtime_last_disconnect_age_sec"] = round(max(0.0, now_mono - rt_last_disconnect_mono), 3)
        rt_last_recovery_mono = _safe_float(getattr(worker, "_smartblog_realtime_last_recovery_mono", 0.0))
        if rt_last_recovery_mono > 0.0:
            fields["smartblog_realtime_last_recovery_age_sec"] = round(max(0.0, now_mono - rt_last_recovery_mono), 3)
        fields["smartblog_realtime_last_recovery_reason"] = str(
            getattr(worker, "_smartblog_realtime_last_recovery_reason", "") or ""
        ).strip() or None

    startup_active = bool(getattr(worker, "_frontend_startup_active", False))
    startup_progress_at = _safe_float(getattr(worker, "_frontend_startup_progress_at", 0.0))
    startup_started_at = _safe_float(getattr(worker, "_frontend_startup_started_at", 0.0))
    if startup_active:
        fields["startup_phase"] = str(getattr(worker, "_frontend_startup_phase", "") or "").strip() or None
        fields["startup_stage"] = str(getattr(worker, "_frontend_startup_stage", "") or "").strip() or None
        fields["startup_step"] = getattr(worker, "_frontend_startup_step", None)
        fields["startup_total_steps"] = getattr(worker, "_frontend_startup_total_steps", None)
        fields["startup_progress_seq"] = int(getattr(worker, "_frontend_startup_progress_seq", 0) or 0)
        fields["startup_detail"] = str(getattr(worker, "_frontend_startup_detail", "") or "").strip() or None
        fields["startup_phase_timeout_sec"] = float(frontend_startup_phase_max_sec())
        if startup_progress_at > 0.0:
            fields["startup_progress_age_sec"] = round(max(0.0, float(time.time()) - startup_progress_at), 3)
        if startup_started_at > 0.0:
            fields["startup_phase_age_sec"] = round(max(0.0, float(time.time()) - startup_started_at), 3)

    state = "warming_up" if startup_active else ("busy" if busy else "running")
    return state, fields


async def run_frontend_worker_heartbeat_loop(worker: Any, heartbeat: ProcessHeartbeat) -> None:
    interval = max(0.5, min(15.0, float(_heartbeat_interval_sec())))
    while True:
        try:
            state, fields = frontend_worker_state_fields(worker)
            heartbeat.set_state(state, **fields)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.warning("Frontend heartbeat loop update failed: %s", e)
        await asyncio.sleep(interval)
