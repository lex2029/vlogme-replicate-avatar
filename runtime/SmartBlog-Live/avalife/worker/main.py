from __future__ import annotations

from .common import *
from avalife.core.heartbeat import (
    ProcessHeartbeat,
    frontend_heartbeat_path,
    frontend_mark_startup_progress,
    run_frontend_worker_heartbeat_loop,
)
from avalife.core.observability import boot_log_enabled, runtime_profile_name, worker_timing_enabled
from .current_log import install_frontend_file_logging
from avalife.core.args import parse_runtime_args

def _install_signal_handlers(worker: Any) -> None:
    def _handle_stop(sig: int, _frame: Any) -> None:
        logging.warning("Signal %s received; stopping...", sig)
        worker.request_stop()

    def _handle_reload(sig: int, _frame: Any) -> None:
        ok, msg = worker.hot_reload_non_model_logic()
        logging.warning("Signal %s received; hot-reload non-model logic result=%s detail=%s", sig, int(ok), msg)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle_stop)
        except Exception:
            pass
    try:
        signal.signal(signal.SIGHUP, _handle_reload)
    except Exception:
        pass


def main() -> None:
    heartbeat = ProcessHeartbeat(path=frontend_heartbeat_path(), component="frontend")
    heartbeat.set_state("starting")
    heartbeat.start()
    try:
        log_path = install_frontend_file_logging()
        logging.info("Active frontend log recorded: %s", log_path)
        heartbeat.set_state("warming_up", frontend_log=str(log_path))
        if boot_log_enabled():
            logging.info(
                "Frontend profile: profile=%s worker_log=%s timing=%d",
                runtime_profile_name(),
                str(os.getenv("WORKER_LOG_LEVEL", "INFO") or "INFO").strip().upper(),
                1 if worker_timing_enabled() else 0,
            )
        args = parse_runtime_args()

        control_plane = str(os.getenv("WORKER_CONTROL_PLANE", "smartblog") or "smartblog").strip().lower()
        if control_plane != "smartblog":
            raise RuntimeError(f"B300 render-only branch supports only WORKER_CONTROL_PLANE=smartblog, got {control_plane!r}")
        from .smartblog_render import SmartBlogRenderOnlyWorker

        worker = SmartBlogRenderOnlyWorker(args=args)
        worker._frontend_heartbeat = heartbeat
        _install_signal_handlers(worker)
        frontend_mark_startup_progress(
            worker,
            phase="frontend_bootstrap",
            stage="worker_constructed",
            step=1,
            total_steps=1,
            detail=control_plane,
        )

        async def _run_worker() -> None:
            hb_task = asyncio.create_task(
                run_frontend_worker_heartbeat_loop(worker, heartbeat),
                name="frontend-heartbeat-loop",
            )
            try:
                await worker.run_forever()
            finally:
                hb_task.cancel()
                await asyncio.gather(hb_task, return_exceptions=True)

        asyncio.run(_run_worker())
    finally:
        heartbeat.close()


if __name__ == "__main__":
    main()
