from __future__ import annotations

import asyncio
import logging
import os
import time

from avalife.core import engine as la
from avalife.core.heartbeat import ProcessHeartbeat, modeld_heartbeat_path, modeld_startup_phase_max_sec
from avalife.core.observability import boot_log_enabled, model_timing_enabled, runtime_profile_name
from avalife.model.server import ModelRuntimeServer, install_model_runtime_signal_handlers
from avalife.core.args import parse_runtime_args
from avalife.worker.current_run import record_current_torchrun_run


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configure_torch_acceleration_defaults() -> None:
    try:
        import torch
    except Exception:
        return

    precision = str(os.getenv("TORCH_FLOAT32_MATMUL_PRECISION", "high") or "high").strip().lower()
    if precision:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception as exc:
            logging.warning("Failed to set torch float32 matmul precision=%s: %s", precision, exc)

    try:
        torch.backends.cuda.matmul.allow_tf32 = _env_flag("TORCH_CUDA_MATMUL_ALLOW_TF32", True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = _env_flag("TORCH_CUDNN_ALLOW_TF32", True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = _env_flag("TORCH_CUDNN_BENCHMARK", True)
    except Exception:
        pass


def main() -> None:
    _configure_torch_acceleration_defaults()
    heartbeat: ProcessHeartbeat | None = None
    if str(os.getenv("LOCAL_RANK", "0") or "0").strip() == "0":
        run_dir = record_current_torchrun_run()
        heartbeat = ProcessHeartbeat(path=modeld_heartbeat_path(), component="modeld")
        heartbeat.set_state("starting", run_dir=(str(run_dir) if run_dir else None))
        heartbeat.start()
        if run_dir:
            logging.info("Active torchrun run recorded: %s", run_dir)
    try:
        startup_started_at = float(time.time())
        if heartbeat is not None:
            heartbeat.mark_startup_progress(
                state="starting",
                phase="model_runtime_bootstrap",
                stage="parse_runtime_args",
                step=1,
                total_steps=3,
                phase_started_at=startup_started_at,
                phase_timeout_sec=float(modeld_startup_phase_max_sec()),
            )
        args = parse_runtime_args()
        training_settings = la.training_config_parser(args.training_config)
        if heartbeat is not None:
            heartbeat.mark_startup_progress(
                state="starting",
                phase="model_runtime_bootstrap",
                stage="training_config_loaded",
                step=2,
                total_steps=3,
                phase_started_at=startup_started_at,
                phase_timeout_sec=float(modeld_startup_phase_max_sec()),
            )

        if boot_log_enabled():
            logging.info(
                "Model runtime profile: profile=%s model_log=%s timing=%d local_rank=%s",
                runtime_profile_name(),
                str(os.getenv("MODEL_LOG_LEVEL", "INFO") or "INFO").strip().upper(),
                1 if model_timing_enabled() else 0,
                str(os.getenv("LOCAL_RANK", "0") or "0").strip(),
            )
        logging.info("Initializing LiveAvatar model runtime...")
        pipeline_started_at = float(time.time())
        if heartbeat is not None:
            heartbeat.mark_startup_progress(
                state="initializing_pipeline",
                phase="model_runtime_pipeline_init",
                stage="begin",
                step=1,
                total_steps=9,
                phase_started_at=pipeline_started_at,
                phase_timeout_sec=float(modeld_startup_phase_max_sec()),
            )

        def _progress_cb(*, stage: str, step: int, total_steps: int, detail: str | None = None) -> None:
            if heartbeat is None:
                return
            heartbeat.mark_startup_progress(
                state="initializing_pipeline",
                phase="model_runtime_pipeline_init",
                stage=str(stage),
                step=int(step),
                total_steps=int(total_steps),
                phase_started_at=pipeline_started_at,
                phase_timeout_sec=float(modeld_startup_phase_max_sec()),
                startup_detail=(str(detail).strip() if detail else None),
            )

        la.initialize_pipeline(args, training_settings, progress_cb=_progress_cb)

        if la.global_rank != 0:
            la.worker_loop()
            return

        if heartbeat is not None:
            heartbeat.clear_startup_progress()
            heartbeat.set_state("ready")
        server = ModelRuntimeServer(heartbeat=heartbeat)
        install_model_runtime_signal_handlers(server)
        asyncio.run(server.serve_forever())
    finally:
        if heartbeat is not None:
            heartbeat.close()


if __name__ == "__main__":
    main()
