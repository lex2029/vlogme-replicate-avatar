from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist

from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.distributed.util import init_distributed_group
from .observability import model_log_level, model_other_rank_log_level


@dataclass
class ModelPipelineRuntimeState:
    pipeline: Any
    args: Any
    cfg: Any
    training_settings: Any
    rank: int
    world_size: int
    save_rank: int
    control_group: Any


def init_model_runtime_logging(rank: int) -> None:
    level = int(model_log_level()) if int(rank) == 0 else int(model_other_rank_log_level())
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )


def _cleanup_existing_pipeline(existing_pipeline: Any) -> None:
    if existing_pipeline is None:
        return
    logging.info("Cleaning up existing pipeline to free GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()


def _override_runtime_sample_fps(args: Any, required_worker_fps: int) -> Any:
    cfg = WAN_CONFIGS[args.task]
    try:
        old_fps = int(getattr(cfg, "sample_fps", 0) or 0)
    except Exception:
        old_fps = 0
    if int(old_fps) != int(required_worker_fps):
        cfg.sample_fps = int(required_worker_fps)
        logging.info(
            "Overriding model sample_fps: %d -> %d",
            int(old_fps),
            int(required_worker_fps),
        )
    raw_block = str(os.getenv("SMARTBLOG_WAN_NUM_FRAMES_PER_BLOCK", "") or "").strip()
    if raw_block:
        try:
            block_frames = int(raw_block)
        except Exception as e:
            raise RuntimeError(f"Invalid SMARTBLOG_WAN_NUM_FRAMES_PER_BLOCK={raw_block!r}") from e
        block_frames = max(1, int(block_frames))
        old_block = int(getattr(cfg, "num_frames_per_block", 0) or 0)
        if int(old_block) != int(block_frames):
            cfg.num_frames_per_block = int(block_frames)
            logging.info(
                "Overriding model num_frames_per_block: %d -> %d",
                int(old_block),
                int(block_frames),
            )
    return cfg


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _select_pipeline_class(args: Any, world_size: int):
    if int(world_size) > 1:
        expected_world_size = int(args.num_gpus_dit) + (1 if args.enable_vae_parallel else 0)
        assert int(world_size) == int(expected_world_size), (
            f"Invalid distributed setup: got WORLD_SIZE={world_size}, "
            f"but expected {expected_world_size} for num_gpus_dit={args.num_gpus_dit} "
            f"and enable_vae_parallel={args.enable_vae_parallel}."
        )
        args.single_gpu = False
        from liveavatar.models.wan.live_stream_pipeline import WanS2V

        logging.info("Using TPP distributed inference.")
        return WanS2V

    if _env_flag("WORKER_FORCE_LIVE_STREAM_PIPELINE", False):
        args.enable_vae_parallel = False
        args.num_gpus_dit = 1
        args.single_gpu = False
        from liveavatar.models.wan.live_stream_pipeline import WanS2V

        logging.info("Using live-stream pipeline on single GPU.")
        return WanS2V

    assert not (args.t5_fsdp or args.dit_fsdp), (
        "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
    )
    assert not (args.ulysses_size > 1), (
        "sequence parallel are not supported in non-distributed environments."
    )
    args.enable_vae_parallel = False
    args.num_gpus_dit = 1
    args.single_gpu = True
    from liveavatar.models.wan.causal_s2v_pipeline import WanS2V

    logging.info("Using single GPU inference with offload mode: %s", args.offload_model)
    return WanS2V


def _ensure_control_group(world_size: int, dist_timeout: timedelta, current_control_group: Any) -> Any:
    if int(world_size) <= 1:
        return current_control_group
    if current_control_group is not None:
        return current_control_group
    try:
        control_group = dist.new_group(backend="gloo", timeout=dist_timeout)
        logging.info("Initialized gloo control group for command broadcasts.")
        return control_group
    except Exception as exc:
        raise RuntimeError(f"Failed to create required gloo control group: {exc}") from exc


def _init_nccl_default_group(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    dist_timeout: timedelta,
) -> None:
    if dist.is_initialized():
        return

    backend = "nccl"
    if int(world_size) <= 1 and os.getenv("SMARTBLOG_SINGLE_GPU_DIST_BACKEND", "gloo").strip().lower() == "gloo":
        backend = "gloo"
    kwargs = {
        "backend": backend,
        "init_method": "env://",
        "rank": int(rank),
        "world_size": int(world_size),
        "timeout": dist_timeout,
    }
    if backend == "nccl" and _env_flag("NCCL_INIT_DEVICE_ID", False):
        try:
            dist.init_process_group(**kwargs, device_id=torch.device("cuda", int(local_rank)))
            return
        except TypeError:
            pass

    dist.init_process_group(**kwargs)


def _prewarm_nccl_p2p(*, world_size: int, local_rank: int) -> None:
    if int(world_size) != 2 or not dist.is_initialized() or not _env_flag("NCCL_P2P_PREWARM", True):
        return
    try:
        rank = int(dist.get_rank())
        token = torch.empty((1,), device=torch.device("cuda", int(local_rank)), dtype=torch.int32)
        token.fill_(rank)
        if rank == 0:
            dist.send(token, dst=1)
            dist.recv(token, src=1)
        else:
            dist.recv(token, src=0)
            dist.send(token, dst=0)
        torch.cuda.synchronize(torch.device("cuda", int(local_rank)))
    except Exception as exc:
        logging.warning("NCCL P2P prewarm failed; continuing with lazy init: %s", exc)


def _load_lora_if_needed(pipeline: Any, args: Any, training_settings: Any) -> None:
    if not args.load_lora or args.lora_path_dmd is None:
        return
    logging.info(
        "Loading LoRA: path=%s, rank=%s, alpha=%s",
        args.lora_path_dmd,
        training_settings["lora_rank"],
        training_settings["lora_alpha"],
    )
    pipeline.noise_model = pipeline.add_lora_to_model(
        pipeline.noise_model,
        lora_rank=training_settings["lora_rank"],
        lora_alpha=training_settings["lora_alpha"],
        lora_target_modules=training_settings["lora_target_modules"],
        init_lora_weights=training_settings["init_lora_weights"],
        pretrained_lora_path=args.lora_path_dmd,
        load_lora_weight_only=False,
    )


def _enable_fp8_if_requested(pipeline: Any, args: Any) -> None:
    if not args.fp8:
        return
    if not hasattr(torch, "_scaled_mm"):
        logging.info("skip fp8_linear, Please update torch vision ")
        return
    from liveavatar.utils.fp8_linear import FP8ScaleLinear, replace_linear_with_scaled_fp8

    rectangular_mode = str(os.getenv("LIVEAVATAR_FP8_RECTANGULAR_MODE", "any") or "any").strip().lower()
    logging.info("Enabling FP8 linear replacement: rectangular_mode=%s", rectangular_mode)
    replace_linear_with_scaled_fp8(
        pipeline.noise_model,
        ignore_keys=[
            "text_embedding",
            "time_embedding",
            "time_projection",
            "head.head",
            "casual_audio_encoder.encoder.final_linear",
        ],
        only_rectangular=True,
        rectangular_mode=rectangular_mode,
    )
    try:
        fp8_layers = sum(1 for mod in pipeline.noise_model.modules() if isinstance(mod, FP8ScaleLinear))
        logging.info("FP8 linear replacement complete: layers=%d rectangular_mode=%s", int(fp8_layers), rectangular_mode)
    except Exception:
        logging.exception("FP8 linear replacement layer count failed")


def _resolve_save_rank(args: Any, world_size: int) -> int:
    if args.enable_vae_parallel:
        return int(args.num_gpus_dit)
    if int(world_size) == 1:
        return 0
    return int(args.num_gpus_dit) - 1


def _emit_init_progress(
    progress_cb: Callable[..., None] | None,
    *,
    stage: str,
    step: int,
    total_steps: int,
    detail: str | None = None,
) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(stage=str(stage), step=int(step), total_steps=int(total_steps), detail=(str(detail) if detail else None))
    except Exception:
        pass


@contextmanager
def _init_progress_keepalive(
    progress_cb: Callable[..., None] | None,
    *,
    stage: str,
    step: int,
    total_steps: int,
    detail: str | None = None,
    pulse_sec: float = 15.0,
    max_sec: float = 600.0,
):
    _emit_init_progress(progress_cb, stage=stage, step=step, total_steps=total_steps, detail=detail)
    if progress_cb is None or float(max_sec) <= 0.0:
        yield
        return
    stop_ev = threading.Event()
    started = float(time.monotonic())

    def _run() -> None:
        while not stop_ev.wait(float(max(1.0, pulse_sec))):
            if (float(time.monotonic()) - started) >= float(max_sec):
                break
            _emit_init_progress(progress_cb, stage=stage, step=step, total_steps=total_steps, detail=detail)

    thread = threading.Thread(target=_run, name=f"model-init-{stage}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_ev.set()
        thread.join(timeout=1.0)


def initialize_pipeline_runtime(
    args: Any,
    training_settings: Any,
    *,
    current_pipeline: Any,
    current_control_group: Any,
    required_worker_fps: int,
    progress_cb: Callable[..., None] | None = None,
) -> ModelPipelineRuntimeState:
    total_steps = 9
    _emit_init_progress(progress_cb, stage="cleanup_previous_pipeline", step=1, total_steps=total_steps)
    _cleanup_existing_pipeline(current_pipeline)

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if int(world_size) == 1:
        rank = 0

    init_model_runtime_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if int(world_size) > 1 else True
        logging.info("offload_model is not specified, set to %s.", args.offload_model)

    torch.cuda.set_device(local_rank)
    _emit_init_progress(progress_cb, stage="cuda_device_selected", step=2, total_steps=total_steps)

    dist_timeout = timedelta(days=int(os.getenv("DIST_TIMEOUT_DAYS", "365")))
    _init_nccl_default_group(
        rank=int(rank),
        world_size=int(world_size),
        local_rank=int(local_rank),
        dist_timeout=dist_timeout,
    )
    _prewarm_nccl_p2p(world_size=int(world_size), local_rank=int(local_rank))
    _emit_init_progress(progress_cb, stage="distributed_initialized", step=3, total_steps=total_steps)

    control_group = _ensure_control_group(world_size, dist_timeout, current_control_group)
    if args.ulysses_size > 1:
        assert int(world_size) > 1, "Sequence parallel requires distributed world_size > 1."
        assert int(args.ulysses_size) <= int(world_size), (
            f"ulysses_size={args.ulysses_size} must be <= world_size={world_size}"
        )
        assert int(world_size) % int(args.ulysses_size) == 0, (
            f"world_size={world_size} must be divisible by ulysses_size={args.ulysses_size}"
        )
        logging.info(
            "Initializing sequence parallel group: ulysses_size=%d world_size=%d",
            int(args.ulysses_size),
            int(world_size),
        )
        init_distributed_group()

    cfg = _override_runtime_sample_fps(args, int(required_worker_fps))
    _emit_init_progress(progress_cb, stage="runtime_configured", step=4, total_steps=total_steps)
    logging.info("Pipeline initialization args: %s", args)
    logging.info("Model config: %s", cfg)
    try:
        from liveavatar.models.wan.wan_2_2.modules import attention as wan_attn

        flash2 = int(bool(getattr(wan_attn, "FLASH_ATTN_2_AVAILABLE", False)))
        flash3 = int(bool(getattr(wan_attn, "FLASH_ATTN_3_AVAILABLE", False)))
        cudnn_sdpa_effective = int(bool(getattr(wan_attn, "_cudnn_sdpa_available", lambda: False)()))
        torch_cudnn_sdp = getattr(wan_attn, "_torch_cudnn_sdp_enabled", lambda: None)()
        torch_flash_sdp = getattr(wan_attn, "_torch_flash_sdp_enabled", lambda: None)()
        torch_mem_sdp = getattr(wan_attn, "_torch_mem_efficient_sdp_enabled", lambda: None)()
        torch_math_sdp = getattr(wan_attn, "_torch_math_sdp_enabled", lambda: None)()
    except Exception:
        flash2 = -1
        flash3 = -1
        cudnn_sdpa_effective = -1
        torch_cudnn_sdp = None
        torch_flash_sdp = None
        torch_mem_sdp = None
        torch_math_sdp = None
    logging.info(
        "Acceleration runtime: compile=%s fp8=%s flash_disabled=%s cudnn_disabled=%s math_only=%s eager_attention=%s cudnn_effective=%s torch_cudnn_sdp=%s torch_flash_sdp=%s torch_mem_sdp=%s torch_math_sdp=%s flash2=%s flash3=%s cuda=%s cudnn=%s",
        int(_env_flag("ENABLE_COMPILE", default=False)),
        int(bool(getattr(args, "fp8", False))),
        int(_env_flag("LIVEAVATAR_DISABLE_FLASH_ATTN", default=False)),
        int(_env_flag("LIVEAVATAR_DISABLE_CUDNN_ATTN", default=False)),
        int(_env_flag("LIVEAVATAR_FORCE_TORCH_SDPA_MATH", default=False)),
        int(_env_flag("LIVEAVATAR_FORCE_EAGER_ATTN", default=False)),
        cudnn_sdpa_effective,
        "n/a" if torch_cudnn_sdp is None else int(bool(torch_cudnn_sdp)),
        "n/a" if torch_flash_sdp is None else int(bool(torch_flash_sdp)),
        "n/a" if torch_mem_sdp is None else int(bool(torch_mem_sdp)),
        "n/a" if torch_math_sdp is None else int(bool(torch_math_sdp)),
        flash2,
        flash3,
        str(torch.version.cuda),
        str(torch.backends.cudnn.version()),
    )

    if dist.is_initialized() and int(world_size) > 1:
        base_seed = [args.base_seed] if int(rank) == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0, group=control_group)
        args.base_seed = base_seed[0]

    WanS2V = _select_pipeline_class(args, world_size)
    _emit_init_progress(progress_cb, stage="pipeline_class_selected", step=5, total_steps=total_steps)
    logging.info("Creating WanS2V pipeline...")
    with _init_progress_keepalive(
        progress_cb,
        stage="pipeline_construction",
        step=6,
        total_steps=total_steps,
        detail="creating WanS2V pipeline",
        pulse_sec=15.0,
        max_sec=float(max(60.0, min(7200.0, float(os.getenv("WORKER_MODELD_STARTUP_PHASE_MAX_SEC", "900") or "900")))),
    ):
        pipeline = WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            merged_noise_model_dir=getattr(args, "merged_noise_model_dir", None),
            device_id=local_rank,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            sp_size=args.ulysses_size,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            single_gpu=args.single_gpu,
            offload_kv_cache=args.offload_kv_cache,
        )
    _emit_init_progress(progress_cb, stage="pipeline_constructed", step=7, total_steps=total_steps)

    _load_lora_if_needed(pipeline, args, training_settings)
    _emit_init_progress(progress_cb, stage="lora_loaded", step=8, total_steps=total_steps)
    _enable_fp8_if_requested(pipeline, args)
    _emit_init_progress(progress_cb, stage="pipeline_initialized", step=9, total_steps=total_steps)

    save_rank = _resolve_save_rank(args, world_size)
    logging.info("Pipeline initialized successfully!")
    return ModelPipelineRuntimeState(
        pipeline=pipeline,
        args=args,
        cfg=cfg,
        training_settings=training_settings,
        rank=int(rank),
        world_size=int(world_size),
        save_rank=int(save_rank),
        control_group=control_group,
    )
