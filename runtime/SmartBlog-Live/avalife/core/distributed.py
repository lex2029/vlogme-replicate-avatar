from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch.distributed as dist


@dataclass(frozen=True)
class DistributedRuntimeState:
    args: Any
    training_settings: Any
    rank: int
    world_size: int
    save_rank: int
    control_group: Any


def run_single_sample(
    prompt: str,
    image_path: str,
    audio_path: str,
    num_clip: int,
    sample_steps: int,
    sample_guide_scale: float,
    infer_frames: int,
    size: str,
    base_seed: int,
    sample_solver: str,
    face_restore: float = 0.0,
    background_restore: float = 0.0,
    *,
    job_id: str | None,
    enable_live_hls: bool,
    live_raw_dir: str | None,
    state: DistributedRuntimeState,
    maybe_square_crop_ref_image: Callable[[str, str, str | None], str],
    run_inference_computation: Callable[..., Any],
    broadcast_command_fn: Callable[..., None],
    build_infer_command_fn: Callable[..., Any],
    build_idle_command_fn: Callable[[], Any],
    save_live_raw_mp4: bool = False,
    video_prompt: str = "",
    negative_prompt: str = "",
    idle_prompt: str = "",
    stream_file_output_path: str = "",
    stream_file_output_width: int = 0,
    stream_file_output_height: int = 0,
    stream_file_output_fps: float = 0.0,
    stream_file_trim_duration_sec: float = 0.0,
    stream_file_interpolation: str = "",
    tpp_cfg_mode: str = "",
    lipsync_audio_path: str = "",
) -> Any:
    image_path = maybe_square_crop_ref_image(str(image_path), str(size), job_id=str(job_id) if job_id else None)

    if dist.is_initialized() and int(state.rank) == 0:
        broadcast_command_fn(
            cmd=build_infer_command_fn(
                prompt=prompt,
                image_path=image_path,
                audio_path=audio_path,
                num_clip=num_clip,
                sample_steps=sample_steps,
                sample_guide_scale=sample_guide_scale,
                infer_frames=infer_frames,
                size=size,
                base_seed=base_seed,
                sample_solver=sample_solver,
                live_raw_dir=live_raw_dir,
                face_restore=face_restore,
                background_restore=background_restore,
                job_id=job_id,
                enable_live_hls=enable_live_hls,
                save_live_raw_mp4=bool(save_live_raw_mp4),
                video_prompt=str(video_prompt or ""),
                negative_prompt=str(negative_prompt or ""),
                idle_prompt=str(idle_prompt or ""),
                stream_file_output_path=str(stream_file_output_path or ""),
                stream_file_output_width=int(stream_file_output_width or 0),
                stream_file_output_height=int(stream_file_output_height or 0),
                stream_file_output_fps=float(stream_file_output_fps or 0.0),
                stream_file_trim_duration_sec=float(stream_file_trim_duration_sec or 0.0),
                stream_file_interpolation=str(stream_file_interpolation or ""),
                tpp_cfg_mode=str(tpp_cfg_mode or ""),
                lipsync_audio_path=str(lipsync_audio_path or ""),
            ),
            src_rank=0,
            control_group=state.control_group,
        )
        logging.info("[Rank 0] Broadcast inference inputs to other ranks")

    try:
        return run_inference_computation(
            prompt,
            image_path,
            audio_path,
            num_clip,
            sample_steps,
            sample_guide_scale,
            infer_frames,
            size,
            base_seed,
            sample_solver,
            face_restore,
            background_restore,
            job_id=job_id,
            enable_live_hls=enable_live_hls,
            live_raw_dir=live_raw_dir,
            save_live_raw_mp4=bool(save_live_raw_mp4),
            video_prompt=str(video_prompt or ""),
            negative_prompt=str(negative_prompt or ""),
            idle_prompt=str(idle_prompt or ""),
            stream_file_output_path=str(stream_file_output_path or ""),
            stream_file_output_width=int(stream_file_output_width or 0),
            stream_file_output_height=int(stream_file_output_height or 0),
            stream_file_output_fps=float(stream_file_output_fps or 0.0),
            stream_file_trim_duration_sec=float(stream_file_trim_duration_sec or 0.0),
            stream_file_interpolation=str(stream_file_interpolation or ""),
            tpp_cfg_mode=str(tpp_cfg_mode or ""),
            lipsync_audio_path=str(lipsync_audio_path or ""),
        )
    finally:
        if dist.is_initialized() and int(state.rank) == 0:
            broadcast_command_fn(
                cmd=build_idle_command_fn(),
                src_rank=0,
                control_group=state.control_group,
            )
            logging.info("[Rank 0] Sent idle signal, workers can continue waiting for next request")


def reload_pipeline_command(
    use_lora: bool,
    lora_path: str | None,
    *,
    state_getter: Callable[[], DistributedRuntimeState],
    initialize_pipeline_fn: Callable[[Any, Any], None],
    broadcast_command_fn: Callable[..., None],
    build_reload_command_fn: Callable[..., Any],
    build_idle_command_fn: Callable[[], Any],
) -> str:
    state = state_getter()
    logging.info("Reloading pipeline with use_lora=%s, lora_path=%s", str(use_lora), str(lora_path))

    state.args.load_lora = bool(use_lora)
    state.args.lora_path_dmd = lora_path if lora_path else None

    if dist.is_initialized() and int(state.rank) == 0:
        broadcast_command_fn(
            cmd=build_reload_command_fn(
                load_lora=state.args.load_lora,
                lora_path=state.args.lora_path_dmd,
            ),
            src_rank=0,
            control_group=state.control_group,
        )
        logging.info("[Rank 0] Broadcast reload signal to other ranks")

    initialize_pipeline_fn(state.args, state.training_settings)

    state = state_getter()
    if dist.is_initialized() and int(state.rank) == 0:
        broadcast_command_fn(
            cmd=build_idle_command_fn(),
            src_rank=0,
            control_group=state.control_group,
        )
        logging.info("[Rank 0] Sent idle signal after reload")

    return "模型重新加载成功 / Pipeline reloaded successfully!"


def worker_loop(
    *,
    state_getter: Callable[[], DistributedRuntimeState],
    receive_command_fn: Callable[..., Any],
    initialize_pipeline_fn: Callable[[Any, Any], None],
    run_inference_computation_fn: Callable[..., Any],
) -> None:
    state = state_getter()
    logging.info("Rank %s entering worker loop, waiting for inference requests...", int(state.rank))

    while True:
        try:
            state = state_getter()
            cmd = receive_command_fn(src_rank=0, control_group=state.control_group)
            command = cmd.command

            if command == "infer":
                if cmd.prompt is not None and cmd.prompt != "":
                    logging.info("[Rank %s] Received valid inference request", int(state.rank))
                    run_inference_computation_fn(
                        cmd.prompt,
                        cmd.image_path,
                        cmd.audio_path,
                        cmd.num_clip,
                        cmd.sample_steps,
                        cmd.sample_guide_scale,
                        cmd.infer_frames,
                        cmd.size,
                        cmd.base_seed,
                        cmd.sample_solver,
                        float(cmd.face_restore or 0.0),
                        float(cmd.background_restore or 0.0),
                        job_id=cmd.job_id,
                        enable_live_hls=bool(cmd.enable_live_hls),
                        live_raw_dir=cmd.live_raw_dir,
                        save_live_raw_mp4=bool(cmd.save_live_raw_mp4),
                        video_prompt=str(cmd.video_prompt or ""),
                        negative_prompt=str(cmd.negative_prompt or ""),
                        idle_prompt=str(cmd.idle_prompt or ""),
                        stream_file_output_path=str(cmd.stream_file_output_path or ""),
                        stream_file_output_width=int(cmd.stream_file_output_width or 0),
                        stream_file_output_height=int(cmd.stream_file_output_height or 0),
                        stream_file_output_fps=float(cmd.stream_file_output_fps or 0.0),
                        stream_file_trim_duration_sec=float(cmd.stream_file_trim_duration_sec or 0.0),
                        stream_file_interpolation=str(cmd.stream_file_interpolation or ""),
                        tpp_cfg_mode=str(cmd.tpp_cfg_mode or ""),
                        lipsync_audio_path=str(cmd.lipsync_audio_path or ""),
                    )
                    logging.info("[Rank %s] Generation completed", int(state.rank))
            elif command == "reload":
                logging.info(
                    "[Rank %s] Received reload request: load_lora=%s, lora_path=%s",
                    int(state.rank),
                    str(cmd.load_lora),
                    str(cmd.lora_path),
                )
                state.args.load_lora = cmd.load_lora
                state.args.lora_path_dmd = cmd.lora_path
                initialize_pipeline_fn(state.args, state.training_settings)
            elif command == "idle":
                logging.debug("[Rank %s] Received idle signal, continuing to wait...", int(state.rank))
            else:
                if command is not None:
                    logging.warning("[Rank %s] Received unknown command: %s", int(state.rank), str(command))
        except Exception as e:
            state = state_getter()
            logging.error("[Rank %s] Error in worker loop: %s", int(state.rank), e)
            import traceback

            traceback.print_exc()
            time.sleep(1)
