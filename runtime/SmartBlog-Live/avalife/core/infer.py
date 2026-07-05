from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    video_prompt: str
    image_path: str
    audio_path: str
    num_clip: int
    sample_steps: int
    sample_guide_scale: float
    infer_frames: int
    size: str
    base_seed: int
    sample_solver: str
    face_restore: float
    background_restore: float
    negative_prompt: str
    job_id: str | None
    enable_live_hls: bool
    live_raw_dir: str | None
    idle_prompt: str = ""
    tpp_cfg_mode: str = ""
    lipsync_audio_path: str = ""


@dataclass(frozen=True)
class InferenceOutputTargets:
    save_file: str | None
    live_hls_dir: str | None
    live_raw_dir: str | None


def log_inference_request(*, rank: int, req: InferenceRequest) -> None:
    logging.info("[Rank %d] Generating video...", int(rank))
    logging.info("  Prompt: %s", str(req.prompt))
    logging.info("  Video_prompt_chars: %d", int(len(str(req.video_prompt or ""))))
    logging.info("  Image: %s", str(req.image_path))
    logging.info("  Audio: %s", str(req.audio_path))
    if str(req.lipsync_audio_path or "").strip():
        logging.info("  Lipsync_audio: %s", str(req.lipsync_audio_path))
    logging.info("  Num_clip: %d", int(req.num_clip))
    logging.info("  Sample_steps: %d", int(req.sample_steps))
    logging.info("  Guide_scale: %s", str(req.sample_guide_scale))
    logging.info("  Infer_frames: %d", int(req.infer_frames))
    logging.info("  Size: %s", str(req.size))
    logging.info("  Seed: %d", int(req.base_seed))
    logging.info("  Solver: %s", str(req.sample_solver))
    logging.info("  Face_restore: %.3f", float(req.face_restore))
    logging.info("  Background_restore: %.3f", float(req.background_restore))
    logging.info("  Negative_prompt_chars: %d", int(len(str(req.negative_prompt or ""))))
    logging.info("  Idle_prompt_chars: %d", int(len(str(req.idle_prompt or ""))))
    logging.info("  TPP_CFG_mode: %s", str(req.tpp_cfg_mode or "-"))
    logging.info("  Live_hls: %s", bool(req.enable_live_hls))
    logging.info("  Live_raw: %s", str(req.live_raw_dir) if req.live_raw_dir else "-")


def _broadcast_object(value: Any, *, src_rank: int, world_size: int, control_group: Any) -> Any:
    if int(world_size) <= 1:
        return value
    payload = [value] if dist.get_rank() == int(src_rank) else [None]
    dist.broadcast_object_list(payload, src=int(src_rank), group=control_group)
    return payload[0]


def _build_save_file(*, save_dir: str, prompt: str, sample_steps: int, lora_path: str | None) -> str:
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = str(prompt).replace(" ", "_").replace("/", "_")[:50]
    save_name = f"{formatted_time}_{int(sample_steps)}step_{formatted_prompt}"

    lora_path_s = str(lora_path or "").strip()
    if lora_path_s:
        path_parts = lora_path_s.split("/")
        if lora_path_s.endswith(".pt") and len(path_parts) >= 3:
            save_name = save_name + "_" + path_parts[-3] + "_" + path_parts[-1].split(".")[0]

    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, save_name + ".mp4")


def prepare_inference_targets(
    *,
    rank: int,
    world_size: int,
    save_dir: str,
    prompt: str,
    sample_steps: int,
    lora_path: str | None,
    enable_live_hls: bool,
    live_raw_dir: str | None,
    job_id: str | None,
    control_group: Any,
    sanitize_job_id: Callable[[str], str],
    make_job_id: Callable[[], str],
    prepare_live_hls_dir: Callable[[str, str], str],
    prepare_live_raw_dir: Callable[[str, str], str],
) -> InferenceOutputTargets:
    save_file = None
    if int(rank) == 0:
        save_file = _build_save_file(
            save_dir=str(save_dir),
            prompt=str(prompt),
            sample_steps=int(sample_steps),
            lora_path=lora_path,
        )
    save_file = _broadcast_object(
        save_file,
        src_rank=0,
        world_size=int(world_size),
        control_group=control_group,
    )

    safe_job_id = sanitize_job_id(str(job_id or "")) if job_id else make_job_id()

    live_hls_dir_eff = None
    if bool(enable_live_hls) and int(rank) == 0:
        live_hls_dir_eff = prepare_live_hls_dir(str(save_dir), str(safe_job_id))
    live_hls_dir_eff = _broadcast_object(
        live_hls_dir_eff,
        src_rank=0,
        world_size=int(world_size),
        control_group=control_group,
    )

    live_raw_dir_eff = None
    raw_dir_in = str(live_raw_dir or "").strip()
    if raw_dir_in and int(rank) == 0:
        if os.path.isabs(raw_dir_in):
            live_raw_dir_eff = raw_dir_in
        else:
            live_raw_dir_eff = os.path.join(str(save_dir), raw_dir_in)
    elif (not raw_dir_in) and int(rank) == 0 and live_raw_dir is not None:
        live_raw_dir_eff = prepare_live_raw_dir(str(save_dir), str(safe_job_id))
    live_raw_dir_eff = _broadcast_object(
        live_raw_dir_eff,
        src_rank=0,
        world_size=int(world_size),
        control_group=control_group,
    )

    return InferenceOutputTargets(
        save_file=save_file,
        live_hls_dir=live_hls_dir_eff,
        live_raw_dir=live_raw_dir_eff,
    )


def save_rank_video_output(
    *,
    rank: int,
    save_rank: int,
    video: Any,
    save_file: str | None,
    fps: int,
    audio_path: str,
    enable_tts: bool,
    task: str,
    save_video_fn: Callable[..., None],
    merge_video_audio_fn: Callable[..., None],
) -> str | None:
    if int(rank) != int(save_rank):
        return save_file
    if not save_file:
        return None

    logging.info("Saving generated video to %s", str(save_file))
    if video is None:
        logging.info("No non-silent frames generated on save rank; skip save for this segment.")
        return None

    save_video_fn(
        tensor=video[None],
        save_file=save_file,
        fps=int(fps),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    if "s2v" in str(task):
        audio_path_s = str(audio_path or "").strip()
        if not bool(enable_tts):
            if audio_path_s.startswith("liveaudio://"):
                logging.info("Skip merge_video_audio for liveaudio stream source: %s", audio_path_s)
            else:
                merge_video_audio_fn(video_path=save_file, audio_path=audio_path_s)
        else:
            merge_video_audio_fn(video_path=save_file, audio_path="tts.wav")

    logging.info("Video saved successfully: %s", str(save_file))
    return str(save_file)


def sync_after_inference(*, rank: int, world_size: int, control_group: Any) -> None:
    if int(world_size) <= 1:
        return
    torch.cuda.synchronize()
    if control_group is not None:
        dist.barrier(group=control_group)
    else:
        dist.barrier()
    logging.info("[Rank %d] Inference completed, synchronized with all ranks", int(rank))
