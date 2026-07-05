from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS
from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video

from .infer import (
    InferenceRequest,
    log_inference_request,
    prepare_inference_targets,
    save_rank_video_output,
    sync_after_inference,
)
from .observability import deep_timing_enabled, log_phase_timing, model_timing_enabled


@dataclass
class ModelInferenceRuntimeState:
    pipeline: Any
    args: Any
    cfg: Any
    rank: int
    world_size: int
    save_rank: int
    control_group: Any


def run_inference_computation(
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
    face_restore: float,
    background_restore: float,
    negative_prompt: str = "",
    video_prompt: str = "",
    idle_prompt: str = "",
    *,
    state: ModelInferenceRuntimeState,
    job_id: str | None,
    enable_live_hls: bool,
    live_raw_dir: str | None,
    sanitize_job_id,
    make_job_id,
    prepare_live_hls_dir,
    prepare_live_raw_dir,
    save_live_raw_mp4: bool = False,
    stream_file_output_path: str = "",
    stream_file_output_width: int = 0,
    stream_file_output_height: int = 0,
    stream_file_output_fps: float = 0.0,
    stream_file_trim_duration_sec: float = 0.0,
    stream_file_interpolation: str = "",
    tpp_cfg_mode: str = "",
    lipsync_audio_path: str = "",
) -> str | None:
    total_t0 = time.perf_counter()
    deep_timing = bool(deep_timing_enabled()) and int(state.rank) == 0
    generate_s = 0.0
    save_s = 0.0
    try:
        request_t0 = time.perf_counter()
        model_prompt = str(prompt or "")
        req = InferenceRequest(
            prompt=str(prompt),
            video_prompt=str(video_prompt or ""),
            image_path=str(image_path),
            audio_path=str(audio_path),
            num_clip=int(num_clip),
            sample_steps=int(sample_steps),
            sample_guide_scale=float(sample_guide_scale),
            infer_frames=int(infer_frames),
            size=str(size),
            base_seed=int(base_seed),
            sample_solver=str(sample_solver),
            face_restore=float(face_restore),
            background_restore=float(background_restore),
            negative_prompt=str(negative_prompt or ""),
            idle_prompt=str(idle_prompt or ""),
            lipsync_audio_path=str(lipsync_audio_path or ""),
            job_id=(str(job_id) if job_id is not None else None),
            enable_live_hls=bool(enable_live_hls),
            live_raw_dir=(str(live_raw_dir) if live_raw_dir else None),
            tpp_cfg_mode=str(tpp_cfg_mode or ""),
        )
        log_inference_request(rank=int(state.rank), req=req)
        log_phase_timing(
            "generation",
            "build_request",
            request_t0,
            enabled=bool(deep_timing),
            job=str(job_id or "-"),
            live_raw=1 if str(live_raw_dir or "").strip() else 0,
            live_hls=1 if bool(enable_live_hls) else 0,
            face=f"{float(face_restore):.2f}",
            bg=f"{float(background_restore):.2f}",
        )

        prepare_targets_t0 = time.perf_counter()
        targets = prepare_inference_targets(
            rank=int(state.rank),
            world_size=int(state.world_size),
            save_dir=str(state.args.save_dir),
            prompt=str(model_prompt),
            sample_steps=int(sample_steps),
            lora_path=getattr(state.args, "lora_path_dmd", None),
            enable_live_hls=bool(enable_live_hls),
            live_raw_dir=(str(live_raw_dir) if live_raw_dir else None),
            job_id=(str(job_id) if job_id is not None else None),
            control_group=state.control_group,
            sanitize_job_id=sanitize_job_id,
            make_job_id=make_job_id,
            prepare_live_hls_dir=prepare_live_hls_dir,
            prepare_live_raw_dir=prepare_live_raw_dir,
        )
        log_phase_timing(
            "generation",
            "prepare_targets",
            prepare_targets_t0,
            enabled=bool(deep_timing),
            job=str(job_id or "-"),
            live_raw=1 if str(targets.live_raw_dir or "").strip() else 0,
            live_hls=1 if bool(targets.live_hls_dir) else 0,
        )

        generate_t0 = time.perf_counter()
        tpp_cfg_mode_s = str(tpp_cfg_mode or "").strip()
        model_audio_path = str(lipsync_audio_path or "").strip() or str(audio_path or "")
        job_id_s = str(job_id or "")
        audio_path_s = str(audio_path or "").strip()
        lipsync_audio_path_s = str(lipsync_audio_path or "").strip()
        render_onepass_liveaudio = bool(
            (
                audio_path_s.startswith("liveaudio://")
                or lipsync_audio_path_s.startswith("liveaudio://")
                or model_audio_path.startswith("liveaudio://")
            )
            and (
                "_avatar_onepass" in job_id_s
                or "_render_avatar_onepass" in audio_path_s
                or "_render_avatar_onepass" in lipsync_audio_path_s
                or "_render_avatar_onepass" in model_audio_path
            )
        )
        old_tpp_cfg_mode = os.environ.get("LIVE_STREAM_TPP_CFG_MODE")
        old_allow_long_clips = os.environ.get("LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS")
        old_max_clip_frames = os.environ.get("LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES")
        if tpp_cfg_mode_s:
            os.environ["LIVE_STREAM_TPP_CFG_MODE"] = tpp_cfg_mode_s
        if render_onepass_liveaudio:
            try:
                configured_max_clip_frames = int(
                    str(
                        os.getenv(
                            "SMARTBLOG_RENDER_ONEPASS_MAX_AUDIO_CLIP_FRAMES",
                            os.getenv(
                                "SMARTBLOG_RENDER_ONEPASS_MAX_CONDITIONING_FRAMES",
                                str(int(infer_frames)),
                            ),
                        )
                    ).strip()
                )
            except Exception:
                configured_max_clip_frames = int(infer_frames)
            max_clip_frames = int(max(int(infer_frames), min(512, int(configured_max_clip_frames))))
            os.environ["LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS"] = "1"
            os.environ["LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES"] = str(max_clip_frames)
            logging.warning(
                "SmartBlog render one-pass liveaudio clip cap applied: job=%s infer_frames=%d max_clip_frames=%d",
                job_id_s or "-",
                int(infer_frames),
                int(max_clip_frames),
            )
        try:
            video, _dataset_info = state.pipeline.generate(
                input_prompt=model_prompt,
                n_prompt=str(negative_prompt or ""),
                video_prompt=str(video_prompt or ""),
                idle_prompt=str(idle_prompt or ""),
                ref_image_path=image_path,
                audio_path=audio_path,
                lipsync_audio_path=str(model_audio_path),
                enable_tts=state.args.enable_tts,
                tts_prompt_audio=None,
                tts_prompt_text=None,
                tts_text=None,
                num_repeat=num_clip,
                pose_video=state.args.pose_video,
                generate_size=size,
                max_area=MAX_AREA_CONFIGS[size],
                infer_frames=infer_frames,
                shift=state.args.sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=base_seed,
                offload_model=state.args.offload_model,
                init_first_frame=state.args.start_from_ref,
                use_dataset=False,
                dataset_sample_idx=0,
                drop_motion_noisy=state.args.drop_motion_noisy,
                num_gpus_dit=state.args.num_gpus_dit,
                enable_vae_parallel=state.args.enable_vae_parallel,
                input_video_for_sam2=None,
                live_hls_dir=targets.live_hls_dir,
                live_raw_dir=targets.live_raw_dir,
                post_vae_face_restore=float(face_restore),
                post_vae_background_restore=float(background_restore),
                job_id=(str(job_id) if job_id is not None else None),
                stream_file_output_path=str(stream_file_output_path or ""),
                stream_file_output_width=int(stream_file_output_width or 0),
                stream_file_output_height=int(stream_file_output_height or 0),
                stream_file_output_fps=float(stream_file_output_fps or 0.0),
                stream_file_trim_duration_sec=float(stream_file_trim_duration_sec or 0.0),
                stream_file_interpolation=str(stream_file_interpolation or ""),
            )
        finally:
            if tpp_cfg_mode_s:
                if old_tpp_cfg_mode is None:
                    os.environ.pop("LIVE_STREAM_TPP_CFG_MODE", None)
                else:
                    os.environ["LIVE_STREAM_TPP_CFG_MODE"] = str(old_tpp_cfg_mode)
            if render_onepass_liveaudio:
                if old_allow_long_clips is None:
                    os.environ.pop("LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS", None)
                else:
                    os.environ["LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS"] = str(old_allow_long_clips)
                if old_max_clip_frames is None:
                    os.environ.pop("LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES", None)
                else:
                    os.environ["LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES"] = str(old_max_clip_frames)
        generate_s = float(time.perf_counter() - generate_t0)
        log_phase_timing(
            "generation",
            "pipeline_generate",
            generate_t0,
            enabled=bool(deep_timing),
            sync_device=getattr(state.pipeline, "device", None),
            job=str(job_id or "-"),
            rank=int(state.rank),
            clips=int(num_clip),
            steps=int(sample_steps),
            infer_frames=int(infer_frames),
            live_raw=1 if str(live_raw_dir or "").strip() else 0,
        )

        logging.info("[Rank %s] Denoising video done", state.rank)
        save_t0 = time.perf_counter()
        if isinstance(video, str) and str(video).startswith("edge-uploaded://"):
            video_path = str(video)
            if int(state.rank) == int(state.save_rank):
                logging.info("Skip saved mp4 for remote edge uploaded output: %s", str(video_path))
        elif str(stream_file_output_path or "").strip():
            video_path = str(stream_file_output_path or "").strip()
            if int(state.rank) == int(state.save_rank):
                logging.info("Skip saved mp4; direct stream-file output was encoded in pipeline: %s", str(video_path))
        elif str(targets.live_raw_dir or "").strip() and not bool(save_live_raw_mp4):
            video_path = targets.save_file
            if int(state.rank) == int(state.save_rank):
                logging.info("Skip saved mp4 for live_raw inference: %s", str(targets.live_raw_dir))
        else:
            video_path = save_rank_video_output(
                rank=int(state.rank),
                save_rank=int(state.save_rank),
                video=video,
                save_file=targets.save_file,
                fps=int(state.cfg.sample_fps),
                audio_path=str(audio_path or ""),
                enable_tts=bool(state.args.enable_tts),
                task=str(state.args.task),
                save_video_fn=save_video,
                merge_video_audio_fn=merge_video_audio,
            )
        save_s = float(time.perf_counter() - save_t0)
        log_phase_timing(
            "generation",
            "save_output",
            save_t0,
            enabled=bool(deep_timing),
            job=str(job_id or "-"),
                output=os.path.basename(str(video_path or "")) or "-",
        )

        del video
        if bool(getattr(state.args, "offload_model", False)):
            torch.cuda.empty_cache()

        sync_s = 0.0
        if dist.is_initialized() and dist.get_world_size() > 1:
            sync_t0 = time.perf_counter()
            sync_after_inference(
                rank=int(state.rank),
                world_size=int(dist.get_world_size()),
                control_group=state.control_group,
            )
            sync_s = float(time.perf_counter() - sync_t0)
            log_phase_timing(
                "generation",
                "sync_after_inference",
                sync_t0,
                enabled=bool(deep_timing),
                job=str(job_id or "-"),
                world=int(dist.get_world_size()),
            )
        total_s = float(time.perf_counter() - total_t0)
        if model_timing_enabled():
            logging.info(
                "Generation rank timing: job=%s rank=%d save_rank=%d generate=%.3fs save=%.3fs sync=%.3fs total=%.3fs output=%s",
                str(job_id or "-"),
                int(state.rank),
                int(state.save_rank),
                float(generate_s),
                float(save_s),
                float(sync_s),
                float(total_s),
                os.path.basename(str(video_path or "")) or "-",
            )
        if model_timing_enabled() and int(state.rank) == 0:
            try:
                device_name = f"cuda:{int(torch.cuda.current_device())}" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_name = "unknown"
            logging.info(
                "Generation timing: job=%s device=%s rank=%d generate=%.3fs save=%.3fs total=%.3fs live_raw=%d live_hls=%d output=%s face=%.2f bg=%.2f",
                str(job_id or "-"),
                device_name,
                int(state.rank),
                float(generate_s),
                float(save_s),
                float(total_s),
                1 if str(live_raw_dir or "").strip() else 0,
                1 if bool(enable_live_hls) else 0,
                os.path.basename(str(video_path or "")) or "-",
                float(face_restore),
                float(background_restore),
            )
        return video_path
    except Exception as exc:
        logging.error("Error during generation: %s", exc)
        import traceback

        traceback.print_exc()
        raise
