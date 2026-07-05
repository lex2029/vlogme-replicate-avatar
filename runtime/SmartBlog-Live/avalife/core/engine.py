# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import sys
import warnings
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from .audio import (
    auto_num_clip_for_audio as _auto_num_clip_for_audio_runtime,
    ensure_silence_wav as _ensure_silence_wav_runtime,
    normalize_infer_frames as _normalize_infer_frames_runtime,
    to_wav_16k_mono as _to_wav_16k_mono_runtime,
    wav_duration_seconds as _wav_duration_seconds_runtime,
)
from .channel_runtime import (
    _ensure_channel_started,
    _live_channel_enqueue_enabled,
    _live_master_enabled,
    _update_channel_state,
    CHANNEL_NAME,
    PUBLIC_OUTPUT_BASE,
    configure_channel_runtime,
)
from .control import (
    broadcast_command,
    build_idle_command,
    build_infer_command,
    build_reload_command,
    receive_command,
)
from .distributed import (
    DistributedRuntimeState,
    reload_pipeline_command as distributed_reload_pipeline_command,
    run_single_sample as distributed_run_single_sample,
    worker_loop as distributed_worker_loop,
)
from .generation import (
    ModelInferenceRuntimeState,
    run_inference_computation as generation_run_inference_computation,
)
from .path import (
    current_sample_fps as _current_sample_fps_runtime,
    is_square_size as _is_square_size,
    make_job_id as _make_job_id,
    maybe_square_crop_ref_image as _maybe_square_crop_ref_image_runtime,
    persist_idle_image as _persist_idle_image_runtime,
    prepare_live_hls_dir as _prepare_live_hls_dir_runtime,
    prepare_live_raw_dir as _prepare_live_raw_dir_runtime,
    required_worker_fps as _required_worker_fps_runtime,
    sanitize_job_id as _sanitize_job_id,
    sha256_file as _sha256_file,
    sha256_text as _sha256_text,
)
from .pipeline import (
    initialize_pipeline_runtime,
    ModelPipelineRuntimeState,
)
from liveavatar.models.wan.wan_2_2.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from liveavatar.utils.args_config import parse_args_for_training_config as training_config_parser

# Global variables for pipeline and config
wan_s2v_pipeline = None
global_args = None
global_cfg = None
global_training_settings = None
global_rank = 0
global_world_size = 1
global_save_rank = 0
global_control_group = None  # CPU (gloo) group for broadcast_object_list() control messages.
GENERATION_LOCK = threading.Lock()


def _write_live_player_html(out_dir: str, *, start_from_beginning: bool = False) -> None:
    start_q = "?start=0" if bool(start_from_beginning) else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartBlog Live Preview</title>
  <style>
    html,body{{margin:0;background:#000;color:#fff;font-family:sans-serif}}
    .wrap{{display:flex;align-items:center;justify-content:center;min-height:100vh}}
    video{{width:min(100vw,1280px);height:auto;background:#000}}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
</head>
<body>
  <div class="wrap"><video id="v" controls autoplay playsinline muted></video></div>
  <script>
    const video = document.getElementById("v");
    const src = "./index.m3u8{start_q}";
    if (video.canPlayType("application/vnd.apple.mpegurl")) {{
      video.src = src;
    }} else if (window.Hls && Hls.isSupported()) {{
      const hls = new Hls({{lowLatencyMode:true}});
      hls.loadSource(src);
      hls.attachMedia(video);
    }}
  </script>
</body>
</html>
"""
    with open(os.path.join(out_dir, "player.html"), "w", encoding="utf-8") as handle:
        handle.write(html)

def _required_worker_fps() -> int:
    return _required_worker_fps_runtime()


def _current_sample_fps() -> int:
    return _current_sample_fps_runtime(global_cfg)


def _prepare_live_hls_dir(save_dir: str, job_id: str) -> str:
    return _prepare_live_hls_dir_runtime(
        save_dir,
        job_id,
        write_live_player_html=_write_live_player_html,
    )


def _prepare_live_raw_dir(save_dir: str, job_id: str) -> str:
    return _prepare_live_raw_dir_runtime(save_dir, job_id)

IDLE_TMP_DIR = os.getenv("LIVE_IDLE_TMP_DIR", "./tmp/channel_idle")


def _to_wav_16k_mono(in_audio_path: str, out_wav_path: str) -> str:
    return _to_wav_16k_mono_runtime(in_audio_path, out_wav_path)


def _wav_duration_seconds(wav_path: str) -> float:
    return _wav_duration_seconds_runtime(wav_path)


def _auto_num_clip_for_audio(audio_wav_path: str, fps: int, infer_frames: int) -> int:
    return _auto_num_clip_for_audio_runtime(audio_wav_path, fps, infer_frames)


def _ensure_silence_wav(out_wav_path: str, samples: int, sample_rate: int = 16000) -> str:
    return _ensure_silence_wav_runtime(out_wav_path, samples, sample_rate=sample_rate)


def _persist_idle_image(src_path: str) -> str:
    return _persist_idle_image_runtime(
        src_path,
        idle_tmp_dir=IDLE_TMP_DIR,
        channel_name=CHANNEL_NAME,
    )


def _maybe_square_crop_ref_image(image_path: str, size: str, job_id: str | None = None) -> str:
    return _maybe_square_crop_ref_image_runtime(
        image_path,
        size,
        idle_tmp_dir=IDLE_TMP_DIR,
        channel_name=CHANNEL_NAME,
        job_id=job_id,
    )

def _normalize_infer_frames(infer_frames: int) -> tuple[int, str | None]:
    return _normalize_infer_frames_runtime(
        infer_frames,
        world_size=int(os.getenv("WORLD_SIZE", "1") or 1),
        cfg=global_cfg,
    )


def initialize_pipeline(args, training_settings, progress_cb=None):
    global wan_s2v_pipeline, global_args, global_cfg, global_training_settings
    global global_rank, global_world_size, global_save_rank
    global global_control_group

    state: ModelPipelineRuntimeState = initialize_pipeline_runtime(
        args,
        training_settings,
        current_pipeline=wan_s2v_pipeline,
        current_control_group=global_control_group,
        required_worker_fps=int(_required_worker_fps()),
        progress_cb=progress_cb,
    )
    wan_s2v_pipeline = state.pipeline
    global_args = state.args
    global_cfg = state.cfg
    global_training_settings = state.training_settings
    global_rank = state.rank
    global_world_size = state.world_size
    global_save_rank = state.save_rank
    global_control_group = state.control_group
    configure_channel_runtime(
        rank=state.rank,
        args=state.args,
        cfg=state.cfg,
        current_sample_fps_getter=_current_sample_fps,
    )


def run_single_sample(prompt, image_path, audio_path, num_clip, 
                     sample_steps, sample_guide_scale, infer_frames,
                     size, base_seed, sample_solver, face_restore=0.0, background_restore=0.0,
                     job_id=None, enable_live_hls=False, live_raw_dir=None, save_live_raw_mp4=False,
                     video_prompt="",
                     negative_prompt="",
                     idle_prompt="",
                     stream_file_output_path="",
                     stream_file_output_width=0,
                     stream_file_output_height=0,
                     stream_file_output_fps=0.0,
                     stream_file_trim_duration_sec=0.0,
                     stream_file_interpolation="",
                     tpp_cfg_mode="",
                     lipsync_audio_path=""):
    return distributed_run_single_sample(
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
        state=_distributed_runtime_state(),
        maybe_square_crop_ref_image=_maybe_square_crop_ref_image,
        run_inference_computation=_run_inference_computation,
        broadcast_command_fn=broadcast_command,
        build_infer_command_fn=build_infer_command,
        build_idle_command_fn=build_idle_command,
    )


def reload_pipeline_command(use_lora, lora_path):
    return distributed_reload_pipeline_command(
        use_lora,
        lora_path,
        state_getter=_distributed_runtime_state,
        initialize_pipeline_fn=initialize_pipeline,
        broadcast_command_fn=broadcast_command,
        build_reload_command_fn=build_reload_command,
        build_idle_command_fn=build_idle_command,
    )


def _run_inference_computation(prompt, image_path, audio_path, num_clip,
                               sample_steps, sample_guide_scale, infer_frames,
                               size, base_seed, sample_solver, face_restore=0.0, background_restore=0.0,
                               job_id=None, enable_live_hls=False, live_raw_dir=None, save_live_raw_mp4=False,
                               video_prompt="",
                               negative_prompt="",
                               idle_prompt="",
                               stream_file_output_path="",
                               stream_file_output_width=0,
                               stream_file_output_height=0,
                               stream_file_output_fps=0.0,
                               stream_file_trim_duration_sec=0.0,
                               stream_file_interpolation="",
                               tpp_cfg_mode="",
                               lipsync_audio_path=""):
    return generation_run_inference_computation(
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
        str(negative_prompt or ""),
        video_prompt=str(video_prompt or ""),
        idle_prompt=str(idle_prompt or ""),
        state=ModelInferenceRuntimeState(
            pipeline=wan_s2v_pipeline,
            args=global_args,
            cfg=global_cfg,
            rank=int(global_rank),
            world_size=int(global_world_size),
            save_rank=int(global_save_rank),
            control_group=global_control_group,
        ),
        job_id=job_id,
        enable_live_hls=enable_live_hls,
        live_raw_dir=live_raw_dir,
        save_live_raw_mp4=bool(save_live_raw_mp4),
        stream_file_output_path=str(stream_file_output_path or ""),
        stream_file_output_width=int(stream_file_output_width or 0),
        stream_file_output_height=int(stream_file_output_height or 0),
        stream_file_output_fps=float(stream_file_output_fps or 0.0),
        stream_file_trim_duration_sec=float(stream_file_trim_duration_sec or 0.0),
        stream_file_interpolation=str(stream_file_interpolation or ""),
        tpp_cfg_mode=str(tpp_cfg_mode or ""),
        lipsync_audio_path=str(lipsync_audio_path or ""),
        sanitize_job_id=_sanitize_job_id,
        make_job_id=_make_job_id,
        prepare_live_hls_dir=_prepare_live_hls_dir,
        prepare_live_raw_dir=_prepare_live_raw_dir,
    )


def _distributed_runtime_state() -> DistributedRuntimeState:
    return DistributedRuntimeState(
        args=global_args,
        training_settings=global_training_settings,
        rank=int(global_rank),
        world_size=int(global_world_size),
        save_rank=int(global_save_rank),
        control_group=global_control_group,
    )


def worker_loop():
    distributed_worker_loop(
        state_getter=_distributed_runtime_state,
        receive_command_fn=receive_command,
        initialize_pipeline_fn=initialize_pipeline,
        run_inference_computation_fn=_run_inference_computation,
    )
