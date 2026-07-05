from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any

from avalife.core.smartblog_profiles import SMARTBLOG_RUNTIME_SIZE_CONFIGS
from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.utils.utils import str2bool


def _required_worker_fps() -> int:
    raw = str(os.getenv("WORKER_FPS", "") or "").strip()
    if not raw:
        raise RuntimeError("Missing required env: WORKER_FPS")
    try:
        fps = int(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid WORKER_FPS={raw!r}") from e
    if fps <= 0:
        raise RuntimeError(f"WORKER_FPS must be > 0, got {fps}")
    return int(fps)


def validate_runtime_args(args: Any) -> Any:
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    if bool(getattr(args, "using_merged_ckpt", False)):
        if not str(getattr(args, "merged_noise_model_dir", "") or "").strip():
            raise RuntimeError("using_merged_ckpt requires merged_noise_model_dir")
        args.load_lora = False
        args.lora_path_dmd = None

    if args.size is None:
        args.size = "384*256"
    if str(args.size) not in SMARTBLOG_RUNTIME_SIZE_CONFIGS:
        raise RuntimeError(
            "Unsupported SmartBlog runtime size "
            f"{str(args.size)!r}; use one of {', '.join(SMARTBLOG_RUNTIME_SIZE_CONFIGS)}"
        )

    cfg = WAN_CONFIGS[args.task]
    forced_sample_fps = int(_required_worker_fps())
    try:
        old_fps = int(getattr(cfg, "sample_fps", 0) or 0)
    except Exception:
        old_fps = 0
    if int(old_fps) != int(forced_sample_fps):
        cfg.sample_fps = int(forced_sample_fps)
        logging.info("Overriding model sample_fps: %d -> %d", int(old_fps), int(forced_sample_fps))

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    return args


def build_runtime_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AvaLife runtime entrypoint")
    parser.add_argument("--task", type=str, default="s2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--size", type=str, default=None, choices=list(SMARTBLOG_RUNTIME_SIZE_CONFIGS), help="The area (width*height) of the generated video.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/Wan2.2-S2V-14B/", help="The path to the checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--save_dir", type=str, default="./output/", help="The directory to save the generated video to.")
    parser.add_argument("--sample_solver", type=str, default="euler", choices=["euler", "unipc", "dpm++"], help="The solver used to sample.")
    parser.add_argument("--sample_steps", type=int, default=4, help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=0.0, help="Classifier free guidance scale.")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")
    parser.add_argument("--base_seed", type=int, default=420, help="The seed to use for generating the video.")
    parser.add_argument("--infer_frames", type=int, default=48, help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v")
    parser.add_argument("--load_lora", action="store_true", default=False, help="Whether to load the LoRA weights.")
    parser.add_argument("--lora_path", type=str, default=None, help="The path to the LoRA weights.")
    parser.add_argument("--lora_path_dmd", type=str, default=None, help="The path to the LoRA weights for DMD.")
    parser.add_argument("--training_config", type=str, default="liveavatar/configs/s2v_causal_sft.yaml", help="The path to the training config file.")
    parser.add_argument("--num_clip", type=int, default=0, help="Optional clip-count hint. 0 lets worker code derive it from audio/runtime context.")
    parser.add_argument("--single_gpu", action="store_true", default=False, help="Whether to use a single GPU.")
    parser.add_argument("--using_merged_ckpt", action="store_true", default=False, help="Whether to use the merged ckpt.")
    parser.add_argument("--merged_noise_model_dir", type=str, default=None, help="Optional path to a pre-merged noise model directory.")
    parser.add_argument("--num_gpus_dit", type=int, default=4, help="The number of GPUs to use for DiT.")
    parser.add_argument("--enable_vae_parallel", action="store_true", default=False, help="Whether to enable VAE parallel decoding on a separate GPU.")
    parser.add_argument("--offload_kv_cache", action="store_true", default=False, help="Whether to offload the KV cache to CPU.")
    parser.add_argument("--enable_tts", action="store_true", default=False, help="Use CosyVoice to synthesis audio")
    parser.add_argument("--pose_video", type=str, default=None, help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument("--start_from_ref", action="store_true", default=False, help="whether set the reference image as the starting point for generation")
    parser.add_argument("--drop_motion_noisy", action="store_true", default=False, help="Whether to drop the motion noisy.")
    parser.add_argument("--server_port", type=int, default=7860, help="Port to run the Gradio server on.")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name for Gradio (0.0.0.0 for public access).")
    parser.add_argument("--fp8", action="store_true", default=False, help="Whether to enable fp8 quantization.")
    return parser


def parse_runtime_args(argv: list[str] | None = None) -> Any:
    parser = build_runtime_arg_parser()
    args = parser.parse_args(argv)
    return validate_runtime_args(args)
