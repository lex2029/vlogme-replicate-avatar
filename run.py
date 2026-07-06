from __future__ import annotations

import asyncio
import faulthandler
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path as SysPath

from cog import BasePredictor, Input, Path, Secret


ROOT = SysPath(__file__).resolve().parent
RUNTIME_ROOT = ROOT / "runtime" / "SmartBlog-Live"


def _log(message: str) -> None:
    print(f"[replicate-avatar] {message}", flush=True)


def _runtime_log_tail(max_chars: int = 12000) -> str:
    log_dir = RUNTIME_ROOT / "logs" / "torchrun"
    if not log_dir.exists():
        return ""
    files = [path for path in log_dir.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    chunks: list[str] = []
    remaining = max_chars
    for path in files[:8]:
        if remaining <= 0:
            break
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tail = raw[-min(len(raw), max(1000, remaining // 2)) :]
        chunk = f"\n--- {path.relative_to(RUNTIME_ROOT)} ---\n{tail}"
        chunks.append(chunk)
        remaining -= len(chunk)
    return "".join(chunks)[-max_chars:]


def _copy_input(src: Path, dst: SysPath) -> SysPath:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src), str(dst))
    return dst


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(str(os.environ.get(name, str(default)) or default).strip())
    except Exception:
        return int(default)


def _gpu_runtime_values() -> dict[str, str]:
    layout = os.environ.get("VLOGME_AVATAR_GPU_LAYOUT", "split").strip().lower() or "split"
    if layout in {"auto", "a100", "a100_auto"}:
        layout = "split"
    os.environ["VLOGME_AVATAR_GPU_LAYOUT_EFFECTIVE"] = str(layout)
    if layout in {"single", "1", "one"}:
        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("VLOGME_AVATAR_CUDA_VISIBLE_DEVICES", "0"),
            "TORCHRUN_NPROC": "1",
            "NUM_GPUS_DIT": "1",
            "ULYSSES_SIZE": os.environ.get("VLOGME_AVATAR_ULYSSES_SIZE", "1"),
            "ENABLE_VAE_PARALLEL": "0",
        }
    if layout in {"split", "split_vae", "dit1_vae1", "vae"}:
        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("VLOGME_AVATAR_CUDA_VISIBLE_DEVICES", "0,1"),
            "TORCHRUN_NPROC": "2",
            "NUM_GPUS_DIT": "1",
            "ULYSSES_SIZE": os.environ.get("VLOGME_AVATAR_ULYSSES_SIZE", "1"),
            "ENABLE_VAE_PARALLEL": "1",
        }
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("VLOGME_AVATAR_CUDA_VISIBLE_DEVICES", "0,1"),
        "TORCHRUN_NPROC": "2",
        "NUM_GPUS_DIT": "2",
        "ULYSSES_SIZE": os.environ.get("VLOGME_AVATAR_ULYSSES_SIZE", "1"),
        "ENABLE_VAE_PARALLEL": "0",
    }


def _set_default_env(asset_root: SysPath) -> None:
    gpu_values = _gpu_runtime_values()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    os.environ.setdefault("TORCH_FLOAT32_MATMUL_PRECISION", "high")
    os.environ.setdefault("TORCH_CUDA_MATMUL_ALLOW_TF32", "1")
    os.environ.setdefault("TORCH_CUDNN_ALLOW_TF32", "1")
    os.environ.setdefault("TORCH_CUDNN_BENCHMARK", "1")
    os.environ.setdefault("NCCL_P2P_PREWARM", "1")
    os.environ.setdefault("WORKER_BOOT_LOG", "1")
    os.environ.setdefault("WORKER_API_KEY", "replicate-local")
    os.environ.setdefault("SMARTBLOG_MOCK_CLAIM_FILE", "/tmp/vlogme-replicate-avatar-unused-claim.json")
    os.environ.setdefault("SMARTBLOG_MOCK_STATE_DIR", str(ROOT / "tmp" / "mock-state"))
    os.environ.setdefault("SUPABASE_URL", "https://replicate.local")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "replicate-local")

    os.environ["WORKER_ASSET_ROOT"] = str(asset_root)
    os.environ["HF_HOME"] = str(asset_root / "hf")
    os.environ["CKPT_DIR"] = str(asset_root / "ckpt" / "Wan2.2-S2V-14B")
    os.environ["LORA_PATH_DMD"] = str(asset_root / "ckpt" / "LiveAvatar" / "liveavatar.safetensors")
    os.environ["MERGED_NOISE_MODEL_DIR"] = str(
        asset_root / "ckpt" / "Wan2.2-S2V-14B-merged-liveavatar-prefp8-test"
    )
    os.environ.setdefault("USE_MERGED_CKPT", "1")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_values["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("TORCHRUN_NPROC", gpu_values["TORCHRUN_NPROC"])
    os.environ.setdefault("NUM_GPUS_DIT", gpu_values["NUM_GPUS_DIT"])
    os.environ.setdefault("ULYSSES_SIZE", gpu_values["ULYSSES_SIZE"])
    os.environ.setdefault("ENABLE_VAE_PARALLEL", gpu_values["ENABLE_VAE_PARALLEL"])
    os.environ.setdefault("MASTER_PORT", "29541")
    os.environ.setdefault("WORKER_TASK", "s2v-14B")
    os.environ.setdefault("WORKER_FPS", "16")
    os.environ.setdefault("INFER_FRAMES", os.environ.get("VLOGME_AVATAR_INFER_FRAMES", "32"))
    os.environ.setdefault("WORKER_SAMPLE_STEPS", os.environ.get("VLOGME_AVATAR_SAMPLE_STEPS", "6"))
    os.environ.setdefault("WORKER_AUDIO_SAMPLE_RATE", "16000")
    os.environ.setdefault("WORKER_AUDIO_NUM_CLIP_PAD_SEC", "0")
    os.environ.setdefault("GUIDE_SCALE", "4")
    os.environ.setdefault("SAMPLE_SOLVER", "euler")
    os.environ.setdefault("BASE_SEED", "420")
    os.environ.setdefault("TRAINING_CONFIG", "liveavatar/configs/s2v_causal_sft.yaml")
    os.environ.setdefault("SAVE_DIR", "./output/")
    os.environ.setdefault("SERVER_PORT", "7861")
    os.environ.setdefault("SERVER_NAME", "127.0.0.1")
    os.environ.setdefault("WORKER_LIVEAUDIO_MICRO_CHUNK_SCHEDULE_SAMPLES", "64000")
    os.environ.setdefault("SMARTBLOG_WAN_NUM_FRAMES_PER_BLOCK", "8")
    os.environ.setdefault("LIVE_AUDIO_STREAM_ASYNC_PRODUCER", "1")
    os.environ.setdefault("LIVE_AUDIO_STREAM_ASYNC_START_AFTER_FIRST_CLIP", "1")
    os.environ.setdefault("LIVE_AUDIO_STREAM_REFILL_DURING_DENOISE", "0")
    os.environ.setdefault("LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS", "24")
    os.environ.setdefault("LIVE_AUDIO_STREAM_MAX_TOTAL_CLIPS", "0")
    os.environ.setdefault("LIVE_AUDIO_STREAM_REPLY_START_MIN_CLIPS", "1")
    os.environ.setdefault("LIVE_AUDIO_STREAM_REPLY_MODEL_QUEUE_TARGET", "1")
    os.environ.setdefault("LIVE_AUDIO_STREAM_TAIL_FILL_MODE", "zero")
    os.environ.setdefault("LIVE_AUDIO_STREAM_FILL_NOISE_STD", "0.0003")
    os.environ.setdefault("LIVE_AUDIO_STREAM_FILL_NOISE_SEED", "420")
    os.environ.setdefault("LIVE_AUDIO_STREAM_CLIP_PROMPT_SWITCH", "1")
    os.environ.setdefault("LIVE_STREAM_UPDATE_REF_LATENTS", "0")
    os.environ.setdefault("LIVE_STREAM_UPDATE_MOTION_LATENTS", "1")
    os.environ.setdefault("LIVE_STREAM_UPDATE_MOTION_LATENTS_MODE", "decoded")
    os.environ.setdefault("LIVE_STREAM_MOTION_LATENTS_SYNC", "1")

    # Replicate should return a local MP4. No VlogMe upload, no remote RTX edge.
    os.environ["REMOTE_EDGE_ENABLED"] = "0"
    os.environ["REMOTE_EDGE_SKIP_LOCAL_DECODE"] = "0"
    os.environ["SMARTBLOG_RENDER_STREAM_FILE"] = "1"
    os.environ["SMARTBLOG_RENDER_FINALIZE_BACKGROUND"] = "0"
    os.environ["SMARTBLOG_RENDER_EDGE_FINALIZER_BACKGROUND"] = "0"
    os.environ["SMARTBLOG_RENDER_BURN_IN_SUBTITLES"] = os.environ.get("SMARTBLOG_RENDER_BURN_IN_SUBTITLES", "0")
    os.environ.setdefault("SMARTBLOG_STREAM_FILE_X264_PRESET", "superfast")
    os.environ.setdefault("SMARTBLOG_STREAM_FILE_X264_CRF", "19")
    os.environ.setdefault("SMARTBLOG_STREAM_FILE_QUEUE_BLOCKS", "4")
    os.environ.setdefault("SMARTBLOG_RENDER_SINGLE_AVATAR_ONE_PASS", "1")
    os.environ.setdefault("SMARTBLOG_RENDER_AVATAR_LIVEAUDIO_ONE_PASS", "1")
    os.environ.setdefault("SMARTBLOG_RENDER_TRIM_TRAILING_SILENCE", "1")
    os.environ.setdefault("USE_FP8", os.environ.get("VLOGME_AVATAR_USE_FP8", "0"))
    os.environ.setdefault("LIVEAVATAR_FP8_QUANT_COMPILE", os.environ.get("VLOGME_AVATAR_USE_FP8", "0"))
    os.environ.setdefault("ENABLE_COMPILE", os.environ.get("VLOGME_AVATAR_ENABLE_COMPILE", "false"))
    os.environ.setdefault("TORCH_COMPILE_DYNAMIC", os.environ.get("VLOGME_AVATAR_TORCH_COMPILE_DYNAMIC", "0"))
    os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
    os.environ.setdefault("TORCH_COMPILE_SAFE_FALLBACK", "1")
    os.environ.setdefault("TORCH_COMPILE_INCLUDE_FUNCS", "CausalHead_S2V.forward")
    os.environ.setdefault(
        "TORCH_COMPILE_SKIP_FUNCS",
        "_forward_inference,rope_apply,rope_apply_cond,CausalWanS2VAttention.forward,CausalWanS2VAttentionBlock._cross_attn_ffn",
    )
    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
    os.environ.setdefault("TORCHINDUCTOR_TRITON_CUDAGRAPH_TREES", "0")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", "0")
    os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "0")
    os.environ.setdefault("TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME", "0")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
    os.environ.setdefault("LIVEAVATAR_DISABLE_FLASH_ATTN", os.environ.get("VLOGME_AVATAR_DISABLE_FLASH_ATTN", "true"))
    os.environ.setdefault("LIVEAVATAR_DISABLE_CUDNN_ATTN", os.environ.get("VLOGME_AVATAR_DISABLE_CUDNN_ATTN", "false"))
    os.environ.setdefault("LIVEAVATAR_FORCE_TORCH_SDPA_MATH", os.environ.get("VLOGME_AVATAR_FORCE_TORCH_SDPA_MATH", "false"))
    os.environ.setdefault("LIVEAVATAR_FORCE_EAGER_ATTN", os.environ.get("VLOGME_AVATAR_FORCE_EAGER_ATTN", "false"))
    os.environ.setdefault("MODEL_TIMING_LOG", "1")
    os.environ.setdefault("POST_VAE_TIMING_LOG", "1")
    os.environ.setdefault("LIVE_AUDIO_TPP_TIMING_LOG", "1")


def _default_prompt() -> str:
    return (
        "A realistic talking-head video. Natural speech, stable face, subtle head motion, "
        "clear lip movement, neutral camera framing."
    )


def _default_negative_prompt() -> str:
    return "distorted face, extra teeth, deformed mouth, flicker, jitter, blur, low quality"


def _append_replicate_profile_overrides(asset_root: SysPath, *, size_profile: str = "b200") -> None:
    gpu_values = _gpu_runtime_values()
    profile_path = RUNTIME_ROOT / "config" / "worker_profile.local.conf"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    base_profile = "b300-avatar-commander" if size_profile == "b300" else "b200-avatar-commander"
    subprocess.run(
        ["bash", "scripts/profile.sh", base_profile],
        cwd=str(RUNTIME_ROOT),
        check=True,
    )

    if size_profile == "b300":
        size = "832*448"
        live_profile = "highres_2x"
    else:
        size = os.environ.get("VLOGME_AVATAR_SIZE", "704*384").strip() or "704*384"
        live_profile = os.environ.get("VLOGME_AVATAR_LIVE_PROFILE", "compact_704").strip() or "compact_704"
    profile_values = {
        "WORKER_PROFILE_NAME": "replicate-avatar",
        "WORKER_ASSET_ROOT": str(asset_root),
        "CUDA_VISIBLE_DEVICES": gpu_values["CUDA_VISIBLE_DEVICES"],
        "TORCHRUN_NPROC": gpu_values["TORCHRUN_NPROC"],
        "NUM_GPUS_DIT": gpu_values["NUM_GPUS_DIT"],
        "ULYSSES_SIZE": gpu_values["ULYSSES_SIZE"],
        "ENABLE_VAE_PARALLEL": gpu_values["ENABLE_VAE_PARALLEL"],
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE", "1"),
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1"),
        "TORCH_FLOAT32_MATMUL_PRECISION": os.environ.get("TORCH_FLOAT32_MATMUL_PRECISION", "high"),
        "TORCH_CUDA_MATMUL_ALLOW_TF32": os.environ.get("TORCH_CUDA_MATMUL_ALLOW_TF32", "1"),
        "TORCH_CUDNN_ALLOW_TF32": os.environ.get("TORCH_CUDNN_ALLOW_TF32", "1"),
        "TORCH_CUDNN_BENCHMARK": os.environ.get("TORCH_CUDNN_BENCHMARK", "1"),
        "NCCL_P2P_PREWARM": os.environ.get("NCCL_P2P_PREWARM", "1"),
        "HF_HOME": str(asset_root / "hf"),
        "CKPT_DIR": str(asset_root / "ckpt" / "Wan2.2-S2V-14B"),
        "LORA_PATH_DMD": str(asset_root / "ckpt" / "LiveAvatar" / "liveavatar.safetensors"),
        "MERGED_NOISE_MODEL_DIR": str(asset_root / "ckpt" / "Wan2.2-S2V-14B-merged-liveavatar-prefp8-test"),
        "USE_MERGED_CKPT": os.environ.get("USE_MERGED_CKPT", "1"),
        "SIZE": size,
        "SMARTBLOG_LIVE_PROFILE": live_profile,
        "SMARTBLOG_RENDER_VIDEO_PROFILE": live_profile,
        "SMARTBLOG_FORBID_LOCAL_MEDIA_SERVICES": "1",
        "SMARTBLOG_HUNYUAN_RENDER_ENABLED": "0",
        "SMARTBLOG_HUNYUAN_SERVICE_ENABLED": "0",
        "SMARTBLOG_HUNYUAN_SERVICE_ON_DEMAND": "0",
        "SMARTBLOG_HUNYUAN_SERVICE_REMOTE": "0",
        "SMARTBLOG_HUNYUAN_SERVICE_URL": "",
        "SMARTBLOG_HUNYUAN_MODEL_ID": "",
        "SMARTBLOG_HUNYUAN_PYTHON": "",
        "SMARTBLOG_HUNYUAN_HF_HOME": "",
        "SMARTBLOG_LTX_RENDER_ENABLED": "0",
        "SMARTBLOG_LTX_SERVICE_URL": "",
        "SMARTBLOG_LTX_SERVICE_WARMUP": "0",
        "SMARTBLOG_MMAUDIO_ENABLED": "0",
        "SMARTBLOG_MMAUDIO_SERVICE_ENABLED": "0",
        "SMARTBLOG_MMAUDIO_SERVICE_REMOTE": "0",
        "SMARTBLOG_MMAUDIO_SERVICE_URL": "",
        "SMARTBLOG_MMAUDIO_ROOT": "",
        "SMARTBLOG_MMAUDIO_PYTHON": "",
        "SMARTBLOG_MMAUDIO_HF_HOME": "",
        "SMARTBLOG_MUSETALK_SERVICE_ENABLED": "0",
        "REMOTE_EDGE_FILE_UPSCALE_ENABLED": "0",
        "REMOTE_EDGE_FILE_UPSCALE_RIFE": "0",
        "REMOTE_EDGE_FILE_REMOTE_FINALIZER": "0",
        "SMARTBLOG_RENDER_PREFETCH_INDEPENDENT_HUNYUAN": "0",
        "REMOTE_EDGE_ENABLED": "0",
        "REMOTE_EDGE_SKIP_LOCAL_DECODE": "0",
        "SMARTBLOG_RENDER_STREAM_FILE": "1",
        "SMARTBLOG_RENDER_FINALIZE_BACKGROUND": "0",
        "SMARTBLOG_RENDER_EDGE_FINALIZER_BACKGROUND": "0",
        "SMARTBLOG_RENDER_BURN_IN_SUBTITLES": "0",
        "SMARTBLOG_STREAM_FILE_X264_PRESET": os.environ.get("SMARTBLOG_STREAM_FILE_X264_PRESET", "superfast"),
        "SMARTBLOG_STREAM_FILE_X264_CRF": os.environ.get("SMARTBLOG_STREAM_FILE_X264_CRF", "19"),
        "SMARTBLOG_STREAM_FILE_QUEUE_BLOCKS": os.environ.get("SMARTBLOG_STREAM_FILE_QUEUE_BLOCKS", "4"),
        "LIVE_STREAM_KV_CACHE_FRAMES": os.environ.get("VLOGME_AVATAR_KV_CACHE_FRAMES", "32"),
        "LIVE_AUDIO_STREAM_ALLOW_LONG_CLIPS": "1",
        "LIVE_AUDIO_STREAM_MAX_CLIP_FRAMES": os.environ.get("VLOGME_AVATAR_MAX_CLIP_FRAMES", "32"),
        "SMARTBLOG_RENDER_ONEPASS_MAX_CONDITIONING_FRAMES": os.environ.get(
            "VLOGME_AVATAR_MAX_CONDITIONING_FRAMES", "32"
        ),
        "SMARTBLOG_RENDER_ONEPASS_MAX_AUDIO_CLIP_FRAMES": os.environ.get(
            "VLOGME_AVATAR_MAX_AUDIO_CLIP_FRAMES", "32"
        ),
        "SMARTBLOG_RENDER_ONEPASS_MIN_TAIL_FRAMES": os.environ.get("VLOGME_AVATAR_MIN_TAIL_FRAMES", "32"),
        "SMARTBLOG_RENDER_ONEPASS_BOUNDARY_PREROLL_FRAMES": os.environ.get(
            "VLOGME_AVATAR_BOUNDARY_PREROLL_FRAMES", "8"
        ),
        "LIVE_AUDIO_STREAM_ASYNC_PRODUCER": os.environ.get("LIVE_AUDIO_STREAM_ASYNC_PRODUCER", "1"),
        "LIVE_AUDIO_STREAM_ASYNC_START_AFTER_FIRST_CLIP": os.environ.get(
            "LIVE_AUDIO_STREAM_ASYNC_START_AFTER_FIRST_CLIP", "1"
        ),
        "LIVE_AUDIO_STREAM_REFILL_DURING_DENOISE": os.environ.get("LIVE_AUDIO_STREAM_REFILL_DURING_DENOISE", "0"),
        "LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS": os.environ.get("LIVE_AUDIO_STREAM_MAX_PENDING_CLIPS", "24"),
        "LIVE_AUDIO_STREAM_MAX_TOTAL_CLIPS": os.environ.get("LIVE_AUDIO_STREAM_MAX_TOTAL_CLIPS", "0"),
        "LIVE_AUDIO_STREAM_REPLY_START_MIN_CLIPS": os.environ.get("LIVE_AUDIO_STREAM_REPLY_START_MIN_CLIPS", "1"),
        "LIVE_AUDIO_STREAM_REPLY_MODEL_QUEUE_TARGET": os.environ.get(
            "LIVE_AUDIO_STREAM_REPLY_MODEL_QUEUE_TARGET", "1"
        ),
        "LIVE_AUDIO_STREAM_TAIL_FILL_MODE": os.environ.get("LIVE_AUDIO_STREAM_TAIL_FILL_MODE", "zero"),
        "LIVE_AUDIO_STREAM_FILL_NOISE_STD": os.environ.get("LIVE_AUDIO_STREAM_FILL_NOISE_STD", "0.0003"),
        "LIVE_AUDIO_STREAM_FILL_NOISE_SEED": os.environ.get("LIVE_AUDIO_STREAM_FILL_NOISE_SEED", "420"),
        "LIVE_AUDIO_STREAM_CLIP_PROMPT_SWITCH": os.environ.get("LIVE_AUDIO_STREAM_CLIP_PROMPT_SWITCH", "1"),
        "LIVE_STREAM_UPDATE_REF_LATENTS": os.environ.get("LIVE_STREAM_UPDATE_REF_LATENTS", "0"),
        "LIVE_STREAM_UPDATE_MOTION_LATENTS": os.environ.get("LIVE_STREAM_UPDATE_MOTION_LATENTS", "1"),
        "LIVE_STREAM_UPDATE_MOTION_LATENTS_MODE": os.environ.get("LIVE_STREAM_UPDATE_MOTION_LATENTS_MODE", "decoded"),
        "LIVE_STREAM_MOTION_LATENTS_SYNC": os.environ.get("LIVE_STREAM_MOTION_LATENTS_SYNC", "1"),
        "WORKER_AUDIO_NUM_CLIP_PAD_SEC": os.environ.get("WORKER_AUDIO_NUM_CLIP_PAD_SEC", "0"),
        "MODEL_TIMING_LOG": os.environ.get("MODEL_TIMING_LOG", "1"),
        "POST_VAE_TIMING_LOG": os.environ.get("POST_VAE_TIMING_LOG", "1"),
        "LIVE_AUDIO_TPP_TIMING_LOG": os.environ.get("LIVE_AUDIO_TPP_TIMING_LOG", "1"),
        "USE_FP8": os.environ.get("VLOGME_AVATAR_USE_FP8", "0"),
        "LIVEAVATAR_FP8_QUANT_COMPILE": os.environ.get("VLOGME_AVATAR_USE_FP8", "0"),
        "ENABLE_COMPILE": os.environ.get("VLOGME_AVATAR_ENABLE_COMPILE", "false"),
        "TORCH_COMPILE_DYNAMIC": os.environ.get("TORCH_COMPILE_DYNAMIC", "0"),
        "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1"),
        "TORCH_COMPILE_SAFE_FALLBACK": os.environ.get("TORCH_COMPILE_SAFE_FALLBACK", "1"),
        "TORCH_COMPILE_INCLUDE_FUNCS": os.environ.get("TORCH_COMPILE_INCLUDE_FUNCS", "CausalHead_S2V.forward"),
        "TORCH_COMPILE_SKIP_FUNCS": os.environ.get(
            "TORCH_COMPILE_SKIP_FUNCS",
            "_forward_inference,rope_apply,rope_apply_cond,CausalWanS2VAttention.forward,CausalWanS2VAttentionBlock._cross_attn_ffn",
        ),
        "TORCHINDUCTOR_CUDAGRAPHS": os.environ.get("TORCHINDUCTOR_CUDAGRAPHS", "0"),
        "TORCHINDUCTOR_TRITON_CUDAGRAPH_TREES": os.environ.get("TORCHINDUCTOR_TRITON_CUDAGRAPH_TREES", "0"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1"),
        "TORCHINDUCTOR_MAX_AUTOTUNE": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", "0"),
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0"),
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", "0"),
        "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "0"),
        "TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME": os.environ.get(
            "TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME", "0"
        ),
        "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "8"),
        "LIVEAVATAR_DISABLE_FLASH_ATTN": os.environ.get("LIVEAVATAR_DISABLE_FLASH_ATTN", "true"),
        "LIVEAVATAR_DISABLE_CUDNN_ATTN": os.environ.get("LIVEAVATAR_DISABLE_CUDNN_ATTN", "false"),
        "LIVEAVATAR_FORCE_TORCH_SDPA_MATH": os.environ.get("LIVEAVATAR_FORCE_TORCH_SDPA_MATH", "false"),
        "LIVEAVATAR_FORCE_EAGER_ATTN": os.environ.get("LIVEAVATAR_FORCE_EAGER_ATTN", "false"),
    }
    os.environ.update(profile_values)

    with profile_path.open("a", encoding="utf-8") as f:
        f.write("\n# Replicate avatar overrides\n")
        for key, value in profile_values.items():
            f.write(f"{key}={value}\n")


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not RUNTIME_ROOT.exists():
            raise RuntimeError(f"Missing runtime snapshot: {RUNTIME_ROOT}")
        sys.path.insert(0, str(RUNTIME_ROOT))
        os.chdir(str(RUNTIME_ROOT))

        self.asset_root = SysPath(os.environ.get("VLOGME_AVATAR_ASSET_ROOT", str(ROOT / "weights"))).resolve()
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self.modeld: subprocess.Popen[bytes] | None = None
        self.runtime_ready = False
        _set_default_env(self.asset_root)
        _append_replicate_profile_overrides(
            self.asset_root,
            size_profile=os.environ.get("VLOGME_AVATAR_SIZE_PROFILE", "b200").strip().lower() or "b200",
        )

    def _apply_cold_runtime_overrides(
        self,
        *,
        gpu_layout: str = "",
        use_fp8: int = -1,
        enable_compile: int = -1,
    ) -> None:
        requested: dict[str, str] = {}
        layout = str(gpu_layout or "").strip().lower()
        if layout:
            if layout not in {"auto", "a100", "a100_auto", "split", "split_vae", "dit1_vae1", "vae", "dit2", "single"}:
                raise RuntimeError(f"Unsupported gpu_layout override: {layout}")
            requested["VLOGME_AVATAR_GPU_LAYOUT"] = layout

        try:
            fp8_override = int(use_fp8)
        except Exception:
            fp8_override = -1
        if fp8_override in {0, 1}:
            requested["VLOGME_AVATAR_USE_FP8"] = "1" if fp8_override == 1 else "0"

        try:
            compile_override = int(enable_compile)
        except Exception:
            compile_override = -1
        if compile_override in {0, 1}:
            requested["VLOGME_AVATAR_ENABLE_COMPILE"] = "true" if compile_override == 1 else "false"

        if not requested:
            return
        if self.runtime_ready:
            raise RuntimeError("Runtime overrides require a cold worker; retry after scaling/restarting the deployment")

        os.environ.update(requested)
        gpu_values = _gpu_runtime_values()
        for key, value in gpu_values.items():
            os.environ[key] = value
        os.environ["USE_FP8"] = os.environ.get("VLOGME_AVATAR_USE_FP8", "0")
        os.environ["LIVEAVATAR_FP8_QUANT_COMPILE"] = os.environ.get("VLOGME_AVATAR_USE_FP8", "0")
        os.environ["ENABLE_COMPILE"] = os.environ.get("VLOGME_AVATAR_ENABLE_COMPILE", "false")
        _append_replicate_profile_overrides(
            self.asset_root,
            size_profile=os.environ.get("VLOGME_AVATAR_SIZE_PROFILE", "b200").strip().lower() or "b200",
        )
        _log(
            "cold runtime overrides applied: "
            f"layout={os.environ.get('VLOGME_AVATAR_GPU_LAYOUT_EFFECTIVE', os.environ.get('VLOGME_AVATAR_GPU_LAYOUT', ''))} "
            f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"num_gpus_dit={os.environ.get('NUM_GPUS_DIT', '')} "
            f"vae_parallel={os.environ.get('ENABLE_VAE_PARALLEL', '')} "
            f"fp8={os.environ.get('USE_FP8', '0')} "
            f"compile={os.environ.get('ENABLE_COMPILE', 'false')}"
        )

    def _ensure_runtime_ready(self) -> None:
        if self.runtime_ready:
            return

        preseed_mode = os.environ.get("VLOGME_AVATAR_PRESEED_MODE", "verify-or-preseed").strip().lower() or "verify-or-preseed"
        if preseed_mode not in {"skip", "verify", "preseed", "verify-or-preseed"}:
            raise RuntimeError("VLOGME_AVATAR_PRESEED_MODE must be skip, verify, preseed, or verify-or-preseed")
        if preseed_mode != "skip":
            started_at = time.monotonic()
            _log(f"runtime preseed mode: {preseed_mode}")
            subprocess.run(
                ["bash", "scripts/preseed_b200_avatar_assets.sh", preseed_mode],
                cwd=str(RUNTIME_ROOT),
                env=os.environ.copy(),
                check=True,
            )
            _log(f"runtime preseed finished in {time.monotonic() - started_at:.1f}s")

        started_at = time.monotonic()
        _log("starting model runtime")
        self.modeld = subprocess.Popen(
            ["bash", "scripts/modeld.sh"],
            cwd=str(RUNTIME_ROOT),
            env=os.environ.copy(),
        )

        from avalife.worker.model_client import ModelRuntimeClient

        client = ModelRuntimeClient()
        try:
            asyncio.run(client.wait_ready(timeout_sec=float(os.environ.get("MODEL_RUNTIME_READY_TIMEOUT_SEC", "1200"))))
        except Exception as exc:
            process_state = "not-started"
            if self.modeld is not None:
                returncode = self.modeld.poll()
                process_state = "running" if returncode is None else f"exited:{returncode}"
            tail = _runtime_log_tail()
            if tail:
                _log(f"model runtime log tail:{tail}")
            raise RuntimeError(f"model runtime failed to become ready; process={process_state}: {exc}") from exc
        self.runtime_ready = True
        _log(f"model runtime is ready in {time.monotonic() - started_at:.1f}s")

    def predict(
        self,
        avatar_image: Path = Input(description="Face/avatar reference image"),
        audio: Path = Input(description="Speech audio to animate"),
        sample_steps: int = Input(
            description="Denoising steps. Use 4 for smoke tests, 6+ for quality checks.",
            default=0,
        ),
        render_timeout_sec: int = Input(
            description="Optional render watchdog in seconds. 0 disables the internal timeout.",
            default=0,
        ),
        gpu_layout: str = Input(
            description="Optional cold-start GPU layout override: split, dit2, or single.",
            default="",
        ),
        use_fp8: int = Input(
            description="Optional cold-start FP8 override: -1 default, 0 disabled, 1 enabled.",
            default=-1,
        ),
        enable_compile: int = Input(
            description="Optional cold-start torch.compile override: -1 default, 0 disabled, 1 enabled.",
            default=-1,
        ),
        hf_token: Secret | None = Input(
            description="Optional Hugging Face token for private model weights",
            default=None,
        ),
    ) -> Path:
        _log("predict request accepted")
        if hf_token is not None:
            token = (hf_token.get_secret_value() or "").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                os.environ["SMARTBLOG_HF_TOKEN"] = token
                _log("HF token provided")
        self._apply_cold_runtime_overrides(
            gpu_layout=gpu_layout,
            use_fp8=use_fp8,
            enable_compile=enable_compile,
        )
        stack_watchdog_sec = _env_int("VLOGME_AVATAR_STACK_WATCHDOG_SEC", 180)
        if int(stack_watchdog_sec) > 0:
            try:
                faulthandler.enable(file=sys.stderr)
                faulthandler.dump_traceback_later(
                    int(stack_watchdog_sec),
                    repeat=True,
                    file=sys.stderr,
                )
                _log(f"stack watchdog enabled: {int(stack_watchdog_sec)}s")
            except Exception as exc:
                _log(f"stack watchdog setup skipped: {exc}")
        try:
            self._ensure_runtime_ready()
            _log("runtime ready; entering avatar render coroutine")
            timeout_sec = int(render_timeout_sec or _env_int("VLOGME_AVATAR_RENDER_TIMEOUT_SEC", 0) or 0)
            result = asyncio.run(
                self._predict_with_optional_timeout(
                    avatar_image=avatar_image,
                    audio=audio,
                    sample_steps_override=sample_steps,
                    timeout_sec=timeout_sec,
                )
            )
            _log("avatar render coroutine completed")
            return result
        except TimeoutError as exc:
            timeout_sec = int(render_timeout_sec or _env_int("VLOGME_AVATAR_RENDER_TIMEOUT_SEC", 0) or 0)
            _log(f"avatar render timed out after {timeout_sec}s")
            try:
                from avalife.worker.model_client import ModelRuntimeClient

                cancel_resp = asyncio.run(
                    ModelRuntimeClient().cancel_active_infer(reason=f"replicate_timeout_{timeout_sec}s")
                )
                _log(
                    "model runtime cancel after timeout: "
                    f"ok={1 if cancel_resp.ok else 0} cancelled={1 if cancel_resp.cancelled else 0} "
                    f"active_job={cancel_resp.active_job_id or '-'} total={cancel_resp.total_s:.1f}s "
                    f"error={cancel_resp.error or '-'}"
                )
            except Exception as cancel_exc:
                _log(f"model runtime cancel after timeout failed: {cancel_exc}")
            tail = _runtime_log_tail(max_chars=30000)
            if tail:
                _log(f"model runtime log tail after timeout:{tail}")
            raise RuntimeError(f"Avatar render timed out after {timeout_sec}s") from exc
        finally:
            if int(stack_watchdog_sec) > 0:
                try:
                    faulthandler.cancel_dump_traceback_later()
                except Exception:
                    pass

    async def _predict_with_optional_timeout(
        self,
        *,
        avatar_image: Path,
        audio: Path,
        sample_steps_override: int = 0,
        timeout_sec: int = 0,
    ) -> Path:
        coro = self._predict_async(
            avatar_image=avatar_image,
            audio=audio,
            sample_steps_override=sample_steps_override,
        )
        if int(timeout_sec or 0) <= 0:
            return await coro
        _log(f"avatar render timeout armed: {int(timeout_sec)}s")
        return await asyncio.wait_for(coro, timeout=float(timeout_sec))

    async def _predict_async(self, *, avatar_image: Path, audio: Path, sample_steps_override: int = 0) -> Path:
        prediction_started_at = time.monotonic()
        _log("starting avatar render")
        os.chdir(str(RUNTIME_ROOT))
        sys.path.insert(0, str(RUNTIME_ROOT))

        sample_steps = int(os.environ.get("VLOGME_AVATAR_SAMPLE_STEPS", "6") or 6)
        if int(sample_steps_override or 0) > 0:
            sample_steps = int(sample_steps_override)
        sample_steps = int(max(1, min(40, int(sample_steps))))
        _log(f"avatar render config resolved: sample_steps={sample_steps}")
        seed = int(os.environ.get("VLOGME_AVATAR_SEED", "420") or 420)
        face_restore = float(os.environ.get("VLOGME_AVATAR_FACE_RESTORE", "0.0") or 0.0)
        background_restore = float(os.environ.get("VLOGME_AVATAR_BACKGROUND_RESTORE", "0.0") or 0.0)
        prompt = _default_prompt()
        negative_prompt = _default_negative_prompt()

        os.environ["WORKER_SAMPLE_STEPS"] = str(int(sample_steps))
        os.environ["BASE_SEED"] = str(int(seed))

        from avalife.core.audio import auto_num_clip_for_audio, to_wav_16k_mono, wav_duration_seconds
        from avalife.model.protocol import InferRequest
        from avalife.worker.model_client import ModelRuntimeClient

        run_root = ROOT / "tmp" / f"run-{int(time.time() * 1000)}"
        input_dir = run_root / "inputs"
        avatar_path = _copy_input(avatar_image, input_dir / "avatar.png")
        raw_audio_path = _copy_input(audio, input_dir / "speech.input")
        audio_path = input_dir / "speech.wav"
        to_wav_16k_mono(str(raw_audio_path), str(audio_path))
        audio_duration_sec = float(wav_duration_seconds(str(audio_path)))
        if audio_duration_sec <= 0.0:
            raise RuntimeError("Input audio is empty or could not be decoded")
        _log(f"avatar inputs prepared: audio={audio_duration_sec:.2f}s run={run_root.name}")

        job_id = f"replicate_avatar_{int(time.time() * 1000)}"
        infer_frames = int(os.environ.get("INFER_FRAMES", "32") or 32)
        fps = int(os.environ.get("WORKER_FPS", "16") or 16)
        size = str(os.environ.get("SIZE", "704*384") or "704*384")
        num_clip = max(1, int(auto_num_clip_for_audio(str(audio_path), fps=int(fps), infer_frames=int(infer_frames))))
        final_path = run_root / "avatar.mp4"
        req = InferRequest(
            prompt=str(prompt or ""),
            image_path=str(avatar_path),
            audio_path=str(audio_path),
            num_clip=int(num_clip),
            sample_steps=int(sample_steps),
            sample_guide_scale=float(os.environ.get("GUIDE_SCALE", "4") or 4),
            infer_frames=int(infer_frames),
            size=str(size),
            base_seed=int(seed),
            sample_solver=str(os.environ.get("SAMPLE_SOLVER", "euler") or "euler"),
            face_restore=float(face_restore),
            background_restore=float(background_restore),
            job_id=str(job_id),
            enable_live_hls=False,
            live_raw_dir=None,
            save_live_raw_mp4=False,
            video_prompt=str(prompt or ""),
            negative_prompt=str(negative_prompt or ""),
            stream_file_output_path=str(final_path),
            stream_file_output_width=int(os.environ.get("VLOGME_AVATAR_OUTPUT_WIDTH", "720") or 720),
            stream_file_output_height=int(os.environ.get("VLOGME_AVATAR_OUTPUT_HEIGHT", "1280") or 1280),
            stream_file_output_fps=float(fps),
            stream_file_trim_duration_sec=float(audio_duration_sec),
            stream_file_interpolation=str(os.environ.get("VLOGME_AVATAR_STREAM_FILE_INTERPOLATION", "") or ""),
        )

        _log(
            "direct avatar infer: "
            f"layout={os.environ.get('VLOGME_AVATAR_GPU_LAYOUT_EFFECTIVE', os.environ.get('VLOGME_AVATAR_GPU_LAYOUT', 'auto')) or 'auto'} "
            f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"num_gpus_dit={os.environ.get('NUM_GPUS_DIT', '')} "
            f"vae_parallel={os.environ.get('ENABLE_VAE_PARALLEL', '')} "
            f"size={size} output={req.stream_file_output_width}x{req.stream_file_output_height} "
            f"fps={fps} audio={audio_duration_sec:.2f}s infer_frames={infer_frames} "
            f"clips={num_clip} steps={sample_steps}"
        )
        infer_started_at = time.monotonic()
        resp = await ModelRuntimeClient().infer(req=req)
        _log(
            "model infer returned: "
            f"ok={1 if resp.ok else 0} lock_wait={resp.lock_wait_s:.1f}s "
            f"run_single={resp.run_single_s:.1f}s total={resp.total_s:.1f}s "
            f"wall={time.monotonic() - infer_started_at:.1f}s"
        )
        if not resp.ok:
            raise RuntimeError(resp.error or "Avatar inference failed")

        output_path = SysPath(str(resp.video_path or final_path))
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Avatar inference finished without a local MP4 output: {output_path}")

        if output_path.resolve() != final_path.resolve():
            shutil.copyfile(str(output_path), str(final_path))
        _log(f"prediction finished in {time.monotonic() - prediction_started_at:.1f}s")
        return Path(str(final_path))
