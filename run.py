from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from argparse import Namespace
from pathlib import Path as SysPath
from typing import Any

from cog import BasePredictor, Input, Path, Secret


ROOT = SysPath(__file__).resolve().parent
RUNTIME_ROOT = ROOT / "runtime" / "SmartBlog-Live"


def _log(message: str) -> None:
    print(f"[replicate-avatar] {message}", flush=True)


def _file_uri(path: SysPath) -> str:
    return "file://" + urllib.parse.quote(str(path.resolve()))


def _copy_input(src: Path, dst: SysPath) -> SysPath:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src), str(dst))
    return dst


def _gpu_runtime_values() -> dict[str, str]:
    layout = os.environ.get("VLOGME_AVATAR_GPU_LAYOUT", "dit2").strip().lower() or "dit2"
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

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_values["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("TORCHRUN_NPROC", gpu_values["TORCHRUN_NPROC"])
    os.environ.setdefault("NUM_GPUS_DIT", gpu_values["NUM_GPUS_DIT"])
    os.environ.setdefault("ULYSSES_SIZE", gpu_values["ULYSSES_SIZE"])
    os.environ.setdefault("ENABLE_VAE_PARALLEL", gpu_values["ENABLE_VAE_PARALLEL"])
    os.environ.setdefault("MASTER_PORT", "29541")
    os.environ.setdefault("WORKER_TASK", "s2v-14B")
    os.environ.setdefault("WORKER_FPS", "16")
    os.environ.setdefault("INFER_FRAMES", "64")
    os.environ.setdefault("WORKER_SAMPLE_STEPS", "8")
    os.environ.setdefault("WORKER_AUDIO_SAMPLE_RATE", "16000")
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

    # Replicate should return a local MP4. No VlogMe upload, no remote RTX edge.
    os.environ["REMOTE_EDGE_ENABLED"] = "0"
    os.environ["REMOTE_EDGE_SKIP_LOCAL_DECODE"] = "0"
    os.environ["SMARTBLOG_RENDER_STREAM_FILE"] = "1"
    os.environ["SMARTBLOG_RENDER_FINALIZE_BACKGROUND"] = "0"
    os.environ["SMARTBLOG_RENDER_EDGE_FINALIZER_BACKGROUND"] = "0"
    os.environ["SMARTBLOG_RENDER_BURN_IN_SUBTITLES"] = os.environ.get("SMARTBLOG_RENDER_BURN_IN_SUBTITLES", "0")
    os.environ.setdefault("SMARTBLOG_RENDER_SINGLE_AVATAR_ONE_PASS", "1")
    os.environ.setdefault("SMARTBLOG_RENDER_AVATAR_LIVEAUDIO_ONE_PASS", "1")
    os.environ.setdefault("SMARTBLOG_RENDER_TRIM_TRAILING_SILENCE", "1")
    os.environ.setdefault("USE_FP8", os.environ.get("VLOGME_AVATAR_USE_FP8", "0"))
    os.environ.setdefault("LIVEAVATAR_FP8_QUANT_COMPILE", os.environ.get("VLOGME_AVATAR_USE_FP8", "0"))
    os.environ.setdefault("ENABLE_COMPILE", os.environ.get("VLOGME_AVATAR_ENABLE_COMPILE", "false"))


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
        size = "720*400"
        live_profile = "highres_1_5x"
    profile_values = {
        "WORKER_PROFILE_NAME": "replicate-avatar",
        "WORKER_ASSET_ROOT": str(asset_root),
        "CUDA_VISIBLE_DEVICES": gpu_values["CUDA_VISIBLE_DEVICES"],
        "TORCHRUN_NPROC": gpu_values["TORCHRUN_NPROC"],
        "NUM_GPUS_DIT": gpu_values["NUM_GPUS_DIT"],
        "ULYSSES_SIZE": gpu_values["ULYSSES_SIZE"],
        "ENABLE_VAE_PARALLEL": gpu_values["ENABLE_VAE_PARALLEL"],
        "HF_HOME": str(asset_root / "hf"),
        "CKPT_DIR": str(asset_root / "ckpt" / "Wan2.2-S2V-14B"),
        "LORA_PATH_DMD": str(asset_root / "ckpt" / "LiveAvatar" / "liveavatar.safetensors"),
        "MERGED_NOISE_MODEL_DIR": str(asset_root / "ckpt" / "Wan2.2-S2V-14B-merged-liveavatar-prefp8-test"),
        "SIZE": size,
        "SMARTBLOG_LIVE_PROFILE": live_profile,
        "SMARTBLOG_RENDER_VIDEO_PROFILE": live_profile,
        "REMOTE_EDGE_ENABLED": "0",
        "REMOTE_EDGE_SKIP_LOCAL_DECODE": "0",
        "SMARTBLOG_RENDER_STREAM_FILE": "1",
        "SMARTBLOG_RENDER_FINALIZE_BACKGROUND": "0",
        "SMARTBLOG_RENDER_EDGE_FINALIZER_BACKGROUND": "0",
        "SMARTBLOG_RENDER_BURN_IN_SUBTITLES": "0",
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
        "USE_FP8": os.environ.get("VLOGME_AVATAR_USE_FP8", "0"),
        "LIVEAVATAR_FP8_QUANT_COMPILE": os.environ.get("VLOGME_AVATAR_USE_FP8", "0"),
        "ENABLE_COMPILE": os.environ.get("VLOGME_AVATAR_ENABLE_COMPILE", "false"),
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
        asyncio.run(client.wait_ready(timeout_sec=float(os.environ.get("MODEL_RUNTIME_READY_TIMEOUT_SEC", "1200"))))
        self.runtime_ready = True
        _log(f"model runtime is ready in {time.monotonic() - started_at:.1f}s")

    def predict(
        self,
        avatar_image: Path = Input(description="Face/avatar reference image"),
        audio: Path = Input(description="Speech audio to animate"),
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
        self._ensure_runtime_ready()
        return asyncio.run(self._predict_async(avatar_image=avatar_image, audio=audio))

    async def _predict_async(self, *, avatar_image: Path, audio: Path) -> Path:
        prediction_started_at = time.monotonic()
        _log("starting avatar render")
        os.chdir(str(RUNTIME_ROOT))
        sys.path.insert(0, str(RUNTIME_ROOT))

        size_profile = os.environ.get("VLOGME_AVATAR_SIZE_PROFILE", "b200").strip().lower() or "b200"
        if size_profile not in {"b200", "b300"}:
            size_profile = "b200"
        sample_steps = int(os.environ.get("VLOGME_AVATAR_SAMPLE_STEPS", "8") or 8)
        seed = int(os.environ.get("VLOGME_AVATAR_SEED", "420") or 420)
        face_restore = float(os.environ.get("VLOGME_AVATAR_FACE_RESTORE", "0.5") or 0.5)
        background_restore = float(os.environ.get("VLOGME_AVATAR_BACKGROUND_RESTORE", "0.0") or 0.0)
        prompt = _default_prompt()
        negative_prompt = _default_negative_prompt()

        _append_replicate_profile_overrides(
            SysPath(os.environ.get("VLOGME_AVATAR_ASSET_ROOT", str(ROOT / "weights"))).resolve(),
            size_profile=str(size_profile),
        )
        os.environ["WORKER_SAMPLE_STEPS"] = str(int(sample_steps))
        os.environ["BASE_SEED"] = str(int(seed))

        from avalife.worker.smartblog_api import LocalSmartBlogMockClient
        from avalife.worker.smartblog_render import SmartBlogRenderOnlyWorker

        class ReplicateAvatarWorker(SmartBlogRenderOnlyWorker):
            async def _smartblog_download_file(self, *, url: str, out_path: str) -> str:
                raw = str(url or "").strip()
                if raw.startswith("file://"):
                    src = SysPath(urllib.parse.unquote(raw[len("file://") :]))
                elif raw and "://" not in raw and SysPath(raw).exists():
                    src = SysPath(raw)
                else:
                    return await super()._smartblog_download_file(url=url, out_path=out_path)
                dst = SysPath(out_path).resolve()
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(src), str(dst))
                return str(dst)

        run_root = ROOT / "tmp" / f"run-{int(time.time() * 1000)}"
        input_dir = run_root / "inputs"
        avatar_path = _copy_input(avatar_image, input_dir / "avatar.png")
        audio_path = _copy_input(audio, input_dir / "speech.input")

        job_id = f"replicate_avatar_{int(time.time() * 1000)}"
        video_config: dict[str, Any] = {
            "mode": "avatar",
            "num_inference_steps": int(sample_steps),
            "seed": int(seed),
            "prompt": str(prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
        }

        claim: dict[str, Any] = {
            "job": {
                "id": job_id,
                "job_type": "render_video",
                "payload_json": {
                    "avatar_url": _file_uri(avatar_path),
                    "audio_url": _file_uri(audio_path),
                    "video": video_config,
                    "filters": {
                        "face_restore": float(face_restore),
                        "background_restore": float(background_restore),
                    },
                },
            },
            "assets": {
                "avatar_url": _file_uri(avatar_path),
                "audio_chunks": [
                    {
                        "url": _file_uri(audio_path),
                        "local_path": str(audio_path),
                        "index": 0,
                        "text": "",
                        "video_prompt": str(prompt or ""),
                        "negative_prompt": str(negative_prompt or ""),
                    }
                ],
            },
            "upload": {},
        }

        worker = ReplicateAvatarWorker(args=Namespace(sample_guide_scale=0.0, sample_solver="euler"))
        worker._smartblog_api = LocalSmartBlogMockClient(
            os.environ["SMARTBLOG_MOCK_CLAIM_FILE"],
            state_dir=str(run_root / "mock-state"),
        )
        try:
            render_started_at = time.monotonic()
            plan = await worker._smartblog_render_video_job(claim)
            _log(f"avatar render job finished in {time.monotonic() - render_started_at:.1f}s")
        finally:
            await worker.aclose()

        output_path = SysPath(str(getattr(plan, "file_path", "") or ""))
        if not output_path.exists():
            raise RuntimeError("Avatar render finished without a local MP4 output")

        final_path = run_root / "avatar.mp4"
        shutil.copyfile(str(output_path), str(final_path))
        _log(f"prediction finished in {time.monotonic() - prediction_started_at:.1f}s")
        return Path(str(final_path))
