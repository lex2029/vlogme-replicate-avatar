from __future__ import annotations

import argparse
import base64
import gc
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from avalife.core.upload_retry import put_file_to_signed_url
from diffusers import HunyuanVideo15ImageToVideoPipeline, attention_backend
from diffusers.utils import export_to_video
from PIL import Image, ImageOps


LOG = logging.getLogger("smartblog-hunyuan-service")
_HUNYUAN_SERVICE_SEMAPHORE: threading.Semaphore | None = None


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _hunyuan_service_semaphore() -> threading.Semaphore:
    global _HUNYUAN_SERVICE_SEMAPHORE
    if _HUNYUAN_SERVICE_SEMAPHORE is None:
        max_concurrent = max(1, _env_int("SMARTBLOG_HUNYUAN_SERVICE_MAX_CONCURRENT", 1))
        _HUNYUAN_SERVICE_SEMAPHORE = threading.Semaphore(max_concurrent)
    return _HUNYUAN_SERVICE_SEMAPHORE


def _request_work_dir(prefix: str) -> Path:
    root = Path(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_REQUEST_DIR", "/tmp/smartblog-hunyuan-service-requests"))
    path = root / f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_suffix(name: str, default: str) -> str:
    suffix = Path(str(name or "")).suffix.lower()
    if not suffix or len(suffix) > 12:
        suffix = str(default)
    return suffix


def _write_base64_file(value: Any, *, out_dir: Path, prefix: str, default_ext: str) -> str:
    filename = ""
    content = value
    if isinstance(value, dict):
        filename = str(value.get("filename") or value.get("name") or "")
        content = value.get("data") or value.get("base64") or value.get("content") or ""
    text = str(content or "").strip()
    if "," in text[:128]:
        text = text.split(",", 1)[1]
    path = out_dir / f"{prefix}_{abs(hash(text[:512])) % 1000000}{_safe_suffix(filename, default_ext)}"
    with open(path, "wb") as f:
        f.write(base64.b64decode(text))
    return str(path)


def _download_url_file(url: str, *, out_dir: Path, prefix: str, default_ext: str) -> str:
    raw = str(url or "").strip()
    suffix = _safe_suffix(urllib.parse.urlparse(raw).path, default_ext)
    path = out_dir / f"{prefix}_{abs(hash(raw)) % 1000000}{suffix}"
    with urllib.request.urlopen(raw, timeout=300) as resp, open(path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return str(path)


def _materialize_hunyuan_request_inputs(request: dict[str, Any]) -> dict[str, Any]:
    out = dict(request or {})
    work_dir: Path | None = None

    def ensure_work_dir() -> Path:
        nonlocal work_dir
        if work_dir is None:
            work_dir = _request_work_dir("input")
        return work_dir

    paths: list[str] = []
    for key in ("conditioning_media_base64", "conditioning_media_b64"):
        values = out.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for i, value in enumerate(values):
            if value:
                paths.append(_write_base64_file(value, out_dir=ensure_work_dir(), prefix=f"conditioning_{i}", default_ext=".png"))
    for key in ("conditioning_media_urls", "conditioning_media_url"):
        values = out.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for i, value in enumerate(values):
            if value:
                paths.append(_download_url_file(str(value), out_dir=ensure_work_dir(), prefix=f"conditioning_url_{i}", default_ext=".png"))
    for key in ("image_base64", "input_media_base64"):
        if out.get(key):
            paths.append(_write_base64_file(out.get(key), out_dir=ensure_work_dir(), prefix=str(key), default_ext=".png"))
            break
    for key in ("image_url", "input_media_url"):
        if out.get(key):
            paths.append(_download_url_file(str(out.get(key)), out_dir=ensure_work_dir(), prefix=str(key), default_ext=".png"))
            break
    if paths:
        existing = out.get("conditioning_media_paths")
        if existing is None:
            existing = []
        if not isinstance(existing, list):
            existing = [existing]
        out["conditioning_media_paths"] = list(paths) + [str(p) for p in existing if str(p or "").strip()]
        out.setdefault("image_path", str(paths[0]))
    return out


def _upload_file_to_signed_url(*, signed_url: str, path: str, content_type: str) -> None:
    url = str(signed_url or "").strip()
    if not url:
        return
    put_file_to_signed_url(
        signed_url=str(url),
        path=str(path),
        content_type=str(content_type or "application/octet-stream"),
        connect_timeout=20.0,
        read_timeout=1800.0,
        env_prefix="SMARTBLOG_HUNYUAN_UPLOAD",
        log_prefix="hunyuan-signed-upload",
    )


def _publish_output_if_requested(response: dict[str, Any], *, output_path: str, request: dict[str, Any], content_type: str) -> dict[str, Any]:
    out = dict(response or {})
    upload_url = str(request.get("output_upload_url") or request.get("upload_url") or "").strip()
    if upload_url:
        _upload_file_to_signed_url(signed_url=upload_url, path=str(output_path), content_type=str(content_type))
        out["uploaded"] = True
        if request.get("output_storage_path"):
            out["output_storage_path"] = str(request.get("output_storage_path"))
        if request.get("output_public_url") or request.get("output_url"):
            out["output_url"] = str(request.get("output_public_url") or request.get("output_url"))
    if _env_flag("SMARTBLOG_HUNYUAN_SERVICE_RETURN_BASE64", "0") or bool(request.get("return_base64")):
        with open(str(output_path), "rb") as f:
            out["output_base64"] = base64.b64encode(f.read()).decode("ascii")
    return out


def _sync_cuda_if_needed(device: str) -> None:
    if torch.cuda.is_available() and str(device or "").startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _cuda_memory_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        return (
            float(torch.cuda.memory_allocated()) / (1024.0 ** 3),
            float(torch.cuda.memory_reserved()) / (1024.0 ** 3),
        )
    except Exception:
        return 0.0, 0.0


def _cleanup_cuda_after_generate(device: str, *, label: str) -> None:
    if not _env_flag("SMARTBLOG_HUNYUAN_CUDA_CLEANUP_AFTER_GENERATE", "1"):
        return
    if not torch.cuda.is_available() or not str(device or "").startswith("cuda"):
        return
    alloc_before, reserved_before = _cuda_memory_gib()
    try:
        _sync_cuda_if_needed(device)
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        _sync_cuda_if_needed(device)
    except Exception:
        LOG.exception("%s CUDA cleanup after generate failed", str(label))
        return
    alloc_after, reserved_after = _cuda_memory_gib()
    LOG.warning(
        "%s CUDA cleanup after generate: cuda_alloc %.2f->%.2f GiB cuda_reserved %.2f->%.2f GiB",
        str(label),
        float(alloc_before),
        float(alloc_after),
        float(reserved_before),
        float(reserved_after),
    )


def _env_text_list(name: str, default: str = "") -> list[str]:
    raw = str(os.getenv(name, default) or "").strip()
    if not raw:
        return []
    values: list[str] = []
    for item in raw.replace(";", ",").split(","):
        text = str(item or "").strip()
        if text and text.lower() not in {"0", "false", "no", "off", "none", "native"}:
            values.append(text)
    return values


def _fit_image(path: str, *, width: int, height: int) -> Image.Image:
    image = Image.open(str(path)).convert("RGB")
    return ImageOps.fit(image, (int(width), int(height)), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _snap_size(width: int, height: int) -> tuple[int, int]:
    multiple = int(max(1, int(os.getenv("SMARTBLOG_HUNYUAN_SIZE_MULTIPLE", "16") or "16")))
    if multiple <= 1:
        return int(width), int(height)
    width_i = max(multiple, int(round(float(width) / float(multiple))) * multiple)
    height_i = max(multiple, int(round(float(height) / float(multiple))) * multiple)
    return int(width_i), int(height_i)


class HunyuanResidentPipeline:
    def __init__(self, *, model_id: str, device: str | None = None, dtype: str = "bf16") -> None:
        self.model_id = str(model_id)
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if str(dtype).strip().lower() in {"fp16", "float16", "half"} else torch.bfloat16
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.ready_at = 0.0
        self.attention_backends = _env_text_list(
            "SMARTBLOG_HUNYUAN_ATTENTION_BACKENDS",
            os.getenv("SMARTBLOG_HUNYUAN_ATTENTION_BACKEND", ""),
        )
        self.attention_fallback = _env_flag("SMARTBLOG_HUNYUAN_ATTENTION_FALLBACK", "1")
        self._working_attention_backend: str | None = None
        self.pipe = self._load_pipeline()
        self.ready_at = time.time()

    def _load_pipeline(self) -> HunyuanVideo15ImageToVideoPipeline:
        LOG.warning(
            "loading Hunyuan pipeline model=%s dtype=%s device=%s",
            self.model_id,
            str(self.dtype),
            self.device,
        )
        pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        try:
            pipe.vae.enable_tiling()
        except Exception:
            LOG.exception("Hunyuan VAE tiling enable failed")
        if str(self.device).startswith("cuda"):
            pipe.to(self.device)
        else:
            pipe.enable_model_cpu_offload()
        if self.attention_backends:
            LOG.warning("Hunyuan attention backend candidates: %s", ",".join(self.attention_backends))
        if _env_flag("SMARTBLOG_HUNYUAN_ENABLE_CACHE", "0"):
            LOG.warning(
                "SMARTBLOG_HUNYUAN_ENABLE_CACHE requested, but the current Diffusers service "
                "does not expose Tencent deepcache/teacache/taylorcache controls; use the "
                "Tencent-source backend for real cache inference."
            )
        if _env_flag("SMARTBLOG_HUNYUAN_ENABLE_FP8_GEMM", "0"):
            LOG.warning(
                "SMARTBLOG_HUNYUAN_ENABLE_FP8_GEMM requested, but the current Diffusers service "
                "does not expose Tencent fp8 GEMM controls; sgl-kernel alone is not enough "
                "without the Tencent-source backend."
            )
        LOG.warning("Hunyuan pipeline loaded in %.2fs", float(time.time() - self.started_at))
        return pipe

    def _run_pipe(self, **kwargs):
        candidates: list[str] = []
        if self._working_attention_backend is not None:
            candidates.append(str(self._working_attention_backend))
        else:
            candidates.extend(self.attention_backends)
            if self.attention_fallback:
                candidates.append("")
        if not candidates:
            candidates = [""]

        last_error: Exception | None = None
        for candidate in candidates:
            backend = str(candidate or "").strip()
            try:
                if backend:
                    LOG.warning("Hunyuan attention backend active: %s", backend)
                    with attention_backend(backend):
                        result = self.pipe(**kwargs)
                else:
                    result = self.pipe(**kwargs)
                if self._working_attention_backend is None:
                    self._working_attention_backend = backend
                    LOG.warning("Hunyuan attention backend selected: %s", backend or "native")
                return result
            except Exception as e:
                last_error = e
                if not self.attention_fallback:
                    raise
                LOG.warning(
                    "Hunyuan attention backend failed: backend=%s err=%s; trying fallback",
                    backend or "native",
                    str(e)[:500],
                )
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                self._working_attention_backend = None
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Hunyuan pipeline failed before inference")

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        request = _materialize_hunyuan_request_inputs(dict(request or {}))
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        negative_prompt = str(request.get("negative_prompt") or "").strip()
        output_dir = Path(str(request.get("output_path") or request.get("output_dir") or "outputs/hunyuan_service")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        requested_width = int(max(32, int(request.get("width") or os.getenv("SMARTBLOG_HUNYUAN_WIDTH", "480"))))
        requested_height = int(max(32, int(request.get("height") or os.getenv("SMARTBLOG_HUNYUAN_HEIGHT", "864"))))
        width, height = _snap_size(int(requested_width), int(requested_height))
        num_frames = int(max(1, int(request.get("num_frames") or os.getenv("SMARTBLOG_HUNYUAN_NUM_FRAMES", "121"))))
        frame_rate = int(max(1, min(60, int(request.get("frame_rate") or os.getenv("SMARTBLOG_HUNYUAN_FPS", "16")))))
        seed = int(request.get("seed") if request.get("seed") is not None else int(os.getenv("SMARTBLOG_HUNYUAN_SEED", "420")))
        steps = int(max(1, int(request.get("num_inference_steps") or os.getenv("SMARTBLOG_HUNYUAN_NUM_INFERENCE_STEPS", "8"))))
        output_type = str(os.getenv("SMARTBLOG_HUNYUAN_OUTPUT_TYPE", "np") or "np").strip().lower()
        if output_type not in {"np", "pil"}:
            output_type = "np"
        conditioning_media_paths = request.get("conditioning_media_paths")
        if conditioning_media_paths is not None and not isinstance(conditioning_media_paths, list):
            conditioning_media_paths = [str(conditioning_media_paths)]
        image_path = ""
        if conditioning_media_paths:
            image_path = str(conditioning_media_paths[0] or "").strip()
        if not image_path:
            image_path = str(request.get("image_path") or request.get("input_media_path") or "").strip()
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Hunyuan I2V requires a conditioning image path, got: {image_path or '-'}")

        with self.lock:
            timings: dict[str, float] = {}
            t0 = time.perf_counter()

            def mark(name: str, started: float) -> None:
                timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - started)

            prep_t0 = time.perf_counter()
            image = _fit_image(image_path, width=int(width), height=int(height))
            generator = torch.Generator(device=self.device if str(self.device).startswith("cuda") else "cpu").manual_seed(seed)
            mark("prepare_request", prep_t0)
            LOG.warning(
                "generate Hunyuan size=%dx%d requested=%dx%d frames=%d fps=%d steps=%d seed=%d output_type=%s image=%s prompt_chars=%d negative_chars=%d",
                int(width),
                int(height),
                int(requested_width),
                int(requested_height),
                int(num_frames),
                int(frame_rate),
                int(steps),
                int(seed),
                str(output_type),
                os.path.basename(str(image_path)),
                int(len(prompt)),
                int(len(negative_prompt)),
            )
            _sync_cuda_if_needed(self.device)
            pipeline_t0 = time.perf_counter()
            result = self._run_pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_frames=int(num_frames),
                num_inference_steps=int(steps),
                generator=generator,
                output_type=str(output_type),
            )
            _sync_cuda_if_needed(self.device)
            mark("pipeline", pipeline_t0)
            frames = result.frames[0]
            actual_width = 0
            actual_height = 0
            try:
                first_frame = frames[0]
                if hasattr(first_frame, "shape"):
                    shape = tuple(int(v) for v in first_frame.shape)
                    actual_height, actual_width = int(shape[0]), int(shape[1])
                elif hasattr(first_frame, "size") and not isinstance(first_frame.size, int):
                    actual_width, actual_height = map(int, first_frame.size)
            except Exception:
                actual_width = 0
                actual_height = 0
            output_path = output_dir / f"hunyuan_{int(time.time())}_{abs(hash((prompt, seed, num_frames))) % 1000000}.mp4"
            write_t0 = time.perf_counter()
            export_to_video(frames, str(output_path), fps=int(frame_rate))
            mark("file_write", write_t0)
            elapsed = float(time.perf_counter() - t0)
            LOG.warning(
                "Hunyuan generate done in %.2fs pipeline=%.2fs prepare=%.2fs file_write=%.2fs frames=%d requested_size=%dx%d actual_size=%dx%d output=%s",
                elapsed,
                float(timings.get("pipeline", 0.0)),
                float(timings.get("prepare_request", 0.0)),
                float(timings.get("file_write", 0.0)),
                int(num_frames),
                int(width),
                int(height),
                int(actual_width),
                int(actual_height),
                str(output_path),
            )
            response = {
                "ok": True,
                "output_path": str(output_path),
                "output_paths": [str(output_path)],
                "elapsed_sec": elapsed,
                "timings_sec": dict(timings),
            }
            response = _publish_output_if_requested(response, output_path=str(output_path), request=request, content_type="video/mp4")
            del result, frames, image, generator
            _cleanup_cuda_after_generate(self.device, label="Hunyuan")
            return response

    def warmup(self) -> dict[str, Any]:
        if not _env_flag("SMARTBLOG_HUNYUAN_SERVICE_WARMUP", "1"):
            return {"ok": True, "skipped": True}
        warm_dir = Path(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_DIR", "/tmp/smartblog_hunyuan_warmup")).resolve()
        warm_dir.mkdir(parents=True, exist_ok=True)
        warm_image = warm_dir / "warmup.png"
        if not warm_image.exists():
            Image.new("RGB", (int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_WIDTH", "480")), int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_HEIGHT", "864"))), (128, 128, 128)).save(warm_image)
        request = {
            "prompt": os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_PROMPT", "A stable cinematic shot with gentle natural motion."),
            "negative_prompt": os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_NEGATIVE_PROMPT", "blurry, distorted, jittery, low quality"),
            "height": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_HEIGHT", "864")),
            "width": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_WIDTH", "480")),
            "num_frames": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_FRAMES", "17")),
            "frame_rate": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_FPS", "16")),
            "seed": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_SEED", "420")),
            "output_path": str(warm_dir),
            "conditioning_media_paths": [str(warm_image)],
            "num_inference_steps": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_STEPS", "4")),
        }
        return self.generate(request)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)


def _save_tencent_video(video: Any, path: str, *, fps: int) -> None:
    import einops
    import imageio
    import numpy as np

    if isinstance(video, torch.Tensor):
        if video.ndim == 5:
            if int(video.shape[0]) != 1:
                raise ValueError(f"expected one generated video, got tensor shape={tuple(video.shape)}")
            video = video[0]
        if video.ndim != 4:
            raise ValueError(f"expected CxFxHxW video tensor, got shape={tuple(video.shape)}")
        frames = (video.detach().float().clamp(0, 1) * 255).to(torch.uint8)
        frames = einops.rearrange(frames, "c f h w -> f h w c").cpu().numpy()
    else:
        frames = np.asarray(video)
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[0] in {1, 3, 4}:
            frames = np.transpose(frames, (1, 2, 3, 0))
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(str(path), frames, fps=int(fps))


class HunyuanTencentResidentPipeline:
    """Resident wrapper around Tencent-Hunyuan/HunyuanVideo-1.5 source runtime."""

    def __init__(self, *, model_id: str, device: str | None = None, dtype: str = "bf16") -> None:
        self.model_id = str(model_id)
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype_name = str(dtype or "bf16").strip().lower()
        self.dtype = torch.float32 if self.dtype_name == "fp32" else torch.bfloat16
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.ready_at = 0.0
        self.backend = "tencent"
        self.root = Path(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_ROOT", "/opt/hunyuan/HunyuanVideo-1.5")).resolve()
        self.model_path = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_MODEL_PATH", self.model_id) or self.model_id)
        self.local_model_dir = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_LOCAL_DIR", "") or "").strip()
        self.glyph_model_id = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_GLYPH_MODEL_ID", "AI-ModelScope/Glyph-SDXL-v2") or "").strip()
        self.auto_download_glyph = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_AUTO_DOWNLOAD_GLYPH", "1")
        self.auto_download_text_encoders = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_AUTO_DOWNLOAD_TEXT_ENCODERS", "1")
        self.llm_model_id = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_LLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct") or "").strip()
        self.byt5_model_id = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_BYT5_MODEL_ID", "google/byt5-small") or "").strip()
        self.vision_model_id = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_VISION_MODEL_ID", "black-forest-labs/FLUX.1-Redux-dev") or "").strip()
        self.llm_allow_patterns = _env_text_list(
            "SMARTBLOG_HUNYUAN_TENCENT_LLM_ALLOW_PATTERNS",
            "*.json,*.safetensors,*.jinja,*.txt,tokenizer*,merges.txt,vocab.json,*.model",
        )
        self.byt5_allow_patterns = _env_text_list(
            "SMARTBLOG_HUNYUAN_TENCENT_BYT5_ALLOW_PATTERNS",
            "config.json,pytorch_model.bin,special_tokens_map.json,tokenizer_config.json,spiece.model",
        )
        self.vision_allow_patterns = _env_text_list(
            "SMARTBLOG_HUNYUAN_TENCENT_VISION_ALLOW_PATTERNS",
            "image_encoder/config.json,image_encoder/model.safetensors,image_encoder/*.json",
        )
        self.allow_patterns = _env_text_list(
            "SMARTBLOG_HUNYUAN_TENCENT_ALLOW_PATTERNS",
            "transformer/480p_i2v_step_distilled/*,transformer/480p_t2v_distilled/*,vae/*,scheduler/*,text_encoder/*,vision_encoder/*",
        )
        self.resolution = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_RESOLUTION", "480p") or "480p").strip()
        self.cfg_distilled = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_CFG_DISTILLED", "0")
        self.step_distilled = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_STEP_DISTILLED", "1")
        self.t2v_cfg_distilled = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_T2V_CFG_DISTILLED", "1")
        self.t2v_step_distilled = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_T2V_STEP_DISTILLED", "0")
        self.sparse_attn = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_SPARSE_ATTN", "0")
        self.enable_sr = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_SR", "0")
        self.prompt_rewrite = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_REWRITE", "0")
        self.enable_cache = _env_flag("SMARTBLOG_HUNYUAN_ENABLE_CACHE", "0")
        self.cache_type = str(os.getenv("SMARTBLOG_HUNYUAN_CACHE_TYPE", "teacache") or "teacache").strip()
        self.cache_start_step = _env_int("SMARTBLOG_HUNYUAN_CACHE_START_STEP", 11)
        self.cache_end_step = _env_int("SMARTBLOG_HUNYUAN_CACHE_END_STEP", 45)
        self.cache_step_interval = _env_int("SMARTBLOG_HUNYUAN_CACHE_STEP_INTERVAL", 4)
        self.no_cache_block_id = str(os.getenv("SMARTBLOG_HUNYUAN_NO_CACHE_BLOCK_ID", "53") or "53").strip()
        self.enable_fp8_gemm = _env_flag("SMARTBLOG_HUNYUAN_ENABLE_FP8_GEMM", "0")
        self.quant_type = str(os.getenv("SMARTBLOG_HUNYUAN_FP8_QUANT_TYPE", "fp8-per-token-sgl") or "fp8-per-token-sgl").strip()
        self.include_patterns = str(os.getenv("SMARTBLOG_HUNYUAN_FP8_INCLUDE_PATTERNS", "double_blocks") or "double_blocks").strip()
        self.use_sageattn = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_SAGE_ATTN", "0")
        self.enable_torch_compile = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_TORCH_COMPILE", "0")
        self.offloading = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_OFFLOADING", "0")
        self.group_offloading = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_GROUP_OFFLOADING", "0")
        self.overlap_group_offloading = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_OVERLAP_GROUP_OFFLOADING", "1")
        self.keep_task_pipelines = _env_flag("SMARTBLOG_HUNYUAN_TENCENT_KEEP_TASK_PIPELINES", "0")
        self.pipe: Any | None = None
        self.pipe_task = ""
        self._pipelines: dict[str, Any] = {}
        self.pipe = self._get_pipeline_unlocked("i2v")
        self.ready_at = time.time()

    def _import_tencent_runtime(self) -> tuple[Any, Any, Any]:
        if not self.root.exists():
            raise RuntimeError(f"Tencent Hunyuan runtime root is missing: {self.root}")
        root_s = str(self.root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(os.getenv("SMARTBLOG_HUNYUAN_TENCENT_MASTER_PORT", "29591") or "29591")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.set_device(0)

        parallel_states = importlib.import_module("hyvideo.commons.parallel_states")
        infer_state_mod = importlib.import_module("hyvideo.commons.infer_state")
        pipeline_mod = importlib.import_module("hyvideo.pipelines.hunyuan_video_pipeline")
        try:
            parallel_states.get_parallel_state()
        except Exception:
            LOG.warning("initializing Tencent Hunyuan parallel state")
            parallel_states.initialize_parallel_state(sp=1, dp_replicate=1)
        return pipeline_mod.HunyuanVideo_1_5_Pipeline, infer_state_mod.InferState, pipeline_mod

    def _resolve_model_path(self) -> str:
        model_path = str(self.model_path).strip()
        if os.path.isdir(model_path):
            return model_path
        if "/" not in model_path:
            return model_path
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {"repo_id": model_path}
        if self.local_model_dir:
            Path(self.local_model_dir).mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = self.local_model_dir
        if self.allow_patterns:
            kwargs["allow_patterns"] = list(self.allow_patterns)
        LOG.warning(
            "downloading/resolving Tencent Hunyuan snapshot: repo=%s local_dir=%s allow_patterns=%s",
            model_path,
            self.local_model_dir or "-",
            ",".join(self.allow_patterns) if self.allow_patterns else "-",
        )
        return str(snapshot_download(**kwargs))

    def _ensure_glyph_checkpoint(self, model_path: str) -> None:
        glyph_root = Path(model_path) / "text_encoder" / "Glyph-SDXL-v2"
        required = (
            glyph_root / "assets" / "color_idx.json",
            glyph_root / "assets" / "multilingual_10-lang_idx.json",
            glyph_root / "checkpoints" / "byt5_model.pt",
        )
        if all(path.exists() for path in required):
            return
        missing = [str(path.relative_to(glyph_root)) for path in required if not path.exists()]
        if not self.auto_download_glyph:
            LOG.warning("Tencent Hunyuan Glyph checkpoint missing at %s: %s", str(glyph_root), ",".join(missing))
            return
        if not self.glyph_model_id:
            raise RuntimeError(f"Tencent Hunyuan Glyph checkpoint is missing at {glyph_root}, and no Glyph model id is configured")

        LOG.warning(
            "downloading/resolving Tencent Hunyuan Glyph checkpoint: model=%s local_dir=%s missing=%s",
            self.glyph_model_id,
            str(glyph_root),
            ",".join(missing),
        )
        glyph_root.mkdir(parents=True, exist_ok=True)
        modelscope_bin = shutil.which("modelscope") or str(Path(sys.executable).with_name("modelscope"))
        subprocess.run(
            [
                modelscope_bin,
                "download",
                "--model",
                self.glyph_model_id,
                "--local_dir",
                str(glyph_root),
                "--include",
                "assets/color_idx.json",
                "assets/multilingual_10-lang_idx.json",
                "checkpoints/byt5_model.pt",
                "--max-workers",
                "4",
            ],
            check=True,
        )
        if not all(path.exists() for path in required):
            missing_after = [str(path.relative_to(glyph_root)) for path in required if not path.exists()]
            raise RuntimeError(
                f"Tencent Hunyuan Glyph checkpoint download did not create required files at {glyph_root}: {missing_after}"
            )

    def _ensure_hf_snapshot_dir(
        self,
        *,
        repo_id: str,
        target_dir: Path,
        label: str,
        required_files: tuple[str, ...],
        allow_patterns: tuple[str, ...] = (),
    ) -> None:
        required = tuple(target_dir / item for item in required_files)
        if all(path.exists() for path in required):
            return
        missing = [str(path.relative_to(target_dir)) for path in required if not path.exists()]
        if not self.auto_download_text_encoders:
            LOG.warning("Tencent Hunyuan %s missing at %s: %s", label, str(target_dir), ",".join(missing))
            return
        if not repo_id:
            raise RuntimeError(f"Tencent Hunyuan {label} is missing at {target_dir}, and no repo id is configured")
        hf_bin = shutil.which("huggingface-cli") or str(Path(sys.executable).with_name("huggingface-cli"))
        LOG.warning(
            "downloading/resolving Tencent Hunyuan %s: repo=%s local_dir=%s missing=%s allow_patterns=%s",
            label,
            repo_id,
            str(target_dir),
            ",".join(missing),
            ",".join(allow_patterns) if allow_patterns else "-",
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = [hf_bin, "download", repo_id, "--local-dir", str(target_dir)]
        for pattern in allow_patterns:
            if str(pattern).strip():
                cmd.extend(["--include", str(pattern).strip()])
        subprocess.run(cmd, check=True)
        if not all(path.exists() for path in required):
            missing_after = [str(path.relative_to(target_dir)) for path in required if not path.exists()]
            raise RuntimeError(
                f"Tencent Hunyuan {label} download did not create required files at {target_dir}: {missing_after}"
            )

    def _ensure_text_encoders(self, model_path: str) -> None:
        text_root = Path(model_path) / "text_encoder"
        self._ensure_hf_snapshot_dir(
            repo_id=self.llm_model_id,
            target_dir=text_root / "llm",
            label="LLM text encoder",
            required_files=("config.json",),
            allow_patterns=tuple(self.llm_allow_patterns),
        )
        self._ensure_hf_snapshot_dir(
            repo_id=self.byt5_model_id,
            target_dir=text_root / "byt5-small",
            label="ByT5 text encoder",
            required_files=("config.json",),
            allow_patterns=tuple(self.byt5_allow_patterns),
        )

    def _ensure_vision_encoder(self, model_path: str) -> None:
        self._ensure_hf_snapshot_dir(
            repo_id=self.vision_model_id,
            target_dir=Path(model_path) / "vision_encoder" / "siglip",
            label="SigLIP vision encoder",
            required_files=("image_encoder/config.json", "image_encoder/model.safetensors"),
            allow_patterns=tuple(self.vision_allow_patterns),
        )

    def _task_cfg_distilled(self, task: str) -> bool:
        return bool(self.t2v_cfg_distilled if str(task) == "t2v" else self.cfg_distilled)

    def _task_step_distilled(self, task: str) -> bool:
        return bool(self.t2v_step_distilled if str(task) == "t2v" else self.step_distilled)

    def _make_infer_state(self, infer_state_cls: Any, *, steps: int, step_distilled: bool | None = None) -> Any:
        enable_cache = bool(self.enable_cache)
        effective_step_distilled = bool(self.step_distilled if step_distilled is None else step_distilled)
        if effective_step_distilled and enable_cache:
            LOG.warning(
                "Tencent Hunyuan disables cache with step-distilled models; "
                "running step_distilled=%d steps=%d without cache",
                1,
                int(steps),
            )
            enable_cache = False
        total_steps = int(max(1, steps))
        cache_end = min(int(self.cache_end_step), total_steps)
        cache_start = min(max(0, int(self.cache_start_step)), cache_end)
        return infer_state_cls(
            enable_sageattn=bool(self.use_sageattn),
            sage_blocks_range=list(range(0, 54)),
            enable_torch_compile=bool(self.enable_torch_compile),
            enable_cache=bool(enable_cache),
            cache_type=str(self.cache_type),
            no_cache_block_id=[int(v) for v in self.no_cache_block_id.replace(";", ",").split(",") if str(v).strip().isdigit()],
            cache_start_step=int(cache_start),
            cache_end_step=int(cache_end),
            total_steps=int(total_steps),
            cache_step_interval=int(max(1, self.cache_step_interval)),
            use_fp8_gemm=bool(self.enable_fp8_gemm),
            quant_type=str(self.quant_type),
            include_patterns=[p.strip() for p in self.include_patterns.replace(";", ",").split(",") if p.strip()],
        )

    def _load_pipeline(self, task: str = "i2v") -> Any:
        pipeline_cls, infer_state_cls, _ = self._import_tencent_runtime()
        task = "t2v" if str(task).strip().lower() == "t2v" else "i2v"
        cfg_distilled = bool(self._task_cfg_distilled(task))
        step_distilled = bool(self._task_step_distilled(task))
        transformer_version = pipeline_cls.get_transformer_version(
            self.resolution,
            task,
            cfg_distilled,
            step_distilled,
            self.sparse_attn,
        )
        steps = _env_int("SMARTBLOG_HUNYUAN_NUM_INFERENCE_STEPS", 8)
        infer_state = self._make_infer_state(infer_state_cls, steps=steps, step_distilled=bool(step_distilled))
        resolved_model_path = self._resolve_model_path()
        self._ensure_glyph_checkpoint(resolved_model_path)
        self._ensure_text_encoders(resolved_model_path)
        if task == "i2v":
            self._ensure_vision_encoder(resolved_model_path)
        LOG.warning(
            "loading Tencent Hunyuan pipeline task=%s root=%s model_path=%s transformer=%s dtype=%s sr=%d cache=%d cache_type=%s fp8_gemm=%d offload=%d group_offload=%d",
            str(task),
            str(self.root),
            str(resolved_model_path),
            str(transformer_version),
            str(self.dtype),
            1 if self.enable_sr else 0,
            1 if bool(getattr(infer_state, "enable_cache", False)) else 0,
            str(self.cache_type),
            1 if bool(self.enable_fp8_gemm) else 0,
            1 if bool(self.offloading) else 0,
            1 if bool(self.group_offloading) else 0,
        )
        t0 = time.time()
        device = torch.device("cuda" if str(self.device).startswith("cuda") else "cpu")
        init_device = torch.device("cpu") if bool(self.group_offloading) else device
        pipe = pipeline_cls.create_pipeline(
            pretrained_model_name_or_path=str(resolved_model_path),
            transformer_version=str(transformer_version),
            create_sr_pipeline=bool(self.enable_sr),
            transformer_dtype=self.dtype,
            device=device,
            transformer_init_device=init_device,
        )
        pipe.apply_infer_optimization(
            infer_state=infer_state,
            enable_offloading=bool(self.offloading),
            enable_group_offloading=bool(self.group_offloading),
            overlap_group_offloading=bool(self.overlap_group_offloading),
        )
        if bool(self.enable_sr) and hasattr(pipe, "sr_pipeline"):
            sr_state = self._make_infer_state(infer_state_cls, steps=steps, step_distilled=bool(step_distilled))
            sr_state.enable_cache = False
            pipe.sr_pipeline.apply_infer_optimization(
                infer_state=sr_state,
                enable_offloading=bool(self.offloading),
                enable_group_offloading=bool(self.group_offloading),
                overlap_group_offloading=bool(self.overlap_group_offloading),
            )
        LOG.warning("Tencent Hunyuan pipeline task=%s loaded in %.2fs", str(task), float(time.time() - t0))
        return pipe

    def _get_pipeline_unlocked(self, task: str) -> Any:
        task = "t2v" if str(task).strip().lower() == "t2v" else "i2v"
        if bool(self.keep_task_pipelines):
            pipe = self._pipelines.get(task)
            if pipe is None:
                pipe = self._load_pipeline(task)
                self._pipelines[task] = pipe
            self.pipe = pipe
            self.pipe_task = task
            return pipe
        if self.pipe is not None and self.pipe_task == task:
            return self.pipe
        if self.pipe is not None:
            LOG.warning("unloading Tencent Hunyuan task=%s before loading task=%s", str(self.pipe_task or "-"), str(task))
            old_pipe = self.pipe
            self.pipe = None
            self.pipe_task = ""
            try:
                del old_pipe
            except Exception:
                pass
            _cleanup_cuda_after_generate(self.device, label=f"Tencent Hunyuan unload {task}")
        self.pipe = self._load_pipeline(task)
        self.pipe_task = task
        return self.pipe

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        request = _materialize_hunyuan_request_inputs(dict(request or {}))
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        negative_prompt = str(request.get("negative_prompt") or "").strip()
        output_dir = Path(str(request.get("output_path") or request.get("output_dir") or "outputs/hunyuan_service")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        requested_width = int(max(32, int(request.get("width") or os.getenv("SMARTBLOG_HUNYUAN_WIDTH", "480"))))
        requested_height = int(max(32, int(request.get("height") or os.getenv("SMARTBLOG_HUNYUAN_HEIGHT", "832"))))
        num_frames = int(max(1, int(request.get("num_frames") or os.getenv("SMARTBLOG_HUNYUAN_NUM_FRAMES", "121"))))
        frame_rate = int(max(1, min(60, int(request.get("frame_rate") or os.getenv("SMARTBLOG_HUNYUAN_FPS", "16")))))
        seed = int(request.get("seed") if request.get("seed") is not None else int(os.getenv("SMARTBLOG_HUNYUAN_SEED", "420")))
        steps = int(max(1, int(request.get("num_inference_steps") or os.getenv("SMARTBLOG_HUNYUAN_NUM_INFERENCE_STEPS", "8"))))
        conditioning_media_paths = request.get("conditioning_media_paths")
        if conditioning_media_paths is not None and not isinstance(conditioning_media_paths, list):
            conditioning_media_paths = [str(conditioning_media_paths)]
        image_path = ""
        if conditioning_media_paths:
            image_path = str(conditioning_media_paths[0] or "").strip()
        if not image_path:
            image_path = str(request.get("image_path") or request.get("input_media_path") or "").strip()
        requested_task = str(request.get("task") or request.get("render_mode") or request.get("renderMode") or "").strip().lower().replace("-", "_")
        task = "t2v" if requested_task in {"t2v", "text_to_video", "text2video", "hunyuan_t2v"} else "i2v"
        if task != "t2v" and (not image_path or not os.path.exists(image_path)):
            raise ValueError(f"Tencent Hunyuan I2V requires a conditioning image path, got: {image_path or '-'}")

        with self.lock:
            timings: dict[str, float] = {}
            t0 = time.perf_counter()

            def mark(name: str, started: float) -> None:
                timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - started)

            prep_t0 = time.perf_counter()
            aspect_ratio = f"{int(requested_width)}:{int(requested_height)}"
            mark("prepare_request", prep_t0)
            LOG.warning(
                "generate Tencent Hunyuan task=%s requested=%dx%d frames=%d fps=%d steps=%d seed=%d sr=%d image=%s prompt_chars=%d negative_chars=%d cache=%d fp8_gemm=%d",
                str(task),
                int(requested_width),
                int(requested_height),
                int(num_frames),
                int(frame_rate),
                int(steps),
                int(seed),
                1 if self.enable_sr else 0,
                os.path.basename(str(image_path)) if image_path else "-",
                int(len(prompt)),
                int(len(negative_prompt)),
                1 if bool(self.enable_cache and not self._task_step_distilled(task)) else 0,
                1 if bool(self.enable_fp8_gemm) else 0,
            )
            _sync_cuda_if_needed(self.device)
            pipeline_t0 = time.perf_counter()
            pipe = self._get_pipeline_unlocked(str(task))
            kwargs: dict[str, Any] = {}
            if task == "i2v":
                kwargs["reference_image"] = str(image_path)
            out = pipe(
                enable_sr=bool(self.enable_sr),
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                num_inference_steps=int(steps),
                sr_num_inference_steps=None,
                video_length=int(num_frames),
                negative_prompt=negative_prompt,
                seed=int(seed),
                output_type="pt",
                prompt_rewrite=bool(self.prompt_rewrite),
                return_pre_sr_video=False,
                **kwargs,
            )
            _sync_cuda_if_needed(self.device)
            mark("pipeline", pipeline_t0)
            video = out.sr_videos if bool(self.enable_sr) and getattr(out, "sr_videos", None) is not None else out.videos
            actual_width = 0
            actual_height = 0
            try:
                tensor = video[0] if isinstance(video, torch.Tensor) and video.ndim == 5 else video
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                    actual_height = int(tensor.shape[-2])
                    actual_width = int(tensor.shape[-1])
            except Exception:
                actual_width = 0
                actual_height = 0
            output_path = output_dir / f"hunyuan_tencent_{int(time.time())}_{abs(hash((prompt, seed, num_frames))) % 1000000}.mp4"
            write_t0 = time.perf_counter()
            _save_tencent_video(video, str(output_path), fps=int(frame_rate))
            mark("file_write", write_t0)
            elapsed = float(time.perf_counter() - t0)
            LOG.warning(
                "Tencent Hunyuan generate done in %.2fs pipeline=%.2fs prepare=%.2fs file_write=%.2fs frames=%d requested_size=%dx%d actual_size=%dx%d output=%s",
                elapsed,
                float(timings.get("pipeline", 0.0)),
                float(timings.get("prepare_request", 0.0)),
                float(timings.get("file_write", 0.0)),
                int(num_frames),
                int(requested_width),
                int(requested_height),
                int(actual_width),
                int(actual_height),
                str(output_path),
            )
            response = {
                "ok": True,
                "task": str(task),
                "output_path": str(output_path),
                "output_paths": [str(output_path)],
                "elapsed_sec": elapsed,
                "timings_sec": dict(timings),
            }
            response = _publish_output_if_requested(response, output_path=str(output_path), request=request, content_type="video/mp4")
            del out, video
            _cleanup_cuda_after_generate(self.device, label="Tencent Hunyuan")
            return response

    def warmup(self) -> dict[str, Any]:
        if not _env_flag("SMARTBLOG_HUNYUAN_SERVICE_WARMUP", "1"):
            return {"ok": True, "skipped": True}
        warm_dir = Path(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_DIR", "/tmp/smartblog_hunyuan_warmup")).resolve()
        warm_dir.mkdir(parents=True, exist_ok=True)
        warm_image = warm_dir / "warmup.png"
        if not warm_image.exists():
            Image.new("RGB", (int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_WIDTH", "480")), int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_HEIGHT", "832"))), (128, 128, 128)).save(warm_image)
        request = {
            "prompt": os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_PROMPT", "A stable cinematic shot with gentle natural motion."),
            "negative_prompt": os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_NEGATIVE_PROMPT", "blurry, distorted, jittery, low quality"),
            "height": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_HEIGHT", "832")),
            "width": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_WIDTH", "480")),
            "num_frames": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_FRAMES", "17")),
            "frame_rate": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_FPS", "16")),
            "seed": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_SEED", "420")),
            "output_path": str(warm_dir),
            "conditioning_media_paths": [str(warm_image)],
            "num_inference_steps": int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_WARMUP_STEPS", "8")),
        }
        return self.generate(request)


def _create_resident(*, model_id: str, device: str | None, dtype: str) -> Any:
    backend = str(os.getenv("SMARTBLOG_HUNYUAN_BACKEND", "diffusers") or "diffusers").strip().lower()
    if backend in {"tencent", "source", "official"}:
        return HunyuanTencentResidentPipeline(model_id=model_id, device=device, dtype=dtype)
    if backend not in {"diffusers", "hf"}:
        LOG.warning("unknown SMARTBLOG_HUNYUAN_BACKEND=%s; falling back to diffusers", backend)
    resident = HunyuanResidentPipeline(model_id=model_id, device=device, dtype=dtype)
    setattr(resident, "backend", "diffusers")
    return resident


class HunyuanHandler(BaseHTTPRequestHandler):
    server_version = "SmartBlogHunyuanService/1.0"

    @property
    def resident(self) -> HunyuanResidentPipeline:
        return getattr(self.server, "resident")  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _run_generate_queued(self, payload: dict[str, Any]) -> dict[str, Any]:
        sem = _hunyuan_service_semaphore()
        queued_at = time.perf_counter()
        acquired_immediately = sem.acquire(blocking=False)
        if not acquired_immediately:
            LOG.warning("Hunyuan service queued: another GPU request is running")
            sem.acquire()
        queue_sec = float(time.perf_counter() - queued_at)
        try:
            result = self.resident.generate(payload)
            if isinstance(result, dict):
                result.setdefault("queue_sec", queue_sec)
            return result
        finally:
            sem.release()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health"}:
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "ready": True,
                    "pid": os.getpid(),
                    "device": self.resident.device,
                    "ready_at": self.resident.ready_at,
                    "model_id": self.resident.model_id,
                    "backend": getattr(self.resident, "backend", "diffusers"),
                },
            )
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            if self.path.rstrip("/") == "/generate":
                _json_response(self, 200, self._run_generate_queued(payload))
                return
            if self.path.rstrip("/") == "/warmup":
                _json_response(self, 200, self.resident.warmup())
                return
            _json_response(self, 404, {"ok": False, "error": "not_found"})
        except Exception as e:
            LOG.exception("request failed")
            _json_response(self, 500, {"ok": False, "error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("SMARTBLOG_HUNYUAN_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_PORT", "8798")))
    parser.add_argument("--model-id", default=os.getenv("SMARTBLOG_HUNYUAN_MODEL_ID", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled"))
    parser.add_argument("--device", default=os.getenv("SMARTBLOG_HUNYUAN_SERVICE_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("SMARTBLOG_HUNYUAN_DTYPE", "bf16"))
    parser.add_argument("--warmup", action="store_true", default=_env_flag("SMARTBLOG_HUNYUAN_SERVICE_WARMUP", "1"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(os.getenv("SMARTBLOG_HUNYUAN_SERVICE_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    resident = _create_resident(model_id=str(args.model_id), device=str(args.device), dtype=str(args.dtype))
    if bool(args.warmup):
        LOG.warning("starting Hunyuan warmup")
        resident.warmup()
        LOG.warning("Hunyuan warmup complete")
    server = ThreadingHTTPServer((str(args.host), int(args.port)), HunyuanHandler)
    setattr(server, "resident", resident)
    LOG.warning("Hunyuan service listening on %s:%d pid=%d", str(args.host), int(args.port), os.getpid())
    server.serve_forever()


if __name__ == "__main__":
    main()
