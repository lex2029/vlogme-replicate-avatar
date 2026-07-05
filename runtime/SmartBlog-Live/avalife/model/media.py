from __future__ import annotations

from contextlib import nullcontext
import logging
import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F

from avalife.model.protocol import MediaProcessRequest, MediaProcessResponse
from avalife.core.observability import media_timing_enabled
from avalife.core.ffmpeg import (
    RGB24VideoPipeReader,
    cuda_decoder_for_codec,
    probe_video_metadata,
)


class MediaProcessCancelled(RuntimeError):
    pass


_VIDEO_ENCODER_AVAILABILITY: dict[str, bool] = {}


def _media_process_device_name() -> str:
    raw = str(os.getenv("SMARTBLOG_MEDIA_PROCESS_DEVICE", "cuda:0") or "cuda:0").strip()
    return raw or "cuda:0"


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SmartBlog offline media processing.")
    return torch.device(_media_process_device_name())


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _output_size(req: MediaProcessRequest, *, src_h: int, src_w: int) -> tuple[int, int]:
    out_w = int(req.output_width or 0)
    out_h = int(req.output_height or 0)
    if out_w > 0 and out_h > 0:
        return out_h, out_w
    if bool(req.upscale):
        return int(src_h * 2), int(src_w * 2)
    return int(src_h), int(src_w)


def _resolve_output_fps(req: MediaProcessRequest, *, input_fps: float) -> float:
    raw = float(req.output_fps or 0.0)
    if raw > 0.0:
        return max(1.0, float(raw))
    if float(input_fps) > 0.0:
        return float(input_fps)
    return 25.0


def _validate_preserve_audio_fps(
    *,
    preserve_audio: bool,
    input_fps: float,
    requested_output_fps: float,
    resolved_output_fps: float,
) -> None:
    if not bool(preserve_audio):
        return
    if float(requested_output_fps) <= 0.0:
        return
    if float(input_fps) <= 0.0 or float(resolved_output_fps) <= 0.0:
        raise RuntimeError(
            "preserve_audio requires source fps for output_fps resampling "
            f"(source_fps={float(input_fps):.6f} requested_fps={float(requested_output_fps):.6f})"
        )


def _motion_duration_sec(*, frames: int, fps: float, fallback_duration_sec: float) -> float:
    if int(frames) > 0 and float(fps) > 0.0:
        return float(frames) / float(fps)
    return max(0.0, float(fallback_duration_sec))


def _media_restore_native_size(req: MediaProcessRequest) -> bool:
    if not bool(req.upscale):
        return False
    return _env_flag("SMARTBLOG_MEDIA_RESTORE_NATIVE_SIZE", "0")


def _media_face_restore_overlay_enabled() -> bool:
    # Match the PostVAE/avatar face path by default. The direct enchenh2d
    # full-frame face paste can leave a visible square edge on low-res clips.
    return _env_flag(
        "SMARTBLOG_MEDIA_FACE_RESTORE_OVERLAY",
        str(os.getenv("SMARTBLOG_MEDIA_FACE_RESTORE_OVERLAY_ONLY", "1") or "1"),
    )


def _media_work_size(
    req: MediaProcessRequest,
    *,
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
    face_overlay_mode: str = "",
) -> tuple[int, int]:
    if face_overlay_mode or _media_restore_native_size(req):
        return int(src_h), int(src_w)
    return _aspect_fit_size(src_h=int(src_h), src_w=int(src_w), target_h=int(out_h), target_w=int(out_w))


def _timing_add(timing: dict[str, float] | None, key: str, dt: float) -> None:
    if timing is None:
        return
    timing[key] = float(timing.get(key, 0.0)) + float(dt)


def _select_enhancer_mode(req: MediaProcessRequest) -> str:
    face_enabled = _clamp01(req.face_restore) > 0.0
    bg_enabled = _clamp01(req.background_restore) > 0.0
    face_overlay_enabled = _media_face_restore_overlay_enabled()
    if bool(req.upscale):
        if _media_restore_native_size(req):
            if bg_enabled:
                return "enhance"
            if face_enabled:
                if face_overlay_enabled:
                    return ""
                return "face"
            return ""
        if face_enabled and not bg_enabled and face_overlay_enabled:
            return ""
        if face_enabled and bg_enabled:
            return "upscale_face_enhance"
        if bg_enabled:
            return "upscale"
        if face_enabled:
            return "face"
        return ""
    if face_enabled and bg_enabled:
        return "enhance"
    if bg_enabled:
        return "enhance"
    if face_enabled:
        if face_overlay_enabled:
            return ""
        return "face"
    return ""


def _restore_face_overlay_mode(req: MediaProcessRequest) -> str:
    face_enabled = _clamp01(req.face_restore) > 0.0
    bg_enabled = _clamp01(req.background_restore) > 0.0
    face_overlay_enabled = _media_face_restore_overlay_enabled()
    if bool(req.upscale) and not _media_restore_native_size(req):
        if face_enabled and not bg_enabled and face_overlay_enabled:
            return "restored"
        return ""
    if not bg_enabled and not face_overlay_enabled:
        return ""
    if face_enabled:
        return "restored"
    return "original"


def _mode_uses_realesrgan(mode: str) -> bool:
    return str(mode or "") in {
        "enhance",
        "upscale",
        "face_upscale",
        "upscale_face",
        "upscale_face_enhance",
        "double_upscale_face",
    }


def _mode_supports_batch(mode: str) -> bool:
    return str(mode or "") in {
        "enhance",
        "upscale",
    }


def _media_batch_frames(req: MediaProcessRequest, *, mode: str) -> int:
    if _restore_face_overlay_mode(req):
        return 1
    if not _mode_supports_batch(mode):
        return 1
    try:
        raw = int(str(os.getenv("SMARTBLOG_MEDIA_BATCH_FRAMES", "1") or "1").strip())
    except Exception:
        raw = 1
    return max(1, min(64, int(raw)))


def _media_face_cache_frames(*, face_overlay_mode: str = "") -> int:
    if str(face_overlay_mode or "").strip():
        # Hunyuan/media face overlays are applied frame-by-frame. Reusing a
        # miss from the first frame of a generated clip can make face restore
        # suddenly appear about a second later, which looks like a square patch.
        raw = str(os.getenv("SMARTBLOG_MEDIA_FACE_OVERLAY_CACHE_FRAMES", "1") or "1").strip()
    else:
        raw = str(os.getenv("SMARTBLOG_ENHANCE_FACE_CACHE_FRAMES", "24") or "24").strip()
    try:
        value = int(raw)
    except Exception:
        value = 1 if str(face_overlay_mode or "").strip() else 24
    return max(1, int(value))


def _enhancer_models_dir() -> str:
    raw = str(os.getenv("LIVE_RAW_POST_VAE_ENHANCER_MODELS_DIR", "") or "").strip()
    if raw:
        return raw
    repo_root = Path(__file__).resolve().parents[2]
    asset_root = Path(str(os.getenv("WORKER_ASSET_ROOT", "") or "").strip())
    candidates: list[Path] = []
    if str(asset_root):
        candidates.append(asset_root / "worker_assets" / "enchenh2d" / "models")
    candidates.extend(
        (
            repo_root / "worker_assets" / "enchenh2d" / "models",
            repo_root / "avalife" / "worker_assets" / "enchenh2d" / "models",
        )
    )
    for path in candidates:
        if (path / "RealESRGAN_x2plus.pth").exists() and (path / "GFPGANv1.4.pth").exists():
            return str(path)
    return str(candidates[0])


def _enhancer_trt_enabled() -> bool:
    raw = str(os.getenv("LIVE_RAW_POST_VAE_ENHANCER_TRT", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _enhancer_trt_max_dim() -> int:
    try:
        raw = int(str(os.getenv("SMARTBLOG_MEDIA_TRT_MAX_DIM", "832") or "832").strip())
    except Exception:
        raw = 832
    return max(64, int(raw))


def _empty_cache_after_media_error() -> bool:
    raw = str(os.getenv("SMARTBLOG_MEDIA_EMPTY_CACHE_ON_ERROR", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _video_encoder_available(encoder: str) -> bool:
    encoder_s = str(encoder or "").strip().lower()
    if encoder_s in {"libx264", "x264"}:
        return True
    if not encoder_s:
        return False
    if encoder_s in _VIDEO_ENCODER_AVAILABILITY:
        return bool(_VIDEO_ENCODER_AVAILABILITY[encoder_s])
    if encoder_s != "h264_nvenc":
        _VIDEO_ENCODER_AVAILABILITY[encoder_s] = False
        return False
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=1:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
        ok = int(proc.returncode or 0) == 0
        if not ok:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            logging.warning("Media h264_nvenc unavailable, falling back to libx264: %s", err or f"rc={proc.returncode}")
    except Exception as e:
        ok = False
        logging.warning("Media h264_nvenc validation failed, falling back to libx264: %s", e)
    _VIDEO_ENCODER_AVAILABILITY[encoder_s] = bool(ok)
    return bool(ok)


def _select_video_encoder() -> str:
    raw = str(
        os.getenv("SMARTBLOG_MEDIA_VIDEO_ENCODER", os.getenv("MEDIA_VIDEO_ENCODER", "auto")) or "auto"
    ).strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "nvenc": "h264_nvenc",
        "h264": "h264_nvenc",
        "h264_nvenc": "h264_nvenc",
        "cpu": "libx264",
        "x264": "libx264",
        "libx264": "libx264",
    }
    encoder = aliases.get(raw, raw)
    if encoder == "auto":
        return "h264_nvenc" if _video_encoder_available("h264_nvenc") else "libx264"
    if encoder == "h264_nvenc" and not _video_encoder_available("h264_nvenc"):
        strict = str(os.getenv("SMARTBLOG_MEDIA_VIDEO_ENCODER_STRICT", "0") or "0").strip().lower()
        if strict in {"1", "true", "yes", "on"}:
            return "h264_nvenc"
        return "libx264"
    if encoder not in {"h264_nvenc", "libx264"}:
        logging.warning("Unsupported media video encoder %r; using auto fallback", raw)
        return "h264_nvenc" if _video_encoder_available("h264_nvenc") else "libx264"
    return encoder


_ENHANCER_CACHE: dict[tuple[str, str, int], object] = {}


def _gpu_guard(gpu_lock):
    return gpu_lock if gpu_lock is not None else nullcontext()


def _check_cancelled(cancel_event) -> None:
    if cancel_event is not None and bool(cancel_event.is_set()):
        raise MediaProcessCancelled("media_process cancelled")


def _aspect_fit_size(*, src_h: int, src_w: int, target_h: int, target_w: int) -> tuple[int, int]:
    if int(src_h) <= 0 or int(src_w) <= 0:
        raise RuntimeError(f"invalid source size for aspect fit: {src_w}x{src_h}")
    if int(target_h) <= 0 or int(target_w) <= 0:
        raise RuntimeError(f"invalid target size for aspect fit: {target_w}x{target_h}")
    scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
    fit_h = max(1, min(int(target_h), int(round(float(src_h) * scale))))
    fit_w = max(1, min(int(target_w), int(round(float(src_w) * scale))))
    return fit_h, fit_w


def _aspect_fill_size(*, src_h: int, src_w: int, target_h: int, target_w: int) -> tuple[int, int]:
    if int(src_h) <= 0 or int(src_w) <= 0:
        raise RuntimeError(f"invalid source size for aspect fill: {src_w}x{src_h}")
    if int(target_h) <= 0 or int(target_w) <= 0:
        raise RuntimeError(f"invalid target size for aspect fill: {target_w}x{target_h}")
    scale = max(float(target_h) / float(src_h), float(target_w) / float(src_w))
    fill_h = max(int(target_h), int(round(float(src_h) * scale)))
    fill_w = max(int(target_w), int(round(float(src_w) * scale)))
    return fill_h, fill_w


def _blur_kernel_size(*, h: int, w: int) -> int:
    dim = int(min(h, w))
    if dim <= 2:
        return 0
    kernel = max(3, int(round(float(dim) * 0.08)))
    if kernel % 2 == 0:
        kernel += 1
    max_kernel_h = int(h) if int(h) % 2 == 1 else int(h) - 1
    max_kernel_w = int(w) if int(w) % 2 == 1 else int(w) - 1
    kernel = min(kernel, max_kernel_h, max_kernel_w)
    return kernel if kernel >= 3 else 0


def _compose_on_blurred_canvas(frames_01: torch.Tensor, *, out_h: int, out_w: int) -> torch.Tensor:
    src_h = int(frames_01.shape[2])
    src_w = int(frames_01.shape[3])
    if int(src_h) == int(out_h) and int(src_w) == int(out_w):
        return frames_01.clamp(0, 1)

    fit_h, fit_w = _aspect_fit_size(src_h=src_h, src_w=src_w, target_h=out_h, target_w=out_w)
    foreground = F.interpolate(frames_01, size=(fit_h, fit_w), mode="bilinear", align_corners=False)
    if int(fit_h) == int(out_h) and int(fit_w) == int(out_w):
        return foreground.clamp(0, 1)

    fill_h, fill_w = _aspect_fill_size(src_h=src_h, src_w=src_w, target_h=out_h, target_w=out_w)
    background = F.interpolate(frames_01, size=(fill_h, fill_w), mode="bilinear", align_corners=False)
    y0 = max(0, int((fill_h - int(out_h)) // 2))
    x0 = max(0, int((fill_w - int(out_w)) // 2))
    background = background[:, :, y0 : y0 + int(out_h), x0 : x0 + int(out_w)].contiguous()
    blur_k = _blur_kernel_size(h=int(out_h), w=int(out_w))
    if blur_k >= 3:
        sigma = max(1.0, float(blur_k) / 6.0)
        background = kornia.filters.gaussian_blur2d(
            background,
            (int(blur_k), int(blur_k)),
            (float(sigma), float(sigma)),
        ).contiguous()
    canvas = background.clone()
    top = max(0, int((int(out_h) - int(fit_h)) // 2))
    left = max(0, int((int(out_w) - int(fit_w)) // 2))
    canvas[:, :, top : top + int(fit_h), left : left + int(fit_w)] = foreground
    return canvas.clamp(0, 1)


def _should_use_trt(mode: str, *, src_h: int, src_w: int) -> bool:
    if not _enhancer_trt_enabled():
        return False
    if not _mode_uses_realesrgan(mode):
        return False
    max_dim = max(int(src_h), int(src_w))
    return max_dim <= int(_enhancer_trt_max_dim())


def _get_enhancer(*, mode: str, device: torch.device, use_trt: bool | None = None):
    trt_enabled = _enhancer_trt_enabled() if use_trt is None else bool(use_trt)
    key = (str(mode or ""), str(device), 1 if trt_enabled else 0)
    cached = _ENHANCER_CACHE.get(key)
    if cached is not None:
        return cached
    if not mode:
        return None
    from liveavatar.vendor.enchenh2d import Enhancer

    enhancer = Enhancer(
        mode=str(mode),
        device=str(device),
        trt=bool(trt_enabled),
        models_dir=_enhancer_models_dir(),
    )
    _ENHANCER_CACHE[key] = enhancer
    return enhancer


def _get_face_affine_pairs(face_enhancer: object, tensor_01: torch.Tensor, *, clip_id: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from liveavatar.vendor.enchenh2d.enhancer import _detect_faces_gpu, _get_affine_matrices

    cached = getattr(face_enhancer, "_cached_affine", None)
    if isinstance(cached, dict) and cached.get("clip_id") == int(clip_id):
        pairs = cached.get("pairs")
        if pairs is not None:
            return list(pairs)

    landmarks = _detect_faces_gpu(
        getattr(face_enhancer, "_face_det"),
        tensor_01,
        priors_cache=getattr(face_enhancer, "_priors_cache", None),
    )
    if not landmarks:
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    else:
        pairs = list(
            _get_affine_matrices(
                landmarks,
                getattr(face_enhancer, "_face_template"),
                getattr(face_enhancer, "_face_size"),
                getattr(face_enhancer, "device"),
            )
        )
    if isinstance(cached, dict):
        cached["clip_id"] = int(clip_id)
        cached["pairs"] = pairs
    return pairs


def _erode_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = int(max(1, int(kernel_size)))
    if kernel <= 1:
        return mask
    if kernel % 2 == 0:
        kernel += 1
    return (-F.max_pool2d(-mask, kernel_size=int(kernel), stride=1, padding=int(kernel // 2))).clamp(0, 1)


def _soften_square_face_mask(mask: torch.Tensor) -> torch.Tensor:
    # Same idea as PostVAEEnhancer._soften_square_face_mask: use the full
    # canonical square face coverage, then erode/blur after inverse affine so
    # the square border does not become visible on low-res generated clips.
    mask = _erode_mask(mask.clamp(0, 1), 3)
    try:
        area = float(mask[:, :1].sum().detach().item()) / float(max(1, int(mask.shape[0])))
    except Exception:
        area = float(mask.shape[-1] * mask.shape[-2]) * 0.04
    w_edge = max(2, int((max(1.0, float(area)) ** 0.5) // 20))
    erosion_kernel = max(3, int(w_edge * 2))
    blur_kernel = max(7, int(w_edge * 2) + 1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    mask = _erode_mask(mask, int(erosion_kernel))
    sigma = max(1.0, float(blur_kernel) / 3.0)
    return kornia.filters.gaussian_blur2d(
        mask,
        (int(blur_kernel), int(blur_kernel)),
        (float(sigma), float(sigma)),
    ).clamp(0, 1).contiguous()


def _apply_restore_face_overlay(
    target_01: torch.Tensor,
    *,
    base_01: torch.Tensor,
    face_enhancer: object,
    overlay_mode: str,
    face_restore: float,
    clip_id: int,
) -> torch.Tensor:
    overlay_mode = str(overlay_mode or "").strip().lower()
    if overlay_mode not in {"original", "restored"}:
        return target_01
    affine_pairs = _get_face_affine_pairs(face_enhancer, base_01, clip_id=int(clip_id))
    if not affine_pairs:
        return target_01

    result = target_01
    face_size = getattr(face_enhancer, "_face_size")
    blend = _clamp01(face_restore)
    for M, M_inv in affine_pairs:
        try:
            original_crop = kornia.geometry.transform.warp_affine(
                base_01,
                M,
                (int(face_size[1]), int(face_size[0])),
                mode="bilinear",
                padding_mode="zeros",
            )
            if overlay_mode == "restored":
                restored = getattr(face_enhancer, "_gfpgan")(
                    original_crop * 2 - 1,
                    return_rgb=False,
                    randomize_noise=False,
                )[0]
                face_patch = (restored.clamp(-1, 1) + 1.0) / 2.0
                if torch.isnan(face_patch).any() or torch.isinf(face_patch).any():
                    continue
                if blend < 1.0:
                    face_patch = (face_patch * blend) + (original_crop * (1.0 - blend))
            else:
                face_patch = original_crop

            h, w = int(base_01.shape[2]), int(base_01.shape[3])
            pasted = kornia.geometry.transform.warp_affine(
                face_patch,
                M_inv,
                (h, w),
                mode="bilinear",
                padding_mode="zeros",
            )
            inner = torch.ones_like(face_patch)
            mask = kornia.geometry.transform.warp_affine(
                inner,
                M_inv,
                (h, w),
                mode="bilinear",
                padding_mode="zeros",
            )
            mask = _soften_square_face_mask(mask)
            result = (result * (1.0 - mask)) + (pasted * mask)
        except Exception:
            continue
    return result.clamp(0, 1)


def _prepare_media_enhancers(
    req: MediaProcessRequest,
    *,
    device: torch.device,
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
) -> tuple[str, object | None, str, object | None]:
    mode = _select_enhancer_mode(req)
    use_trt = _should_use_trt(mode, src_h=src_h, src_w=src_w)
    enhancer = _get_enhancer(mode=mode, device=device, use_trt=use_trt)
    face_overlay_mode = _restore_face_overlay_mode(req)
    face_overlay_enhancer = None
    if face_overlay_mode:
        face_overlay_enhancer = _get_enhancer(mode="face", device=device, use_trt=False)
    work_h, work_w = _media_work_size(
        req,
        src_h=int(src_h),
        src_w=int(src_w),
        out_h=int(out_h),
        out_w=int(out_w),
        face_overlay_mode=str(face_overlay_mode),
    )

    for inst in (enhancer, face_overlay_enhancer):
        if inst is None:
            continue
        try:
            inst.preallocate(src_h, src_w, out_h=work_h, out_w=work_w)
        except Exception:
            pass
    return mode, enhancer, face_overlay_mode, face_overlay_enhancer


def _bgr_frame_to_tensor(bgr: np.ndarray, *, device: torch.device) -> torch.Tensor:
    arr = np.ascontiguousarray(bgr)
    tensor = torch.from_numpy(arr).to(device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = tensor[:, [2, 1, 0], :, :].to(dtype=torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


def _bgr_frames_to_tensor(frames_bgr: list[np.ndarray], *, device: torch.device) -> torch.Tensor:
    if not frames_bgr:
        raise RuntimeError("frames_bgr batch is empty")
    arr = np.ascontiguousarray(np.stack(frames_bgr, axis=0))
    tensor = torch.from_numpy(arr).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    tensor = tensor[:, [2, 1, 0], :, :].to(dtype=torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


def _tensor01_to_bgr_u8(frame_01: torch.Tensor) -> np.ndarray:
    bgr_u8 = (
        frame_01[0]
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)[[2, 1, 0], :, :]
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(bgr_u8)


def _tensor01_batch_to_bgr_u8_list(frames_01: torch.Tensor) -> list[np.ndarray]:
    bgr_u8 = (
        frames_01
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)[:, [2, 1, 0], :, :]
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    out: list[np.ndarray] = []
    for idx in range(int(bgr_u8.shape[0])):
        out.append(np.ascontiguousarray(bgr_u8[idx]))
    return out


def _process_frame_tensor(
    tensor_nchw: torch.Tensor,
    *,
    enhancer: object | None,
    mode: str,
    out_h: int,
    out_w: int,
    face_restore: float,
    background_restore: float,
    clip_id: int,
    frame_idx: int,
    face_overlay_mode: str = "",
    face_overlay_enhancer: object | None = None,
    restore_native_size: bool = False,
    timing: dict[str, float] | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    base_01 = (tensor_nchw.clamp(-1, 1) + 1.0) / 2.0
    _timing_add(timing, "base", time.perf_counter() - t0)
    if face_overlay_mode or bool(restore_native_size):
        work_h = int(base_01.shape[2])
        work_w = int(base_01.shape[3])
    else:
        work_h, work_w = _aspect_fit_size(
            src_h=int(base_01.shape[2]),
            src_w=int(base_01.shape[3]),
            target_h=int(out_h),
            target_w=int(out_w),
        )
    if (enhancer is None or not mode) and not face_overlay_mode:
        t0 = time.perf_counter()
        out = _compose_on_blurred_canvas(base_01, out_h=int(out_h), out_w=int(out_w))
        _timing_add(timing, "compose", time.perf_counter() - t0)
        return out
    if enhancer is None or not mode:
        enhanced_01 = base_01
    else:
        t0 = time.perf_counter()
        try:
            setattr(enhancer, "blend", _clamp01(face_restore))
        except Exception:
            pass
        _timing_add(timing, "setup", time.perf_counter() - t0)

        t0 = time.perf_counter()
        enhanced_01 = enhancer.enhance_gpu(
            tensor_nchw,
            out_h=int(work_h),
            out_w=int(work_w),
            clip_id=int(clip_id),
            frame_idx=int(frame_idx),
        ).clamp(0, 1)
        _timing_add(timing, "enhance", time.perf_counter() - t0)

        t0 = time.perf_counter()
        bg_strength = _clamp01(background_restore)
        if (bg_strength < 1.0) and _mode_uses_realesrgan(mode):
            base_resized = F.interpolate(base_01, size=(work_h, work_w), mode="bilinear", align_corners=False)
            enhanced_01 = (enhanced_01 * bg_strength) + (base_resized * (1.0 - bg_strength))
        _timing_add(timing, "bg_blend", time.perf_counter() - t0)

    if face_overlay_mode and face_overlay_enhancer is not None:
        t0 = time.perf_counter()
        enhanced_01 = _apply_restore_face_overlay(
            enhanced_01.clamp(0, 1),
            base_01=base_01,
            face_enhancer=face_overlay_enhancer,
            overlay_mode=face_overlay_mode,
            face_restore=float(face_restore),
            clip_id=int(clip_id),
        )
        _timing_add(timing, "face_overlay", time.perf_counter() - t0)
    t0 = time.perf_counter()
    out = _compose_on_blurred_canvas(enhanced_01, out_h=int(out_h), out_w=int(out_w))
    _timing_add(timing, "compose", time.perf_counter() - t0)
    return out


def _process_frame_tensor_batch(
    tensor_nchw: torch.Tensor,
    *,
    enhancer: object | None,
    mode: str,
    out_h: int,
    out_w: int,
    background_restore: float,
    restore_native_size: bool = False,
    timing: dict[str, float] | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    base_01 = (tensor_nchw.clamp(-1, 1) + 1.0) / 2.0
    _timing_add(timing, "base", time.perf_counter() - t0)
    if bool(restore_native_size):
        work_h, work_w = int(base_01.shape[2]), int(base_01.shape[3])
    else:
        work_h, work_w = _aspect_fit_size(
            src_h=int(base_01.shape[2]),
            src_w=int(base_01.shape[3]),
            target_h=int(out_h),
            target_w=int(out_w),
        )
    if enhancer is None or not mode:
        t0 = time.perf_counter()
        out = _compose_on_blurred_canvas(base_01, out_h=int(out_h), out_w=int(out_w))
        _timing_add(timing, "compose", time.perf_counter() - t0)
        return out
    if not _mode_supports_batch(mode):
        raise RuntimeError(f"batch processing is not supported for enhancer mode={mode}")
    t0 = time.perf_counter()
    enhanced_01 = enhancer.enhance_gpu(
        tensor_nchw,
        out_h=int(work_h),
        out_w=int(work_w),
    ).clamp(0, 1)
    _timing_add(timing, "enhance", time.perf_counter() - t0)
    t0 = time.perf_counter()
    bg_strength = _clamp01(background_restore)
    if (bg_strength < 1.0) and _mode_uses_realesrgan(mode):
        base_resized = F.interpolate(base_01, size=(work_h, work_w), mode="bilinear", align_corners=False)
        enhanced_01 = (enhanced_01 * bg_strength) + (base_resized * (1.0 - bg_strength))
    _timing_add(timing, "bg_blend", time.perf_counter() - t0)
    t0 = time.perf_counter()
    out = _compose_on_blurred_canvas(enhanced_01, out_h=int(out_h), out_w=int(out_w))
    _timing_add(timing, "compose", time.perf_counter() - t0)
    return out


def _jpeg_params(quality: int) -> list[int]:
    return [int(cv2.IMWRITE_JPEG_QUALITY), int(max(50, min(100, int(quality))))]


def _bgr_buffer_to_tensor(buf: bytearray, *, src_h: int, src_w: int, device: torch.device) -> torch.Tensor:
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    tensor = tensor.view(src_h, src_w, 3).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = tensor.to(device=device)
    tensor = tensor[:, [2, 1, 0], :, :].to(dtype=torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


def _bgr_buffers_to_tensor(
    buffers: list[bytearray],
    *,
    src_h: int,
    src_w: int,
    device: torch.device,
) -> torch.Tensor:
    if not buffers:
        raise RuntimeError("buffers batch is empty")
    frames = [torch.frombuffer(buf, dtype=torch.uint8).view(src_h, src_w, 3) for buf in buffers]
    tensor = torch.stack(frames, dim=0).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device=device)
    tensor = tensor[:, [2, 1, 0], :, :].to(dtype=torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


def _process_image_sync(
    req: MediaProcessRequest,
    *,
    device: torch.device,
    gpu_lock=None,
    cancel_event=None,
) -> MediaProcessResponse:
    total_t0 = time.perf_counter()
    _check_cancelled(cancel_event)
    src_path = os.path.abspath(str(req.source_path or "").strip())
    out_path = os.path.abspath(str(req.output_path or "").strip())
    if not src_path or not os.path.exists(src_path):
        raise FileNotFoundError(f"source image missing: {src_path}")
    if not out_path:
        raise RuntimeError("output_path is required for media_process image")

    img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {src_path}")
    src_h, src_w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    out_h, out_w = _output_size(req, src_h=src_h, src_w=src_w)
    _check_cancelled(cancel_event)
    gpu_dt = 0.0
    with _gpu_guard(gpu_lock):
        gpu_t0 = time.perf_counter()
        mode, enhancer, face_overlay_mode, face_overlay_enhancer = _prepare_media_enhancers(
            req,
            device=device,
            src_h=src_h,
            src_w=src_w,
            out_h=out_h,
            out_w=out_w,
        )

        with torch.inference_mode():
            _check_cancelled(cancel_event)
            frame_tensor = _bgr_frame_to_tensor(img_bgr, device=device)
            out_01 = _process_frame_tensor(
                frame_tensor,
                enhancer=enhancer,
                mode=mode,
                out_h=out_h,
                out_w=out_w,
                face_restore=float(req.face_restore),
                background_restore=float(req.background_restore),
                clip_id=0,
                frame_idx=0,
                face_overlay_mode=face_overlay_mode,
                face_overlay_enhancer=face_overlay_enhancer,
                restore_native_size=_media_restore_native_size(req),
            )
            _check_cancelled(cancel_event)
            out_bgr = _tensor01_to_bgr_u8(out_01)
        gpu_dt = float(time.perf_counter() - gpu_t0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    suffix = str(Path(out_path).suffix or "").lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = bool(cv2.imwrite(out_path, out_bgr, _jpeg_params(int(req.jpeg_quality or 95))))
    else:
        ok = bool(cv2.imwrite(out_path, out_bgr))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out_path}")
    if media_timing_enabled():
        logging.info(
            "Media image timing: device=%s mode=%s src=%sx%s out=%sx%s gpu=%.3fs total=%.3fs face=%.2f bg=%.2f",
            str(device),
            str(mode or "-"),
            int(src_w),
            int(src_h),
            int(out_w),
            int(out_h),
            float(gpu_dt),
            float(time.perf_counter() - total_t0),
            float(req.face_restore),
            float(req.background_restore),
        )
    return MediaProcessResponse(ok=True, output_path=out_path, frames_written=1)


def _video_ffmpeg_cmd(
    *,
    out_path: str,
    src_path: str,
    out_w: int,
    out_h: int,
    out_fps: float,
    preserve_audio: bool,
    trim_duration_sec: float = 0.0,
    video_encoder: str = "h264_nvenc",
) -> list[str]:
    encoder = str(video_encoder or "h264_nvenc").strip().lower()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(out_w)}x{int(out_h)}",
        "-r",
        f"{float(out_fps):.6f}",
        "-i",
        "pipe:0",
    ]
    if preserve_audio:
        cmd.extend(["-i", str(src_path), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        cmd.extend(["-map", "0:v:0"])
    if float(trim_duration_sec or 0.0) > 0.0:
        cmd.extend(["-t", f"{float(trim_duration_sec):.6f}", "-shortest"])
    gop = str(max(1, int(round(float(out_fps) * 2.0))))
    if encoder == "libx264":
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                str(os.getenv("SMARTBLOG_MEDIA_X264_PRESET", "veryfast") or "veryfast"),
                "-crf",
                str(os.getenv("SMARTBLOG_MEDIA_X264_CRF", "23") or "23"),
                "-pix_fmt",
                "yuv420p",
                "-g",
                gop,
                "-keyint_min",
                gop,
                "-sc_threshold",
                "0",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-profile:v",
                "high",
                "-preset",
                "p4",
                "-pix_fmt",
                "yuv420p",
                "-g",
                gop,
                "-keyint_min",
                gop,
                "-sc_threshold",
                "0",
            ]
        )
    if preserve_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"])
    cmd.extend(["-movflags", "+faststart", str(out_path)])
    return cmd


def _process_video_sync(
    req: MediaProcessRequest,
    *,
    device: torch.device,
    gpu_lock=None,
    cancel_event=None,
) -> MediaProcessResponse:
    total_t0 = time.perf_counter()
    _check_cancelled(cancel_event)
    src_path = os.path.abspath(str(req.source_path or "").strip())
    out_path = os.path.abspath(str(req.output_path or "").strip())
    if not src_path or not os.path.exists(src_path):
        raise FileNotFoundError(f"source video missing: {src_path}")
    if not out_path:
        raise RuntimeError("output_path is required for media_process video")

    phase_s: dict[str, float] = {}
    probe_t0 = time.perf_counter()
    probe = probe_video_metadata(src_path)
    src_w = int(probe.width)
    src_h = int(probe.height)
    input_fps = float(probe.fps)
    total_frames = int(probe.frames)
    codec_name = str(probe.codec_name)
    _timing_add(phase_s, "probe", time.perf_counter() - probe_t0)

    out_h, out_w = _output_size(req, src_h=src_h, src_w=src_w)
    out_fps = float(_resolve_output_fps(req, input_fps=float(input_fps)))
    _validate_preserve_audio_fps(
        preserve_audio=bool(req.preserve_audio),
        input_fps=float(input_fps),
        requested_output_fps=float(req.output_fps or 0.0),
        resolved_output_fps=float(out_fps),
    )

    mode = _select_enhancer_mode(req)
    restore_native_size = _media_restore_native_size(req)
    batch_frames = _media_batch_frames(req, mode=mode)
    face_cache_frames = _media_face_cache_frames(face_overlay_mode=str(_restore_face_overlay_mode(req)))
    frame_bytes = int(src_w * src_h * 3)
    prep_t0 = time.perf_counter()
    with _gpu_guard(gpu_lock):
        mode, enhancer, face_overlay_mode, face_overlay_enhancer = _prepare_media_enhancers(
            req,
            device=device,
            src_h=src_h,
            src_w=src_w,
            out_h=out_h,
            out_w=out_w,
        )
    _timing_add(phase_s, "prepare_enhancer", time.perf_counter() - prep_t0)
    if media_timing_enabled():
        work_h, work_w = _media_work_size(
            req,
            src_h=int(src_h),
            src_w=int(src_w),
            out_h=int(out_h),
            out_w=int(out_w),
            face_overlay_mode=str(face_overlay_mode),
        )
        logging.info(
            "Media video config: device=%s mode=%s native_restore=%d face_overlay=%s src=%sx%s work=%sx%s out=%sx%s in_fps=%.3f out_fps=%.3f frames=%d batch=%d face=%.2f bg=%.2f",
            str(device),
            str(mode or "-"),
            1 if bool(restore_native_size) else 0,
            str(face_overlay_mode or "-"),
            int(src_w),
            int(src_h),
            int(work_w),
            int(work_h),
            int(out_w),
            int(out_h),
            float(input_fps),
            float(out_fps),
            int(total_frames),
            int(batch_frames),
            float(req.face_restore),
            float(req.background_restore),
        )
        if str(face_overlay_mode or "").strip():
            logging.info(
                "Media face overlay config: cache_frames=%d overlay=%s",
                int(face_cache_frames),
                str(face_overlay_mode or "-"),
            )

    _ = cuda_decoder_for_codec(codec_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    encoder_t0 = time.perf_counter()
    video_encoder = _select_video_encoder()
    logging.info("Media video encoder selected: %s", str(video_encoder))
    encode_cmd = _video_ffmpeg_cmd(
        out_path=out_path,
        src_path=src_path,
        out_w=out_w,
        out_h=out_h,
        out_fps=out_fps,
        preserve_audio=bool(req.preserve_audio),
        trim_duration_sec=float(req.trim_duration_sec or 0.0),
        video_encoder=str(video_encoder),
    )
    _timing_add(phase_s, "encoder_setup", time.perf_counter() - encoder_t0)
    decode_reader = RGB24VideoPipeReader(src_path, out_w=src_w, out_h=src_h, device=device, pix_fmt="bgr24")
    proc = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    frame_idx = 0
    source_frame_idx = 0
    raw_buffers = [bytearray(frame_bytes) for _ in range(int(batch_frames))]
    cancelled = False
    decode_s = 0.0
    gpu_s = 0.0
    encode_s = 0.0
    wait_s = 0.0

    fps_delta = abs(float(out_fps) - float(input_fps))
    fps_tol = max(0.01, float(input_fps) * 0.01) if float(input_fps) > 0.0 else 0.01
    resample_output_fps = bool(float(input_fps) > 0.0 and float(out_fps) > 0.0 and fps_delta > fps_tol)
    trim_duration_sec = float(max(0.0, float(req.trim_duration_sec or 0.0)))
    max_source_frames = int(total_frames)
    if trim_duration_sec > 0.0 and float(input_fps) > 0.0:
        max_source_frames = int(
            min(
                int(total_frames),
                max(1, int(math.ceil(trim_duration_sec * float(input_fps)))),
            )
        )

    def _write_encoded_frame(out_bgr) -> None:
        nonlocal frame_idx, encode_s
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        encode_t0 = time.perf_counter()
        proc.stdin.write(memoryview(out_bgr))
        encode_s += float(time.perf_counter() - encode_t0)
        frame_idx += 1

    def _write_resampled_frame(out_bgr, *, src_frame_idx: int) -> None:
        if not resample_output_fps:
            _write_encoded_frame(out_bgr)
            return
        source_end_sec = float(int(src_frame_idx) + 1) / float(input_fps)
        while (float(frame_idx) / float(out_fps)) < source_end_sec - 1e-9:
            _write_encoded_frame(out_bgr)

    try:
        with torch.inference_mode():
            while True:
                _check_cancelled(cancel_event)
                remaining_source_frames = int(max_source_frames) - int(source_frame_idx)
                if remaining_source_frames <= 0:
                    break
                frame_buffers: list[bytearray] = []
                decode_t0 = time.perf_counter()
                for idx in range(min(int(batch_frames), int(remaining_source_frames))):
                    frame_view = decode_reader.read_frame_view()
                    if frame_view is None:
                        break
                    raw_buffers[idx][:] = frame_view
                    frame_buffers.append(raw_buffers[idx])
                decode_s += float(time.perf_counter() - decode_t0)
                if not frame_buffers:
                    break
                if int(batch_frames) > 1 and _mode_supports_batch(mode):
                    with _gpu_guard(gpu_lock):
                        gpu_t0 = time.perf_counter()
                        _check_cancelled(cancel_event)
                        upload_t0 = time.perf_counter()
                        tensor_batch = _bgr_buffers_to_tensor(frame_buffers, src_h=src_h, src_w=src_w, device=device)
                        _timing_add(phase_s, "tensor_upload", time.perf_counter() - upload_t0)
                        out_batch_01 = _process_frame_tensor_batch(
                            tensor_batch,
                            enhancer=enhancer,
                            mode=mode,
                            out_h=out_h,
                            out_w=out_w,
                            background_restore=float(req.background_restore),
                            restore_native_size=restore_native_size,
                            timing=phase_s,
                        )
                        _check_cancelled(cancel_event)
                        cpu_t0 = time.perf_counter()
                        out_frames_bgr = _tensor01_batch_to_bgr_u8_list(out_batch_01)
                        _timing_add(phase_s, "to_cpu", time.perf_counter() - cpu_t0)
                        gpu_s += float(time.perf_counter() - gpu_t0)
                    for out_bgr in out_frames_bgr:
                        _check_cancelled(cancel_event)
                        _write_resampled_frame(out_bgr, src_frame_idx=int(source_frame_idx))
                        source_frame_idx += 1
                    continue

                for frame_buf in frame_buffers:
                    with _gpu_guard(gpu_lock):
                        gpu_t0 = time.perf_counter()
                        _check_cancelled(cancel_event)
                        upload_t0 = time.perf_counter()
                        tensor = _bgr_buffer_to_tensor(frame_buf, src_h=src_h, src_w=src_w, device=device)
                        _timing_add(phase_s, "tensor_upload", time.perf_counter() - upload_t0)
                        out_01 = _process_frame_tensor(
                            tensor,
                            enhancer=enhancer,
                            mode=mode,
                            out_h=out_h,
                            out_w=out_w,
                            face_restore=float(req.face_restore),
                            background_restore=float(req.background_restore),
                            clip_id=int(source_frame_idx // face_cache_frames),
                            frame_idx=int(source_frame_idx),
                            face_overlay_mode=face_overlay_mode,
                            face_overlay_enhancer=face_overlay_enhancer,
                            restore_native_size=restore_native_size,
                            timing=phase_s,
                        )
                        _check_cancelled(cancel_event)
                        cpu_t0 = time.perf_counter()
                        out_bgr = _tensor01_to_bgr_u8(out_01)
                        _timing_add(phase_s, "to_cpu", time.perf_counter() - cpu_t0)
                        gpu_s += float(time.perf_counter() - gpu_t0)
                    _check_cancelled(cancel_event)
                    _write_resampled_frame(out_bgr, src_frame_idx=int(source_frame_idx))
                    source_frame_idx += 1
    except MediaProcessCancelled:
        cancelled = True
        raise
    finally:
        try:
            decode_reader.close()
        except Exception:
            pass
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if cancelled:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
    stderr_data = b""
    try:
        wait_t0 = time.perf_counter()
        stderr_data = (proc.stderr.read() if proc.stderr is not None else b"")
        proc.wait(timeout=600)
        wait_s += float(time.perf_counter() - wait_t0)
    except subprocess.TimeoutExpired:
        wait_s += float(time.perf_counter() - wait_t0)
        proc.kill()
        stderr_data = (proc.stderr.read() if proc.stderr is not None else b"")
        proc.wait()
        raise RuntimeError("ffmpeg encode timeout")
    if int(proc.returncode or 0) != 0:
        err_text = (stderr_data or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg encode failed rc={proc.returncode}: {err_text or 'no stderr'}")
    if frame_idx <= 0 and total_frames > 0:
        raise RuntimeError(f"no frames were written for video: {src_path}")
    src_motion_sec = _motion_duration_sec(
        frames=int(source_frame_idx),
        fps=float(input_fps),
        fallback_duration_sec=(
            min(float(getattr(probe, "duration_sec", 0.0) or 0.0), float(trim_duration_sec))
            if trim_duration_sec > 0.0
            else float(getattr(probe, "duration_sec", 0.0) or 0.0)
        ),
    )
    out_motion_sec = _motion_duration_sec(
        frames=int(frame_idx),
        fps=float(out_fps),
        fallback_duration_sec=0.0,
    )
    if bool(req.preserve_audio):
        drift_sec = abs(float(out_motion_sec) - float(src_motion_sec))
        drift_tol = max(0.20, 2.0 / float(max(1.0, out_fps)))
        if drift_sec > drift_tol:
            raise RuntimeError(
                "encoded motion duration drifted from preserved-audio source "
                f"(source_motion_sec={float(src_motion_sec):.3f} "
                f"output_motion_sec={float(out_motion_sec):.3f} "
                f"output_fps={float(out_fps):.6f})"
            )
    logging.info(
        "media_process video done: src=%s frames=%d src_fps=%.6f out_fps=%.6f preserve_audio=%d src_motion=%.3fs out_motion=%.3fs",
        os.path.basename(src_path),
        int(frame_idx),
        float(input_fps),
        float(out_fps),
        1 if bool(req.preserve_audio) else 0,
        float(src_motion_sec),
        float(out_motion_sec),
    )
    if media_timing_enabled():
        logging.info(
            "Media video timing: device=%s mode=%s native_restore=%d frames=%d source_frames=%d src=%sx%s out=%sx%s decode=%.3fs gpu=%.3fs encode=%.3fs wait=%.3fs total=%.3fs batch=%d face=%.2f bg=%.2f",
            str(device),
            str(mode or "-"),
            1 if bool(restore_native_size) else 0,
            int(frame_idx),
            int(source_frame_idx),
            int(src_w),
            int(src_h),
            int(out_w),
            int(out_h),
            float(decode_s),
            float(gpu_s),
            float(encode_s),
            float(wait_s),
            float(time.perf_counter() - total_t0),
            int(batch_frames),
            float(req.face_restore),
            float(req.background_restore),
        )
        logging.info(
            "Media video phase timing: probe=%.3fs prepare=%.3fs encoder_setup=%.3fs tensor_upload=%.3fs base=%.3fs setup=%.3fs enhance=%.3fs bg_blend=%.3fs face_overlay=%.3fs compose=%.3fs to_cpu=%.3fs decode=%.3fs encode=%.3fs wait=%.3fs",
            float(phase_s.get("probe", 0.0)),
            float(phase_s.get("prepare_enhancer", 0.0)),
            float(phase_s.get("encoder_setup", 0.0)),
            float(phase_s.get("tensor_upload", 0.0)),
            float(phase_s.get("base", 0.0)),
            float(phase_s.get("setup", 0.0)),
            float(phase_s.get("enhance", 0.0)),
            float(phase_s.get("bg_blend", 0.0)),
            float(phase_s.get("face_overlay", 0.0)),
            float(phase_s.get("compose", 0.0)),
            float(phase_s.get("to_cpu", 0.0)),
            float(decode_s),
            float(encode_s),
            float(wait_s),
        )
    return MediaProcessResponse(ok=True, output_path=out_path, frames_written=int(frame_idx))


def process_media_sync(req: MediaProcessRequest, *, gpu_lock=None, cancel_event=None) -> MediaProcessResponse:
    t0 = time.perf_counter()
    device = _require_cuda_device()
    had_error = False
    try:
        kind = str(req.source_kind or "").strip().lower()
        if kind == "image":
            resp = _process_image_sync(req, device=device, gpu_lock=gpu_lock, cancel_event=cancel_event)
        elif kind == "video":
            resp = _process_video_sync(req, device=device, gpu_lock=gpu_lock, cancel_event=cancel_event)
        else:
            raise RuntimeError(f"unsupported media_process source_kind={kind!r}")
        return MediaProcessResponse(
            ok=True,
            output_path=str(resp.output_path or ""),
            frames_written=int(resp.frames_written or 0),
            total_s=float(time.perf_counter() - t0),
        )
    except MediaProcessCancelled:
        return MediaProcessResponse(
            ok=False,
            error="media_process cancelled",
            frames_written=0,
            total_s=float(time.perf_counter() - t0),
        )
    except Exception as e:
        had_error = True
        logging.exception("media_process failed: source=%s kind=%s err=%s", req.source_path, req.source_kind, e)
        return MediaProcessResponse(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            frames_written=0,
            total_s=float(time.perf_counter() - t0),
        )
    finally:
        if had_error and _empty_cache_after_media_error():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
