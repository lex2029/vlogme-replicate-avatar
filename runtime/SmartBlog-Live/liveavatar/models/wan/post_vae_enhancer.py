from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import kornia
import torch
import torch.nn.functional as F
from avalife.core.observability import deep_timing_enabled, log_phase_timing, post_vae_timing_enabled


def _env_flag(name: str, default: str = "0") -> bool:
    try:
        raw = str(os.getenv(name, default) or default).strip().lower()
    except Exception:
        raw = str(default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _device_type(device: object) -> str:
    raw = getattr(device, "type", None)
    if raw is not None:
        return str(raw).strip().lower()
    return str(device).strip().lower().split(":", 1)[0]


def _cuda_device_context(device: object):
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(torch_device)
    return nullcontext()


def _finite_clamp01(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def _post_vae_phase_timing_enabled() -> bool:
    return bool(post_vae_timing_enabled()) or _env_flag("LIVE_RAW_POST_VAE_PHASE_TIMING", "1")


def _post_vae_phase_sync_enabled() -> bool:
    return _env_flag("LIVE_RAW_POST_VAE_PHASE_SYNC", "0")


def _maybe_sync_phase(device: object) -> None:
    if not bool(_post_vae_phase_sync_enabled()):
        return
    if _device_type(device) != "cuda":
        return
    try:
        torch.cuda.synchronize(torch.device(device))
    except Exception:
        torch.cuda.synchronize()


def _phase_dt(started_at: float, *, device: object) -> float:
    _maybe_sync_phase(device)
    return max(0.0, float(time.perf_counter() - float(started_at)))


def _select_primary_landmarks(
    landmarks: list[torch.Tensor],
    *,
    max_faces: int,
    height: int,
    width: int,
) -> list[torch.Tensor]:
    if not landmarks:
        return []
    frame_center = torch.tensor(
        [float(width) * 0.5, float(height) * 0.5],
        dtype=torch.float32,
        device=landmarks[0].device,
    )
    frame_diag = max(1.0, float((float(width) ** 2 + float(height) ** 2) ** 0.5))
    scored: list[tuple[float, int, torch.Tensor]] = []
    for idx, item in enumerate(landmarks):
        lm = item.to(dtype=torch.float32)
        mins = lm.amin(dim=0)
        maxs = lm.amax(dim=0)
        extent = (maxs - mins).clamp(min=1.0)
        area = float((extent[0] * extent[1]).item())
        center = (mins + maxs) * 0.5
        center_dist = float(torch.linalg.vector_norm(center - frame_center).item()) / frame_diag
        score = float(area) * max(0.15, 1.0 - float(center_dist))
        scored.append((score, -idx, lm))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    limit = max(1, int(max_faces))
    return [lm for _, _, lm in scored[:limit]]


def _similarity_affine_cpu(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit small face affine matrices on CPU.

    Some Blackwell containers fail when PyTorch JITs CUDA reduction kernels for
    the tiny SVD/determinant path used by the GPU implementation. The affine
    input is only five facial landmarks, so doing this math with OpenCV on CPU is
    effectively free and avoids those architecture-specific NVRTC failures.
    """
    import cv2
    import numpy as np

    src_f = src.detach().to(dtype=torch.float32)
    dst_f = dst.detach().to(dtype=torch.float32)
    squeeze = False
    if src_f.ndim == 2:
        src_f = src_f.unsqueeze(0)
        squeeze = True
    if dst_f.ndim == 2:
        dst_f = dst_f.unsqueeze(0).expand(src_f.shape[0], -1, -1)
    src_np = src_f.cpu().numpy()
    dst_np = dst_f.cpu().numpy()
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mats: list[np.ndarray] = []
    invs: list[np.ndarray] = []
    for src_i, dst_i in zip(src_np, dst_np):
        try:
            m = cv2.estimateAffinePartial2D(src_i.astype(np.float32), dst_i.astype(np.float32), method=cv2.LMEDS)[0]
            if m is None:
                raise ValueError("estimateAffinePartial2D returned None")
            m = np.asarray(m, dtype=np.float32).reshape(2, 3)
            m_inv = np.asarray(cv2.invertAffineTransform(m), dtype=np.float32).reshape(2, 3)
        except Exception:
            m = identity.copy()
            m_inv = identity.copy()
        mats.append(m)
        invs.append(m_inv)
    m_t = torch.as_tensor(np.stack(mats, axis=0), dtype=torch.float32, device=device)
    m_inv_t = torch.as_tensor(np.stack(invs, axis=0), dtype=torch.float32, device=device)
    if squeeze:
        return m_t[:1], m_inv_t[:1]
    return m_t, m_inv_t


@dataclass(frozen=True)
class PostVAEEnhancerConfig:
    enabled: bool
    models_dir: str | None
    use_trt: bool
    trt_max_dim: int

    @classmethod
    def from_env(cls) -> "PostVAEEnhancerConfig":
        models_dir = str(os.getenv("LIVE_RAW_POST_VAE_ENHANCER_MODELS_DIR", "") or "").strip()
        try:
            trt_max_dim = int(str(os.getenv("SMARTBLOG_MEDIA_TRT_MAX_DIM", "832") or "832").strip())
        except Exception:
            trt_max_dim = 832
        return cls(
            enabled=bool(_env_flag("LIVE_RAW_POST_VAE_ENHANCER", "0")),
            models_dir=models_dir or None,
            use_trt=bool(_env_flag("LIVE_RAW_POST_VAE_ENHANCER_TRT", "0")),
            trt_max_dim=max(64, int(trt_max_dim)),
        )


@dataclass(frozen=True)
class PostVAEEnhanceSettings:
    face_restore: float
    background_restore: float

    @classmethod
    def create(cls, *, face_restore: float, background_restore: float) -> "PostVAEEnhanceSettings":
        return cls(
            face_restore=_clamp01(face_restore),
            background_restore=_clamp01(background_restore),
        )

    @property
    def face_enabled(self) -> bool:
        return float(self.face_restore) > 0.0

    @property
    def background_enabled(self) -> bool:
        return float(self.background_restore) > 0.0

    @property
    def active(self) -> bool:
        return bool(self.face_enabled or self.background_enabled)


class PostVAEEnhancer:
    _LIVE_FACE_LAYOUT_REFRESH_CLIPS = 4

    def __init__(self, *, device: torch.device | str):
        self.device = torch.device(device)
        self.cfg = PostVAEEnhancerConfig.from_env()
        self._settings = PostVAEEnhanceSettings.create(face_restore=0.0, background_restore=0.0)
        self._enhancers: dict[tuple[str, int], object] = {}
        self._enhancer_failures: set[tuple[str, int]] = set()
        self._preallocated_shape: dict[tuple[str, int], tuple[int, int] | None] = {}
        self._face_layout_cache: dict[tuple[int, int], dict[str, object]] = {}
        self._face_restore_prewarm_shapes: set[tuple[int, int]] = set()
        self._last_debug_info: dict[str, object] = {}
        self._face_chunk_seq = 0
        try:
            self._max_faces = max(1, int(str(os.getenv("LIVE_RAW_POST_VAE_MAX_FACES", "1") or "1").strip()))
        except Exception:
            self._max_faces = 1
        try:
            self._face_detect_score_threshold = float(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_DETECT_SCORE", "0.97") or "0.97").strip()
            )
        except Exception:
            self._face_detect_score_threshold = 0.97
        try:
            self._face_mask_radius_x = float(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_MASK_RADIUS_X", "0.34") or "0.34").strip()
            )
        except Exception:
            self._face_mask_radius_x = 0.34
        try:
            self._face_mask_radius_y = float(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_MASK_RADIUS_Y", "0.46") or "0.46").strip()
            )
        except Exception:
            self._face_mask_radius_y = 0.46
        try:
            self._face_mask_center_y = float(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_MASK_CENTER_Y", "0.52") or "0.52").strip()
            )
        except Exception:
            self._face_mask_center_y = 0.52
        try:
            self._face_mask_softness = float(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_MASK_SOFTNESS", "0.14") or "0.14").strip()
            )
        except Exception:
            self._face_mask_softness = 0.14
        self._face_mask_mode = self._normalize_face_mask_mode(
            os.getenv("LIVE_RAW_POST_VAE_FACE_MASK_MODE", "inner_square") or "inner_square"
        )
        self._face_mask_radius_x = max(0.12, min(0.49, float(self._face_mask_radius_x)))
        self._face_mask_radius_y = max(0.12, min(0.60, float(self._face_mask_radius_y)))
        self._face_mask_center_y = max(0.35, min(0.70, float(self._face_mask_center_y)))
        self._face_mask_softness = max(0.02, min(0.35, float(self._face_mask_softness)))
        self._face_detect_batch_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_FACE_DETECT_BATCH", "0"))
        self._face_affine_cpu_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_FACE_AFFINE_CPU", "0"))
        try:
            self._face_detect_stride = max(
                1,
                int(str(os.getenv("LIVE_RAW_POST_VAE_FACE_DETECT_STRIDE", "1") or "1").strip()),
            )
        except Exception:
            self._face_detect_stride = 1
        self._face_aligned_layout_mode = str(
            os.getenv("LIVE_RAW_POST_VAE_FACE_ALIGNED_LAYOUT_MODE", "frame_loop") or "frame_loop"
        ).strip().lower()
        try:
            self._face_layout_refresh_clips = max(
                1,
                int(str(os.getenv("LIVE_RAW_POST_VAE_FACE_LAYOUT_REFRESH_CLIPS", "4") or "4").strip()),
            )
        except Exception:
            self._face_layout_refresh_clips = 4
        try:
            self._face_layout_change_threshold = max(
                0.0,
                min(
                    1.0,
                    float(str(os.getenv("LIVE_RAW_POST_VAE_FACE_LAYOUT_CHANGE_THRESHOLD", "0.08") or "0.08").strip()),
                ),
            )
        except Exception:
            self._face_layout_change_threshold = 0.08
        try:
            self._face_layout_ema = max(
                0.0,
                min(0.98, float(str(os.getenv("LIVE_RAW_POST_VAE_FACE_LAYOUT_EMA", "0.0") or "0.0").strip())),
            )
        except Exception:
            self._face_layout_ema = 0.0
        self._logged_batch_detect_fallback = False
        self._face_restore_amp_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_FACE_RESTORE_AMP", "1"))
        self._face_restore_cudnn_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_FACE_RESTORE_CUDNN", "0"))
        self._face_restore_amp_failed = False
        try:
            self._face_restore_batch_size = max(
                1,
                int(str(os.getenv("LIVE_RAW_POST_VAE_FACE_RESTORE_BATCH_SIZE", "1") or "1").strip()),
            )
        except Exception:
            self._face_restore_batch_size = 1
        self._face_restore_empty_cache_each_batch = bool(
            _env_flag("LIVE_RAW_POST_VAE_FACE_RESTORE_EMPTY_CACHE", "0")
        )
        try:
            self._background_batch_size = max(
                1,
                int(str(os.getenv("LIVE_RAW_POST_VAE_BACKGROUND_BATCH_SIZE", "1") or "1").strip()),
            )
        except Exception:
            self._background_batch_size = 1
        self._logged_face_restore_oom = False
        self._logged_face_restore_failure = False
        self._logged_background_trt_fallback = False
        self._logged_background_failure = False
        self._background_pytorch_fallback_enabled = bool(
            _env_flag("LIVE_RAW_POST_VAE_BACKGROUND_PYTORCH_FALLBACK", "1")
        )
        self._trt_finite_check_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_TRT_FINITE_CHECK", "0"))
        self._upscale_x2_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_UPSCALE_X2", "1"))
        self._face_source_x2_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_FACE_SOURCE_X2", "0"))
        self._face_restore_stage = self._normalize_face_restore_stage(
            os.getenv("LIVE_RAW_POST_VAE_FACE_RESTORE_STAGE", "post_vae") or "post_vae"
        )
        self._face_restore_small_crop_enabled = bool(
            _env_flag("LIVE_RAW_POST_VAE_FACE_RESTORE_SMALL_CROP_ENABLED", "1")
        )
        try:
            self._face_restore_small_crop_max_strength = _clamp01(
                float(
                    str(
                        os.getenv(
                            "LIVE_RAW_POST_VAE_FACE_RESTORE_SMALL_CROP_MAX_STRENGTH",
                            "0.15",
                        )
                        or "0.15"
                    ).strip()
                )
            )
        except Exception:
            self._face_restore_small_crop_max_strength = 0.15
        try:
            self._face_restore_small_crop_size = int(
                str(os.getenv("LIVE_RAW_POST_VAE_FACE_RESTORE_SMALL_CROP_SIZE", "512") or "512").strip()
            )
        except Exception:
            self._face_restore_small_crop_size = 512
        # GFPGAN clean uses fixed 512x512 face features. Smaller crops can pass
        # shape checks until a linear layer fails and poisons the CUDA context.
        self._face_restore_small_crop_size = max(512, int(self._face_restore_small_crop_size))
        self._debug_face_crops_enabled = bool(_env_flag("LIVE_RAW_POST_VAE_DEBUG_FACE_CROPS", "0"))
        self._debug_face_crops_dir = str(
            os.getenv("LIVE_RAW_POST_VAE_DEBUG_FACE_CROPS_DIR", "/tmp/vlogme-avatar-face-debug")
            or "/tmp/vlogme-avatar-face-debug"
        )
        try:
            self._debug_face_crops_max = max(
                1,
                int(str(os.getenv("LIVE_RAW_POST_VAE_DEBUG_FACE_CROPS_MAX", "6") or "6").strip()),
            )
        except Exception:
            self._debug_face_crops_max = 6
        self._debug_face_crops_saved = 0
        self._logged_ready = False
        self._logged_first_invoke = False
        self._logged_first_face_overlay = False
        self._logged_face_prewarm = False
        self._logged_face_overlay_failure = False

    def reset_session(self) -> None:
        self._face_layout_cache.clear()
        self._face_chunk_seq = 0
        self._last_debug_info = {}
        for enhancer in self._enhancers.values():
            cached = getattr(enhancer, "_cached_affine", None)
            if isinstance(cached, dict):
                cached.clear()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def set_restore_strengths(self, *, face_restore: float, background_restore: float) -> None:
        self._settings = PostVAEEnhanceSettings.create(
            face_restore=float(face_restore),
            background_restore=float(background_restore),
        )

    def _get_enhancer(self, *, mode: str, use_trt: bool) -> object | None:
        key = (str(mode), 1 if bool(use_trt) else 0)
        if key in self._enhancer_failures:
            return None
        cached = self._enhancers.get(key)
        if cached is not None:
            return cached
        try:
            from liveavatar.vendor.enchenh2d import Enhancer

            enhancer = Enhancer(
                mode=str(mode),
                device=str(self.device),
                trt=bool(use_trt),
                models_dir=self.cfg.models_dir,
            )
            self._enhancers[key] = enhancer
            if not self._logged_ready:
                logging.info(
                    "Post-VAE enhancer ready: device=%s models=%s trt=%d",
                    str(self.device),
                    str(self.cfg.models_dir or "-"),
                    1 if bool(self.cfg.use_trt) else 0,
                )
                self._logged_ready = True
            return enhancer
        except Exception as e:
            self._enhancer_failures.add(key)
            logging.exception("Post-VAE enhancer init failed: mode=%s trt=%d err=%s", str(mode), int(bool(use_trt)), e)
            return None

    def _should_use_trt(self, *, height: int, width: int) -> bool:
        return bool(self.cfg.use_trt) and max(int(height), int(width)) <= int(self.cfg.trt_max_dim)

    def _get_background_enhancer(self, *, use_trt: bool) -> tuple[object | None, bool]:
        bg_enhancer = self._get_enhancer(mode="upscale", use_trt=bool(use_trt))
        if bg_enhancer is not None or not bool(use_trt):
            return bg_enhancer, bool(use_trt)
        if not self._logged_background_trt_fallback:
            logging.warning(
                "Post-VAE background TensorRT unavailable; fallback=%s",
                "PyTorch" if bool(self._background_pytorch_fallback_enabled) else "bicubic",
            )
            self._logged_background_trt_fallback = True
        if not bool(self._background_pytorch_fallback_enabled):
            return None, bool(use_trt)
        bg_enhancer = self._get_enhancer(mode="upscale", use_trt=False)
        return bg_enhancer, False

    def _enhance_background_x2(
        self,
        bg_enhancer: object,
        frames_tchw: torch.Tensor,
        *,
        x2_h: int,
        x2_w: int,
        clip_id: int,
    ) -> torch.Tensor:
        frame_count = int(frames_tchw.shape[0])
        batch_size = max(1, int(getattr(self, "_background_batch_size", 1)))
        if frame_count <= batch_size:
            return bg_enhancer.enhance_gpu(
                frames_tchw,
                out_h=int(x2_h),
                out_w=int(x2_w),
                clip_id=int(clip_id),
                frame_idx=0,
            ).clamp(0, 1)
        chunks: list[torch.Tensor] = []
        for start in range(0, frame_count, batch_size):
            stop = min(frame_count, start + batch_size)
            chunk = bg_enhancer.enhance_gpu(
                frames_tchw[start:stop],
                out_h=int(x2_h),
                out_w=int(x2_w),
                clip_id=int(clip_id),
                frame_idx=int(start),
            ).clamp(0, 1)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def _preallocate(self, enhancer: object, *, mode: str, use_trt: bool, height: int, width: int) -> None:
        key = (str(mode), 1 if bool(use_trt) else 0)
        shape = (int(height), int(width))
        if self._preallocated_shape.get(key) == shape:
            return
        try:
            enhancer.preallocate(int(height), int(width), out_h=int(height), out_w=int(width))
            self._preallocated_shape[key] = shape
        except Exception as e:
            logging.exception(
                "Post-VAE enhancer preallocate failed: mode=%s trt=%d shape=%s err=%s",
                str(mode),
                int(bool(use_trt)),
                shape,
                e,
            )

    @staticmethod
    def _expand_affine(M: torch.Tensor, batch: int) -> torch.Tensor:
        if int(M.shape[0]) == int(batch):
            return M
        return M.expand(int(batch), -1, -1).contiguous()

    def _aligned_face_mask(self, *, height: int, width: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, int(height), dtype=torch.float32, device=device).view(1, 1, int(height), 1)
        x = torch.linspace(0.0, 1.0, int(width), dtype=torch.float32, device=device).view(1, 1, 1, int(width))
        rx = float(getattr(self, "_face_mask_radius_x", 0.34))
        ry = float(getattr(self, "_face_mask_radius_y", 0.46))
        cy = float(getattr(self, "_face_mask_center_y", 0.52))
        softness = float(getattr(self, "_face_mask_softness", 0.14))
        dist = ((x - 0.5) / rx).pow(2) + ((y - cy) / ry).pow(2)
        mask = ((1.0 - dist) / softness).clamp(0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        return mask.expand(1, 3, -1, -1).contiguous()

    @staticmethod
    def _full_square_face_mask(*, height: int, width: int, device: torch.device) -> torch.Tensor:
        return torch.ones((1, 3, int(height), int(width)), dtype=torch.float32, device=device).contiguous()

    @staticmethod
    def _inner_square_face_mask(*, height: int, width: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((1, 3, int(height), int(width)), dtype=torch.float32, device=device)
        # GFPGAN/facexlib paste compatibility: restore the aligned square crop,
        # but paste only the stable inner face area and feather it back.
        margin_y = max(1, min(int(height) // 2 - 1, int(round(float(height) * (64.0 / 512.0)))))
        margin_x = max(1, min(int(width) // 2 - 1, int(round(float(width) * (64.0 / 512.0)))))
        mask[:, :, int(margin_y) : int(height) - int(margin_y), int(margin_x) : int(width) - int(margin_x)] = 1.0
        return mask.contiguous()

    @staticmethod
    def _normalize_face_mask_mode(mode: object) -> str:
        raw = str(mode or "inner_square").strip().lower().replace("-", "_")
        if raw in {"gfpgan", "gfpgan_square", "gfpgan_inner", "square_inner", "inner_square", "legacy_square"}:
            return "inner_square"
        if raw in {"ellipse", "oval"}:
            return "ellipse"
        if raw in {"full_square", "square_full", "square_soft", "full"}:
            return "square_soft"
        return "inner_square"

    @staticmethod
    def _normalize_face_restore_stage(stage: object) -> str:
        raw = str(stage or "native_first").strip().lower().replace("-", "_")
        if raw in {"native", "native_first", "pre_upscale", "preupscale", "vae", "vae_native"}:
            return "native_first"
        if raw in {"post", "post_vae", "x2", "layer", "post_upscale", "postupscale"}:
            return "post_vae"
        return "native_first"

    @staticmethod
    def _erode_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel = int(max(1, kernel_size))
        if kernel <= 1:
            return mask
        if kernel % 2 == 0:
            kernel += 1
        return (-F.max_pool2d(-mask, kernel_size=int(kernel), stride=1, padding=int(kernel // 2))).clamp(0, 1)

    def _soften_square_face_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # GFPGAN/facexlib uses a square paste mask, but softens it after inverse
        # affine based on the visible face area. This keeps the canonical square
        # coverage without leaving a hard rectangular paste edge.
        mask = self._erode_mask(mask.clamp(0, 1), 3)
        try:
            area = float(mask[:, :1].sum().detach().item()) / float(max(1, int(mask.shape[0])))
        except Exception:
            area = float(mask.shape[-1] * mask.shape[-2]) * 0.04
        w_edge = max(2, int((max(1.0, area) ** 0.5) // 20))
        erosion_kernel = max(3, int(w_edge * 2))
        blur_kernel = max(7, int(w_edge * 2) + 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = self._erode_mask(mask, int(erosion_kernel))
        sigma = max(1.0, float(blur_kernel) / 3.0)
        return kornia.filters.gaussian_blur2d(
            mask,
            (int(blur_kernel), int(blur_kernel)),
            (float(sigma), float(sigma)),
        ).clamp(0, 1).contiguous()

    def _debug_save_face_crops(
        self,
        *,
        aligned: torch.Tensor,
        restored: torch.Tensor,
        prefix: str,
        composited: torch.Tensor | None = None,
    ) -> None:
        if not bool(getattr(self, "_debug_face_crops_enabled", False)):
            return
        remaining = int(getattr(self, "_debug_face_crops_max", 6)) - int(getattr(self, "_debug_face_crops_saved", 0))
        if remaining <= 0:
            return
        try:
            from pathlib import Path

            from PIL import Image

            debug_dir = Path(str(getattr(self, "_debug_face_crops_dir", "") or "/tmp/vlogme-avatar-face-debug"))
            debug_dir.mkdir(parents=True, exist_ok=True)
            aligned_cpu = aligned.detach().to(dtype=torch.float32).clamp(0.0, 1.0).cpu()
            restored_cpu = restored.detach().to(dtype=torch.float32).clamp(0.0, 1.0).cpu()
            composited_cpu = (
                composited.detach().to(dtype=torch.float32).clamp(0.0, 1.0).cpu()
                if composited is not None
                else None
            )
            count = min(int(remaining), int(aligned_cpu.shape[0]), int(restored_cpu.shape[0]))
            if composited_cpu is not None:
                count = min(int(count), int(composited_cpu.shape[0]))
            for batch_idx in range(int(count)):
                save_idx = int(getattr(self, "_debug_face_crops_saved", 0))
                tensors: list[tuple[str, torch.Tensor]] = [("aligned", aligned_cpu), ("restored", restored_cpu)]
                if composited_cpu is not None:
                    tensors.append(("composited", composited_cpu))
                for label, tensor in tensors:
                    arr = (
                        tensor[int(batch_idx)]
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .contiguous()
                        .numpy()
                    )
                    Image.fromarray(arr, mode="RGB").save(
                        debug_dir / f"{str(prefix)}_{save_idx:03d}_{str(label)}.jpg",
                        quality=95,
                    )
                self._debug_face_crops_saved = int(save_idx) + 1
        except Exception:
            if not bool(getattr(self, "_logged_debug_face_crops_failure", False)):
                logging.warning("Post-VAE debug face crop export failed", exc_info=True)
                self._logged_debug_face_crops_failure = True

    @staticmethod
    def _resize_cover_crop_to_output(tensor_01: torch.Tensor, *, out_h: int | None, out_w: int | None) -> torch.Tensor:
        target_h = int(out_h or 0)
        target_w = int(out_w or 0)
        if target_h <= 0 or target_w <= 0:
            return tensor_01
        src_h = int(tensor_01.shape[2])
        src_w = int(tensor_01.shape[3])
        if src_h == target_h and src_w == target_w:
            return tensor_01
        scale = max(float(target_h) / float(max(1, src_h)), float(target_w) / float(max(1, src_w)))
        new_h = max(1, int(round(float(src_h) * float(scale))))
        new_w = max(1, int(round(float(src_w) * float(scale))))
        resized = F.interpolate(tensor_01, size=(int(new_h), int(new_w)), mode="bicubic", align_corners=False)
        top = max(0, (int(new_h) - int(target_h)) // 2)
        left = max(0, (int(new_w) - int(target_w)) // 2)
        return resized[:, :, int(top) : int(top) + int(target_h), int(left) : int(left) + int(target_w)].contiguous()

    @staticmethod
    def _face_layout_anchor(tensor_01: torch.Tensor) -> torch.Tensor | None:
        try:
            return (
                F.interpolate(
                    tensor_01[:1].detach().to(dtype=torch.float32),
                    size=(32, 32),
                    mode="bilinear",
                    align_corners=False,
                )
                .detach()
                .contiguous()
            )
        except Exception:
            return None

    @staticmethod
    def _face_layout_anchor_diff(anchor: torch.Tensor | None, previous: object) -> float | None:
        if anchor is None or not torch.is_tensor(previous):
            return None
        try:
            prev = previous.to(device=anchor.device, dtype=anchor.dtype, copy=False)
            if tuple(int(v) for v in prev.shape) != tuple(int(v) for v in anchor.shape):
                return None
            return float((anchor - prev).abs().mean().detach().item())
        except Exception:
            return None

    def _blend_affine_pairs(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        cached_pairs: object,
        *,
        weight: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        ema = float(max(0.0, min(0.98, float(weight or 0.0))))
        if ema <= 0.0 or not isinstance(cached_pairs, list) or len(cached_pairs) != len(pairs):
            return pairs
        blended: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for (m_new, minv_new), (m_old, minv_old) in zip(pairs, cached_pairs):
                old_m = torch.as_tensor(m_old, device=m_new.device, dtype=m_new.dtype)
                old_minv = torch.as_tensor(minv_old, device=minv_new.device, dtype=minv_new.dtype)
                if tuple(old_m.shape) != tuple(m_new.shape) or tuple(old_minv.shape) != tuple(minv_new.shape):
                    return pairs
                blended.append(
                    (
                        (old_m * ema + m_new * (1.0 - ema)).contiguous(),
                        (old_minv * ema + minv_new * (1.0 - ema)).contiguous(),
                    )
                )
            return blended
        except Exception:
            return pairs

    def _get_face_affine_pairs(self, face_enhancer: object, tensor_01: torch.Tensor, *, clip_id: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch = int(tensor_01.shape[0])
        height = int(tensor_01.shape[2])
        width = int(tensor_01.shape[3])
        cache_key = (int(height), int(width))
        refresh_clips = int(max(1, int(getattr(self, "_face_layout_refresh_clips", self._LIVE_FACE_LAYOUT_REFRESH_CLIPS))))
        anchor = self._face_layout_anchor(tensor_01)

        cached_layout = self._face_layout_cache.get(cache_key)
        cached_pairs: object | None = None
        anchor_diff: float | None = None
        content_changed = False
        if isinstance(cached_layout, dict):
            cached_pairs = cached_layout.get("pairs")
            last_clip_id = int(cached_layout.get("clip_id") or 0)
            age = int(max(0, int(clip_id) - int(last_clip_id)))
            anchor_diff = self._face_layout_anchor_diff(anchor, cached_layout.get("anchor"))
            threshold = float(getattr(self, "_face_layout_change_threshold", 0.08) or 0.0)
            content_changed = bool(anchor_diff is not None and threshold > 0.0 and float(anchor_diff) > float(threshold))
            if cached_pairs is not None and age < int(refresh_clips) and not bool(content_changed):
                self._last_debug_info.update(
                    {
                        "face_layout_cache": "hit",
                        "face_layout_cache_age": int(age),
                        "face_layout_anchor_diff": None if anchor_diff is None else float(anchor_diff),
                    }
                )
                return list(cached_pairs)

        cached = getattr(face_enhancer, "_cached_affine", None)
        if isinstance(cached, dict) and cached.get("clip_id") == int(clip_id):
            pairs = cached.get("pairs")
            if pairs is not None:
                self._face_layout_cache[cache_key] = {
                    "pairs": list(pairs),
                    "clip_id": int(clip_id),
                    "anchor": anchor.detach().clone().contiguous() if anchor is not None else None,
                }
                self._last_debug_info.update({"face_layout_cache": "enhancer"})
                return list(pairs)

        from liveavatar.vendor.enchenh2d.enhancer import _detect_faces_gpu, _get_affine_matrices

        detect_src = tensor_01[:1].contiguous()
        landmarks = _detect_faces_gpu(
            getattr(face_enhancer, "_face_det"),
            detect_src,
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
        if not pairs and isinstance(cached_pairs, list) and cached_pairs and not bool(content_changed):
            self._last_debug_info.update(
                {
                    "face_layout_cache": "hit_detect_miss",
                    "face_layout_anchor_diff": None if anchor_diff is None else float(anchor_diff),
                }
            )
            return list(cached_pairs)
        if pairs and isinstance(cached_layout, dict) and not bool(content_changed):
            pairs = self._blend_affine_pairs(
                pairs,
                cached_layout.get("pairs"),
                weight=float(getattr(self, "_face_layout_ema", 0.0) or 0.0),
            )
        if isinstance(cached, dict):
            cached["clip_id"] = int(clip_id)
            cached["pairs"] = pairs
        self._face_layout_cache[cache_key] = {
            "pairs": list(pairs),
            "clip_id": int(clip_id),
            "anchor": anchor.detach().clone().contiguous() if anchor is not None else None,
        }
        self._last_debug_info.update(
            {
                "face_layout_cache": "miss_changed" if bool(content_changed) else "miss",
                "face_layout_anchor_diff": None if anchor_diff is None else float(anchor_diff),
            }
        )
        return pairs

    def _get_face_overlay_entries(
        self,
        face_enhancer: object,
        tensor_01: torch.Tensor,
        *,
        clip_id: int,
        output_height: int | None = None,
        output_width: int | None = None,
        affine_scale: float = 1.0,
        mask_mode: str = "",
        layout_mode: str = "",
    ) -> list[dict[str, torch.Tensor]]:
        batch = int(tensor_01.shape[0])
        height = int(tensor_01.shape[2])
        width = int(tensor_01.shape[3])
        out_h = int(output_height or height)
        out_w = int(output_width or width)
        scale = float(affine_scale or 1.0)
        detect_t0 = time.perf_counter()
        affine_batches = self._get_face_affine_batches(
            face_enhancer,
            tensor_01,
            clip_id=int(clip_id),
            layout_mode=str(layout_mode or ""),
        )
        detect_sec = _phase_dt(detect_t0, device=self.device)
        self._last_debug_info["face_detect_sec"] = float(detect_sec)
        log_phase_timing(
            "post_vae",
            "face_detect",
            detect_t0,
            enabled=bool(deep_timing_enabled()),
            sync_device=self.device,
            device=str(self.device),
            frames=int(batch),
            size=f"{int(width)}x{int(height)}",
            clip=int(clip_id),
            layout=str(layout_mode or ""),
        )
        if not affine_batches:
            return []
        face_h = int(getattr(face_enhancer, "_face_size")[1])
        face_w = int(getattr(face_enhancer, "_face_size")[0])
        mode = self._normalize_face_mask_mode(mask_mode or getattr(self, "_face_mask_mode", "inner_square"))
        self._last_debug_info["face_mask_mode"] = str(mode)
        if mode in {"ellipse", "oval"}:
            inner = self._aligned_face_mask(height=int(face_h), width=int(face_w), device=tensor_01.device)
        elif mode == "inner_square":
            inner = self._inner_square_face_mask(height=int(face_h), width=int(face_w), device=tensor_01.device)
        else:
            inner = self._full_square_face_mask(height=int(face_h), width=int(face_w), device=tensor_01.device)
        stable_layout = str(layout_mode or "").strip().lower() in {"stable_first", "stable", "cached_first", "first"}
        inner_batch = inner.expand((1 if bool(stable_layout) else int(batch)), -1, -1, -1).contiguous()

        entries: list[dict[str, torch.Tensor]] = []
        mask_t0 = time.perf_counter()
        for M_batch, M_inv_batch, valid_mask in affine_batches:
            if scale != 1.0:
                M_inv_out = M_inv_batch * float(scale)
            else:
                M_inv_out = M_inv_batch
            if bool(stable_layout):
                M_inv_for_mask = M_inv_out[:1].contiguous()
                valid_for_mask = valid_mask[:1].contiguous()
            else:
                M_inv_for_mask = M_inv_out
                valid_for_mask = valid_mask
            mask = kornia.geometry.transform.warp_affine(
                inner_batch,
                M_inv_for_mask,
                (int(out_h), int(out_w)),
                mode="bilinear",
                padding_mode="zeros",
            )
            if mode in {"ellipse", "inner_square"}:
                mask = kornia.filters.gaussian_blur2d(mask, (51, 51), (17.0, 17.0)).contiguous()
            else:
                mask = self._soften_square_face_mask(mask)
            mask = (mask * valid_for_mask.to(device=mask.device, dtype=mask.dtype)).contiguous()
            if float(mask.sum().detach().item()) <= 0.0:
                continue
            if bool(stable_layout) and int(batch) > 1:
                mask = mask.expand(int(batch), -1, -1, -1).contiguous()
            entries.append(
                {
                    "M_batch": M_batch,
                    "M_inv_batch": M_inv_out.contiguous(),
                    "mask": mask,
                }
            )
        self._last_debug_info["face_mask_sec"] = float(_phase_dt(mask_t0, device=self.device))
        return entries

    def _get_face_affine_batches(
        self,
        face_enhancer: object,
        tensor_01: torch.Tensor,
        *,
        clip_id: int = 0,
        layout_mode: str = "",
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        mode = str(layout_mode or "").strip().lower()
        if mode in {"stable_first", "stable", "cached_first", "first"}:
            return self._get_face_affine_batches_stable_first(
                face_enhancer,
                tensor_01,
                clip_id=int(clip_id or 0),
            )
        if not bool(getattr(self, "_face_detect_batch_enabled", False)):
            return self._get_face_affine_batches_frame_loop(face_enhancer, tensor_01)
        try:
            return self._get_face_affine_batches_detect_once(face_enhancer, tensor_01)
        except Exception as e:
            if not bool(getattr(self, "_logged_batch_detect_fallback", False)):
                logging.warning("Post-VAE batch face detect failed; falling back to frame loop: %s", e)
                self._logged_batch_detect_fallback = True
        return self._get_face_affine_batches_frame_loop(face_enhancer, tensor_01)

    def _get_face_affine_batches_stable_first(
        self,
        face_enhancer: object,
        tensor_01: torch.Tensor,
        *,
        clip_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch = int(tensor_01.shape[0])
        pairs = self._get_face_affine_pairs(face_enhancer, tensor_01, clip_id=int(clip_id))
        if not pairs:
            return []
        valid_mask = torch.ones((int(batch), 1, 1, 1), dtype=torch.float32, device=tensor_01.device)
        out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for M, M_inv in pairs:
            try:
                M_t = torch.as_tensor(M, dtype=torch.float32, device=tensor_01.device).view(1, 2, 3)
                M_inv_t = torch.as_tensor(M_inv, dtype=torch.float32, device=tensor_01.device).view(1, 2, 3)
            except Exception:
                continue
            out.append(
                (
                    M_t.expand(int(batch), -1, -1).contiguous(),
                    M_inv_t.expand(int(batch), -1, -1).contiguous(),
                    valid_mask.contiguous(),
                )
            )
        self._last_debug_info["face_layout_mode"] = "stable_first"
        self._last_debug_info["face_detect_stride"] = int(batch)
        self._last_debug_info["face_detect_frames"] = 1
        self._last_debug_info["face_detect_reused_frames"] = max(0, int(batch) - 1)
        return out

    def _get_face_affine_batches_detect_once(
        self,
        face_enhancer: object,
        tensor_01: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        from facexlib.detection.retinaface_utils import PriorBox, decode, decode_landm
        from liveavatar.vendor.enchenh2d.enhancer import (
            _face_detect_forward,
            _nms_boxes_torch,
            _similarity_affine_torch,
        )

        batch = int(tensor_01.shape[0])
        height = int(tensor_01.shape[2])
        width = int(tensor_01.shape[3])
        face_det = getattr(face_enhancer, "_face_det")
        img_bgr = tensor_01[:, [2, 1, 0], :, :] * 255.0
        if bool(getattr(face_det, "half_inference", False)):
            img_bgr = img_bgr.half()
        img_bgr = img_bgr - face_det.mean_tensor.to(img_bgr.device)
        loc, conf, landmarks_raw = _face_detect_forward(face_det, img_bgr)
        priors_cache = getattr(face_enhancer, "_priors_cache", None)
        if isinstance(priors_cache, dict) and (height, width) in priors_cache:
            priors = priors_cache[(height, width)]
        else:
            priorbox = PriorBox(face_det.cfg, image_size=(height, width))
            priors = priorbox.forward().to(img_bgr.device)
        scale = torch.tensor([width, height, width, height], device=img_bgr.device, dtype=torch.float32)
        scale1 = torch.tensor([width, height] * 5, device=img_bgr.device, dtype=torch.float32)
        template = getattr(face_enhancer, "_face_template").to(device=img_bgr.device, dtype=torch.float32)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=img_bgr.device,
        )
        selected: list[torch.Tensor] = []
        valid: list[bool] = []
        threshold = float(getattr(self, "_face_detect_score_threshold", 0.97))
        max_faces = max(1, int(getattr(self, "_max_faces", 1)))
        for idx in range(batch):
            loc_i = loc[idx] if loc.ndim == 3 else loc
            conf_i = conf[idx] if conf.ndim == 3 else conf
            landmarks_i = landmarks_raw[idx] if landmarks_raw.ndim == 3 else landmarks_raw
            boxes = decode(loc_i.data, priors.data, face_det.cfg["variance"])
            boxes = (boxes * scale).detach()
            scores = conf_i.data[:, 1].detach().float()
            decoded_landmarks = decode_landm(landmarks_i, priors, face_det.cfg["variance"])
            decoded_landmarks = (decoded_landmarks * scale1).detach().view(-1, 5, 2).float()
            candidate_idx = torch.nonzero(scores > threshold, as_tuple=False).flatten()
            if int(candidate_idx.numel()) == 0:
                selected.append(template)
                valid.append(False)
                continue
            keep = _nms_boxes_torch(boxes[candidate_idx], scores[candidate_idx], iou_threshold=0.4)
            if int(keep.numel()) > 0:
                candidate_idx = candidate_idx[keep]
            candidates = [decoded_landmarks[int(i.item())] for i in candidate_idx]
            primary = _select_primary_landmarks(
                candidates,
                max_faces=max_faces,
                height=int(height),
                width=int(width),
            )
            if not primary:
                selected.append(template)
                valid.append(False)
                continue
            selected.append(primary[0])
            valid.append(True)
        src = torch.stack(selected, dim=0).to(device=img_bgr.device, dtype=torch.float32)
        dst = template.unsqueeze(0).expand(int(batch), -1, -1).contiguous()
        if bool(getattr(self, "_face_affine_cpu_enabled", False)):
            M_batch, M_inv_batch = _similarity_affine_cpu(src, dst, device=img_bgr.device)
        else:
            M_batch, M_inv_batch = _similarity_affine_torch(src, dst)
        if not all(valid):
            for idx, ok in enumerate(valid):
                if not ok:
                    M_batch[idx] = identity
                    M_inv_batch[idx] = identity
        valid_mask = torch.tensor(valid, dtype=torch.float32, device=img_bgr.device).view(batch, 1, 1, 1)
        if float(valid_mask.sum().detach().item()) <= 0.0:
            return []
        return [(M_batch.contiguous(), M_inv_batch.contiguous(), valid_mask.contiguous())]

    def _get_face_affine_batches_frame_loop(
        self,
        face_enhancer: object,
        tensor_01: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        from liveavatar.vendor.enchenh2d.enhancer import _detect_faces_gpu, _similarity_affine_torch

        batch = int(tensor_01.shape[0])
        height = int(tensor_01.shape[2])
        width = int(tensor_01.shape[3])
        template = getattr(face_enhancer, "_face_template").to(device=tensor_01.device, dtype=torch.float32)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=tensor_01.device,
        )
        selected: list[torch.Tensor] = []
        valid: list[bool] = []
        max_faces = max(1, int(getattr(self, "_max_faces", 1)))
        detect_stride = max(1, int(getattr(self, "_face_detect_stride", 1) or 1))
        detections = 0
        reuses = 0
        for idx in range(batch):
            if int(detect_stride) > 1 and idx > 0 and (idx % int(detect_stride)) != 0 and selected and valid[-1]:
                selected.append(selected[-1].clone())
                valid.append(True)
                reuses += 1
                continue
            landmarks = _detect_faces_gpu(
                getattr(face_enhancer, "_face_det"),
                tensor_01[idx : idx + 1].contiguous(),
                priors_cache=getattr(face_enhancer, "_priors_cache", None),
            )
            detections += 1
            candidates = [torch.as_tensor(item, dtype=torch.float32, device=tensor_01.device) for item in landmarks]
            primary = _select_primary_landmarks(
                candidates,
                max_faces=max_faces,
                height=int(height),
                width=int(width),
            )
            if not primary:
                selected.append(template)
                valid.append(False)
            else:
                selected.append(primary[0])
                valid.append(True)
        self._last_debug_info["face_detect_stride"] = int(detect_stride)
        self._last_debug_info["face_detect_frames"] = int(detections)
        self._last_debug_info["face_detect_reused_frames"] = int(reuses)
        src = torch.stack(selected, dim=0).to(device=tensor_01.device, dtype=torch.float32)
        dst = template.unsqueeze(0).expand(int(batch), -1, -1).contiguous()
        if bool(getattr(self, "_face_affine_cpu_enabled", False)):
            M_batch, M_inv_batch = _similarity_affine_cpu(src, dst, device=tensor_01.device)
        else:
            M_batch, M_inv_batch = _similarity_affine_torch(src, dst)
        if not all(valid):
            for idx, ok in enumerate(valid):
                if not ok:
                    M_batch[idx] = identity
                    M_inv_batch[idx] = identity
        valid_mask = torch.tensor(valid, dtype=torch.float32, device=tensor_01.device).view(batch, 1, 1, 1)
        if float(valid_mask.sum().detach().item()) <= 0.0:
            return []
        return [(M_batch.contiguous(), M_inv_batch.contiguous(), valid_mask.contiguous())]

    def _apply_face_overlay_batch(
        self,
        target_01: torch.Tensor,
        *,
        base_01: torch.Tensor,
        face_enhancer: object,
        overlay_mode: str,
        face_restore: float,
        clip_id: int,
        overlay_entries: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        overlay_mode = str(overlay_mode or "").strip().lower()
        if overlay_mode not in {"original", "restored"}:
            return target_01
        if overlay_entries is None:
            overlay_entries = self._get_face_overlay_entries(face_enhancer, base_01, clip_id=int(clip_id))
        if not overlay_entries:
            return target_01

        result = target_01
        face_h = int(getattr(face_enhancer, "_face_size")[1])
        face_w = int(getattr(face_enhancer, "_face_size")[0])
        blend = _clamp01(face_restore)
        profile = {
            "crop_sec": 0.0,
            "restore_sec": 0.0,
            "blend_sec": 0.0,
            "paste_sec": 0.0,
            "composite_sec": 0.0,
            "entries": int(len(overlay_entries)),
        }

        for entry in overlay_entries:
            try:
                M_batch = entry["M_batch"]
                M_inv_batch = entry["M_inv_batch"]
                mask = entry["mask"]
                crop_t0 = time.perf_counter()
                original_crop = kornia.geometry.transform.warp_affine(
                    base_01,
                    M_batch,
                    (face_h, face_w),
                    mode="bilinear",
                    padding_mode="zeros",
                )
                profile["crop_sec"] += _phase_dt(crop_t0, device=self.device)
                if overlay_mode == "restored":
                    restore_t0 = time.perf_counter()
                    restored_face = self._run_face_restore_model(
                        face_enhancer,
                        original_crop,
                        face_restore=float(face_restore),
                    )
                    profile["restore_sec"] += _phase_dt(restore_t0, device=self.device)
                    if torch.isnan(restored_face).any() or torch.isinf(restored_face).any():
                        continue
                    face_patch = restored_face
                    if blend < 1.0:
                        blend_t0 = time.perf_counter()
                        face_patch = (face_patch * blend) + (original_crop * (1.0 - blend))
                        profile["blend_sec"] += _phase_dt(blend_t0, device=self.device)
                else:
                    face_patch = original_crop

                paste_t0 = time.perf_counter()
                pasted = kornia.geometry.transform.warp_affine(
                    face_patch,
                    M_inv_batch,
                    (int(target_01.shape[2]), int(target_01.shape[3])),
                    mode="bilinear",
                    padding_mode="zeros",
                )
                profile["paste_sec"] += _phase_dt(paste_t0, device=self.device)
                composite_t0 = time.perf_counter()
                result = (result * (1.0 - mask)) + (pasted * mask)
                profile["composite_sec"] += _phase_dt(composite_t0, device=self.device)
                if overlay_mode == "restored" and bool(getattr(self, "_debug_face_crops_enabled", False)):
                    debug_M = M_batch
                    if int(result.shape[2]) != int(base_01.shape[2]) or int(result.shape[3]) != int(base_01.shape[3]):
                        scale_x = float(result.shape[3]) / float(max(1, int(base_01.shape[3])))
                        scale_y = float(result.shape[2]) / float(max(1, int(base_01.shape[2])))
                        debug_M = M_batch.clone()
                        debug_M[:, 0, 0] = debug_M[:, 0, 0] / float(scale_x)
                        debug_M[:, 0, 1] = debug_M[:, 0, 1] / float(scale_y)
                        debug_M[:, 1, 0] = debug_M[:, 1, 0] / float(scale_x)
                        debug_M[:, 1, 1] = debug_M[:, 1, 1] / float(scale_y)
                    composited_crop = kornia.geometry.transform.warp_affine(
                        result,
                        debug_M,
                        (face_h, face_w),
                        mode="bilinear",
                        padding_mode="zeros",
                    )
                    self._debug_save_face_crops(
                        aligned=original_crop,
                        restored=restored_face,
                        composited=composited_crop,
                        prefix=str(overlay_mode or "face"),
                    )
            except Exception:
                if not bool(getattr(self, "_logged_face_overlay_failure", False)):
                    logging.warning("Post-VAE face overlay failed; leaving target frame unpatched", exc_info=True)
                    self._logged_face_overlay_failure = True
                continue
        self._last_debug_info["face_overlay_profile"] = dict(profile)
        return result.clamp(0, 1)

    def apply_face_aligned_transform_tchw(
        self,
        frames_01: torch.Tensor,
        *,
        transform_callback: object,
        face_restore: float,
    ) -> torch.Tensor | None:
        """Run a caller transform inside the canonical face crop and paste once.

        This is used by the inline MuseTalk path: the face enhancer owns face
        detection, affine crop, optional GFPGAN restore, and inverse-affine paste.
        MuseTalk only modifies the aligned crop, so the final frame does not go
        through the older full-frame bbox mouth paste plus a second face paste.
        """
        if not self.enabled:
            return None
        if frames_01 is None or frames_01.ndim != 4 or int(frames_01.shape[1]) != 3:
            raise ValueError(f"Expected frames [T,3,H,W], got {tuple(frames_01.shape)}")
        if not callable(transform_callback):
            return None

        face = _clamp01(float(face_restore or 0.0))
        started_total = time.perf_counter()
        with torch.inference_mode(), _cuda_device_context(self.device):
            base_01 = _finite_clamp01(frames_01.to(device=self.device, dtype=torch.float32, copy=False).contiguous())
            batch, _, height, width = (int(v) for v in base_01.shape)
            face_init_t0 = time.perf_counter()
            face_enhancer = self._get_enhancer(mode="face", use_trt=False)
            if face_enhancer is None:
                self._last_debug_info.update({"face_aligned_transform": "no_face_enhancer"})
                return None
            self._preallocate(face_enhancer, mode="face", use_trt=False, height=int(height), width=int(width))
            face_init_sec = _phase_dt(face_init_t0, device=self.device)

            clip_id = int(self._face_chunk_seq) + 1
            self._face_chunk_seq = int(clip_id)
            layout_t0 = time.perf_counter()
            overlay_entries = self._get_face_overlay_entries(
                face_enhancer,
                base_01,
                clip_id=int(clip_id),
                output_height=int(height),
                output_width=int(width),
                affine_scale=1.0,
                mask_mode=str(getattr(self, "_face_mask_mode", "inner_square") or "inner_square"),
                layout_mode=str(getattr(self, "_face_aligned_layout_mode", "frame_loop") or "frame_loop"),
            )
            layout_sec = _phase_dt(layout_t0, device=self.device)
            if not overlay_entries:
                self._last_debug_info.update({"face_aligned_transform": "no_face_entries"})
                return None

            face_h = int(getattr(face_enhancer, "_face_size")[1])
            face_w = int(getattr(face_enhancer, "_face_size")[0])
            result = base_01
            profile: dict[str, float | int] = {
                "entries": int(len(overlay_entries)),
                "face_init_sec": float(face_init_sec),
                "layout_sec": float(layout_sec),
                "crop_sec": 0.0,
                "transform_sec": 0.0,
                "restore_sec": 0.0,
                "blend_sec": 0.0,
                "paste_sec": 0.0,
                "composite_sec": 0.0,
            }
            transformed_any = False

            for entry in overlay_entries:
                try:
                    M_batch = entry["M_batch"]
                    M_inv_batch = entry["M_inv_batch"]
                    mask = entry["mask"]
                    crop_t0 = time.perf_counter()
                    aligned_crop = kornia.geometry.transform.warp_affine(
                        base_01,
                        M_batch,
                        (int(face_h), int(face_w)),
                        mode="bilinear",
                        padding_mode="zeros",
                    ).clamp(0.0, 1.0)
                    profile["crop_sec"] = float(profile["crop_sec"]) + _phase_dt(crop_t0, device=self.device)

                    transform_t0 = time.perf_counter()
                    transformed_crop = transform_callback(aligned_crop)
                    profile["transform_sec"] = float(profile["transform_sec"]) + _phase_dt(transform_t0, device=self.device)
                    if transformed_crop is None:
                        transformed_crop = aligned_crop
                    transformed_crop = getattr(transformed_crop, "tensor", transformed_crop)
                    if not torch.is_tensor(transformed_crop):
                        transformed_crop = aligned_crop
                    transformed_crop = transformed_crop.to(device=self.device, dtype=torch.float32, copy=False).contiguous().clamp(0.0, 1.0)
                    if tuple(int(v) for v in transformed_crop.shape[-2:]) != (int(face_h), int(face_w)):
                        transformed_crop = F.interpolate(
                            transformed_crop,
                            size=(int(face_h), int(face_w)),
                            mode="bilinear",
                            align_corners=False,
                        ).clamp(0.0, 1.0)

                    face_patch = transformed_crop
                    if face > 0.0:
                        restore_t0 = time.perf_counter()
                        restored = self._run_face_restore_model(
                            face_enhancer,
                            transformed_crop,
                            face_restore=float(face),
                        )
                        profile["restore_sec"] = float(profile["restore_sec"]) + _phase_dt(restore_t0, device=self.device)
                        if restored is not None and not torch.isnan(restored).any() and not torch.isinf(restored).any():
                            face_patch = restored.clamp(0.0, 1.0)
                            if face < 1.0:
                                blend_t0 = time.perf_counter()
                                face_patch = (face_patch * face) + (transformed_crop * (1.0 - face))
                                profile["blend_sec"] = float(profile["blend_sec"]) + _phase_dt(blend_t0, device=self.device)

                    paste_t0 = time.perf_counter()
                    pasted = kornia.geometry.transform.warp_affine(
                        face_patch,
                        M_inv_batch,
                        (int(height), int(width)),
                        mode="bilinear",
                        padding_mode="zeros",
                    )
                    profile["paste_sec"] = float(profile["paste_sec"]) + _phase_dt(paste_t0, device=self.device)

                    composite_t0 = time.perf_counter()
                    result = (result * (1.0 - mask)) + (pasted * mask)
                    profile["composite_sec"] = float(profile["composite_sec"]) + _phase_dt(composite_t0, device=self.device)
                    transformed_any = True
                except Exception:
                    if not bool(getattr(self, "_logged_face_aligned_transform_failure", False)):
                        logging.warning("Post-VAE face-aligned transform failed; leaving frame unpatched", exc_info=True)
                        self._logged_face_aligned_transform_failure = True
                    continue

            total_sec = max(0.0, time.perf_counter() - float(started_total))
            profile["total_sec"] = float(total_sec)
            self._last_debug_info["face_aligned_transform_profile"] = dict(profile)
            self._last_debug_info.update(
                {
                    "face_aligned_transform": "applied" if transformed_any else "no_valid_entries",
                    "face_aligned_transform_entries": int(len(overlay_entries)),
                    "face_aligned_transform_shape": (int(width), int(height)),
                }
            )
            logging.warning(
                "Post-VAE face-aligned MuseTalk profile: device=%s frames=%d size=%dx%d face=%.2f entries=%d layout_mode=%s layout_cache=%s layout_diff=%s face_init=%.3fs layout=%.3fs crop=%.3fs mouth=%.3fs restore=%.3fs paste=%.3fs comp=%.3fs total=%.3fs",
                str(self.device),
                int(batch),
                int(width),
                int(height),
                float(face),
                int(len(overlay_entries)),
                str(getattr(self, "_face_aligned_layout_mode", "frame_loop") or "frame_loop"),
                str(self._last_debug_info.get("face_layout_cache", "-")),
                str(self._last_debug_info.get("face_layout_anchor_diff", "-")),
                float(profile.get("face_init_sec", 0.0) or 0.0),
                float(profile.get("layout_sec", 0.0) or 0.0),
                float(profile.get("crop_sec", 0.0) or 0.0),
                float(profile.get("transform_sec", 0.0) or 0.0),
                float(profile.get("restore_sec", 0.0) or 0.0),
                float(profile.get("paste_sec", 0.0) or 0.0),
                float(profile.get("composite_sec", 0.0) or 0.0),
                float(total_sec),
            )
            return result.clamp(0.0, 1.0) if transformed_any else None

    def _run_face_restore_model(
        self,
        face_enhancer: object,
        original_crop: torch.Tensor,
        *,
        face_restore: float,
    ) -> torch.Tensor:
        crop = original_crop
        restore_size = tuple(int(v) for v in original_crop.shape[-2:])
        profile = {
            "input_resize_sec": 0.0,
            "model_sec": 0.0,
            "output_resize_sec": 0.0,
            "chunks": 0,
            "small_crop": 0,
            "restore_h": int(restore_size[0]),
            "restore_w": int(restore_size[1]),
        }
        use_small_crop = bool(self._face_restore_small_crop_enabled) and (
            float(face_restore) <= float(self._face_restore_small_crop_max_strength)
        )
        if use_small_crop:
            target_dim = int(self._face_restore_small_crop_size)
            if min(restore_size) > int(target_dim):
                resize_t0 = time.perf_counter()
                crop = F.interpolate(
                    original_crop,
                    size=(int(target_dim), int(target_dim)),
                    mode="bilinear",
                    align_corners=False,
                )
                profile["input_resize_sec"] = _phase_dt(resize_t0, device=self.device)
                profile["small_crop"] = 1

        def _run_model(inp: torch.Tensor) -> torch.Tensor:
            cudnn_context = (
                torch.backends.cudnn.flags(enabled=bool(self._face_restore_cudnn_enabled))
                if _device_type(self.device) == "cuda"
                else nullcontext()
            )
            with _cuda_device_context(self.device), cudnn_context:
                return getattr(face_enhancer, "_gfpgan")(
                    inp * 2.0 - 1.0,
                    return_rgb=False,
                    randomize_noise=False,
                )[0]

        def _empty_cuda_cache() -> None:
            if _device_type(self.device) != "cuda":
                return
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        def _original_as_model_output(inp: torch.Tensor) -> torch.Tensor:
            return (inp * 2.0) - 1.0

        def _restore_chunk(inp: torch.Tensor) -> torch.Tensor:
            inp = inp.to(device=self.device, dtype=torch.float32, copy=False).contiguous()
            restored_chunk = None
            use_amp = (
                bool(self._face_restore_amp_enabled)
                and (not bool(self._face_restore_amp_failed))
                and _device_type(self.device) == "cuda"
            )
            if use_amp:
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        model_t0 = time.perf_counter()
                        restored_chunk = _run_model(inp)
                        profile["model_sec"] += _phase_dt(model_t0, device=self.device)
                except torch.cuda.OutOfMemoryError as e:
                    self._face_restore_amp_failed = True
                    if not bool(self._logged_face_restore_oom):
                        logging.warning("Post-VAE face AMP path OOM; falling back to fp32 micro-batch: %s", e)
                        self._logged_face_restore_oom = True
                    _empty_cuda_cache()
                    restored_chunk = None
                except Exception as e:
                    self._face_restore_amp_failed = True
                    if not bool(self._logged_face_restore_failure):
                        logging.warning("Post-VAE face AMP path failed; falling back to fp32 micro-batch: %s", e)
                        self._logged_face_restore_failure = True
                    restored_chunk = None
            if restored_chunk is not None:
                profile["chunks"] += 1
                return restored_chunk
            try:
                model_t0 = time.perf_counter()
                out = _run_model(inp)
                profile["model_sec"] += _phase_dt(model_t0, device=self.device)
                profile["chunks"] += 1
                return out
            except torch.cuda.OutOfMemoryError as e:
                if not bool(self._logged_face_restore_oom):
                    logging.warning("Post-VAE face restore OOM; using original face crop for this frame: %s", e)
                    self._logged_face_restore_oom = True
                _empty_cuda_cache()
                return _original_as_model_output(inp)
            except Exception as e:
                if not bool(self._logged_face_restore_failure):
                    logging.warning("Post-VAE face restore failed; using original face crop for this frame: %s", e)
                    self._logged_face_restore_failure = True
                return _original_as_model_output(inp)
            finally:
                if bool(getattr(self, "_face_restore_empty_cache_each_batch", False)):
                    _empty_cuda_cache()

        batch_size = max(1, int(getattr(self, "_face_restore_batch_size", int(crop.shape[0]) or 1)))
        if int(crop.shape[0]) > batch_size:
            restored = torch.cat([_restore_chunk(chunk) for chunk in crop.split(batch_size, dim=0)], dim=0)
        else:
            restored = _restore_chunk(crop)

        restored_01 = (restored.clamp(-1, 1) + 1.0) / 2.0
        if tuple(int(v) for v in restored_01.shape[-2:]) != restore_size:
            resize_t0 = time.perf_counter()
            restored_01 = F.interpolate(
                restored_01,
                size=restore_size,
                mode="bilinear",
                align_corners=False,
            )
            profile["output_resize_sec"] = _phase_dt(resize_t0, device=self.device)
        self._last_debug_info["face_restore_profile"] = dict(profile)
        return restored_01

    def prewarm_face_restore(self, *, height: int, width: int) -> bool:
        if not self.enabled:
            return False
        h = int(max(1, int(height)))
        w = int(max(1, int(width)))
        shape = (h, w)
        if shape in self._face_restore_prewarm_shapes:
            return True

        face_enhancer = self._get_enhancer(mode="face", use_trt=False)
        if face_enhancer is None:
            return False
        self._preallocate(face_enhancer, mode="face", use_trt=False, height=h, width=w)

        try:
            from liveavatar.vendor.enchenh2d.enhancer import _detect_faces_gpu
        except Exception:
            _detect_faces_gpu = None

        try:
            face_h = int(getattr(face_enhancer, "_face_size")[1])
            face_w = int(getattr(face_enhancer, "_face_size")[0])
        except Exception:
            face_h = 512
            face_w = 512
        face_h = int(max(64, face_h))
        face_w = int(max(64, face_w))

        with torch.inference_mode(), _cuda_device_context(self.device):
            dummy_01 = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)

            if _detect_faces_gpu is not None:
                try:
                    _ = _detect_faces_gpu(
                        getattr(face_enhancer, "_face_det"),
                        dummy_01,
                        priors_cache=getattr(face_enhancer, "_priors_cache", None),
                    )
                except Exception:
                    pass

            try:
                dummy_crop = torch.zeros((1, 3, face_h, face_w), dtype=torch.float32, device=self.device)
                face_model = getattr(face_enhancer, "_gfpgan")
                amp_attempted = False
                if (
                    bool(self._face_restore_amp_enabled)
                    and (not bool(self._face_restore_amp_failed))
                    and _device_type(self.device) == "cuda"
                ):
                    amp_attempted = True
                    try:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            _ = face_model(
                                dummy_crop * 2.0 - 1.0,
                                return_rgb=False,
                                randomize_noise=False,
                            )[0]
                    except Exception as e:
                        self._face_restore_amp_failed = True
                        logging.warning(
                            "Post-VAE face AMP prewarm failed; disabling AMP before live path: %s",
                            e,
                        )
                if not bool(amp_attempted) or bool(self._face_restore_amp_failed):
                    _ = face_model(
                        dummy_crop * 2.0 - 1.0,
                        return_rgb=False,
                        randomize_noise=False,
                    )[0]
            except Exception:
                pass

            try:
                M = torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=torch.float32,
                    device=self.device,
                )
                _ = kornia.geometry.transform.warp_affine(
                    dummy_01,
                    M,
                    (face_h, face_w),
                    mode="bilinear",
                    padding_mode="zeros",
                )
                mode = self._normalize_face_mask_mode(getattr(self, "_face_mask_mode", "inner_square"))
                if mode in {"ellipse", "oval"}:
                    inner = self._aligned_face_mask(height=int(face_h), width=int(face_w), device=self.device)
                elif mode == "inner_square":
                    inner = self._inner_square_face_mask(height=int(face_h), width=int(face_w), device=self.device)
                else:
                    inner = self._full_square_face_mask(height=int(face_h), width=int(face_w), device=self.device)
                mask = kornia.geometry.transform.warp_affine(
                    inner,
                    M,
                    (h, w),
                    mode="bilinear",
                    padding_mode="zeros",
                )
                if mode in {"ellipse", "inner_square"}:
                    _ = kornia.filters.gaussian_blur2d(mask, (51, 51), (17.0, 17.0))
                else:
                    _ = self._soften_square_face_mask(mask)
            except Exception:
                pass

        self._face_restore_prewarm_shapes.add(shape)
        if not self._logged_face_prewarm:
            logging.info(
                "Post-VAE face restore prewarm ready: device=%s size=%dx%d",
                str(self.device),
                int(w),
                int(h),
            )
            self._logged_face_prewarm = True
        return True

    def restore_known_face_crops_tchw(
        self,
        crops_01: torch.Tensor,
        *,
        face_restore: float,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        if crops_01 is None:
            return None
        if crops_01.ndim != 4 or int(crops_01.shape[1]) != 3:
            raise ValueError(f"Expected known face crops [T,3,H,W], got {tuple(crops_01.shape)}")
        face = _clamp01(float(face_restore or 0.0))
        if face <= 0.0:
            return crops_01

        with torch.inference_mode(), _cuda_device_context(self.device):
            face_enhancer = self._get_enhancer(mode="face", use_trt=False)
            if face_enhancer is None:
                self._last_debug_info.update({"known_face_restore": "no_face_enhancer"})
                return None
            crops_01 = crops_01.to(device=self.device, dtype=torch.float32, copy=False).contiguous().clamp(0.0, 1.0)
            _, _, height, width = (int(v) for v in crops_01.shape)
            self._preallocate(face_enhancer, mode="face", use_trt=False, height=int(height), width=int(width))
            started = time.perf_counter()
            restored_01 = self._run_face_restore_model(
                face_enhancer,
                crops_01,
                face_restore=float(face),
            )
            if restored_01 is None:
                return None
            if face < 1.0:
                restored_01 = restored_01 * face + crops_01 * (1.0 - face)
            elapsed = _phase_dt(started, device=self.device)
            self._last_debug_info.update(
                {
                    "known_face_restore": "applied",
                    "known_face_restore_frames": int(crops_01.shape[0]),
                    "known_face_restore_size": (int(width), int(height)),
                    "known_face_restore_sec": float(elapsed),
                }
            )
            return restored_01.clamp(0.0, 1.0)

    def prewarm_live_shape(self, *, height: int, width: int, face_restore: float, background_restore: float) -> bool:
        if not self.enabled:
            return False
        h = int(max(1, int(height)))
        w = int(max(1, int(width)))
        face = _clamp01(float(face_restore))
        background = _clamp01(float(background_restore))
        warmed = False

        if face > 0.0:
            warmed = bool(self.prewarm_face_restore(height=h, width=w)) or bool(warmed)

        if background > 0.0:
            use_trt = self._should_use_trt(height=h, width=w)
            bg_enhancer, use_trt = self._get_background_enhancer(use_trt=bool(use_trt))
            if bg_enhancer is not None:
                self._preallocate(bg_enhancer, mode="upscale", use_trt=bool(use_trt), height=h, width=w)
                try:
                    with torch.inference_mode():
                        dummy = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)
                        _ = self._enhance_background_x2(
                            bg_enhancer,
                            dummy,
                            x2_h=h * 2,
                            x2_w=w * 2,
                            clip_id=0,
                        )
                    warmed = True
                except Exception as e:
                    self._enhancer_failures.add(("upscale", 1 if bool(use_trt) else 0))
                    if bool(use_trt):
                        fallback, fallback_trt = self._get_background_enhancer(use_trt=False)
                        if fallback is not None:
                            self._preallocate(fallback, mode="upscale", use_trt=bool(fallback_trt), height=h, width=w)
                            try:
                                with torch.inference_mode():
                                    dummy = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)
                                    _ = self._enhance_background_x2(
                                        fallback,
                                        dummy,
                                        x2_h=h * 2,
                                        x2_w=w * 2,
                                        clip_id=0,
                                    )
                                warmed = True
                            except Exception as fallback_exc:
                                self._enhancer_failures.add(("upscale", 0))
                                if not self._logged_background_failure:
                                    logging.warning(
                                        "Post-VAE background prewarm failed; live path will use bicubic fallback: %s",
                                        fallback_exc,
                                    )
                                    self._logged_background_failure = True
                        elif not self._logged_background_failure:
                            logging.warning(
                                "Post-VAE background prewarm failed; live path will use bicubic fallback: %s",
                                e,
                            )
                            self._logged_background_failure = True
                    elif not self._logged_background_failure:
                        logging.warning(
                            "Post-VAE background prewarm failed; live path will use bicubic fallback: %s",
                            e,
                        )
                        self._logged_background_failure = True

        if warmed:
            logging.warning(
                "Post-VAE live prewarm ready: device=%s size=%dx%d face=%.2f background=%.2f",
                str(self.device),
                int(w),
                int(h),
                float(face),
                float(background),
            )
        return bool(warmed)

    def enhance_batch_tchw(
        self,
        frames_tchw: torch.Tensor,
        *,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> torch.Tensor | None:
        if not self.enabled:
            self._last_debug_info = {"enabled": False, "return": "disabled"}
            return None
        if frames_tchw is None:
            self._last_debug_info = {"enabled": True, "return": "missing_input"}
            return None
        if frames_tchw.ndim != 4 or int(frames_tchw.shape[1]) != 3:
            raise ValueError(f"Expected [T,3,H,W], got {tuple(frames_tchw.shape)}")
        settings = self._settings
        if not settings.active:
            self._last_debug_info = {
                "enabled": True,
                "active": False,
                "return": "inactive_settings",
                "face_restore": float(settings.face_restore),
                "background_restore": float(settings.background_restore),
            }
            return None
        total_t0 = time.perf_counter()
        deep_timing = bool(deep_timing_enabled())
        self._last_debug_info = {
            "enabled": True,
            "active": True,
            "return": "started",
            "input_shape": tuple(int(v) for v in frames_tchw.shape),
            "input_dtype": str(frames_tchw.dtype),
            "input_device": str(frames_tchw.device),
            "face_restore": float(settings.face_restore),
            "background_restore": float(settings.background_restore),
            "face_enabled": bool(settings.face_enabled),
            "background_enabled": bool(settings.background_enabled),
            "requested_output": (
                None if output_height is None else int(output_height),
                None if output_width is None else int(output_width),
            ),
        }

        if not self._logged_first_invoke:
            logging.info(
                "Post-VAE enhancer invoked: shape=%s dtype=%s device=%s face_restore=%.2f background_restore=%.2f",
                tuple(int(v) for v in frames_tchw.shape),
                str(frames_tchw.dtype),
                str(frames_tchw.device),
                float(settings.face_restore),
                float(settings.background_restore),
            )
            self._logged_first_invoke = True

        with torch.inference_mode(), _cuda_device_context(self.device):
            phase_profile: dict[str, float | int | str] = {
                "phase_sync": 1 if bool(_post_vae_phase_sync_enabled()) else 0,
            }
            base_t0 = time.perf_counter()
            base_01 = _finite_clamp01((frames_tchw.clamp(-1, 1) + 1.0) / 2.0)
            phase_profile["input_normalize_sec"] = _phase_dt(base_t0, device=self.device)
            _, _, height, width = (int(v) for v in frames_tchw.shape)
            use_x2 = bool(getattr(self, "_upscale_x2_enabled", True))
            layer_h = int(height) * 2 if bool(use_x2) else int(height)
            layer_w = int(width) * 2 if bool(use_x2) else int(width)
            affine_scale = 2.0 if bool(use_x2) else 1.0
            clip_id = int(self._face_chunk_seq) + 1
            self._face_chunk_seq = int(clip_id)

            face_source_x2 = bool(getattr(self, "_face_source_x2_enabled", False)) and bool(use_x2)
            face_stage = self._normalize_face_restore_stage(getattr(self, "_face_restore_stage", "native_first"))

            face_enhancer = None
            overlay_entries: list[dict[str, torch.Tensor]] = []
            face_applied_native = False
            phase_profile["face_init_sec"] = 0.0
            phase_profile["face_layout_sec"] = 0.0
            phase_profile["face_overlay_sec"] = 0.0
            if settings.face_enabled:
                face_init_t0 = time.perf_counter()
                face_enhancer = self._get_enhancer(mode="face", use_trt=False)
                if face_enhancer is None:
                    self._last_debug_info.update({"return": "no_face_enhancer"})
                    return None
                face_prealloc_h = int(height) if face_stage == "native_first" else (int(layer_h) if bool(face_source_x2) else int(height))
                face_prealloc_w = int(width) if face_stage == "native_first" else (int(layer_w) if bool(face_source_x2) else int(width))
                self._preallocate(
                    face_enhancer,
                    mode="face",
                    use_trt=False,
                    height=int(face_prealloc_h),
                    width=int(face_prealloc_w),
                )
                phase_profile["face_init_sec"] = _phase_dt(face_init_t0, device=self.device)

                face_layout_t0 = time.perf_counter()
                if face_stage == "native_first":
                    overlay_entries = self._get_face_overlay_entries(
                        face_enhancer,
                        base_01,
                        clip_id=int(clip_id),
                        output_height=int(height),
                        output_width=int(width),
                        affine_scale=1.0,
                        mask_mode=str(getattr(self, "_face_mask_mode", "inner_square") or "inner_square"),
                        layout_mode=str(getattr(self, "_face_aligned_layout_mode", "frame_loop") or "frame_loop"),
                    )
                    phase_profile["face_layout_sec"] = _phase_dt(face_layout_t0, device=self.device)
                    native_overlay_t0 = time.perf_counter()
                    base_01 = self._apply_face_overlay_batch(
                        base_01,
                        base_01=base_01,
                        face_enhancer=face_enhancer,
                        overlay_mode="restored",
                        face_restore=float(settings.face_restore),
                        clip_id=int(clip_id),
                        overlay_entries=overlay_entries,
                    )
                    phase_profile["face_overlay_sec"] = _phase_dt(native_overlay_t0, device=self.device)
                    face_applied_native = bool(overlay_entries)
                    overlay_entries = []
                else:
                    # Detect/crop only when face restore is enabled. The post-VAE
                    # stage can opt into x2 source, but native_first is safer for
                    # file renders because it avoids inverse-warping a 512 crop
                    # directly into the enlarged layer.
                    face_source_01 = None
                    face_affine_scale = 1.0 if bool(face_source_x2) else float(affine_scale)
                    overlay_entries = []
                    phase_profile["face_layout_sec"] = _phase_dt(face_layout_t0, device=self.device)
            layer_t0 = time.perf_counter()
            if bool(use_x2):
                base_layer = F.interpolate(
                    base_01,
                    size=(int(layer_h), int(layer_w)),
                    mode="bicubic",
                    align_corners=False,
                )
                base_layer = _finite_clamp01(base_layer)
            else:
                base_layer = base_01
            phase_profile["bicubic_x2_sec"] = _phase_dt(layer_t0, device=self.device)
            phase_profile["post_vae_x2"] = 1 if bool(use_x2) else 0
            face_source_01 = base_layer if bool(face_source_x2) else base_01
            background_input_tchw = (base_01 * 2.0 - 1.0).contiguous() if bool(face_applied_native) else frames_tchw
            if settings.face_enabled and face_stage == "post_vae" and face_enhancer is not None:
                face_layout_t0 = time.perf_counter()
                face_affine_scale = 1.0 if bool(face_source_x2) else float(affine_scale)
                overlay_entries = self._get_face_overlay_entries(
                    face_enhancer,
                    face_source_01,
                    clip_id=int(clip_id),
                    output_height=int(layer_h),
                    output_width=int(layer_w),
                    affine_scale=float(face_affine_scale),
                    mask_mode=str(getattr(self, "_face_mask_mode", "inner_square") or "inner_square"),
                    layout_mode=str(getattr(self, "_face_aligned_layout_mode", "frame_loop") or "frame_loop"),
                )
                phase_profile["face_layout_sec"] = _phase_dt(face_layout_t0, device=self.device)
            self._last_debug_info.update(
                {
                    "render_size": (int(width), int(height)),
                    "post_vae_layer_size": (int(layer_w), int(layer_h)),
                    "post_vae_upscale_x2": bool(use_x2),
                    "face_source": "native_first" if bool(face_applied_native) else ("x2" if bool(face_source_x2) else "native"),
                    "face_restore_stage": str(face_stage),
                    "face_applied_native": bool(face_applied_native),
                    "clip_id": int(clip_id),
                    "face_entries": int(len(overlay_entries)) if not bool(face_applied_native) else 1,
                }
            )

            if settings.background_enabled:
                use_trt = self._should_use_trt(height=int(height), width=int(width))
                bg_init_t0 = time.perf_counter()
                background_mode = "upscale" if bool(use_x2) else "enhance"
                background_mode_label = "upscale_x2" if bool(use_x2) else "enhance_native"
                if bool(use_x2):
                    bg_enhancer, use_trt = self._get_background_enhancer(use_trt=bool(use_trt))
                else:
                    bg_enhancer = self._get_enhancer(mode="enhance", use_trt=bool(use_trt))
                    if bg_enhancer is None and bool(use_trt) and bool(self._background_pytorch_fallback_enabled):
                        bg_enhancer = self._get_enhancer(mode="enhance", use_trt=False)
                        use_trt = False
                self._last_debug_info.update({"background_trt": bool(use_trt)})
                if bg_enhancer is None:
                    if not settings.face_enabled:
                        self._last_debug_info.update({"return": "no_background_enhancer"})
                        return None
                    enhanced_01 = base_layer
                    self._last_debug_info.update({"background_mode": "bicubic_no_enhancer"})
                    use_trt = False
                else:
                    self._preallocate(
                        bg_enhancer,
                        mode=str(background_mode),
                        use_trt=bool(use_trt),
                        height=int(height),
                        width=int(width),
                    )
                    phase_profile["background_init_sec"] = _phase_dt(bg_init_t0, device=self.device)
                    bg_t0 = time.perf_counter()
                    try:
                        if bool(use_x2):
                            enhanced_01 = self._enhance_background_x2(
                                bg_enhancer,
                                background_input_tchw,
                                x2_h=int(layer_h),
                                x2_w=int(layer_w),
                                clip_id=int(clip_id),
                            )
                        else:
                            enhanced_01 = bg_enhancer.enhance_gpu(
                                background_input_tchw,
                                out_h=int(height),
                                out_w=int(width),
                                clip_id=int(clip_id),
                                frame_idx=0,
                            ).clamp(0, 1)
                        phase_profile["background_enhance_sec"] = _phase_dt(bg_t0, device=self.device)
                        if (
                            bool(use_trt)
                            and bool(self._trt_finite_check_enabled)
                            and not bool(torch.isfinite(enhanced_01).all().item())
                        ):
                            raise RuntimeError("Post-VAE background TensorRT returned non-finite values")
                        enhanced_01 = _finite_clamp01(enhanced_01)
                        log_phase_timing(
                            "post_vae",
                            "background_enhance",
                            bg_t0,
                            enabled=bool(deep_timing),
                            sync_device=self.device,
                            device=str(self.device),
                            frames=int(frames_tchw.shape[0]),
                            size=f"{int(width)}x{int(height)}",
                            trt=1 if bool(use_trt) else 0,
                            batch=int(getattr(self, "_background_batch_size", 1)),
                        )
                        bg_strength = float(settings.background_restore)
                        if bg_strength < 1.0:
                            bg_blend_t0 = time.perf_counter()
                            enhanced_01 = (enhanced_01 * bg_strength) + (base_layer * (1.0 - bg_strength))
                            phase_profile["background_blend_sec"] = _phase_dt(bg_blend_t0, device=self.device)
                        self._last_debug_info.update(
                            {
                                "background_mode": str(background_mode_label),
                                "background_trt": bool(use_trt),
                                "background_strength": float(bg_strength),
                            }
                        )
                    except Exception as e:
                        if not bool(use_trt):
                            self._enhancer_failures.add((str(background_mode), 0))
                        self._last_debug_info.update(
                            {
                                "background_mode": "bicubic_after_error",
                                "background_error": str(e),
                            }
                        )
                        if not self._logged_background_failure:
                            logging.exception(
                                "Post-VAE background enhance failed; using base layer and keeping face restore: %s",
                                e,
                            )
                            self._logged_background_failure = True
                        if not settings.face_enabled:
                            self._last_debug_info.update({"return": "background_error_no_face"})
                            return None
                        enhanced_01 = base_layer
                overlay_t0 = time.perf_counter()
                face_overlay_pending = bool(settings.face_enabled and not bool(face_applied_native))
                overlay_mode = "restored" if bool(face_overlay_pending) else "native_first" if bool(face_applied_native) else "none"
                if face_overlay_pending:
                    out = self._apply_face_overlay_batch(
                        enhanced_01,
                        base_01=face_source_01,
                        face_enhancer=face_enhancer,
                        overlay_mode=overlay_mode,
                        face_restore=float(settings.face_restore),
                        clip_id=int(clip_id),
                        overlay_entries=overlay_entries,
                    )
                else:
                    out = enhanced_01
                phase_profile["face_overlay_sec"] = float(phase_profile.get("face_overlay_sec", 0.0) or 0.0) + _phase_dt(
                    overlay_t0, device=self.device
                )
                if face_overlay_pending:
                    log_phase_timing(
                        "post_vae",
                        "face_overlay",
                        overlay_t0,
                        enabled=bool(deep_timing),
                        sync_device=self.device,
                        device=str(self.device),
                        frames=int(frames_tchw.shape[0]),
                        size=f"{int(width)}x{int(height)}",
                        overlay=str(overlay_mode),
                    )
                if settings.face_enabled and not self._logged_first_face_overlay:
                    logging.info(
                        "Post-VAE enhancer face overlay ready: clip=%d overlay=%s shape=%s",
                        int(clip_id),
                        str(overlay_mode),
                        tuple(int(v) for v in out.shape),
                    )
                    self._logged_first_face_overlay = True
                resize_t0 = time.perf_counter()
                out = self._resize_cover_crop_to_output(
                    out,
                    out_h=int(output_height or height),
                    out_w=int(output_width or width),
                )
                phase_profile["output_resize_sec"] = _phase_dt(resize_t0, device=self.device)
                total_sec = max(0.0, time.perf_counter() - total_t0)
                phase_profile["total_sec"] = float(total_sec)
                self._last_debug_info["phase_profile"] = dict(phase_profile)
                self._last_debug_info.update(
                    {
                        "return": "ok",
                        "overlay_mode": str(overlay_mode),
                        "face_overlay_requested": bool(settings.face_enabled),
                        "face_overlay_applied": bool(face_applied_native or (settings.face_enabled and overlay_entries)),
                        "output_shape": tuple(int(v) for v in out.shape),
                        "dt_sec": float(total_sec),
                    }
                )
                if bool(_post_vae_phase_timing_enabled()) or float(total_sec) >= 0.75:
                    logging.warning(
                        "Post-VAE phase profile: device=%s frames=%d size=%dx%d out=%dx%d face=%.2f bg=%.2f trt=%d x2=%d face_source=%s face_restore_crop=%dx%d face_small=%d sync=%d face_stride=%d face_detect_frames=%d face_reused=%d face_entries=%d mask_mode=%s input=%.3fs face_init=%.3fs face_layout=%.3fs face_detect=%.3fs face_mask=%.3fs bicubic=%.3fs bg_init=%.3fs bg=%.3fs bg_blend=%.3fs face_overlay=%.3fs face_crop=%.3fs face_model=%.3fs face_paste=%.3fs face_comp=%.3fs resize=%.3fs total=%.3fs",
                        str(self.device),
                        int(frames_tchw.shape[0]),
                        int(width),
                        int(height),
                        int(out.shape[3]),
                        int(out.shape[2]),
                        float(settings.face_restore),
                        float(settings.background_restore),
                        1 if bool(use_trt) else 0,
                        int(phase_profile.get("post_vae_x2", 0) or 0),
                        str(self._last_debug_info.get("face_source") or "native"),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("restore_w", 0) or 0),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("restore_h", 0) or 0),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("small_crop", 0) or 0),
                        int(phase_profile.get("phase_sync", 0) or 0),
                        int(self._last_debug_info.get("face_detect_stride", 1) or 1),
                        int(self._last_debug_info.get("face_detect_frames", int(frames_tchw.shape[0])) or 0),
                        int(self._last_debug_info.get("face_detect_reused_frames", 0) or 0),
                        int(len(overlay_entries)) if not bool(face_applied_native) else 1,
                        str(self._last_debug_info.get("face_mask_mode") or "-"),
                        float(phase_profile.get("input_normalize_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_init_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_layout_sec", 0.0) or 0.0),
                        float(self._last_debug_info.get("face_detect_sec", 0.0) or 0.0),
                        float(self._last_debug_info.get("face_mask_sec", 0.0) or 0.0),
                        float(phase_profile.get("bicubic_x2_sec", 0.0) or 0.0),
                        float(phase_profile.get("background_init_sec", 0.0) or 0.0),
                        float(phase_profile.get("background_enhance_sec", 0.0) or 0.0),
                        float(phase_profile.get("background_blend_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_overlay_sec", 0.0) or 0.0),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("crop_sec", 0.0)),
                        float((self._last_debug_info.get("face_restore_profile") or {}).get("model_sec", 0.0)),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("paste_sec", 0.0)),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("composite_sec", 0.0)),
                        float(phase_profile.get("output_resize_sec", 0.0) or 0.0),
                        float(total_sec),
                    )
                if post_vae_timing_enabled():
                    logging.info(
                        "Post-VAE timing: device=%s frames=%d size=%dx%d out=%dx%d face=%.2f bg=%.2f bg_mode=%s overlay=%s trt=%d dt=%.3fs",
                        str(self.device),
                        int(frames_tchw.shape[0]),
                        int(width),
                        int(height),
                        int(out.shape[3]),
                        int(out.shape[2]),
                        float(settings.face_restore),
                        float(settings.background_restore),
                        str(background_mode_label),
                        str(overlay_mode),
                        1 if bool(use_trt) else 0,
                        float(time.perf_counter() - total_t0),
                    )
                return out

            if settings.face_enabled:
                face_only_t0 = time.perf_counter()
                if bool(face_applied_native):
                    overlay_mode = "native_first"
                    out = base_layer
                    phase_profile["face_overlay_sec"] = float(phase_profile.get("face_overlay_sec", 0.0) or 0.0) + _phase_dt(
                        face_only_t0, device=self.device
                    )
                else:
                    overlay_mode = "restored"
                    out = self._apply_face_overlay_batch(
                        base_layer,
                        base_01=face_source_01,
                        face_enhancer=face_enhancer,
                        overlay_mode=overlay_mode,
                        face_restore=float(settings.face_restore),
                        clip_id=int(clip_id),
                        overlay_entries=overlay_entries,
                    )
                    phase_profile["face_overlay_sec"] = _phase_dt(face_only_t0, device=self.device)
                    log_phase_timing(
                        "post_vae",
                        "face_restore_only",
                        face_only_t0,
                        enabled=bool(deep_timing),
                        sync_device=self.device,
                        device=str(self.device),
                        frames=int(frames_tchw.shape[0]),
                        size=f"{int(width)}x{int(height)}",
                    )
                resize_t0 = time.perf_counter()
                out = self._resize_cover_crop_to_output(
                    out,
                    out_h=int(output_height or height),
                    out_w=int(output_width or width),
                )
                phase_profile["output_resize_sec"] = _phase_dt(resize_t0, device=self.device)
                total_sec = max(0.0, time.perf_counter() - total_t0)
                phase_profile["total_sec"] = float(total_sec)
                self._last_debug_info["phase_profile"] = dict(phase_profile)
                self._last_debug_info.update(
                    {
                        "return": "ok",
                        "background_mode": "none",
                        "background_trt": False,
                        "overlay_mode": str(overlay_mode),
                        "face_overlay_requested": True,
                        "face_overlay_applied": bool(face_applied_native or overlay_entries),
                        "output_shape": tuple(int(v) for v in out.shape),
                        "dt_sec": float(total_sec),
                    }
                )
                if bool(_post_vae_phase_timing_enabled()) or float(total_sec) >= 0.75:
                    logging.warning(
                        "Post-VAE phase profile: device=%s frames=%d size=%dx%d out=%dx%d face=%.2f bg=%.2f trt=0 x2=%d face_source=%s face_restore_crop=%dx%d face_small=%d sync=%d face_stride=%d face_detect_frames=%d face_reused=%d face_entries=%d mask_mode=%s input=%.3fs face_init=%.3fs face_layout=%.3fs face_detect=%.3fs face_mask=%.3fs bicubic=%.3fs bg_init=0.000s bg=0.000s bg_blend=0.000s face_overlay=%.3fs face_crop=%.3fs face_model=%.3fs face_paste=%.3fs face_comp=%.3fs resize=%.3fs total=%.3fs",
                        str(self.device),
                        int(frames_tchw.shape[0]),
                        int(width),
                        int(height),
                        int(out.shape[3]),
                        int(out.shape[2]),
                        float(settings.face_restore),
                        float(settings.background_restore),
                        int(phase_profile.get("post_vae_x2", 0) or 0),
                        str(self._last_debug_info.get("face_source") or "native"),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("restore_w", 0) or 0),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("restore_h", 0) or 0),
                        int((self._last_debug_info.get("face_restore_profile") or {}).get("small_crop", 0) or 0),
                        int(phase_profile.get("phase_sync", 0) or 0),
                        int(self._last_debug_info.get("face_detect_stride", 1) or 1),
                        int(self._last_debug_info.get("face_detect_frames", int(frames_tchw.shape[0])) or 0),
                        int(self._last_debug_info.get("face_detect_reused_frames", 0) or 0),
                        int(len(overlay_entries)) if not bool(face_applied_native) else 1,
                        str(self._last_debug_info.get("face_mask_mode") or "-"),
                        float(phase_profile.get("input_normalize_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_init_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_layout_sec", 0.0) or 0.0),
                        float(self._last_debug_info.get("face_detect_sec", 0.0) or 0.0),
                        float(self._last_debug_info.get("face_mask_sec", 0.0) or 0.0),
                        float(phase_profile.get("bicubic_x2_sec", 0.0) or 0.0),
                        float(phase_profile.get("face_overlay_sec", 0.0) or 0.0),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("crop_sec", 0.0)),
                        float((self._last_debug_info.get("face_restore_profile") or {}).get("model_sec", 0.0)),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("paste_sec", 0.0)),
                        float((self._last_debug_info.get("face_overlay_profile") or {}).get("composite_sec", 0.0)),
                        float(phase_profile.get("output_resize_sec", 0.0) or 0.0),
                        float(total_sec),
                    )
                if post_vae_timing_enabled():
                    logging.info(
                        "Post-VAE timing: device=%s frames=%d size=%dx%d out=%dx%d face=%.2f bg=%.2f bg_mode=- overlay=%s trt=0 dt=%.3fs",
                        str(self.device),
                        int(frames_tchw.shape[0]),
                        int(width),
                        int(height),
                        int(out.shape[3]),
                        int(out.shape[2]),
                        float(settings.face_restore),
                        float(settings.background_restore),
                        str(overlay_mode),
                        float(time.perf_counter() - total_t0),
                    )
                return out

            self._last_debug_info.update({"return": "no_enabled_restore_branch"})
        return None
