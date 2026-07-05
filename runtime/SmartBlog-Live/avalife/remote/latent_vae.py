from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Any


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class CudaReadyTensor:
    """Tensor plus a CUDA event that marks when the producing stream is done."""

    __slots__ = ("tensor", "ready_event", "ready_device")

    def __init__(self, tensor: Any, ready_event: Any | None = None, ready_device: str = "") -> None:
        self.tensor = tensor
        self.ready_event = ready_event
        self.ready_device = str(ready_device or "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tensor, name)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(0.0, min(1.0, float(parsed)))


def _env_float01(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    return _clamp01(raw if raw not in {None, ""} else default, default=default)


def _timing_enabled() -> bool:
    return _env_flag("POST_VAE_TIMING_LOG", "0") or _env_flag("REMOTE_EDGE_VAE_PHASE_TIMING", "1")


def _vae_phase_sync_enabled() -> bool:
    return _env_flag("REMOTE_EDGE_VAE_PHASE_SYNC", "0")


def _dtype_from_env(raw: str) -> Any:
    import torch

    value = str(raw or "").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return torch.bfloat16


def _post_vae_device_from_env(default: str) -> str:
    raw = str(
        os.getenv("REMOTE_EDGE_POST_VAE_DEVICE")
        or os.getenv("LIVE_RAW_POST_VAE_ENHANCER_DEVICE")
        or "auto"
    ).strip()
    if not raw or raw.lower() in {"auto", "default", "same"}:
        return str(default)
    return raw


def _parse_shape(raw: str) -> tuple[int, ...]:
    value = str(raw or "").strip()
    if not value:
        raise RuntimeError("raw latent payload requires shape")
    for ch in "()[]":
        value = value.replace(ch, "")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise RuntimeError(f"invalid raw latent shape: {raw!r}")
    shape = tuple(int(part) for part in parts)
    if any(int(dim) <= 0 for dim in shape):
        raise RuntimeError(f"invalid raw latent shape: {raw!r}")
    return shape


def _tensor_01_to_rgb24_frames(tensor_01: Any) -> list[bytes]:
    import torch

    tensor_01 = getattr(tensor_01, "tensor", tensor_01)
    if not torch.is_tensor(tensor_01):
        raise TypeError("expected decoded frames tensor")
    if int(tensor_01.ndim) != 4:
        raise ValueError(f"expected decoded frames as T,C,H,W tensor, got {tuple(tensor_01.shape)}")
    if int(tensor_01.shape[0]) <= 0:
        return []
    rgb = (tensor_01.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    rgb = rgb.permute(0, 2, 3, 1).contiguous()
    rgb_cpu = rgb.cpu().numpy()
    return [bytes(memoryview(frame).cast("B")) for frame in rgb_cpu]


class WanLatentDecoder:
    """Decode remote Wan S2V VAE latents into RGB24 frame buffers on the edge GPU."""

    def __init__(self) -> None:
        import torch
        from liveavatar.models.wan.wan_2_2.modules.vae_streaming import WanVAE

        self.torch = torch
        self.device = str(os.getenv("REMOTE_EDGE_LATENT_DEVICE", "cuda:0") or "cuda:0")
        self.dtype = _dtype_from_env(str(os.getenv("REMOTE_EDGE_LATENT_DTYPE", "bf16") or "bf16"))
        ckpt = str(os.getenv("REMOTE_EDGE_LATENT_VAE_CHECKPOINT", "") or "").strip()
        if not ckpt:
            asset_root = str(os.getenv("WORKER_ASSET_ROOT", os.getcwd()) or os.getcwd()).strip()
            ckpt_dir = str(
                os.getenv("CKPT_DIR", os.path.join(asset_root, "ckpt", "Wan2.2-S2V-14B")) or ""
            ).strip()
            ckpt = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"missing Wan VAE checkpoint: {ckpt}")
        self.keep_last_frames_default = max(0, int(os.getenv("REMOTE_EDGE_LATENT_KEEP_LAST_FRAMES", "0") or 0))
        self.resize_to_livekit = _env_flag("REMOTE_EDGE_LATENT_RESIZE_TO_LIVEKIT", "1")
        self.post_vae_enabled = _env_flag("REMOTE_EDGE_POST_VAE_ENHANCER", "1")
        self.default_face_restore = _env_float01("REMOTE_EDGE_FACE_RESTORE", 0.0)
        self.default_background_restore = _env_float01("REMOTE_EDGE_BACKGROUND_RESTORE", 0.0)
        self.max_face_restore = _env_float01("REMOTE_EDGE_FACE_RESTORE_MAX", 1.0)
        self.max_background_restore = _env_float01("REMOTE_EDGE_BACKGROUND_RESTORE_MAX", 1.0)
        self.post_vae_device = _post_vae_device_from_env(self.device)
        self.post_vae = None
        self._last_restore_log_key: tuple[float, float, bool] | None = None
        self._logged_post_vae_disabled = False
        self._decode_block_count = 0
        self._debug_capture_count = 0
        self._decode_lock = threading.RLock()
        self._post_vae_lock = threading.RLock()
        self.vae = WanVAE(vae_pth=ckpt, dtype=self.dtype, device=self.device)
        logging.warning(
            "Remote latent VAE ready: ckpt=%s device=%s dtype=%s post_vae=%d post_vae_device=%s",
            ckpt,
            self.device,
            self.dtype,
            1 if bool(self.post_vae_enabled) else 0,
            self.post_vae_device,
        )

    def _resize_cover_crop_frames_01(self, frames_01: Any, *, target_h: int, target_w: int) -> Any:
        target_h = int(target_h or 0)
        target_w = int(target_w or 0)
        if target_h <= 0 or target_w <= 0:
            return frames_01
        if int(frames_01.shape[-2]) == target_h and int(frames_01.shape[-1]) == target_w:
            return frames_01.contiguous()
        import torch.nn.functional as F

        src_h = int(frames_01.shape[-2])
        src_w = int(frames_01.shape[-1])
        scale = max(float(target_h) / float(max(1, src_h)), float(target_w) / float(max(1, src_w)))
        new_h = max(1, int(round(float(src_h) * float(scale))))
        new_w = max(1, int(round(float(src_w) * float(scale))))
        out = F.interpolate(frames_01, size=(int(new_h), int(new_w)), mode="bicubic", align_corners=False)
        top = max(0, (int(new_h) - int(target_h)) // 2)
        left = max(0, (int(new_w) - int(target_w)) // 2)
        return out[..., int(top) : int(top) + int(target_h), int(left) : int(left) + int(target_w)].contiguous()

    @staticmethod
    def _int_env(name: str, default: int, *, low: int = 0, high: int = 1_000_000) -> int:
        try:
            value = int(str(os.getenv(name, str(default)) or str(default)).strip())
        except Exception:
            value = int(default)
        return max(int(low), min(int(high), int(value)))

    def _maybe_capture_post_vae_debug(
        self,
        *,
        block_index: int,
        baseline_01: Any,
        final_01: Any,
        tensor_shape: tuple[int, ...],
        codec: str,
        face_restore: float,
        background_restore: float,
        post_vae_requested: bool,
        enhanced_returned: bool,
        target_h: int,
        target_w: int,
        debug_info: dict[str, object] | None,
        decode_dt_sec: float,
    ) -> None:
        if not _env_flag("REMOTE_EDGE_POST_VAE_DEBUG_CAPTURE", "0"):
            return
        max_captures = self._int_env("REMOTE_EDGE_POST_VAE_DEBUG_MAX", 6, low=1, high=500)
        if int(self._debug_capture_count) >= int(max_captures):
            return
        every = self._int_env("REMOTE_EDGE_POST_VAE_DEBUG_EVERY_BLOCKS", 1, low=1, high=100_000)
        if int(block_index) % int(every) != 0:
            return
        try:
            import json
            from pathlib import Path

            import cv2
            import numpy as np

            torch = self.torch
            baseline = baseline_01
            if self.resize_to_livekit and int(target_h) > 0 and int(target_w) > 0:
                baseline = self._resize_cover_crop_frames_01(
                    baseline,
                    target_h=int(target_h),
                    target_w=int(target_w),
                ).clamp(0.0, 1.0)
            final = final_01.clamp(0.0, 1.0)
            if tuple(int(v) for v in baseline.shape) != tuple(int(v) for v in final.shape):
                baseline = self._resize_cover_crop_frames_01(
                    baseline,
                    target_h=int(final.shape[-2]),
                    target_w=int(final.shape[-1]),
                ).clamp(0.0, 1.0)

            frame_count = int(final.shape[0])
            frame_index = min(
                max(0, self._int_env("REMOTE_EDGE_POST_VAE_DEBUG_FRAME_INDEX", 0, low=0, high=10_000)),
                max(0, frame_count - 1),
            )
            before = baseline[frame_index : frame_index + 1].detach().to(dtype=torch.float32).clamp(0.0, 1.0)
            after = final[frame_index : frame_index + 1].detach().to(dtype=torch.float32).clamp(0.0, 1.0)
            diff = (after - before).abs()
            flat = diff.flatten()
            if int(flat.numel()) > 0:
                p95 = float(torch.quantile(flat, 0.95).detach().cpu().item())
                mad = float(diff.mean().detach().cpu().item())
                max_abs = float(diff.max().detach().cpu().item())
            else:
                p95 = 0.0
                mad = 0.0
                max_abs = 0.0

            out_dir = Path(
                str(os.getenv("REMOTE_EDGE_POST_VAE_DEBUG_DIR", "/tmp/smartblog-postvae-debug") or "").strip()
                or "/tmp/smartblog-postvae-debug"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            capture_id = int(self._debug_capture_count) + 1
            stem = f"block_{int(block_index):06d}_capture_{capture_id:03d}_frame_{int(frame_index):03d}"

            def write_png(path: Path, tensor_01: Any) -> None:
                arr = (
                    tensor_01[0]
                    .detach()
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
                cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

            write_png(out_dir / f"{stem}_before.png", before)
            write_png(out_dir / f"{stem}_after.png", after)
            scale = max(1.0e-6, max_abs)
            write_png(out_dir / f"{stem}_diff.png", (diff / scale).clamp(0.0, 1.0))

            metrics = {
                "block_index": int(block_index),
                "capture_index": int(capture_id),
                "frame_index": int(frame_index),
                "frames": int(frame_count),
                "tensor_shape": [int(v) for v in tensor_shape],
                "baseline_shape": [int(v) for v in baseline.shape],
                "final_shape": [int(v) for v in final.shape],
                "codec": str(codec),
                "face_restore": float(face_restore),
                "background_restore": float(background_restore),
                "post_vae_enabled": bool(self.post_vae_enabled),
                "post_vae_requested": bool(post_vae_requested),
                "enhanced_returned": bool(enhanced_returned),
                "resize_to_livekit": bool(self.resize_to_livekit),
                "target_size": [int(target_w), int(target_h)],
                "mean_abs_diff": float(mad),
                "p95_abs_diff": float(p95),
                "max_abs_diff": float(max_abs),
                "decode_dt_sec": float(decode_dt_sec),
                "post_vae_debug": dict(debug_info or {}),
                "files": {
                    "before": str(out_dir / f"{stem}_before.png"),
                    "after": str(out_dir / f"{stem}_after.png"),
                    "diff": str(out_dir / f"{stem}_diff.png"),
                },
            }
            with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=True, indent=2, sort_keys=True)
            self._debug_capture_count = int(capture_id)
            logging.warning(
                "Remote edge PostVAE debug capture: block=%d frame=%d capture=%d mad=%.6f p95=%.6f max=%.6f enhanced=%d info=%s dir=%s",
                int(block_index),
                int(frame_index),
                int(capture_id),
                float(mad),
                float(p95),
                float(max_abs),
                1 if bool(enhanced_returned) else 0,
                json.dumps(dict(debug_info or {}), ensure_ascii=True, sort_keys=True),
                str(out_dir),
            )
        except Exception as e:
            logging.exception("Remote edge PostVAE debug capture failed: %s", e)

    def _tensor_stats(self, tensor: Any, *, max_values: int | None = None) -> dict[str, object]:
        if not self.torch.is_tensor(tensor):
            return {"is_tensor": False, "type": type(tensor).__name__}
        try:
            data = tensor.detach()
            stats_max = self._int_env("REMOTE_EDGE_LATENT_STATS_MAX_VALUES", 2_000_000, low=1, high=50_000_000)
            if max_values is not None:
                stats_max = max(1, min(int(stats_max), int(max_values)))
            flat = data.to(dtype=self.torch.float32).flatten()
            numel = int(flat.numel())
            if numel > int(stats_max):
                step = max(1, int((numel + int(stats_max) - 1) // int(stats_max)))
                flat = flat[::step]
            if int(flat.numel()) <= 0:
                return {
                    "shape": [int(v) for v in data.shape],
                    "dtype": str(data.dtype).replace("torch.", ""),
                    "device": str(data.device),
                    "numel": int(numel),
                    "sampled": 0,
                }
            return {
                "shape": [int(v) for v in data.shape],
                "dtype": str(data.dtype).replace("torch.", ""),
                "device": str(data.device),
                "numel": int(numel),
                "sampled": int(flat.numel()),
                "mean": float(flat.mean().detach().cpu().item()),
                "std": float(flat.std(unbiased=False).detach().cpu().item()),
                "min": float(flat.min().detach().cpu().item()),
                "max": float(flat.max().detach().cpu().item()),
            }
        except Exception as e:
            return {
                "shape": [int(v) for v in getattr(tensor, "shape", ())],
                "dtype": str(getattr(tensor, "dtype", "")),
                "device": str(getattr(tensor, "device", "")),
                "error": str(e),
            }

    def _maybe_log_latent_stats(
        self,
        *,
        block_index: int,
        tensor: Any,
        latents: list[Any],
        decoded: Any,
        baseline_01: Any,
        final_01: Any,
        codec: str,
        keep: int,
        face_restore: float | None,
        background_restore: float | None,
        output_width: int | None,
        output_height: int | None,
        decode_dt_sec: float,
    ) -> None:
        if not _env_flag("REMOTE_EDGE_LATENT_STATS_LOG", "0"):
            return
        try:
            import json

            max_blocks = self._int_env("REMOTE_EDGE_LATENT_STATS_MAX_BLOCKS", 8, low=0, high=100_000)
            dark_threshold_raw = str(os.getenv("REMOTE_EDGE_LATENT_DARK_MEAN_THRESHOLD", "0.02") or "0.02")
            try:
                dark_threshold = max(0.0, min(1.0, float(dark_threshold_raw)))
            except Exception:
                dark_threshold = 0.02

            final_stats = self._tensor_stats(final_01)
            baseline_stats = self._tensor_stats(baseline_01)
            final_mean = float(final_stats.get("mean", 1.0) or 0.0)
            baseline_mean = float(baseline_stats.get("mean", 1.0) or 0.0)
            should_log = int(block_index) <= int(max_blocks)
            should_log = bool(should_log or final_mean <= float(dark_threshold) or baseline_mean <= float(dark_threshold))
            if not should_log:
                return
            payload_stats = self._tensor_stats(tensor)
            latent0_stats = self._tensor_stats(latents[0]) if latents else {"empty": True}
            decoded_stats = self._tensor_stats(decoded)
            logging.warning(
                "Remote latent stats: block=%d codec=%s keep=%d target=%sx%s face=%s background=%s dt=%.3fs payload=%s latent0=%s decoded=%s baseline_01=%s final_01=%s",
                int(block_index),
                str(codec),
                int(keep),
                str(output_width or ""),
                str(output_height or ""),
                "none" if face_restore is None else f"{float(_clamp01(face_restore)):.3f}",
                "none" if background_restore is None else f"{float(_clamp01(background_restore)):.3f}",
                float(decode_dt_sec),
                json.dumps(payload_stats, ensure_ascii=True, sort_keys=True),
                json.dumps(latent0_stats, ensure_ascii=True, sort_keys=True),
                json.dumps(decoded_stats, ensure_ascii=True, sort_keys=True),
                json.dumps(baseline_stats, ensure_ascii=True, sort_keys=True),
                json.dumps(final_stats, ensure_ascii=True, sort_keys=True),
            )
        except Exception as e:
            logging.exception("Remote latent stats failed: %s", e)

    def _maybe_fail_dark_frames(
        self,
        *,
        block_index: int,
        baseline_01: Any,
        final_01: Any,
        codec: str,
        face_restore: float,
        background_restore: float,
        post_vae_requested: bool,
        enhanced_returned: bool,
        debug_info: dict[str, object] | None,
    ) -> None:
        if not _env_flag("REMOTE_EDGE_LATENT_DARK_FRAME_FAIL", "0"):
            return
        try:
            threshold_raw = str(os.getenv("REMOTE_EDGE_LATENT_DARK_MEAN_THRESHOLD", "0.02") or "0.02")
            try:
                threshold = max(0.0, min(1.0, float(threshold_raw)))
            except Exception:
                threshold = 0.02
            final_stats = self._tensor_stats(final_01, max_values=500_000)
            final_mean = float(final_stats.get("mean", 1.0) or 0.0)
            if final_mean > float(threshold):
                return
            baseline_stats = self._tensor_stats(baseline_01, max_values=500_000)
            import json

            raise RuntimeError(
                "remote edge decoded dark frames; refusing to publish black video "
                f"block={int(block_index)} codec={codec} threshold={float(threshold):.4f} "
                f"face_restore={float(face_restore):.3f} background_restore={float(background_restore):.3f} "
                f"post_vae_requested={1 if bool(post_vae_requested) else 0} "
                f"enhanced_returned={1 if bool(enhanced_returned) else 0} "
                f"baseline_01={json.dumps(baseline_stats, ensure_ascii=True, sort_keys=True)} "
                f"final_01={json.dumps(final_stats, ensure_ascii=True, sort_keys=True)} "
                f"post_vae_debug={json.dumps(dict(debug_info or {}), ensure_ascii=True, sort_keys=True)}"
            )
        except RuntimeError:
            raise
        except Exception as e:
            logging.exception("Remote latent dark-frame guard failed: %s", e)

    def reset(self) -> None:
        model = getattr(self.vae, "model", None)
        if model is not None:
            if hasattr(model, "first_decode"):
                model.first_decode = True
            if hasattr(model, "clear_cache_decode"):
                model.clear_cache_decode()
        self._decode_block_count = 0
        self._last_restore_log_key = None
        self._debug_capture_count = 0
        post_vae_reset = getattr(getattr(self, "post_vae", None), "reset_session", None)
        if callable(post_vae_reset):
            with self._post_vae_lock:
                post_vae_reset()

    def load_tensor(self, payload: bytes, *, codec: str = "torch.save", shape: str = "", dtype: str = "") -> Any:
        codec_s = str(codec or "torch.save").strip().lower()
        if codec_s in {"raw", "raw_tensor", "tensor.raw"}:
            shape_t = _parse_shape(str(shape or ""))
            dtype_t = _dtype_from_env(str(dtype or "").strip() or str(self.dtype).replace("torch.", ""))
            expected = int(self.torch.empty((), dtype=dtype_t).element_size())
            for dim in shape_t:
                expected *= int(dim)
            if len(payload) != int(expected):
                raise RuntimeError(
                    f"raw latent payload size mismatch: got={len(payload)} expected={expected} "
                    f"shape={shape_t} dtype={dtype_t}"
                )
            raw = self.torch.frombuffer(bytearray(payload), dtype=self.torch.uint8)
            return raw.view(dtype_t).reshape(shape_t).contiguous()
        if codec_s not in {"torch.save", "pt", "pth"}:
            raise RuntimeError(f"unsupported latent codec: {codec}")
        obj = self.torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("latents", "latent", "block_latents", "tensor"):
                value = obj.get(key)
                if self.torch.is_tensor(value):
                    return value
        if self.torch.is_tensor(obj):
            return obj
        raise RuntimeError("latent payload must be a Tensor or dict containing a Tensor")

    def _normalise_latents(self, tensor: Any) -> list[Any]:
        if not self.torch.is_tensor(tensor):
            raise RuntimeError("latents must be a torch Tensor")
        lat = tensor.detach()
        if lat.ndim == 4:
            items = [lat]
        elif lat.ndim == 5:
            items = [lat[i] for i in range(int(lat.shape[0]))]
        else:
            raise RuntimeError(f"latents must have shape C,T,H,W or B,C,T,H,W; got {tuple(lat.shape)}")
        return [x.contiguous().to(device=self.device, dtype=self.dtype, non_blocking=True) for x in items]

    def _sync_cuda_device(self, device: Any) -> None:
        value = str(device or "").strip()
        if not value.startswith("cuda"):
            return
        try:
            self.torch.cuda.synchronize(device=self.torch.device(value))
        except Exception:
            self.torch.cuda.synchronize()

    def _record_ready_tensor(self, tensor: Any) -> Any:
        if not bool(_env_flag("REMOTE_EDGE_NATIVE_M11_READY_EVENT", "1")):
            return tensor
        if not bool(self.torch.is_tensor(tensor)) or not bool(getattr(tensor, "is_cuda", False)):
            return tensor
        try:
            device = getattr(tensor, "device", self.device)
            with self.torch.cuda.device(device):
                event = self.torch.cuda.Event(blocking=False)
                event.record(self.torch.cuda.current_stream(device=device))
            return CudaReadyTensor(tensor, event, str(device))
        except Exception:
            logging.exception("Failed to record CUDA ready event for remote edge tensor; falling back to raw tensor")
            return tensor

    def _wait_ready_tensor(self, value: Any) -> Any:
        tensor = getattr(value, "tensor", value)
        event = getattr(value, "ready_event", None)
        if event is None:
            return tensor
        if not bool(self.torch.is_tensor(tensor)) or not bool(getattr(tensor, "is_cuda", False)):
            return tensor
        try:
            device = getattr(tensor, "device", self.device)
            with self.torch.cuda.device(device):
                self.torch.cuda.current_stream(device=device).wait_event(event)
        except Exception:
            logging.exception("Failed to wait for CUDA ready event on remote edge tensor")
            raise
        return tensor

    def _vae_phase_dt(self, started_at: float) -> float:
        if bool(_vae_phase_sync_enabled()):
            self._sync_cuda_device(self.device)
        return max(0.0, float(time.perf_counter() - float(started_at)))

    def _ensure_post_vae(self) -> Any:
        if self.post_vae is None:
            from liveavatar.models.wan.post_vae_enhancer import PostVAEEnhancer

            self.post_vae = PostVAEEnhancer(device=self.post_vae_device)
        return self.post_vae

    def _stage_frames_for_post_vae(self, frames_01: Any) -> Any:
        target = self.torch.device(self.post_vae_device)
        frames_01 = self._wait_ready_tensor(frames_01)
        source = frames_01.detach()
        sync_stage = _env_flag("REMOTE_EDGE_POST_VAE_STAGE_SYNC", "0")
        direct_p2p = _env_flag("REMOTE_EDGE_POST_VAE_DIRECT_P2P", "0")
        source_device = self.torch.device(source.device) if bool(getattr(source, "is_cuda", False)) else None
        if source_device == target and source.dtype in {
            self.torch.float16,
            self.torch.bfloat16,
            self.torch.float32,
        }:
            staged = source.contiguous()
            return staged
        if bool(sync_stage) and bool(getattr(source, "is_cuda", False)):
            self._sync_cuda_device(source.device)
        if (
            bool(getattr(source, "is_cuda", False))
            and target.type == "cuda"
            and source_device != target
            and not bool(direct_p2p)
        ):
            # This host hop crosses CUDA devices. Keep it blocking so GPU1 never
            # starts GFPGAN/background kernels from an unfinished host buffer.
            source = source.to(device="cpu", dtype=self.torch.float32, copy=True, non_blocking=False).contiguous()
            staged = source.to(device=target, dtype=self.torch.float32, copy=True, non_blocking=False).contiguous()
        else:
            staged = source.to(device=target, dtype=self.torch.float32, copy=True, non_blocking=True).contiguous()
        if bool(sync_stage) and bool(getattr(staged, "is_cuda", False)):
            self._sync_cuda_device(staged.device)
        return staged

    def postprocess_frames_tensor_01(
        self,
        frames_01: Any,
        *,
        face_restore: float | None = None,
        background_restore: float | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        resize_output: bool = True,
        apply_post_vae: bool = True,
        block_index: int | None = None,
        tensor_shape: tuple[int, ...] | None = None,
        codec: str = "tensor",
        decode_started_at: float | None = None,
        input_range: str = "01",
    ) -> Any:
        t0 = time.perf_counter() if decode_started_at is None else float(decode_started_at)
        frames_01 = self._wait_ready_tensor(frames_01)
        target_h = int(output_height or 0)
        target_w = int(output_width or 0)
        input_is_m11 = str(input_range or "").strip().lower() in {"m11", "-1,1", "minus1_1", "neg1_1"}
        if not bool(apply_post_vae) and not (
            bool(resize_output) and self.resize_to_livekit and target_h > 0 and target_w > 0
        ):
            with self.torch.inference_mode():
                frames_01 = frames_01.contiguous()
                self._sync_cuda_device(getattr(frames_01, "device", self.device))
            return frames_01
        raw_face = self.default_face_restore if face_restore is None else _clamp01(face_restore)
        raw_background = (
            self.default_background_restore if background_restore is None else _clamp01(background_restore)
        )
        face = min(float(raw_face), float(self.max_face_restore))
        background = min(float(raw_background), float(self.max_background_restore))
        post_vae_requested = bool(apply_post_vae) and bool(self.post_vae_enabled) and (face > 0.0 or background > 0.0)
        restore_log_key = (round(float(face), 3), round(float(background), 3), bool(self.post_vae_enabled))
        if restore_log_key != self._last_restore_log_key:
            logging.warning(
                "Remote edge restore config: post_vae=%d face_restore=%.2f background_restore=%.2f source=%s post_vae_device=%s",
                1 if bool(self.post_vae_enabled) else 0,
                float(face),
                float(background),
                "default" if face_restore is None and background_restore is None else "latent_message",
                self.post_vae_device,
            )
            self._last_restore_log_key = restore_log_key
        debug_baseline_01 = frames_01
        post_vae_debug_info: dict[str, object] = {}
        enhanced_returned = False
        phase_stage_sec = 0.0
        phase_range_sec = 0.0
        phase_enhance_sec = 0.0
        phase_resize_sec = 0.0
        phase_finalize_sec = 0.0
        phase_host_hop = False
        phase_source_device = str(getattr(frames_01, "device", ""))
        phase_target_device = str(self.post_vae_device)
        with self.torch.inference_mode():
            if bool(post_vae_requested):
                with self._post_vae_lock:
                    post_vae = self._ensure_post_vae()
                    if bool(getattr(post_vae, "enabled", False)):
                        post_vae.set_restore_strengths(face_restore=float(face), background_restore=float(background))
                        stage_started = time.perf_counter()
                        source_device = getattr(frames_01, "device", None)
                        phase_source_device = str(source_device or "")
                        phase_target_device = str(self.post_vae_device)
                        phase_host_hop = bool(
                            getattr(frames_01, "is_cuda", False)
                            and self.torch.device(source_device) != self.torch.device(self.post_vae_device)
                            and not _env_flag("REMOTE_EDGE_POST_VAE_DIRECT_P2P", "0")
                        )
                        staged_01 = self._stage_frames_for_post_vae(frames_01)
                        phase_stage_sec = max(0.0, time.perf_counter() - float(stage_started))
                        range_started = time.perf_counter()
                        if bool(input_is_m11):
                            frames_m11 = staged_01.contiguous()
                        else:
                            frames_m11 = (
                                staged_01.mul(2.0)
                                .sub(1.0)
                                .contiguous()
                            )
                        debug_baseline_01 = (frames_m11 + 1.0).mul(0.5).clamp(0.0, 1.0)
                        phase_range_sec = max(0.0, time.perf_counter() - float(range_started))
                        # PostVAEEnhancer's live contract is VAE RGB in [-1, 1].
                        # Passing [0, 1] here remaps colors to [0.5, 1] inside the
                        # enhancer and makes RTMP output look washed out.
                        enhance_started = time.perf_counter()
                        enhanced = post_vae.enhance_batch_tchw(
                            frames_m11,
                            output_height=target_h if bool(resize_output) and self.resize_to_livekit and target_h > 0 else None,
                            output_width=target_w if bool(resize_output) and self.resize_to_livekit and target_w > 0 else None,
                        )
                        phase_enhance_sec = max(0.0, time.perf_counter() - float(enhance_started))
                        post_vae_debug_info = dict(getattr(post_vae, "_last_debug_info", {}) or {})
                        if enhanced is not None:
                            enhanced_returned = True
                            frames_01 = enhanced.clamp(0.0, 1.0)
                    elif not bool(self._logged_post_vae_disabled):
                        logging.warning(
                            "Remote edge PostVAE requested but disabled by LIVE_RAW_POST_VAE_ENHANCER: face_restore=%.2f background_restore=%.2f",
                            float(face),
                            float(background),
                        )
                        self._logged_post_vae_disabled = True
            if bool(input_is_m11) and not bool(enhanced_returned):
                frames_01 = (frames_01.clamp(-1.0, 1.0) + 1.0).mul(0.5).clamp(0.0, 1.0)
                debug_baseline_01 = frames_01
            if bool(resize_output) and self.resize_to_livekit and target_h > 0 and target_w > 0:
                resize_started = time.perf_counter()
                frames_01 = self._resize_cover_crop_frames_01(
                    frames_01,
                    target_h=int(target_h),
                    target_w=int(target_w),
                ).clamp(0.0, 1.0)
                phase_resize_sec = max(0.0, time.perf_counter() - float(resize_started))
            self._maybe_capture_post_vae_debug(
                block_index=int(block_index or self._decode_block_count),
                baseline_01=debug_baseline_01,
                final_01=frames_01,
                tensor_shape=tuple(int(v) for v in (tensor_shape or tuple(int(x) for x in frames_01.shape))),
                codec=str(codec),
                face_restore=float(face),
                background_restore=float(background),
                post_vae_requested=bool(post_vae_requested),
                enhanced_returned=bool(enhanced_returned),
                target_h=int(target_h),
                target_w=int(target_w),
                debug_info=post_vae_debug_info,
                decode_dt_sec=float(time.perf_counter() - t0),
            )
            self._maybe_fail_dark_frames(
                block_index=int(block_index or self._decode_block_count),
                baseline_01=debug_baseline_01,
                final_01=frames_01,
                codec=str(codec),
                face_restore=float(face),
                background_restore=float(background),
                post_vae_requested=bool(post_vae_requested),
                enhanced_returned=bool(enhanced_returned),
                debug_info=post_vae_debug_info,
            )
            finalize_started = time.perf_counter()
            frames_01 = frames_01.contiguous()
            self._sync_cuda_device(getattr(frames_01, "device", self.device))
            phase_finalize_sec = max(0.0, time.perf_counter() - float(finalize_started))
            total_phase_sec = max(0.0, time.perf_counter() - float(t0))
            if bool(_env_flag("POST_VAE_TIMING_LOG", "0")) or total_phase_sec >= 0.75:
                logging.warning(
                    "Remote edge PostVAE phase timing: block=%d frames=%d total=%.3fs stage=%.3fs host_hop=%d source=%s target=%s input_range=%s range=%.3fs enhance=%.3fs resize=%.3fs finalize_sync=%.3fs face=%.2f background=%.2f returned=%d",
                    int(block_index or self._decode_block_count),
                    int(frames_01.shape[0]) if hasattr(frames_01, "shape") and len(frames_01.shape) > 0 else 0,
                    float(total_phase_sec),
                    float(phase_stage_sec),
                    1 if bool(phase_host_hop) else 0,
                    str(phase_source_device),
                    str(phase_target_device),
                    "m11" if bool(input_is_m11) else "01",
                    float(phase_range_sec),
                    float(phase_enhance_sec),
                    float(phase_resize_sec),
                    float(phase_finalize_sec),
                    float(face),
                    float(background),
                    1 if bool(enhanced_returned) else 0,
                )
        return frames_01

    def decode_payload_tensor_01(
        self,
        payload: bytes,
        *,
        codec: str = "torch.save",
        shape: str = "",
        dtype: str = "",
        keep_last_frames: int | None = None,
        reset: bool = False,
        prime_only: bool = False,
        face_restore: float | None = None,
        background_restore: float | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        resize_output: bool = True,
        apply_post_vae: bool = True,
        return_m11: bool = False,
    ) -> Any:
        if reset:
            self.reset()
        t0 = time.perf_counter()
        load_sec = 0.0
        normalise_sec = 0.0
        vae_sec = 0.0
        keep_sec = 0.0
        range_sec = 0.0
        post_sec = 0.0
        sync_sec = 0.0
        load_t0 = time.perf_counter()
        tensor = self.load_tensor(payload, codec=codec, shape=shape, dtype=dtype)
        load_sec = self._vae_phase_dt(load_t0)
        normalise_t0 = time.perf_counter()
        latents = self._normalise_latents(tensor)
        normalise_sec = self._vae_phase_dt(normalise_t0)
        with self.torch.inference_mode():
            vae_t0 = time.perf_counter()
            decoded = self.torch.stack(self.vae.stream_decode(latents))
            vae_sec = self._vae_phase_dt(vae_t0)
            if prime_only:
                sync_t0 = time.perf_counter()
                self.torch.cuda.synchronize() if str(self.device).startswith("cuda") else None
                sync_sec = self._vae_phase_dt(sync_t0)
                logging.info("Remote latent prime consumed: shape=%s dt=%.3fs", tuple(decoded.shape), time.perf_counter() - t0)
                return decoded.new_empty((0, 3, 1, 1))
            keep_t0 = time.perf_counter()
            video = decoded[0]
            keep = self.keep_last_frames_default if keep_last_frames is None else max(0, int(keep_last_frames))
            if keep > 0:
                video = video[:, -keep:]
            self._decode_block_count += 1
            block_index = int(self._decode_block_count)
            keep_sec = self._vae_phase_dt(keep_t0)
            range_t0 = time.perf_counter()
            frames_m11 = video.permute(1, 0, 2, 3).contiguous()
            post_vae_face = self.default_face_restore if face_restore is None else _clamp01(face_restore)
            post_vae_background = self.default_background_restore if background_restore is None else _clamp01(background_restore)
            post_vae_requested = bool(apply_post_vae) and bool(self.post_vae_enabled) and (
                float(post_vae_face) > 0.0 or float(post_vae_background) > 0.0
            )
            native_m11_return = bool(return_m11) and not bool(apply_post_vae) and not (
                bool(resize_output) and self.resize_to_livekit and int(output_height or 0) > 0 and int(output_width or 0) > 0
            )
            if bool(native_m11_return) or bool(post_vae_requested):
                frames_01 = frames_m11.contiguous()
                baseline_01 = None
                if bool(native_m11_return) and _env_flag("REMOTE_EDGE_NATIVE_M11_RETURN_SYNC", "0") and not _env_flag(
                    "REMOTE_EDGE_NATIVE_M11_READY_EVENT", "1"
                ):
                    sync_t0 = time.perf_counter()
                    self._sync_cuda_device(getattr(frames_01, "device", self.device))
                    sync_sec = max(0.0, time.perf_counter() - float(sync_t0))
                frames_01 = self._record_ready_tensor(frames_01)
            else:
                frames_01 = (frames_m11 + 1.0).mul(0.5).clamp(0.0, 1.0)
                baseline_01 = frames_01
            range_sec = self._vae_phase_dt(range_t0)
            post_t0 = time.perf_counter()
            if not bool(native_m11_return):
                frames_01 = self.postprocess_frames_tensor_01(
                    frames_01,
                    face_restore=face_restore,
                    background_restore=background_restore,
                    output_width=output_width,
                    output_height=output_height,
                    resize_output=bool(resize_output),
                    apply_post_vae=bool(apply_post_vae),
                    block_index=int(block_index),
                    tensor_shape=tuple(int(v) for v in tensor.shape),
                    codec=str(codec),
                    decode_started_at=float(t0),
                    input_range="m11" if bool(post_vae_requested) else "01",
                )
            post_sec = self._vae_phase_dt(post_t0)
            total_sec = max(0.0, time.perf_counter() - t0)
            if bool(_timing_enabled()) or float(total_sec) >= 0.75:
                logging.warning(
                    "Remote edge VAE phase timing: block=%d frames=%d total=%.3fs sync=%d load=%.3fs normalise=%.3fs vae_stream=%.3fs keep=%.3fs range=%.3fs postprocess=%.3fs return_sync=%.3fs codec=%s tensor_shape=%s latent_shape=%s final_shape=%s dtype=%s device=%s keep_frames=%d output_range=%s",
                    int(block_index),
                    int(frames_01.shape[0]) if hasattr(frames_01, "shape") and len(frames_01.shape) > 0 else 0,
                    float(total_sec),
                    1 if bool(_vae_phase_sync_enabled()) else 0,
                    float(load_sec),
                    float(normalise_sec),
                    float(vae_sec),
                    float(keep_sec),
                    float(range_sec),
                    float(post_sec),
                    float(sync_sec),
                    str(codec),
                    tuple(int(v) for v in tensor.shape),
                    tuple(tuple(int(x) for x in item.shape) for item in latents),
                    tuple(int(v) for v in frames_01.shape),
                    str(self.dtype),
                    str(self.device),
                    int(keep),
                    "m11" if bool(native_m11_return) else "01",
                )
            if baseline_01 is not None:
                self._maybe_log_latent_stats(
                    block_index=int(block_index),
                    tensor=tensor,
                    latents=latents,
                    decoded=decoded,
                    baseline_01=baseline_01,
                    final_01=frames_01,
                    codec=str(codec),
                    keep=int(keep),
                    face_restore=face_restore,
                    background_restore=background_restore,
                    output_width=output_width,
                    output_height=output_height,
                    decode_dt_sec=float(time.perf_counter() - t0),
                )
        logging.info(
            "Remote latent decoded: frames=%d shape=%s codec=%s dt=%.3fs format=tensor",
            int(frames_01.shape[0]),
            tuple(tensor.shape),
            str(codec),
            time.perf_counter() - t0,
        )
        return frames_01

    def decode_payload(
        self,
        payload: bytes,
        *,
        codec: str = "torch.save",
        shape: str = "",
        dtype: str = "",
        keep_last_frames: int | None = None,
        reset: bool = False,
        prime_only: bool = False,
        face_restore: float | None = None,
        background_restore: float | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        resize_output: bool = True,
    ) -> list[bytes]:
        frames_01 = self.decode_payload_tensor_01(
            payload,
            codec=codec,
            shape=shape,
            dtype=dtype,
            keep_last_frames=keep_last_frames,
            reset=reset,
            prime_only=prime_only,
            face_restore=face_restore,
            background_restore=background_restore,
            output_width=output_width,
            output_height=output_height,
            resize_output=bool(resize_output),
        )
        return _tensor_01_to_rgb24_frames(frames_01)

    def decode_payload_threadsafe(self, *args: Any, **kwargs: Any) -> list[bytes]:
        with self._decode_lock:
            return self.decode_payload(*args, **kwargs)

    def decode_payload_tensor_01_threadsafe(self, *args: Any, **kwargs: Any) -> Any:
        with self._decode_lock:
            return self.decode_payload_tensor_01(*args, **kwargs)

    def postprocess_frames_tensor_01_threadsafe(self, *args: Any, **kwargs: Any) -> Any:
        return self.postprocess_frames_tensor_01(*args, **kwargs)

    def restore_known_face_crops_tensor_01_threadsafe(
        self,
        crops_01: Any,
        *,
        face_restore: float,
    ) -> Any | None:
        if not bool(self.post_vae_enabled):
            return None
        face = _clamp01(float(face_restore or 0.0))
        if face <= 0.0:
            return crops_01
        with self._post_vae_lock:
            post_vae = self._ensure_post_vae()
            restore = getattr(post_vae, "restore_known_face_crops_tchw", None)
            if not callable(restore):
                return None
            return restore(crops_01, face_restore=float(face))

    def apply_face_aligned_transform_tensor_01_threadsafe(
        self,
        frames_01: Any,
        *,
        transform_callback: Any,
        face_restore: float,
    ) -> Any | None:
        if not bool(self.post_vae_enabled):
            return None
        frames_01 = self._wait_ready_tensor(frames_01)
        if frames_01 is None:
            return None
        with self._post_vae_lock:
            post_vae = self._ensure_post_vae()
            transform = getattr(post_vae, "apply_face_aligned_transform_tchw", None)
            if not callable(transform):
                return None
            staged_01 = self._stage_frames_for_post_vae(frames_01)
            return transform(
                staged_01,
                transform_callback=transform_callback,
                face_restore=float(face_restore or 0.0),
            )

    def prewarm_post_vae(
        self,
        *,
        height: int,
        width: int,
        face_restore: float,
        background_restore: float,
    ) -> bool:
        if not bool(self.post_vae_enabled):
            return False
        if self.post_vae is None:
            self._ensure_post_vae()
        if not bool(getattr(self.post_vae, "enabled", False)):
            if not bool(self._logged_post_vae_disabled):
                logging.warning("Remote edge PostVAE prewarm skipped: LIVE_RAW_POST_VAE_ENHANCER is disabled")
                self._logged_post_vae_disabled = True
            return False
        face = min(_clamp01(face_restore), float(self.max_face_restore))
        background = min(_clamp01(background_restore), float(self.max_background_restore))
        with self._decode_lock:
            self.post_vae.set_restore_strengths(face_restore=float(face), background_restore=float(background))
            prewarm = getattr(self.post_vae, "prewarm_live_shape", None)
            if callable(prewarm):
                return bool(prewarm(height=int(height), width=int(width), face_restore=float(face), background_restore=float(background)))
            if face > 0.0:
                face_prewarm = getattr(self.post_vae, "prewarm_face_restore", None)
                if callable(face_prewarm):
                    return bool(face_prewarm(height=int(height), width=int(width)))
        return False


_SHARED_DECODER: WanLatentDecoder | None = None
_SHARED_DECODER_LOCK = threading.RLock()


def get_shared_wan_latent_decoder() -> WanLatentDecoder:
    global _SHARED_DECODER
    with _SHARED_DECODER_LOCK:
        if _SHARED_DECODER is None:
            _SHARED_DECODER = WanLatentDecoder()
        return _SHARED_DECODER


def _parse_render_sizes(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for part in str(raw or "").replace(";", ",").split(","):
        value = part.strip().lower().replace("x", "*")
        if not value or "*" not in value:
            continue
        left, right = value.split("*", 1)
        try:
            size = (int(left.strip()), int(right.strip()))
        except Exception:
            continue
        if int(size[0]) <= 0 or int(size[1]) <= 0 or size in seen:
            continue
        seen.add(size)
        out.append(size)
    return out


def prewarm_shared_wan_latent_decoder_from_env() -> WanLatentDecoder:
    decoder = get_shared_wan_latent_decoder()
    if not _env_flag("REMOTE_EDGE_PREWARM_POST_VAE", "1"):
        return decoder
    raw_sizes = str(os.getenv("REMOTE_EDGE_PREWARM_RENDER_SIZES", "") or "").strip()
    if not raw_sizes:
        raw_sizes = str(os.getenv("SIZE", "384*256") or "384*256").strip()
    sizes = _parse_render_sizes(raw_sizes)
    if not sizes:
        sizes = [(384, 256)]
    face = _env_float01("REMOTE_EDGE_PREWARM_FACE_RESTORE", 0.9)
    background = _env_float01("REMOTE_EDGE_PREWARM_BACKGROUND_RESTORE", 0.5)
    for height, width in sizes:
        decoder.prewarm_post_vae(
            height=int(height),
            width=int(width),
            face_restore=float(face),
            background_restore=float(background),
        )
    logging.warning(
        "Remote edge shared latent decoder prewarmed: sizes=%s face=%.2f background=%.2f",
        ",".join(f"{h}x{w}" for h, w in sizes),
        float(face),
        float(background),
    )
    return decoder
