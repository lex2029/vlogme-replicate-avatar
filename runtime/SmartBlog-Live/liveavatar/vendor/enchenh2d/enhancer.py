"""GPU-accelerated image enhancement with TensorRT + GFPGAN + kornia."""

import os
import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia


def _install_torchvision_compat() -> None:
    """Backfill removed torchvision module names expected by basicsr/gfpgan."""
    try:
        import torchvision.transforms._functional_tensor as _functional_tensor
    except Exception:
        return
    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor",
        _functional_tensor,
    )


_install_torchvision_compat()

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODES = [
    "enhance",              # RealESRGAN sharpen (no upscale)
    "upscale",              # RealESRGAN x2
    "face",                 # GFPGAN only
    "face_upscale",         # GFPGAN → RealESRGAN x2
    "upscale_face",         # RealESRGAN x2 → GFPGAN
    "upscale_face_enhance", # RealESRGAN x2 → GFPGAN → RealESRGAN sharpen
    "double_upscale_face",  # RealESRGAN x2 → GFPGAN → RealESRGAN x2
]


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class TRTInfer:
    """TensorRT inference: GPU tensor in → GPU tensor out."""

    def _device_context(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.device(self.device)
        return nullcontext()

    def __init__(self, engine_path, device="cuda"):
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        with self._device_context():
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"TensorRT failed to deserialize engine: {engine_path}")
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(
                    f"TensorRT failed to create execution context for {engine_path}; "
                    "this usually means the GPU cannot reserve the engine memory"
                )
        self.output_names = []
        self.input_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        self._sync = str(os.getenv("SMARTBLOG_TRT_SYNC", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        with self._device_context():
            self._stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None

    def _torch_dtype(self, name, fallback):
        try:
            dtype = self.engine.get_tensor_dtype(name)
        except Exception:
            return fallback
        if dtype == trt.DataType.HALF:
            return torch.float16
        if dtype == trt.DataType.FLOAT:
            return torch.float32
        if dtype == trt.DataType.INT32:
            return torch.int32
        if dtype == trt.DataType.INT8:
            return torch.int8
        if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
            return torch.bfloat16
        return fallback

    def __call__(self, input_tensor):
        if self.context is None:
            raise RuntimeError("TensorRT execution context is not available")
        if self.device.type == "cuda" and input_tensor.device != self.device:
            input_tensor = input_tensor.to(device=self.device, non_blocking=False)
        current_stream = torch.cuda.current_stream(input_tensor.device) if input_tensor.is_cuda else None
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        if self._sync and current_stream is not None:
            current_stream.synchronize()
        if self._stream is not None and current_stream is not None:
            self._stream.wait_stream(current_stream)
            stream_context = torch.cuda.stream(self._stream)
            cuda_stream = self._stream.cuda_stream
        else:
            stream_context = torch.cuda.stream(current_stream) if current_stream is not None else nullcontext()
            cuda_stream = current_stream.cuda_stream if current_stream is not None else 0
        b, c, h, w = input_tensor.shape
        with self._device_context(), stream_context:
            if not self.context.set_input_shape(self.input_name, (b, c, h, w)):
                raise ValueError(
                    f"TensorRT engine does not support input shape {(b, c, h, w)} "
                    f"for tensor {self.input_name}"
                )
            self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
            outputs = {}
            for name in self.output_names:
                shape = self.context.get_tensor_shape(name)
                dtype = self._torch_dtype(name, input_tensor.dtype)
                out = torch.empty(tuple(shape), dtype=dtype, device=self.device)
                self.context.set_tensor_address(name, out.data_ptr())
                outputs[name] = out
            self.context.execute_async_v3(cuda_stream)
        if self._stream is not None and current_stream is not None:
            current_stream.wait_stream(self._stream)
        if self._sync and current_stream is not None:
            current_stream.synchronize()
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return tuple(outputs.values())


def _detect_faces_gpu(face_det, img_rgb_01, priors_cache=None):
    """Face detection on GPU tensor. Returns list of 5-point landmarks."""
    h, w = img_rgb_01.shape[2], img_rgb_01.shape[3]
    img_bgr = img_rgb_01[:, [2, 1, 0], :, :] * 255.0
    if face_det.half_inference:
        img_bgr = img_bgr.half()
    img_bgr = img_bgr - face_det.mean_tensor.to(img_bgr.device)

    from facexlib.detection.retinaface_utils import decode, decode_landm, py_cpu_nms, PriorBox

    scale = torch.tensor([w, h, w, h], device=img_bgr.device, dtype=torch.float32)
    scale1 = torch.tensor([w, h] * 5, device=img_bgr.device, dtype=torch.float32)

    loc, conf, landmarks_raw = _face_detect_forward(face_det, img_bgr)

    if priors_cache and (h, w) in priors_cache:
        priors = priors_cache[(h, w)]
    else:
        priorbox = PriorBox(face_det.cfg, image_size=(h, w))
        priors = priorbox.forward().to(img_bgr.device)

    boxes = decode(loc.data.squeeze(0), priors.data, face_det.cfg['variance'])
    boxes = (boxes * scale).detach().cpu().numpy()
    scores = conf.squeeze(0).data.detach().cpu().numpy()[:, 1]
    landmarks_np = decode_landm(landmarks_raw.squeeze(0), priors, face_det.cfg['variance'])
    landmarks_np = (landmarks_np * scale1).detach().cpu().numpy()

    inds = np.where(scores > 0.97)[0]
    if len(inds) == 0:
        return []
    boxes, landmarks_np, scores = boxes[inds], landmarks_np[inds], scores[inds]
    order = scores.argsort()[::-1]
    boxes, landmarks_np, scores = boxes[order], landmarks_np[order], scores[order]
    bboxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(bboxes, 0.4)
    return [landmarks_np[k].reshape(5, 2) for k in keep]


def _face_detect_forward(face_det, img_bgr):
    """Run RetinaFace without cuDNN engine search when requested.

    On the live edge, cuDNN can sporadically fail RetinaFace with
    "FIND was unable to find an engine", then poison the shared CUDA context.
    The non-cuDNN path is slightly slower but stable for the small per-frame
    detection workload.
    """
    use_cudnn = str(os.getenv("LIVE_RAW_POST_VAE_FACE_DETECT_CUDNN", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with torch.no_grad():
        if bool(use_cudnn) or not bool(getattr(img_bgr, "is_cuda", False)):
            return face_det(img_bgr)
        with torch.backends.cudnn.flags(enabled=False):
            return face_det(img_bgr)


def _nms_boxes_torch(boxes, scores, iou_threshold=0.4):
    """Torch NMS that keeps indices on the input device."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    boxes_f = boxes.float()
    scores_f = scores.float()
    x1, y1, x2, y2 = boxes_f.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores_f, descending=True)
    keep = []
    while int(order.numel()) > 0:
        i = order[0]
        keep.append(i)
        if int(order.numel()) == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= float(iou_threshold)]
    if not keep:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(keep).to(dtype=torch.long, device=boxes.device)


def _similarity_affine_torch(src, dst):
    """Fit batched similarity affine matrices using torch only; returns matrix and inverse."""
    src_f = src.float()
    dst_f = dst.float()
    squeeze = False
    if src_f.ndim == 2:
        src_f = src_f.unsqueeze(0)
        dst_f = dst_f.unsqueeze(0)
        squeeze = True
    if dst_f.ndim == 2:
        dst_f = dst_f.unsqueeze(0).expand(src_f.shape[0], -1, -1)
    device_type = "cuda" if src_f.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        src_mean = src_f.mean(dim=1, keepdim=True)
        dst_mean = dst_f.mean(dim=1, keepdim=True)
        src_c = src_f - src_mean
        dst_c = dst_f - dst_mean
        cov = dst_c.transpose(1, 2).matmul(src_c) / max(1, int(src_f.shape[1]))
        u, s, vh = torch.linalg.svd(cov)
        r = u.matmul(vh)
        det = torch.linalg.det(r)
        if bool((det < 0).any()):
            fix = torch.ones((src_f.shape[0], 2), dtype=torch.float32, device=src_f.device)
            fix[:, 1] = torch.where(det < 0, -1.0, 1.0)
            r = u.matmul(torch.diag_embed(fix)).matmul(vh)
            s = s * fix
        var = (src_c * src_c).sum(dim=(1, 2)) / max(1, int(src_f.shape[1]))
        scale = (s.sum(dim=1) / var.clamp(min=1e-8)).view(-1, 1, 1)
        linear = scale * r
        trans = dst_mean.transpose(1, 2) - linear.matmul(src_mean.transpose(1, 2))
        m = torch.cat((linear, trans), dim=2)
        linear = m[:, :, :2]
        linear_inv = torch.linalg.inv(linear)
        trans_inv = -linear_inv.matmul(m[:, :, 2:3])
        m_inv = torch.cat((linear_inv, trans_inv), dim=2)
    if squeeze:
        return m[:1], m_inv[:1]
    return m, m_inv


def _get_affine_matrices(landmarks_list, face_template, face_size, device):
    """Compute affine + inverse from landmarks. CPU math only (6 numbers)."""
    results = []
    template_np = face_template.cpu().numpy()
    for landmark in landmarks_list:
        M = cv2.estimateAffinePartial2D(landmark, template_np, method=cv2.LMEDS)[0]
        M_t = torch.tensor(M, dtype=torch.float32, device=device).unsqueeze(0)
        M_inv = cv2.invertAffineTransform(M)
        M_inv_t = torch.tensor(M_inv, dtype=torch.float32, device=device).unsqueeze(0)
        results.append((M_t, M_inv_t))
    return results


class Enhancer:
    """GPU image enhancer with multiple modes."""

    def _device_context(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.device(device)
        return nullcontext()

    def __init__(self, mode="upscale_face", device="cuda", blend=0.9, trt=True,
                 models_dir=None):
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Available: {MODES}")

        self.mode = mode
        self.device = device
        self.blend = blend
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._realesrgan_is_trt = False
        self._realesrgan_half = _env_flag("LIVE_RAW_POST_VAE_BACKGROUND_FP16", "1")
        self._realesrgan_cudnn = _env_flag("LIVE_RAW_POST_VAE_BACKGROUND_CUDNN", "1")

        self._needs_realesrgan = mode in ("enhance", "upscale", "face_upscale",
                                          "upscale_face", "upscale_face_enhance",
                                          "double_upscale_face")
        self._needs_gfpgan = mode in ("face", "face_upscale", "upscale_face",
                                      "upscale_face_enhance", "double_upscale_face")

        # Load RealESRGAN
        if self._needs_realesrgan:
            trt_path = self.models_dir / "realesrgan_x2.engine"
            if trt and HAS_TRT and trt_path.exists():
                self._realesrgan = TRTInfer(str(trt_path), device=device)
                self._realesrgan_is_trt = True
                logging.info("RealESRGAN: TensorRT")
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=2)
                self._realesrgan_obj = RealESRGANer(
                    scale=2, model_path=str(self.models_dir / "RealESRGAN_x2plus.pth"),
                    model=rrdb, tile=0, half=bool(self._realesrgan_half), device=device)
                self._realesrgan = lambda x: self._realesrgan_obj.model(x)
                logging.info(
                    "RealESRGAN: PyTorch %s cudnn=%d",
                    "FP16" if bool(self._realesrgan_half) else "FP32",
                    1 if bool(self._realesrgan_cudnn) else 0,
                )
            default_sync = "0" if bool(self._realesrgan_is_trt) else "1"
            self._realesrgan_output_sync = _env_flag(
                "LIVE_RAW_POST_VAE_BACKGROUND_OUTPUT_SYNC",
                default_sync,
            )

        # Load GFPGAN
        if self._needs_gfpgan:
            from gfpgan import GFPGANer
            self._gfpgan_wrapper = GFPGANer(
                model_path=str(self.models_dir / "GFPGANv1.4.pth"),
                upscale=1, arch="clean", channel_multiplier=2,
                bg_upsampler=None, device=device)
            self._gfpgan = self._gfpgan_wrapper.gfpgan
            self._face_det = self._gfpgan_wrapper.face_helper.face_det
            self._face_template = torch.tensor(
                self._gfpgan_wrapper.face_helper.face_template,
                dtype=torch.float32, device=device)
            self._face_size = self._gfpgan_wrapper.face_helper.face_size

            # Pre-compute priors for common sizes
            from facexlib.detection.retinaface_utils import PriorBox
            self._priors_cache = {}
            for rh, rw in [(448, 256), (256, 448), (832, 448), (448, 832),
                           (896, 512), (512, 896), (1664, 896), (896, 1664)]:
                pb = PriorBox(self._face_det.cfg, image_size=(rh, rw))
                self._priors_cache[(rh, rw)] = pb.forward().to(device)

        self._cached_affine = {"pairs": None, "clip_id": None}

        # Pre-allocated buffers (set by preallocate())
        self._buffers = None

    def preallocate(self, in_h, in_w, out_h=None, out_w=None):
        """Pre-allocate GPU buffers for given input/output size.

        Call once after init. Avoids GPU memory allocation during inference.

        Args:
            in_h, in_w: input frame size (render resolution)
            out_h, out_w: output size (optional, for resize)

        Example:
            enh = Enhancer(mode="upscale_face")
            enh.preallocate(in_h=448, in_w=256, out_h=1280, out_w=720)
            # Now feed frames:
            result = enh.enhance_gpu(frame_tensor)
        """
        sr_h, sr_w = in_h * 2, in_w * 2
        self._buffers = {
            "input": torch.zeros(1, 3, in_h, in_w, dtype=torch.float32, device=self.device),
            "sr": torch.zeros(1, 3, sr_h, sr_w, dtype=torch.float32, device=self.device),
            "face_crop": torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=self.device),
            "face_restored": torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=self.device),
            "output": torch.zeros(1, 3, out_h or sr_h, out_w or sr_w,
                                  dtype=torch.float32, device=self.device),
            "in_h": in_h, "in_w": in_w, "out_h": out_h, "out_w": out_w,
        }
        logging.info(f"Pre-allocated buffers: input={in_h}x{in_w}, sr={sr_h}x{sr_w}, "
                     f"output={out_h}x{out_w}")

    def enhance_gpu(self, frame_tensor, out_h=None, out_w=None,
                    clip_id=0, frame_idx=0):
        """Enhance GPU tensor → return GPU tensor. No CPU transfer.

        Args:
            frame_tensor: GPU tensor [1, 3, H, W] in RGB [-1, 1]
            out_h, out_w: resize output (optional)
            clip_id, frame_idx: for cached face detection

        Returns:
            GPU tensor [1, 3, out_h, out_w] in RGB [0, 1]
        """
        x = (frame_tensor.clamp(-1, 1) + 1) / 2

        if self.mode == "enhance":
            x = self._run_realesrgan(x)
            x = F.interpolate(x, size=(frame_tensor.shape[2], frame_tensor.shape[3]),
                              mode='bilinear', align_corners=False)
        elif self.mode == "upscale":
            x = self._run_realesrgan(x)
        elif self.mode == "face":
            x = self._run_gfpgan(x, clip_id, frame_idx)
        elif self.mode == "face_upscale":
            x = self._run_gfpgan(x, clip_id, frame_idx)
            x = self._run_realesrgan(x)
        elif self.mode == "upscale_face":
            x = self._run_realesrgan(x)
            x = self._run_gfpgan(x, clip_id, frame_idx)
        elif self.mode == "upscale_face_enhance":
            x = self._run_realesrgan(x)
            x = self._run_gfpgan(x, clip_id, frame_idx)
            sr2 = self._run_realesrgan(x)
            x = F.interpolate(sr2, size=(x.shape[2], x.shape[3]),
                              mode='bilinear', align_corners=False)
        elif self.mode == "double_upscale_face":
            x = self._run_realesrgan(x)
            x = self._run_gfpgan(x, clip_id, frame_idx)
            x = self._run_realesrgan(x)

        if out_h and out_w:
            x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)

        return x  # GPU tensor [1, 3, H, W] in [0, 1]

    def _run_realesrgan(self, tensor_01):
        """RealESRGAN x2 on GPU tensor [1,3,H,W] in [0,1]."""
        dtype = torch.float16 if bool(self._realesrgan_half or self._realesrgan_is_trt) else torch.float32
        cudnn_context = (
            torch.backends.cudnn.flags(enabled=bool(self._realesrgan_cudnn))
            if bool(getattr(tensor_01, "is_cuda", False))
            else nullcontext()
        )
        with torch.no_grad(), self._device_context(), cudnn_context:
            x = tensor_01.to(device=self.device, dtype=dtype, non_blocking=False).contiguous()
            out = self._realesrgan(x).clamp(0, 1).float()
            if bool(getattr(out, "is_cuda", False)) and bool(getattr(self, "_realesrgan_output_sync", True)):
                torch.cuda.synchronize(device=out.device)
            return out

    def _run_gfpgan(self, tensor_01, clip_id=0, frame_idx=0):
        """GFPGAN face restore on GPU tensor [1,3,H,W] in [0,1]. Returns same tensor."""
        # Cached face detection per clip
        if self._cached_affine["clip_id"] == clip_id and self._cached_affine["pairs"] is not None:
            affine_pairs = self._cached_affine["pairs"]
        else:
            landmarks = _detect_faces_gpu(
                self._face_det, tensor_01, priors_cache=self._priors_cache)
            if landmarks:
                affine_pairs = _get_affine_matrices(
                    landmarks, self._face_template, self._face_size, self.device)
                self._cached_affine["pairs"] = affine_pairs
                self._cached_affine["clip_id"] = clip_id
            else:
                affine_pairs = []

        if not affine_pairs:
            return tensor_01

        result = tensor_01.clone()
        for M, M_inv in affine_pairs:
            try:
                cropped = kornia.geometry.transform.warp_affine(
                    tensor_01, M, (self._face_size[1], self._face_size[0]),
                    mode='bilinear', padding_mode='zeros')
                with torch.no_grad():
                    restored = self._gfpgan(
                        cropped * 2 - 1,
                        return_rgb=False,
                        randomize_noise=False,
                    )[0]
                restored_01 = (restored.clamp(-1, 1) + 1) / 2

                if torch.isnan(restored_01).any() or torch.isinf(restored_01).any():
                    continue

                h, w = tensor_01.shape[2], tensor_01.shape[3]
                pasted = kornia.geometry.transform.warp_affine(
                    restored_01, M_inv, (h, w), mode='bilinear', padding_mode='zeros')

                inner = torch.zeros_like(restored_01)
                inner[:, :, 64:-64, 64:-64] = 1.0
                mask = kornia.geometry.transform.warp_affine(
                    inner, M_inv, (h, w), mode='bilinear', padding_mode='zeros')
                mask = kornia.filters.gaussian_blur2d(mask, (51, 51), (17.0, 17.0))

                result = result * (1 - mask) + pasted * mask
            except Exception:
                continue

        if self.blend < 1.0:
            result = result * self.blend + tensor_01 * (1 - self.blend)

        return result

    def __call__(self, frame_tensor, out_h=None, out_w=None,
                 clip_id=0, frame_idx=0):
        """Enhance GPU tensor → BGR numpy.

        Args:
            frame_tensor: GPU tensor [1, 3, H, W] in RGB [-1, 1]

        Returns:
            BGR numpy [out_h, out_w, 3] uint8
        """
        x = self.enhance_gpu(frame_tensor, out_h, out_w, clip_id, frame_idx)
        # Single GPU→CPU transfer
        result_np = (x[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        return result_np[:, :, ::-1].copy()

    def from_numpy(self, bgr_image, out_h=None, out_w=None, clip_id=0, frame_idx=0):
        """Enhance from BGR numpy image."""
        rgb = bgr_image[:, :, ::-1].copy().astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = tensor * 2 - 1  # [0,1] → [-1,1]
        return self(tensor, out_h, out_w, clip_id, frame_idx)

    def from_file(self, path, out_h=None, out_w=None):
        """Enhance from image file."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return self.from_numpy(img, out_h, out_w)
