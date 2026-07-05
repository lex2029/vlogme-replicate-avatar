from __future__ import annotations

import importlib.util
import os
import threading
from contextlib import nullcontext
from typing import Any


_DEFAULT_MODEL_DIR = "/opt/RIFE-safetensors"
_DEFAULT_WEIGHTS = os.path.join(_DEFAULT_MODEL_DIR, "flownet.safetensors")
_SHARED_LOCK = threading.Lock()
_SHARED: dict[tuple[str, str, str, str, int], "TorchRifeInterpolator"] = {}


def _fit_tensor_frame_count(tensor: Any, target_frames: int) -> Any:
    import torch

    target = max(0, int(target_frames))
    source = int(tensor.shape[0])
    if target <= 0:
        return tensor[:0].contiguous()
    if source <= 0 or source == target:
        return tensor.contiguous()
    if source < target:
        tail = tensor[-1:].expand(int(target - source), *tuple(tensor.shape[1:]))
        return torch.cat((tensor, tail), dim=0).contiguous()
    if target == 1:
        return tensor[:1].contiguous()
    idx = torch.linspace(0, source - 1, target, device=tensor.device).round().to(torch.long)
    return tensor.index_select(0, idx).contiguous()


def rgb24_frames_to_tensor_01(
    frames: list[bytes],
    *,
    width: int,
    height: int,
    device: Any,
    dtype: Any,
) -> Any:
    import numpy as np
    import torch

    source_frames = [bytes(frame) for frame in frames]
    frame_bytes = int(width) * int(height) * 3
    for frame in source_frames:
        if len(frame) != frame_bytes:
            raise ValueError(f"invalid rgb24 frame size for torch-rife: got={len(frame)} expected={frame_bytes}")
    if not source_frames:
        return torch.empty((0, 3, int(height), int(width)), device=device, dtype=dtype)
    arr = np.frombuffer(b"".join(source_frames), dtype=np.uint8).copy()
    arr = arr.reshape((len(source_frames), int(height), int(width), 3))
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype, non_blocking=True)
    return tensor.permute(0, 3, 1, 2).contiguous().div_(255.0)


def tensor_01_to_rgb24_frames(tensor_01: Any) -> list[bytes]:
    import torch

    if not torch.is_tensor(tensor_01):
        raise TypeError("tensor_01_to_rgb24_frames expects a torch Tensor")
    if int(tensor_01.ndim) != 4:
        raise ValueError(f"tensor_01_to_rgb24_frames expects T,C,H,W tensor, got shape={tuple(tensor_01.shape)}")
    if int(tensor_01.shape[0]) <= 0:
        return []
    rgb = (tensor_01.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    rgb = rgb.permute(0, 2, 3, 1).contiguous()
    rgb_cpu = rgb.cpu().numpy()
    return [bytes(memoryview(frame).cast("B")) for frame in rgb_cpu]


def _resolve_module_path(model_dir: str) -> str:
    path = os.path.join(str(model_dir or _DEFAULT_MODEL_DIR), "interpolation_model.py")
    if not os.path.exists(path):
        raise RuntimeError(
            "torch-rife live interpolation requires interpolation_model.py; "
            "set REMOTE_EDGE_TORCH_RIFE_MODEL_DIR or install TensorForger/RIFE-safetensors"
        )
    return path


def _load_ifnet_module(model_dir: str, *, device: Any, dtype: Any) -> Any:
    module_path = _resolve_module_path(model_dir)
    spec = importlib.util.spec_from_file_location("_smartblog_rife_ifnet", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load RIFE model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # TensorForger/RIFE-safetensors keeps device/dtype globals used by warp().
    module.device = device
    module.dtype = dtype
    return module


class TorchRifeInterpolator:
    """In-memory x2 RIFE interpolation for live RGB24 frame batches."""

    def __init__(
        self,
        *,
        model_dir: str = _DEFAULT_MODEL_DIR,
        weights_path: str = _DEFAULT_WEIGHTS,
        device: str = "cuda:0",
        dtype_name: str = "float16",
        batch_pairs: int = 4,
    ) -> None:
        import torch
        from safetensors.torch import load_file

        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("torch-rife live interpolation requires CUDA")
        dtype = torch.float16 if str(dtype_name).lower() in {"fp16", "float16", "half"} else torch.float32
        torch_device = torch.device(device)
        if not os.path.exists(weights_path):
            raise RuntimeError(
                f"torch-rife live interpolation weights not found: {weights_path}; "
                "set REMOTE_EDGE_TORCH_RIFE_WEIGHTS"
            )

        module = _load_ifnet_module(model_dir, device=torch_device, dtype=dtype)
        model = module.IFNet()
        state = load_file(weights_path, device="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(device=torch_device, dtype=dtype)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        self.torch = torch
        self.model = model
        self.device = torch_device
        self.dtype = dtype
        self.batch_pairs = max(1, int(batch_pairs))
        stream_enabled = str(os.getenv("REMOTE_EDGE_TORCH_RIFE_DEDICATED_STREAM", "1") or "1").strip().lower()
        self.stream = (
            torch.cuda.Stream(device=torch_device)
            if torch_device.type == "cuda" and stream_enabled in {"1", "true", "yes", "on"}
            else None
        )

    def interpolate_tensor_x2(
        self,
        frames_01: Any,
        *,
        target_frames: int | None = None,
    ) -> Any:
        torch = self.torch
        if not torch.is_tensor(frames_01):
            raise TypeError("interpolate_tensor_x2 expects a torch Tensor")
        if int(frames_01.ndim) != 4 or int(frames_01.shape[1]) != 3:
            raise ValueError(f"interpolate_tensor_x2 expects T,3,H,W tensor, got shape={tuple(frames_01.shape)}")
        tensor = frames_01.detach().to(device=self.device, dtype=self.dtype, non_blocking=True).contiguous()
        source_count = int(tensor.shape[0])
        wanted = int(target_frames or (source_count * 2))
        if source_count < 2:
            return _fit_tensor_frame_count(tensor, int(wanted))

        mids: list[Any] = []
        pair_count = int(source_count - 1)
        stream = getattr(self, "stream", None)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device=self.device))
        stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with torch.inference_mode(), stream_ctx:
            for start in range(0, pair_count, int(self.batch_pairs)):
                end = min(pair_count, start + int(self.batch_pairs))
                inp = torch.cat((tensor[start:end], tensor[start + 1 : end + 1]), dim=1)
                mids.append(self.model(inp).clamp_(0.0, 1.0))
            mid_tensor = torch.cat(mids, dim=0) if mids else tensor[:0]
            out = torch.empty(
                (int(pair_count * 2 + 1), int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            out[0:-1:2].copy_(tensor[:-1])
            out[1:-1:2].copy_(mid_tensor)
            out[-1].copy_(tensor[-1])
            fitted = _fit_tensor_frame_count(out, int(wanted))
        if stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(stream)
        return fitted

    def interpolate_x2(
        self,
        frames: list[bytes],
        *,
        width: int,
        height: int,
        target_frames: int | None = None,
    ) -> list[bytes]:
        source_frames = [bytes(frame) for frame in frames]
        if len(source_frames) < 2:
            return source_frames
        tensor = rgb24_frames_to_tensor_01(
            source_frames,
            width=int(width),
            height=int(height),
            device=self.device,
            dtype=self.dtype,
        )
        out_tensor = self.interpolate_tensor_x2(tensor, target_frames=target_frames)
        return tensor_01_to_rgb24_frames(out_tensor)


def get_shared_torch_rife_interpolator(
    *,
    model_dir: str = _DEFAULT_MODEL_DIR,
    weights_path: str = _DEFAULT_WEIGHTS,
    device: str = "cuda:0",
    dtype_name: str = "float16",
    batch_pairs: int = 4,
) -> TorchRifeInterpolator:
    key = (str(model_dir), str(weights_path), str(device), str(dtype_name), int(batch_pairs))
    with _SHARED_LOCK:
        cached = _SHARED.get(key)
        if cached is not None:
            return cached
        created = TorchRifeInterpolator(
            model_dir=str(model_dir),
            weights_path=str(weights_path),
            device=str(device),
            dtype_name=str(dtype_name),
            batch_pairs=int(batch_pairs),
        )
        _SHARED[key] = created
        return created
