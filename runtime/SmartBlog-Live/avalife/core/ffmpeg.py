from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frames: int
    duration_sec: float
    codec_name: str


def _parse_ffprobe_rate(raw: object) -> float:
    text = str(raw or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


def _safe_float(raw: object) -> float:
    try:
        return float(raw or 0.0)
    except Exception:
        return 0.0


def probe_video_metadata(src_path: str) -> VideoProbe:
    abs_path = os.path.abspath(str(src_path or "").strip())
    if not abs_path or not os.path.exists(abs_path):
        raise FileNotFoundError(f"video missing: {abs_path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(abs_path),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if int(proc.returncode or 0) != 0:
        err_text = str(proc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed rc={proc.returncode}: {err_text or 'no stderr'}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}") from e
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {abs_path}")
    stream = streams[0] or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid input video size: {abs_path}")
    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate")) or _parse_ffprobe_rate(stream.get("r_frame_rate"))
    frames = int(stream.get("nb_frames") or 0)
    codec_name = str(stream.get("codec_name") or "").strip().lower()
    fmt = payload.get("format") or {}
    duration_sec = _safe_float(fmt.get("duration"))
    if duration_sec <= 0.0 and fps > 0.0 and frames > 0:
        duration_sec = float(frames) / float(fps)
    return VideoProbe(
        width=int(width),
        height=int(height),
        fps=float(fps),
        frames=int(frames),
        duration_sec=float(duration_sec),
        codec_name=codec_name,
    )


def video_duration_sec(src_path: str) -> float:
    try:
        return float(probe_video_metadata(src_path).duration_sec)
    except Exception:
        return 0.0


def _cuda_device_index(device: Any) -> int:
    if device is None:
        return 0
    idx = getattr(device, "index", None)
    if idx is not None and (not callable(idx)):
        return int(idx)
    text = str(device or "cuda:0").strip().lower()
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except Exception:
            return 0
    try:
        return int(text)
    except Exception:
        return 0


def cuda_decoder_for_codec(codec_name: str) -> str:
    name = str(codec_name or "").strip().lower()
    mapping = {
        "av1": "av1_cuvid",
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "mjpeg": "mjpeg_cuvid",
        "mpeg1video": "mpeg1_cuvid",
        "mpeg2video": "mpeg2_cuvid",
        "mpeg4": "mpeg4_cuvid",
        "vc1": "vc1_cuvid",
        "vp8": "vp8_cuvid",
        "vp9": "vp9_cuvid",
    }
    decoder = mapping.get(name)
    if decoder:
        return decoder
    raise RuntimeError(f"unsupported codec for CUDA video decode: {codec_name or 'unknown'}")


def video_decode_rgb24_ffmpeg_cmd(
    *,
    src_path: str,
    codec_name: str,
    out_w: int,
    out_h: int,
    device: Any = "cuda:0",
    frame_limit: int = 0,
    pix_fmt: str = "rgb24",
) -> list[str]:
    probe = probe_video_metadata(src_path)
    gpu_idx = _cuda_device_index(device)
    decoder = cuda_decoder_for_codec(codec_name or probe.codec_name)
    pix_fmt_norm = str(pix_fmt or "rgb24").strip().lower() or "rgb24"
    # hwdownload cannot convert CUDA frames directly into packed RGB/BGR.
    # First land on a CPU-friendly planar format, then convert to the target.
    vf = f"hwdownload,format=nv12,format={pix_fmt_norm}"
    if int(out_w) != int(probe.width) or int(out_h) != int(probe.height):
        vf = (
            f"scale_cuda=w={int(out_w)}:h={int(out_h)}:interp_algo=lanczos,"
            f"hwdownload,format=nv12,format={pix_fmt_norm}"
        )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-extra_hw_frames",
        "8",
        "-c:v",
        str(decoder),
        "-gpu",
        str(gpu_idx),
        "-i",
        str(os.path.abspath(src_path)),
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
    ]
    if int(frame_limit) > 0:
        cmd.extend(["-frames:v", str(int(frame_limit))])
    cmd.extend(
        [
            "-vf",
            vf,
            "-pix_fmt",
            pix_fmt_norm,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    return cmd


def read_exact_into(stream, buf: bytearray) -> bool:
    view = memoryview(buf)
    offset = 0
    total = int(len(view))
    while offset < total:
        n = stream.readinto(view[offset:])
        if n is None:
            continue
        n = int(n)
        if n <= 0:
            if offset == 0:
                return False
            raise RuntimeError(f"short rawvideo read: expected={total} got={offset}")
        offset += n
    return True


class RGB24VideoPipeReader:
    def __init__(
        self,
        video_path: str,
        *,
        out_w: int,
        out_h: int,
        device: Any = "cuda:0",
        frame_limit: int = 0,
        pix_fmt: str = "rgb24",
    ) -> None:
        self.video_path = os.path.abspath(str(video_path or "").strip())
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.device = device
        self.frame_limit = int(max(0, int(frame_limit)))
        self.pix_fmt = str(pix_fmt or "rgb24").strip().lower() or "rgb24"
        self.probe = probe_video_metadata(self.video_path)
        self._frame_bytes = int(self.out_w * self.out_h * 3)
        self._buf = bytearray(self._frame_bytes)
        self._closed = False
        self._proc = subprocess.Popen(
            video_decode_rgb24_ffmpeg_cmd(
                src_path=self.video_path,
                codec_name=self.probe.codec_name,
                out_w=self.out_w,
                out_h=self.out_h,
                device=self.device,
                frame_limit=self.frame_limit,
                pix_fmt=self.pix_fmt,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def read_frame_view(self) -> memoryview | None:
        if self._closed:
            return None
        if self._proc.stdout is None:
            raise RuntimeError("ffmpeg decode stdout is not available")
        ok = read_exact_into(self._proc.stdout, self._buf)
        if ok:
            return memoryview(self._buf)
        self._finish_and_check()
        return None

    def _finish_and_check(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
        stderr_data = b""
        try:
            stderr_data = self._proc.stderr.read() if self._proc.stderr is not None else b""
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            stderr_data = self._proc.stderr.read() if self._proc.stderr is not None else b""
            self._proc.wait()
            raise RuntimeError("ffmpeg decode timeout")
        if int(self._proc.returncode or 0) != 0:
            err_text = (stderr_data or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg decode failed rc={self._proc.returncode}: {err_text or 'no stderr'}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            if self._proc.stderr is not None:
                self._proc.stderr.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            pass
        self._closed = True


def read_first_frame_rgb(path: str, *, width: int = 0, height: int = 0, device: Any = "cuda:0") -> np.ndarray | None:
    abs_path = os.path.abspath(str(path or "").strip())
    if not abs_path or not os.path.exists(abs_path):
        return None
    probe = probe_video_metadata(abs_path)
    out_w = int(width or probe.width)
    out_h = int(height or probe.height)
    reader = RGB24VideoPipeReader(abs_path, out_w=out_w, out_h=out_h, device=device, frame_limit=1)
    try:
        frame = reader.read_frame_view()
        if frame is None:
            return None
        arr = np.frombuffer(frame, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
        return np.ascontiguousarray(arr, dtype=np.uint8)
    finally:
        reader.close()
