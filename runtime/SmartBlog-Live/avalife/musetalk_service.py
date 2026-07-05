from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
from aiohttp import web
from avalife.core.upload_retry import put_file_to_signed_url


LOG = logging.getLogger("smartblog-musetalk-service")


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(str(os.getenv(name, str(default)) or str(default)).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return float(default)


def _text(value: Any, default: str = "") -> str:
    raw = str(value if value is not None else default).strip()
    return raw if raw else str(default or "")


def _run(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None, timeout: float, label: str) -> None:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(timeout),
        check=False,
        text=True,
    )
    elapsed = float(time.perf_counter() - started)
    if int(proc.returncode or 0) != 0:
        stderr = str(proc.stderr or "")[-6000:]
        stdout = str(proc.stdout or "")[-2000:]
        raise RuntimeError(f"{label} failed rc={proc.returncode} elapsed={elapsed:.2f}s: {stderr or stdout}")
    if proc.stderr:
        LOG.info("%s stderr: %s", label, str(proc.stderr)[-2000:])


def _ffprobe(path: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if int(proc.returncode or 0) != 0:
        raise RuntimeError(f"ffprobe failed: {str(proc.stderr or '')[-2000:]}")
    return json.loads(proc.stdout or "{}")


def _video_has_audio(path: str) -> bool:
    try:
        data = _ffprobe(path)
    except Exception:
        return False
    for stream in data.get("streams") or []:
        if str(stream.get("codec_type") or "") == "audio":
            return True
    return False


def _probe_duration(path: str) -> float:
    try:
        data = _ffprobe(path)
        duration = float((data.get("format") or {}).get("duration") or 0.0)
        return float(max(0.0, duration))
    except Exception:
        return 0.0


def _probe_video_fps(path: str, default: float = 16.0) -> float:
    try:
        data = _ffprobe(path)
        for stream in data.get("streams") or []:
            if str(stream.get("codec_type") or "") != "video":
                continue
            for key in ("avg_frame_rate", "r_frame_rate"):
                raw = str(stream.get(key) or "").strip()
                if not raw or raw == "0/0":
                    continue
                if "/" in raw:
                    left, right = raw.split("/", 1)
                    fps = float(left) / float(right)
                else:
                    fps = float(raw)
                if fps > 0:
                    return float(fps)
    except Exception:
        pass
    return float(default)


def _safe_suffix(name: str, default: str) -> str:
    suffix = Path(str(name or "")).suffix.lower()
    if not suffix or len(suffix) > 16:
        return str(default)
    return suffix


def _download_url_to_file(*, url: str, path: str) -> dict[str, Any]:
    started = time.perf_counter()
    bytes_written = 0
    with requests.get(
        str(url),
        stream=True,
        timeout=(
            max(3.0, _env_float("SMARTBLOG_MUSETALK_SOURCE_CONNECT_TIMEOUT_SEC", 20.0)),
            max(30.0, _env_float("SMARTBLOG_MUSETALK_SOURCE_READ_TIMEOUT_SEC", 1800.0)),
        ),
    ) as resp:
        resp.raise_for_status()
        with open(str(path), "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += int(len(chunk))
    if bytes_written <= 0:
        raise RuntimeError(f"downloaded zero bytes from {url}")
    return {
        "source_bytes": int(bytes_written),
        "source_download_sec": float(time.perf_counter() - started),
    }


def _upload_file_to_signed_url(*, signed_url: str, path: str, content_type: str = "video/mp4") -> dict[str, Any]:
    return put_file_to_signed_url(
        signed_url=str(signed_url),
        path=str(path),
        content_type=str(content_type or "video/mp4"),
        connect_timeout=max(3.0, _env_float("SMARTBLOG_MUSETALK_UPLOAD_CONNECT_TIMEOUT_SEC", 20.0)),
        read_timeout=max(30.0, _env_float("SMARTBLOG_MUSETALK_UPLOAD_READ_TIMEOUT_SEC", 1800.0)),
        env_prefix="SMARTBLOG_MUSETALK_UPLOAD",
        log_prefix="musetalk-signed-upload",
    )


def _write_base64_file(value: str, *, path: str) -> None:
    raw = str(value or "").strip()
    if "," in raw[:128]:
        raw = raw.split(",", 1)[1]
    with open(str(path), "wb") as f:
        f.write(base64.b64decode(raw))


def _extract_audio(*, video_path: str, output_audio_path: str) -> dict[str, Any]:
    if not _video_has_audio(str(video_path)):
        raise RuntimeError("input video has no audio stream and no separate audio was provided")
    started = time.perf_counter()
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(max(8000, _env_int("SMARTBLOG_MUSETALK_AUDIO_SAMPLE_RATE", 16000))),
            str(output_audio_path),
        ],
        timeout=max(60.0, _probe_duration(video_path) * 5.0 + 60.0),
        label="extract audio for MuseTalk",
    )
    return {
        "audio_source": "video",
        "audio_extract_sec": float(time.perf_counter() - started),
    }


def _transcode_video_fps(*, input_path: str, output_path: str, fps: int) -> dict[str, Any]:
    fps_i = int(max(1, int(fps or 25)))
    started = time.perf_counter()
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-vf",
            f"fps={fps_i}",
            "-c:v",
            "libx264",
            "-crf",
            "15",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        timeout=max(60.0, _probe_duration(input_path) * 8.0 + 60.0),
        label="prepare MuseTalk fps input",
    )
    return {
        "prepared_fps": int(fps_i),
        "prepare_fps_sec": float(time.perf_counter() - started),
    }


def _remux_final_audio(*, video_path: str, audio_path: str, output_path: str) -> None:
    duration = max(_probe_duration(video_path), _probe_duration(audio_path), 1.0)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        timeout=max(60.0, duration * 4.0 + 60.0),
        label="remux MuseTalk output audio",
    )


def _write_musetalk_config(*, path: str, video_path: str, audio_path: str, bbox_shift: int) -> None:
    # Paths are local temp paths generated by the service; quote for YAML safety.
    content = (
        "task_0:\n"
        f"  video_path: {json.dumps(str(video_path))}\n"
        f"  audio_path: {json.dumps(str(audio_path))}\n"
        f"  bbox_shift: {int(bbox_shift)}\n"
    )
    Path(path).write_text(content, encoding="utf-8")


def _musetalk_paths(root: str, version: str) -> dict[str, str]:
    root_path = Path(root).resolve()
    version_s = str(version or "v15").strip().lower()
    if version_s in {"v1.5", "1.5", "v15"}:
        return {
            "version": "v15",
            "unet_model_path": str(root_path / "models/musetalkV15/unet.pth"),
            "unet_config": str(root_path / "models/musetalkV15/musetalk.json"),
        }
    return {
        "version": "v1",
        "unet_model_path": str(root_path / "models/musetalk/pytorch_model.bin"),
        "unet_config": str(root_path / "models/musetalk/musetalk.json"),
    }


def _check_musetalk_ready(root: str, python: str, version: str) -> dict[str, Any]:
    paths = _musetalk_paths(root, version)
    require_mmpose = _env_flag("SMARTBLOG_MUSETALK_REQUIRE_MMPOSE", "0")
    checks = {
        "root": os.path.isdir(str(root)),
        "python": os.path.exists(str(python)) and os.access(str(python), os.X_OK),
        "script": os.path.exists(os.path.join(str(root), "scripts", "inference.py")),
        "unet_model": os.path.exists(paths["unet_model_path"]),
        "unet_config": os.path.exists(paths["unet_config"]),
        "whisper": os.path.exists(os.path.join(str(root), "models", "whisper", "pytorch_model.bin")),
        "sd_vae": os.path.exists(os.path.join(str(root), "models", "sd-vae", "diffusion_pytorch_model.bin")),
        "face_parse": os.path.exists(os.path.join(str(root), "models", "face-parse-bisent", "79999_iter.pth")),
    }
    if bool(require_mmpose):
        checks["dwpose"] = os.path.exists(os.path.join(str(root), "models", "dwpose", "dw-ll_ucoco_384.pth"))
    ready = all(bool(v) for v in checks.values())
    return {
        "ok": True,
        "ready": bool(ready),
        "pid": os.getpid(),
        "root": str(root),
        "python": str(python),
        "version": paths["version"],
        "checks": checks,
    }


class _MuseTalkResidentRuntime:
    def __init__(
        self,
        *,
        root: str,
        paths: dict[str, str],
        use_float16: bool,
        parsing_mode: str,
        extra_margin: int,
        left_cheek_width: int,
        right_cheek_width: int,
    ) -> None:
        self.root = str(Path(root).resolve())
        self.paths = dict(paths)
        self.use_float16 = bool(use_float16)
        self.parsing_mode = str(parsing_mode or "jaw")
        self.extra_margin = int(max(0, extra_margin))
        self.left_cheek_width = int(max(0, left_cheek_width))
        self.right_cheek_width = int(max(0, right_cheek_width))
        self.key = (
            self.root,
            str(self.paths.get("version") or ""),
            bool(self.use_float16),
            self.parsing_mode,
            int(self.extra_margin),
            int(self.left_cheek_width),
            int(self.right_cheek_width),
        )
        self._load()

    def _load(self) -> None:
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        old_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            import torch
            from transformers import WhisperModel

            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.utils.blending import get_image_blending, get_image_prepare_material
            from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
            from musetalk.utils.utils import datagen, load_all_model
            from musetalk.utils.face_parsing import FaceParsing

            self.torch = torch
            self.get_landmark_and_bbox = get_landmark_and_bbox
            self.coord_placeholder = coord_placeholder
            self.get_image_prepare_material = get_image_prepare_material
            self.get_image_blending = get_image_blending
            self.datagen = datagen
            gpu_id = _env_int("SMARTBLOG_MUSETALK_GPU_ID", 0)
            self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            started = time.perf_counter()
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(self.paths["unet_model_path"]),
                vae_type="sd-vae",
                unet_config=str(self.paths["unet_config"]),
                device=self.device,
            )
            self.timesteps = torch.tensor([0], device=self.device)
            if bool(self.use_float16):
                self.pe = self.pe.half()
                self.vae.vae = self.vae.vae.half()
                self.unet.model = self.unet.model.half()
            self.pe = self.pe.to(self.device)
            self.vae.vae = self.vae.vae.to(self.device)
            self.unet.model = self.unet.model.to(self.device)
            self.weight_dtype = self.unet.model.dtype
            whisper_dir = os.path.join(self.root, "models", "whisper")
            self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
            self.whisper = WhisperModel.from_pretrained(whisper_dir)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
            if str(self.paths.get("version") or "") == "v15":
                self.fp = FaceParsing(
                    left_cheek_width=int(self.left_cheek_width),
                    right_cheek_width=int(self.right_cheek_width),
                )
            else:
                self.fp = FaceParsing()
            self.load_sec = float(time.perf_counter() - started)
            LOG.warning(
                "MuseTalk resident runtime loaded: version=%s device=%s fp16=%d elapsed=%.3fs",
                str(self.paths.get("version") or ""),
                str(self.device),
                1 if bool(self.use_float16) else 0,
                float(self.load_sec),
            )
        finally:
            os.chdir(old_cwd)

    def _extract_frames(self, *, video_path: str, frames_dir: str) -> tuple[list[str], float]:
        os.makedirs(str(frames_dir), exist_ok=True)
        started = time.perf_counter()
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-i",
                str(video_path),
                "-an",
                "-vsync",
                "0",
                os.path.join(str(frames_dir), "%08d.png"),
            ],
            timeout=max(60.0, _probe_duration(video_path) * 8.0 + 60.0),
            label="extract MuseTalk resident frames",
        )
        frame_paths = sorted(str(p) for p in Path(frames_dir).glob("*.png"))
        if not frame_paths:
            raise RuntimeError("MuseTalk resident extracted no frames")
        return frame_paths, float(time.perf_counter() - started)

    def _open_video_writer(self, *, output_video_path: str, width: int, height: int, fps: float) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{int(width)}x{int(height)}",
                "-r",
                f"{float(fps):.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                str(os.getenv("SMARTBLOG_MUSETALK_RESIDENT_ENCODE_PRESET", "veryfast") or "veryfast"),
                "-crf",
                str(os.getenv("SMARTBLOG_MUSETALK_RESIDENT_ENCODE_CRF", "18") or "18"),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_video_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )

    def _batch_latents_for_unet(self, crop_frames: list[Any], *, batch_size: int) -> list[Any]:
        import cv2
        import numpy as np

        if not crop_frames:
            return []
        torch = self.torch
        latents: list[Any] = []
        bs = int(max(1, int(batch_size or 16)))
        mask = (self.vae._mask_tensor > 0.5).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        for start in range(0, len(crop_frames), bs):
            batch = crop_frames[start : start + bs]
            rgb = [cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB) for img in batch]
            x = np.asarray(rgb, dtype=np.float32) / 255.0
            x = np.transpose(x, (0, 3, 1, 2))
            tensor = torch.from_numpy(x)
            masked = tensor * mask
            masked = self.vae.transform(masked).to(self.vae.vae.device, dtype=self.vae.vae.dtype)
            full = self.vae.transform(tensor).to(self.vae.vae.device, dtype=self.vae.vae.dtype)
            with torch.no_grad():
                masked_latents = self.vae.scaling_factor * self.vae.vae.encode(masked).latent_dist.sample()
                ref_latents = self.vae.scaling_factor * self.vae.vae.encode(full).latent_dist.sample()
            batch_latents = torch.cat([masked_latents, ref_latents], dim=1)
            for idx in range(int(batch_latents.shape[0])):
                latents.append(batch_latents[idx : idx + 1])
        return latents

    def process(
        self,
        *,
        video_path: str,
        audio_path: str,
        output_path: str,
        work_dir: str,
        bbox_shift: int,
        batch_size: int,
        force_processing_fps: int,
        fixed_bbox: bool,
        bbox_sample_frames: int,
        mask_stride: int,
    ) -> dict[str, Any]:
        import cv2
        import numpy as np

        started = time.perf_counter()
        timings: dict[str, float] = {}

        def mark(name: str, t0: float) -> None:
            timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - t0)

        source_fps = float(_probe_video_fps(str(video_path), default=16.0))
        fps = float(force_processing_fps or source_fps or 16.0)
        fps = float(max(1.0, min(60.0, fps)))
        input_video = str(video_path)
        if int(force_processing_fps or 0) > 0 and abs(float(force_processing_fps) - float(source_fps)) > 0.01:
            t0 = time.perf_counter()
            prepared_video = os.path.join(str(work_dir), "resident_input_fps.mp4")
            _transcode_video_fps(input_path=str(video_path), output_path=str(prepared_video), fps=int(force_processing_fps))
            input_video = str(prepared_video)
            mark("prepare_fps", t0)

        frames_dir = os.path.join(str(work_dir), "resident_frames")
        frame_paths, extract_sec = self._extract_frames(video_path=str(input_video), frames_dir=str(frames_dir))
        timings["extract_frames"] = float(extract_sec)

        t0 = time.perf_counter()
        audio_features, librosa_length = self.audio_processor.get_audio_feature(str(audio_path), weight_dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            audio_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=int(round(float(fps))),
            audio_padding_length_left=_env_int("SMARTBLOG_MUSETALK_AUDIO_PADDING_LEFT", 2),
            audio_padding_length_right=_env_int("SMARTBLOG_MUSETALK_AUDIO_PADDING_RIGHT", 2),
        )
        mark("audio_features", t0)

        shift = 0 if str(self.paths.get("version") or "") == "v15" else int(bbox_shift)
        fixed_bbox_used = False
        t0 = time.perf_counter()
        if bool(fixed_bbox):
            sample_count = int(max(1, min(len(frame_paths), int(bbox_sample_frames or 5))))
            if sample_count >= len(frame_paths):
                sample_indices = list(range(len(frame_paths)))
            else:
                sample_indices = sorted(set(int(round(v)) for v in np.linspace(0, len(frame_paths) - 1, sample_count)))
            sample_paths = [frame_paths[int(i)] for i in sample_indices]
            sample_coords, _sample_frames = self.get_landmark_and_bbox(sample_paths, shift)
            valid = [tuple(int(v) for v in bbox) for bbox in sample_coords if bbox != self.coord_placeholder]
            frame_list = []
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame_list.append(frame)
            if valid and frame_list:
                arr = np.asarray(valid, dtype=np.float32)
                x1, y1, x2, y2 = [float(v) for v in np.median(arr, axis=0).tolist()]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_ratio = float(max(0.0, min(0.5, _env_float("SMARTBLOG_MUSETALK_FIXED_BBOX_PAD_RATIO", 0.08))))
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw *= 1.0 + pad_ratio
                bh *= 1.0 + pad_ratio
                h0, w0 = int(frame_list[0].shape[0]), int(frame_list[0].shape[1])
                fixed = (
                    max(0, min(w0 - 1, int(round(cx - bw * 0.5)))),
                    max(0, min(h0 - 1, int(round(cy - bh * 0.5)))),
                    max(1, min(w0, int(round(cx + bw * 0.5)))),
                    max(1, min(h0, int(round(cy + bh * 0.5)))),
                )
                if fixed[2] > fixed[0] and fixed[3] > fixed[1]:
                    coord_list = [fixed for _ in frame_list]
                    fixed_bbox_used = True
                else:
                    coord_list, frame_list = self.get_landmark_and_bbox(frame_paths, shift)
            else:
                coord_list, frame_list = self.get_landmark_and_bbox(frame_paths, shift)
        else:
            coord_list, frame_list = self.get_landmark_and_bbox(frame_paths, shift)
        mark("bbox", t0)
        if not frame_list:
            raise RuntimeError("MuseTalk resident has no decoded frames")
        frame_height, frame_width = int(frame_list[0].shape[0]), int(frame_list[0].shape[1])

        t0 = time.perf_counter()
        coord_placeholder = self.coord_placeholder
        crop_frames: list[Any] = []
        cleaned_coord_list: list[Any] = []
        cleaned_frame_list: list[Any] = []
        last_valid: tuple[int, int, int, int] | None = None
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                if last_valid is None:
                    continue
                bbox = last_valid
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if str(self.paths.get("version") or "") == "v15":
                y2 = min(int(y2) + int(self.extra_margin), int(frame.shape[0]))
            x1 = max(0, min(int(frame.shape[1]) - 1, int(x1)))
            x2 = max(x1 + 1, min(int(frame.shape[1]), int(x2)))
            y1 = max(0, min(int(frame.shape[0]) - 1, int(y1)))
            y2 = max(y1 + 1, min(int(frame.shape[0]), int(y2)))
            bbox = (x1, y1, x2, y2)
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            crop_frames.append(resized_crop_frame)
            cleaned_coord_list.append(bbox)
            cleaned_frame_list.append(frame)
            last_valid = bbox
        vae_batch_size = int(max(1, _env_int("SMARTBLOG_MUSETALK_VAE_ENCODE_BATCH_SIZE", 16)))
        input_latent_list = self._batch_latents_for_unet(crop_frames, batch_size=int(vae_batch_size))
        if not input_latent_list:
            raise RuntimeError("MuseTalk resident found no usable face latents")
        mark("vae_encode", t0)

        t0 = time.perf_counter()
        frame_list_cycle = cleaned_frame_list + cleaned_frame_list[::-1]
        coord_list_cycle = cleaned_coord_list + cleaned_coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        mask_list_cycle: list[Any] = []
        mask_coords_list_cycle: list[Any] = []
        mask_stride_i = int(max(1, int(mask_stride or 1)))
        mask_cache: dict[int, tuple[Any, Any]] = {}
        for idx, (frame, bbox) in enumerate(zip(frame_list_cycle, coord_list_cycle)):
            if bool(fixed_bbox_used):
                cache_idx = 0
            else:
                cache_idx = int(idx) if mask_stride_i <= 1 else int(idx // mask_stride_i) * int(mask_stride_i)
            cached = mask_cache.get(cache_idx)
            if cached is None:
                source_frame = frame_list_cycle[min(cache_idx, len(frame_list_cycle) - 1)]
                source_bbox = coord_list_cycle[min(cache_idx, len(coord_list_cycle) - 1)]
                mask, crop_box = self.get_image_prepare_material(
                    source_frame,
                    [int(v) for v in source_bbox],
                    fp=self.fp,
                    mode=str(self.parsing_mode or "jaw") if str(self.paths.get("version") or "") == "v15" else "raw",
                )
                cached = (mask, crop_box)
                mask_cache[cache_idx] = cached
            mask_list_cycle.append(cached[0])
            mask_coords_list_cycle.append(cached[1])
        mark("mask_prepare", t0)

        t0 = time.perf_counter()
        temp_video_path = os.path.join(str(work_dir), "resident_video_only.mp4")
        writer = self._open_video_writer(
            output_video_path=str(temp_video_path),
            width=int(frame_width),
            height=int(frame_height),
            fps=float(fps),
        )
        frames_written = 0
        try:
            gen = self.datagen(
                whisper_chunks,
                input_latent_list_cycle,
                max(1, int(batch_size or 8)),
                device=self.device,
            )
            total_batches = int(np.ceil(float(len(whisper_chunks)) / float(max(1, int(batch_size or 8)))))
            LOG.warning(
                "MuseTalk resident inference start: frames=%d batches=%d batch=%d fps=%.3f size=%dx%d",
                int(len(whisper_chunks)),
                int(total_batches),
                int(batch_size),
                float(fps),
                int(frame_width),
                int(frame_height),
            )
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                pred_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                recon = self.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    if frames_written >= int(len(whisper_chunks)):
                        break
                    bbox = coord_list_cycle[frames_written % len(coord_list_cycle)]
                    ori_frame = frame_list_cycle[frames_written % len(frame_list_cycle)]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    try:
                        face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception:
                        face = np.asarray(res_frame, dtype=np.uint8)
                    combine_frame = self.get_image_blending(
                        ori_frame,
                        face,
                        [x1, y1, x2, y2],
                        mask_list_cycle[frames_written % len(mask_list_cycle)],
                        mask_coords_list_cycle[frames_written % len(mask_coords_list_cycle)],
                    )
                    if writer.stdin is None:
                        raise RuntimeError("MuseTalk resident ffmpeg stdin is closed")
                    writer.stdin.write(np.ascontiguousarray(combine_frame).tobytes())
                    frames_written += 1
            if writer.stdin is not None:
                writer.stdin.close()
            stderr = (writer.stderr.read() if writer.stderr is not None else b"").decode("utf-8", errors="replace")
            rc = writer.wait(timeout=max(60.0, float(len(whisper_chunks)) / float(max(1.0, fps)) * 5.0 + 60.0))
            if int(rc or 0) != 0:
                raise RuntimeError(f"MuseTalk resident video encode failed rc={rc}: {stderr[-4000:]}")
        except Exception:
            try:
                if writer.stdin is not None:
                    writer.stdin.close()
            except Exception:
                pass
            try:
                writer.kill()
            except Exception:
                pass
            raise
        mark("inference_encode", t0)

        t0 = time.perf_counter()
        _remux_final_audio(video_path=str(temp_video_path), audio_path=str(audio_path), output_path=str(output_path))
        mark("remux", t0)
        elapsed = float(time.perf_counter() - started)
        LOG.warning(
            "MuseTalk resident complete: frames=%d fps=%.3f size=%dx%d fixed_bbox=%d mask_stride=%d elapsed=%.3fs timings=%s",
            int(frames_written),
            float(fps),
            int(frame_width),
            int(frame_height),
            1 if bool(fixed_bbox_used) else 0,
            int(mask_stride_i),
            float(elapsed),
            json.dumps(timings, ensure_ascii=True, sort_keys=True),
        )
        return {
            "ok": True,
            "backend": "resident",
            "output_path": str(output_path),
            "version": str(self.paths.get("version") or ""),
            "bbox_shift": int(bbox_shift),
            "batch_size": int(batch_size),
            "use_float16": bool(self.use_float16),
            "force_processing_fps": int(force_processing_fps or 0),
            "source_fps": float(source_fps),
            "fps": float(fps),
            "frames": int(frames_written),
            "width": int(frame_width),
            "height": int(frame_height),
            "fixed_bbox": bool(fixed_bbox_used),
            "mask_stride": int(mask_stride_i),
            "duration_sec": float(_probe_duration(str(output_path))),
            "bytes": int(os.path.getsize(str(output_path))) if os.path.exists(str(output_path)) else 0,
            "timings_sec": dict(timings),
            "elapsed_sec": float(elapsed),
        }


_RESIDENT_LOCK = threading.Lock()
_RESIDENT_RUNTIME: _MuseTalkResidentRuntime | None = None


def _resident_runtime(
    *,
    root: str,
    version: str,
    use_float16: bool,
    parsing_mode: str,
    extra_margin: int,
    left_cheek_width: int,
    right_cheek_width: int,
) -> _MuseTalkResidentRuntime:
    global _RESIDENT_RUNTIME
    paths = _musetalk_paths(root, version)
    key = (
        str(Path(root).resolve()),
        str(paths.get("version") or ""),
        bool(use_float16),
        str(parsing_mode or "jaw"),
        int(extra_margin),
        int(left_cheek_width),
        int(right_cheek_width),
    )
    with _RESIDENT_LOCK:
        if _RESIDENT_RUNTIME is None or getattr(_RESIDENT_RUNTIME, "key", None) != key:
            _RESIDENT_RUNTIME = _MuseTalkResidentRuntime(
                root=str(root),
                paths=paths,
                use_float16=bool(use_float16),
                parsing_mode=str(parsing_mode),
                extra_margin=int(extra_margin),
                left_cheek_width=int(left_cheek_width),
                right_cheek_width=int(right_cheek_width),
            )
        return _RESIDENT_RUNTIME


def _process_lipsync_resident(
    *,
    video_path: str,
    audio_path: str,
    output_path: str,
    work_dir: str,
    root: str,
    version: str,
    bbox_shift: int,
    batch_size: int,
    use_float16: bool,
    parsing_mode: str,
    extra_margin: int,
    left_cheek_width: int,
    right_cheek_width: int,
    force_processing_fps: int,
    fixed_bbox: bool,
    bbox_sample_frames: int,
    mask_stride: int,
) -> dict[str, Any]:
    runtime = _resident_runtime(
        root=str(root),
        version=str(version),
        use_float16=bool(use_float16),
        parsing_mode=str(parsing_mode),
        extra_margin=int(extra_margin),
        left_cheek_width=int(left_cheek_width),
        right_cheek_width=int(right_cheek_width),
    )
    return runtime.process(
        video_path=str(video_path),
        audio_path=str(audio_path),
        output_path=str(output_path),
        work_dir=str(work_dir),
        bbox_shift=int(bbox_shift),
        batch_size=int(batch_size),
        force_processing_fps=int(force_processing_fps),
        fixed_bbox=bool(fixed_bbox),
        bbox_sample_frames=int(bbox_sample_frames),
        mask_stride=int(mask_stride),
    )


def _process_lipsync(
    *,
    video_path: str,
    audio_path: str,
    output_path: str,
    work_dir: str,
    root: str,
    python: str,
    version: str,
    bbox_shift: int,
    batch_size: int,
    use_float16: bool,
    parsing_mode: str,
    extra_margin: int,
    left_cheek_width: int,
    right_cheek_width: int,
    force_processing_fps: int,
) -> dict[str, Any]:
    ready = _check_musetalk_ready(root, python, version)
    if not ready.get("ready"):
        raise RuntimeError(f"MuseTalk is not ready: {ready}")

    started = time.perf_counter()
    timings: dict[str, float] = {}

    def mark(name: str, t0: float) -> None:
        timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - t0)

    input_video = str(video_path)
    if int(force_processing_fps or 0) > 0:
        t0 = time.perf_counter()
        prepared_video = os.path.join(str(work_dir), "musetalk_input_fps.mp4")
        prep = _transcode_video_fps(input_path=input_video, output_path=prepared_video, fps=int(force_processing_fps))
        input_video = prepared_video
        timings.update({k: float(v) for k, v in prep.items() if str(k).endswith("_sec")})

    cfg_path = os.path.join(str(work_dir), "musetalk_request.yaml")
    result_dir = os.path.join(str(work_dir), "musetalk_results")
    paths = _musetalk_paths(root, version)
    _write_musetalk_config(path=cfg_path, video_path=input_video, audio_path=str(audio_path), bbox_shift=int(bbox_shift))
    expected = os.path.join(result_dir, paths["version"], "output.mp4")
    cmd = [
        str(python),
        "-m",
        "scripts.inference",
        "--inference_config",
        str(cfg_path),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        paths["unet_model_path"],
        "--unet_config",
        paths["unet_config"],
        "--whisper_dir",
        os.path.join(str(root), "models", "whisper"),
        "--vae_type",
        "sd-vae",
        "--version",
        str(paths["version"]),
        "--batch_size",
        str(max(1, int(batch_size or 8))),
        "--output_vid_name",
        "output.mp4",
        "--parsing_mode",
        str(parsing_mode or "jaw"),
        "--extra_margin",
        str(max(0, int(extra_margin or 10))),
        "--left_cheek_width",
        str(max(0, int(left_cheek_width or 90))),
        "--right_cheek_width",
        str(max(0, int(right_cheek_width or 90))),
    ]
    if bool(use_float16):
        cmd.append("--use_float16")
    if paths["version"] == "v1":
        cmd.extend(["--bbox_shift", str(int(bbox_shift))])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(os.getenv("SMARTBLOG_MUSETALK_CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0")))
    env["HF_HOME"] = str(os.getenv("SMARTBLOG_MUSETALK_HF_HOME", os.path.join(str(root), ".hf")))
    env["HUGGING_FACE_HUB_TOKEN"] = str(
        os.getenv("SMARTBLOG_HF_TOKEN")
        or os.getenv("AVALIFE_HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or ""
    )
    if env["HUGGING_FACE_HUB_TOKEN"] and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = env["HUGGING_FACE_HUB_TOKEN"]

    LOG.warning(
        "MuseTalk lipsync start video=%s audio=%s version=%s batch=%d fp16=%d bbox_shift=%d force_fps=%d",
        os.path.basename(str(video_path)),
        os.path.basename(str(audio_path)),
        paths["version"],
        int(batch_size),
        1 if bool(use_float16) else 0,
        int(bbox_shift),
        int(force_processing_fps or 0),
    )
    t0 = time.perf_counter()
    _run(
        cmd,
        cwd=str(root),
        env=env,
        timeout=max(300.0, _probe_duration(str(audio_path)) * _env_float("SMARTBLOG_MUSETALK_TIMEOUT_PER_SEC", 60.0) + 300.0),
        label="MuseTalk inference",
    )
    mark("musetalk_inference", t0)
    if not os.path.exists(expected) or os.path.getsize(expected) <= 0:
        raise RuntimeError(f"MuseTalk output missing: {expected}")

    # The official script already muxes audio, but remux explicitly so separate
    # audio is always the final track and source-video audio is preserved when no
    # separate audio was supplied.
    t0 = time.perf_counter()
    _remux_final_audio(video_path=str(expected), audio_path=str(audio_path), output_path=str(output_path))
    mark("remux", t0)
    elapsed = float(time.perf_counter() - started)
    return {
        "ok": True,
        "output_path": str(output_path),
        "version": str(paths["version"]),
        "bbox_shift": int(bbox_shift),
        "batch_size": int(batch_size),
        "use_float16": bool(use_float16),
        "force_processing_fps": int(force_processing_fps or 0),
        "duration_sec": float(_probe_duration(str(output_path))),
        "bytes": int(os.path.getsize(str(output_path))) if os.path.exists(str(output_path)) else 0,
        "timings_sec": dict(timings),
        "elapsed_sec": float(elapsed),
    }


_SEM: asyncio.Semaphore | None = None


def _musetalk_semaphore() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(max(1, _env_int("SMARTBLOG_MUSETALK_MAX_CONCURRENT", 1)))
    return _SEM


async def health(_: web.Request) -> web.Response:
    root = str(os.getenv("SMARTBLOG_MUSETALK_ROOT", "/opt/MuseTalk") or "/opt/MuseTalk")
    python = str(os.getenv("SMARTBLOG_MUSETALK_PYTHON", "/opt/MuseTalk/venv/bin/python") or "/opt/MuseTalk/venv/bin/python")
    version = str(os.getenv("SMARTBLOG_MUSETALK_VERSION", "v15") or "v15")
    payload = _check_musetalk_ready(root, python, version)
    payload["backend_default"] = str(os.getenv("SMARTBLOG_MUSETALK_BACKEND", "resident") or "resident")
    payload["resident_loaded"] = bool(_RESIDENT_RUNTIME is not None)
    if _RESIDENT_RUNTIME is not None:
        payload["resident_key"] = list(getattr(_RESIDENT_RUNTIME, "key", ()) or ())
    return web.json_response(payload)


async def lipsync(request: web.Request) -> web.StreamResponse:
    secret = str(os.getenv("SMARTBLOG_MUSETALK_SHARED_SECRET", "") or "").strip()
    if secret:
        auth = str(request.headers.get("Authorization", "") or "").strip()
        header_secret = str(request.headers.get("X-SmartBlog-MuseTalk-Secret", "") or "").strip()
        if auth != f"Bearer {secret}" and header_secret != secret:
            raise web.HTTPUnauthorized(text="unauthorized")

    root = str(request.query.get("root") or os.getenv("SMARTBLOG_MUSETALK_ROOT", "/opt/MuseTalk") or "/opt/MuseTalk")
    python = str(
        request.query.get("python")
        or os.getenv("SMARTBLOG_MUSETALK_PYTHON", "/opt/MuseTalk/venv/bin/python")
        or "/opt/MuseTalk/venv/bin/python"
    )
    version = str(request.query.get("version") or os.getenv("SMARTBLOG_MUSETALK_VERSION", "v15") or "v15")
    bbox_shift = _env_int("SMARTBLOG_MUSETALK_BBOX_SHIFT", 0)
    batch_size = max(1, _env_int("SMARTBLOG_MUSETALK_BATCH_SIZE", 8))
    use_float16 = _env_flag("SMARTBLOG_MUSETALK_USE_FLOAT16", "1")
    parsing_mode = str(request.query.get("parsing_mode") or os.getenv("SMARTBLOG_MUSETALK_PARSING_MODE", "jaw") or "jaw")
    extra_margin = _env_int("SMARTBLOG_MUSETALK_EXTRA_MARGIN", 10)
    left_cheek_width = _env_int("SMARTBLOG_MUSETALK_LEFT_CHEEK_WIDTH", 90)
    right_cheek_width = _env_int("SMARTBLOG_MUSETALK_RIGHT_CHEEK_WIDTH", 90)
    force_processing_fps = _env_int("SMARTBLOG_MUSETALK_FORCE_PROCESSING_FPS", 0)
    backend = str(request.query.get("backend") or os.getenv("SMARTBLOG_MUSETALK_BACKEND", "resident") or "resident").strip().lower()
    fixed_bbox = _env_flag("SMARTBLOG_MUSETALK_FIXED_BBOX", "0")
    bbox_sample_frames = max(1, _env_int("SMARTBLOG_MUSETALK_BBOX_SAMPLE_FRAMES", 5))
    mask_stride = max(1, _env_int("SMARTBLOG_MUSETALK_MASK_STRIDE", 1))
    upload_url = str(request.query.get("upload_url") or request.query.get("output_upload_url") or "").strip()
    upload_content_type = str(request.query.get("upload_content_type") or "video/mp4").strip() or "video/mp4"
    source_url = str(request.query.get("source_url") or request.query.get("video_url") or "").strip()
    audio_url = str(request.query.get("audio_url") or "").strip()

    work_root = Path(os.getenv("SMARTBLOG_MUSETALK_WORK_DIR", "/tmp/smartblog-musetalk"))
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job-", dir=str(work_root)) as td:
        td_path = Path(td)
        video_path = str(td_path / "input.mp4")
        audio_path = str(td_path / "input_audio.wav")
        output_path = str(td_path / "output.mp4")
        got_video = False
        got_audio = False
        download_result: dict[str, Any] = {}
        audio_source = "separate"

        ctype = str(request.headers.get("Content-Type") or "").lower()
        if "multipart/" in ctype:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                name = str(part.name or "")
                if name in {"file", "video", "input", "source"}:
                    got_video = True
                    suffix = _safe_suffix(str(part.filename or ""), ".mp4")
                    video_path = str(td_path / f"input{suffix}")
                    with open(video_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(size=1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    continue
                if name == "audio":
                    got_audio = True
                    suffix = _safe_suffix(str(part.filename or ""), ".wav")
                    audio_path = str(td_path / f"audio{suffix}")
                    with open(audio_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(size=1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    continue
                text = (await part.text()).strip()
                if name in {"source_url", "video_url", "input_url"}:
                    source_url = text
                elif name == "audio_url":
                    audio_url = text
                elif name in {"upload_url", "output_upload_url"}:
                    upload_url = text
                elif name in {"bbox_shift", "bboxShift"}:
                    bbox_shift = int(float(text or bbox_shift))
                elif name in {"batch_size", "batchSize"}:
                    batch_size = max(1, int(float(text or batch_size)))
                elif name in {"version"}:
                    version = text or version
                elif name in {"use_float16", "fp16"}:
                    use_float16 = str(text).lower() not in {"0", "false", "no", "off"}
                elif name in {"force_processing_fps", "processing_fps", "fps"}:
                    force_processing_fps = max(0, int(float(text or 0)))
                elif name in {"backend", "mode"}:
                    backend = str(text or backend).strip().lower()
                elif name in {"fixed_bbox", "fixedBbox", "fixed_crop", "fixedCrop"}:
                    fixed_bbox = str(text).lower() not in {"0", "false", "no", "off"}
                elif name in {"bbox_sample_frames", "bboxSampleFrames"}:
                    bbox_sample_frames = max(1, int(float(text or bbox_sample_frames)))
                elif name in {"mask_stride", "maskStride"}:
                    mask_stride = max(1, int(float(text or mask_stride)))
                elif name in {"parsing_mode", "parsingMode"}:
                    parsing_mode = text or parsing_mode
                elif name in {"extra_margin", "extraMargin"}:
                    extra_margin = max(0, int(float(text or extra_margin)))
                elif name in {"left_cheek_width", "leftCheekWidth"}:
                    left_cheek_width = max(0, int(float(text or left_cheek_width)))
                elif name in {"right_cheek_width", "rightCheekWidth"}:
                    right_cheek_width = max(0, int(float(text or right_cheek_width)))
        else:
            body = await request.read()
            payload: dict[str, Any] = {}
            if body:
                payload = json.loads(body.decode("utf-8") or "{}")
            source_url = _text(payload.get("source_url") or payload.get("video_url") or source_url)
            audio_url = _text(payload.get("audio_url") or audio_url)
            upload_url = _text(payload.get("upload_url") or payload.get("output_upload_url") or upload_url)
            if payload.get("video_base64"):
                got_video = True
                _write_base64_file(str(payload.get("video_base64")), path=video_path)
            if payload.get("audio_base64"):
                got_audio = True
                audio_path = str(td_path / "audio.wav")
                _write_base64_file(str(payload.get("audio_base64")), path=audio_path)
            bbox_shift = int(payload.get("bbox_shift") if payload.get("bbox_shift") is not None else bbox_shift)
            batch_size = max(1, int(payload.get("batch_size") or batch_size))
            version = _text(payload.get("version") or version)
            if payload.get("use_float16") is not None:
                use_float16 = bool(payload.get("use_float16"))
            force_processing_fps = max(0, int(payload.get("force_processing_fps") or payload.get("processing_fps") or force_processing_fps))
            backend = _text(payload.get("backend") or payload.get("mode") or backend).strip().lower()
            if payload.get("fixed_bbox") is not None or payload.get("fixedBbox") is not None:
                fixed_bbox = bool(payload.get("fixed_bbox") if payload.get("fixed_bbox") is not None else payload.get("fixedBbox"))
            bbox_sample_frames = max(1, int(payload.get("bbox_sample_frames") or payload.get("bboxSampleFrames") or bbox_sample_frames))
            mask_stride = max(1, int(payload.get("mask_stride") or payload.get("maskStride") or mask_stride))
            parsing_mode = _text(payload.get("parsing_mode") or parsing_mode)
            extra_margin = max(0, int(payload.get("extra_margin") or extra_margin))
            left_cheek_width = max(0, int(payload.get("left_cheek_width") or left_cheek_width))
            right_cheek_width = max(0, int(payload.get("right_cheek_width") or right_cheek_width))

        if not got_video:
            if not source_url:
                raise web.HTTPBadRequest(text="video file, video_base64, source_url, or video_url is required")
            download_result.update(
                await asyncio.to_thread(_download_url_to_file, url=str(source_url), path=str(video_path))
            )
            got_video = True
        if not got_audio:
            if audio_url:
                audio_source = "audio_url"
                audio_path = str(td_path / f"audio{_safe_suffix(audio_url, '.wav')}")
                audio_download = await asyncio.to_thread(_download_url_to_file, url=str(audio_url), path=str(audio_path))
                download_result["audio_source_bytes"] = int(audio_download.get("source_bytes") or 0)
                download_result["audio_download_sec"] = float(audio_download.get("source_download_sec") or 0.0)
                got_audio = True
            else:
                audio_source = "video"
                extract_result = await asyncio.to_thread(_extract_audio, video_path=str(video_path), output_audio_path=str(audio_path))
                download_result.update(extract_result)
                got_audio = True

        queue_started = time.perf_counter()
        sem = _musetalk_semaphore()
        if sem.locked():
            LOG.warning("MuseTalk lipsync queued: max_concurrent=%s", _env_int("SMARTBLOG_MUSETALK_MAX_CONCURRENT", 1))
        async with sem:
            queue_sec = float(time.perf_counter() - queue_started)
            if backend in {"resident", "fast", "realtime"}:
                try:
                    result = await asyncio.to_thread(
                        _process_lipsync_resident,
                        video_path=str(video_path),
                        audio_path=str(audio_path),
                        output_path=str(output_path),
                        work_dir=str(td_path),
                        root=str(root),
                        version=str(version),
                        bbox_shift=int(bbox_shift),
                        batch_size=int(batch_size),
                        use_float16=bool(use_float16),
                        parsing_mode=str(parsing_mode),
                        extra_margin=int(extra_margin),
                        left_cheek_width=int(left_cheek_width),
                        right_cheek_width=int(right_cheek_width),
                        force_processing_fps=int(force_processing_fps),
                        fixed_bbox=bool(fixed_bbox),
                        bbox_sample_frames=int(bbox_sample_frames),
                        mask_stride=int(mask_stride),
                    )
                except Exception:
                    if not _env_flag("SMARTBLOG_MUSETALK_RESIDENT_FALLBACK_SUBPROCESS", "0"):
                        raise
                    LOG.exception("MuseTalk resident backend failed; falling back to subprocess backend")
                    result = await asyncio.to_thread(
                        _process_lipsync,
                        video_path=str(video_path),
                        audio_path=str(audio_path),
                        output_path=str(output_path),
                        work_dir=str(td_path),
                        root=str(root),
                        python=str(python),
                        version=str(version),
                        bbox_shift=int(bbox_shift),
                        batch_size=int(batch_size),
                        use_float16=bool(use_float16),
                        parsing_mode=str(parsing_mode),
                        extra_margin=int(extra_margin),
                        left_cheek_width=int(left_cheek_width),
                        right_cheek_width=int(right_cheek_width),
                        force_processing_fps=int(force_processing_fps),
                    )
                    result["backend"] = "subprocess_fallback"
            else:
                result = await asyncio.to_thread(
                    _process_lipsync,
                    video_path=str(video_path),
                    audio_path=str(audio_path),
                    output_path=str(output_path),
                    work_dir=str(td_path),
                    root=str(root),
                    python=str(python),
                    version=str(version),
                    bbox_shift=int(bbox_shift),
                    batch_size=int(batch_size),
                    use_float16=bool(use_float16),
                    parsing_mode=str(parsing_mode),
                    extra_margin=int(extra_margin),
                    left_cheek_width=int(left_cheek_width),
                    right_cheek_width=int(right_cheek_width),
                    force_processing_fps=int(force_processing_fps),
                )
                result["backend"] = "subprocess"
            result["queue_sec"] = float(queue_sec)
            result["audio_source"] = str(audio_source)
            result.update(download_result)
            if upload_url:
                upload_result = await asyncio.to_thread(
                    _upload_file_to_signed_url,
                    signed_url=str(upload_url),
                    path=str(output_path),
                    content_type=str(upload_content_type),
                )
                result.update(upload_result)
                LOG.warning(
                    "MuseTalk lipsync uploaded: bytes=%s duration=%.3fs queue=%.3fs elapsed=%.3fs source_audio=%s",
                    result.get("bytes"),
                    float(result.get("duration_sec") or 0.0),
                    float(result.get("queue_sec") or 0.0),
                    float(result.get("elapsed_sec") or 0.0),
                    str(audio_source),
                )
                return web.json_response(result)

            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "video/mp4",
                    "X-SmartBlog-MuseTalk-Elapsed-Sec": f"{float(result.get('elapsed_sec') or 0.0):.3f}",
                    "X-SmartBlog-MuseTalk-Audio-Source": str(audio_source),
                    "X-SmartBlog-MuseTalk-Version": str(result.get("version") or ""),
                },
            )
            await response.prepare(request)
            with open(str(output_path), "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    await response.write(chunk)
            await response.write_eof()
            return response


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, str(os.getenv("SMARTBLOG_MUSETALK_SERVICE_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app = web.Application(client_max_size=max(1, _env_int("SMARTBLOG_MUSETALK_CLIENT_MAX_MB", 4096)) * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/lipsync", lipsync)
    host = str(os.getenv("SMARTBLOG_MUSETALK_SERVICE_HOST", "0.0.0.0") or "0.0.0.0")
    port = _env_int("SMARTBLOG_MUSETALK_SERVICE_PORT", 8800)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
