from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from avalife.core.upload_retry import put_file_to_signed_url


LOG = logging.getLogger("vlogme_render_worker")


class VlogMeJobStopped(Exception):
    pass


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


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


def _text(value: Any) -> str:
    return str(value or "").strip()


def _run(cmd: list[str], *, timeout: float, label: str) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(timeout), text=True)
    if int(proc.returncode or 0) != 0:
        stderr = str(proc.stderr or "")[-4000:]
        stdout = str(proc.stdout or "")[-1200:]
        raise RuntimeError(f"{label} failed rc={proc.returncode}: {stderr or stdout}")


def _run_ffmpeg_encode(
    base_cmd: list[str],
    output_path: str,
    *,
    timeout: float,
    label: str,
    crf: int = 20,
) -> None:
    requested = _text(os.getenv("VLOGME_VIDEOEDIT_ENCODER") or os.getenv("SMARTBLOG_VIDEOEDIT_ENCODER") or "h264_nvenc")
    encoders = [requested]
    if requested != "libx264":
        encoders.append("libx264")
    last_error: Exception | None = None
    for encoder in encoders:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if Path(output_path).exists():
                Path(output_path).unlink()
            if encoder == "libx264":
                video_args = ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf))]
            else:
                video_args = [
                    "-c:v",
                    str(encoder),
                    "-preset",
                    _text(os.getenv("VLOGME_VIDEOEDIT_NVENC_PRESET") or "p4"),
                    "-cq",
                    str(int(crf)),
                    "-b:v",
                    "0",
                ]
            cmd = [
                *base_cmd,
                *video_args,
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            _run(cmd, timeout=float(timeout), label=f"{label} ({encoder})")
            return
        except Exception as e:
            last_error = e
            if encoder == "libx264":
                break
            LOG.warning("%s failed with %s, retrying libx264", label, encoder)
    raise RuntimeError(str(last_error or f"{label} failed"))


def _probe(path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, text=True)
    if int(proc.returncode or 0) != 0:
        raise RuntimeError(str(proc.stderr or "")[-2000:])
    try:
        return dict(json.loads(str(proc.stdout or "{}")))
    except Exception as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}") from e


def _has_audio_stream(path: str) -> bool:
    try:
        data = _probe(path)
        return any(s.get("codec_type") == "audio" for s in list(data.get("streams") or []))
    except Exception:
        return False


def _probe_video_summary(path: str) -> dict[str, Any]:
    data = _probe(path)
    streams = list(data.get("streams") or [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = dict(data.get("format") or {})
    fps = 0.0
    raw_rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "").strip()
    if "/" in raw_rate:
        a, b = raw_rate.split("/", 1)
        try:
            fps = float(a) / max(1.0, float(b))
        except Exception:
            fps = 0.0
    else:
        try:
            fps = float(raw_rate or 0.0)
        except Exception:
            fps = 0.0
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": float(fps or 0.0),
        "frames": int(float(str(video.get("nb_frames") or 0) or 0)),
        "duration": float(fmt.get("duration") or video.get("duration") or 0.0),
        "bytes": int(float(str(fmt.get("size") or 0) or 0)),
    }


def _download(url: str, path: str) -> int:
    src = _text(url)
    if not src:
        raise RuntimeError("download URL is empty")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with requests.get(src, stream=True, timeout=(20, 1800)) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
    if total <= 0:
        raise RuntimeError(f"downloaded zero bytes from {src}")
    return int(total)


def _ffmpeg_filter_escape(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace(",", "\\,")
    text = text.replace("\n", " ")
    return text


def _concat_demuxer_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        out = int(float(value))
        return int(out)
    except Exception:
        return int(default)


def _even(value: int, minimum: int = 2) -> int:
    out = max(int(minimum), int(value))
    return out - (out % 2)


def _atempo_chain(factor: float) -> list[str]:
    factor = max(0.25, min(4.0, float(factor or 1.0)))
    parts: list[str] = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.6f}")
    return parts


def _write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _ceil_to_multiple_plus_one(frames: int, *, multiple: int) -> int:
    frames_i = max(1, int(frames))
    multiple_i = max(1, int(multiple))
    if frames_i <= 1:
        return 1
    return int(math.ceil(float(frames_i - 1) / float(multiple_i)) * multiple_i + 1)


def _aspect_is_landscape(job: dict[str, Any]) -> bool:
    assets = _claim_assets(job)
    orientation = _text(assets.get("orientation")).lower()
    if orientation in {"landscape", "horizontal", "16:9"}:
        return True
    if orientation in {"portrait", "vertical", "9:16"}:
        return False
    video_size = assets.get("video_size") if isinstance(assets.get("video_size"), dict) else {}
    try:
        width = int(video_size.get("width") or 0)
        height = int(video_size.get("height") or 0)
        if width > 0 and height > 0:
            return width > height
    except Exception:
        pass
    output = job.get("output") if isinstance(job.get("output"), dict) else {}
    aspect = _text(output.get("aspect_ratio") or output.get("aspectRatio")).lower()
    if aspect in {"16:9", "landscape", "horizontal"}:
        return True
    if aspect in {"9:16", "portrait", "vertical"}:
        return False
    try:
        width = int(output.get("width") or 0)
        height = int(output.get("height") or 0)
        if width > 0 and height > 0:
            return width > height
    except Exception:
        pass
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    aspect = _text(settings.get("aspect_ratio") or settings.get("aspectRatio")).lower()
    return aspect in {"16:9", "landscape", "horizontal"}


def _output_size(job: dict[str, Any]) -> tuple[int, int]:
    assets = _claim_assets(job)
    video_size = assets.get("video_size") if isinstance(assets.get("video_size"), dict) else {}
    try:
        width = int(video_size.get("width") or 0)
        height = int(video_size.get("height") or 0)
        if width > 0 and height > 0:
            return width - width % 2, height - height % 2
    except Exception:
        pass
    output = job.get("output") if isinstance(job.get("output"), dict) else {}
    try:
        width = int(output.get("width") or 0)
        height = int(output.get("height") or 0)
        if width > 0 and height > 0:
            return width - width % 2, height - height % 2
    except Exception:
        pass
    return (1280, 720) if _aspect_is_landscape(job) else (720, 1280)


def _duration_seconds(job: dict[str, Any]) -> float:
    assets = _claim_assets(job)
    video = _claim_video(job)
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    candidates = [
        assets.get("duration_seconds"),
        assets.get("durationSeconds"),
        video.get("duration_seconds"),
        video.get("durationSeconds"),
        settings.get("duration_seconds"),
        settings.get("durationSeconds"),
        settings.get("duration_sec"),
        settings.get("duration"),
        job.get("duration_seconds"),
        job.get("durationSeconds"),
    ]
    for value in candidates:
        try:
            out = float(value)
            if out > 0:
                return float(out)
        except Exception:
            continue
    return _env_float("VLOGME_RENDER_DEFAULT_DURATION_SEC", 5.0)


def _select_image_url(job: dict[str, Any]) -> str:
    assets = _claim_assets(job)
    direct = _text(assets.get("avatar_url") or assets.get("image_url") or assets.get("first_frame_url"))
    if direct:
        return direct
    avatar_urls = assets.get("avatar_urls")
    if isinstance(avatar_urls, list):
        for item in avatar_urls:
            url = _text(item)
            if url:
                return url
    asset_inputs = assets.get("inputs")
    if isinstance(asset_inputs, list):
        ranked_assets: list[tuple[int, str]] = []
        priority = {
            "first_frame": 0,
            "start_frame": 0,
            "character_photo": 0,
            "style_reference": 1,
            "style": 1,
            "image": 1,
            "source_image": 1,
            "input_image": 1,
            "reference": 2,
            "reference_image": 2,
            "source": 3,
            "input": 3,
        }
        for item in asset_inputs:
            if not isinstance(item, dict):
                continue
            url = _text(item.get("url") or item.get("signed_url") or item.get("signedUrl"))
            if not url:
                continue
            role = _text(item.get("role") or item.get("slot") or item.get("id")).lower()
            rank = priority.get(role, 50)
            ranked_assets.append((int(rank), url))
        if ranked_assets:
            ranked_assets.sort(key=lambda x: x[0])
            return str(ranked_assets[0][1])
    inputs = job.get("inputs")
    if not isinstance(inputs, list):
        inputs = []
    priority = {
        "first_frame": 0,
        "start_frame": 0,
        "image": 1,
        "source_image": 1,
        "input_image": 1,
        "reference": 2,
        "reference_image": 2,
        "source": 3,
        "input": 3,
    }
    ranked: list[tuple[int, str]] = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        url = _text(item.get("url") or item.get("signed_url") or item.get("signedUrl"))
        if not url:
            continue
        role = _text(item.get("role") or item.get("slot") or item.get("id")).lower()
        rank = min(priority.get(part, 50) for part in [role] if part) if role else 50
        ranked.append((int(rank), url))
    if ranked:
        ranked.sort(key=lambda x: x[0])
        return str(ranked[0][1])
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    return _text(settings.get("image_url") or settings.get("imageUrl") or settings.get("first_frame_url"))


def _settings_audio(job: dict[str, Any]) -> dict[str, Any]:
    video = job.get("video") if isinstance(job.get("video"), dict) else {}
    audio = video.get("audio")
    if isinstance(audio, dict):
        return audio
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    audio = settings.get("audio")
    return audio if isinstance(audio, dict) else {}


def _claim_job(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("job")
    return value if isinstance(value, dict) else {}


def _claim_assets(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("assets")
    return value if isinstance(value, dict) else {}


def _claim_video(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("video")
    return value if isinstance(value, dict) else {}


def _claim_upload(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("upload")
    return value if isinstance(value, dict) else {}


def _job_id(job: dict[str, Any]) -> str:
    claim_job = _claim_job(job)
    return _text(claim_job.get("id") or job.get("job_id") or job.get("id"))


def _job_type(job: dict[str, Any]) -> str:
    claim_job = _claim_job(job)
    return _text(claim_job.get("job_type") or job.get("job_type") or job.get("operation_type"))


def _provider(job: dict[str, Any]) -> str:
    claim_job = _claim_job(job)
    return _text(claim_job.get("provider") or job.get("provider"))


def _render_mode(job: dict[str, Any], *, image_url: str = "") -> str:
    assets = _claim_assets(job)
    mode = _text(assets.get("render_mode") or assets.get("renderMode")).lower()
    if mode in {"i2v", "image_to_video", "image2video"}:
        return "i2v"
    if mode in {"t2v", "text_to_video", "text2video"}:
        return "t2v"
    return "i2v" if _text(image_url) else "t2v"


def _prompt(job: dict[str, Any]) -> str:
    video = _claim_video(job)
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    return _text(
        video.get("effective_prompt")
        or video.get("effectivePrompt")
        or video.get("prompt")
        or job.get("effective_prompt")
        or job.get("prompt")
        or settings.get("effective_prompt")
        or settings.get("prompt")
    )


def _negative_prompt(job: dict[str, Any]) -> str:
    video = _claim_video(job)
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    return _text(
        video.get("effective_negative_prompt")
        or video.get("effectiveNegativePrompt")
        or video.get("negative_prompt")
        or video.get("negativePrompt")
        or job.get("negative_prompt")
        or settings.get("negative_prompt")
        or settings.get("negativePrompt")
    )


def _mux_audio(*, video_path: str, audio_path: str, out_path: str, duration_sec: float) -> None:
    cmd = [
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
        "-t",
        f"{max(0.1, float(duration_sec)):.3f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run(cmd, timeout=max(120.0, float(duration_sec) * 30.0), label="mux audio")


def _mux_silence(*, video_path: str, out_path: str, duration_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(video_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{max(0.1, float(duration_sec)):.3f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run(cmd, timeout=max(120.0, float(duration_sec) * 30.0), label="mux silence")


class VlogMeRenderWorker:
    def __init__(self) -> None:
        self.api_mode = _text(os.getenv("VLOGME_RENDER_API_MODE") or "worker_api").lower()
        if self.api_mode in {"jsonrpc", "json_rpc", "worker-api"}:
            self.api_mode = "worker_api"
        if self.api_mode in {"legacy", "legacy-render", "render"}:
            self.api_mode = "legacy_render"
        self.base_url = _text(
            os.getenv("VLOGME_RENDER_LEGACY_API_BASE")
            or os.getenv("VLOGME_RENDER_API_BASE")
            or "https://vlogme.ai/api/public/v1/render"
        ).rstrip("/")
        configured_worker_api = _text(
            os.getenv("VLOGME_WORKER_API_URL")
            or os.getenv("VLOGME_RENDER_WORKER_API_URL")
            or os.getenv("VLOGME_WORKER_API_BASE")
        ).rstrip("/")
        if not configured_worker_api and "worker-api" in self.base_url:
            configured_worker_api = self.base_url
        self.worker_api_url = configured_worker_api or "https://vlogme.ai/api/public/v1/worker-api"
        self.token = _text(
            os.getenv("VLOGME_WORKER_API_KEY")
            or os.getenv("WORKER_API_KEY")
            or os.getenv("POSTPROCESSING_WORKER_TOKEN")
            or os.getenv("VLOGME_RENDER_WORKER_TOKEN")
        )
        self.worker_id = _text(os.getenv("VLOGME_RENDER_WORKER_ID")) or f"{socket.gethostname()}-render"
        self.job_types = [
            item.strip()
            for item in _text(os.getenv("VLOGME_RENDER_JOB_TYPES") or os.getenv("VLOGME_WORKER_JOB_TYPES") or "visual_scene").split(",")
            if item.strip()
        ] or ["visual_scene"]
        self.poll_sleep_sec = max(1.0, _env_float("VLOGME_RENDER_POLL_SLEEP_SEC", 5.0))
        self.enabled = _env_flag("VLOGME_RENDER_WORKER_ENABLED", "0")
        self.work_root = Path(os.getenv("VLOGME_RENDER_WORK_DIR", "/tmp/vlogme-render-worker")).resolve()
        self.hunyuan_url = _text(
            os.getenv("VLOGME_RENDER_HUNYUAN_SERVICE_URL")
            or os.getenv("SMARTBLOG_HUNYUAN_SERVICE_URL")
            or "http://127.0.0.1:8798"
        ).rstrip("/")
        self.mmaudio_url = _text(
            os.getenv("VLOGME_RENDER_MMAUDIO_SERVICE_URL")
            or os.getenv("SMARTBLOG_MMAUDIO_SERVICE_URL")
            or "http://127.0.0.1:8799"
        ).rstrip("/")
        self.finalizer_url = _text(
            os.getenv("VLOGME_RENDER_FINALIZER_URL")
            or os.getenv("SMARTBLOG_FILE_UPSCALE_SERVICE_URL")
            or os.getenv("REMOTE_EDGE_FILE_UPSCALE_SERVICE_URL")
            or "http://127.0.0.1:8888/upscale"
        )
        self.finalizer_secret = _text(
            os.getenv("VLOGME_RENDER_FINALIZER_SHARED_SECRET")
            or os.getenv("SMARTBLOG_UPSCALE_SHARED_SECRET")
            or os.getenv("REMOTE_EDGE_FILE_UPSCALE_SHARED_SECRET")
        )
        # Hunyuan is a fixed 16fps generator in our pipeline. Public APIs may
        # carry fps fields for other providers, but this worker must ignore
        # them and normalize to 16fps before the finalizer interpolates to 30.
        self.hunyuan_fps = 16
        self.output_fps = max(1, _env_int("VLOGME_RENDER_OUTPUT_FPS", 30))
        self.hunyuan_steps = max(1, _env_int("VLOGME_RENDER_HUNYUAN_STEPS", 8))
        self.max_duration_sec = max(1.0, _env_float("VLOGME_RENDER_MAX_DURATION_SEC", 30.0))
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        self._progress_lock = threading.Lock()
        self._last_progress_payload: tuple[str, int, str] | None = None
        self._last_progress_mono = 0.0

    def run_forever(self) -> None:
        if not self.enabled:
            LOG.info("VlogMe render worker disabled")
            return
        if not self.token:
            raise RuntimeError(
                "VLOGME_WORKER_API_KEY/WORKER_API_KEY is required when VLOGME_RENDER_WORKER_ENABLED=1"
            )
        self.work_root.mkdir(parents=True, exist_ok=True)
        LOG.warning(
            "VlogMe render worker started worker_id=%s mode=%s api=%s job_types=%s",
            self.worker_id,
            self.api_mode,
            self.worker_api_url if self.api_mode == "worker_api" else self.base_url,
            ",".join(self.job_types),
        )
        while True:
            try:
                job = self.claim_next()
                if not job:
                    time.sleep(float(self.poll_sleep_sec))
                    continue
                self.handle_job(job)
            except KeyboardInterrupt:
                raise
            except Exception:
                LOG.exception("VlogMe render worker loop error")
                time.sleep(float(self.poll_sleep_sec))

    def worker_api_action(self, payload: dict[str, Any], *, timeout: tuple[float, float] = (10, 120)) -> dict[str, Any]:
        resp = requests.post(self.worker_api_url, headers=self.headers, json=payload, timeout=timeout)
        if int(resp.status_code) == 404:
            LOG.warning("VlogMe worker-api not found: %s", self.worker_api_url)
            time.sleep(30)
            raise RuntimeError(f"VlogMe worker-api not found: {self.worker_api_url}")
        if int(resp.status_code) < 200 or int(resp.status_code) >= 300:
            raise RuntimeError(f"VlogMe worker-api {payload.get('action')} failed HTTP {resp.status_code}: {resp.text[:2000]}")
        parsed = resp.json()
        if not isinstance(parsed, dict):
            raise RuntimeError(f"VlogMe worker-api {payload.get('action')} returned non-object response")
        if parsed.get("error"):
            raise RuntimeError(f"VlogMe worker-api {payload.get('action')} error: {parsed.get('error')}")
        return parsed

    def resolve_library_asset(self, *, job_id: str, item_id: str) -> dict[str, Any]:
        out = self.worker_api_action(
            {"action": "resolve_library_asset", "job_id": str(job_id), "item_id": str(item_id)},
            timeout=(10, 120),
        )
        url = _text(out.get("url") or out.get("signed_url") or out.get("download_url"))
        if not url:
            raise RuntimeError(f"resolve_library_asset returned no URL for item={item_id}")
        return out

    def claim_next_legacy(self) -> dict[str, Any] | None:
        url = f"{self.base_url}/next"
        resp = requests.get(url, params={"worker_id": self.worker_id}, headers={"Authorization": f"Bearer {self.token}"}, timeout=(10, 60))
        if int(resp.status_code) == 204:
            return None
        if int(resp.status_code) == 404:
            LOG.warning("VlogMe render API not found: %s", url)
            time.sleep(30)
            return None
        if int(resp.status_code) == 409:
            return None
        resp.raise_for_status()
        parsed = resp.json()
        if not isinstance(parsed, dict) or not _text(parsed.get("job_id")):
            LOG.warning("VlogMe render claim returned unexpected payload: %s", str(parsed)[:1000])
            return None
        return parsed

    def claim_next(self) -> dict[str, Any] | None:
        if self.api_mode == "legacy_render":
            return self.claim_next_legacy()
        for job_type in self.job_types:
            poll = self.worker_api_action(
                {
                    "action": "poll",
                    "job_type": str(job_type),
                    "worker_id": str(self.worker_id),
                    "limit": max(1, _env_int("VLOGME_RENDER_POLL_LIMIT", 10)),
                },
                timeout=(10, 60),
            )
            jobs = poll.get("jobs") if isinstance(poll.get("jobs"), list) else []
            for item in jobs:
                if not isinstance(item, dict):
                    continue
                job_id = _text(item.get("id") or item.get("job_id"))
                if not job_id:
                    continue
                claim = self.worker_api_action(
                    {"action": "claim", "job_id": str(job_id), "worker_id": str(self.worker_id)},
                    timeout=(10, 120),
                )
                if bool(claim.get("claimed")):
                    return claim
                reason = _text(claim.get("reason") or claim.get("details") or "not_claimed")
                LOG.warning("VlogMe worker-api claim skipped job=%s reason=%s", job_id, reason)
                if bool(claim.get("stop")):
                    continue
        return None

    def post_progress(self, job_id: str, progress: int, stage: str, *, lease_extend_minutes: int = 15) -> None:
        try:
            progress_i = int(max(0, min(99, int(progress))))
            if self.api_mode == "worker_api":
                out = self.worker_api_action(
                    {
                        "action": "progress",
                        "job_id": str(job_id),
                        "progress": int(progress_i),
                        "stage": str(stage),
                        "stage_label": str(stage).replace("_", " ").title(),
                    },
                    timeout=(10, 60),
                )
                if bool(out.get("stop")):
                    raise VlogMeJobStopped(str(out.get("reason") or "VlogMe API requested stop"))
                with self._progress_lock:
                    self._last_progress_payload = (str(job_id), int(progress_i), str(stage))
                    self._last_progress_mono = float(time.monotonic())
                return
            payload = {
                "progress": int(progress_i),
                "stage": str(stage),
                "lease_extend_minutes": int(max(1, lease_extend_minutes)),
            }
            resp = requests.post(
                f"{self.base_url}/{urllib.parse.quote(str(job_id), safe='')}/progress",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
                timeout=(10, 60),
            )
            if int(resp.status_code) >= 400:
                LOG.warning("VlogMe progress failed job=%s status=%s body=%s", job_id, resp.status_code, resp.text[:500])
            else:
                with self._progress_lock:
                    self._last_progress_payload = (str(job_id), int(progress_i), str(stage))
                    self._last_progress_mono = float(time.monotonic())
        except VlogMeJobStopped:
            raise
        except Exception:
            LOG.exception("VlogMe progress exception job=%s stage=%s", job_id, stage)

    def post_fail(self, job_id: str, error: str, *, retryable: bool = True) -> None:
        try:
            if self.api_mode == "worker_api":
                self.worker_api_action(
                    {
                        "action": "fail",
                        "job_id": str(job_id),
                        "error_text": str(error)[-2000:] or "worker failed",
                        "retryable": bool(retryable),
                    },
                    timeout=(10, 60),
                )
                return
            payload = {"error": str(error)[-4000:], "retryable": bool(retryable)}
            resp = requests.post(
                f"{self.base_url}/{urllib.parse.quote(str(job_id), safe='')}/fail",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
                timeout=(10, 60),
            )
            if int(resp.status_code) >= 400:
                LOG.warning("VlogMe fail failed job=%s status=%s body=%s", job_id, resp.status_code, resp.text[:1000])
        except Exception:
            LOG.exception("VlogMe fail exception job=%s", job_id)

    def post_complete(
        self,
        job_id: str,
        *,
        storage_path: str,
        duration_sec: float,
        width: int,
        height: int,
        file_size_bytes: int,
        output_url: str = "",
    ) -> None:
        payload: dict[str, Any] = {
            "duration_seconds": float(duration_sec),
            "width": int(width),
            "height": int(height),
            "file_size_bytes": int(file_size_bytes),
        }
        if output_url:
            payload["output_url"] = str(output_url)
        if storage_path:
            payload["storage_path"] = str(storage_path)
        if self.api_mode == "worker_api":
            payload["action"] = "complete"
            payload["job_id"] = str(job_id)
            self.worker_api_action(payload, timeout=(10, 120))
            return
        resp = requests.post(
            f"{self.base_url}/{urllib.parse.quote(str(job_id), safe='')}/complete",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
            timeout=(10, 120),
        )
        if int(resp.status_code) < 200 or int(resp.status_code) >= 300:
            raise RuntimeError(f"VlogMe complete failed HTTP {resp.status_code}: {resp.text[:2000]}")

    def start_progress_keepalive(self, job_id: str) -> threading.Event:
        stop = threading.Event()
        interval = max(5.0, min(45.0, _env_float("VLOGME_RENDER_JOB_HEARTBEAT_SEC", 20.0)))
        stale_sec = max(interval, min(55.0, _env_float("VLOGME_RENDER_JOB_HEARTBEAT_STALE_SEC", 35.0)))

        def _loop() -> None:
            while not stop.wait(float(interval)):
                with self._progress_lock:
                    payload = self._last_progress_payload
                    last_mono = float(self._last_progress_mono or 0.0)
                if last_mono > 0.0 and (time.monotonic() - last_mono) < float(stale_sec):
                    continue
                if payload is None or payload[0] != str(job_id):
                    progress_i, stage_s = 1, "heartbeat"
                else:
                    _, progress_i, stage_s = payload
                try:
                    self.post_progress(str(job_id), int(progress_i), str(stage_s or "heartbeat"))
                except VlogMeJobStopped as e:
                    LOG.warning("VlogMe render keepalive stopped by API job=%s reason=%s", job_id, e)
                    stop.set()
                    return
                except Exception:
                    LOG.exception("VlogMe render keepalive failed open job=%s", job_id)

        thread = threading.Thread(target=_loop, name=f"vlogme-render-keepalive-{job_id}", daemon=True)
        thread.start()
        return stop

    def handle_job(self, job: dict[str, Any]) -> None:
        job_id = _job_id(job)
        if not job_id:
            raise RuntimeError("VlogMe claim has no job id")
        run_dir = self.work_root / f"{int(time.time())}_{job_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(str(run_dir / "claim.json"), job)
        started = time.perf_counter()
        keepalive_stop = self.start_progress_keepalive(str(job_id))
        try:
            LOG.warning(
                "VlogMe render job claimed id=%s type=%s provider=%s mode=%s",
                job_id,
                _job_type(job),
                _provider(job),
                _text(_claim_assets(job).get("render_mode") or "-"),
            )
            self.post_progress(job_id, 1, "claimed")

            if _job_type(job).lower() == "video_edit" or _provider(job).lower() == "vlogme_videoedit":
                self.handle_video_edit_job(job, run_dir=run_dir, started=started)
                return

            image_url = _select_image_url(job)
            render_mode = _render_mode(job, image_url=image_url)
            if render_mode != "t2v" and not image_url:
                raise RuntimeError("render job has no input image URL")
            image_path = run_dir / "input_image"
            image_bytes = 0
            if image_url:
                self.post_progress(job_id, 4, "download_input")
                image_bytes = _download(image_url, str(image_path))

            duration = min(float(self.max_duration_sec), max(0.5, _duration_seconds(job)))
            prompt = _prompt(job)
            negative_prompt = _negative_prompt(job)
            if not prompt:
                prompt = "cinematic natural motion, high quality video"

            is_landscape = _aspect_is_landscape(job)
            src_w = _env_int("VLOGME_RENDER_HUNYUAN_LANDSCAPE_WIDTH", 832) if is_landscape else _env_int("VLOGME_RENDER_HUNYUAN_PORTRAIT_WIDTH", 480)
            src_h = _env_int("VLOGME_RENDER_HUNYUAN_LANDSCAPE_HEIGHT", 480) if is_landscape else _env_int("VLOGME_RENDER_HUNYUAN_PORTRAIT_HEIGHT", 832)
            out_w, out_h = _output_size(job)
            raw_frames = int(math.ceil(float(duration) * float(self.hunyuan_fps))) + 1
            num_frames = max(9, _ceil_to_multiple_plus_one(raw_frames, multiple=_env_int("VLOGME_RENDER_HUNYUAN_FRAME_MULTIPLE", 8)))

            hunyuan_out_dir = run_dir / "hunyuan_out"
            hunyuan_path = Path("")
            self.post_progress(job_id, 8, "hunyuan_start", lease_extend_minutes=30)
            hy_started = time.perf_counter()
            hy_payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": int(src_w),
                "height": int(src_h),
                "num_frames": int(num_frames),
                "frame_rate": int(self.hunyuan_fps),
                "num_inference_steps": int(self.hunyuan_steps),
                "output_path": str(hunyuan_out_dir),
                "task": str(render_mode),
                "render_mode": str(render_mode),
            }
            if render_mode != "t2v":
                hy_payload.update(
                    {
                        "image_path": str(image_path),
                        "conditioning_media_paths": [str(image_path)],
                    }
                )
            hy_resp = requests.post(f"{self.hunyuan_url}/generate", json=hy_payload, timeout=(20, max(900, int(duration * 180))))
            if int(hy_resp.status_code) >= 400:
                raise RuntimeError(f"Hunyuan generate failed HTTP {hy_resp.status_code}: {hy_resp.text[:2000]}")
            hy_json = hy_resp.json()
            hy_outputs = hy_json.get("output_paths") if isinstance(hy_json.get("output_paths"), list) else []
            hy_output = _text(hy_json.get("output_path") or (hy_outputs[0] if hy_outputs else ""))
            if hy_output:
                hunyuan_path = Path(hy_output)
            if not str(hunyuan_path) or not hunyuan_path.exists() or not hunyuan_path.is_file():
                raise RuntimeError(f"Hunyuan output not found: {hunyuan_path}")
            hy_sec = time.perf_counter() - hy_started
            LOG.warning(
                "VlogMe Hunyuan done job=%s duration=%.3fs frames=%d fps=%d size=%sx%s elapsed=%.3fs input_bytes=%d",
                job_id,
                duration,
                num_frames,
                self.hunyuan_fps,
                src_w,
                src_h,
                hy_sec,
                image_bytes,
            )
            self.post_progress(job_id, 55, "hunyuan_done", lease_extend_minutes=20)

            voiced_path = run_dir / "with_audio.mp4"
            audio_mode = _text(_settings_audio(job).get("mode") or "auto").lower()
            if audio_mode in {"off", "none", "mute", "muted"}:
                _mux_silence(video_path=str(hunyuan_path), out_path=str(voiced_path), duration_sec=duration)
                LOG.warning("VlogMe audio skipped job=%s mode=off", job_id)
            else:
                audio_asset_url = _text(_settings_audio(job).get("audio_url") or _settings_audio(job).get("audioUrl"))
                if audio_mode == "asset":
                    if not audio_asset_url:
                        raise RuntimeError("video.audio.mode=asset requires audio_url")
                    audio_path = run_dir / "asset_audio"
                    self.post_progress(job_id, 60, "download_audio_asset")
                    _download(audio_asset_url, str(audio_path))
                else:
                    audio_path = run_dir / "mmaudio.wav"
                    audio_prompt = _text(_settings_audio(job).get("prompt"))
                    if not audio_prompt:
                        audio_prompt = f"Natural cinematic ambience and Foley matching the scene. {prompt}"
                    audio_negative = _text(_settings_audio(job).get("negative_prompt") or _settings_audio(job).get("negativePrompt"))
                    if not audio_negative:
                        audio_negative = "speech, talking, narration, vocals, singing, music, melody, distorted, clipping"
                    self.post_progress(job_id, 60, "mmaudio_start", lease_extend_minutes=15)
                    mm_payload = {
                        "video_path": str(hunyuan_path),
                        "prompt": audio_prompt,
                        "negative_prompt": audio_negative,
                        "duration": float(duration),
                        "output_path": str(audio_path),
                        "variant": _text(os.getenv("VLOGME_RENDER_MMAUDIO_VARIANT") or os.getenv("SMARTBLOG_MMAUDIO_VARIANT") or "large_44k_v2"),
                        "num_steps": _env_int("VLOGME_RENDER_MMAUDIO_STEPS", _env_int("SMARTBLOG_MMAUDIO_NUM_STEPS", 25)),
                        "cfg_strength": _env_float("VLOGME_RENDER_MMAUDIO_CFG", _env_float("SMARTBLOG_MMAUDIO_CFG_STRENGTH", 4.5)),
                    }
                    mm_resp = requests.post(f"{self.mmaudio_url}/generate", json=mm_payload, timeout=(20, max(600, int(duration * 120))))
                    if int(mm_resp.status_code) >= 400:
                        raise RuntimeError(f"MMAudio generate failed HTTP {mm_resp.status_code}: {mm_resp.text[:2000]}")
                    mm_json = mm_resp.json()
                    if not audio_path.exists():
                        mm_output = _text(mm_json.get("output_path"))
                        if mm_output:
                            audio_path = Path(mm_output)
                    if not audio_path.exists():
                        raise RuntimeError(f"MMAudio output not found: {audio_path}")
                _mux_audio(video_path=str(hunyuan_path), audio_path=str(audio_path), out_path=str(voiced_path), duration_sec=duration)

            self.post_progress(job_id, 70, "finalize_start", lease_extend_minutes=15)
            final_path = run_dir / "final.mp4"
            fin_started = time.perf_counter()
            self.run_finalizer(
                source_path=str(voiced_path),
                output_path=str(final_path),
                source_fps=int(self.hunyuan_fps),
                target_fps=int(self.output_fps),
                width=int(out_w),
                height=int(out_h),
            )
            final_summary = _probe_video_summary(str(final_path))
            fin_sec = time.perf_counter() - fin_started
            self.post_progress(job_id, 90, "upload", lease_extend_minutes=15)

            storage_path = self.upload_via_vlogme_api(job=job, job_id=job_id, file_path=str(final_path))
            self.post_complete(
                job_id,
                storage_path=str(storage_path),
                duration_sec=float(final_summary.get("duration") or duration),
                width=int(final_summary.get("width") or out_w),
                height=int(final_summary.get("height") or out_h),
                file_size_bytes=int(os.path.getsize(str(final_path))),
            )
            total_sec = time.perf_counter() - started
            LOG.warning(
                "VlogMe render complete job=%s total=%.3fs hy=%.3fs finalizer=%.3fs output=%sx%s fps=%.3f bytes=%d",
                job_id,
                total_sec,
                hy_sec,
                fin_sec,
                int(final_summary.get("width") or 0),
                int(final_summary.get("height") or 0),
                float(final_summary.get("fps") or 0.0),
                int(os.path.getsize(str(final_path))),
            )
        except VlogMeJobStopped as e:
            LOG.warning("VlogMe render stopped by API job=%s reason=%s", job_id, e)
        except Exception as e:
            LOG.exception("VlogMe render failed job=%s", job_id)
            self.post_fail(job_id, str(e), retryable=True)
        finally:
            keepalive_stop.set()

    def handle_video_edit_job(self, job: dict[str, Any], *, run_dir: Path, started: float) -> None:
        job_id = _job_id(job)
        claim = _claim_job(job)
        source_url = _text(claim.get("source_url") or job.get("source_url"))
        if not source_url:
            raise RuntimeError("video_edit claim has no source_url")

        self.post_progress(job_id, 4, "download_source")
        source_path = run_dir / "source_video"
        source_bytes = _download(source_url, str(source_path))
        source_summary = _probe_video_summary(str(source_path))
        kind = _text(claim.get("kind") or "edit").lower() or "edit"
        output_path = run_dir / "video_edit_output.mp4"

        if kind == "edit":
            self.post_progress(job_id, 12, "edit_start")
            output_path = self.render_library_video_edit(job, source_path=source_path, run_dir=run_dir)
        elif kind == "upscale":
            self.post_progress(job_id, 12, "upscale_start")
            output_path = self.render_library_video_upscale(job, source_path=source_path, run_dir=run_dir)
        elif kind == "interpolate":
            self.post_progress(job_id, 12, "interpolate_start")
            output_path = self.render_library_video_interpolate(job, source_path=source_path, run_dir=run_dir)
        elif kind == "matte":
            raise RuntimeError("video_edit kind=matte is not installed on this worker yet")
        else:
            raise RuntimeError(f"unsupported video_edit kind: {kind}")

        final_summary = _probe_video_summary(str(output_path))
        self.post_progress(job_id, 90, "upload")
        storage_path = self.upload_via_vlogme_api(job=job, job_id=job_id, file_path=str(output_path))
        self.post_complete(
            job_id,
            storage_path=str(storage_path),
            duration_sec=float(final_summary.get("duration") or source_summary.get("duration") or 0.0),
            width=int(final_summary.get("width") or 0),
            height=int(final_summary.get("height") or 0),
            file_size_bytes=int(os.path.getsize(str(output_path))),
        )
        LOG.warning(
            "VlogMe video_edit complete job=%s kind=%s total=%.3fs source=%sx%s %.3ffps output=%sx%s %.3ffps bytes_in=%d bytes_out=%d",
            job_id,
            kind,
            time.perf_counter() - started,
            int(source_summary.get("width") or 0),
            int(source_summary.get("height") or 0),
            float(source_summary.get("fps") or 0.0),
            int(final_summary.get("width") or 0),
            int(final_summary.get("height") or 0),
            float(final_summary.get("fps") or 0.0),
            source_bytes,
            int(os.path.getsize(str(output_path))),
        )

    def _video_edit_target(self, job: dict[str, Any], source_summary: dict[str, Any]) -> tuple[int, int, int, int]:
        claim = _claim_job(job)
        target = claim.get("target") if isinstance(claim.get("target"), dict) else {}
        width = _safe_int(target.get("width"), 720)
        height = _safe_int(target.get("height"), 1280)
        fps = _safe_int(target.get("fps"), 30)
        crf = _safe_int(target.get("crf"), 20)
        if width <= 0 or height <= 0:
            width = _safe_int(source_summary.get("width"), 720)
            height = _safe_int(source_summary.get("height"), 1280)
        if fps <= 0:
            fps = _safe_int(round(_safe_float(source_summary.get("fps"), 30.0)), 30)
        return _even(width), _even(height), max(1, fps), max(12, min(35, crf))

    def render_library_video_upscale(self, job: dict[str, Any], *, source_path: Path, run_dir: Path) -> Path:
        claim = _claim_job(job)
        ai_params = claim.get("ai_params") if isinstance(claim.get("ai_params"), dict) else {}
        summary = _probe_video_summary(str(source_path))
        factor = max(1.0, min(4.0, _safe_float(ai_params.get("factor"), 2.0)))
        target_w = _even(int(round(_safe_int(summary.get("width"), 720) * factor)))
        target_h = _even(int(round(_safe_int(summary.get("height"), 1280) * factor)))
        source_fps = max(1, _safe_int(round(_safe_float(summary.get("fps"), 30.0)), 30))
        out_path = run_dir / "video_upscale.mp4"
        self.run_finalizer(
            source_path=str(source_path),
            output_path=str(out_path),
            source_fps=source_fps,
            target_fps=source_fps,
            width=target_w,
            height=target_h,
            upscale_enabled=True,
            rife_enabled=False,
            quality=_text(ai_params.get("quality") or "DEBLUR_HIGH"),
        )
        return out_path

    def render_library_video_interpolate(self, job: dict[str, Any], *, source_path: Path, run_dir: Path) -> Path:
        claim = _claim_job(job)
        ai_params = claim.get("ai_params") if isinstance(claim.get("ai_params"), dict) else {}
        summary = _probe_video_summary(str(source_path))
        source_fps = max(1, _safe_int(round(_safe_float(summary.get("fps"), 30.0)), 30))
        target_fps = max(source_fps, min(120, _safe_int(ai_params.get("target_fps"), 60)))
        out_path = run_dir / "video_interpolate.mp4"
        self.run_finalizer(
            source_path=str(source_path),
            output_path=str(out_path),
            source_fps=source_fps,
            target_fps=target_fps,
            width=_even(_safe_int(summary.get("width"), 720)),
            height=_even(_safe_int(summary.get("height"), 1280)),
            upscale_enabled=False,
            rife_enabled=True,
            quality="FAST",
        )
        return out_path

    def render_library_video_edit(self, job: dict[str, Any], *, source_path: Path, run_dir: Path) -> Path:
        claim = _claim_job(job)
        ops = claim.get("ops") if isinstance(claim.get("ops"), list) else []
        append_ops = [op for op in ops if isinstance(op, dict) and _text(op.get("type")).lower() == "concat_append"]
        base_ops = [op for op in ops if not (isinstance(op, dict) and _text(op.get("type")).lower() == "concat_append")]
        summary = _probe_video_summary(str(source_path))
        width, height, fps, crf = self._video_edit_target(job, summary)
        base_out = run_dir / "video_edit_base.mp4"
        self.render_video_edit_base(
            job,
            source_path=source_path,
            output_path=base_out,
            run_dir=run_dir,
            ops=base_ops,
            width=width,
            height=height,
            fps=fps,
            crf=crf,
            label="video_edit_base",
        )
        if not append_ops:
            return base_out

        concat_inputs = [base_out]
        for index, op in enumerate(append_ops, start=1):
            clip_item_id = _text(op.get("clip_item_id") or op.get("item_id") or op.get("asset_id"))
            if not clip_item_id:
                raise RuntimeError("concat_append requires clip_item_id")
            asset = self.resolve_library_asset(job_id=_job_id(job), item_id=clip_item_id)
            clip_src = run_dir / f"concat_append_{index}_source"
            _download(_text(asset.get("url")), str(clip_src))
            clip_out = run_dir / f"concat_append_{index}.mp4"
            self.render_video_edit_base(
                job,
                source_path=clip_src,
                output_path=clip_out,
                run_dir=run_dir,
                ops=[],
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                label=f"concat_append_{index}",
            )
            concat_inputs.append(clip_out)

        final_out = run_dir / "video_edit_concat.mp4"
        concat_file = run_dir / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{_concat_demuxer_escape(path)}'\n" for path in concat_inputs),
            encoding="utf-8",
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(final_out),
            ],
            timeout=max(180.0, sum(max(1.0, _probe_video_summary(str(p)).get("duration") or 1.0) for p in concat_inputs) * 20.0),
            label="video_edit_concat",
        )
        return final_out

    def render_video_edit_base(
        self,
        job: dict[str, Any],
        *,
        source_path: Path,
        output_path: Path,
        run_dir: Path,
        ops: list[Any],
        width: int,
        height: int,
        fps: int,
        crf: int,
        label: str,
    ) -> None:
        summary = _probe_video_summary(str(source_path))
        source_duration = max(0.1, _safe_float(summary.get("duration"), 1.0))
        has_audio = _has_audio_stream(str(source_path))
        trim_start = 0.0
        trim_end = source_duration
        speed = 1.0
        reverse = False
        volume_level = 1.0
        fade_in = 0.0
        fade_out = 0.0
        color = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0}
        text_ops: list[dict[str, Any]] = []
        audio_replacements: list[dict[str, Any]] = []
        color_presets = {
            "cinematic": {"brightness": -0.05, "contrast": 1.15, "saturation": 0.85, "gamma": 1.0},
            "vivid": {"brightness": 0.02, "contrast": 1.20, "saturation": 1.40, "gamma": 1.0},
            "warm": {"brightness": 0.02, "contrast": 1.05, "saturation": 1.10, "gamma": 1.0},
            "cool": {"brightness": -0.02, "contrast": 1.05, "saturation": 1.05, "gamma": 1.0},
            "noir": {"brightness": 0.0, "contrast": 1.30, "saturation": 0.0, "gamma": 1.0},
            "vintage": {"brightness": 0.05, "contrast": 0.90, "saturation": 0.75, "gamma": 1.10},
        }

        for op in ops:
            if not isinstance(op, dict):
                continue
            op_type = _text(op.get("type")).lower()
            if op_type == "trim":
                trim_start = max(0.0, _safe_float(op.get("start_sec"), trim_start))
                end_value = op.get("end_sec")
                if end_value is not None:
                    trim_end = min(source_duration, max(trim_start + 0.05, _safe_float(end_value, trim_end)))
            elif op_type == "speed":
                speed *= max(0.25, min(4.0, _safe_float(op.get("factor"), 1.0)))
            elif op_type == "reverse":
                reverse = not reverse
            elif op_type == "volume":
                volume_level *= max(0.0, min(4.0, _safe_float(op.get("level"), 1.0)))
                fade_in = max(fade_in, max(0.0, _safe_float(op.get("fade_in_sec"), 0.0)))
                fade_out = max(fade_out, max(0.0, _safe_float(op.get("fade_out_sec"), 0.0)))
            elif op_type == "color":
                preset = _text(op.get("preset")).lower()
                if preset in color_presets:
                    color.update(color_presets[preset])
                for key in ("brightness", "contrast", "saturation", "gamma"):
                    if op.get(key) is not None:
                        color[key] = _safe_float(op.get(key), color[key])
            elif op_type == "text":
                if _text(op.get("text")):
                    text_ops.append(op)
            elif op_type == "audio_replace":
                audio_replacements.append(op)

        speed = max(0.25, min(4.0, speed))
        trimmed_duration = max(0.1, trim_end - trim_start)
        output_duration = max(0.1, trimmed_duration / speed)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source_path)]
        audio_base_index = 0
        if not has_audio:
            cmd += [
                "-f",
                "lavfi",
                "-t",
                f"{trimmed_duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            audio_base_index = 1
        replacement_labels: list[str] = []
        replacement_filter_parts: list[str] = []
        next_input_index = 1 if has_audio else 2
        for idx, op in enumerate(audio_replacements, start=1):
            item_id = _text(op.get("audio_item_id") or op.get("item_id") or op.get("asset_id"))
            if not item_id:
                continue
            asset = self.resolve_library_asset(job_id=_job_id(job), item_id=item_id)
            audio_path = run_dir / f"audio_replace_{idx}"
            _download(_text(asset.get("url")), str(audio_path))
            cmd += ["-i", str(audio_path)]
            level = max(0.0, min(4.0, _safe_float(op.get("level"), 1.0)))
            replacement_labels.append(f"a{idx}")
            filters = [
                "aresample=48000",
                f"atrim=start=0:end={output_duration:.6f}",
                "asetpts=PTS-STARTPTS",
                f"volume={level:.6f}",
            ]
            replacement_filter_parts.append(f"[{next_input_index}:a]{','.join(filters)}[a{idx}]")
            next_input_index += 1

        vf = [
            f"trim=start={trim_start:.6f}:end={trim_end:.6f}",
            f"setpts=(PTS-STARTPTS)/{speed:.6f}",
        ]
        if reverse:
            vf.append("reverse")
        vf.extend(
            [
                f"fps={int(fps)}",
                f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase",
                f"crop={int(width)}:{int(height)}",
                "setsar=1",
                (
                    "eq="
                    f"brightness={float(color['brightness']):.6f}:"
                    f"contrast={float(color['contrast']):.6f}:"
                    f"saturation={float(color['saturation']):.6f}:"
                    f"gamma={float(color['gamma']):.6f}"
                ),
            ]
        )
        for op in text_ops:
            text = _ffmpeg_filter_escape(op.get("text"))
            x = max(0.0, min(1.0, _safe_float(op.get("x"), 0.5)))
            y = max(0.0, min(1.0, _safe_float(op.get("y"), 0.8)))
            size = max(0.01, min(0.25, _safe_float(op.get("size"), 0.055)))
            color_raw = _text(op.get("color") or "white")
            font_color = color_raw if re.match(r"^[A-Za-z0-9#@._+-]+$", color_raw) else "white"
            draw = (
                "drawtext="
                f"text='{text}':"
                f"x=(w-text_w)*{x:.6f}:"
                f"y=(h-text_h)*{y:.6f}:"
                f"fontsize=h*{size:.6f}:"
                f"fontcolor={font_color}"
            )
            if bool(op.get("shadow")):
                draw += ":shadowcolor=black@0.7:shadowx=2:shadowy=2"
            start_t = op.get("start_sec")
            end_t = op.get("end_sec")
            if start_t is not None or end_t is not None:
                start = max(0.0, _safe_float(start_t, 0.0))
                end = max(start + 0.05, _safe_float(end_t, output_duration))
                draw += f":enable='between(t,{start:.6f},{end:.6f})'"
            vf.append(draw)
        vf.append("format=yuv420p")

        audio_trim_start = trim_start if has_audio else 0.0
        audio_trim_end = trim_end if has_audio else trimmed_duration
        af = [
            "aresample=48000",
            f"atrim=start={audio_trim_start:.6f}:end={audio_trim_end:.6f}",
            "asetpts=PTS-STARTPTS",
            *_atempo_chain(speed),
        ]
        if reverse:
            af.append("areverse")
        if abs(volume_level - 1.0) > 0.0001:
            af.append(f"volume={volume_level:.6f}")
        if fade_in > 0.0:
            af.append(f"afade=t=in:st=0:d={min(fade_in, output_duration):.6f}")
        if fade_out > 0.0:
            af.append(f"afade=t=out:st={max(0.0, output_duration - fade_out):.6f}:d={min(fade_out, output_duration):.6f}")

        filter_parts = [f"[0:v]{','.join(vf)}[v]", f"[{audio_base_index}:a]{','.join(af)}[a0]"]
        filter_parts.extend(replacement_filter_parts)
        if replacement_labels:
            original_level = max(0.0, min(1.0, _safe_float(audio_replacements[-1].get("original_level"), 1.0)))
            labels = ["[a0]", *[f"[{label}]" for label in replacement_labels]]
            weights = " ".join([f"{original_level:.6f}", *["1.000000" for _ in replacement_labels]])
            filter_parts.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0:weights='{weights}'[a]"
            )
            audio_label = "[a]"
        else:
            audio_label = "[a0]"

        base_cmd = [
            *cmd,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[v]",
            "-map",
            audio_label,
            "-shortest",
        ]
        _run_ffmpeg_encode(
            base_cmd,
            str(output_path),
            timeout=max(180.0, output_duration * 30.0),
            label=label,
            crf=int(crf),
        )

    def run_finalizer(
        self,
        *,
        source_path: str,
        output_path: str,
        source_fps: int,
        target_fps: int,
        width: int,
        height: int,
        upscale_enabled: bool | None = None,
        rife_enabled: bool | None = None,
        quality: str | None = None,
        scale: str | None = None,
    ) -> None:
        upscale = _env_flag("VLOGME_RENDER_FINALIZER_UPSCALE", "1") if upscale_enabled is None else bool(upscale_enabled)
        rife = _env_flag("VLOGME_RENDER_FINALIZER_RIFE", "1") if rife_enabled is None else bool(rife_enabled)
        params = {
            "upscale": "1" if upscale else "0",
            "rife": "1" if rife else "0",
            "rife_stage": _text(os.getenv("VLOGME_RENDER_FINALIZER_RIFE_STAGE") or "pre_vfx"),
            "source_fps": str(int(source_fps)),
            "target_fps": str(int(target_fps)),
            "target_width": str(int(width)),
            "target_height": str(int(height)),
            "quality": _text(quality or os.getenv("VLOGME_RENDER_FINALIZER_QUALITY") or "DEBLUR_HIGH"),
            "scale": _text(scale or os.getenv("VLOGME_RENDER_FINALIZER_SCALE") or "1"),
        }
        headers = {}
        if self.finalizer_secret:
            headers["Authorization"] = f"Bearer {self.finalizer_secret}"
        with open(source_path, "rb") as f:
            files = {"file": (os.path.basename(source_path), f, "video/mp4")}
            resp = requests.post(self.finalizer_url, params=params, headers=headers, files=files, timeout=(20, 3600))
        if int(resp.status_code) < 200 or int(resp.status_code) >= 300:
            raise RuntimeError(f"finalizer failed HTTP {resp.status_code}: {resp.text[:2000]}")
        content_type = str(resp.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            payload = resp.json()
            raise RuntimeError(f"finalizer returned JSON without upload_url support: {json.dumps(payload, ensure_ascii=False)[:2000]}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(resp.content)
        if os.path.getsize(output_path) <= 0:
            raise RuntimeError("finalizer returned empty output")

    def upload_via_vlogme_api(self, *, job: dict[str, Any], job_id: str, file_path: str) -> str:
        upload = _claim_upload(job)
        signed_url = _text(upload.get("signed_url") or upload.get("upload_url") or upload.get("signedUrl"))
        storage_path = _text(upload.get("path") or upload.get("storage_path") or upload.get("storagePath"))
        if signed_url:
            if not storage_path:
                raise RuntimeError("worker-api claim upload.signed_url is present but upload.path is empty")
            put_file_to_signed_url(
                signed_url=str(signed_url),
                path=str(file_path),
                content_type="video/mp4",
                connect_timeout=20.0,
                read_timeout=1800.0,
                env_prefix="VLOGME_SIGNED_UPLOAD",
                log_prefix="vlogme-render-signed-upload",
            )
            return str(storage_path)

        resp = requests.post(
            f"{self.base_url}/{urllib.parse.quote(str(job_id), safe='')}/upload-url",
            headers={"Authorization": f"Bearer {self.token}"},
            json={},
            timeout=(10, 60),
        )
        if int(resp.status_code) < 200 or int(resp.status_code) >= 300:
            raise RuntimeError(f"VlogMe upload-url failed HTTP {resp.status_code}: {resp.text[:2000]}")
        payload = resp.json()
        upload_url = _text(payload.get("upload_url") or payload.get("signed_url") or payload.get("signedUrl"))
        storage_path = _text(payload.get("storage_path") or payload.get("path"))
        if not upload_url:
            raise RuntimeError("VlogMe upload-url returned no upload_url")
        if not storage_path:
            raise RuntimeError("VlogMe upload-url returned no storage_path")
        put_file_to_signed_url(
            signed_url=str(upload_url),
            path=str(file_path),
            content_type="video/mp4",
            connect_timeout=20.0,
            read_timeout=1800.0,
            env_prefix="VLOGME_SIGNED_UPLOAD",
            log_prefix="vlogme-render-signed-upload",
        )
        return str(storage_path)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, _text(os.getenv("VLOGME_RENDER_LOG_LEVEL") or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    VlogMeRenderWorker().run_forever()


if __name__ == "__main__":
    main()
