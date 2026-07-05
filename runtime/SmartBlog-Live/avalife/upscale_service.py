from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web
from avalife.core.upload_retry import put_file_to_signed_url
from avalife.core.watermark import normalize_watermark_text, watermark_drawtext_filter, write_watermark_text_file
from avalife.worker.render_subtitles import RenderSubtitleChunk, write_render_subtitles_ass


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


def _bool_text(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "on", "y", "enable", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "n", "disable", "disabled"}:
        return False
    return bool(default)


def _even_dim(value: int) -> int:
    value_i = max(2, int(value or 2))
    return int(value_i - (value_i % 2))


def _run(cmd: list[str], *, timeout: float, label: str) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(timeout))
    if int(proc.returncode or 0) != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-4000:]
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"{label} failed ({proc.returncode}): {stderr or stdout}")


def _upload_file_to_signed_url(*, signed_url: str, path: str, content_type: str = "video/mp4") -> dict[str, Any]:
    url = str(signed_url or "").strip()
    if not url:
        raise RuntimeError("upload_url is empty")
    return put_file_to_signed_url(
        signed_url=str(url),
        path=str(path),
        content_type=str(content_type or "video/mp4"),
        connect_timeout=max(3.0, _env_float("SMARTBLOG_UPSCALE_UPLOAD_CONNECT_TIMEOUT_SEC", 20.0)),
        read_timeout=max(30.0, _env_float("SMARTBLOG_UPSCALE_UPLOAD_READ_TIMEOUT_SEC", 1800.0)),
        env_prefix="SMARTBLOG_UPSCALE_UPLOAD",
        log_prefix="upscale-signed-upload",
    )


def _extract_nonblack_video_poster(*, video_path: str, poster_path: str) -> dict[str, Any]:
    src = os.path.abspath(str(video_path or "").strip())
    dst = os.path.abspath(str(poster_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"poster source video not found: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video for poster: {src}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(total_frames / fps) if fps > 0.0 and total_frames > 0 else 0.0
        scan_seconds: list[float] = []
        for part in str(os.getenv("SMARTBLOG_UPSCALE_POSTER_SCAN_SECONDS", "0.4,0.8,1.2,1.8,2.5") or "").split(","):
            try:
                value = float(part.strip())
                if value >= 0.0:
                    scan_seconds.append(value)
            except Exception:
                pass
        if duration > 0.0:
            scan_seconds.extend(
                [
                    min(max(0.0, duration * 0.08), max(0.0, duration - 0.05)),
                    min(max(0.0, duration * 0.16), max(0.0, duration - 0.05)),
                ]
            )
        if not scan_seconds:
            scan_seconds = [0.0]
        min_mean = float(_env_float("SMARTBLOG_UPSCALE_POSTER_MIN_MEAN", 10.0))
        min_std = float(_env_float("SMARTBLOG_UPSCALE_POSTER_MIN_STD", 3.0))
        best_frame: Any | None = None
        best_score = -1.0
        best_mean = 0.0
        best_std = 0.0
        best_sec = 0.0
        for sec in scan_seconds:
            if fps > 0.0:
                frame_idx = int(max(0, round(float(sec) * float(fps))))
                if total_frames > 0:
                    frame_idx = min(max(0, total_frames - 1), frame_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(sec)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            std = float(np.std(gray))
            score = mean + std * 2.0
            if score > best_score:
                best_frame = frame
                best_score = score
                best_mean = mean
                best_std = std
                best_sec = float(sec)
            if mean >= min_mean and std >= min_std:
                best_frame = frame
                best_score = score
                best_mean = mean
                best_std = std
                best_sec = float(sec)
                break
        if best_frame is None:
            raise RuntimeError("could not read any frame for poster")
        height, width = int(best_frame.shape[0]), int(best_frame.shape[1])
        max_width = int(max(0, _env_int("SMARTBLOG_UPSCALE_POSTER_MAX_WIDTH", 720)))
        if max_width > 0 and width > max_width:
            scale = float(max_width) / float(width)
            best_frame = cv2.resize(
                best_frame,
                (int(max_width), max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = int(best_frame.shape[0]), int(best_frame.shape[1])
        quality = int(max(40, min(100, _env_int("SMARTBLOG_UPSCALE_POSTER_JPEG_QUALITY", 90))))
        ok = cv2.imwrite(dst, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError(f"cv2 failed to write poster: {dst}")
        return {
            "poster_generated": True,
            "poster_width": int(width),
            "poster_height": int(height),
            "poster_mean": float(best_mean),
            "poster_std": float(best_std),
            "poster_second": float(best_sec),
            "poster_bytes": int(os.path.getsize(dst)),
        }
    finally:
        cap.release()


def _download_url_to_file(*, url: str, path: str) -> dict[str, Any]:
    import requests

    src = str(url or "").strip()
    if not src:
        raise RuntimeError("source_url is empty")
    dst = str(path or "").strip()
    if not dst:
        raise RuntimeError("source path is empty")
    started = time.perf_counter()
    bytes_written = 0
    with requests.get(
        src,
        stream=True,
        timeout=(
            max(3.0, _env_float("SMARTBLOG_UPSCALE_SOURCE_CONNECT_TIMEOUT_SEC", 20.0)),
            max(30.0, _env_float("SMARTBLOG_UPSCALE_SOURCE_READ_TIMEOUT_SEC", 1800.0)),
        ),
    ) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += len(chunk)
    if int(bytes_written) <= 0:
        raise RuntimeError("source_url downloaded zero bytes")
    return {
        "downloaded": True,
        "source_bytes": int(bytes_written),
        "source_download_sec": float(time.perf_counter() - float(started)),
    }


def _video_has_audio_stream(path: str) -> bool:
    src = os.path.abspath(str(path or "").strip())
    if not src or not os.path.exists(src):
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(src),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode or 0) == 0 and bool(str(proc.stdout or "").strip())


def _ffmpeg_volume_db(value: float) -> str:
    try:
        gain = float(value)
    except Exception:
        gain = 0.0
    if not math.isfinite(gain):
        gain = 0.0
    gain = float(max(-60.0, min(24.0, gain)))
    return f"{gain:.3f}dB"


def _ffmpeg_filter_value(value: str) -> str:
    out = str(value or "")
    out = out.replace("\\", "\\\\")
    out = out.replace(":", "\\:")
    out = out.replace(",", "\\,")
    out = out.replace("'", "\\'")
    out = out.replace("[", "\\[")
    out = out.replace("]", "\\]")
    return out


def _final_touch_filters(*, enabled: bool, override: str = "") -> list[str]:
    if not bool(enabled):
        return []
    raw = str(override or os.getenv("SMARTBLOG_UPSCALE_FINAL_TOUCH_FILTERS", "") or "").strip()
    if raw:
        if raw.lower() in {"0", "false", "no", "off", "none"}:
            return []
        return [str(raw)]
    return [
        str(os.getenv("SMARTBLOG_UPSCALE_FINAL_TOUCH_EQ", "eq=contrast=1.025:saturation=1.035:gamma=0.995") or "").strip(),
        str(os.getenv("SMARTBLOG_UPSCALE_FINAL_TOUCH_UNSHARP", "unsharp=3:3:0.35:3:3:0.0") or "").strip(),
    ]


def _subtitle_chunks_from_json(value: str) -> list[RenderSubtitleChunk]:
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"invalid subtitle_chunks_json: {e}") from e
    if not isinstance(parsed, list):
        raise RuntimeError("subtitle_chunks_json must be a list")
    chunks: list[RenderSubtitleChunk] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            start_sec = float(item.get("start_sec") if item.get("start_sec") is not None else item.get("start") or 0.0)
            end_sec = float(item.get("end_sec") if item.get("end_sec") is not None else item.get("end") or 0.0)
        except Exception:
            continue
        if end_sec <= start_sec:
            continue
        chunks.append(
            RenderSubtitleChunk(
                index=int(item.get("index") or idx),
                text=str(text),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                alignment_offset_sec=float(item.get("alignment_offset_sec") or item.get("alignmentOffsetSec") or 0.0),
                alignment=dict(item.get("alignment")) if isinstance(item.get("alignment"), dict) else None,
                normalized_alignment=(
                    dict(item.get("normalized_alignment"))
                    if isinstance(item.get("normalized_alignment"), dict)
                    else dict(item.get("normalizedAlignment"))
                    if isinstance(item.get("normalizedAlignment"), dict)
                    else None
                ),
            )
        )
    return chunks


def _mix_background_music(
    *,
    video_path: str,
    music_path: str,
    out_path: str,
    duration_sec: float,
    sample_rate: int = 48000,
    gain_db: float = 0.0,
    loop: bool = True,
    duck_voice_db: float = 0.0,
    fade_in_seconds: float = 0.0,
    fade_out_seconds: float = 0.0,
    start_offset_seconds: float = 0.0,
) -> str:
    src = os.path.abspath(str(video_path or "").strip())
    music = os.path.abspath(str(music_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"background music video source missing: {src}")
    if not music or not os.path.exists(music):
        raise RuntimeError(f"background music source missing: {music}")
    duration = max(0.001, float(duration_sec or (_probe(src).get("format") or {}).get("duration") or 0.0))
    has_main_audio = bool(_video_has_audio_stream(src))
    fade_in = float(max(0.0, min(60.0, float(fade_in_seconds or 0.0))))
    fade_out = float(max(0.0, min(60.0, float(fade_out_seconds or 0.0))))
    if duration > 0.001 and fade_in + fade_out > duration * 0.9:
        scale = (duration * 0.9) / max(0.001, fade_in + fade_out)
        fade_in *= scale
        fade_out *= scale
    start_offset = float(max(0.0, float(start_offset_seconds or 0.0)))
    duck_db = float(max(-24.0, min(0.0, float(duck_voice_db or 0.0))))

    def build_cmd(*, use_ducking: bool) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src)]
        if bool(loop):
            cmd.extend(["-stream_loop", "-1"])
        if start_offset > 0.001:
            cmd.extend(["-ss", f"{float(start_offset):.6f}"])
        cmd.extend(["-i", str(music)])

        filter_parts: list[str] = []
        music_chain = (
            f"[1:a:0]aformat=channel_layouts=stereo,aresample={int(max(1, int(sample_rate)))},"
            f"apad,atrim=0:{float(duration):.6f},asetpts=PTS-STARTPTS,"
            f"volume={_ffmpeg_volume_db(float(gain_db or 0.0))}"
        )
        if fade_in > 0.001:
            music_chain += f",afade=t=in:st=0:d={float(min(fade_in, duration)):.6f}"
        if fade_out > 0.001:
            out_start = max(0.0, float(duration) - float(fade_out))
            music_chain += f",afade=t=out:st={float(out_start):.6f}:d={float(min(fade_out, duration)):.6f}"
        music_chain += "[music]"
        filter_parts.append(music_chain)

        if has_main_audio:
            filter_parts.append(
                f"[0:a:0]aformat=channel_layouts=stereo,aresample={int(max(1, int(sample_rate)))},"
                f"apad,atrim=0:{float(duration):.6f},asetpts=PTS-STARTPTS[main]"
            )
            if use_ducking and duck_db < -0.001:
                ratio = float(max(2.0, min(20.0, abs(float(duck_db)))))
                filter_parts.append("[main]asplit=2[main_mix][main_sc]")
                filter_parts.append(
                    f"[music][main_sc]sidechaincompress=threshold=0.035:ratio={float(ratio):.3f}:"
                    "attack=20:release=350[ducked]"
                )
                main_label = "[main_mix]"
                music_label = "[ducked]"
            else:
                main_label = "[main]"
                music_label = "[music]"
            filter_parts.append(
                f"{main_label}{music_label}amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                f"atrim=0:{float(duration):.6f}[a]"
            )
        else:
            filter_parts.append(f"[music]atrim=0:{float(duration):.6f}[a]")

        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-t",
                f"{float(duration):.6f}",
                "-shortest",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                str(os.getenv("REMOTE_EDGE_FILE_AUDIO_BITRATE", "160k") or "160k"),
                "-ar",
                str(int(max(1, int(sample_rate)))),
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(out),
            ]
        )
        return cmd

    cmd = build_cmd(use_ducking=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if int(proc.returncode or 0) != 0 and duck_db < -0.001:
        logging.warning(
            "SmartBlog upscale background music ducking failed, retrying plain mix: rc=%s stderr=%s",
            int(proc.returncode or 0),
            str(proc.stderr or "").strip()[-1000:],
        )
        cmd = build_cmd(use_ducking=False)
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if int(proc.returncode or 0) != 0:
        raise RuntimeError(
            "background music mix failed: "
            f"rc={proc.returncode} stderr={str(proc.stderr or '').strip()[-2000:]}"
        )
    logging.warning(
        "SmartBlog upscale background music mixed: duration=%.3fs gain=%.1fdB loop=%d duck=%.1fdB fade_in=%.2fs fade_out=%.2fs offset=%.2fs main_audio=%d",
        float(duration),
        float(gain_db or 0.0),
        1 if bool(loop) else 0,
        float(duck_db),
        float(fade_in),
        float(fade_out),
        float(start_offset),
        1 if bool(has_main_audio) else 0,
    )
    return str(out)


_ENCODER_CACHE: str | None = None
_UPSCALE_SEMAPHORE: asyncio.Semaphore | None = None
_UPSCALE_JOBS: dict[str, dict[str, Any]] = {}


def _require_upscale_auth(request: web.Request) -> None:
    secret = str(os.getenv("SMARTBLOG_UPSCALE_SHARED_SECRET", "") or "").strip()
    if not secret:
        return
    auth = str(request.headers.get("Authorization", "") or "").strip()
    header_secret = str(request.headers.get("X-SmartBlog-Upscale-Secret", "") or "").strip()
    if auth != f"Bearer {secret}" and header_secret != secret:
        raise web.HTTPUnauthorized(text="unauthorized")


def _tail_text_file(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "tail": ""}
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(max(0, size - max_bytes))
            data = f.read(max_bytes)
        return {
            "path": str(path),
            "exists": True,
            "size": int(size),
            "tail": data.decode("utf-8", errors="replace"),
        }
    except Exception as e:
        return {"path": str(path), "exists": True, "error": str(e), "tail": ""}


def _current_log_path(pointer: Path) -> Path | None:
    try:
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None
    if not text:
        return None
    return Path(text)


async def media_debug_logs(request: web.Request) -> web.Response:
    _require_upscale_auth(request)
    root = Path(os.getenv("SMARTBLOG_APP_DIR", "/workspace/SmartBlog-Live-media"))
    runtime = root / "runtime"
    max_bytes = max(4096, min(512 * 1024, _env_int("SMARTBLOG_DEBUG_LOG_TAIL_BYTES", 65536)))
    files: dict[str, Path] = {
        "entrypoint": runtime / "media_worker_registry_register.out",
        "sshd": runtime / "sshd.out",
        "upscale": runtime / "current_upscale_service_log.txt",
        "hunyuan": runtime / "current_hunyuan_service_log.txt",
        "mmaudio": runtime / "current_mmaudio_service_log.txt",
        "musetalk": runtime / "current_musetalk_service_log.txt",
        "vlogme_render_worker": runtime / "current_vlogme_render_worker_log.txt",
    }
    logs: dict[str, Any] = {}
    for name, path in files.items():
        target = _current_log_path(path) if path.name.startswith("current_") else path
        logs[name] = _tail_text_file(target or path, max_bytes=max_bytes)
    return web.json_response(
        {
            "ok": True,
            "root": str(root),
            "max_bytes": int(max_bytes),
            "logs": logs,
        }
    )


def _url_with_query(url: str, query_string: str) -> str:
    base = str(url or "").strip()
    query = str(query_string or "").strip()
    if not query:
        return base
    return f"{base}{'&' if '?' in base else '?'}{query}"


def _upscale_semaphore() -> asyncio.Semaphore:
    global _UPSCALE_SEMAPHORE
    if _UPSCALE_SEMAPHORE is None:
        _UPSCALE_SEMAPHORE = asyncio.Semaphore(max(1, _env_int("SMARTBLOG_UPSCALE_MAX_CONCURRENT", 1)))
    return _UPSCALE_SEMAPHORE


def _job_now() -> float:
    return float(time.time())


def _set_upscale_job(job_id: str, **updates: Any) -> dict[str, Any]:
    job = _UPSCALE_JOBS.get(str(job_id))
    if job is None:
        job = {"id": str(job_id), "created_at": _job_now()}
        _UPSCALE_JOBS[str(job_id)] = job
    job.update(updates)
    job["updated_at"] = _job_now()
    return job


def _public_upscale_job(job: dict[str, Any]) -> dict[str, Any]:
    hidden = {"task", "work_dir"}
    return {str(k): v for k, v in dict(job).items() if str(k) not in hidden}


def _list_public_upscale_jobs(*, status_filter: str = "", limit: int = 50) -> list[dict[str, Any]]:
    status_s = str(status_filter or "").strip().lower()
    jobs = sorted(
        (_public_upscale_job(job) for job in _UPSCALE_JOBS.values()),
        key=lambda item: float(item.get("created_at") or 0.0),
        reverse=True,
    )
    if status_s:
        jobs = [job for job in jobs if str(job.get("status") or "").strip().lower() == status_s]
    return jobs[: max(1, min(500, int(limit or 50)))]


def _abort_upscale_worker_on_stall(job_id: str, reason: str) -> None:
    _set_upscale_job(
        job_id,
        status="failed",
        stage="failed",
        progress=100,
        error=str(reason),
    )
    logging.error("SmartBlog async upscale watchdog abort: job=%s reason=%s", str(job_id), str(reason))
    if _env_flag("SMARTBLOG_UPSCALE_ASYNC_EXIT_ON_STALL", "1"):
        logging.error("SmartBlog async upscale watchdog exiting upscale service for supervisor restart")
        os._exit(70)
    raise TimeoutError(str(reason))


def _encoder() -> str:
    global _ENCODER_CACHE
    raw = str(os.getenv("SMARTBLOG_UPSCALE_VIDEO_ENCODER", "auto") or "auto").strip().lower()
    if raw not in {"auto", "h264_nvenc", "nvenc", "libx264", "x264"}:
        raw = "auto"
    if raw in {"libx264", "x264"}:
        return "libx264"
    if raw in {"h264_nvenc", "nvenc"}:
        return "h264_nvenc"
    if _ENCODER_CACHE:
        return str(_ENCODER_CACHE)
    with tempfile.TemporaryDirectory(prefix="smartblog-nvenc-test-") as td:
        out = os.path.join(td, "test.mp4")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x256:rate=1:duration=0.1",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            out,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    _ENCODER_CACHE = "h264_nvenc" if int(proc.returncode or 0) == 0 else "libx264"
    logging.warning("SmartBlog upscale encoder selected: %s", _ENCODER_CACHE)
    return str(_ENCODER_CACHE)


def _video_encode_args(*, output_fps: int) -> list[str]:
    encoder = _encoder()
    gop = max(1, int(round(float(output_fps) * _env_float("SMARTBLOG_UPSCALE_KEYFRAME_SEC", 2.0))))
    if encoder == "h264_nvenc":
        bitrate = str(os.getenv("SMARTBLOG_UPSCALE_VIDEO_BITRATE", "7000k") or "7000k")
        return [
            "-c:v",
            "h264_nvenc",
            "-profile:v",
            "high",
            "-preset",
            str(os.getenv("SMARTBLOG_UPSCALE_NVENC_PRESET", "p4") or "p4"),
            "-rc",
            "vbr",
            "-cq",
            str(os.getenv("SMARTBLOG_UPSCALE_NVENC_CQ", "19") or "19"),
            "-b:v",
            bitrate,
            "-maxrate",
            str(os.getenv("SMARTBLOG_UPSCALE_VIDEO_MAXRATE", bitrate) or bitrate),
            "-bufsize",
            str(os.getenv("SMARTBLOG_UPSCALE_VIDEO_BUFSIZE", "14000k") or "14000k"),
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-preset",
        str(os.getenv("SMARTBLOG_UPSCALE_X264_PRESET", "veryfast") or "veryfast"),
        "-crf",
        str(os.getenv("SMARTBLOG_UPSCALE_X264_CRF", "20") or "20"),
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
    ]


def _probe(path: str) -> dict[str, Any]:
    import json

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
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if int(proc.returncode or 0) != 0:
        raise RuntimeError((proc.stderr or b"").decode("utf-8", errors="replace")[-2000:])
    return dict(json.loads((proc.stdout or b"{}").decode("utf-8", errors="replace")))


def process_video(
    *,
    input_video: str,
    output_video: str,
    quality_name: str,
    scale: int,
    gpu: int,
    upscale_enabled: bool = True,
    rife_enabled: bool = False,
    rife_stage: str = "post_vfx",
    source_fps: int = 0,
    target_fps: int = 0,
    target_width: int = 0,
    target_height: int = 0,
    rife_batch_source_frames: int = 32,
    background_music_path: str = "",
    background_music_gain_db: float = 0.0,
    background_music_loop: bool = True,
    background_music_duck_voice_db: float = 0.0,
    background_music_fade_in_seconds: float = 0.0,
    background_music_fade_out_seconds: float = 0.0,
    background_music_start_offset_seconds: float = 0.0,
    subtitle_chunks_json: str = "",
    watermark_text: str = "",
    final_touch_enabled: bool = True,
    final_touch_filters: str = "",
    work_dir: str = "",
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    import av
    import numpy as np

    started = time.perf_counter()
    upscale_enabled = bool(upscale_enabled)
    requested_quality_s = str(quality_name or "DEBLUR_HIGH").strip().upper()
    quality_s = requested_quality_s
    if quality_s.startswith(("DEBLUR_", "DENOISE_", "HIGHBITRATE_")):
        mapped_quality = str(os.getenv("SMARTBLOG_UPSCALE_VSR_EFFECT_FALLBACK_QUALITY", "HIGH") or "HIGH").strip().upper()
        logging.warning(
            "SmartBlog upscale VSR quality mapped: requested=%s effective=%s",
            str(quality_s),
            str(mapped_quality),
        )
        quality_s = mapped_quality
    scale_i = max(1, min(4, int(scale or 1)))
    probe = _probe(str(input_video))
    video_stream_meta = next((s for s in probe.get("streams") or [] if s.get("codec_type") == "video"), {})
    duration = float((probe.get("format") or {}).get("duration") or video_stream_meta.get("duration") or 0.0)

    input_container = av.open(str(input_video))
    try:
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = "AUTO"
        source_w = max(1, int(input_stream.codec_context.width or video_stream_meta.get("width") or 1))
        source_h = max(1, int(input_stream.codec_context.height or video_stream_meta.get("height") or 1))
        requested_output_w = _even_dim(int(target_width)) if int(target_width or 0) > 0 else 0
        requested_output_h = _even_dim(int(target_height)) if int(target_height or 0) > 0 else 0
        vfx_enabled = bool(upscale_enabled)
        integer_vsr_scale = _bool_text(os.getenv("SMARTBLOG_UPSCALE_VSR_INTEGER_SCALE", "1"), default=True)
        if bool(vfx_enabled) and bool(integer_vsr_scale) and int(scale_i) <= 1 and requested_output_w > 0 and requested_output_h > 0:
            ratio = max(
                float(requested_output_w) / float(max(1, int(source_w))),
                float(requested_output_h) / float(max(1, int(source_h))),
            )
            if ratio > 1.0:
                scale_i = max(1, min(4, int(math.ceil(float(ratio)))))
            else:
                vfx_enabled = False
        elif bool(vfx_enabled) and requested_output_w > 0 and requested_output_h > 0:
            scale_i = 1
            vfx_enabled = False
        elif bool(vfx_enabled) and int(scale_i) <= 1:
            vfx_enabled = False
        process_w = int(source_w * int(scale_i)) if bool(vfx_enabled) else int(source_w)
        process_h = int(source_h * int(scale_i)) if bool(vfx_enabled) else int(source_h)
        output_w = int(requested_output_w) if requested_output_w > 0 else int(process_w)
        output_h = int(requested_output_h) if requested_output_h > 0 else int(process_h)
        overlay_work_dir = str(work_dir or os.path.dirname(str(output_video)) or ".")
        os.makedirs(overlay_work_dir, exist_ok=True)
        subtitle_blocks = 0
        ass_path = ""
        subtitle_chunks = _subtitle_chunks_from_json(str(subtitle_chunks_json or ""))
        if subtitle_chunks:
            ass_path = os.path.join(overlay_work_dir, "render_subtitles.ass")
            subtitle_blocks = int(
                write_render_subtitles_ass(
                    list(subtitle_chunks),
                    out_path=str(ass_path),
                    width=int(output_w),
                    height=int(output_h),
                )
            )
            if subtitle_blocks <= 0:
                ass_path = ""
        watermark_s = normalize_watermark_text(str(watermark_text or ""))
        watermark_filter = ""
        if watermark_s:
            watermark_text_file = write_watermark_text_file(
                path=os.path.join(overlay_work_dir, "watermark.txt"),
                text=str(watermark_s),
                width=int(output_w),
                height=int(output_h),
                env_prefixes=("SMARTBLOG_UPSCALE", "SMARTBLOG"),
            )
            watermark_filter = watermark_drawtext_filter(
                text_file=str(watermark_text_file),
                width=int(output_w),
                height=int(output_h),
                env_prefixes=("SMARTBLOG_UPSCALE", "SMARTBLOG"),
            )
        probed_fps_f = float(input_stream.average_rate or 0.0)
        requested_fps_f = float(source_fps or 0.0)
        fps_f = float(probed_fps_f or requested_fps_f or 30.0)
        if requested_fps_f > 0.0:
            if probed_fps_f > 0.0:
                fps_delta = abs(float(requested_fps_f) - float(probed_fps_f)) / max(1.0, float(probed_fps_f))
                if fps_delta > 0.08:
                    logging.warning(
                        "SmartBlog upscale source_fps override ignored: requested=%.3f probed=%.3f input=%s",
                        float(requested_fps_f),
                        float(probed_fps_f),
                        str(input_video),
                    )
                else:
                    fps_f = float(requested_fps_f)
            else:
                fps_f = float(requested_fps_f)
        fps_i = max(1, int(round(fps_f)))
        output_fps_i = max(1, int(target_fps or (fps_i * 2 if bool(rife_enabled) else fps_i)))
        total_source_frames = int(
            getattr(input_stream, "frames", 0)
            or video_stream_meta.get("nb_frames")
            or (float(duration) * float(fps_i) if float(duration or 0.0) > 0.0 else 0)
            or 0
        )
        if bool(rife_enabled) and int(output_fps_i) == int(fps_i):
            logging.warning(
                "SmartBlog upscale RIFE skipped: source_fps=%d target_fps=%d input=%s",
                int(fps_i),
                int(output_fps_i),
                str(input_video),
            )
            rife_enabled = False
        rife_stage_s = str(rife_stage or "post_vfx").strip().lower()
        rife_before_vfx = bool(vfx_enabled) and bool(rife_enabled) and rife_stage_s in {
            "pre",
            "pre_vfx",
            "before_vfx",
            "before_upscale",
        }

        torch = None
        sr = None
        stream_ptr = 0
        load_sec = 0.0
        if bool(vfx_enabled):
            import torch as torch_mod
            from nvvfx import VideoSuperRes

            torch = torch_mod
            if quality_s not in VideoSuperRes.QualityLevel.__members__:
                raise RuntimeError(f"unknown NVIDIA VFX quality: {quality_s}")
            torch.cuda.set_device(int(gpu))
            stream_ptr = torch.cuda.current_stream().cuda_stream
            sr = VideoSuperRes(device=int(gpu), quality=VideoSuperRes.QualityLevel[quality_s])
            sr.input_width = int(source_w)
            sr.input_height = int(source_h)
            sr.output_width = int(process_w)
            sr.output_height = int(process_h)
            load_started = time.perf_counter()
            sr.load()
            torch.cuda.synchronize()
            load_sec = float(time.perf_counter() - float(load_started))

        base = os.path.splitext(str(output_video))[0]
        video_tmp = f"{base}.video.mp4"
        mux_tmp = f"{base}.mux.mp4"
        for path in (video_tmp, mux_tmp, output_video):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(process_w)}x{int(process_h)}",
            "-r",
            str(int(output_fps_i)),
            "-i",
            "pipe:0",
            "-an",
        ]
        video_filters: list[str] = []
        if int(output_w) != int(process_w) or int(output_h) != int(process_h):
            video_filters.extend(
                [
                    f"scale={int(output_w)}:{int(output_h)}:force_original_aspect_ratio=increase:flags=lanczos",
                    f"crop={int(output_w)}:{int(output_h)}",
                ]
            )
        touch_filters = [
            item
            for item in _final_touch_filters(
                enabled=bool(final_touch_enabled) and bool(upscale_enabled),
                override=str(final_touch_filters or ""),
            )
            if str(item or "").strip()
        ]
        video_filters.extend(touch_filters)
        if ass_path:
            video_filters.append(f"ass={_ffmpeg_filter_value(str(ass_path))}")
        if watermark_filter:
            video_filters.append(str(watermark_filter))
        video_filters.append("setsar=1")
        if video_filters:
            cmd.extend(["-vf", ",".join(video_filters)])
        cmd.extend(_video_encode_args(output_fps=int(output_fps_i)))
        cmd.append(str(video_tmp))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdin is None:
            raise RuntimeError("failed to open encoder stdin")
        frames_in = 0
        frames_out = 0
        vfx_sec = 0.0
        rife_sec = 0.0
        write_sec = 0.0
        vfx_runtime_disabled = False
        vfx_failed_open = False
        rife_chunk: list[bytes] = []
        rife_carry: bytes | None = None
        rife_batch = max(2, int(rife_batch_source_frames or 32))
        interpolator: Any | None = None
        if bool(rife_enabled):
            from avalife.remote.torch_rife import get_shared_torch_rife_interpolator

            model_dir = str(os.getenv("SMARTBLOG_UPSCALE_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
            weights_path = str(
                os.getenv("SMARTBLOG_UPSCALE_RIFE_WEIGHTS", os.path.join(model_dir, "flownet.safetensors"))
                or os.path.join(model_dir, "flownet.safetensors")
            )
            interpolator = get_shared_torch_rife_interpolator(
                model_dir=model_dir,
                weights_path=weights_path,
                device=f"cuda:{int(gpu)}",
                dtype_name=str(os.getenv("SMARTBLOG_UPSCALE_RIFE_DTYPE", "float16") or "float16"),
                batch_pairs=max(1, _env_int("SMARTBLOG_UPSCALE_RIFE_BATCH_PAIRS", 4)),
            )

        def _write_raw_frame_bytes(items: list[bytes]) -> None:
            nonlocal frames_out, write_sec
            if not items:
                return
            t0 = time.perf_counter()
            for item in items:
                proc.stdin.write(bytes(item))
            write_sec += float(time.perf_counter() - t0)
            frames_out += int(len(items))

        def _resize_source_frame_bytes(arr: np.ndarray) -> bytes:
            if int(arr.shape[1]) == int(process_w) and int(arr.shape[0]) == int(process_h):
                return bytes(np.asarray(arr, dtype=np.uint8).tobytes())
            import cv2

            resized = cv2.resize(
                np.asarray(arr, dtype=np.uint8),
                (int(process_w), int(process_h)),
                interpolation=cv2.INTER_LANCZOS4,
            )
            return bytes(np.asarray(resized, dtype=np.uint8).tobytes())

        def _vfx_source_frame_bytes(frame_bytes: bytes) -> bytes:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8).copy().reshape((int(source_h), int(source_w), 3))
            nonlocal vfx_sec, vfx_runtime_disabled, vfx_failed_open
            if not bool(vfx_enabled) or bool(vfx_runtime_disabled):
                return _resize_source_frame_bytes(arr)
            if torch is None or sr is None:
                raise RuntimeError("NVIDIA VFX runtime is not initialized")
            v0 = time.perf_counter()
            try:
                tensor = torch.from_numpy(arr).to(f"cuda:{int(gpu)}")
                tensor = tensor.permute(2, 0, 1).float().div_(255.0).contiguous()
                vfx_output = sr.run(tensor, stream_ptr=stream_ptr)
                rgb_output = torch.from_dlpack(vfx_output.image)
                frame_np = (
                    rgb_output.clamp(0.0, 1.0)
                    .mul(255.0)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                vfx_sec += float(time.perf_counter() - v0)
                return bytes(frame_np.tobytes())
            except Exception as e:
                vfx_runtime_disabled = True
                vfx_failed_open = True
                logging.warning(
                    "SmartBlog upscale NVIDIA VSR failed open; using lanczos resize for this request: "
                    "source=%sx%s process=%sx%s quality=%s scale=%s error=%s",
                    int(source_w),
                    int(source_h),
                    int(process_w),
                    int(process_h),
                    str(quality_s),
                    int(scale_i),
                    str(e),
                )
                return _resize_source_frame_bytes(arr)

        def _flush_rife_chunk(*, final: bool) -> None:
            nonlocal rife_chunk, rife_carry, rife_sec
            if not bool(rife_enabled):
                return
            if not rife_chunk:
                if bool(final) and rife_carry is not None:
                    _write_raw_frame_bytes([bytes(rife_carry)])
                    rife_carry = None
                return
            combined = ([bytes(rife_carry)] if rife_carry is not None else []) + [bytes(x) for x in rife_chunk]
            if len(combined) < 2 or interpolator is None:
                if bool(final):
                    _write_raw_frame_bytes([bytes(x) for x in combined])
                    rife_carry = None
                else:
                    rife_carry = bytes(rife_chunk[-1])
                rife_chunk = []
                return
            target_frames = max(1, int(round(float(len(combined)) * float(output_fps_i) / float(max(1, fps_i)))))
            interp_w = int(source_w) if bool(rife_before_vfx) else int(process_w)
            interp_h = int(source_h) if bool(rife_before_vfx) else int(process_h)
            t0 = time.perf_counter()
            out = interpolator.interpolate_x2(
                combined,
                width=int(interp_w),
                height=int(interp_h),
                target_frames=int(target_frames),
            )
            rife_sec += float(time.perf_counter() - t0)
            if rife_carry is not None and out:
                out = out[1:]
            if not bool(final) and out:
                out = out[:-1]
            if bool(rife_before_vfx):
                _write_raw_frame_bytes([_vfx_source_frame_bytes(bytes(x)) for x in out])
            else:
                _write_raw_frame_bytes([bytes(x) for x in out])
            rife_carry = bytes(rife_chunk[-1])
            rife_chunk = []

        try:
            for frame in input_container.decode(input_stream):
                arr = frame.to_ndarray(format="rgb24")
                frames_in += 1
                if progress_cb is not None and (
                    int(frames_in) == 1 or int(frames_in) % max(1, int(fps_i)) == 0
                ):
                    try:
                        progress_cb(
                            phase="processing",
                            frames=int(frames_in),
                            total_frames=int(total_source_frames),
                            progress=(
                                min(0.98, float(frames_in) / float(max(1, int(total_source_frames))))
                                if int(total_source_frames) > 0
                                else 0.5
                            ),
                        )
                    except Exception:
                        logging.exception("SmartBlog upscale progress callback failed")
                if bool(rife_before_vfx):
                    frame_bytes = bytes(np.asarray(arr).tobytes())
                    rife_chunk.append(frame_bytes)
                    if len(rife_chunk) >= int(rife_batch):
                        _flush_rife_chunk(final=False)
                    continue
                frame_bytes = _vfx_source_frame_bytes(bytes(np.asarray(arr).tobytes()))
                if bool(rife_enabled):
                    rife_chunk.append(frame_bytes)
                    if len(rife_chunk) >= int(rife_batch):
                        _flush_rife_chunk(final=False)
                else:
                    _write_raw_frame_bytes([frame_bytes])
            if bool(rife_enabled):
                _flush_rife_chunk(final=True)
            if progress_cb is not None:
                try:
                    progress_cb(
                        phase="processing",
                        frames=int(frames_in),
                        total_frames=int(total_source_frames),
                        progress=0.99,
                    )
                except Exception:
                    logging.exception("SmartBlog upscale progress callback failed")
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=max(60.0, _env_float("SMARTBLOG_UPSCALE_ENCODE_WAIT_SEC", 600.0 + duration * 6.0)))
        except subprocess.TimeoutExpired as e:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError("upscale encode timed out") from e
        stdout = proc.stdout.read() if proc.stdout is not None else b""
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if int(proc.returncode or 0) != 0:
            raise RuntimeError(
                f"upscale encode failed ({proc.returncode}): "
                f"{(stderr or stdout or b'').decode('utf-8', errors='replace')[-4000:]}"
            )
        mux_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(video_tmp),
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-shortest",
            str(mux_tmp),
        ]
        _run(mux_cmd, timeout=max(60.0, 60.0 + duration * 4.0), label="upscale mux")
        os.replace(str(mux_tmp), str(output_video))
        music_sec = 0.0
        music_path = str(background_music_path or "").strip()
        if music_path:
            music_tmp = f"{base}.music.mp4"
            t_music = time.perf_counter()
            _mix_background_music(
                video_path=str(output_video),
                music_path=str(music_path),
                out_path=str(music_tmp),
                duration_sec=float(duration),
                sample_rate=48000,
                gain_db=float(background_music_gain_db or 0.0),
                loop=bool(background_music_loop),
                duck_voice_db=float(background_music_duck_voice_db or 0.0),
                fade_in_seconds=float(background_music_fade_in_seconds or 0.0),
                fade_out_seconds=float(background_music_fade_out_seconds or 0.0),
                start_offset_seconds=float(background_music_start_offset_seconds or 0.0),
            )
            os.replace(str(music_tmp), str(output_video))
            music_sec = float(time.perf_counter() - float(t_music))
        try:
            os.unlink(str(video_tmp))
        except Exception:
            pass
    finally:
        input_container.close()

    elapsed = float(time.perf_counter() - float(started))
    return {
        "quality": quality_s,
        "requested_quality": requested_quality_s,
        "scale": int(scale_i),
        "source_width": int(source_w),
        "source_height": int(source_h),
        "process_width": int(process_w),
        "process_height": int(process_h),
        "output_width": int(output_w),
        "output_height": int(output_h),
        "frames": int(frames_out),
        "source_frames": int(frames_in),
        "duration_sec": float(duration),
        "source_fps": int(fps_i),
        "output_fps": int(output_fps_i),
        "upscale_enabled": bool(upscale_enabled),
        "vfx_enabled": bool(vfx_enabled) and not bool(vfx_runtime_disabled),
        "vfx_failed_open": bool(vfx_failed_open),
        "load_sec": float(load_sec),
        "vfx_sec": float(vfx_sec),
        "rife_enabled": bool(rife_enabled),
        "rife_stage": str(rife_stage_s),
        "rife_sec": float(rife_sec),
        "background_music": bool(str(background_music_path or "").strip()),
        "background_music_sec": float(locals().get("music_sec", 0.0) or 0.0),
        "subtitles": int(subtitle_blocks),
        "watermark": bool(watermark_s),
        "final_touch": bool(touch_filters),
        "final_touch_filters": ",".join(str(item) for item in touch_filters),
        "write_sec": float(write_sec),
        "elapsed_sec": float(elapsed),
        "encoder": _encoder(),
        "bytes": int(os.path.getsize(str(output_video))) if os.path.exists(str(output_video)) else 0,
    }


async def health(_: web.Request) -> web.Response:
    model_dir = str(os.getenv("SMARTBLOG_UPSCALE_RIFE_MODEL_DIR", "/opt/RIFE-safetensors") or "/opt/RIFE-safetensors")
    return web.json_response({
        "ok": True,
        "encoder": _encoder(),
        "rife_weights": os.path.exists(os.path.join(model_dir, "flownet.safetensors")),
    })


async def musetalk_health(_: web.Request) -> web.Response:
    target = str(
        os.getenv("SMARTBLOG_UPSCALE_MUSETALK_HEALTH_URL")
        or os.getenv("SMARTBLOG_MUSETALK_HEALTH_URL")
        or ""
    ).strip()
    if not target:
        lipsync_url = str(
            os.getenv("SMARTBLOG_UPSCALE_MUSETALK_URL")
            or os.getenv("SMARTBLOG_MUSETALK_SERVICE_URL")
            or "http://127.0.0.1:8800/lipsync"
        ).strip()
        target = lipsync_url.rsplit("/", 1)[0].rstrip("/") + "/health"
    timeout = ClientTimeout(total=max(3.0, _env_float("SMARTBLOG_UPSCALE_MUSETALK_HEALTH_TIMEOUT_SEC", 15.0)))
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.get(target) as resp:
                body = await resp.read()
                headers = {"Content-Type": str(resp.headers.get("Content-Type") or "application/json")}
                return web.Response(status=int(resp.status), body=body, headers=headers)
    except Exception as e:
        return web.json_response({"ok": False, "target": target, "error": str(e)}, status=502)


async def musetalk_lipsync_proxy(request: web.Request) -> web.StreamResponse:
    _require_upscale_auth(request)
    target_base = str(
        os.getenv("SMARTBLOG_UPSCALE_MUSETALK_URL")
        or os.getenv("SMARTBLOG_MUSETALK_LOCAL_URL")
        or "http://127.0.0.1:8800/lipsync"
    ).strip()
    if not target_base:
        raise web.HTTPServiceUnavailable(text="musetalk target is not configured")
    target = _url_with_query(target_base, str(request.query_string or ""))
    body = await request.read()
    headers: dict[str, str] = {}
    for name in ("Content-Type", "Authorization", "X-SmartBlog-MuseTalk-Secret", "X-SmartBlog-Upscale-Secret"):
        value = str(request.headers.get(name, "") or "").strip()
        if value:
            headers[name] = value
    timeout = ClientTimeout(total=max(30.0, _env_float("SMARTBLOG_UPSCALE_MUSETALK_TIMEOUT_SEC", 3600.0)))
    started = time.perf_counter()
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(target, data=body, headers=headers) as resp:
                response_headers = {
                    "Content-Type": str(resp.headers.get("Content-Type") or "application/octet-stream"),
                    "X-SmartBlog-MuseTalk-Proxy": "1",
                }
                proxied = web.StreamResponse(status=int(resp.status), headers=response_headers)
                await proxied.prepare(request)
                bytes_out = 0
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    if not chunk:
                        continue
                    bytes_out += len(chunk)
                    await proxied.write(chunk)
                await proxied.write_eof()
                logging.warning(
                    "SmartBlog MuseTalk proxy complete: status=%s bytes=%s elapsed=%.3fs target=%s",
                    int(resp.status),
                    int(bytes_out),
                    float(time.perf_counter() - float(started)),
                    target_base,
                )
                return proxied
    except Exception as e:
        logging.exception("SmartBlog MuseTalk proxy failed: target=%s", target_base)
        raise web.HTTPBadGateway(text=f"musetalk proxy failed: {e}") from e


def _upscale_request_defaults(request: web.Request) -> dict[str, Any]:
    quality = str(request.query.get("quality") or os.getenv("SMARTBLOG_UPSCALE_DEFAULT_QUALITY", "DEBLUR_HIGH"))
    scale = max(1, min(4, int(float(str(request.query.get("scale") or os.getenv("SMARTBLOG_UPSCALE_DEFAULT_SCALE", "1"))))))
    return {
        "quality": quality,
        "scale": int(scale),
        "gpu": max(0, _env_int("SMARTBLOG_UPSCALE_GPU", 0)),
        "upscale_enabled": _bool_text(
            request.query.get("upscale")
            or request.query.get("upscale_enabled")
            or request.query.get("vsr")
            or os.getenv("SMARTBLOG_UPSCALE_ENABLED", "1"),
            default=True,
        ),
        "rife_enabled": str(
            request.query.get("rife")
            or request.query.get("interpolate")
            or os.getenv("SMARTBLOG_UPSCALE_RIFE", "0")
            or "0"
        ).strip().lower() in {"1", "true", "yes", "on", "rife", "torch-rife", "x2"},
        "rife_stage": str(
            request.query.get("rife_stage")
            or request.query.get("interpolation_stage")
            or os.getenv("SMARTBLOG_UPSCALE_RIFE_STAGE", "post_vfx")
            or "post_vfx"
        ).strip().lower(),
        "source_fps": max(0, int(float(str(request.query.get("source_fps") or os.getenv("SMARTBLOG_UPSCALE_SOURCE_FPS", "0") or "0")))),
        "target_fps": max(0, int(float(str(request.query.get("target_fps") or os.getenv("SMARTBLOG_UPSCALE_TARGET_FPS", "0") or "0")))),
        "target_width": max(0, int(float(str(request.query.get("target_width") or os.getenv("SMARTBLOG_UPSCALE_TARGET_WIDTH", "0") or "0")))),
        "target_height": max(0, int(float(str(request.query.get("target_height") or os.getenv("SMARTBLOG_UPSCALE_TARGET_HEIGHT", "0") or "0")))),
        "background_music_url": str(
            request.query.get("background_music_url")
            or request.query.get("music_url")
            or request.query.get("backgroundMusicUrl")
            or ""
        ).strip(),
        "background_music_gain_db": _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_GAIN_DB", 0.0),
        "background_music_duck_voice_db": _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_DUCK_VOICE_DB", 0.0),
        "background_music_fade_in_seconds": _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_FADE_IN_SECONDS", 0.0),
        "background_music_fade_out_seconds": _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_FADE_OUT_SECONDS", 0.0),
        "background_music_start_offset_seconds": _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_START_OFFSET_SECONDS", 0.0),
        "background_music_loop": str(request.query.get("background_music_loop") or "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        },
        "subtitle_chunks_json": str(
            request.query.get("subtitle_chunks_json") or request.query.get("subtitles_json") or ""
        ).strip(),
        "watermark_text": normalize_watermark_text(
            request.query.get("watermark_text") or request.query.get("watermarkText") or ""
        ),
        "final_touch_enabled": _bool_text(
            request.query.get("final_touch")
            or request.query.get("finalTouch")
            or request.query.get("sharpen")
            or os.getenv("SMARTBLOG_UPSCALE_FINAL_TOUCH", "1"),
            default=True,
        ),
        "final_touch_filters": str(
            request.query.get("final_touch_filters")
            or request.query.get("finalTouchFilters")
            or request.query.get("post_filter")
            or request.query.get("postFilter")
            or ""
        ).strip(),
        "rife_batch_source_frames": max(
            2,
            int(float(str(request.query.get("rife_batch_source_frames") or os.getenv("SMARTBLOG_UPSCALE_RIFE_BATCH_SOURCE_FRAMES", "32") or "32"))),
        ),
        "upload_url": str(request.query.get("upload_url") or "").strip(),
        "upload_content_type": str(request.query.get("upload_content_type") or "video/mp4").strip() or "video/mp4",
        "poster_upload_url": str(request.query.get("poster_upload_url") or "").strip(),
        "poster_upload_content_type": str(request.query.get("poster_upload_content_type") or "image/jpeg").strip()
        or "image/jpeg",
        "poster_storage_path": str(request.query.get("poster_storage_path") or "").strip(),
        "source_url": str(request.query.get("source_url") or "").strip(),
    }


async def _parse_upscale_async_job_request(request: web.Request) -> dict[str, Any]:
    opts = _upscale_request_defaults(request)
    if request.can_read_body:
        try:
            reader = await request.multipart()
        except Exception:
            reader = None
        if reader is not None:
            while True:
                part = await reader.next()
                if part is None:
                    break
                name = str(part.name or "")
                text_names = {
                    "source_url",
                    "input_url",
                    "source_signed_url",
                    "upload_url",
                    "output_upload_url",
                    "upload_content_type",
                    "content_type",
                    "poster_upload_url",
                    "posterUploadUrl",
                    "poster_upload_content_type",
                    "posterUploadContentType",
                    "poster_storage_path",
                    "posterStoragePath",
                    "source_fps",
                    "input_fps",
                    "sourceFps",
                    "upscale",
                    "upscale_enabled",
                    "upscaleEnabled",
                    "vsr",
                    "rife_stage",
                    "interpolation_stage",
                    "rifeStage",
                    "interpolationStage",
                    "background_music_url",
                    "music_url",
                    "backgroundMusicUrl",
                    "background_music_gain_db",
                    "music_gain_db",
                    "backgroundMusicGainDb",
                    "background_music_duck_voice_db",
                    "music_duck_voice_db",
                    "backgroundMusicDuckVoiceDb",
                    "background_music_fade_in_seconds",
                    "music_fade_in_seconds",
                    "backgroundMusicFadeInSeconds",
                    "background_music_fade_out_seconds",
                    "music_fade_out_seconds",
                    "backgroundMusicFadeOutSeconds",
                    "background_music_start_offset_seconds",
                    "music_start_offset_seconds",
                    "backgroundMusicStartOffsetSeconds",
                    "background_music_loop",
                    "music_loop",
                    "backgroundMusicLoop",
                    "subtitle_chunks_json",
                    "subtitles_json",
                    "subtitleChunksJson",
                    "subtitlesJson",
                    "watermark_text",
                    "watermarkText",
                    "final_touch",
                    "finalTouch",
                    "sharpen",
                    "final_touch_filters",
                    "finalTouchFilters",
                    "post_filter",
                    "postFilter",
                    "target_fps",
                    "target_width",
                    "target_height",
                    "rife_batch_source_frames",
                }
                if name in text_names:
                    value = (await part.text()).strip()
                    if name in {"source_url", "input_url", "source_signed_url"}:
                        opts["source_url"] = value
                    elif name in {"upload_url", "output_upload_url"}:
                        opts["upload_url"] = value
                    elif name in {"upload_content_type", "content_type"}:
                        opts["upload_content_type"] = value or "video/mp4"
                    elif name in {"poster_upload_url", "posterUploadUrl"}:
                        opts["poster_upload_url"] = value
                    elif name in {"poster_upload_content_type", "posterUploadContentType"}:
                        opts["poster_upload_content_type"] = value or "image/jpeg"
                    elif name in {"poster_storage_path", "posterStoragePath"}:
                        opts["poster_storage_path"] = value
                    elif name in {"source_fps", "input_fps", "sourceFps", "target_fps", "target_width", "target_height", "rife_batch_source_frames"}:
                        try:
                            opts[{
                                "source_fps": "source_fps",
                                "input_fps": "source_fps",
                                "sourceFps": "source_fps",
                                "target_fps": "target_fps",
                                "target_width": "target_width",
                                "target_height": "target_height",
                                "rife_batch_source_frames": "rife_batch_source_frames",
                            }[name]] = max(0, int(float(value or "0")))
                        except Exception:
                            pass
                    elif name in {"upscale", "upscale_enabled", "upscaleEnabled", "vsr"}:
                        opts["upscale_enabled"] = _bool_text(value, default=bool(opts["upscale_enabled"]))
                    elif name in {"rife_stage", "interpolation_stage", "rifeStage", "interpolationStage"}:
                        opts["rife_stage"] = str(value or opts["rife_stage"]).lower()
                    elif name in {"background_music_url", "music_url", "backgroundMusicUrl"}:
                        opts["background_music_url"] = value
                    elif name in {"background_music_loop", "music_loop", "backgroundMusicLoop"}:
                        opts["background_music_loop"] = str(value).lower() not in {"0", "false", "no", "off"}
                    elif name in {"subtitle_chunks_json", "subtitles_json", "subtitleChunksJson", "subtitlesJson"}:
                        opts["subtitle_chunks_json"] = value
                    elif name in {"watermark_text", "watermarkText"}:
                        opts["watermark_text"] = normalize_watermark_text(value)
                    elif name in {"final_touch", "finalTouch", "sharpen"}:
                        opts["final_touch_enabled"] = _bool_text(value, default=bool(opts["final_touch_enabled"]))
                    elif name in {"final_touch_filters", "finalTouchFilters", "post_filter", "postFilter"}:
                        opts["final_touch_filters"] = value
                    else:
                        key_map = {
                            "background_music_gain_db": "background_music_gain_db",
                            "music_gain_db": "background_music_gain_db",
                            "backgroundMusicGainDb": "background_music_gain_db",
                            "background_music_duck_voice_db": "background_music_duck_voice_db",
                            "music_duck_voice_db": "background_music_duck_voice_db",
                            "backgroundMusicDuckVoiceDb": "background_music_duck_voice_db",
                            "background_music_fade_in_seconds": "background_music_fade_in_seconds",
                            "music_fade_in_seconds": "background_music_fade_in_seconds",
                            "backgroundMusicFadeInSeconds": "background_music_fade_in_seconds",
                            "background_music_fade_out_seconds": "background_music_fade_out_seconds",
                            "music_fade_out_seconds": "background_music_fade_out_seconds",
                            "backgroundMusicFadeOutSeconds": "background_music_fade_out_seconds",
                            "background_music_start_offset_seconds": "background_music_start_offset_seconds",
                            "music_start_offset_seconds": "background_music_start_offset_seconds",
                            "backgroundMusicStartOffsetSeconds": "background_music_start_offset_seconds",
                        }
                        if name in key_map:
                            try:
                                opts[key_map[name]] = float(value or 0.0)
                            except Exception:
                                pass
                    continue
                while await part.read_chunk(size=1024 * 1024):
                    pass
    return opts


async def _run_upscale_async_job(job_id: str, opts: dict[str, Any]) -> None:
    work_root = Path(os.getenv("SMARTBLOG_UPSCALE_WORK_DIR", "/tmp/smartblog-upscale")) / "async-jobs"
    job_dir = work_root / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_upscale_job(job_id, status="running", stage="download", progress=5, work_dir=str(job_dir))
    started = time.perf_counter()
    input_path = str(job_dir / "input.mp4")
    output_path = str(job_dir / "output.mp4")
    background_music_path = ""
    try:
        source_url = str(opts.get("source_url") or "").strip()
        upload_url = str(opts.get("upload_url") or "").strip()
        if not source_url:
            raise RuntimeError("source_url is required for async upscale job")
        if not upload_url:
            raise RuntimeError("upload_url is required for async upscale job")
        download_result = await asyncio.to_thread(_download_url_to_file, url=source_url, path=input_path)
        music_download: dict[str, Any] = {}
        if str(opts.get("background_music_url") or "").strip():
            background_music_path = str(job_dir / "background_music.audio")
            music_download = await asyncio.to_thread(
                _download_url_to_file,
                url=str(opts.get("background_music_url") or ""),
                path=str(background_music_path),
            )
        _set_upscale_job(job_id, status="running", stage="queued", progress=20)
        sem = _upscale_semaphore()
        queue_started = time.perf_counter()
        async with sem:
            queue_sec = float(time.perf_counter() - float(queue_started))
            _set_upscale_job(job_id, status="running", stage="processing", progress=35, queue_sec=queue_sec)
            last_processing_progress = 35

            def _processing_progress(**payload: Any) -> None:
                nonlocal last_processing_progress
                try:
                    frac = float(payload.get("progress") or 0.0)
                except Exception:
                    frac = 0.0
                frac = float(max(0.0, min(1.0, frac)))
                progress_i = int(max(35, min(86, round(35.0 + 51.0 * frac))))
                if progress_i <= int(last_processing_progress):
                    return
                last_processing_progress = int(progress_i)
                _set_upscale_job(
                    job_id,
                    status="running",
                    stage=str(payload.get("phase") or "processing"),
                    progress=int(progress_i),
                    frames=int(payload.get("frames") or 0),
                    total_frames=int(payload.get("total_frames") or 0),
                    queue_sec=queue_sec,
                )

            process_task = asyncio.create_task(
                asyncio.to_thread(
                    process_video,
                    input_video=input_path,
                    output_video=output_path,
                    quality_name=str(opts.get("quality") or "DEBLUR_HIGH"),
                    scale=int(opts.get("scale") or 1),
                    gpu=int(opts.get("gpu") or 0),
                    upscale_enabled=bool(opts.get("upscale_enabled", True)),
                    rife_enabled=bool(opts.get("rife_enabled", False)),
                    rife_stage=str(opts.get("rife_stage") or "post_vfx"),
                    source_fps=int(opts.get("source_fps") or 0),
                    target_fps=int(opts.get("target_fps") or 0),
                    target_width=int(opts.get("target_width") or 0),
                    target_height=int(opts.get("target_height") or 0),
                    rife_batch_source_frames=int(opts.get("rife_batch_source_frames") or 32),
                    background_music_path=str(background_music_path),
                    background_music_gain_db=float(opts.get("background_music_gain_db") or 0.0),
                    background_music_loop=bool(opts.get("background_music_loop", True)),
                    background_music_duck_voice_db=float(opts.get("background_music_duck_voice_db") or 0.0),
                    background_music_fade_in_seconds=float(opts.get("background_music_fade_in_seconds") or 0.0),
                    background_music_fade_out_seconds=float(opts.get("background_music_fade_out_seconds") or 0.0),
                    background_music_start_offset_seconds=float(opts.get("background_music_start_offset_seconds") or 0.0),
                    subtitle_chunks_json=str(opts.get("subtitle_chunks_json") or ""),
                    watermark_text=str(opts.get("watermark_text") or ""),
                    final_touch_enabled=bool(opts.get("final_touch_enabled", True)),
                    final_touch_filters=str(opts.get("final_touch_filters") or ""),
                    work_dir=str(job_dir),
                    progress_cb=_processing_progress,
                ),
                name=f"upscale-process-{job_id}",
            )
            process_started_mono = float(time.monotonic())
            process_hard_timeout = max(
                0.0,
                _env_float("SMARTBLOG_UPSCALE_ASYNC_PROCESS_TIMEOUT_SEC", 1800.0),
            )
            process_stall_timeout = max(
                0.0,
                _env_float("SMARTBLOG_UPSCALE_ASYNC_PROGRESS_TIMEOUT_SEC", 300.0),
            )
            watchdog_poll = max(1.0, _env_float("SMARTBLOG_UPSCALE_ASYNC_WATCHDOG_POLL_SEC", 5.0))
            while True:
                done, _ = await asyncio.wait({process_task}, timeout=float(watchdog_poll))
                if process_task in done:
                    result = process_task.result()
                    break
                elapsed_mono = float(time.monotonic()) - float(process_started_mono)
                if process_hard_timeout > 0.0 and elapsed_mono > process_hard_timeout:
                    _abort_upscale_worker_on_stall(
                        job_id,
                        f"process hard timeout after {elapsed_mono:.1f}s",
                    )
                job_snapshot = _UPSCALE_JOBS.get(str(job_id)) or {}
                updated_at = float(job_snapshot.get("updated_at") or _job_now())
                stale_sec = float(_job_now()) - float(updated_at)
                if process_stall_timeout > 0.0 and stale_sec > process_stall_timeout:
                    _abort_upscale_worker_on_stall(
                        job_id,
                        "process progress stalled "
                        f"for {stale_sec:.1f}s stage={job_snapshot.get('stage')} "
                        f"progress={job_snapshot.get('progress')} "
                        f"frames={job_snapshot.get('frames')}/{job_snapshot.get('total_frames')}",
                    )
            result["queue_sec"] = queue_sec
        result.update(download_result)
        if music_download:
            result["background_music_source_bytes"] = int(music_download.get("source_bytes") or 0)
            result["background_music_download_sec"] = float(music_download.get("source_download_sec") or 0.0)
        poster_upload_url = str(opts.get("poster_upload_url") or "").strip()
        if poster_upload_url and _env_flag("SMARTBLOG_UPSCALE_POSTER_ENABLED", "1"):
            _set_upscale_job(job_id, status="running", stage="poster", progress=87)
            poster_path = str(job_dir / "poster.jpg")
            try:
                poster_result = await asyncio.to_thread(
                    _extract_nonblack_video_poster,
                    video_path=str(output_path),
                    poster_path=str(poster_path),
                )
                poster_upload = await asyncio.to_thread(
                    _upload_file_to_signed_url,
                    signed_url=str(poster_upload_url),
                    path=str(poster_path),
                    content_type=str(opts.get("poster_upload_content_type") or "image/jpeg"),
                )
                result.update(poster_result)
                result["poster_uploaded"] = bool(poster_upload.get("uploaded"))
                result["poster_storage_path"] = str(opts.get("poster_storage_path") or "")
                result["poster_upload_bytes"] = int(poster_upload.get("bytes") or poster_upload.get("upload_bytes") or 0)
            except Exception as poster_exc:
                logging.exception("SmartBlog async upscale poster failed: job=%s", str(job_id))
                result["poster_uploaded"] = False
                result["poster_error"] = str(poster_exc)[:1000]
        _set_upscale_job(job_id, status="running", stage="upload", progress=88)
        upload_result = await asyncio.to_thread(
            _upload_file_to_signed_url,
            signed_url=str(upload_url),
            path=str(output_path),
            content_type=str(opts.get("upload_content_type") or "video/mp4"),
        )
        result.update(upload_result)
        result["async_job_id"] = str(job_id)
        result["elapsed_sec"] = float(time.perf_counter() - float(started))
        logging.warning(
            "SmartBlog async upscale complete: job=%s source=%sx%s output=%sx%s fps=%s->%s frames=%s->%s upscale=%d uploaded=%d elapsed=%.3fs",
            str(job_id),
            result.get("source_width"),
            result.get("source_height"),
            result.get("output_width"),
            result.get("output_height"),
            result.get("source_fps"),
            result.get("output_fps"),
            result.get("source_frames"),
            result.get("frames"),
            1 if bool(result.get("upscale_enabled")) else 0,
            1 if bool(result.get("uploaded")) else 0,
            float(result.get("elapsed_sec") or 0.0),
        )
        _set_upscale_job(job_id, status="completed", stage="complete", progress=100, result=result)
        if _env_flag("SMARTBLOG_UPSCALE_ASYNC_CLEANUP", "1"):
            shutil.rmtree(str(job_dir), ignore_errors=True)
    except Exception as e:
        logging.exception("SmartBlog async upscale failed: job=%s", str(job_id))
        _set_upscale_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            error=str(e),
            elapsed_sec=float(time.perf_counter() - float(started)),
        )


async def upscale_start_job(request: web.Request) -> web.Response:
    _require_upscale_auth(request)
    opts = await _parse_upscale_async_job_request(request)
    if not str(opts.get("source_url") or "").strip():
        raise web.HTTPBadRequest(text="source_url is required for async upscale job")
    if not str(opts.get("upload_url") or "").strip():
        raise web.HTTPBadRequest(text="upload_url is required for async upscale job")
    job_id = str(uuid.uuid4())
    _set_upscale_job(
        job_id,
        status="queued",
        stage="queued",
        progress=0,
        source_url_present=bool(str(opts.get("source_url") or "").strip()),
        upload_url_present=bool(str(opts.get("upload_url") or "").strip()),
    )
    task = asyncio.create_task(_run_upscale_async_job(job_id, opts), name=f"upscale-job-{job_id}")
    _UPSCALE_JOBS[job_id]["task"] = task
    return web.json_response(
        {
            "ok": True,
            "accepted": True,
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/upscale/jobs/{job_id}",
        },
        status=202,
    )


async def upscale_jobs(request: web.Request) -> web.Response:
    _require_upscale_auth(request)
    status_filter = str(request.query.get("status") or "").strip()
    limit = max(1, min(500, int(float(str(request.query.get("limit") or "50")))))
    return web.json_response(
        {
            "ok": True,
            "count": len(_UPSCALE_JOBS),
            "max_concurrent": max(1, _env_int("SMARTBLOG_UPSCALE_MAX_CONCURRENT", 1)),
            "jobs": _list_public_upscale_jobs(status_filter=status_filter, limit=limit),
        }
    )


async def upscale_job_status(request: web.Request) -> web.Response:
    _require_upscale_auth(request)
    job_id = str(request.match_info.get("job_id") or "").strip()
    job = _UPSCALE_JOBS.get(job_id)
    if not job:
        raise web.HTTPNotFound(text="upscale job not found")
    return web.json_response(_public_upscale_job(job))


async def upscale_cancel_job(request: web.Request) -> web.Response:
    _require_upscale_auth(request)
    job_id = str(request.match_info.get("job_id") or "").strip()
    job = _UPSCALE_JOBS.get(job_id)
    if not job:
        raise web.HTTPNotFound(text="upscale job not found")
    task = job.get("task")
    if isinstance(task, asyncio.Task) and not task.done():
        task.cancel()
    _set_upscale_job(
        job_id,
        status="cancelled",
        stage="cancelled",
        progress=100,
        error="cancelled by admin request",
    )
    if str(job.get("stage") or "").strip().lower() == "processing" and _env_flag(
        "SMARTBLOG_UPSCALE_ASYNC_EXIT_ON_CANCEL", "1"
    ):
        logging.error("SmartBlog async upscale admin cancel exits service for clean GPU/process state: job=%s", job_id)
        os._exit(71)
    return web.json_response(_public_upscale_job(_UPSCALE_JOBS[job_id]))


async def upscale(request: web.Request) -> web.StreamResponse:
    _require_upscale_auth(request)

    quality = str(request.query.get("quality") or os.getenv("SMARTBLOG_UPSCALE_DEFAULT_QUALITY", "DEBLUR_HIGH"))
    scale = max(1, min(4, int(float(str(request.query.get("scale") or os.getenv("SMARTBLOG_UPSCALE_DEFAULT_SCALE", "1"))))))
    gpu = max(0, _env_int("SMARTBLOG_UPSCALE_GPU", 0))
    upscale_enabled = _bool_text(
        request.query.get("upscale")
        or request.query.get("upscale_enabled")
        or request.query.get("vsr")
        or os.getenv("SMARTBLOG_UPSCALE_ENABLED", "1"),
        default=True,
    )
    rife_enabled = str(
        request.query.get("rife")
        or request.query.get("interpolate")
        or os.getenv("SMARTBLOG_UPSCALE_RIFE", "0")
        or "0"
    ).strip().lower() in {"1", "true", "yes", "on", "rife", "torch-rife", "x2"}
    rife_stage = str(
        request.query.get("rife_stage")
        or request.query.get("interpolation_stage")
        or os.getenv("SMARTBLOG_UPSCALE_RIFE_STAGE", "post_vfx")
        or "post_vfx"
    ).strip().lower()
    source_fps = max(0, int(float(str(request.query.get("source_fps") or os.getenv("SMARTBLOG_UPSCALE_SOURCE_FPS", "0") or "0"))))
    target_fps = max(0, int(float(str(request.query.get("target_fps") or os.getenv("SMARTBLOG_UPSCALE_TARGET_FPS", "0") or "0"))))
    target_width = max(0, int(float(str(request.query.get("target_width") or os.getenv("SMARTBLOG_UPSCALE_TARGET_WIDTH", "0") or "0"))))
    target_height = max(0, int(float(str(request.query.get("target_height") or os.getenv("SMARTBLOG_UPSCALE_TARGET_HEIGHT", "0") or "0"))))
    background_music_url = str(
        request.query.get("background_music_url")
        or request.query.get("music_url")
        or request.query.get("backgroundMusicUrl")
        or ""
    ).strip()
    background_music_gain_db = _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_GAIN_DB", 0.0)
    background_music_duck_voice_db = _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_DUCK_VOICE_DB", 0.0)
    background_music_fade_in_seconds = _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_FADE_IN_SECONDS", 0.0)
    background_music_fade_out_seconds = _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_FADE_OUT_SECONDS", 0.0)
    background_music_start_offset_seconds = _env_float("SMARTBLOG_UPSCALE_BACKGROUND_MUSIC_START_OFFSET_SECONDS", 0.0)
    background_music_loop = str(request.query.get("background_music_loop") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    subtitle_chunks_json = str(request.query.get("subtitle_chunks_json") or request.query.get("subtitles_json") or "").strip()
    watermark_text = normalize_watermark_text(request.query.get("watermark_text") or request.query.get("watermarkText") or "")
    final_touch_enabled = _bool_text(
        request.query.get("final_touch")
        or request.query.get("finalTouch")
        or request.query.get("sharpen")
        or os.getenv("SMARTBLOG_UPSCALE_FINAL_TOUCH", "1"),
        default=True,
    )
    final_touch_filters = str(
        request.query.get("final_touch_filters")
        or request.query.get("finalTouchFilters")
        or request.query.get("post_filter")
        or request.query.get("postFilter")
        or ""
    ).strip()
    rife_batch_source_frames = max(
        2,
        int(float(str(request.query.get("rife_batch_source_frames") or os.getenv("SMARTBLOG_UPSCALE_RIFE_BATCH_SOURCE_FRAMES", "32") or "32"))),
    )

    work_root = Path(os.getenv("SMARTBLOG_UPSCALE_WORK_DIR", "/tmp/smartblog-upscale"))
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job-", dir=str(work_root)) as td:
        input_path = os.path.join(td, "input.mp4")
        output_path = os.path.join(td, "output.mp4")
        background_music_path = ""
        upload_url = str(request.query.get("upload_url") or "").strip()
        upload_content_type = str(request.query.get("upload_content_type") or "video/mp4").strip() or "video/mp4"
        poster_upload_url = str(request.query.get("poster_upload_url") or "").strip()
        poster_upload_content_type = str(request.query.get("poster_upload_content_type") or "image/jpeg").strip() or "image/jpeg"
        poster_storage_path = str(request.query.get("poster_storage_path") or "").strip()
        source_url = str(request.query.get("source_url") or "").strip()
        got_file = False
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            name = str(part.name or "")
            if name == "file":
                got_file = True
                with open(input_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk(size=1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                continue
            if name in {"upload_url", "output_upload_url"}:
                upload_url = (await part.text()).strip()
                continue
            if name in {"upload_content_type", "content_type"}:
                upload_content_type = (await part.text()).strip() or "video/mp4"
                continue
            if name in {"poster_upload_url", "posterUploadUrl"}:
                poster_upload_url = (await part.text()).strip()
                continue
            if name in {"poster_upload_content_type", "posterUploadContentType"}:
                poster_upload_content_type = (await part.text()).strip() or "image/jpeg"
                continue
            if name in {"poster_storage_path", "posterStoragePath"}:
                poster_storage_path = (await part.text()).strip()
                continue
            if name in {"source_url", "input_url", "source_signed_url"}:
                source_url = (await part.text()).strip()
                continue
            if name in {"source_fps", "input_fps", "sourceFps"}:
                try:
                    source_fps = max(0, int(float((await part.text()).strip())))
                except Exception:
                    source_fps = 0
                continue
            if name in {"upscale", "upscale_enabled", "upscaleEnabled", "vsr"}:
                upscale_enabled = _bool_text(await part.text(), default=bool(upscale_enabled))
                continue
            if name in {"rife_stage", "interpolation_stage", "rifeStage", "interpolationStage"}:
                rife_stage = str((await part.text()).strip() or rife_stage).lower()
                continue
            if name in {"background_music_url", "music_url", "backgroundMusicUrl"}:
                background_music_url = (await part.text()).strip()
                continue
            if name in {"background_music_gain_db", "music_gain_db", "backgroundMusicGainDb"}:
                try:
                    background_music_gain_db = float((await part.text()).strip())
                except Exception:
                    background_music_gain_db = 0.0
                continue
            if name in {"background_music_duck_voice_db", "music_duck_voice_db", "backgroundMusicDuckVoiceDb"}:
                try:
                    background_music_duck_voice_db = float((await part.text()).strip())
                except Exception:
                    background_music_duck_voice_db = 0.0
                continue
            if name in {"background_music_fade_in_seconds", "music_fade_in_seconds", "backgroundMusicFadeInSeconds"}:
                try:
                    background_music_fade_in_seconds = float((await part.text()).strip())
                except Exception:
                    background_music_fade_in_seconds = 0.0
                continue
            if name in {"background_music_fade_out_seconds", "music_fade_out_seconds", "backgroundMusicFadeOutSeconds"}:
                try:
                    background_music_fade_out_seconds = float((await part.text()).strip())
                except Exception:
                    background_music_fade_out_seconds = 0.0
                continue
            if name in {"background_music_start_offset_seconds", "music_start_offset_seconds", "backgroundMusicStartOffsetSeconds"}:
                try:
                    background_music_start_offset_seconds = float((await part.text()).strip())
                except Exception:
                    background_music_start_offset_seconds = 0.0
                continue
            if name in {"background_music_loop", "music_loop", "backgroundMusicLoop"}:
                background_music_loop = str((await part.text()).strip()).lower() not in {"0", "false", "no", "off"}
                continue
            if name in {"subtitle_chunks_json", "subtitles_json", "subtitleChunksJson", "subtitlesJson"}:
                subtitle_chunks_json = (await part.text()).strip()
                continue
            if name in {"watermark_text", "watermarkText"}:
                watermark_text = normalize_watermark_text(await part.text())
                continue
            if name in {"final_touch", "finalTouch", "sharpen"}:
                final_touch_enabled = _bool_text(await part.text(), default=bool(final_touch_enabled))
                continue
            if name in {"final_touch_filters", "finalTouchFilters", "post_filter", "postFilter"}:
                final_touch_filters = (await part.text()).strip()
                continue
            while await part.read_chunk(size=1024 * 1024):
                pass
        if not got_file:
            if not source_url:
                raise web.HTTPBadRequest(text="multipart field 'file' or 'source_url' is required")
            download_result = await asyncio.to_thread(
                _download_url_to_file,
                url=str(source_url),
                path=str(input_path),
            )
            got_file = True
        else:
            download_result = {}
        if background_music_url:
            background_music_path = os.path.join(td, "background_music.audio")
            music_download = await asyncio.to_thread(
                _download_url_to_file,
                url=str(background_music_url),
                path=str(background_music_path),
            )
        else:
            music_download = {}
        queue_started = time.perf_counter()
        sem = _upscale_semaphore()
        if sem.locked():
            logging.warning("SmartBlog upscale queued: max_concurrent=%s", _env_int("SMARTBLOG_UPSCALE_MAX_CONCURRENT", 1))
        async with sem:
            queue_sec = float(time.perf_counter() - float(queue_started))
            started = time.perf_counter()
            result = await asyncio.to_thread(
                process_video,
                input_video=input_path,
                output_video=output_path,
                quality_name=quality,
                scale=scale,
                gpu=gpu,
                upscale_enabled=bool(upscale_enabled),
                rife_enabled=bool(rife_enabled),
                rife_stage=str(rife_stage),
                source_fps=int(source_fps),
                target_fps=int(target_fps),
                target_width=int(target_width),
                target_height=int(target_height),
                rife_batch_source_frames=int(rife_batch_source_frames),
                background_music_path=str(background_music_path),
                background_music_gain_db=float(background_music_gain_db),
                background_music_loop=bool(background_music_loop),
                background_music_duck_voice_db=float(background_music_duck_voice_db),
                background_music_fade_in_seconds=float(background_music_fade_in_seconds),
                background_music_fade_out_seconds=float(background_music_fade_out_seconds),
                background_music_start_offset_seconds=float(background_music_start_offset_seconds),
                subtitle_chunks_json=str(subtitle_chunks_json or ""),
                watermark_text=str(watermark_text or ""),
                final_touch_enabled=bool(final_touch_enabled),
                final_touch_filters=str(final_touch_filters or ""),
                work_dir=str(td),
            )
            result["queue_sec"] = queue_sec
            if download_result:
                result.update(download_result)
            if music_download:
                result["background_music_source_bytes"] = int(music_download.get("source_bytes") or 0)
                result["background_music_download_sec"] = float(music_download.get("source_download_sec") or 0.0)
            if poster_upload_url and _env_flag("SMARTBLOG_UPSCALE_POSTER_ENABLED", "1"):
                poster_path = os.path.join(td, "poster.jpg")
                try:
                    poster_result = await asyncio.to_thread(
                        _extract_nonblack_video_poster,
                        video_path=str(output_path),
                        poster_path=str(poster_path),
                    )
                    poster_upload = await asyncio.to_thread(
                        _upload_file_to_signed_url,
                        signed_url=str(poster_upload_url),
                        path=str(poster_path),
                        content_type=str(poster_upload_content_type),
                    )
                    result.update(poster_result)
                    result["poster_uploaded"] = bool(poster_upload.get("uploaded"))
                    result["poster_storage_path"] = str(poster_storage_path or "")
                    result["poster_upload_bytes"] = int(poster_upload.get("bytes") or poster_upload.get("upload_bytes") or 0)
                except Exception as poster_exc:
                    logging.exception("SmartBlog upscale poster failed")
                    result["poster_uploaded"] = False
                    result["poster_error"] = str(poster_exc)[:1000]
            if upload_url:
                upload_result = await asyncio.to_thread(
                    _upload_file_to_signed_url,
                    signed_url=str(upload_url),
                    path=str(output_path),
                    content_type=str(upload_content_type),
                )
                result.update(upload_result)
                logging.warning(
                    "SmartBlog upscale direct upload complete: source=%sx%s process=%sx%s output=%sx%s fps=%s->%s frames=%s->%s upscale=%d vfx=%d quality=%s scale=%s final_touch=%d vfx_sec=%.3fs rife=%d/%s rife_sec=%.3fs subtitles=%s watermark=%d bytes=%s queue=%.3fs music=%d music_sec=%.3fs upload=%.3fs elapsed=%.3fs",
                    result.get("source_width"),
                    result.get("source_height"),
                    result.get("process_width"),
                    result.get("process_height"),
                    result.get("output_width"),
                    result.get("output_height"),
                    result.get("source_fps"),
                    result.get("output_fps"),
                    result.get("source_frames"),
                    result.get("frames"),
                    1 if bool(result.get("upscale_enabled")) else 0,
                    1 if bool(result.get("vfx_enabled")) else 0,
                    result.get("quality"),
                    result.get("scale"),
                    1 if bool(result.get("final_touch")) else 0,
                    float(result.get("vfx_sec") or 0.0),
                    1 if bool(result.get("rife_enabled")) else 0,
                    str(result.get("rife_stage") or ""),
                    float(result.get("rife_sec") or 0.0),
                    result.get("subtitles"),
                    1 if bool(result.get("watermark")) else 0,
                    result.get("bytes"),
                    float(result.get("queue_sec") or 0.0),
                    1 if bool(result.get("background_music")) else 0,
                    float(result.get("background_music_sec") or 0.0),
                    float(result.get("upload_sec") or 0.0),
                    float(time.perf_counter() - float(started)),
                )
                return web.json_response(result)
            logging.warning(
                "SmartBlog upscale complete: source=%sx%s output=%sx%s frames=%s->%s fps=%s->%s upscale=%d quality=%s scale=%s final_touch=%d rife=%s/%s subtitles=%s watermark=%d queue=%.3fs vfx=%.3fs rife_sec=%.3fs encoder=%s elapsed=%.3fs",
                result.get("source_width"),
                result.get("source_height"),
                result.get("output_width"),
                result.get("output_height"),
                result.get("source_frames"),
                result.get("frames"),
                result.get("source_fps"),
                result.get("output_fps"),
                1 if bool(result.get("upscale_enabled")) else 0,
                result.get("quality"),
                result.get("scale"),
                1 if bool(result.get("final_touch")) else 0,
                result.get("rife_enabled"),
                str(result.get("rife_stage") or ""),
                result.get("subtitles"),
                1 if bool(result.get("watermark")) else 0,
                float(result.get("queue_sec") or 0.0),
                float(result.get("vfx_sec") or 0.0),
                float(result.get("rife_sec") or 0.0),
                result.get("encoder"),
                float(time.perf_counter() - float(started)),
            )
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "video/mp4",
                    "X-SmartBlog-Upscale-Frames": str(result.get("frames") or 0),
                    "X-SmartBlog-Upscale-Source-Frames": str(result.get("source_frames") or 0),
                    "X-SmartBlog-Upscale-Width": str(result.get("output_width") or 0),
                    "X-SmartBlog-Upscale-Height": str(result.get("output_height") or 0),
                    "X-SmartBlog-Upscale-Fps": str(result.get("output_fps") or 0),
                    "X-SmartBlog-Upscale-Enabled": "1" if bool(result.get("upscale_enabled")) else "0",
                    "X-SmartBlog-Upscale-Rife": "1" if bool(result.get("rife_enabled")) else "0",
                    "X-SmartBlog-Upscale-Rife-Stage": str(result.get("rife_stage") or ""),
                    "X-SmartBlog-Upscale-Elapsed-Sec": f"{float(result.get('elapsed_sec') or 0.0):.3f}",
                    "X-SmartBlog-Upscale-Rife-Sec": f"{float(result.get('rife_sec') or 0.0):.3f}",
                    "X-SmartBlog-Upscale-Queue-Sec": f"{float(result.get('queue_sec') or 0.0):.3f}",
                    "X-SmartBlog-Upscale-Encoder": str(result.get("encoder") or ""),
                },
            )
            await response.prepare(request)
            with open(output_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    await response.write(data)
            await response.write_eof()
            return response


def main() -> None:
    logging.basicConfig(level=str(os.getenv("SMARTBLOG_UPSCALE_LOG_LEVEL", "INFO") or "INFO").upper())
    app = web.Application(client_max_size=max(1, _env_int("SMARTBLOG_UPSCALE_CLIENT_MAX_MB", 4096)) * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/debug/media-logs", media_debug_logs)
    app.router.add_get("/musetalk/health", musetalk_health)
    app.router.add_post("/musetalk/lipsync", musetalk_lipsync_proxy)
    app.router.add_get("/upscale/jobs", upscale_jobs)
    app.router.add_post("/upscale/jobs", upscale_start_job)
    app.router.add_get("/upscale/jobs/{job_id}", upscale_job_status)
    app.router.add_post("/upscale/jobs/{job_id}/cancel", upscale_cancel_job)
    app.router.add_delete("/upscale/jobs/{job_id}", upscale_cancel_job)
    app.router.add_post("/upscale", upscale)
    host = str(os.getenv("SMARTBLOG_UPSCALE_HOST", "0.0.0.0") or "0.0.0.0")
    port = _env_int("SMARTBLOG_UPSCALE_PORT", 19300)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
