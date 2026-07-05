from __future__ import annotations

import functools
import logging
import os
import subprocess

from avalife.core.watermark import watermark_drawtext_filter


def resolve_channel_outputs() -> tuple[bool, bool, str, str]:
    rtmp_url = (os.getenv("LIVE_CHANNEL_RTMP_URL") or os.getenv("LIVEKIT_RTMP_URL") or "").strip()
    output_mode = os.getenv("LIVE_CHANNEL_OUTPUT", "").strip().lower()
    if not output_mode:
        output_mode = "both" if rtmp_url else "hls"
    if output_mode not in ("hls", "rtmp", "both"):
        output_mode = "both" if rtmp_url else "hls"
    want_hls = output_mode in ("hls", "both")
    want_rtmp = output_mode in ("rtmp", "both")
    return want_hls, want_rtmp, rtmp_url, output_mode


def recreate_fifos(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        os.mkfifo(path, 0o600)


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return float(default)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if bool(default) else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@functools.lru_cache(maxsize=1)
def nvenc_runtime_available() -> bool:
    if not _truthy_env("LIVE_CHANNEL_RTMP_NVENC_VALIDATE", True):
        return True
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=256x256:rate=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, _safe_float_env("LIVE_CHANNEL_RTMP_NVENC_VALIDATE_TIMEOUT_SEC", 5.0)),
            check=False,
        )
    except Exception as e:
        logging.warning("h264_nvenc runtime validation failed: %s", e)
        return False
    if int(proc.returncode) == 0:
        return True
    tail = " ".join(str(proc.stderr or "").strip().splitlines()[-3:])
    logging.warning("h264_nvenc runtime unavailable: %s", tail or f"rc={proc.returncode}")
    return False


def resolve_rtmp_standard_canvas(width: int, height: int) -> tuple[int, int, str]:
    w = max(1, int(width))
    h = max(1, int(height))
    if not _truthy_env("LIVE_CHANNEL_RTMP_STANDARD_PAD", True):
        return int(w), int(h), ""
    if int(w) == 720 and int(h) == 1080:
        return 720, 1280, "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1"
    if int(w) == 1080 and int(h) == 720:
        return 1280, 720, "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,setsar=1"
    return int(w), int(h), ""


def resolve_rtmp_publish_canvas(width: int, height: int) -> tuple[int, int, list[str]]:
    canvas_w, canvas_h, canvas_filter = resolve_rtmp_standard_canvas(width=int(width), height=int(height))
    filters: list[str] = []
    if canvas_filter:
        filters.append(str(canvas_filter))
    if _truthy_env("LIVE_CHANNEL_RTMP_SOCIAL_UPSCALE", False):
        target_w = int(canvas_w)
        target_h = int(canvas_h)
        if int(canvas_h) > int(canvas_w):
            target_w, target_h = 1080, 1920
        elif int(canvas_w) > int(canvas_h):
            target_w, target_h = 1920, 1080
        if target_w != int(canvas_w) or target_h != int(canvas_h):
            scale_flags = str(os.getenv("LIVE_CHANNEL_RTMP_SCALE_FLAGS", "bicubic") or "bicubic").strip() or "bicubic"
            filters.append(f"scale={int(target_w)}:{int(target_h)}:flags={scale_flags}")
            canvas_w, canvas_h = int(target_w), int(target_h)
    return int(canvas_w), int(canvas_h), filters


def _rtmp_unsharp_filters() -> list[str]:
    raw = str(os.getenv("LIVE_CHANNEL_RTMP_UNSHARP", "") or "").strip()
    if not raw or raw.lower() in {"0", "false", "no", "off", "none"}:
        return []
    if raw.lower() in {"1", "true", "yes", "on"}:
        raw = "3:3:0.30:3:3:0.0"
    if raw.startswith("unsharp="):
        return [raw]
    return [f"unsharp={raw}"]


def _rtmp_aspect_filters(*, width: int, height: int) -> list[str]:
    w = max(1, int(width))
    h = max(1, int(height))
    return ["setsar=1", f"setdar=dar={w}/{h}"]


def _rtmp_aspect_args(*, width: int, height: int) -> list[str]:
    w = max(1, int(width))
    h = max(1, int(height))
    return ["-aspect", f"{w}:{h}"]


def rtmp_output_fps(input_fps: int) -> int:
    return max(30, max(1, int(input_fps)), _safe_int_env("LIVE_CHANNEL_RTMP_OUTPUT_FPS", 30))


def _rtmp_video_filter_args(
    *,
    width: int,
    height: int,
    input_fps: int,
    output_fps: int,
    watermark_text_file: str | None = None,
) -> list[str]:
    filters: list[str] = []
    if int(output_fps) != int(input_fps):
        filters.append(f"fps={int(output_fps)}")
    canvas_w, canvas_h, canvas_filters = resolve_rtmp_publish_canvas(width=int(width), height=int(height))
    filters.extend(canvas_filters)
    filters.extend(_rtmp_unsharp_filters())
    watermark_filter = watermark_drawtext_filter(
        text_file=str(watermark_text_file or ""),
        width=int(canvas_w),
        height=int(canvas_h),
        env_prefixes=("LIVE_CHANNEL", "REMOTE_EDGE", "SMARTBLOG"),
    )
    if watermark_filter:
        filters.append(str(watermark_filter))
    filters.extend(_rtmp_aspect_filters(width=int(canvas_w), height=int(canvas_h)))
    return ["-vf", ",".join(filters)] if filters else []


def _rtmp_clip_split_filter_complex(
    *,
    width: int,
    height: int,
    input_fps: int,
    output_fps: int,
    watermark_text_file: str | None = None,
) -> str:
    filters: list[str] = []
    if int(output_fps) != int(input_fps):
        filters.append(f"fps={int(output_fps)}")
    canvas_w, canvas_h, canvas_filters = resolve_rtmp_publish_canvas(width=int(width), height=int(height))
    filters.extend(canvas_filters)
    filters.extend(_rtmp_unsharp_filters())
    watermark_filter = watermark_drawtext_filter(
        text_file=str(watermark_text_file or ""),
        width=int(canvas_w),
        height=int(canvas_h),
        env_prefixes=("LIVE_CHANNEL", "REMOTE_EDGE", "SMARTBLOG"),
    )
    if watermark_filter:
        filters.append(str(watermark_filter))
    filters.extend(_rtmp_aspect_filters(width=int(canvas_w), height=int(canvas_h)))
    live_filters = ",".join(filters) if filters else "null"
    clip_filters: list[str] = []
    clip_watermark_filter = watermark_drawtext_filter(
        text_file=str(watermark_text_file or ""),
        width=int(width),
        height=int(height),
        env_prefixes=("LIVE_CHANNEL", "REMOTE_EDGE", "SMARTBLOG"),
    )
    if clip_watermark_filter:
        clip_filters.append(str(clip_watermark_filter))
    clip_filter_chain = ",".join(clip_filters) if clip_filters else "null"
    return f"[0:v]split=2[vrtmp_src][vclip_src];[vrtmp_src]{live_filters}[vrtmp];[vclip_src]{clip_filter_chain}[vclip]"


def _rtmp_video_encoder_args(
    *,
    gop: int,
    video_bitrate: str,
    video_maxrate: str,
    video_bufsize: str,
    encoder: str | None = None,
) -> list[str]:
    encoder_s = str(encoder or os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "libx264") or "libx264").strip().lower()
    if encoder_s in {"nvenc", "h264_nvenc"}:
        return [
            "-c:v",
            "h264_nvenc",
            "-profile:v",
            "high",
            "-preset",
            str(os.getenv("LIVE_CHANNEL_RTMP_NVENC_PRESET", "p1") or "p1"),
            "-tune",
            str(os.getenv("LIVE_CHANNEL_RTMP_NVENC_TUNE", "ll") or "ll"),
            "-rc",
            "cbr",
            "-zerolatency",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            "-b:v",
            video_bitrate,
            "-maxrate",
            video_maxrate,
            "-bufsize",
            video_bufsize,
        ]
    return [
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-bf",
        "0",
        "-sc_threshold",
        "0",
        "-b:v",
        video_bitrate,
        "-maxrate",
        video_maxrate,
        "-bufsize",
        video_bufsize,
        "-x264-params",
        "nal-hrd=cbr:force-cfr=1:bframes=0",
    ]


def _tee_escape_url(value: str) -> str:
    raw = str(value or "")
    return raw.replace("\\", "\\\\").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")


def _diagnostic_flv_tee_output(path: str | None = None) -> str:
    path = str(path if path is not None else (os.getenv("LIVE_CHANNEL_RTMP_DIAG_FLV_PATH", "") or "")).strip()
    if not path:
        return ""
    return f"[f=flv:onfail=ignore]{_tee_escape_url(path)}"


def build_hls_ffmpeg_cmd(
    *,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    video_fifo: str,
    audio_fifo: str,
    playlist_path: str,
    segment_pattern: str,
) -> list[str]:
    try:
        hls_list_size = max(8, int(os.getenv("LIVE_HLS_LIST_SIZE", "30") or 30))
    except Exception:
        hls_list_size = 30
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-thread_queue_size",
        "512",
        "-i",
        video_fifo,
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-thread_queue_size",
        "512",
        "-i",
        audio_fifo,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_list_size",
        str(hls_list_size),
        "-hls_flags",
        "delete_segments+append_list+program_date_time",
        "-hls_segment_type",
        "fmp4",
        "-hls_fmp4_init_filename",
        "init.mp4",
        "-hls_segment_filename",
        segment_pattern,
        playlist_path,
    ]


def build_rtmp_ffmpeg_cmd(
    *,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    video_fifo: str,
    audio_fifo: str,
    rtmp_url: str,
    segment_pattern: str | None = None,
    segment_time_sec: float | None = None,
    input_readrate: bool = False,
    diagnostic_flv_path: str | None = None,
    watermark_text_file: str | None = None,
) -> list[str]:
    input_fps = max(1, int(fps))
    output_fps = rtmp_output_fps(input_fps)
    canvas_w, canvas_h, _ = resolve_rtmp_publish_canvas(width=int(width), height=int(height))
    keyframe_sec = max(1.0, min(4.0, _safe_float_env("LIVE_CHANNEL_RTMP_KEYFRAME_SEC", 2.0)))
    gop = max(1, int(round(float(output_fps) * float(keyframe_sec))))
    video_bitrate = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_BITRATE", "4500k") or "4500k").strip() or "4500k"
    video_maxrate = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_MAXRATE", video_bitrate) or video_bitrate).strip() or video_bitrate
    video_bufsize = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_BUFSIZE", "9000k") or "9000k").strip() or "9000k"
    audio_bitrate = str(os.getenv("LIVE_CHANNEL_RTMP_AUDIO_BITRATE", "128k") or "128k").strip() or "128k"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
    ]
    if bool(input_readrate):
        cmd.append("-re")
    cmd.extend([
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(input_fps),
        "-thread_queue_size",
        "512",
        "-i",
        video_fifo,
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
    ])
    if bool(input_readrate):
        cmd.append("-re")
    cmd.extend([
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-thread_queue_size",
        "512",
        "-i",
        audio_fifo,
    ])
    segment_pattern_s = str(segment_pattern or "").strip()
    _, _, canvas_filter = resolve_rtmp_standard_canvas(width=int(width), height=int(height))
    split_native_clip_ring = bool(
        segment_pattern_s
        and (canvas_filter or _truthy_env("REMOTE_EDGE_RTMP_SPLIT_CLIP_RING", False))
    )
    if segment_pattern_s and not split_native_clip_ring:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    if split_native_clip_ring:
        live_encoder = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "libx264") or "libx264").strip()
        clip_encoder = str(os.getenv("REMOTE_EDGE_CLIP_VIDEO_ENCODER", live_encoder) or live_encoder).strip()
        seg_sec = max(0.5, min(30.0, float(segment_time_sec if segment_time_sec is not None else 2.0)))
        rtmp_encoder_args = _rtmp_video_encoder_args(
            gop=gop,
            video_bitrate=video_bitrate,
            video_maxrate=video_maxrate,
            video_bufsize=video_bufsize,
            encoder=live_encoder,
        )
        clip_gop = max(1, int(round(float(input_fps) * float(keyframe_sec))))
        clip_encoder_args = _rtmp_video_encoder_args(
            gop=clip_gop,
            video_bitrate=video_bitrate,
            video_maxrate=video_maxrate,
            video_bufsize=video_bufsize,
            encoder=clip_encoder,
        )
        filter_complex = _rtmp_clip_split_filter_complex(
            width=int(width),
            height=int(height),
            input_fps=int(input_fps),
            output_fps=int(output_fps),
            watermark_text_file=watermark_text_file,
        )
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(
            [
                "-map",
                "[vrtmp]",
                "-map",
                "1:a:0",
                "-r",
                str(output_fps),
                *_rtmp_aspect_args(width=int(canvas_w), height=int(canvas_h)),
                *rtmp_encoder_args,
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-ar",
                "48000",
                "-ac",
                "2",
                "-f",
                "flv",
                rtmp_url,
                "-map",
                "[vclip]",
                "-map",
                "1:a:0",
                "-r",
                str(input_fps),
                *_rtmp_aspect_args(width=int(width), height=int(height)),
                *clip_encoder_args,
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-ar",
                "48000",
                "-ac",
                "2",
                "-f",
                "segment",
                "-segment_time",
                f"{seg_sec:.3f}",
                "-segment_format",
                "mpegts",
                "-reset_timestamps",
                "1",
                segment_pattern_s,
            ]
        )
        return cmd
    cmd.extend(
        _rtmp_video_filter_args(
            width=int(width),
            height=int(height),
            input_fps=int(input_fps),
            output_fps=int(output_fps),
            watermark_text_file=watermark_text_file,
        )
    )
    cmd.extend(
        [
            "-r",
            str(output_fps),
            *_rtmp_aspect_args(width=int(canvas_w), height=int(canvas_h)),
            *_rtmp_video_encoder_args(
                gop=gop,
                video_bitrate=video_bitrate,
                video_maxrate=video_maxrate,
                video_bufsize=video_bufsize,
                encoder=str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_ENCODER", "libx264") or "libx264"),
            ),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
    )
    if segment_pattern_s:
        seg_sec = max(0.5, min(30.0, float(segment_time_sec if segment_time_sec is not None else 2.0)))
        tee_outputs = [
            f"[f=flv]{_tee_escape_url(str(rtmp_url))}",
            (
                "[f=segment:onfail=ignore"
                f":segment_time={seg_sec:.3f}"
                ":segment_format=mpegts"
                ":reset_timestamps=1"
                f"]{_tee_escape_url(segment_pattern_s)}"
            ),
        ]
        diagnostic_flv = _diagnostic_flv_tee_output(diagnostic_flv_path)
        if diagnostic_flv:
            tee_outputs.append(str(diagnostic_flv))
        cmd.extend(["-f", "tee", "|".join(tee_outputs)])
    else:
        cmd.extend(["-f", "flv", rtmp_url])
    return cmd

def build_hls_rtmp_ffmpeg_cmd(
    *,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    video_fifo: str,
    audio_fifo: str,
    playlist_path: str,
    segment_pattern: str,
    rtmp_url: str,
    watermark_text_file: str | None = None,
) -> list[str]:
    input_fps = max(1, int(fps))
    output_fps = rtmp_output_fps(input_fps)
    canvas_w, canvas_h, _ = resolve_rtmp_publish_canvas(width=int(width), height=int(height))
    keyframe_sec = max(1.0, min(4.0, _safe_float_env("LIVE_CHANNEL_RTMP_KEYFRAME_SEC", 2.0)))
    rtmp_gop = max(1, int(round(float(output_fps) * float(keyframe_sec))))
    video_bitrate = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_BITRATE", "4500k") or "4500k").strip() or "4500k"
    video_maxrate = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_MAXRATE", video_bitrate) or video_bitrate).strip() or video_bitrate
    video_bufsize = str(os.getenv("LIVE_CHANNEL_RTMP_VIDEO_BUFSIZE", "9000k") or "9000k").strip() or "9000k"
    audio_bitrate = str(os.getenv("LIVE_CHANNEL_RTMP_AUDIO_BITRATE", "128k") or "128k").strip() or "128k"
    try:
        hls_list_size = max(8, int(os.getenv("LIVE_HLS_LIST_SIZE", "30") or 30))
    except Exception:
        hls_list_size = 30

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(input_fps),
        "-thread_queue_size",
        "512",
        "-i",
        video_fifo,
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-thread_queue_size",
        "512",
        "-i",
        audio_fifo,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(input_fps),
        "-keyint_min",
        str(input_fps),
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_list_size",
        str(hls_list_size),
        "-hls_flags",
        "delete_segments+append_list+program_date_time",
        "-hls_segment_type",
        "fmp4",
        "-hls_fmp4_init_filename",
        "init.mp4",
        "-hls_segment_filename",
        segment_pattern,
        playlist_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    cmd.extend(
        _rtmp_video_filter_args(
            width=int(width),
            height=int(height),
            input_fps=int(input_fps),
            output_fps=int(output_fps),
            watermark_text_file=watermark_text_file,
        )
    )
    cmd.extend(
        [
            "-r",
            str(output_fps),
            *_rtmp_aspect_args(width=int(canvas_w), height=int(canvas_h)),
            *_rtmp_video_encoder_args(
                gop=rtmp_gop,
                video_bitrate=video_bitrate,
                video_maxrate=video_maxrate,
                video_bufsize=video_bufsize,
            ),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "flv",
            rtmp_url,
        ]
    )
    return cmd
