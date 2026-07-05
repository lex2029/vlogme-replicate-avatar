from __future__ import annotations

import os
import re
import subprocess
import textwrap
from typing import Any


def normalize_watermark_text(value: Any, *, max_chars: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    return text[: int(max(1, int(max_chars)))]


def watermark_text_from_sources(*sources: Any) -> str:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in ("watermark_text", "watermarkText"):
            text = normalize_watermark_text(src.get(key))
            if text:
                return text
    return ""


def _safe_int_env(names: tuple[str, ...], default: int, *, low: int, high: int) -> int:
    for name in names:
        raw = os.getenv(str(name))
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(str(raw).strip())
        except Exception:
            continue
        return int(max(int(low), min(int(high), int(value))))
    return int(max(int(low), min(int(high), int(default))))


def _safe_float_env(names: tuple[str, ...], default: float, *, low: float, high: float) -> float:
    for name in names:
        raw = os.getenv(str(name))
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(str(raw).strip())
        except Exception:
            continue
        return float(max(float(low), min(float(high), float(value))))
    return float(max(float(low), min(float(high), float(default))))


def _safe_str_env(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(str(name))
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return str(default)


def watermark_font_size(width: int, height: int, *, env_prefixes: tuple[str, ...] = ("SMARTBLOG",)) -> int:
    h = int(max(1, int(height)))
    default = int(max(18, min(72, round(float(h) * 0.030))))
    names = tuple(f"{prefix}_WATERMARK_FONT_SIZE" for prefix in env_prefixes)
    return _safe_int_env(names, default, low=12, high=160)


def wrap_watermark_text(
    text: str,
    *,
    width: int,
    height: int,
    env_prefixes: tuple[str, ...] = ("SMARTBLOG",),
) -> str:
    text_s = normalize_watermark_text(text)
    if not text_s:
        return ""
    font_size = watermark_font_size(int(width), int(height), env_prefixes=env_prefixes)
    margin = int(round(float(max(1, int(width))) * 0.10))
    available = max(120, int(width) - 2 * int(margin))
    char_width = max(1.0, float(font_size) * 0.55)
    default_chars = int(max(14, min(48, int(available // char_width))))
    names = tuple(f"{prefix}_WATERMARK_CHARS_PER_LINE" for prefix in env_prefixes)
    line_chars = _safe_int_env(names, default_chars, low=8, high=80)
    lines = textwrap.wrap(
        text_s,
        width=int(line_chars),
        break_long_words=False,
        break_on_hyphens=False,
    )
    max_lines = _safe_int_env(
        tuple(f"{prefix}_WATERMARK_MAX_LINES" for prefix in env_prefixes),
        2,
        low=1,
        high=4,
    )
    if len(lines) > int(max_lines):
        kept = lines[: int(max_lines)]
        if kept:
            suffix = "..."
            if len(kept[-1]) + len(suffix) > int(line_chars):
                kept[-1] = kept[-1][: max(1, int(line_chars) - len(suffix))].rstrip()
            kept[-1] = f"{kept[-1]}{suffix}"
        lines = kept
    return "\n".join(lines) if lines else text_s


def write_watermark_text_file(
    *,
    path: str,
    text: str,
    width: int,
    height: int,
    env_prefixes: tuple[str, ...] = ("SMARTBLOG",),
) -> str:
    text_s = wrap_watermark_text(str(text or ""), width=int(width), height=int(height), env_prefixes=env_prefixes)
    if not text_s:
        return ""
    out = os.path.abspath(str(path or ""))
    if not out:
        return ""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = f"{out}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text_s)
        f.write("\n")
    os.replace(tmp, out)
    return str(out)


def _ffmpeg_filter_value(value: str) -> str:
    out = str(value or "")
    out = out.replace("\\", "\\\\")
    out = out.replace(":", "\\:")
    out = out.replace(",", "\\,")
    out = out.replace("'", "\\'")
    out = out.replace("[", "\\[")
    out = out.replace("]", "\\]")
    return out


def _font_file(env_prefixes: tuple[str, ...]) -> str:
    explicit = _safe_str_env(tuple(f"{prefix}_WATERMARK_FONT" for prefix in env_prefixes), "")
    candidates = [
        explicit,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        path_s = str(path or "").strip()
        if path_s and os.path.exists(path_s):
            return path_s
    return ""


def watermark_drawtext_filter(
    *,
    text_file: str,
    width: int,
    height: int,
    env_prefixes: tuple[str, ...] = ("SMARTBLOG",),
) -> str:
    raw_text_file = str(text_file or "").strip()
    if not raw_text_file:
        return ""
    text_path = os.path.abspath(raw_text_file)
    if not os.path.isfile(text_path) or os.path.getsize(text_path) <= 0:
        return ""
    w = int(max(1, int(width)))
    h = int(max(1, int(height)))
    font_size = watermark_font_size(w, h, env_prefixes=env_prefixes)
    position = _safe_str_env(tuple(f"{prefix}_WATERMARK_POSITION" for prefix in env_prefixes), "top_center").lower()
    margin_x = _safe_int_env(
        tuple(f"{prefix}_WATERMARK_MARGIN_X" for prefix in env_prefixes),
        int(max(18, round(float(w) * 0.045))),
        low=0,
        high=max(1, w),
    )
    margin_y = _safe_int_env(
        tuple(f"{prefix}_WATERMARK_MARGIN_Y" for prefix in env_prefixes),
        int(max(20, round(float(h) * 0.055))),
        low=0,
        high=max(1, h),
    )
    if position in {"top_right", "right_top", "tr"}:
        x_expr = f"w-text_w-{int(margin_x)}"
        y_expr = str(int(margin_y))
    elif position in {"top_left", "left_top", "tl"}:
        x_expr = str(int(margin_x))
        y_expr = str(int(margin_y))
    elif position in {"bottom_right", "right_bottom", "br"}:
        x_expr = f"w-text_w-{int(margin_x)}"
        y_expr = f"h-text_h-{int(margin_y)}"
    elif position in {"bottom_left", "left_bottom", "bl"}:
        x_expr = str(int(margin_x))
        y_expr = f"h-text_h-{int(margin_y)}"
    else:
        x_expr = "(w-text_w)/2"
        y_expr = str(int(margin_y))
    alpha = _safe_float_env(tuple(f"{prefix}_WATERMARK_TEXT_ALPHA" for prefix in env_prefixes), 0.88, low=0.10, high=1.0)
    box_alpha = _safe_float_env(tuple(f"{prefix}_WATERMARK_BOX_ALPHA" for prefix in env_prefixes), 0.34, low=0.0, high=1.0)
    border = _safe_int_env(
        tuple(f"{prefix}_WATERMARK_BOX_BORDER" for prefix in env_prefixes),
        int(max(6, round(float(font_size) * 0.36))),
        low=0,
        high=80,
    )
    shadow = _safe_int_env(
        tuple(f"{prefix}_WATERMARK_SHADOW" for prefix in env_prefixes),
        int(max(1, round(float(font_size) * 0.05))),
        low=0,
        high=20,
    )
    font_file = _font_file(env_prefixes)
    parts = [
        f"textfile={_ffmpeg_filter_value(text_path)}",
        f"fontsize={int(font_size)}",
        f"fontcolor=white@{float(alpha):.3f}",
        f"x={x_expr}",
        f"y={y_expr}",
        "box=1",
        f"boxcolor=black@{float(box_alpha):.3f}",
        f"boxborderw={int(border)}",
        "shadowcolor=black@0.70",
        f"shadowx={int(shadow)}",
        f"shadowy={int(shadow)}",
    ]
    if font_file:
        parts.insert(1, f"fontfile={_ffmpeg_filter_value(font_file)}")
    return "drawtext=" + ":".join(parts)


def burn_watermark_video(
    *,
    input_path: str,
    output_path: str,
    text: str,
    width: int,
    height: int,
    work_dir: str,
    env_prefixes: tuple[str, ...] = ("SMARTBLOG",),
) -> str:
    text_s = normalize_watermark_text(text)
    if not text_s:
        return str(input_path)
    src = os.path.abspath(str(input_path or ""))
    out = os.path.abspath(str(output_path or ""))
    if not src or not os.path.exists(src):
        raise RuntimeError(f"watermark input missing: {input_path}")
    if not out:
        raise RuntimeError("watermark output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    text_file = write_watermark_text_file(
        path=os.path.join(str(work_dir or os.path.dirname(out) or "."), "watermark.txt"),
        text=text_s,
        width=int(width),
        height=int(height),
        env_prefixes=env_prefixes,
    )
    filt = watermark_drawtext_filter(
        text_file=str(text_file),
        width=int(width),
        height=int(height),
        env_prefixes=env_prefixes,
    )
    if not filt:
        return str(input_path)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(src),
            "-vf",
            str(filt),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            str(os.getenv("SMARTBLOG_WATERMARK_VIDEO_ENCODER", "libx264") or "libx264"),
            "-preset",
            str(os.getenv("SMARTBLOG_WATERMARK_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("SMARTBLOG_WATERMARK_CRF", "18") or "18"),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return str(out)
