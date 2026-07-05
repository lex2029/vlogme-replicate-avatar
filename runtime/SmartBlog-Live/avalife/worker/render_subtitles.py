from __future__ import annotations

import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_BRACKET_TAG_RE = re.compile(r"^\[[\w\s-]{1,64}\]$", flags=re.UNICODE)
_RTL_TEXT_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]", flags=re.UNICODE)
_HEBREW_TEXT_RE = re.compile(r"[\u0590-\u05FF]", flags=re.UNICODE)
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", flags=re.UNICODE)
_JAPANESE_TEXT_RE = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF]", flags=re.UNICODE)
_CJK_TEXT_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", flags=re.UNICODE)
_HANGUL_TEXT_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", flags=re.UNICODE)
_THAI_TEXT_RE = re.compile(r"[\u0E00-\u0E7F]", flags=re.UNICODE)
_DEVANAGARI_TEXT_RE = re.compile(r"[\u0900-\u097F]", flags=re.UNICODE)
_EMOJI_TEXT_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", flags=re.UNICODE)


@dataclass(frozen=True)
class RenderSubtitleChunk:
    index: int
    text: str
    start_sec: float
    end_sec: float
    alignment_offset_sec: float = 0.0
    alignment: dict[str, Any] | None = None
    normalized_alignment: dict[str, Any] | None = None


@dataclass(frozen=True)
class _CharTiming:
    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class _WordTiming:
    text: str
    start_sec: float
    end_sec: float
    chars: tuple[_CharTiming, ...] = ()


@dataclass(frozen=True)
class _SubtitleBlock:
    text: str
    start_sec: float
    end_sec: float
    chunk_index: int
    word_count: int
    words: tuple[_WordTiming, ...] = ()


@dataclass(frozen=True)
class _SubtitleLayout:
    margin_l: int
    margin_r: int
    margin_v: int
    max_lines: int
    font_name: str
    font_size: int
    line_char_limit: int
    outline: float
    shadow: float
    secondary_colour: str
    safe_zone_preset: str


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(str(name), str(default)) or str(default)).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.getenv(str(name), str(default)) or default)
    except Exception:
        value = int(default)
    return int(max(int(low), min(int(high), int(value))))


def _float_env(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.getenv(str(name), str(default)) or default)
    except Exception:
        value = float(default)
    return float(max(float(low), min(float(high), float(value))))


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(str(name))
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _ass_colour_rgb(r: int, g: int, b: int, *, alpha: int = 0) -> str:
    a = max(0, min(255, int(alpha)))
    rr = max(0, min(255, int(r)))
    gg = max(0, min(255, int(g)))
    bb = max(0, min(255, int(b)))
    return f"&H{a:02X}{bb:02X}{gg:02X}{rr:02X}"


def _subtitle_min_font_scale() -> float:
    return _float_env("SMARTBLOG_RENDER_SUBTITLE_MIN_FONT_SCALE", 0.86, low=0.75, high=1.0)


def _contains_rtl_text(text: str) -> bool:
    return bool(_RTL_TEXT_RE.search(str(text or "")))


def _subtitle_font_key(text: str) -> str:
    text_s = str(text or "")
    if _JAPANESE_TEXT_RE.search(text_s):
        return "japanese"
    if _HANGUL_TEXT_RE.search(text_s):
        return "korean"
    if _CJK_TEXT_RE.search(text_s):
        return "cjk"
    if _ARABIC_TEXT_RE.search(text_s):
        return "arabic"
    if _HEBREW_TEXT_RE.search(text_s):
        return "hebrew"
    if _THAI_TEXT_RE.search(text_s):
        return "thai"
    if _DEVANAGARI_TEXT_RE.search(text_s):
        return "devanagari"
    if _EMOJI_TEXT_RE.search(text_s):
        return "emoji"
    return "default"


def _subtitle_font_file_candidates(font_name: str = "", font_key: str = "default") -> list[str]:
    preferred_path = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_FILE", "") or "").strip()
    configured_path = str(font_name or "").strip() if os.path.exists(str(font_name or "").strip()) else ""
    script_candidates: dict[str, list[str]] = {
        "japanese": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ],
        "cjk": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ],
        "korean": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ],
        "arabic": [
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        ],
        "hebrew": [
            "/usr/share/fonts/truetype/noto/NotoSansHebrew-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
        ],
        "thai": [
            "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        ],
        "devanagari": [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        ],
        "emoji": [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
        ],
    }
    generic = [
        preferred_path,
        configured_path,
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    if str(font_key) == "default":
        return generic + [path for paths in script_candidates.values() for path in paths]
    return list(script_candidates.get(str(font_key), ())) + generic


@lru_cache(maxsize=128)
def _measure_font(font_size: int, font_key: str = "default") -> Any:
    try:
        from PIL import ImageFont
    except Exception:
        return None
    font_name = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT", "Noto Sans") or "Noto Sans").strip()
    for path in _subtitle_font_file_candidates(font_name, str(font_key or "default")):
        if not path:
            continue
        try:
            return ImageFont.truetype(path, int(font_size))
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _measure_text_width_px(text: str, *, font_size: int, width: int, height: int) -> int:
    value = str(text or "")
    font = _measure_font(int(font_size), _subtitle_font_key(value))
    if font is not None:
        try:
            bbox = font.getbbox(value)
            return int(max(1, int(bbox[2] - bbox[0])))
        except Exception:
            pass
        try:
            return int(max(1, round(float(font.getlength(value)))))
        except Exception:
            pass
    ratio = 0.58 if int(height) > int(width) else 0.54
    return int(max(1, round(float(len(value)) * float(font_size) * float(ratio))))


def _subtitle_line_width_limit_px(width: int, height: int, *, font_size: int | None = None) -> int:
    width_i = int(max(1, int(width)))
    height_i = int(max(1, int(height)))
    layout = _subtitle_layout(width_i, height_i, font_size_override=font_size)
    safe_w = int(max(1, width_i - int(layout.margin_l) - int(layout.margin_r)))
    hard_pad = int(max(2, round(max(float(layout.outline), float(layout.shadow), 2.0))))
    screen_w = int(max(1, width_i - 2 * int(hard_pad)))
    overflow_px = _int_env(
        "SMARTBLOG_RENDER_SUBTITLE_SAFE_OVERFLOW_PX",
        int(round(float(width_i) * 0.06)),
        low=0,
        high=max(0, width_i),
    )
    overflow_ratio = _float_env("SMARTBLOG_RENDER_SUBTITLE_SAFE_OVERFLOW_RATIO", 0.08, low=0.0, high=0.50)
    soft_extra = int(max(int(overflow_px), int(round(float(safe_w) * float(overflow_ratio)))))
    draw_pad = int(max(8, round(float(layout.outline) + float(layout.shadow) + 2.0)))
    return int(max(1, min(int(screen_w), int(safe_w) + int(soft_extra)) - 2 * int(draw_pad)))


def _word_group_width_px(words: list["_WordTiming"] | tuple["_WordTiming", ...], *, width: int, height: int, font_size: int | None = None) -> int:
    layout = _subtitle_layout(int(width), int(height), font_size_override=font_size)
    size = int(layout.font_size)
    clean = [_clean_word_text(word.text) for word in list(words or []) if _clean_word_text(word.text)]
    if not clean:
        return 0
    total = 0
    for idx, text in enumerate(clean):
        if idx > 0:
            total += _measure_text_width_px(" ", font_size=int(size), width=int(width), height=int(height))
        total += _measure_text_width_px(str(text), font_size=int(size), width=int(width), height=int(height))
    return int(total)


def _alignment_from_chunk(chunk: RenderSubtitleChunk) -> dict[str, Any] | None:
    if isinstance(chunk.normalized_alignment, dict) and chunk.normalized_alignment:
        return chunk.normalized_alignment
    if isinstance(chunk.alignment, dict) and chunk.alignment:
        return chunk.alignment
    return None


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            val = float(item)
        except Exception:
            val = math.nan
        out.append(val)
    return out


def _char_list(value: Any, fallback_text: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(ch) for ch in value]
    if isinstance(value, str):
        return list(value)
    return list(str(fallback_text or ""))


def _strip_alignment_bracket_tags(
    chars: list[str],
    starts: list[float],
    ends: list[float],
    *,
    chunk_index: int,
) -> tuple[list[str], list[float], list[float]]:
    if not _env_flag("SMARTBLOG_RENDER_SUBTITLE_STRIP_BRACKET_TAGS", "1"):
        return chars, starts, ends
    n = min(len(chars), len(starts), len(ends))
    out_chars: list[str] = []
    out_starts: list[float] = []
    out_ends: list[float] = []
    tags: list[str] = []
    collapse_timing = _env_flag("SMARTBLOG_RENDER_SUBTITLE_COLLAPSE_BRACKET_TAG_TIMING", "1")
    tag_gap_sec = _float_env("SMARTBLOG_RENDER_SUBTITLE_BRACKET_TAG_GAP_SEC", 0.04, low=0.0, high=0.25)
    timing_shift_sec = 0.0

    def append_shifted(ch: str, start: float, end: float) -> None:
        shifted_start = max(0.0, float(start) - float(timing_shift_sec))
        shifted_end = max(float(shifted_start), float(end) - float(timing_shift_sec))
        if out_ends:
            shifted_start = max(float(shifted_start), float(out_ends[-1]))
            shifted_end = max(float(shifted_end), float(shifted_start))
        out_chars.append(str(ch))
        out_starts.append(float(shifted_start))
        out_ends.append(float(shifted_end))

    i = 0
    while i < n:
        ch = str(chars[i])
        if ch == "[":
            max_j = min(n, i + 66)
            j = i + 1
            while j < max_j and str(chars[j]) != "]":
                j += 1
            if j < n and str(chars[j]) == "]":
                tag = "".join(str(item) for item in chars[i : j + 1])
                if _BRACKET_TAG_RE.match(tag):
                    tags.append(tag)
                    if bool(collapse_timing):
                        tag_start = float(starts[i])
                        tag_end = max(float(tag_start), float(ends[j]))
                        tag_duration = max(0.0, float(tag_end) - float(tag_start))
                        kept_gap = min(float(tag_gap_sec), float(tag_duration))
                        append_shifted(" ", tag_start, tag_start + kept_gap)
                        timing_shift_sec += max(0.0, float(tag_duration) - float(kept_gap))
                    else:
                        append_shifted(" ", float(starts[i]), float(ends[j]))
                    i = j + 1
                    continue
        append_shifted(ch, float(starts[i]), float(ends[i]))
        i += 1
    if tags:
        logging.warning(
            "SmartBlog subtitles stripped bracket tags: chunk_index=%s count=%d collapsed=%s shift=%.3fs tags=%s",
            int(chunk_index),
            int(len(tags)),
            bool(collapse_timing),
            float(timing_shift_sec),
            ",".join(tags[:8]),
        )
    return out_chars, out_starts, out_ends


def _exact_alignment_char_timings(
    chars: list[str],
    starts: list[float],
    ends: list[float],
    *,
    offset_sec: float,
    chunk_start: float,
    chunk_end: float,
    chunk_index: int,
) -> list[_CharTiming]:
    n = min(len(chars), len(starts), len(ends))
    if n <= 0:
        return []

    out: list[_CharTiming] = []
    invalid = 0
    flat = 0
    for i in range(n):
        ch = str(chars[i])
        try:
            start = float(starts[i])
            end = float(ends[i])
        except Exception:
            invalid += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            invalid += 1
            continue
        global_start = max(float(chunk_start), float(start) + float(offset_sec))
        global_end = min(float(chunk_end), float(end) + float(offset_sec))
        if global_end < global_start:
            global_end = float(global_start)
            flat += 1
        elif global_end <= global_start + 0.001:
            flat += 1
        out.append(_CharTiming(text=str(ch), start_sec=float(global_start), end_sec=float(global_end)))

    if invalid or flat > max(8, n // 8):
        logging.warning(
            "SmartBlog subtitles using exact alignment timings: chunk_index=%s chars=%d invalid=%d flat=%d",
            int(chunk_index),
            int(n),
            int(invalid),
            int(flat),
        )
    return out


def _extract_words_from_alignment(chunk: RenderSubtitleChunk) -> list[_WordTiming]:
    alignment = _alignment_from_chunk(chunk)
    if not alignment:
        logging.warning(
            "SmartBlog subtitles skipped chunk: chunk_index=%s reason=missing_alignment",
            int(chunk.index),
        )
        return []

    chars = _char_list(alignment.get("characters") or alignment.get("chars"), str(chunk.text or ""))
    starts = _as_float_list(
        alignment.get("character_start_times_seconds")
        or alignment.get("characterStartTimesSeconds")
        or alignment.get("character_start_times")
        or alignment.get("start_times")
        or alignment.get("starts")
    )
    ends = _as_float_list(
        alignment.get("character_end_times_seconds")
        or alignment.get("characterEndTimesSeconds")
        or alignment.get("character_end_times")
        or alignment.get("end_times")
        or alignment.get("ends")
    )
    n = min(len(chars), len(starts), len(ends))
    if n <= 0:
        logging.warning(
            "SmartBlog subtitles skipped chunk: chunk_index=%s reason=empty_alignment chars=%d starts=%d ends=%d",
            int(chunk.index),
            int(len(chars)),
            int(len(starts)),
            int(len(ends)),
        )
        return []

    offset = float(chunk.alignment_offset_sec or 0.0)
    chunk_start = float(chunk.start_sec)
    chunk_end = float(chunk.end_sec)
    chars, starts, ends = _strip_alignment_bracket_tags(chars[:n], starts[:n], ends[:n], chunk_index=int(chunk.index))
    char_records = _exact_alignment_char_timings(
        chars,
        starts,
        ends,
        offset_sec=float(offset),
        chunk_start=float(chunk_start),
        chunk_end=float(chunk_end),
        chunk_index=int(chunk.index),
    )
    if not char_records:
        logging.warning(
            "SmartBlog subtitles skipped chunk: chunk_index=%s reason=no_char_records_after_alignment_parse",
            int(chunk.index),
        )
        return []

    words: list[_WordTiming] = []
    buf: list[str] = []
    char_buf: list[_CharTiming] = []
    word_start: float | None = None
    word_end: float | None = None

    def flush_word() -> None:
        nonlocal buf, char_buf, word_start, word_end
        text = "".join(buf).strip()
        if text and word_start is not None and word_end is not None:
            start = max(chunk_start, float(word_start))
            end = min(chunk_end, float(word_end))
            chars_t = tuple(ch for ch in char_buf if str(ch.text))
            words.append(_WordTiming(text=text, start_sec=float(start), end_sec=float(end), chars=chars_t))
        buf = []
        char_buf = []
        word_start = None
        word_end = None

    for record in char_records:
        ch = str(record.text)
        start = float(record.start_sec)
        end = float(record.end_sec)
        if not ch or ch.isspace():
            flush_word()
            continue
        if word_start is None:
            word_start = start
        buf.append(ch)
        char_buf.append(_CharTiming(text=ch, start_sec=float(start), end_sec=float(end)))
        word_end = max(float(word_end if word_end is not None else end), float(end))
    flush_word()
    if not words:
        logging.warning(
            "SmartBlog subtitles skipped chunk: chunk_index=%s reason=no_words_after_alignment_parse",
            int(chunk.index),
        )
    return words


def _subtitle_layout(width: int, height: int, *, font_size_override: int | None = None) -> _SubtitleLayout:
    width_i = int(max(1, int(width)))
    height_i = int(max(1, int(height)))
    portrait = bool(height_i > width_i)
    preset = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_SAFE_ZONE_PRESET", "universal_social") or "universal_social").strip().lower()
    max_lines = _int_env("SMARTBLOG_RENDER_SUBTITLE_MAX_LINES", 2, low=1, high=3)
    font_name = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT", "Noto Sans") or "Noto Sans").strip()
    font_size_default = int(round(float(height_i) * (0.040 if portrait else 0.052)))
    if font_size_override is None:
        font_size = _int_env("SMARTBLOG_RENDER_SUBTITLE_FONT_SIZE", font_size_default, low=18, high=120)
    else:
        font_size = int(max(12, min(120, int(font_size_override))))

    legacy_lr = _optional_int_env("SMARTBLOG_RENDER_SUBTITLE_MARGIN_LR")
    if preset in {"legacy", "old"}:
        margin_l_default = int(legacy_lr if legacy_lr is not None else round(float(width_i) * 0.10))
        margin_r_default = int(legacy_lr if legacy_lr is not None else round(float(width_i) * 0.10))
        margin_v_default = int(round(float(height_i) * (0.17 if portrait else 0.11)))
    elif portrait:
        margin_l_default = int(max(round(float(width_i) * 100.0 / 1080.0), round(float(width_i) * 0.09)))
        margin_r_default = int(max(round(float(width_i) * 150.0 / 1080.0), round(float(width_i) * 0.13)))
        margin_v_default = int(max(round(float(height_i) * 450.0 / 1920.0), round(float(height_i) * 0.23)))
    else:
        margin_l_default = int(max(round(float(width_i) * 0.08), 64))
        margin_r_default = int(max(round(float(width_i) * 0.08), 64))
        margin_v_default = int(max(round(float(height_i) * 0.12), 72))

    margin_l = _optional_int_env("SMARTBLOG_RENDER_SUBTITLE_MARGIN_L")
    margin_r = _optional_int_env("SMARTBLOG_RENDER_SUBTITLE_MARGIN_R")
    margin_v = _optional_int_env("SMARTBLOG_RENDER_SUBTITLE_MARGIN_V")
    if margin_l is None and legacy_lr is not None:
        margin_l = int(legacy_lr)
    if margin_r is None and legacy_lr is not None:
        margin_r = int(legacy_lr)
    margin_l = int(max(0, min(width_i // 2, margin_l_default if margin_l is None else int(margin_l))))
    margin_r = int(max(0, min(width_i // 2, margin_r_default if margin_r is None else int(margin_r))))
    margin_v = int(max(0, min(height_i // 2, margin_v_default if margin_v is None else int(margin_v))))

    available_width = int(max(120, width_i - margin_l - margin_r))
    char_width = float(font_size) * (0.58 if portrait else 0.54)
    default_limit = int(max(10, math.floor(float(available_width) / max(1.0, char_width))))
    if portrait:
        line_char_limit = _int_env("SMARTBLOG_RENDER_SUBTITLE_CHARS_PER_LINE", max(24, default_limit), low=18, high=32)
    else:
        line_char_limit = _int_env("SMARTBLOG_RENDER_SUBTITLE_CHARS_PER_LINE", default_limit, low=18, high=48)
    outline_default = max(3.0, float(font_size) * 0.085)
    shadow_default = max(2.0, float(font_size) * 0.045)
    outline = _float_env("SMARTBLOG_RENDER_SUBTITLE_OUTLINE", outline_default, low=0.0, high=14.0)
    shadow = _float_env("SMARTBLOG_RENDER_SUBTITLE_SHADOW", shadow_default, low=0.0, high=10.0)
    secondary_alpha = _int_env("SMARTBLOG_RENDER_SUBTITLE_SECONDARY_ALPHA", 180, low=0, high=255)
    return _SubtitleLayout(
        margin_l=int(margin_l),
        margin_r=int(margin_r),
        margin_v=int(margin_v),
        max_lines=int(max_lines),
        font_name=str(font_name),
        font_size=int(font_size),
        line_char_limit=int(line_char_limit),
        outline=float(outline),
        shadow=float(shadow),
        secondary_colour=_ass_colour_rgb(255, 255, 255, alpha=int(secondary_alpha)),
        safe_zone_preset=str(preset),
    )


def _subtitle_ass_font_name_for_text(text: str, default_font: str) -> str:
    key = _subtitle_font_key(str(text or ""))
    if key == "japanese":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_JA", "Noto Sans CJK JP") or "Noto Sans CJK JP").strip()
    if key == "korean":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_KO", "Noto Sans CJK KR") or "Noto Sans CJK KR").strip()
    if key == "cjk":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_ZH", "Noto Sans CJK SC") or "Noto Sans CJK SC").strip()
    if key == "arabic":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_AR", "Noto Sans Arabic") or "Noto Sans Arabic").strip()
    if key == "hebrew":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_HE", "Noto Sans Hebrew") or "Noto Sans Hebrew").strip()
    if key == "thai":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_TH", "Noto Sans Thai") or "Noto Sans Thai").strip()
    if key == "devanagari":
        return str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_FONT_HI", "Noto Sans Devanagari") or "Noto Sans Devanagari").strip()
    return str(default_font or "").strip()


def _ass_font_tag_for_text(text: str, default_font: str) -> str:
    font_name = _subtitle_ass_font_name_for_text(str(text or ""), str(default_font or ""))
    if not font_name or font_name == str(default_font or ""):
        return ""
    cleaned = str(font_name).replace("\\", "").replace("{", "").replace("}", "")
    return f"{{\\fn{cleaned}}}" if cleaned else ""


def _line_char_limit(width: int, height: int, *, font_size: int | None = None) -> int:
    return int(_subtitle_layout(int(width), int(height), font_size_override=font_size).line_char_limit)


def _clean_word_text(text: str) -> str:
    return str(text or "").strip()


def _wrap_word_groups(
    words: list[_WordTiming],
    *,
    width: int,
    height: int,
    font_size: int | None = None,
) -> list[list[_WordTiming]]:
    clean_words = [word for word in list(words or []) if _clean_word_text(word.text)]
    if not clean_words:
        return []
    limit = max(8, int(_line_char_limit(int(width), int(height), font_size=font_size)))
    pixel_limit = int(_subtitle_line_width_limit_px(int(width), int(height), font_size=font_size))
    soft_limit = int(max(limit, math.floor(float(limit) / max(0.1, _subtitle_min_font_scale()))))
    texts = [_clean_word_text(word.text) for word in clean_words]
    total = " ".join(texts)
    total_px = _word_group_width_px(clean_words, width=int(width), height=int(height), font_size=font_size)
    if (len(total) <= limit and int(total_px) <= int(pixel_limit)) or (
        _ends_boundary_symbol(texts[-1])
        and len(total) <= int(soft_limit)
        and int(total_px) <= int(pixel_limit)
    ):
        return [clean_words]
    if len(clean_words) == 2:
        if (
            len(texts[0]) <= limit
            and len(texts[1]) <= limit
            and _word_group_width_px([clean_words[0]], width=int(width), height=int(height), font_size=font_size) <= int(pixel_limit)
            and _word_group_width_px([clean_words[1]], width=int(width), height=int(height), font_size=font_size) <= int(pixel_limit)
        ):
            return [[clean_words[0]], [clean_words[1]]]
        return [clean_words]

    best_split: int | None = None
    best_score = 10**9
    for split in range(1, len(clean_words)):
        left = " ".join(texts[:split])
        right = " ".join(texts[split:])
        if len(left) > limit or len(right) > limit:
            continue
        if (
            _word_group_width_px(clean_words[:split], width=int(width), height=int(height), font_size=font_size)
            > int(pixel_limit)
            or _word_group_width_px(clean_words[split:], width=int(width), height=int(height), font_size=font_size)
            > int(pixel_limit)
        ):
            continue
        single_penalty = 1000 if split == 1 or len(clean_words) - split == 1 else 0
        boundary_tail_penalty = 1500 if len(clean_words) - split == 1 and _ends_boundary_symbol(texts[-1]) else 0
        score = abs(len(left) - len(right)) + single_penalty + boundary_tail_penalty
        if score < best_score:
            best_score = score
            best_split = split
    if best_split is not None:
        return [clean_words[:best_split], clean_words[best_split:]]

    lines: list[list[_WordTiming]] = []
    cur: list[_WordTiming] = []
    cur_len = 0
    cur_px = 0
    space_px = _measure_text_width_px(" ", font_size=int(_subtitle_layout(int(width), int(height), font_size_override=font_size).font_size), width=int(width), height=int(height))
    for word, text in zip(clean_words, texts):
        add_len = len(text) if not cur else len(text) + 1
        word_px = _measure_text_width_px(str(text), font_size=int(_subtitle_layout(int(width), int(height), font_size_override=font_size).font_size), width=int(width), height=int(height))
        add_px = int(word_px + (space_px if cur else 0))
        if cur and (cur_len + add_len > limit or cur_px + add_px > int(pixel_limit)):
            lines.append(cur)
            cur = [word]
            cur_len = len(text)
            cur_px = int(word_px)
        else:
            cur.append(word)
            cur_len += add_len
            cur_px += int(add_px)
    if cur:
        lines.append(cur)
    if (
        len(lines) >= 2
        and len(lines[-1]) == 1
        and _ends_boundary_symbol(str(lines[-1][0].text))
        and len(" ".join(_clean_word_text(word.text) for word in lines[-2] + lines[-1])) <= int(soft_limit)
        and _word_group_width_px(lines[-2] + lines[-1], width=int(width), height=int(height), font_size=font_size) <= int(pixel_limit)
    ):
        lines[-2].extend(lines[-1])
        lines.pop()
    if len(lines) >= 2 and len(lines[-1]) == 1 and len(lines[-2]) > 1:
        moved = lines[-2].pop()
        lines[-1].insert(0, moved)
    return lines


def _wrap_subtitle_lines(text: str, *, width: int, height: int, font_size: int | None = None) -> list[str]:
    words = re.findall(r"\S+", str(text or "").strip())
    if not words:
        return []
    synthetic = tuple(_WordTiming(text=str(word), start_sec=0.0, end_sec=0.0) for word in words)
    groups = _wrap_word_groups(list(synthetic), width=int(width), height=int(height), font_size=font_size)
    return [" ".join(str(word.text) for word in group) for group in groups]


def _wrap_subtitle_text(text: str, *, width: int, height: int, max_lines: int, font_size: int | None = None) -> str:
    lines = _wrap_subtitle_lines(text, width=int(width), height=int(height), font_size=font_size)
    if not lines:
        return ""
    if len(lines) <= int(max_lines):
        return r"\N".join(lines)

    # Last-resort compression if a very long word or phrase still overflowed.
    limit = max(8, int(_line_char_limit(int(width), int(height), font_size=font_size)))
    kept = lines[: max(0, int(max_lines) - 1)]
    tail = " ".join(lines[max(0, int(max_lines) - 1) :])
    kept.append(tail[: max(1, limit - 3)].rstrip() + "...")
    return r"\N".join(kept)


def _would_fit_text(
    text: str,
    *,
    width: int,
    height: int,
    max_lines: int,
    font_size: int | None = None,
) -> bool:
    lines = _wrap_subtitle_lines(text, width=int(width), height=int(height), font_size=font_size)
    limit = max(8, int(_line_char_limit(int(width), int(height), font_size=font_size)))
    pixel_limit = int(_subtitle_line_width_limit_px(int(width), int(height), font_size=font_size))
    return (
        bool(lines)
        and len(lines) <= int(max_lines)
        and all(len(line) <= limit for line in lines)
        and all(
            _measure_text_width_px(line, font_size=int(_subtitle_layout(int(width), int(height), font_size_override=font_size).font_size), width=int(width), height=int(height))
            <= int(pixel_limit)
            for line in lines
        )
    )


_BAD_TERMINAL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "nor",
    "of",
    "on",
    "or",
    "the",
    "to",
    "very",
    "with",
    "\u0430",
    "\u0431\u0435\u0437",
    "\u0432",
    "\u0432\u043e",
    "\u0434\u043b\u044f",
    "\u0434\u043e",
    "\u0437\u0430",
    "\u0438",
    "\u0438\u043b\u0438",
    "\u0438\u0437",
    "\u043a",
    "\u043a\u043e",
    "\u043d\u0430",
    "\u043d\u0430\u0434",
    "\u043d\u043e",
    "\u043e",
    "\u043e\u0431",
    "\u043e\u0442",
    "\u043f\u043e",
    "\u043f\u043e\u0434",
    "\u043f\u0440\u0438",
    "\u043f\u0440\u043e",
    "\u0441",
    "\u0441\u043e",
    "\u0443",
    "\u0447\u0435\u0440\u0435\u0437",
}


def _plain_word(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(text or "").strip().lower(), flags=re.UNICODE)


def _ends_strong(text: str) -> bool:
    return bool(re.search(r"[.!?]+[\"')\]}]*$", str(text or "").strip()))


def _ends_weak(text: str) -> bool:
    return bool(re.search(r"[,;:]+[\"')\]}]*$", str(text or "").strip()))


def _ends_boundary_symbol(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and not str(value[-1]).isalnum()


def _bad_terminal_word(word: _WordTiming) -> bool:
    plain = _plain_word(str(word.text))
    if not plain:
        return False
    if plain in _BAD_TERMINAL_WORDS:
        return True
    return len(plain) <= 2 and not _ends_strong(str(word.text)) and not _ends_weak(str(word.text))


def _words_text(items: list[_WordTiming] | tuple[_WordTiming, ...]) -> str:
    return " ".join(_clean_word_text(item.text) for item in list(items or []) if _clean_word_text(item.text))


def _block_from_words(words: list[_WordTiming], *, chunk_index: int) -> _SubtitleBlock | None:
    clean = [word for word in list(words or []) if _clean_word_text(word.text)]
    if not clean:
        return None
    if clean[-1].end_sec <= clean[0].start_sec + 0.05:
        return None
    return _SubtitleBlock(
        text=_words_text(clean),
        start_sec=float(clean[0].start_sec),
        end_sec=float(clean[-1].end_sec),
        chunk_index=int(chunk_index),
        word_count=int(len(clean)),
        words=tuple(clean),
    )


def _word_group_fits(
    words: list[_WordTiming],
    *,
    width: int,
    height: int,
    max_lines: int,
    max_words: int,
    max_duration: float,
) -> bool:
    clean = [word for word in list(words or []) if _clean_word_text(word.text)]
    if not clean:
        return False
    duration = float(clean[-1].end_sec) - float(clean[0].start_sec)
    if len(clean) > int(max_words) or duration > float(max_duration):
        return False
    return _would_fit_text(_words_text(clean), width=int(width), height=int(height), max_lines=int(max_lines))


def _word_group_soft_fits(
    words: list[_WordTiming],
    *,
    width: int,
    height: int,
    max_lines: int,
    max_words: int,
    max_duration: float,
) -> bool:
    clean = [word for word in list(words or []) if _clean_word_text(word.text)]
    if not clean:
        return False
    duration = float(clean[-1].end_sec) - float(clean[0].start_sec)
    if len(clean) > int(max_words) or duration > float(max_duration):
        return False
    layout = _subtitle_layout(int(width), int(height))
    min_size = int(max(12, math.floor(float(layout.font_size) * _subtitle_min_font_scale())))
    return _would_fit_text(
        _words_text(clean),
        width=int(width),
        height=int(height),
        max_lines=int(max_lines),
        font_size=int(min_size),
    )


def _rebalance_subtitle_blocks(
    blocks: list[_SubtitleBlock],
    *,
    width: int,
    height: int,
    min_words: int,
    max_words: int,
    max_lines: int,
    max_duration: float,
) -> list[_SubtitleBlock]:
    if len(blocks) < 2:
        return blocks
    orphan_words = _int_env("SMARTBLOG_RENDER_SUBTITLE_ORPHAN_WORDS", 2, low=1, high=4)
    short_block_sec = _float_env("SMARTBLOG_RENDER_SUBTITLE_SHORT_BLOCK_SEC", 0.75, low=0.0, high=2.0)
    relaxed_words = int(max_words) + _int_env("SMARTBLOG_RENDER_SUBTITLE_RELAX_WORDS", 2, low=0, high=6)
    relaxed_duration = float(max_duration) + _float_env("SMARTBLOG_RENDER_SUBTITLE_RELAX_DURATION_SEC", 1.2, low=0.0, high=4.0)
    out = list(blocks)

    for _ in range(4):
        changed = False
        rebuilt: list[_SubtitleBlock] = []
        idx = 0
        while idx < len(out):
            cur = out[idx]
            nxt = out[idx + 1] if idx + 1 < len(out) else None
            if nxt is not None and int(cur.chunk_index) == int(nxt.chunk_index):
                cur_words = list(cur.words or ())
                next_words = list(nxt.words or ())
                merged_words = cur_words + next_words
                cur_duration = float(cur.end_sec) - float(cur.start_sec)
                next_duration = float(nxt.end_sec) - float(nxt.start_sec)
                if cur_words and next_words and _ends_boundary_symbol(str(next_words[0].text)):
                    pulled_cur_words = cur_words + [next_words[0]]
                    kept_next_words = next_words[1:]
                    if _word_group_soft_fits(
                        pulled_cur_words,
                        width=int(width),
                        height=int(height),
                        max_lines=int(max_lines),
                        max_words=int(relaxed_words),
                        max_duration=float(relaxed_duration),
                    ) and (
                        not kept_next_words
                        or _word_group_soft_fits(
                            kept_next_words,
                            width=int(width),
                            height=int(height),
                            max_lines=int(max_lines),
                            max_words=int(relaxed_words),
                            max_duration=float(relaxed_duration),
                        )
                    ):
                        pulled = _block_from_words(pulled_cur_words, chunk_index=int(cur.chunk_index))
                        kept_next = _block_from_words(kept_next_words, chunk_index=int(nxt.chunk_index)) if kept_next_words else None
                        if pulled is not None:
                            rebuilt.append(pulled)
                            if kept_next is not None:
                                rebuilt.append(kept_next)
                            idx += 2
                            changed = True
                            continue
                should_merge = (
                    len(next_words) <= int(orphan_words)
                    or (next_words and _ends_strong(str(next_words[-1].text)) and len(next_words) <= int(orphan_words) + 1)
                    or (cur_words and _bad_terminal_word(cur_words[-1]))
                    or (float(short_block_sec) > 0.0 and float(cur_duration) <= float(short_block_sec))
                    or (float(short_block_sec) > 0.0 and float(next_duration) <= float(short_block_sec))
                )
                if should_merge and _word_group_fits(
                    merged_words,
                    width=int(width),
                    height=int(height),
                    max_lines=int(max_lines),
                    max_words=int(relaxed_words),
                    max_duration=float(relaxed_duration),
                ):
                    merged = _block_from_words(merged_words, chunk_index=int(cur.chunk_index))
                    if merged is not None:
                        rebuilt.append(merged)
                        idx += 2
                        changed = True
                        continue
                if (
                    len(next_words) <= int(orphan_words)
                    and len(cur_words) > int(min_words)
                    and not _bad_terminal_word(cur_words[-2])
                ):
                    moved_words = [cur_words[-1]] + next_words
                    kept_words = cur_words[:-1]
                    if _word_group_fits(
                        moved_words,
                        width=int(width),
                        height=int(height),
                        max_lines=int(max_lines),
                        max_words=int(relaxed_words),
                        max_duration=float(relaxed_duration),
                    ) and _word_group_fits(
                        kept_words,
                        width=int(width),
                        height=int(height),
                        max_lines=int(max_lines),
                        max_words=int(relaxed_words),
                        max_duration=float(relaxed_duration),
                    ):
                        kept = _block_from_words(kept_words, chunk_index=int(cur.chunk_index))
                        moved = _block_from_words(moved_words, chunk_index=int(nxt.chunk_index))
                        if kept is not None and moved is not None:
                            rebuilt.append(kept)
                            rebuilt.append(moved)
                            idx += 2
                            changed = True
                            continue
                if cur_words and _bad_terminal_word(cur_words[-1]) and len(cur_words) > int(min_words):
                    moved_words = [cur_words[-1]] + next_words
                    kept_words = cur_words[:-1]
                    if _word_group_fits(
                        moved_words,
                        width=int(width),
                        height=int(height),
                        max_lines=int(max_lines),
                        max_words=int(relaxed_words),
                        max_duration=float(relaxed_duration),
                    ):
                        kept = _block_from_words(kept_words, chunk_index=int(cur.chunk_index))
                        moved = _block_from_words(moved_words, chunk_index=int(nxt.chunk_index))
                        if kept is not None and moved is not None:
                            rebuilt.append(kept)
                            rebuilt.append(moved)
                            idx += 2
                            changed = True
                            continue
            rebuilt.append(cur)
            idx += 1
        out = rebuilt
        if not changed:
            break
    return out


def _words_to_blocks(
    words: list[_WordTiming],
    *,
    chunk_index: int,
    width: int,
    height: int,
) -> list[_SubtitleBlock]:
    if not words:
        return []
    min_words = _int_env("SMARTBLOG_RENDER_SUBTITLE_MIN_WORDS", 3, low=1, high=10)
    max_words = _int_env("SMARTBLOG_RENDER_SUBTITLE_MAX_WORDS", 6, low=2, high=14)
    max_lines = _int_env("SMARTBLOG_RENDER_SUBTITLE_MAX_LINES", 2, low=1, high=3)
    max_duration = _float_env("SMARTBLOG_RENDER_SUBTITLE_MAX_DURATION_SEC", 2.5, low=0.8, high=6.0)
    pause_sec = _float_env("SMARTBLOG_RENDER_SUBTITLE_PAUSE_SEC", 0.42, low=0.10, high=2.0)

    blocks: list[_SubtitleBlock] = []
    cur: list[_WordTiming] = []

    def cur_text(items: list[_WordTiming]) -> str:
        return _words_text(items)

    def emit() -> None:
        nonlocal cur
        if not cur:
            return
        block = _block_from_words(cur, chunk_index=int(chunk_index))
        if block is not None:
            blocks.append(block)
        cur = []

    for word in words:
        if not str(word.text).strip():
            continue
        if cur:
            gap = float(word.start_sec) - float(cur[-1].end_sec)
            candidate = cur + [word]
            candidate_text = cur_text(candidate)
            candidate_duration = float(candidate[-1].end_sec) - float(candidate[0].start_sec)
            should_emit = (
                gap > float(pause_sec)
                or (len(cur) >= int(min_words) and candidate_duration > float(max_duration))
                or (len(cur) >= int(min_words) and len(candidate) > int(max_words))
                or not _would_fit_text(candidate_text, width=int(width), height=int(height), max_lines=int(max_lines))
            )
            if (
                should_emit
                and len(cur) >= int(min_words)
                and _ends_boundary_symbol(str(word.text))
                and _word_group_soft_fits(
                    candidate,
                    width=int(width),
                    height=int(height),
                    max_lines=int(max_lines),
                    max_words=int(max_words) + 1,
                    max_duration=float(max_duration) + 0.8,
                )
            ):
                should_emit = False
            if should_emit:
                emit()
        cur.append(word)
        if len(cur) >= int(min_words):
            text = str(cur[-1].text)
            duration = float(cur[-1].end_sec) - float(cur[0].start_sec)
            if _ends_strong(text) or len(cur) >= int(max_words) or duration >= float(max_duration):
                emit()
            elif _ends_weak(text) and len(cur) >= max(int(min_words), 4):
                emit()
    emit()

    return _rebalance_subtitle_blocks(
        blocks,
        width=int(width),
        height=int(height),
        min_words=int(min_words),
        max_words=int(max_words),
        max_lines=int(max_lines),
        max_duration=float(max_duration),
    )


def _ass_time(sec: float) -> str:
    value = max(0.0, float(sec or 0.0))
    centis = int(round(value * 100.0))
    hours = centis // 360000
    centis -= int(hours) * 360000
    minutes = centis // 6000
    centis -= int(minutes) * 6000
    seconds = centis // 100
    centis -= int(seconds) * 100
    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}.{int(centis):02d}"


def _ass_escape(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = value.replace("\\", r"\\")
    value = value.replace("{", r"\{").replace("}", r"\}")
    return value


def _ass_escape_char(text: str) -> str:
    value = str(text or "")
    if value == "\n":
        return ""
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _ass_file_path(path: str) -> str:
    value = os.path.abspath(str(path or ""))
    return value.replace("\\", "/").replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


def _centiseconds(sec: float) -> int:
    return int(max(1, round(max(0.01, float(sec or 0.0)) * 100.0)))


def _centisecond_mark(sec: float) -> int:
    return int(max(0, round(max(0.0, float(sec or 0.0)) * 100.0)))


def _block_reveal_start(block: _SubtitleBlock) -> float:
    for word in list(block.words or ()):
        for ch in list(word.chars or ()):
            return max(float(block.start_sec), float(ch.start_sec))
    return float(block.start_sec)


def _block_reveal_end(block: _SubtitleBlock) -> float:
    last = float(block.end_sec)
    for word in list(block.words or ()):
        for ch in list(word.chars or ()):
            last = max(last, float(ch.end_sec))
    return min(float(block.end_sec), last)


def _subtitle_reveal_mode() -> str:
    mode = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_REVEAL_MODE", "karaoke") or "karaoke").strip().lower()
    if mode in {"0", "off", "false", "none", "static"}:
        return "static"
    return "karaoke"


def _subtitle_reveal_backend() -> str:
    backend = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_REVEAL_BACKEND", "snapshot") or "snapshot").strip().lower()
    if backend in {"ass", "kf", "karaoke", "legacy"}:
        return "ass_karaoke"
    return "snapshot"


def _block_font_size(block: _SubtitleBlock, *, width: int, height: int, max_lines: int) -> int:
    layout = _subtitle_layout(int(width), int(height))
    base_size = int(layout.font_size)
    min_size = int(max(12, math.floor(float(base_size) * _subtitle_min_font_scale())))
    words = list(block.words or ())
    if not words:
        return int(base_size)
    for size in range(int(base_size), int(min_size) - 1, -1):
        groups = _wrap_word_groups(words, width=int(width), height=int(height), font_size=int(size))
        if not groups:
            continue
        if len(groups) > int(max_lines):
            continue
        limit = max(8, int(_line_char_limit(int(width), int(height), font_size=int(size))))
        pixel_limit = int(_subtitle_line_width_limit_px(int(width), int(height), font_size=int(size)))
        if all(len(_words_text(group)) <= int(limit) for group in groups) and all(
            _word_group_width_px(group, width=int(width), height=int(height), font_size=int(size)) <= int(pixel_limit)
            for group in groups
        ):
            return int(size)
    return int(min_size)


def _block_ass_text(block: _SubtitleBlock, *, width: int, height: int, max_lines: int) -> str:
    font_size = _block_font_size(block, width=int(width), height=int(height), max_lines=int(max_lines))
    layout = _subtitle_layout(int(width), int(height))
    prefix = _ass_font_tag_for_text(str(block.text), str(layout.font_name))
    if int(font_size) != int(layout.font_size):
        prefix += f"{{\\fs{int(font_size)}}}"
    if _subtitle_reveal_mode() == "static" or not tuple(block.words or ()):
        return prefix + _wrap_subtitle_text(
            _ass_escape(str(block.text)),
            width=int(width),
            height=int(height),
            max_lines=int(max_lines),
            font_size=int(font_size),
        )
    groups = _wrap_word_groups(list(block.words or ()), width=int(width), height=int(height), font_size=int(font_size))
    if not groups:
        return ""
    if len(groups) > int(max_lines):
        kept = groups[: max(0, int(max_lines) - 1)]
        tail: list[_WordTiming] = []
        for group in groups[max(0, int(max_lines) - 1) :]:
            tail.extend(group)
        groups = kept + ([tail] if tail else [])
    event_start = _block_reveal_start(block)
    event_start_cs = _centisecond_mark(float(event_start))
    event_end_cs = max(event_start_cs, _centisecond_mark(float(_block_reveal_end(block))))
    total_cs = max(0, int(event_end_cs) - int(event_start_cs))
    cursor_cs = 0
    parts: list[str] = []

    def rel_cs(sec: float) -> int:
        return int(max(0, min(int(total_cs), _centisecond_mark(float(sec)) - int(event_start_cs))))

    def append_duration_text(duration_cs: int, text: str) -> None:
        nonlocal cursor_cs
        duration = int(max(0, int(duration_cs)))
        parts.append(f"{{\\kf{duration}}}{text}")
        cursor_cs = int(min(int(total_cs), int(cursor_cs) + int(duration)))

    def append_gap_until(target_cs: int) -> None:
        nonlocal cursor_cs
        target = int(max(int(cursor_cs), min(int(total_cs), int(target_cs))))
        if target > int(cursor_cs):
            append_duration_text(int(target) - int(cursor_cs), " ")
        else:
            parts.append(" ")

    def append_hidden_gap_until(target_cs: int) -> None:
        nonlocal cursor_cs
        target = int(max(int(cursor_cs), min(int(total_cs), int(target_cs))))
        if target <= int(cursor_cs):
            return
        duration = int(target) - int(cursor_cs)
        parts.append(f"{{\\alpha&HFF&\\kf{duration}}}{chr(8203)}{{\\alpha&H00&}}")
        cursor_cs = int(target)

    rtl_block = _contains_rtl_text(str(block.text))
    for line_idx, line_words in enumerate(groups):
        if line_idx > 0:
            parts.append(r"\N")
        for word_idx, word in enumerate(line_words):
            if word_idx > 0:
                append_gap_until(rel_cs(float(word.start_sec)))
            else:
                append_hidden_gap_until(rel_cs(float(word.start_sec)))
            chars = tuple(word.chars or ())
            if rtl_block or _contains_rtl_text(str(word.text)) or not chars:
                start_cs = rel_cs(float(word.start_sec))
                if start_cs > int(cursor_cs):
                    append_hidden_gap_until(int(start_cs))
                end_cs = rel_cs(float(word.end_sec))
                append_duration_text(max(0, int(end_cs) - int(cursor_cs)), _ass_escape(str(word.text)))
                continue
            for ch in chars:
                start_cs = rel_cs(float(ch.start_sec))
                end_cs = rel_cs(float(ch.end_sec))
                if start_cs > int(cursor_cs):
                    append_hidden_gap_until(int(start_cs))
                append_duration_text(max(0, int(end_cs) - int(cursor_cs)), _ass_escape_char(str(ch.text)))
    return prefix + "".join(parts)


def _block_wrapped_word_groups(
    block: _SubtitleBlock,
    *,
    width: int,
    height: int,
    max_lines: int,
    font_size: int,
) -> list[list[_WordTiming]]:
    groups = _wrap_word_groups(list(block.words or ()), width=int(width), height=int(height), font_size=int(font_size))
    if not groups:
        return []
    if len(groups) <= int(max_lines):
        return groups
    kept = groups[: max(0, int(max_lines) - 1)]
    tail: list[_WordTiming] = []
    for group in groups[max(0, int(max_lines) - 1) :]:
        tail.extend(group)
    return kept + ([tail] if tail else [])


def _block_ass_snapshot_text(block: _SubtitleBlock, *, width: int, height: int, max_lines: int, elapsed_sec: float) -> str:
    font_size = _block_font_size(block, width=int(width), height=int(height), max_lines=int(max_lines))
    layout = _subtitle_layout(int(width), int(height))
    parts: list[str] = [_ass_font_tag_for_text(str(block.text), str(layout.font_name))]
    if int(font_size) != int(layout.font_size):
        parts.append(f"{{\\fs{int(font_size)}}}")
    groups = _block_wrapped_word_groups(
        block,
        width=int(width),
        height=int(height),
        max_lines=int(max_lines),
        font_size=int(font_size),
    )
    if not groups:
        return _block_ass_text(block, width=int(width), height=int(height), max_lines=int(max_lines))
    if _contains_rtl_text(str(block.text)):
        return (
            "".join(parts)
            + _wrap_subtitle_text(
                _ass_escape(str(block.text)),
                width=int(width),
                height=int(height),
                max_lines=int(max_lines),
                font_size=int(font_size),
            )
        )

    inactive_alpha = _int_env("SMARTBLOG_RENDER_SUBTITLE_SECONDARY_ALPHA", 180, low=0, high=255)
    active_alpha = 0
    current_alpha: int | None = None

    def set_alpha(alpha: int) -> None:
        nonlocal current_alpha
        value = int(max(0, min(255, int(alpha))))
        if current_alpha == value:
            return
        parts.append(f"{{\\1a&H{value:02X}&}}")
        current_alpha = int(value)

    def char_active(ch: _CharTiming) -> bool:
        text = str(ch.text or "")
        try:
            start = float(ch.start_sec)
        except Exception:
            start = float(block.start_sec)
        try:
            end = float(ch.end_sec)
        except Exception:
            end = float(start)
        if not math.isfinite(start):
            start = float(block.start_sec)
        if not math.isfinite(end):
            end = float(start)
        probe = float(elapsed_sec) + 0.001
        if not text.isalnum() and probe >= float(start):
            return True
        return bool(probe >= float(start))

    for line_idx, line_words in enumerate(groups):
        if line_idx > 0:
            parts.append(r"\N")
        for word_idx, word in enumerate(line_words):
            if word_idx > 0:
                parts.append(" ")
            chars = tuple(word.chars or ())
            if not chars:
                set_alpha(active_alpha if float(elapsed_sec) + 0.001 >= float(word.start_sec) else inactive_alpha)
                parts.append(_ass_escape(str(word.text)))
                continue
            for ch in chars:
                text = str(ch.text or "")
                if not text:
                    continue
                set_alpha(active_alpha if char_active(ch) else inactive_alpha)
                parts.append(_ass_escape_char(text))
    parts.append("{\\1a&H00&}")
    return "".join(parts)


def _block_reveal_points_cs(block: _SubtitleBlock) -> list[int]:
    start_cs = _centisecond_mark(float(block.start_sec))
    end_cs = max(start_cs + 1, _centisecond_mark(max(float(block.end_sec), float(_block_reveal_end(block)))))
    if _contains_rtl_text(str(block.text)):
        return [int(start_cs), int(end_cs)]
    points = {int(start_cs), int(end_cs)}
    for word in list(block.words or ()):
        if not tuple(word.chars or ()):
            points.add(int(max(start_cs, min(end_cs, _centisecond_mark(float(word.start_sec))))))
            continue
        for ch in list(word.chars or ()):
            points.add(int(max(start_cs, min(end_cs, _centisecond_mark(float(ch.start_sec))))))
    return sorted(point for point in points if int(start_cs) <= int(point) <= int(end_cs))


def _block_ass_snapshot_dialogues(block: _SubtitleBlock, *, width: int, height: int, max_lines: int) -> list[str]:
    points = _block_reveal_points_cs(block)
    if len(points) < 2:
        return []
    lines: list[str] = []
    for start_cs, end_cs in zip(points, points[1:]):
        if int(end_cs) <= int(start_cs):
            continue
        start_sec = float(start_cs) / 100.0
        end_sec = float(end_cs) / 100.0
        text = _block_ass_snapshot_text(
            block,
            width=int(width),
            height=int(height),
            max_lines=int(max_lines),
            elapsed_sec=float(start_sec),
        )
        if not text:
            continue
        lines.append(
            "Dialogue: "
            f"0,{_ass_time(float(start_sec))},{_ass_time(float(end_sec))},"
            f"Default,chunk_{int(block.chunk_index)},0,0,0,,{text}"
        )
    return lines


def build_render_subtitle_blocks(
    chunks: list[RenderSubtitleChunk],
    *,
    width: int,
    height: int,
) -> list[_SubtitleBlock]:
    blocks: list[_SubtitleBlock] = []
    for chunk in list(chunks or []):
        words = _extract_words_from_alignment(chunk)
        if not words:
            continue
        chunk_blocks = _words_to_blocks(words, chunk_index=int(chunk.index), width=int(width), height=int(height))
        chunk_start = float(chunk.start_sec)
        chunk_end = float(chunk.end_sec)
        lead = _float_env("SMARTBLOG_RENDER_SUBTITLE_LEAD_SEC", 0.0, low=0.0, high=0.3)
        hold = _float_env("SMARTBLOG_RENDER_SUBTITLE_HOLD_SEC", 0.12, low=0.0, high=0.6)
        hold_last_to_chunk_end = _env_flag("SMARTBLOG_RENDER_SUBTITLE_HOLD_LAST_BLOCK_TO_CHUNK_END", "1")
        for block_idx, block in enumerate(chunk_blocks):
            start = max(chunk_start, float(block.start_sec) - float(lead))
            end = min(chunk_end, float(block.end_sec) + float(hold))
            if bool(hold_last_to_chunk_end) and int(block_idx) == int(len(chunk_blocks) - 1):
                end = float(chunk_end)
            if end <= start + 0.08:
                continue
            blocks.append(
                _SubtitleBlock(
                    text=str(block.text),
                    start_sec=float(start),
                    end_sec=float(end),
                    chunk_index=int(chunk.index),
                    word_count=int(block.word_count),
                    words=tuple(block.words or ()),
                )
            )
    blocks.sort(key=lambda item: (float(item.start_sec), int(item.chunk_index)))
    non_overlapping: list[_SubtitleBlock] = []
    for idx, block in enumerate(blocks):
        end = float(block.end_sec)
        if idx + 1 < len(blocks):
            end = min(end, max(float(block.start_sec), float(blocks[idx + 1].start_sec) - 0.01))
        if end <= float(block.start_sec) + 0.08:
            continue
        non_overlapping.append(
            _SubtitleBlock(
                text=str(block.text),
                start_sec=float(block.start_sec),
                end_sec=float(end),
                chunk_index=int(block.chunk_index),
                word_count=int(block.word_count),
                words=tuple(block.words or ()),
            )
        )
    return non_overlapping


def write_render_subtitles_ass(
    chunks: list[RenderSubtitleChunk],
    *,
    out_path: str,
    width: int,
    height: int,
) -> int:
    blocks = build_render_subtitle_blocks(list(chunks or []), width=int(width), height=int(height))
    out = os.path.abspath(str(out_path or ""))
    if not out:
        raise RuntimeError("subtitle ASS output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    layout = _subtitle_layout(int(width), int(height))
    max_lines = int(layout.max_lines)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{layout.font_name},{int(layout.font_size)},&H00FFFFFF,{layout.secondary_colour},"
            f"&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,"
            f"{float(layout.outline):.1f},{float(layout.shadow):.1f},2,"
            f"{int(layout.margin_l)},{int(layout.margin_r)},{int(layout.margin_v)},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = list(header)
    reveal_backend = _subtitle_reveal_backend()
    dialogue_count = 0
    for block in blocks:
        if _subtitle_reveal_mode() != "static" and str(reveal_backend) == "snapshot":
            dialogues = _block_ass_snapshot_dialogues(
                block,
                width=int(width),
                height=int(height),
                max_lines=int(max_lines),
            )
            lines.extend(dialogues)
            dialogue_count += int(len(dialogues))
        else:
            wrapped = _block_ass_text(block, width=int(width), height=int(height), max_lines=int(max_lines))
            if not wrapped:
                continue
            start_sec = _block_reveal_start(block) if _subtitle_reveal_mode() != "static" else float(block.start_sec)
            end_sec = max(float(block.end_sec), _block_reveal_end(block))
            lines.append(
                "Dialogue: "
                f"0,{_ass_time(float(start_sec))},{_ass_time(float(end_sec))},"
                f"Default,chunk_{int(block.chunk_index)},0,0,0,,{wrapped}"
            )
            dialogue_count += 1
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    logging.warning(
        "SmartBlog render subtitle layout: blocks=%d dialogues=%d size=%dx%d safe_zone=%s margins=%d/%d/%d font=%d line_chars=%d reveal=%s backend=%s",
        int(len(blocks)),
        int(dialogue_count),
        int(width),
        int(height),
        str(layout.safe_zone_preset),
        int(layout.margin_l),
        int(layout.margin_r),
        int(layout.margin_v),
        int(layout.font_size),
        int(layout.line_char_limit),
        str(_subtitle_reveal_mode()),
        str(reveal_backend),
    )
    return int(len(blocks))


def burn_ass_subtitles(
    *,
    input_path: str,
    ass_path: str,
    output_path: str,
) -> str:
    src = os.path.abspath(str(input_path or ""))
    ass = os.path.abspath(str(ass_path or ""))
    out = os.path.abspath(str(output_path or ""))
    if not src or not os.path.exists(src):
        raise RuntimeError(f"subtitle burn input missing: {input_path}")
    if not ass or not os.path.exists(ass):
        raise RuntimeError(f"subtitle ASS file missing: {ass_path}")
    if not out:
        raise RuntimeError("subtitle burn output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
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
            f"ass={_ass_file_path(str(ass))}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_VIDEO_ENCODER", "libx264") or "libx264"),
            "-preset",
            str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_CRF", "18") or "18"),
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
