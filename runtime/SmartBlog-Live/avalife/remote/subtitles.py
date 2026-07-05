from __future__ import annotations

import math
import os
import re
import time
from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace
from typing import Any


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(str(name), str(default)) or str(default)).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.getenv(str(name), str(default)) or default)
    except Exception:
        value = int(default)
    return int(max(int(low), min(int(high), int(value))))


def _int_env_any(names: tuple[str, ...], default: int, *, low: int, high: int) -> int:
    for name in names:
        raw = os.getenv(str(name))
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        return int(max(int(low), min(int(high), int(value))))
    return int(max(int(low), min(int(high), int(default))))


def _float_env_any(names: tuple[str, ...], default: float, *, low: float, high: float) -> float:
    for name in names:
        raw = os.getenv(str(name))
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        return float(max(float(low), min(float(high), float(value))))
    return float(max(float(low), min(float(high), float(default))))


def _float_env(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.getenv(str(name), str(default)) or default)
    except Exception:
        value = float(default)
    return float(max(float(low), min(float(high), float(value))))


def _str_env_any(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(str(name))
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return str(default)


_BRACKET_TAG_TEXT_RE = re.compile(r"\[[\w\s-]{1,64}\]", flags=re.UNICODE)
_RTL_TEXT_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]", flags=re.UNICODE)
_HEBREW_TEXT_RE = re.compile(r"[\u0590-\u05FF]", flags=re.UNICODE)
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", flags=re.UNICODE)
_JAPANESE_TEXT_RE = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF]", flags=re.UNICODE)
_CJK_TEXT_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", flags=re.UNICODE)
_HANGUL_TEXT_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", flags=re.UNICODE)
_THAI_TEXT_RE = re.compile(r"[\u0E00-\u0E7F]", flags=re.UNICODE)
_DEVANAGARI_TEXT_RE = re.compile(r"[\u0900-\u097F]", flags=re.UNICODE)
_EMOJI_TEXT_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", flags=re.UNICODE)


def _strip_bracket_tags(text: str) -> str:
    if not (
        _env_flag("REMOTE_EDGE_SUBTITLE_STRIP_BRACKET_TAGS", "1")
        and _env_flag("SMARTBLOG_RENDER_SUBTITLE_STRIP_BRACKET_TAGS", "1")
    ):
        return str(text or "")
    return _BRACKET_TAG_TEXT_RE.sub(" ", str(text or ""))


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


def _subtitle_font_file_candidates(font_path: str = "", font_key: str = "default") -> list[str]:
    configured_path = str(font_path or "").strip()
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
def _load_font(font_path: str, font_size: int, font_key: str = "default") -> Any:
    from PIL import ImageFont

    for path in _subtitle_font_file_candidates(str(font_path or ""), str(font_key or "default")):
        if not path:
            continue
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, int(font_size))
        except Exception:
            continue
    return ImageFont.load_default()


def _clean_text(text: str) -> str:
    text = _strip_bracket_tags(str(text or ""))
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    # Avoid huge headers if a model repeats itself or the source sends a full transcript.
    return text[-2000:] if len(text) > 2000 else text


class SubtitleRenderer:
    """Draws cached lightweight karaoke-style burn-in captions on RGB24 frames."""

    def __init__(self, *, width: int, height: int, fps: int, is_live: bool = False) -> None:
        self.width = int(max(1, int(width)))
        self.height = int(max(1, int(height)))
        self.fps = int(max(1, int(fps)))
        self.is_live = bool(is_live)
        self.enabled = _env_flag("REMOTE_EDGE_BURN_IN_SUBTITLES", "0")
        self.max_words = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_MAX_WORDS", "SMARTBLOG_RENDER_SUBTITLE_MAX_WORDS"),
            6,
            low=4,
            high=32,
        )
        self.max_lines = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_MAX_LINES", "SMARTBLOG_RENDER_SUBTITLE_MAX_LINES"),
            2,
            low=1,
            high=3,
        )
        self.safe_zone_preset = _str_env_any(
            ("REMOTE_EDGE_SUBTITLE_SAFE_ZONE_PRESET", "SMARTBLOG_RENDER_SUBTITLE_SAFE_ZONE_PRESET"),
            "universal_social",
        ).lower()
        self.margin_l, self.margin_r, self.margin_v = self._safe_margins()
        self.reveal_mode = _str_env_any(
            ("REMOTE_EDGE_SUBTITLE_REVEAL_MODE", "SMARTBLOG_RENDER_SUBTITLE_REVEAL_MODE"),
            "char",
        ).lower()
        if self.reveal_mode == "karaoke":
            self.reveal_mode = "char"
        self.reveal_backend = _str_env_any(
            ("REMOTE_EDGE_SUBTITLE_REVEAL_BACKEND", "SMARTBLOG_RENDER_SUBTITLE_REVEAL_BACKEND"),
            "snapshot",
        ).lower()
        self.cache_size = _int_env("REMOTE_EDGE_SUBTITLE_OVERLAY_CACHE_SIZE", 192, low=8, high=1024)
        self.font_size = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_FONT_SIZE", "SMARTBLOG_RENDER_SUBTITLE_FONT_SIZE"),
            int(max(26, min(64, round(float(self.height) * (0.040 if self.height > self.width else 0.052))))),
            low=12,
            high=120,
        )
        self.min_font_scale = _float_env_any(
            ("REMOTE_EDGE_SUBTITLE_MIN_FONT_SCALE", "SMARTBLOG_RENDER_SUBTITLE_MIN_FONT_SCALE"),
            0.86,
            low=0.75,
            high=1.0,
        )
        self.base_font_size = int(self.font_size)
        self.font_path = str(os.getenv("REMOTE_EDGE_SUBTITLE_FONT", "") or "").strip()
        self.line_char_limit = self._line_char_limit()
        self.outline = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_OUTLINE", "SMARTBLOG_RENDER_SUBTITLE_OUTLINE"),
            max(3, int(round(float(self.font_size) * 0.085))),
            low=0,
            high=14,
        )
        self.shadow = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_SHADOW", "SMARTBLOG_RENDER_SUBTITLE_SHADOW"),
            max(2, int(round(float(self.font_size) * 0.045))),
            low=0,
            high=10,
        )
        self.secondary_alpha = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_SECONDARY_ALPHA", "SMARTBLOG_RENDER_SUBTITLE_SECONDARY_ALPHA"),
            180,
            low=0,
            high=255,
        )
        self.final_hold_max_sec = _float_env(
            "REMOTE_EDGE_SUBTITLE_FINAL_HOLD_MAX_SEC",
            0.0,
            low=0.0,
            high=6.0,
        )
        self.inter_block_hold_max_sec = _float_env(
            "REMOTE_EDGE_SUBTITLE_INTER_BLOCK_HOLD_MAX_SEC",
            0.0,
            low=0.0,
            high=2.0,
        )
        self.live_lead_sec = _float_env(
            "REMOTE_EDGE_SUBTITLE_LIVE_LEAD_SEC",
            0.0,
            low=0.0,
            high=0.6,
        )
        self.live_first_block_max_lead_sec = _float_env(
            "REMOTE_EDGE_SUBTITLE_FIRST_BLOCK_MAX_LEAD_SEC",
            0.35,
            low=0.0,
            high=1.0,
        )
        self.min_visible_sec = _float_env(
            "REMOTE_EDGE_SUBTITLE_MIN_VISIBLE_SEC",
            1.05,
            low=0.0,
            high=2.5,
        )
        self.exact_timeline = _env_flag(
            "REMOTE_EDGE_SUBTITLE_EXACT_TIMELINE",
            "0",
        )
        self.linear_fallback_enabled = _env_flag("REMOTE_EDGE_SUBTITLE_LINEAR_FALLBACK", "0")
        self.font = None
        self._overlay_cache: OrderedDict[tuple[Any, ...], tuple[int, int, int, int, bytes]] = OrderedDict()
        self._block_cache: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
        self._render_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_stats_log = 0.0
        self._last_alignment_warning_log = 0.0
        self._last_no_active_block_log = 0.0
        self._last_prepare_error_log = 0.0
        self._last_render_error_log = 0.0
        if bool(self.enabled):
            try:
                self.font = _load_font(self.font_path, self.font_size)
            except Exception:
                self.enabled = False

    def prepare(self, *, text: str, alignment: dict[str, Any] | None = None) -> None:
        if not bool(self.enabled):
            return
        text_s = _clean_text(text)
        if not text_s or not isinstance(alignment, dict) or not alignment:
            return
        try:
            self._subtitle_blocks(text_s, alignment=alignment)
        except Exception:
            now = time.monotonic()
            if now - float(self._last_prepare_error_log or 0.0) < 10.0:
                return
            self._last_prepare_error_log = float(now)
            try:
                import logging

                logging.exception(
                    "Remote edge subtitles prepare failed: text_chars=%d alignment=%s",
                    int(len(text_s)),
                    bool(alignment),
                )
            except Exception:
                return

    def render(
        self,
        rgb24: bytes,
        *,
        text: str,
        progress: float,
        alignment: dict[str, Any] | None = None,
        sample_offset_samples: int | None = None,
        sample_rate: int | None = None,
        segment_end_samples: int | None = None,
    ) -> bytes:
        if not bool(self.enabled):
            return rgb24
        text_s = _clean_text(text)
        if not text_s:
            return rgb24
        try:
            words = re.findall(r"\S+", text_s)
            if not words:
                return rgb24
            elapsed_sec = self._elapsed_sec(
                sample_offset_samples=sample_offset_samples,
                sample_rate=sample_rate,
            )
            alignment_blocks = self._subtitle_blocks(text_s, alignment=alignment)
            if alignment_blocks and elapsed_sec is not None:
                final_hold_until_sec = self._elapsed_sec(
                    sample_offset_samples=segment_end_samples,
                    sample_rate=sample_rate,
                )
                block_idx, block, held = self._active_render_block_index_status(
                    alignment_blocks,
                    elapsed_sec=float(elapsed_sec),
                    final_hold_until_sec=final_hold_until_sec,
                )
                if block is None:
                    self._maybe_log_no_active_block(
                        text_s,
                        alignment_blocks,
                        elapsed_sec=float(elapsed_sec),
                        final_hold_until_sec=final_hold_until_sec,
                    )
                    return rgb24
                block_key = self._block_key(block)
                render_elapsed_sec = float(elapsed_sec)
                if bool(held):
                    render_elapsed_sec = min(float(render_elapsed_sec), float(self._block_reveal_end(block)))
                active_chars = self._active_chars_in_block(block, elapsed_sec=float(render_elapsed_sec))
                reveal_tick = int(round(float(render_elapsed_sec) * float(max(1, int(self.fps)))))
                key = (
                    "timed",
                    id(alignment),
                    text_s,
                    block_key,
                    int(active_chars),
                    int(reveal_tick),
                    int(self.width),
                    int(self.height),
                    int(self.font_size),
                    int(self.margin_l),
                    int(self.margin_r),
                    int(self.margin_v),
                    int(self.max_lines),
                    str(self.reveal_mode),
                    str(self.reveal_backend),
                    int(self.secondary_alpha),
                )
                overlay = self._overlay_cache.get(key)
                if overlay is not None:
                    self._overlay_cache.move_to_end(key)
                    self._cache_hits += 1
                else:
                    overlay = self._build_timed_overlay(block, elapsed_sec=float(render_elapsed_sec))
                    self._overlay_cache[key] = overlay
                    self._cache_misses += 1
                    while len(self._overlay_cache) > int(self.cache_size):
                        self._overlay_cache.popitem(last=False)
                self._render_count += 1
                self._maybe_log_stats()
                if overlay is None:
                    return rgb24
                x0, y0, crop_w, crop_h, crop_rgba = overlay
                if int(crop_w) <= 0 or int(crop_h) <= 0 or not crop_rgba:
                    return rgb24
                return self._composite_crop(
                    rgb24,
                    x=int(x0),
                    y=int(y0),
                    w=int(crop_w),
                    h=int(crop_h),
                    rgba=bytes(crop_rgba),
                )
            char_total = sum(len(str(word)) for word in words)
            active_chars_exact = self._active_chars_from_alignment(
                alignment,
                sample_offset_samples=sample_offset_samples,
                sample_rate=sample_rate,
            )
            if active_chars_exact is None:
                if not bool(self.linear_fallback_enabled):
                    self._maybe_log_alignment_warning(
                        text_s,
                        alignment=alignment,
                        sample_offset_samples=sample_offset_samples,
                        sample_rate=sample_rate,
                    )
                    return rgb24
                progress_f = max(0.0, min(1.0, float(progress)))
                active_chars = int(max(0, min(char_total, round(progress_f * max(1, char_total)))))
                if progress_f > 0.002 and char_total > 0:
                    active_chars = max(1, int(active_chars))
            else:
                active_chars = int(max(0, min(char_total, int(active_chars_exact))))
            active_word = self._active_word_index(words, active_chars)
            start = self._window_start(len(words), active_word)
            display_words = words[start : start + int(self.max_words)]
            active_chars_before_window = sum(len(str(word)) for word in words[:start])
            active_in_window_chars = max(0, int(active_chars) - int(active_chars_before_window))
            key = (
                text_s,
                int(start),
                int(active_in_window_chars if str(self.reveal_mode) != "word" else self._active_word_index(display_words, active_in_window_chars)),
                int(self.width),
                int(self.height),
                int(self.font_size),
                int(self.margin_l),
                int(self.margin_r),
                int(self.margin_v),
                int(self.max_lines),
                int(self.max_words),
                str(self.reveal_mode),
            )
            overlay = self._overlay_cache.get(key)
            if overlay is not None:
                self._overlay_cache.move_to_end(key)
                self._cache_hits += 1
            else:
                overlay = self._build_overlay(display_words, active_chars=int(active_in_window_chars))
                self._overlay_cache[key] = overlay
                self._cache_misses += 1
                while len(self._overlay_cache) > int(self.cache_size):
                    self._overlay_cache.popitem(last=False)
            self._render_count += 1
            self._maybe_log_stats()
            if overlay is None:
                return rgb24
            x0, y0, crop_w, crop_h, crop_rgba = overlay
            if int(crop_w) <= 0 or int(crop_h) <= 0 or not crop_rgba:
                return rgb24
            return self._composite_crop(rgb24, x=int(x0), y=int(y0), w=int(crop_w), h=int(crop_h), rgba=bytes(crop_rgba))
        except Exception:
            now = time.monotonic()
            if now - float(self._last_render_error_log or 0.0) >= 10.0:
                self._last_render_error_log = float(now)
                try:
                    import logging

                    logging.exception(
                        "Remote edge subtitles render failed: text_chars=%d sample_offset=%s sample_rate=%s",
                        int(len(text_s)),
                        "-" if sample_offset_samples is None else str(int(sample_offset_samples)),
                        "-" if sample_rate is None else str(int(sample_rate)),
                    )
                except Exception:
                    pass
            return rgb24

    def _maybe_log_no_active_block(
        self,
        text: str,
        blocks: tuple[Any, ...],
        *,
        elapsed_sec: float,
        final_hold_until_sec: float | None,
    ) -> None:
        now = time.monotonic()
        if now - float(self._last_no_active_block_log or 0.0) < 5.0:
            return
        self._last_no_active_block_log = float(now)
        try:
            import logging

            first_start = None
            first_end = None
            last_start = None
            last_end = None
            if blocks:
                first = blocks[0]
                last = blocks[-1]
                first_start = float(getattr(first, "start_sec", 0.0) or 0.0)
                first_end = float(getattr(first, "end_sec", first_start) or first_start)
                last_start = float(getattr(last, "start_sec", 0.0) or 0.0)
                last_end = float(getattr(last, "end_sec", last_start) or last_start)
            logging.warning(
                "Remote edge subtitles no active block: elapsed=%.3f final_hold=%s blocks=%d first=%s-%s last=%s-%s text_chars=%d",
                float(elapsed_sec),
                "-" if final_hold_until_sec is None else f"{float(final_hold_until_sec):.3f}",
                int(len(blocks or ())),
                "-" if first_start is None else f"{float(first_start):.3f}",
                "-" if first_end is None else f"{float(first_end):.3f}",
                "-" if last_start is None else f"{float(last_start):.3f}",
                "-" if last_end is None else f"{float(last_end):.3f}",
                int(len(str(text or ""))),
            )
        except Exception:
            return

    def _maybe_log_alignment_warning(
        self,
        text: str,
        *,
        alignment: dict[str, Any] | None,
        sample_offset_samples: int | None,
        sample_rate: int | None,
    ) -> None:
        now = time.monotonic()
        if now - float(self._last_alignment_warning_log or 0.0) < 10.0:
            return
        self._last_alignment_warning_log = float(now)
        try:
            import logging

            chars = []
            starts = []
            ends = []
            if isinstance(alignment, dict):
                chars = alignment.get("characters") or alignment.get("chars") or []
                starts = (
                    alignment.get("character_start_times_seconds")
                    or alignment.get("characterStartTimesSeconds")
                    or alignment.get("character_start_times")
                    or alignment.get("start_times")
                    or alignment.get("starts")
                    or []
                )
                ends = (
                    alignment.get("character_end_times_seconds")
                    or alignment.get("characterEndTimesSeconds")
                    or alignment.get("character_end_times")
                    or alignment.get("end_times")
                    or alignment.get("ends")
                    or []
                )
            logging.warning(
                "Remote edge subtitles skipped without exact timing: text_chars=%d alignment=%s chars=%d starts=%d ends=%d sample_offset=%s sample_rate=%s",
                int(len(str(text or ""))),
                bool(isinstance(alignment, dict) and alignment),
                int(len(chars) if isinstance(chars, (list, tuple, str)) else 0),
                int(len(starts) if isinstance(starts, (list, tuple)) else 0),
                int(len(ends) if isinstance(ends, (list, tuple)) else 0),
                "-" if sample_offset_samples is None else str(int(sample_offset_samples)),
                "-" if sample_rate is None else str(int(sample_rate)),
            )
        except Exception:
            return

    def _elapsed_sec(self, *, sample_offset_samples: int | None, sample_rate: int | None) -> float | None:
        if not isinstance(sample_offset_samples, (int, float)):
            return None
        sr = int(sample_rate or 0)
        if sr <= 0:
            return None
        return float(max(0, int(sample_offset_samples))) / float(max(1, int(sr)))

    def _subtitle_blocks(self, text: str, *, alignment: dict[str, Any] | None) -> tuple[Any, ...]:
        if not isinstance(alignment, dict) or not alignment:
            return ()
        chars = alignment.get("characters") or alignment.get("chars")
        starts = (
            alignment.get("character_start_times_seconds")
            or alignment.get("characterStartTimesSeconds")
            or alignment.get("character_start_times")
            or alignment.get("start_times")
            or alignment.get("starts")
        )
        ends = (
            alignment.get("character_end_times_seconds")
            or alignment.get("characterEndTimesSeconds")
            or alignment.get("character_end_times")
            or alignment.get("end_times")
            or alignment.get("ends")
        )
        char_count = len(chars) if isinstance(chars, (list, tuple, str)) else 0
        start_list = self._float_list(starts)
        end_list = self._float_list(ends)
        if int(char_count) <= 0 or not start_list or not end_list:
            return ()
        end_sec = max([0.1, *[float(v) for v in end_list if self._is_finite(v)]])
        key = (
            id(alignment),
            str(text or ""),
            int(char_count),
            round(float(start_list[0]), 3) if start_list else 0.0,
            round(float(end_sec), 3),
            int(self.width),
            int(self.height),
        )
        cached = self._block_cache.get(key)
        if cached is not None:
            self._block_cache.move_to_end(key)
            return cached
        try:
            from avalife.worker.render_subtitles import RenderSubtitleChunk, build_render_subtitle_blocks

            chunk = RenderSubtitleChunk(
                index=0,
                text=str(text or ""),
                start_sec=0.0,
                end_sec=float(end_sec) + 0.6,
                alignment_offset_sec=0.0,
                normalized_alignment=dict(alignment),
            )
            blocks = tuple(build_render_subtitle_blocks([chunk], width=int(self.width), height=int(self.height)))
        except Exception:
            blocks = ()
        self._block_cache[key] = blocks
        while len(self._block_cache) > 32:
            self._block_cache.popitem(last=False)
        return blocks

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return bool(float(value) == float(value) and abs(float(value)) != float("inf"))
        except Exception:
            return False

    def _active_render_block_index_status(
        self,
        blocks: tuple[Any, ...],
        *,
        elapsed_sec: float,
        final_hold_until_sec: float | None = None,
    ) -> tuple[int, Any | None, bool]:
        """Select a stable live subtitle block without changing reveal timing."""
        now = float(elapsed_sec)
        if bool(self.exact_timeline) and not bool(self.is_live):
            selected: tuple[int, Any | None, bool] = (-1, None, False)
            for idx, block in enumerate(blocks):
                start = float(getattr(block, "start_sec", 0.0) or 0.0)
                end = float(getattr(block, "end_sec", start) or start)
                if now + 0.001 >= float(start) and now <= float(end) + 0.001:
                    selected = (int(idx), block, bool(now > float(self._block_reveal_end(block)) + 0.001))
            return selected

        selected: tuple[int, Any | None, bool] = (-1, None, False)
        for idx, block in enumerate(blocks):
            visible_start, visible_end, natural_end = self._block_visible_window(
                blocks,
                int(idx),
                final_hold_until_sec=final_hold_until_sec,
            )
            if now + 0.001 >= float(visible_start) and now <= float(visible_end) + 0.001:
                selected = (int(idx), block, bool(now > float(natural_end) + 0.001))
        return selected

    def _block_visible_window(
        self,
        blocks: tuple[Any, ...],
        idx: int,
        *,
        final_hold_until_sec: float | None = None,
    ) -> tuple[float, float, float]:
        block = blocks[int(idx)]
        start = float(getattr(block, "start_sec", 0.0) or 0.0)
        reveal_end = self._block_reveal_end(block)
        natural_end = max(float(getattr(block, "end_sec", start) or start), float(reveal_end))
        visible_start = float(start)

        if bool(self.is_live):
            reveal_start = self._block_reveal_start(block)
            # build_render_subtitle_blocks already applies the same pre-roll
            # lead used by render-video. Keep that start in live too, so the
            # card is on screen before the first highlighted character instead
            # of appearing late or already partially revealed.
            visible_start = max(0.0, min(float(start), float(reveal_start)) - float(self.live_lead_sec))
            if int(idx) == 0 and float(reveal_start) <= float(self.live_first_block_max_lead_sec):
                # The first few live frames often arrive before ElevenLabs' first
                # non-space character timestamp. Show the first subtitle card at
                # the audio boundary, while keeping the per-character reveal tied
                # to the exact alignment timestamps.
                visible_start = 0.0
            if float(self.min_visible_sec) > 0.0:
                natural_end = max(float(natural_end), float(visible_start) + float(self.min_visible_sec))

        visible_end = float(natural_end)
        if int(idx) + 1 < len(blocks):
            next_start = self._block_visible_window(
                blocks,
                int(idx) + 1,
                final_hold_until_sec=final_hold_until_sec,
            )[0]
            if bool(self.is_live):
                # Live must never jump to the next subtitle screen halfway
                # through a long speech pause, but it also should not expose a
                # blank subtitle gap between two timed blocks. Keep the current
                # block fully revealed until the next block's own reveal window.
                visible_end = max(float(visible_start), float(next_start) - 0.001)
            elif float(next_start) > float(reveal_end):
                visible_end = float(next_start) - 0.001
        elif bool(self.is_live) and isinstance(final_hold_until_sec, (int, float)) and self._is_finite(final_hold_until_sec):
            visible_end = max(float(visible_end), float(final_hold_until_sec))
        elif (
            not bool(self.is_live)
            and float(self.final_hold_max_sec) > 0.0
            and isinstance(final_hold_until_sec, (int, float))
            and self._is_finite(final_hold_until_sec)
        ):
            visible_end = min(float(final_hold_until_sec), float(visible_end) + float(self.final_hold_max_sec))

        if float(visible_end) < float(visible_start):
            visible_end = float(visible_start)
        return float(visible_start), float(visible_end), float(natural_end)

    def _block_reveal_start(self, block: Any) -> float:
        fallback = float(getattr(block, "start_sec", 0.0) or 0.0)
        try:
            starts: list[float] = []
            for word in list(getattr(block, "words", ()) or ()):
                for ch in list(getattr(word, "chars", ()) or ()):
                    text = str(getattr(ch, "text", "") or "")
                    if text and not text.isspace():
                        starts.append(float(getattr(ch, "start_sec", fallback)))
            if starts:
                return max(0.0, min(float(v) for v in starts if self._is_finite(v)))
        except Exception:
            pass
        return float(fallback)

    def _block_reveal_end(self, block: Any) -> float:
        last = float(getattr(block, "end_sec", 0.0) or 0.0)
        for word in list(getattr(block, "words", ()) or ()):
            for ch in list(getattr(word, "chars", ()) or ()):
                try:
                    last = max(float(last), float(getattr(ch, "end_sec", last)))
                except Exception:
                    continue
        return float(last)

    def _block_key(self, block: Any) -> tuple[Any, ...]:
        return (
            str(getattr(block, "text", "") or ""),
            round(float(getattr(block, "start_sec", 0.0) or 0.0), 3),
            round(float(getattr(block, "end_sec", 0.0) or 0.0), 3),
            int(getattr(block, "word_count", 0) or 0),
        )

    def _active_chars_in_block(self, block: Any, *, elapsed_sec: float) -> int:
        active = 0
        for word in list(getattr(block, "words", ()) or ()):
            for ch in list(getattr(word, "chars", ()) or ()):
                text = str(getattr(ch, "text", "") or "")
                if not text or text.isspace():
                    continue
                try:
                    start = float(getattr(ch, "start_sec", 0.0) or 0.0)
                except Exception:
                    start = 0.0
                if float(elapsed_sec) + 0.001 >= float(start):
                    active += 1
        return int(active)

    def _active_chars_from_alignment(
        self,
        alignment: dict[str, Any] | None,
        *,
        sample_offset_samples: int | None,
        sample_rate: int | None,
    ) -> int | None:
        if not isinstance(alignment, dict) or not alignment:
            return None
        if not isinstance(sample_offset_samples, (int, float)):
            return None
        sr = int(sample_rate or 0)
        if sr <= 0:
            return None
        chars_raw = alignment.get("characters") or alignment.get("chars")
        starts_raw = (
            alignment.get("character_start_times_seconds")
            or alignment.get("characterStartTimesSeconds")
            or alignment.get("character_start_times")
            or alignment.get("start_times")
            or alignment.get("starts")
        )
        if isinstance(chars_raw, str):
            chars = list(chars_raw)
        elif isinstance(chars_raw, (list, tuple)):
            chars = [str(item) for item in chars_raw]
        else:
            return None
        starts = self._float_list(starts_raw)
        n = min(len(chars), len(starts))
        if n <= 0:
            return None
        elapsed_sec = float(max(0, int(sample_offset_samples))) / float(max(1, int(sr)))
        active = 0
        for idx in range(n):
            ch = str(chars[idx])
            if not ch or ch.isspace():
                continue
            try:
                start = float(starts[idx])
            except Exception:
                continue
            if elapsed_sec + 0.001 >= float(start):
                active += 1
        return int(active)

    @staticmethod
    def _float_list(value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return []
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                out.append(float("nan"))
        return out

    def _safe_margins(self) -> tuple[int, int, int]:
        if self.safe_zone_preset in {"legacy", "old"}:
            margin = int(round(float(self.width) * 0.10))
            bottom = int(round(float(self.height) * _float_env("REMOTE_EDGE_SUBTITLE_BOTTOM_MARGIN_PCT", 0.075, low=0.02, high=0.30)))
            return margin, margin, bottom
        if self.height > self.width:
            left = int(max(round(float(self.width) * 100.0 / 1080.0), round(float(self.width) * 0.09)))
            right = int(max(round(float(self.width) * 150.0 / 1080.0), round(float(self.width) * 0.13)))
            bottom = int(max(round(float(self.height) * 450.0 / 1920.0), round(float(self.height) * 0.23)))
        else:
            left = int(max(round(float(self.width) * 0.08), 64))
            right = int(max(round(float(self.width) * 0.08), 64))
            bottom = int(max(round(float(self.height) * 0.12), 72))
        left = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_MARGIN_L", "SMARTBLOG_RENDER_SUBTITLE_MARGIN_L"),
            left,
            low=0,
            high=max(0, self.width // 2),
        )
        right = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_MARGIN_R", "SMARTBLOG_RENDER_SUBTITLE_MARGIN_R"),
            right,
            low=0,
            high=max(0, self.width // 2),
        )
        bottom = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_MARGIN_V", "SMARTBLOG_RENDER_SUBTITLE_MARGIN_V"),
            bottom,
            low=0,
            high=max(0, self.height // 2),
        )
        return int(left), int(right), int(bottom)

    def _line_char_limit(self) -> int:
        available_width = max(120, int(self.width) - int(self.margin_l) - int(self.margin_r))
        char_width = max(1.0, float(self.font_size) * (0.58 if self.height > self.width else 0.54))
        default = int(max(10, available_width // char_width))
        if self.height > self.width:
            return _int_env_any(
                ("REMOTE_EDGE_SUBTITLE_CHARS_PER_LINE", "SMARTBLOG_RENDER_SUBTITLE_CHARS_PER_LINE"),
                max(24, default),
                low=18,
                high=32,
            )
        return _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_CHARS_PER_LINE", "SMARTBLOG_RENDER_SUBTITLE_CHARS_PER_LINE"),
            default,
            low=18,
            high=48,
        )

    def _subtitle_box_x_width(self, desired_w: int) -> tuple[int, int]:
        safe_w = max(1, int(self.width) - int(self.margin_l) - int(self.margin_r))
        hard_pad = int(max(2, round(max(float(self.outline), float(self.shadow), 2.0))))
        screen_w = max(1, int(self.width) - 2 * int(hard_pad))
        extra_px_default = int(round(float(self.width) * 0.06))
        extra_px = _int_env_any(
            ("REMOTE_EDGE_SUBTITLE_SAFE_OVERFLOW_PX", "SMARTBLOG_RENDER_SUBTITLE_SAFE_OVERFLOW_PX"),
            extra_px_default,
            low=0,
            high=max(0, int(self.width)),
        )
        extra_ratio = _float_env(
            "REMOTE_EDGE_SUBTITLE_SAFE_OVERFLOW_RATIO",
            0.08,
            low=0.0,
            high=1.0,
        )
        soft_extra = max(int(extra_px), int(round(float(safe_w) * float(extra_ratio))))
        max_w = int(min(int(screen_w), int(safe_w) + int(soft_extra)))
        box_w = int(min(max(1, int(desired_w)), max(1, int(max_w))))
        preferred_x = int(round(float(self.margin_l) + (float(safe_w) - float(box_w)) / 2.0))
        min_x = int(hard_pad)
        max_x = int(max(int(hard_pad), int(self.width) - int(hard_pad) - int(box_w)))
        x0 = int(max(int(min_x), min(int(max_x), int(preferred_x))))
        return int(x0), int(box_w)

    def _set_font_size(self, font_size: int) -> tuple[Any, int]:
        old = (self.font, int(self.font_size))
        size = int(max(12, min(120, int(font_size))))
        if int(size) != int(self.font_size):
            self.font_size = int(size)
            self.font = _load_font(self.font_path, int(size))
        return old

    def _restore_font(self, old: tuple[Any, int]) -> None:
        self.font, self.font_size = old

    def _font_for_text(self, text: str) -> Any:
        try:
            return _load_font(self.font_path, int(self.font_size), _subtitle_font_key(str(text or "")))
        except Exception:
            return self.font

    def _candidate_font_sizes(self) -> list[int]:
        base = int(getattr(self, "base_font_size", self.font_size) or self.font_size)
        min_size = int(max(12, math.floor(float(base) * float(self.min_font_scale))))
        sizes = list(range(int(base), int(min_size) - 1, -1))
        return sizes or [int(base)]

    def _active_word_index(self, words: list[str], active_chars: int) -> int:
        if not words:
            return 0
        remaining = int(max(0, int(active_chars)))
        for idx, word in enumerate(words):
            count = len(str(word))
            if remaining <= count:
                return int(idx)
            remaining -= int(count)
        return int(max(0, len(words) - 1))

    def _build_overlay(self, display_words: list[str], *, active_chars: int) -> tuple[int, int, int, int, bytes] | None:
        if not display_words:
            return None
        from PIL import Image, ImageDraw

        scratch = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        draw_scratch = ImageDraw.Draw(scratch)
        old_font = (self.font, int(self.font_size))
        selected: tuple[list[dict[str, Any]], int, int] | None = None
        try:
            for size in self._candidate_font_sizes():
                self._set_font_size(int(size))
                lines = self._layout_lines(draw_scratch, display_words)
                if not lines:
                    continue
                pad_x = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
                text_w = max(int(line["width"]) for line in lines)
                _x0, box_w = self._subtitle_box_x_width(int(text_w + 2 * pad_x))
                selected = (lines, int(text_w), int(size))
                if int(text_w + 2 * pad_x) <= int(box_w):
                    break
            if selected is None:
                return None
            lines, text_w, selected_size = selected
            self._set_font_size(int(selected_size))
            line_h = int(round(float(self.font_size) * 1.25))
            pad_x = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
            pad_y = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
            gap_y = int(max(0, round(float(self.font_size) * 0.08)))
            text_h = int(len(lines) * line_h + max(0, len(lines) - 1) * gap_y)
            x0, box_w = self._subtitle_box_x_width(int(text_w + 2 * pad_x))
            box_h = int(text_h + 2 * pad_y)
            y0 = int(self.height - int(self.margin_v) - box_h)
            y0 = int(max(0, min(int(self.height) - box_h, y0)))
            overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            remaining_active = int(max(0, int(active_chars)))
            y = int(pad_y)
            for line in lines:
                line_x = int(max(0, (box_w - int(line["width"])) // 2))
                line_words = [str(item[0]).strip() for item in list(line["items"] or []) if str(item[0]).strip()]
                line_text = " ".join(line_words)
                if _contains_rtl_text(line_text):
                    inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
                    alpha = 255 if int(remaining_active) > 0 else int(inactive_alpha)
                    self._draw_text(draw, line_text, x=int(line_x), y=int(y), fill=(255, 255, 255, int(alpha)))
                    remaining_active = max(0, int(remaining_active) - len(line_text.replace(" ", "")))
                    y += int(line_h + gap_y)
                    continue
                x = int(line_x)
                for word, word_w in line["items"]:
                    word_s = str(word)
                    leading = ""
                    core = word_s
                    if word_s.startswith(" "):
                        leading = " "
                        core = word_s[1:]
                        x += self._text_width(draw, " ")
                    done_chars = 0
                    if str(self.reveal_mode) == "word":
                        done_chars = len(core) if remaining_active >= len(core) else 0
                    else:
                        done_chars = max(0, min(len(core), int(remaining_active)))
                    self._draw_word(draw, core, x=x, y=y, done_chars=int(done_chars))
                    remaining_active -= len(core)
                    x += int(word_w) - (self._text_width(draw, leading) if leading else 0)
                y += int(line_h + gap_y)
            return int(x0), int(y0), int(box_w), int(box_h), overlay.tobytes()
        finally:
            self._restore_font(old_font)

    def _build_timed_overlay(self, block: Any, *, elapsed_sec: float) -> tuple[int, int, int, int, bytes] | None:
        words = list(getattr(block, "words", ()) or ())
        if not words:
            return None
        from PIL import Image, ImageDraw

        scratch = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        draw_scratch = ImageDraw.Draw(scratch)
        old_font = (self.font, int(self.font_size))

        def build_line_items() -> tuple[list[dict[str, Any]], int, int] | None:
            groups = self._timed_word_groups(draw_scratch, words, elapsed_sec=float(elapsed_sec))
            if not groups:
                return None
            space = self._text_width(draw_scratch, " ")
            items_out: list[dict[str, Any]] = []
            widest = 1
            for group in groups:
                items: list[tuple[Any, int]] = []
                line_w = 0
                for idx, word in enumerate(list(group or [])):
                    text = str(getattr(word, "text", "") or "").strip()
                    if not text:
                        continue
                    word_w = self._text_width(draw_scratch, text)
                    if idx > 0:
                        line_w += int(space)
                    items.append((word, int(word_w)))
                    line_w += int(word_w)
                if items:
                    widest = max(int(widest), int(line_w))
                    items_out.append({"items": items, "width": int(line_w)})
            if not items_out:
                return None
            return items_out, int(widest), int(space)

        selected: tuple[list[dict[str, Any]], int, int, int] | None = None
        try:
            for size in self._candidate_font_sizes():
                self._set_font_size(int(size))
                built = build_line_items()
                if built is None:
                    continue
                line_items, max_line_w, space_w = built
                pad_x = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
                _x0, crop_w = self._subtitle_box_x_width(int(max_line_w + 2 * pad_x))
                selected = (line_items, int(max_line_w), int(space_w), int(size))
                if int(max_line_w + 2 * pad_x) <= int(crop_w):
                    break
            if selected is None:
                return None
            line_items, max_line_w, space_w, selected_size = selected
            self._set_font_size(int(selected_size))

            line_h = int(round(float(self.font_size) * 1.25))
            gap_y = int(max(0, round(float(self.font_size) * 0.08)))
            pad_x = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
            pad_y = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
            x0, crop_w = self._subtitle_box_x_width(int(max_line_w + 2 * pad_x))
            crop_h = int(len(line_items) * line_h + max(0, len(line_items) - 1) * gap_y + 2 * pad_y)
            y0 = int(self.height - int(self.margin_v) - crop_h)
            y0 = int(max(0, min(int(self.height) - crop_h, y0)))
            overlay = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            y = int(pad_y)
            for line in line_items:
                x = int(max(0, (crop_w - int(line["width"])) // 2))
                line_words = [str(getattr(item[0], "text", "") or "").strip() for item in list(line["items"] or []) if str(getattr(item[0], "text", "") or "").strip()]
                line_text = " ".join(line_words)
                if _contains_rtl_text(line_text):
                    try:
                        line_start = min(float(getattr(item[0], "start_sec", 0.0) or 0.0) for item in list(line["items"] or []))
                    except Exception:
                        line_start = 0.0
                    inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
                    alpha = 255 if float(elapsed_sec) + 0.001 >= float(line_start) else int(inactive_alpha)
                    self._draw_text(draw, line_text, x=int(x), y=int(y), fill=(255, 255, 255, int(alpha)))
                    y += int(line_h + gap_y)
                    continue
                for word_idx, (word, word_w) in enumerate(list(line["items"] or [])):
                    if word_idx > 0:
                        x += int(space_w)
                    self._draw_timed_word(draw, word, x=x, y=y, elapsed_sec=float(elapsed_sec))
                    x += int(word_w)
                y += int(line_h + gap_y)
            return int(x0), int(y0), int(crop_w), int(crop_h), overlay.tobytes()
        finally:
            self._restore_font(old_font)

    def _timed_word_groups(self, draw: Any, words: list[Any], *, elapsed_sec: float) -> list[list[Any]]:
        lines = self._timed_word_lines_all(draw, words)
        max_lines = int(max(1, int(self.max_lines)))
        if len(lines) <= int(max_lines):
            return lines
        kept = lines[: max(0, int(max_lines) - 1)]
        tail: list[Any] = []
        for line in lines[max(0, int(max_lines) - 1) :]:
            tail.extend(line)
        return kept + ([tail] if tail else [])

    def _timed_word_lines_all(self, draw: Any, words: list[Any]) -> list[list[Any]]:
        clean_words = [word for word in list(words or []) if str(getattr(word, "text", "") or "").strip()]
        if not clean_words:
            return []
        try:
            from avalife.worker.render_subtitles import _wrap_word_groups

            groups = _wrap_word_groups(
                list(clean_words),
                width=int(self.width),
                height=int(self.height),
                font_size=int(self.font_size),
            )
            if groups:
                return [list(group) for group in groups]
        except Exception:
            pass
        pad_x = int(max(8, round(float(self.outline) + float(self.shadow) + 2)))
        max_width = int(
            max(
                1,
                int(self.width)
                - int(self.margin_l)
                - int(self.margin_r)
                - (2 * int(pad_x))
                - int(max(8, round(float(self.outline) + float(self.shadow)))),
            )
        )
        space_w = self._text_width(draw, " ")
        lines: list[list[Any]] = []
        cur: list[Any] = []
        cur_w = 0
        for word in self._split_timed_words_to_fit(draw, clean_words, max_width=int(max_width)):
            text = str(getattr(word, "text", "") or "").strip()
            word_w = self._text_width(draw, text)
            add_w = int(word_w + (space_w if cur else 0))
            if cur and cur_w + add_w > int(max_width):
                lines.append(cur)
                cur = [word]
                cur_w = int(word_w)
            else:
                cur.append(word)
                cur_w += int(add_w)
        if cur:
            lines.append(cur)
        return lines

    def _split_timed_words_to_fit(self, draw: Any, words: list[Any], *, max_width: int) -> list[Any]:
        out: list[Any] = []
        max_w = int(max(1, int(max_width)))
        for word in list(words or []):
            text = str(getattr(word, "text", "") or "").strip()
            if not text:
                continue
            if self._text_width(draw, text) <= int(max_w):
                out.append(word)
                continue
            chars = list(getattr(word, "chars", ()) or ())
            if not chars:
                out.extend(self._split_plain_word(draw, word, max_width=int(max_w)))
                continue
            part_chars: list[Any] = []
            part_text = ""
            part_w = 0
            for ch in chars:
                ch_text = str(getattr(ch, "text", "") or "")
                if not ch_text:
                    continue
                ch_w = self._text_width(draw, ch_text)
                if part_chars and part_w + int(ch_w) > int(max_w):
                    out.append(self._word_part_from_chars(part_text, part_chars))
                    part_chars = []
                    part_text = ""
                    part_w = 0
                part_chars.append(ch)
                part_text += ch_text
                part_w += int(ch_w)
            if part_chars:
                out.append(self._word_part_from_chars(part_text, part_chars))
        return out

    def _split_plain_word(self, draw: Any, word: Any, *, max_width: int) -> list[Any]:
        text = str(getattr(word, "text", "") or "").strip()
        if not text:
            return []
        max_w = int(max(1, int(max_width)))
        parts: list[Any] = []
        part = ""
        part_w = 0
        for ch in text:
            ch_w = self._text_width(draw, ch)
            if part and part_w + int(ch_w) > int(max_w):
                parts.append(
                    SimpleNamespace(
                        text=str(part),
                        start_sec=float(getattr(word, "start_sec", 0.0) or 0.0),
                        end_sec=float(getattr(word, "end_sec", 0.0) or 0.0),
                        chars=(),
                    )
                )
                part = ""
                part_w = 0
            part += ch
            part_w += int(ch_w)
        if part:
            parts.append(
                SimpleNamespace(
                    text=str(part),
                    start_sec=float(getattr(word, "start_sec", 0.0) or 0.0),
                    end_sec=float(getattr(word, "end_sec", 0.0) or 0.0),
                    chars=(),
                )
            )
        return parts

    @staticmethod
    def _word_part_from_chars(text: str, chars: list[Any]) -> Any:
        start = 0.0
        end = 0.0
        if chars:
            try:
                start = float(getattr(chars[0], "start_sec", 0.0) or 0.0)
            except Exception:
                start = 0.0
            try:
                end = float(getattr(chars[-1], "end_sec", start) or start)
            except Exception:
                end = float(start)
        return SimpleNamespace(text=str(text or ""), start_sec=float(start), end_sec=float(end), chars=tuple(chars))

    @staticmethod
    def _active_timed_word_index(words: list[Any], *, elapsed_sec: float) -> int:
        if not words:
            return 0
        active = 0
        for idx, word in enumerate(list(words or [])):
            try:
                start = float(getattr(word, "start_sec", 0.0) or 0.0)
            except Exception:
                start = 0.0
            if float(elapsed_sec) + 0.001 >= float(start):
                active = int(idx)
            else:
                break
        return int(max(0, min(int(active), len(words) - 1)))

    def _draw_word(self, draw: Any, word: str, *, x: int, y: int, done_chars: int) -> None:
        if _contains_rtl_text(str(word)):
            inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
            alpha = 255 if int(done_chars) > 0 else int(inactive_alpha)
            self._draw_text(draw, str(word), x=int(x), y=int(y), fill=(255, 255, 255, int(alpha)))
            return
        cursor = int(x)
        done = int(max(0, min(len(str(word)), int(done_chars))))
        inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
        for idx, ch in enumerate(str(word)):
            active = int(idx) < int(done)
            fill = (255, 255, 255, 255) if active else (255, 255, 255, int(inactive_alpha))
            self._draw_text(draw, str(ch), x=cursor, y=int(y), fill=fill)
            cursor += self._text_width(draw, str(ch))

    def _draw_timed_word(self, draw: Any, word: Any, *, x: int, y: int, elapsed_sec: float) -> None:
        cursor = int(x)
        word_text = str(getattr(word, "text", "") or "")
        if _contains_rtl_text(word_text):
            try:
                start = float(getattr(word, "start_sec", 0.0) or 0.0)
            except Exception:
                start = 0.0
            inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
            alpha = 255 if float(elapsed_sec) + 0.001 >= float(start) else int(inactive_alpha)
            self._draw_text(draw, word_text, x=int(x), y=int(y), fill=(255, 255, 255, int(alpha)))
            return
        chars = list(getattr(word, "chars", ()) or ())
        if not chars:
            text = word_text
            try:
                start = float(getattr(word, "start_sec", 0.0) or 0.0)
            except Exception:
                start = 0.0
            done_chars = len(text) if float(elapsed_sec) + 0.001 >= float(start) else 0
            self._draw_word(draw, text, x=int(x), y=int(y), done_chars=int(done_chars))
            return
        inactive_alpha = int(max(0, min(255, 255 - int(self.secondary_alpha))))
        for ch in chars:
            text = str(getattr(ch, "text", "") or "")
            if not text or text.isspace():
                cursor += self._text_width(draw, text or " ")
                continue
            try:
                start = float(getattr(ch, "start_sec", 0.0) or 0.0)
            except Exception:
                start = 0.0
            try:
                end = float(getattr(ch, "end_sec", start) or start)
            except Exception:
                end = float(start)
            if str(self.reveal_backend) in {"fade", "legacy_fade"} and text.isalnum():
                if float(elapsed_sec) + 0.001 >= float(end):
                    alpha = 255
                elif float(elapsed_sec) + 0.001 >= float(start):
                    duration = max(0.01, float(end) - float(start))
                    progress = max(0.0, min(1.0, (float(elapsed_sec) - float(start)) / float(duration)))
                    alpha = int(round(float(inactive_alpha) + (255.0 - float(inactive_alpha)) * progress))
                else:
                    alpha = int(inactive_alpha)
            elif float(elapsed_sec) + 0.001 >= float(start):
                alpha = 255
            else:
                alpha = int(inactive_alpha)
            fill = (255, 255, 255, int(max(0, min(255, int(alpha)))))
            self._draw_text(draw, text, x=int(cursor), y=int(y), fill=fill)
            cursor += self._text_width(draw, text)

    def _draw_text(self, draw: Any, text: str, *, x: int, y: int, fill: tuple[int, int, int, int]) -> None:
        text_s = str(text or "")
        font = self._font_for_text(text_s)
        kwargs: dict[str, Any] = {}
        if _contains_rtl_text(text_s):
            kwargs = {"direction": "rtl", "language": "ar"}
        shadow = int(round(float(self.shadow)))
        if shadow > 0:
            try:
                draw.text(
                    (int(x) + int(shadow), int(y) + int(shadow)),
                    text_s,
                    font=font,
                    fill=(0, 0, 0, 112),
                    stroke_width=0,
                    **kwargs,
                )
            except Exception:
                draw.text(
                    (int(x) + int(shadow), int(y) + int(shadow)),
                    text_s,
                    font=font,
                    fill=(0, 0, 0, 112),
                    stroke_width=0,
                )
        try:
            draw.text(
                (int(x), int(y)),
                text_s,
                font=font,
                fill=fill,
                stroke_width=int(max(1, round(float(self.outline)))),
                stroke_fill=(0, 0, 0, 255),
                **kwargs,
            )
        except Exception:
            draw.text(
                (int(x), int(y)),
                text_s,
                font=font,
                fill=fill,
                stroke_width=int(max(1, round(float(self.outline)))),
                stroke_fill=(0, 0, 0, 255),
            )

    def _composite_crop(self, rgb24: bytes, *, x: int, y: int, w: int, h: int, rgba: bytes) -> bytes:
        try:
            import numpy as np

            frame = np.frombuffer(rgb24, dtype=np.uint8).reshape((int(self.height), int(self.width), 3)).copy()
            crop = np.frombuffer(rgba, dtype=np.uint8).reshape((int(h), int(w), 4))
            x0 = int(max(0, min(int(self.width), int(x))))
            y0 = int(max(0, min(int(self.height), int(y))))
            x1 = int(max(x0, min(int(self.width), int(x0 + w))))
            y1 = int(max(y0, min(int(self.height), int(y0 + h))))
            if x1 <= x0 or y1 <= y0:
                return rgb24
            crop = crop[: y1 - y0, : x1 - x0, :]
            alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
            base = frame[y0:y1, x0:x1, :].astype(np.float32)
            over = crop[:, :, :3].astype(np.float32)
            frame[y0:y1, x0:x1, :] = np.clip(over * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)
            return frame.tobytes()
        except Exception:
            try:
                from PIL import Image

                img = Image.frombytes("RGB", (int(self.width), int(self.height)), bytes(rgb24)).convert("RGBA")
                overlay = Image.frombytes("RGBA", (int(w), int(h)), bytes(rgba))
                img.alpha_composite(overlay, (int(x), int(y)))
                return img.convert("RGB").tobytes()
            except Exception:
                return rgb24

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - float(self._last_stats_log or 0.0) < 30.0:
            return
        self._last_stats_log = float(now)
        if int(self._render_count) <= 0:
            return
        total = int(self._cache_hits) + int(self._cache_misses)
        hit_rate = float(self._cache_hits) / float(max(1, total))
        try:
            import logging

            logging.warning(
                "Remote edge subtitles: frames=%d cache=%d hits=%d misses=%d hit_rate=%.2f margins=%d/%d/%d mode=%s",
                int(self._render_count),
                int(len(self._overlay_cache)),
                int(self._cache_hits),
                int(self._cache_misses),
                float(hit_rate),
                int(self.margin_l),
                int(self.margin_r),
                int(self.margin_v),
                f"{str(self.reveal_mode)}/{str(self.reveal_backend)}",
            )
        except Exception:
            pass

    def _window_start(self, total_words: int, active_count: int) -> int:
        total = int(max(0, int(total_words)))
        if total <= int(self.max_words):
            return 0
        active = int(max(0, min(int(total), int(active_count))))
        # Keep the currently spoken word near the right side, karaoke-style.
        return int(max(0, min(total - int(self.max_words), active - int(self.max_words) + 3)))

    def _layout_lines(self, draw: Any, words: list[str]) -> list[dict[str, Any]]:
        if not words:
            return []
        max_width = int(max(1, int(self.width) - int(self.margin_l) - int(self.margin_r) - max(24, self.font_size)))
        space_w = self._text_width(draw, " ")
        lines: list[dict[str, Any]] = []
        cur: list[tuple[str, int]] = []
        cur_w = 0
        for word in words:
            word_w = self._text_width(draw, str(word))
            add_w = int(word_w + (space_w if cur else 0))
            if cur and cur_w + add_w > max_width and len(lines) < int(self.max_lines) - 1:
                lines.append({"items": cur, "width": cur_w})
                cur = []
                cur_w = 0
                add_w = int(word_w)
            item_w = int(word_w + (space_w if cur else 0))
            cur.append((str(word) if not cur else " " + str(word), item_w))
            cur_w += item_w
        if cur:
            lines.append({"items": cur, "width": cur_w})
        return lines[-int(self.max_lines) :]

    def _text_width(self, draw: Any, text: str) -> int:
        text_s = str(text)
        font = self._font_for_text(text_s)
        kwargs: dict[str, Any] = {}
        if _contains_rtl_text(text_s):
            kwargs = {"direction": "rtl", "language": "ar"}
        try:
            bbox = draw.textbbox((0, 0), text_s, font=font, **kwargs)
            return int(max(1, int(bbox[2] - bbox[0])))
        except Exception:
            try:
                return int(max(1, round(draw.textlength(text_s, font=font, **kwargs))))
            except Exception:
                return int(max(1, len(text_s) * max(6, self.font_size // 2)))
