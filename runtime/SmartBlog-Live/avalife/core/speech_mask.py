from __future__ import annotations

import math
from typing import Any


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            out.append(math.nan)
    return out


def _char_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(ch) for ch in value]
    if isinstance(value, str):
        return list(value)
    return []


def _alignment_chars_starts_ends(alignment: dict[str, Any] | None) -> tuple[list[str], list[float], list[float]]:
    if not isinstance(alignment, dict) or not alignment:
        return [], [], []
    chars = _char_list(alignment.get("characters") or alignment.get("chars"))
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
    return chars[:n], starts[:n], ends[:n]


def _speech_character_flags(chars: list[str]) -> list[bool]:
    flags: list[bool] = []
    bracket_until = -1
    for idx, raw in enumerate(chars):
        ch = str(raw or "")
        if idx <= bracket_until:
            flags.append(False)
            continue
        if ch == "[":
            close = -1
            for pos in range(idx + 1, min(len(chars), idx + 66)):
                if str(chars[pos] or "") == "]":
                    close = int(pos)
                    break
            if close > idx:
                bracket_until = int(close)
                flags.append(False)
                continue
        flags.append(bool(ch and any(part.isalnum() for part in ch)))
    return flags


def speech_intervals_from_alignment(
    alignment: dict[str, Any] | None,
    *,
    alignment_offset_sec: float = 0.0,
    duration_sec: float | None = None,
    pad_before_sec: float = 0.07,
    pad_after_sec: float = 0.22,
    merge_gap_sec: float = 0.55,
    min_segment_sec: float = 0.12,
) -> list[tuple[float, float]]:
    """Build speech intervals from ElevenLabs-style character alignment.

    The render-only cleanup removed the old WAV speech-mask path, but the
    upstream LiveAvatar pipeline still imports this helper for feature-level
    speech envelopes.
    """
    chars, starts, ends = _alignment_chars_starts_ends(alignment)
    if not chars:
        return []
    speech_flags = _speech_character_flags(chars)
    duration = None if duration_sec is None else max(0.0, float(duration_sec))
    offset = float(alignment_offset_sec or 0.0)
    raw_intervals: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None
    for is_speech, start_raw, end_raw in zip(speech_flags, starts, ends, strict=False):
        if not bool(is_speech):
            if current_start is not None and current_end is not None:
                raw_intervals.append((float(current_start), float(current_end)))
            current_start = None
            current_end = None
            continue
        start = float(start_raw) + offset
        end = float(end_raw) + offset
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end <= start:
            end = start + 0.015
        if duration is not None:
            if end <= 0.0 or start >= duration:
                continue
            start = max(0.0, min(duration, start))
            end = max(0.0, min(duration, end))
        if current_start is None:
            current_start = start
            current_end = end
        else:
            current_end = max(float(current_end if current_end is not None else end), end)
    if current_start is not None and current_end is not None:
        raw_intervals.append((float(current_start), float(current_end)))
    if not raw_intervals:
        return []

    padded: list[tuple[float, float]] = []
    for start, end in raw_intervals:
        s = float(start) - float(max(0.0, pad_before_sec))
        e = float(end) + float(max(0.0, pad_after_sec))
        if duration is not None:
            s = max(0.0, min(duration, s))
            e = max(0.0, min(duration, e))
        else:
            s = max(0.0, s)
        if e - s >= float(max(0.0, min_segment_sec)):
            padded.append((s, e))
    if not padded:
        return []

    padded.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= float(max(0.0, merge_gap_sec)):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
