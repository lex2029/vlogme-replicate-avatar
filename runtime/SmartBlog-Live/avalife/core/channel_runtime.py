from __future__ import annotations

import os
from typing import Any

CHANNEL_NAME = os.getenv("LIVE_CHANNEL_NAME", "main")
PUBLIC_OUTPUT_BASE = os.getenv("LIVE_PUBLIC_OUTPUT_BASE", "/live")
_current_sample_fps_getter = None


def configure_channel_runtime(*, rank: int, args: Any, cfg: Any, current_sample_fps_getter) -> None:
    global _current_sample_fps_getter
    _current_sample_fps_getter = current_sample_fps_getter


def _live_master_enabled() -> bool:
    v = os.getenv("LIVE_MASTER", "0").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _live_channel_enqueue_enabled(*, master: bool) -> bool:
    v = os.getenv("LIVE_CHANNEL_ENQUEUE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return not bool(master)


def _update_channel_state(**updates) -> None:
    return None


def _ensure_channel_started() -> None:
    return None
