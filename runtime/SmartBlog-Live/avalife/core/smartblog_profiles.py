from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SmartBlogLiveProfile:
    orientation: str
    render_size: str
    render_height: int
    render_width: int
    output_width: int
    output_height: int


SMARTBLOG_LIVE_PROFILE_BASE = "base"
SMARTBLOG_LIVE_PROFILE_COMPACT_704 = "compact_704"
SMARTBLOG_LIVE_PROFILE_HIGH_RES_1_5X = "highres_1_5x"
SMARTBLOG_LIVE_PROFILE_HIGH_RES_2X = "highres_2x"
SMARTBLOG_LIVE_PROFILE_NATIVE_720P = "native720p"

_SMARTBLOG_LIVE_BASE_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    SmartBlogLiveProfile(
        orientation="portrait",
        render_size="384*256",
        render_height=384,
        render_width=256,
        output_width=720,
        output_height=1280,
    ),
    SmartBlogLiveProfile(
        orientation="landscape",
        render_size="256*384",
        render_height=256,
        render_width=384,
        output_width=1280,
        output_height=720,
    ),
)

_SMARTBLOG_LIVE_COMPACT_704_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    SmartBlogLiveProfile(
        orientation="portrait",
        render_size="704*384",
        render_height=704,
        render_width=384,
        output_width=720,
        output_height=1280,
    ),
    SmartBlogLiveProfile(
        orientation="landscape",
        render_size="384*704",
        render_height=384,
        render_width=704,
        output_width=1280,
        output_height=720,
    ),
)

_SMARTBLOG_LIVE_HIGH_RES_1_5X_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    SmartBlogLiveProfile(
        orientation="portrait",
        render_size="720*400",
        # Wan pads the nominal 720*400 profile down to the effective 64-aligned
        # generation size. Keep render_size as the request key, but use the real
        # frame dimensions for reference prep and downstream shape checks.
        render_height=704,
        render_width=384,
        output_width=720,
        output_height=1280,
    ),
    SmartBlogLiveProfile(
        orientation="landscape",
        render_size="400*720",
        render_height=384,
        render_width=704,
        output_width=1280,
        output_height=720,
    ),
)

_SMARTBLOG_LIVE_HIGH_RES_2X_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    SmartBlogLiveProfile(
        orientation="portrait",
        render_size="832*448",
        render_height=832,
        render_width=448,
        output_width=720,
        output_height=1280,
    ),
    SmartBlogLiveProfile(
        orientation="landscape",
        render_size="448*832",
        render_height=448,
        render_width=832,
        output_width=1280,
        output_height=720,
    ),
)

_SMARTBLOG_LIVE_NATIVE_720P_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    SmartBlogLiveProfile(
        orientation="portrait",
        render_size="1280*720",
        render_height=1280,
        render_width=720,
        output_width=720,
        output_height=1280,
    ),
    SmartBlogLiveProfile(
        orientation="landscape",
        render_size="720*1280",
        render_height=720,
        render_width=1280,
        output_width=1280,
        output_height=720,
    ),
)

SMARTBLOG_LIVE_PROFILES: tuple[SmartBlogLiveProfile, ...] = (
    *_SMARTBLOG_LIVE_BASE_PROFILES,
    *_SMARTBLOG_LIVE_COMPACT_704_PROFILES,
    *_SMARTBLOG_LIVE_HIGH_RES_1_5X_PROFILES,
    *_SMARTBLOG_LIVE_HIGH_RES_2X_PROFILES,
    *_SMARTBLOG_LIVE_NATIVE_720P_PROFILES,
)

SMARTBLOG_LIVE_PROFILES_BY_SIZE = {profile.render_size: profile for profile in SMARTBLOG_LIVE_PROFILES}
SMARTBLOG_LIVE_PROFILES_BY_ORIENTATION = {
    profile.orientation: profile for profile in _SMARTBLOG_LIVE_BASE_PROFILES
}
SMARTBLOG_LIVE_COMPACT_704_PROFILES_BY_ORIENTATION = {
    profile.orientation: profile for profile in _SMARTBLOG_LIVE_COMPACT_704_PROFILES
}
SMARTBLOG_LIVE_HIGH_RES_1_5X_PROFILES_BY_ORIENTATION = {
    profile.orientation: profile for profile in _SMARTBLOG_LIVE_HIGH_RES_1_5X_PROFILES
}
SMARTBLOG_LIVE_HIGH_RES_2X_PROFILES_BY_ORIENTATION = {
    profile.orientation: profile for profile in _SMARTBLOG_LIVE_HIGH_RES_2X_PROFILES
}
SMARTBLOG_LIVE_NATIVE_720P_PROFILES_BY_ORIENTATION = {
    profile.orientation: profile for profile in _SMARTBLOG_LIVE_NATIVE_720P_PROFILES
}
SMARTBLOG_RUNTIME_SIZE_CONFIGS = tuple(profile.render_size for profile in SMARTBLOG_LIVE_PROFILES)

_PORTRAIT_WORDS = {
    "portrait",
    "vertical",
    "vert",
    "v",
    "short",
    "shorts",
    "reel",
    "reels",
    "story",
    "stories",
    "2:3",
    "9:16",
}
_LANDSCAPE_WORDS = {
    "landscape",
    "horizontal",
    "horiz",
    "wide",
    "h",
    "3:2",
    "16:9",
}


def smartblog_live_profile_for_size(size: str) -> SmartBlogLiveProfile:
    size_s = str(size or "").strip()
    try:
        return SMARTBLOG_LIVE_PROFILES_BY_SIZE[size_s]
    except KeyError as e:
        raise RuntimeError(
            "Unsupported SmartBlog live size "
            f"{size_s!r}; use one of {', '.join(SMARTBLOG_RUNTIME_SIZE_CONFIGS)}"
        ) from e


def smartblog_live_profile_variant() -> str:
    raw = str(os.getenv("SMARTBLOG_LIVE_PROFILE", "") or "").strip().lower()
    raw = raw.replace("-", "_").replace(".", "_")
    if raw in {"native720p", "native_720p", "native_720", "native_social", "social_native"}:
        return SMARTBLOG_LIVE_PROFILE_NATIVE_720P
    if raw in {"highres_2x", "high_res_2x", "2x", "832", "832p"}:
        return SMARTBLOG_LIVE_PROFILE_HIGH_RES_2X
    if str(os.getenv("SMARTBLOG_LIVE_HIGH_RES_2X", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return SMARTBLOG_LIVE_PROFILE_HIGH_RES_2X
    if raw in {
        "compact",
        "compact_704",
        "compact-704",
        "704",
        "704p",
        "b200",
        "b200_safe",
        "b200-safe",
    }:
        return SMARTBLOG_LIVE_PROFILE_COMPACT_704
    if raw in {"highres", "high_res", "highres_1_5x", "high_res_1_5x", "1_5x", "1_5", "720", "720p", "hd"}:
        return SMARTBLOG_LIVE_PROFILE_HIGH_RES_1_5X
    if str(os.getenv("SMARTBLOG_LIVE_HIGH_RES", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return SMARTBLOG_LIVE_PROFILE_HIGH_RES_1_5X
    return SMARTBLOG_LIVE_PROFILE_BASE


def smartblog_live_profile_for_orientation(orientation: str) -> SmartBlogLiveProfile:
    orientation_s = str(orientation or "portrait").strip().lower()
    if orientation_s != "landscape":
        orientation_s = "portrait"
    variant = smartblog_live_profile_variant()
    if variant == SMARTBLOG_LIVE_PROFILE_NATIVE_720P:
        return SMARTBLOG_LIVE_NATIVE_720P_PROFILES_BY_ORIENTATION[orientation_s]
    if variant == SMARTBLOG_LIVE_PROFILE_HIGH_RES_2X:
        return SMARTBLOG_LIVE_HIGH_RES_2X_PROFILES_BY_ORIENTATION[orientation_s]
    if variant == SMARTBLOG_LIVE_PROFILE_COMPACT_704:
        return SMARTBLOG_LIVE_COMPACT_704_PROFILES_BY_ORIENTATION[orientation_s]
    if variant == SMARTBLOG_LIVE_PROFILE_HIGH_RES_1_5X:
        return SMARTBLOG_LIVE_HIGH_RES_1_5X_PROFILES_BY_ORIENTATION[orientation_s]
    return SMARTBLOG_LIVE_PROFILES_BY_ORIENTATION[orientation_s]


def smartblog_normalize_orientation(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("_", "-").replace(" ", "-")
    if raw in _LANDSCAPE_WORDS:
        return "landscape"
    if raw in _PORTRAIT_WORDS:
        return "portrait"
    if raw in SMARTBLOG_LIVE_PROFILES_BY_SIZE:
        return SMARTBLOG_LIVE_PROFILES_BY_SIZE[raw].orientation

    normalized = raw.replace("*", "x").replace(":", "x").replace("/", "x")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) == 2:
        try:
            a = float(parts[0])
            b = float(parts[1])
        except Exception:
            return ""
        if a <= 0.0 or b <= 0.0:
            return ""
        if abs(a - b) < 0.001:
            return ""
        return "landscape" if a > b else "portrait"
    return ""


def _smartblog_size_pair(value: Any) -> tuple[float, float] | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    normalized = raw.replace("_", "-").replace(" ", "-")
    if normalized in SMARTBLOG_LIVE_PROFILES_BY_SIZE:
        profile = SMARTBLOG_LIVE_PROFILES_BY_SIZE[normalized]
        return float(profile.render_height), float(profile.render_width)
    normalized = normalized.replace("*", "x").replace(":", "x").replace("/", "x")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) != 2:
        return None
    try:
        first = float(parts[0])
        second = float(parts[1])
    except Exception:
        return None
    if first <= 0.0 or second <= 0.0 or abs(first - second) < 0.001:
        return None
    return first, second


def smartblog_normalize_model_size_orientation(value: Any) -> str:
    orientation = smartblog_normalize_orientation(value)
    raw = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if raw in _LANDSCAPE_WORDS or raw in _PORTRAIT_WORDS or raw in SMARTBLOG_LIVE_PROFILES_BY_SIZE:
        return orientation
    pair = _smartblog_size_pair(value)
    if pair is None:
        return orientation
    height, width = pair
    return "landscape" if width > height else "portrait"


def smartblog_normalize_output_size_orientation(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("_", "-").replace(" ", "-")
    if raw in _LANDSCAPE_WORDS:
        return "landscape"
    if raw in _PORTRAIT_WORDS:
        return "portrait"
    pair = _smartblog_size_pair(value)
    if pair is None:
        return ""
    width, height = pair
    return "landscape" if width > height else "portrait"


def _smartblog_size_dict_orientation(value: Any, *, order: str) -> str:
    if not isinstance(value, dict):
        return ""
    try:
        width = float(value.get("width") or value.get("w") or 0)
        height = float(value.get("height") or value.get("h") or 0)
    except Exception:
        return ""
    if width <= 0.0 or height <= 0.0 or abs(width - height) < 0.001:
        return ""
    if str(order or "").strip().lower() == "model_hw":
        return "landscape" if width > height else "portrait"
    return "landscape" if width > height else "portrait"


def _smartblog_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _smartblog_nested_payload(src: dict[str, Any]) -> dict[str, Any]:
    for key in ("payload", "payload_json"):
        payload = src.get(key)
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def smartblog_orientation_from_claim(claim: dict[str, Any] | None, *, default: str = "") -> str:
    src = _smartblog_dict(claim)
    job = _smartblog_dict(src.get("job"))
    payload = _smartblog_nested_payload(src) or _smartblog_nested_payload(job)
    runtime_profile_hints = _smartblog_dict(src.get("_runtime_profile_hints"))
    assets = _smartblog_dict(src.get("assets"))
    video = _smartblog_dict(src.get("video"))
    prompt_config = _smartblog_dict(src.get("prompt_config"))
    persona = _smartblog_dict(src.get("persona"))
    persona_video = _smartblog_dict(persona.get("video"))
    persona_prompt_config = _smartblog_dict(persona.get("prompt_config"))
    live_session = _smartblog_dict(src.get("live_session"))
    live_metadata = _smartblog_dict(live_session.get("metadata_json"))
    consultation = _smartblog_dict(src.get("consultation"))
    job_prompt_config = _smartblog_dict(job.get("prompt_config"))
    payload_video = _smartblog_dict(payload.get("video"))
    payload_prompt_config = _smartblog_dict(payload.get("prompt_config"))
    payload_assets = _smartblog_dict(payload.get("assets"))
    payload_persona = _smartblog_dict(payload.get("persona"))
    payload_persona_video = _smartblog_dict(payload_persona.get("video"))
    payload_persona_prompt_config = _smartblog_dict(payload_persona.get("prompt_config"))
    payload_live = _smartblog_dict(payload.get("live_session"))
    payload_live_metadata = _smartblog_dict(payload_live.get("metadata_json"))

    direct_keys = (
        "orientation",
        "video_orientation",
        "videoOrientation",
        "output_orientation",
        "outputOrientation",
        "render_orientation",
        "renderOrientation",
        "target_orientation",
        "targetOrientation",
        "aspect",
        "aspect_ratio",
        "aspectRatio",
        "video_aspect",
        "videoAspect",
        "layout",
        "format",
        "shape",
    )
    model_size_keys = (
        "render_size",
        "renderSize",
        "model_size",
        "modelSize",
    )
    output_size_keys = (
        "video_size",
        "videoSize",
        "output_size",
        "outputSize",
        "resolution",
        "dimensions",
    )
    generic_size_keys = (
        "size",
    )

    sources = (
        runtime_profile_hints,
        video,
        persona,
        persona_video,
        payload_video,
        payload_persona,
        payload_persona_video,
        prompt_config,
        persona_prompt_config,
        payload_prompt_config,
        payload_persona_prompt_config,
        job_prompt_config,
        live_session,
        live_metadata,
        consultation,
        payload_live,
        payload_live_metadata,
        payload,
        job,
        src,
        payload_assets,
        assets,
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in direct_keys:
            orientation = smartblog_normalize_orientation(source.get(key))
            if orientation:
                return orientation

    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in model_size_keys:
            raw_size = source.get(key)
            if isinstance(raw_size, dict):
                orientation = _smartblog_size_dict_orientation(raw_size, order="model_hw")
                if orientation:
                    return orientation
            orientation = smartblog_normalize_model_size_orientation(raw_size)
            if orientation:
                return orientation
        for key in output_size_keys:
            raw_size = source.get(key)
            if isinstance(raw_size, dict):
                orientation = _smartblog_size_dict_orientation(raw_size, order="output_wh")
                if orientation:
                    return orientation
            orientation = smartblog_normalize_output_size_orientation(raw_size)
            if orientation:
                return orientation
        for key in generic_size_keys:
            raw_size = source.get(key)
            if isinstance(raw_size, dict):
                orientation = _smartblog_size_dict_orientation(raw_size, order="output_wh")
                if orientation:
                    return orientation
            orientation = smartblog_normalize_output_size_orientation(raw_size)
            if orientation:
                return orientation

    for source in sources:
        if not isinstance(source, dict):
            continue
        for width_key, height_key in (
            ("output_width", "output_height"),
            ("outputWidth", "outputHeight"),
            ("video_width", "video_height"),
            ("videoWidth", "videoHeight"),
        ):
            try:
                width = float(source.get(width_key) or 0)
                height = float(source.get(height_key) or 0)
            except Exception:
                continue
            if width > 0.0 and height > 0.0 and abs(width - height) >= 0.001:
                return "landscape" if width > height else "portrait"

    for source in (
        video,
        persona_video,
        live_session,
        live_metadata,
        consultation,
        payload_video,
        payload_live,
        payload_live_metadata,
        payload,
        job,
        src,
    ):
        if not isinstance(source, dict):
            continue
        try:
            width = float(source.get("width") or 0)
            height = float(source.get("height") or 0)
        except Exception:
            continue
        if width > 0.0 and height > 0.0 and abs(width - height) >= 0.001:
            return "landscape" if width > height else "portrait"

    fallback = smartblog_normalize_orientation(default)
    return fallback if fallback else str(default or "").strip().lower()
