from __future__ import annotations

from .common import *
import aiohttp
import hashlib
import mimetypes
import re
import cv2
import math
import subprocess
import random
from urllib.parse import urlparse, urlunparse
from avalife.core.smartblog_profiles import (
    SMARTBLOG_RUNTIME_SIZE_CONFIGS,
    smartblog_orientation_from_claim,
    smartblog_live_profile_for_orientation,
)
from avalife.model.protocol import InferRequest, MediaProcessRequest
from avalife.core.audio import (
    auto_num_clip_for_duration,
    to_wav_16k_mono,
    wav_audible_sample_count,
    wav_duration_seconds,
    wav_sample_count,
)
from avalife.core.ffmpeg import probe_video_metadata, video_duration_sec
from avalife.core.path import prepare_live_raw_dir, sanitize_job_id
from avalife.core.upload_retry import put_file_to_signed_url
from avalife.core.watermark import burn_watermark_video, normalize_watermark_text, watermark_text_from_sources
from avalife.worker.live_audio_shm import attach_shared_memory_no_tracker
from avalife.worker.live_raw_shm import live_raw_shm_read_header
from .render_subtitles import RenderSubtitleChunk, burn_ass_subtitles, write_render_subtitles_ass
from .smartblog_api import (
    smartblog_api_rejection_reason,
    smartblog_is_transient_api_error,
    smartblog_supabase_service_role_key,
    smartblog_supabase_url,
    smartblog_validate_action_response,
    smartblog_validate_complete_response,
)


SMARTBLOG_JOB_TYPE_RENDER_VIDEO = "render_video"
SMARTBLOG_JOB_TYPE_VIDEO_TEST = "video_test"
SMARTBLOG_JOB_TYPE_TEST_VIDEO = "test_video"
SMARTBLOG_REMOTE_EDGE_UPLOADED_PREFIX = "edge-uploaded://"
SMARTBLOG_RENDER_DEFAULT_SPEAKING_PROMPT = (
    "A realistic person speaking naturally to camera, accurate lip sync, subtle facial expressions, "
    "stable posture, steady framing."
)
SMARTBLOG_RENDER_DEFAULT_IDLE_PROMPT = (
    "The person is silent with lips gently closed, relaxed expression, stable posture, minimal head motion."
)
_SMARTBLOG_NEXTFRAME_RE = re.compile(r"(?i)(?<!\w)@nextframe(?!\w)")


def _smartblog_render_is_local_runtime_failure(error: Any) -> bool:
    text = str(error or "").strip().lower()
    if not text:
        return False
    transient_markers = (
        "http 429",
        "429 too many requests",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "524",
        "cloudflare",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "network is unreachable",
        "temporary failure",
        "signed url failed",
        "supabase",
        "upload",
    )
    if any(marker in text for marker in transient_markers):
        return False
    fatal_markers = (
        "outofmemory",
        "out of memory",
        "cuda error",
        "cudnn",
        "no valid execution plans",
        "unspecified launch failure",
        "device-side assert",
        "model runtime",
        "modeld",
        "liveaudio producer failed",
        "producer failed",
        "remote_edge",
        "ffmpeg file encode failed",
        "openencodesessionex failed",
        "no capable devices found",
        "cannot access local variable",
        "name '",
        "is not defined",
    )
    return any(marker in text for marker in fatal_markers)


def _smartblog_canonical_media_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc and parsed.path:
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        pass
    return text


def _smartblog_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _smartblog_service_url_is_local(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


_SMARTBLOG_MEDIA_WORKER_POOL_CACHE: dict[str, Any] = {
    "path": "",
    "mtime": -1.0,
    "size": -1,
    "data": None,
}


def _smartblog_media_worker_pool_file() -> str:
    configured = str(
        os.getenv("VLOGME_MEDIA_WORKER_POOL_FILE")
        or os.getenv("SMARTBLOG_MEDIA_WORKER_POOL_FILE")
        or os.getenv("SMARTBLOG_HUNYUAN_WORKER_POOL_FILE")
        or ""
    ).strip()
    if configured:
        return configured
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root_dir, "runtime", "media_worker_pool.json")


def _smartblog_load_media_worker_pool() -> dict[str, Any]:
    if not _env_flag("VLOGME_MEDIA_WORKER_POOL_ENABLED", os.getenv("SMARTBLOG_MEDIA_WORKER_POOL_ENABLED", "1")):
        return {}
    path = _smartblog_media_worker_pool_file()
    if not path or not os.path.exists(path):
        return {}
    try:
        st = os.stat(path)
        cache = _SMARTBLOG_MEDIA_WORKER_POOL_CACHE
        if (
            cache.get("path") == str(path)
            and float(cache.get("mtime") or -1.0) == float(st.st_mtime)
            and int(cache.get("size") or -1) == int(st.st_size)
            and isinstance(cache.get("data"), dict)
        ):
            return dict(cache["data"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("pool root is not an object")
        cache["path"] = str(path)
        cache["mtime"] = float(st.st_mtime)
        cache["size"] = int(st.st_size)
        cache["data"] = dict(data)
        return dict(data)
    except Exception as e:
        logging.warning("VlogMe media worker pool ignored: path=%s err=%s", str(path), str(e))
        return {}


def _smartblog_media_pool_candidates(service: str) -> list[str]:
    service_name = str(service or "").strip().lower()
    if not service_name:
        return []
    pool = _smartblog_load_media_worker_pool()
    workers = pool.get("workers") if isinstance(pool, dict) else []
    if not isinstance(workers, list):
        return []
    url_keys_by_service: dict[str, tuple[str, ...]] = {
        "hunyuan": ("hunyuan", "ltx", "hunyuan_url", "ltx_url"),
        "mmaudio": ("mmaudio", "mmaudio_url"),
        "finalizer": ("finalizer", "upscale", "upscale_url", "finalizer_url"),
        "musetalk": ("musetalk", "lipsync", "musetalk_url", "lipsync_url"),
    }
    wanted_keys = url_keys_by_service.get(service_name, (service_name, f"{service_name}_url"))

    candidates: list[tuple[int, float, float, str]] = []
    for worker in workers:
        if not isinstance(worker, dict) or bool(worker.get("disabled")):
            continue
        status = str(worker.get("status") or worker.get("state") or "").strip().lower()
        if status and status not in {"ready", "healthy", "active", "ok"}:
            continue
        urls = worker.get("urls") if isinstance(worker.get("urls"), dict) else {}
        endpoints = worker.get("endpoints") if isinstance(worker.get("endpoints"), dict) else {}
        url = ""
        for key in wanted_keys:
            url = str(urls.get(key) or endpoints.get(key) or "").strip()
            if url:
                break
        if not url:
            continue
        try:
            priority = int(float(worker.get("priority", 100)))
        except Exception:
            priority = 100
        last_health = str(worker.get("last_health_at") or worker.get("updated_at") or "").strip()
        try:
            # ISO strings sort lexically, but this also handles epoch seconds.
            health_score = float(last_health)
        except Exception:
            health_score = float(sum(ord(ch) for ch in last_health[-32:]))
        candidates.append((int(priority), -float(health_score), random.random(), url.rstrip("/")))
    candidates.sort(key=lambda item: (int(item[0]), float(item[2]), float(item[1]), str(item[3])))
    return [str(item[3]) for item in candidates]


def _smartblog_media_pool_service_url(service: str, fallback: str) -> str:
    for url in _smartblog_media_pool_candidates(str(service)):
        if url:
            logging.info(
                "VlogMe media worker pool selected: service=%s url=%s pool=%s",
                str(service),
                str(url),
                _smartblog_media_worker_pool_file(),
            )
            return str(url)
    return str(fallback or "").strip()


def _smartblog_worker_api_url() -> str:
    return str(
        os.getenv("VLOGME_WORKER_API_URL")
        or os.getenv("SMARTBLOG_WORKER_API_URL")
        or "https://vlogme.ai/api/public/v1/worker-api"
    ).strip()


def _smartblog_worker_api_key() -> str:
    return str(
        os.getenv("WORKER_API_KEY")
        or os.getenv("VLOGME_WORKER_API_KEY")
        or os.getenv("POSTPROCESSING_WORKER_TOKEN")
        or ""
    ).strip()


def _smartblog_media_worker_owner_id() -> str:
    return str(
        os.getenv("VLOGME_RENDER_WORKER_ID")
        or os.getenv("SMARTBLOG_MEDIA_LEASE_OWNER_ID")
        or os.getenv("SMARTBLOG_WORKER_ID")
        or os.getenv("RUNPOD_POD_ID")
        or os.getenv("HOSTNAME")
        or "b200-render-worker"
    ).strip()


def _smartblog_media_worker_lease_enabled() -> bool:
    return _env_flag(
        "VLOGME_MEDIA_WORKER_LEASE_ENABLED",
        os.getenv("SMARTBLOG_MEDIA_WORKER_LEASE_ENABLED", "1"),
    )


async def _smartblog_worker_api_json(payload: dict[str, Any], *, timeout_sec: float = 20.0) -> dict[str, Any]:
    api_url = _smartblog_worker_api_url()
    api_key = _smartblog_worker_api_key()
    if not api_url or not api_key:
        raise RuntimeError("VlogMe worker-api URL/key is not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; VlogMeMediaWorker/1.0)",
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_sec))) as session:
        async with session.post(str(api_url), json=dict(payload or {}), headers=headers) as resp:
            text = await resp.text()
            if int(resp.status) >= 400:
                raise RuntimeError(f"worker-api HTTP {int(resp.status)}: {text[-1000:]}")
            try:
                data = json.loads(text or "{}")
            except Exception as e:
                raise RuntimeError(f"worker-api returned invalid JSON: {text[-1000:]}") from e
    return dict(data or {})


async def _smartblog_acquire_media_worker_lease(
    service: str,
    fallback_url: str,
    *,
    log_prefix: str = "",
) -> tuple[str, dict[str, Any]]:
    service_name = str(service or "").strip().lower()
    fallback_selected = _smartblog_media_pool_service_url(service_name, str(fallback_url or ""))
    if not _smartblog_media_worker_lease_enabled():
        return str(fallback_selected or "").strip(), {}
    if not _smartblog_worker_api_key():
        if fallback_selected:
            logging.warning("%s VlogMe media lease skipped: worker-api key missing, using fallback/pool", str(log_prefix))
            return str(fallback_selected or "").strip(), {}
        raise RuntimeError("VlogMe media lease requires WORKER_API_KEY/VLOGME_WORKER_API_KEY")

    wait_timeout = max(0.0, _safe_float_env("VLOGME_MEDIA_WORKER_LEASE_WAIT_TIMEOUT_SEC", _safe_float_env("SMARTBLOG_MEDIA_WORKER_LEASE_WAIT_TIMEOUT_SEC", 1800.0)))
    poll_sec = max(0.5, _safe_float_env("VLOGME_MEDIA_WORKER_LEASE_POLL_SEC", _safe_float_env("SMARTBLOG_MEDIA_WORKER_LEASE_POLL_SEC", 3.0)))
    lease_sec = int(max(30, min(7200, _safe_int_env("VLOGME_MEDIA_WORKER_LEASE_SECONDS", _safe_int_env("SMARTBLOG_MEDIA_WORKER_LEASE_SECONDS", 3600)))))
    role = str(os.getenv("VLOGME_MEDIA_WORKER_ROLE") or os.getenv("SMARTBLOG_MEDIA_WORKER_ROLE") or "rtxpro6000-media").strip()
    lock_key = str(os.getenv("VLOGME_MEDIA_WORKER_LOCK_KEY") or os.getenv("SMARTBLOG_MEDIA_WORKER_LOCK_KEY") or "gpu").strip() or "gpu"
    started = float(time.monotonic())
    warned = False
    while True:
        try:
            data = await _smartblog_worker_api_json(
                {
                    "action": "acquire_media_worker",
                    "service": service_name,
                    "lock_key": lock_key,
                    "role": role,
                    "lease_seconds": int(lease_sec),
                    "owner_id": _smartblog_media_worker_owner_id(),
                    "job_id": str(log_prefix or "")[:200],
                },
                timeout_sec=max(5.0, _safe_float_env("VLOGME_MEDIA_WORKER_LEASE_HTTP_TIMEOUT_SEC", 20.0)),
            )
            if bool(data.get("acquired")):
                url = str(data.get("url") or "").strip()
                lease = data.get("lease") if isinstance(data.get("lease"), dict) else {}
                if not url:
                    raise RuntimeError(f"VlogMe media lease acquired without URL: {data}")
                logging.warning(
                    "%s VlogMe media lease acquired: service=%s worker=%s lock=%s expires=%s url=%s",
                    str(log_prefix),
                    service_name,
                    str((data.get("worker") or {}).get("worker_id") or (lease or {}).get("worker_id") or ""),
                    str((lease or {}).get("lock_key") or lock_key),
                    str((lease or {}).get("expires_at") or ""),
                    str(url),
                )
                return str(url).rstrip("/"), dict(lease or {})
            if not warned:
                logging.warning(
                    "%s waiting for free VlogMe media worker: service=%s reason=%s candidates=%s",
                    str(log_prefix),
                    service_name,
                    str(data.get("reason") or "busy"),
                    str(data.get("candidates") or ""),
                )
                warned = True
        except Exception as e:
            if fallback_selected and "unknown_action:acquire_media_worker" in str(e):
                logging.warning(
                    "%s VlogMe media lease API is not deployed yet, using registry fallback without lease: service=%s url=%s",
                    str(log_prefix),
                    service_name,
                    str(fallback_selected),
                )
                return str(fallback_selected or "").strip(), {}
            if fallback_selected and _env_flag("VLOGME_MEDIA_WORKER_LEASE_ALLOW_FALLBACK", os.getenv("SMARTBLOG_MEDIA_WORKER_LEASE_ALLOW_FALLBACK", "0")):
                logging.warning("%s VlogMe media lease failed open: %s", str(log_prefix), e)
                return str(fallback_selected or "").strip(), {}
            if float(time.monotonic()) - float(started) >= float(wait_timeout):
                raise
            logging.warning("%s VlogMe media lease acquire failed, retrying: %s", str(log_prefix), e)
        if float(time.monotonic()) - float(started) >= float(wait_timeout):
            raise RuntimeError(f"no free VlogMe media worker for service={service_name} after {wait_timeout:.1f}s")
        await asyncio.sleep(float(poll_sec))


async def _smartblog_release_media_worker_lease(lease: dict[str, Any] | None, *, log_prefix: str = "") -> None:
    if not isinstance(lease, dict):
        return
    token = str(lease.get("lease_token") or "").strip()
    if not token:
        return
    try:
        data = await _smartblog_worker_api_json(
            {
                "action": "release_media_worker",
                "lease_token": token,
                "worker_id": str(lease.get("worker_id") or ""),
                "lock_key": str(lease.get("lock_key") or ""),
            },
            timeout_sec=max(5.0, _safe_float_env("VLOGME_MEDIA_WORKER_LEASE_HTTP_TIMEOUT_SEC", 20.0)),
        )
        logging.warning(
            "%s VlogMe media lease released: worker=%s lock=%s released=%s",
            str(log_prefix),
            str(lease.get("worker_id") or ""),
            str(lease.get("lock_key") or ""),
            str(data.get("released")),
        )
    except Exception as e:
        logging.warning("%s VlogMe media lease release failed: %s", str(log_prefix), e)


def _smartblog_guess_content_type(path: str, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(str(path or ""))
    return str(guessed or fallback or "application/octet-stream")

_SMARTBLOG_RENDER_720P_PROFILES: dict[str, tuple[str, int, int]] = {
    # render_size is Wan HEIGHT*WIDTH; output dimensions are width,height.
    # These near-16:9 / 9:16 sizes become 800x1440 / 1440x800 after x2
    # PostVAE, then final-scale cleanly to 720x1280 / 1280x720.
    "portrait": ("720*400", 720, 1280),
    "landscape": ("400*720", 1280, 720),
}

_SMARTBLOG_RENDER_COMPACT_704_PROFILES: dict[str, tuple[str, int, int]] = {
    # B200 single-GPU fallback: use the real 64-aligned Wan canvas directly
    # instead of the nominal 720*400 key, while keeping the same social output.
    "portrait": ("704*384", 720, 1280),
    "landscape": ("384*704", 1280, 720),
}

_SMARTBLOG_RENDER_832P_PROFILES: dict[str, tuple[str, int, int]] = {
    # Next S2V step above 720*400 / 400*720. Keep this profile on the actual
    # native tensor size and final-scale to the social 720p canvas.
    "portrait": ("832*448", 720, 1280),
    "landscape": ("448*832", 1280, 720),
}

_SMARTBLOG_RENDER_NATIVE_720P_PROFILES: dict[str, tuple[str, int, int]] = {
    # Full social canvas generation. render_size is Wan HEIGHT*WIDTH;
    # output dimensions are conventional width,height.
    "portrait": ("1280*720", 720, 1280),
    "landscape": ("720*1280", 1280, 720),
}

_SMARTBLOG_VALID_RENDER_SIZES = set(SMARTBLOG_RUNTIME_SIZE_CONFIGS) | {
    str(profile[0]) for profile in _SMARTBLOG_RENDER_720P_PROFILES.values()
} | {
    str(profile[0]) for profile in _SMARTBLOG_RENDER_COMPACT_704_PROFILES.values()
} | {str(profile[0]) for profile in _SMARTBLOG_RENDER_832P_PROFILES.values()} | {
    str(profile[0]) for profile in _SMARTBLOG_RENDER_NATIVE_720P_PROFILES.values()
}

_SMARTBLOG_RENDER_PROGRESS_STAGES: dict[str, tuple[str, ...]] = {
    SMARTBLOG_JOB_TYPE_RENDER_VIDEO: ("prepare", "tts", "face_detect", "inference", "encode", "upload"),
    SMARTBLOG_JOB_TYPE_VIDEO_TEST: ("prepare", "tts", "face_detect", "inference", "encode", "upload"),
    SMARTBLOG_JOB_TYPE_TEST_VIDEO: ("prepare", "tts", "face_detect", "inference", "encode", "upload"),
}

_SMARTBLOG_RENDER_PROGRESS_WEIGHTS: dict[str, dict[str, int]] = {
    SMARTBLOG_JOB_TYPE_RENDER_VIDEO: {
        "prepare": 5,
        "tts": 10,
        "face_detect": 5,
        "inference": 50,
        "encode": 20,
        "upload": 10,
    },
    SMARTBLOG_JOB_TYPE_VIDEO_TEST: {
        "prepare": 5,
        "tts": 10,
        "face_detect": 5,
        "inference": 50,
        "encode": 20,
        "upload": 10,
    },
    SMARTBLOG_JOB_TYPE_TEST_VIDEO: {
        "prepare": 5,
        "tts": 10,
        "face_detect": 5,
        "inference": 50,
        "encode": 20,
        "upload": 10,
    },
}

_SMARTBLOG_STAGE_LABELS: dict[str, str] = {
    "prepare": "Preparing assets",
    "tts": "Preparing voiced audio",
    "face_detect": "Preparing avatar framing",
    "inference": "Running model",
    "encode": "Encoding output",
    "upload": "Uploading result",
    "streaming": "Streaming live",
    "heartbeat": "Session heartbeat",
}


class SmartBlogJobStoppedByServer(RuntimeError):
    pass


@dataclass
class SmartBlogRenderFinalizePlan:
    job_id: str
    job_type: str
    signed_url: str
    upload_path: str
    file_path: str
    content_type: str
    complete_kwargs: dict[str, Any]
    run_dir: str = ""
    skip_upload: bool = False
    remote_finalizer_source_path: str = ""
    remote_finalizer_source_url: str = ""
    remote_finalizer_source_fps: int = 0
    remote_finalizer_target_width: int = 0
    remote_finalizer_target_height: int = 0
    remote_finalizer_target_fps: int = 0
    remote_finalizer_upscale_enabled: bool = False
    remote_finalizer_background_music_url: str = ""
    remote_finalizer_background_music_gain_db: float = 0.0
    remote_finalizer_background_music_loop: bool = True
    remote_finalizer_background_music_duck_voice_db: float = 0.0
    remote_finalizer_background_music_fade_in_seconds: float = 0.0
    remote_finalizer_background_music_fade_out_seconds: float = 0.0
    remote_finalizer_background_music_start_offset_seconds: float = 0.0
    remote_finalizer_subtitle_chunks_json: str = ""
    remote_finalizer_watermark_text: str = ""


def smartblog_render_job_types() -> tuple[str, ...]:
    return (
        SMARTBLOG_JOB_TYPE_RENDER_VIDEO,
        SMARTBLOG_JOB_TYPE_VIDEO_TEST,
        SMARTBLOG_JOB_TYPE_TEST_VIDEO,
    )


def smartblog_render_finalize_background_enabled() -> bool:
    return _env_flag("SMARTBLOG_RENDER_FINALIZE_BACKGROUND", "0")


def _smartblog_file_upscale_fallback_service_url() -> str:
    return str(
        os.getenv("REMOTE_EDGE_FILE_UPSCALE_SERVICE_URL", os.getenv("SMARTBLOG_FILE_UPSCALE_SERVICE_URL", ""))
        or ""
    ).strip()


def _smartblog_file_upscale_service_url() -> str:
    fallback = _smartblog_file_upscale_fallback_service_url()
    return _smartblog_media_pool_service_url("finalizer", fallback)


def _smartblog_file_musetalk_service_url() -> str:
    fallback = str(
        os.getenv(
            "SMARTBLOG_MUSETALK_SERVICE_URL",
            os.getenv("REMOTE_EDGE_FILE_MUSETALK_SERVICE_URL", os.getenv("SMARTBLOG_FILE_MUSETALK_SERVICE_URL", "")),
        )
        or ""
    ).strip()
    return _smartblog_media_pool_service_url("musetalk", fallback)


def _smartblog_render_avatar_musetalk_enabled(claim: dict[str, Any] | None = None) -> bool:
    payload = _smartblog_job_payload(claim if isinstance(claim, dict) else {})
    video = _smartblog_video_config(claim if isinstance(claim, dict) else {})
    sources = [src for src in (video, payload, claim) if isinstance(src, dict)]
    for src in sources:
        for key in (
            "musetalk",
            "musetalk_enabled",
            "muse_talk",
            "museTalk",
            "external_lipsync",
            "externalLipSync",
        ):
            if key not in src:
                continue
            parsed = _smartblog_optional_bool(src.get(key))
            if parsed is not None:
                return bool(parsed)
            nested = src.get(key)
            if isinstance(nested, dict):
                nested_parsed = _smartblog_optional_bool(
                    nested.get("enabled") if nested.get("enabled") is not None else nested.get("active")
                )
                if nested_parsed is not None:
                    return bool(nested_parsed)
    env_value = _smartblog_optional_bool(
        os.getenv("SMARTBLOG_RENDER_AVATAR_MUSETALK_ENABLED")
        if os.getenv("SMARTBLOG_RENDER_AVATAR_MUSETALK_ENABLED") is not None
        else os.getenv("SMARTBLOG_RENDER_MUSETALK_ENABLED")
    )
    return bool(env_value) if env_value is not None else False


def _smartblog_render_edge_finalizer_background_enabled() -> bool:
    return bool(
        _env_flag("SMARTBLOG_RENDER_EDGE_FINALIZER_BACKGROUND", "0")
        and bool(_smartblog_file_upscale_service_url())
    )


def _smartblog_render_tpp_cfg_mode() -> str:
    return str(os.getenv("SMARTBLOG_RENDER_TPP_CFG_MODE", "") or "").strip()


_SMARTBLOG_RENDER_SAMPLE_STEP_KEYS = (
    "sample_steps",
    "sampleSteps",
    "render_sample_steps",
    "renderSampleSteps",
    "denoise_steps",
    "denoiseSteps",
    "num_inference_steps",
    "numInferenceSteps",
)


def _smartblog_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _smartblog_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = str(value).strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
            return False
    if isinstance(value, dict):
        for key in ("enabled", "enable", "burn_in", "burnIn", "subtitles", "captions"):
            if key in value:
                parsed = _smartblog_optional_bool(value.get(key))
                if parsed is not None:
                    return bool(parsed)
    return None


def _smartblog_first_int_from_sources(
    keys: tuple[str, ...],
    *sources: tuple[str, Any],
) -> tuple[int | None, str]:
    for source_name, src in sources:
        if not isinstance(src, dict):
            continue
        for key in keys:
            if key not in src:
                continue
            value = _smartblog_optional_int(src.get(key))
            if value is not None:
                return int(value), f"{str(source_name)}.{key}"
    return None, ""


def _smartblog_render_sample_steps(claim: dict[str, Any] | None = None) -> int:
    forced_value = _smartblog_optional_int(os.getenv("SMARTBLOG_RENDER_FORCE_SAMPLE_STEPS"))
    if forced_value is not None:
        value = int(smartblog_clamp_sample_steps(forced_value))
        logging.warning(
            "SmartBlog render sample steps forced: job=%s sample_steps=%d source=SMARTBLOG_RENDER_FORCE_SAMPLE_STEPS",
            str((claim or {}).get("job_id") or ((claim or {}).get("job") or {}).get("id") or "-"),
            int(value),
        )
        return int(value)

    default_value = int(_safe_int_env("SMARTBLOG_RENDER_SAMPLE_STEPS", int(FORCED_SAMPLE_STEPS)))
    value = int(smartblog_clamp_sample_steps(default_value))
    source = "env"

    if isinstance(claim, dict):
        payload = _smartblog_job_payload(claim)
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        video = _smartblog_video_config(claim)
        nested_sources: list[tuple[str, Any]] = [("video", video)]
        for source_name, src in (("payload", payload), ("job", job), ("claim", claim)):
            if not isinstance(src, dict):
                continue
            nested_sources.append((source_name, src))
            for key in ("video", "render", "render_options", "generation", "inference", "model", "settings", "options"):
                nested = src.get(key)
                if isinstance(nested, dict):
                    nested_sources.append((f"{source_name}.{key}", nested))
        candidate, candidate_source = _smartblog_first_int_from_sources(
            _SMARTBLOG_RENDER_SAMPLE_STEP_KEYS,
            *nested_sources,
        )
        if candidate is not None:
            value = int(smartblog_clamp_sample_steps(candidate))
            source = candidate_source or "payload"

    max_value = int(
        min(
            int(SMARTBLOG_MAX_SAMPLE_STEPS),
            int(_safe_int_env("SMARTBLOG_RENDER_MAX_SAMPLE_STEPS", int(SMARTBLOG_MAX_SAMPLE_STEPS))),
        )
    )
    if max_value > 0:
        value = int(min(value, max_value))
    value = int(smartblog_clamp_sample_steps(value))
    if source != "env":
        logging.warning(
            "SmartBlog render sample steps override: job=%s sample_steps=%d source=%s",
            str((claim or {}).get("job_id") or ((claim or {}).get("job") or {}).get("id") or "-"),
            int(value),
            str(source),
        )
    return int(value)


def smartblog_is_render_job(claim_or_job: dict[str, Any] | None) -> bool:
    from .smartblog_claims import smartblog_job_type_value

    return str(smartblog_job_type_value(claim_or_job)).strip().lower() in set(smartblog_render_job_types())


def _smartblog_job_payload(claim: dict[str, Any]) -> dict[str, Any]:
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    for cand in (
        claim.get("payload"),
        claim.get("payload_json"),
        job.get("payload"),
        job.get("payload_json"),
    ):
        if isinstance(cand, dict):
            return dict(cand)
    return {}


def _smartblog_filters(claim: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    def merge_from(src: Any) -> None:
        if not isinstance(src, dict):
            return
        for key in ("face_restore", "background_restore"):
            if key in src:
                merged[key] = src.get(key)
        filters = src.get("filters")
        if isinstance(filters, dict):
            merged.update(filters)

    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    persona_payload = payload.get("persona") if isinstance(payload.get("persona"), dict) else {}
    live_session = claim.get("live_session") if isinstance(claim.get("live_session"), dict) else {}
    consultation = claim.get("consultation") if isinstance(claim.get("consultation"), dict) else {}

    for src in (persona_payload, persona):
        avatar_config = src.get("avatar_config") if isinstance(src, dict) else None
        if isinstance(avatar_config, dict):
            merge_from(avatar_config)

    for src in (persona_payload, persona, live_session, consultation, job, payload, claim):
        if isinstance(src, dict):
            merge_from(src.get("prompt_config"))
        merge_from(src)

    return merged


def _smartblog_watermark_text(claim: dict[str, Any] | None) -> str:
    src = claim if isinstance(claim, dict) else {}
    payload = _smartblog_job_payload(src)
    job = src.get("job") if isinstance(src.get("job"), dict) else {}
    live_session = src.get("live_session") if isinstance(src.get("live_session"), dict) else {}
    live_metadata = live_session.get("metadata_json") if isinstance(live_session.get("metadata_json"), dict) else {}
    content = src.get("content_item") if isinstance(src.get("content_item"), dict) else {}
    content_metadata = content.get("metadata_json") if isinstance(content.get("metadata_json"), dict) else {}
    return watermark_text_from_sources(src, payload, job, live_metadata, live_session, content_metadata, content)


def _smartblog_render_burn_in_subtitles_enabled(claim: dict[str, Any] | None) -> bool:
    if not bool(_env_flag("SMARTBLOG_RENDER_BURN_IN_SUBTITLES", "1")):
        return False
    src = claim if isinstance(claim, dict) else {}
    payload = _smartblog_job_payload(src)
    job = src.get("job") if isinstance(src.get("job"), dict) else {}
    content = src.get("content_item") if isinstance(src.get("content_item"), dict) else {}
    content_metadata = content.get("metadata_json") if isinstance(content.get("metadata_json"), dict) else {}
    assets = src.get("assets") if isinstance(src.get("assets"), dict) else {}
    for source in (payload, content_metadata, content, assets, job, src):
        if not isinstance(source, dict):
            continue
        for key in (
            "subtitles",
            "subtitles_enabled",
            "subtitlesEnabled",
            "burn_in_subtitles",
            "burnInSubtitles",
            "captions",
        ):
            if key not in source:
                continue
            parsed = _smartblog_optional_bool(source.get(key))
            if parsed is not None:
                return bool(parsed)
    return True


def _smartblog_http_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(str(value or "").strip())
    except Exception:
        return False
    return str(parsed.scheme or "").lower() in {"http", "https"} and bool(parsed.netloc)


def _smartblog_generated_assets_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlparse(raw)
        parsed_path = urllib.parse.unquote(str(parsed.path or ""))
    except Exception:
        parsed_path = ""
    for marker in (
        "/storage/v1/object/public/generated-assets/",
        "/storage/v1/object/sign/generated-assets/",
        "/storage/v1/object/authenticated/generated-assets/",
        "/storage/v1/render/image/authenticated/generated-assets/",
    ):
        pos = parsed_path.find(marker)
        if pos >= 0:
            return parsed_path[pos + len(marker) :].lstrip("/")
    if _smartblog_http_url(raw):
        return ""
    path = raw.lstrip("/")
    if path.startswith("generated-assets/"):
        path = path[len("generated-assets/") :]
    return path


def _smartblog_upload_url_folder(value: str) -> str:
    folder = str(value or "").replace("\\", "/").strip("/")
    for bucket in ("builder-renders", "generated-assets"):
        if folder == bucket:
            return ""
        if folder.startswith(f"{bucket}/"):
            folder = folder[len(bucket) + 1 :].strip("/")
            break
    while folder == "worker-uploads" or folder.startswith("worker-uploads/"):
        if folder == "worker-uploads":
            return ""
        folder = folder[len("worker-uploads/") :].strip("/")
    return folder


def _smartblog_float_filter(filters: dict[str, Any], key: str, default: float) -> float:
    raw = filters.get(key)
    try:
        out = float(raw)
    except Exception:
        out = float(default)
    return float(max(0.0, min(1.0, out)))


def _smartblog_render_background_restore_filter(filters: dict[str, Any], *, job_id: Any) -> float:
    default = float(_safe_float_env("SMARTBLOG_RENDER_DEFAULT_BACKGROUND_RESTORE", 0.0))
    requested = _smartblog_float_filter(filters, "background_restore", default)
    if bool(_env_flag("SMARTBLOG_RENDER_BACKGROUND_RESTORE_ENABLED", "0")):
        return float(requested)
    if float(requested) > 0.0:
        logging.warning(
            "SmartBlog render background restore ignored: job=%s requested=%.2f enabled=0",
            str(job_id),
            float(requested),
        )
    return 0.0


def _smartblog_first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _smartblog_first_present(src: dict[str, Any], *keys: str) -> tuple[bool, Any]:
    if not isinstance(src, dict):
        return False, None
    for key in keys:
        if key in src:
            return True, src.get(key)
    return False, None


def _smartblog_same_asset_url(a: Any, b: Any) -> bool:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True
    try:
        left_parsed = urllib.parse.urlparse(left)
        right_parsed = urllib.parse.urlparse(right)
        return bool(left_parsed.path and left_parsed.path == right_parsed.path)
    except Exception:
        return False


def _smartblog_transition_code(*sources: dict[str, Any] | None) -> int:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in ("transition", "transition_code", "transitionCode"):
            if key not in src:
                continue
            try:
                value = int(src.get(key) or 0)
            except Exception:
                value = 0
            if value <= 0:
                return 0
            return int(max(1, min(100, value)))
    return 0


def _smartblog_render_default_speaking_prompt() -> str:
    return str(os.getenv("SMARTBLOG_RENDER_DEFAULT_SPEAKING_PROMPT", SMARTBLOG_RENDER_DEFAULT_SPEAKING_PROMPT) or "").strip()


def _smartblog_render_default_idle_prompt() -> str:
    return str(os.getenv("SMARTBLOG_RENDER_DEFAULT_IDLE_PROMPT", SMARTBLOG_RENDER_DEFAULT_IDLE_PROMPT) or "").strip()


def _smartblog_render_default_hunyuan_prompt() -> str:
    return str(
        os.getenv(
            "SMARTBLOG_RENDER_DEFAULT_HUNYUAN_PROMPT",
            "A stable cinematic shot with natural motion.",
        )
        or ""
    ).strip()


def _smartblog_render_default_mmaudio_prompt() -> str:
    return str(
        os.getenv(
            "SMARTBLOG_MMAUDIO_DEFAULT_PROMPT",
            "subtle synchronized environmental sound effects",
        )
        or ""
    ).strip()


def _smartblog_render_asset_url(claim: dict[str, Any], kind: str) -> str:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    persona_payload = payload.get("persona") if isinstance(payload.get("persona"), dict) else {}
    sources = (assets, payload_assets, payload, job, claim, persona, persona_payload)
    if str(kind) == "audio":
        keys = ("audio_url", "audioUrl", "audio_file_url", "audioFileUrl", "sound_url", "speech_url")
    else:
        keys = (
            "avatar_url",
            "avatarUrl",
            "image_url",
            "imageUrl",
            "photo_url",
            "photoUrl",
            "reference_image_url",
            "referenceImageUrl",
        )
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in keys:
            text = str(src.get(key) or "").strip()
            if text:
                return text
    return ""


def _smartblog_render_asset_urls(claim: dict[str, Any], kind: str) -> list[str]:
    if str(kind) == "audio":
        url = _smartblog_render_asset_url(claim, kind)
        return [url] if url else []
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    persona_payload = payload.get("persona") if isinstance(payload.get("persona"), dict) else {}
    sources = (assets, payload_assets, payload, job, claim, persona, persona_payload)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, dict):
                text = str(
                    item.get("url")
                    or item.get("avatar_url")
                    or item.get("avatarUrl")
                    or item.get("image_url")
                    or item.get("imageUrl")
                    or item.get("photo_url")
                    or item.get("photoUrl")
                    or ""
                ).strip()
            else:
                text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)

    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in ("avatar_urls", "avatarUrls", "image_urls", "imageUrls", "photo_urls", "photoUrls", "photos", "images"):
            add(src.get(key))
        if str(kind) != "audio":
            for frame in _smartblog_render_frames(src):
                for frame_key in (
                    "avatar_url",
                    "avatarUrl",
                    "image_url",
                    "imageUrl",
                    "photo_url",
                    "photoUrl",
                    "reference_image_url",
                    "referenceImageUrl",
                ):
                    frame_url = str((frame or {}).get(frame_key) or "").strip()
                    if frame_url:
                        add(frame_url)
                        break
    primary = _smartblog_render_asset_url(claim, kind)
    if primary:
        if primary in seen:
            ordered = [primary] + [url for url in ordered if url != primary]
        else:
            ordered.insert(0, primary)
    return ordered


def _smartblog_render_frames(src: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(src, dict):
        return []
    for key in ("frames", "video_frames", "videoFrames", "frame_assets", "frameAssets"):
        raw = src.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        frames = [dict(item) for item in raw if isinstance(item, dict)]
        if not frames:
            return []

        def sort_key(item: dict[str, Any]) -> int:
            for idx_key in ("index", "frame_index", "frameIndex", "order", "position"):
                if idx_key not in item:
                    continue
                try:
                    return int(item.get(idx_key))
                except Exception:
                    pass
            return int(frames.index(item))

        return sorted(frames, key=sort_key)
    return []


def _smartblog_render_claim_frames(claim: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    for src in (assets, payload_assets, payload, job, claim):
        if not isinstance(src, dict):
            continue
        frames = _smartblog_render_frames(src)
        if frames:
            return frames
    return []


def _smartblog_frame_kind(frame: dict[str, Any]) -> str:
    text = str((frame or {}).get("kind") or (frame or {}).get("type") or "avatar").strip().lower()
    if text in {"ltx", "ltxv", "ltx-video", "ltx_video", "text_to_video", "text-to-video"}:
        return "ltx"
    return "avatar"


def _smartblog_frame_ltx_config(frame: dict[str, Any]) -> dict[str, Any]:
    cfg = dict((frame or {}).get("ltx") or {}) if isinstance((frame or {}).get("ltx"), dict) else {}
    for key in (
        "prompt",
        "negative_prompt",
        "negativePrompt",
        "duration_seconds",
        "durationSeconds",
        "duration_sec",
        "durationSec",
        "seed",
        "width",
        "height",
        "num_frames",
        "numFrames",
        "frame_rate",
        "frameRate",
        "fps",
        "conditioning_strength",
        "conditioningStrength",
    ):
        if key in frame and key not in cfg:
            cfg[key] = frame.get(key)
    return cfg


_SMARTBLOG_LTX_DURATION_KEYS = ("duration_seconds", "durationSeconds", "duration_sec", "durationSec", "seconds", "duration")
_SMARTBLOG_LTX_FRAME_COUNT_KEYS = (
    "ltx_num_frames",
    "ltxNumFrames",
    "num_frames",
    "numFrames",
    "frames",
    "frame_count",
    "frameCount",
)


def _smartblog_ltx_duration_value(src: dict[str, Any]) -> float | None:
    if not isinstance(src, dict):
        return None
    for key in _SMARTBLOG_LTX_DURATION_KEYS:
        if key not in src:
            continue
        try:
            value = float(src.get(key))
            if math.isfinite(value) and value > 0.0:
                return float(max(0.1, min(60.0, value)))
        except Exception:
            pass
    return None


def _smartblog_ltx_duration_seconds(cfg: dict[str, Any], *, default: float = 3.0) -> float:
    value = _smartblog_ltx_duration_value(cfg)
    if value is not None:
        return float(value)
    return float(max(0.1, min(60.0, float(default))))


def _smartblog_ltx_has_frame_count(src: dict[str, Any]) -> bool:
    return isinstance(src, dict) and any(key in src for key in _SMARTBLOG_LTX_FRAME_COUNT_KEYS)


def _smartblog_ltx_claim_duration_seconds(claim: dict[str, Any], *, default: float = 0.0) -> float:
    for src in _smartblog_ltx_source_dicts(claim):
        value = _smartblog_ltx_duration_value(src)
        if value is not None:
            return float(value)
    return float(max(0.0, min(60.0, float(default))))


def _smartblog_segment_file_index(info: dict[str, Any], *, fallback: int) -> int:
    for key in ("index", "segment_index", "segmentIndex"):
        try:
            value = int((info or {}).get(key))
            if value >= 0:
                return int(value)
        except Exception:
            pass
    return int(max(0, int(fallback)))


def _smartblog_render_timeline_entries(claim: dict[str, Any]) -> list[dict[str, Any]]:
    frames = _smartblog_render_claim_frames(claim)
    if not frames:
        return []
    timeline: list[dict[str, Any]] = []
    next_index = 0

    def frame_index_for(frame: dict[str, Any], *, fallback: int) -> int:
        for idx_key in ("index", "frame_index", "frameIndex", "order", "position"):
            if idx_key not in frame:
                continue
            try:
                return int(frame.get(idx_key))
            except Exception:
                pass
        return int(fallback)

    def frame_avatar_url(frame: dict[str, Any]) -> str:
        return _smartblog_render_audio_entry_avatar_url(dict(frame or {}))

    def add_frame_overrides(entry: dict[str, Any], frame: dict[str, Any]) -> None:
        frame_video = (frame or {}).get("video") if isinstance((frame or {}).get("video"), dict) else {}
        frame_filters = (frame or {}).get("filters") if isinstance((frame or {}).get("filters"), dict) else {}
        present, value = _smartblog_first_present(
            frame,
            "video_prompt",
            "videoPrompt",
            "prompt",
            "render_prompt",
            "renderPrompt",
        )
        if not present:
            present, value = _smartblog_first_present(
                frame_video,
                "video_prompt",
                "videoPrompt",
                "prompt",
                "render_prompt",
                "renderPrompt",
            )
        if present:
            entry["_smartblog_frame_video_prompt"] = value
        present, value = _smartblog_first_present(
            frame,
            "video_negative_prompt",
            "videoNegativePrompt",
            "negative_prompt",
            "negativePrompt",
        )
        if not present:
            present, value = _smartblog_first_present(
                frame_video,
                "video_negative_prompt",
                "videoNegativePrompt",
                "negative_prompt",
                "negativePrompt",
            )
        if present:
            entry["_smartblog_frame_video_negative_prompt"] = value
        present, value = _smartblog_first_present(frame, "face_restore", "faceRestore")
        if not present:
            present, value = _smartblog_first_present(frame_filters, "face_restore", "faceRestore")
        if present:
            entry["_smartblog_frame_face_restore"] = value
        present, value = _smartblog_first_present(frame, "background_restore", "backgroundRestore")
        if not present:
            present, value = _smartblog_first_present(frame_filters, "background_restore", "backgroundRestore")
        if present:
            entry["_smartblog_frame_background_restore"] = value

    def add_ltx_entry(
        *,
        frame: dict[str, Any],
        frame_index: int,
        frame_pos: int,
        cfg: dict[str, Any],
        image_url: str,
        insert_after_chunk: int | None = None,
        mode: str = "cut",
        t_offset_seconds: float = 0.0,
        use_previous_last_frame: bool = False,
        audio: dict[str, Any] | None = None,
        transition_code: int = 0,
        direct_clip: bool = False,
        direct_clip_video_url: str = "",
        append: bool = True,
    ) -> dict[str, Any]:
        nonlocal next_index
        prompt = _smartblog_first_text(cfg.get("prompt"), cfg.get("ltx_prompt"), cfg.get("ltxPrompt"))
        if not prompt:
            prompt = _smartblog_first_text(
                (frame or {}).get("video_prompt"),
                (frame or {}).get("videoPrompt"),
                (frame or {}).get("prompt"),
            )
        entry: dict[str, Any] = {
            "index": int(next_index),
            "_smartblog_timeline_kind": "ltx",
            "_smartblog_timeline_order": int(next_index),
            "_smartblog_frame_index": int(frame_index),
            "_smartblog_frame_pos": int(frame_pos),
            "_smartblog_ltx_config": dict(cfg or {}),
            "_smartblog_ltx_prompt": str(prompt or ""),
            "_smartblog_ltx_negative_prompt": _smartblog_first_text(
                cfg.get("negative_prompt"),
                cfg.get("negativePrompt"),
                (frame or {}).get("video_negative_prompt"),
                (frame or {}).get("videoNegativePrompt"),
            ),
            "_smartblog_ltx_duration_sec": float(_smartblog_ltx_duration_seconds(dict(cfg or {}))),
            "_smartblog_ltx_image_url": str(image_url or frame_avatar_url(frame) or "").strip(),
            "_smartblog_ltx_insert_after_chunk": insert_after_chunk,
            "_smartblog_ltx_mode": str(mode or "cut").strip().lower(),
            "_smartblog_ltx_t_offset_seconds": float(max(0.0, float(t_offset_seconds or 0.0))),
            "_smartblog_ltx_use_previous_last_frame": bool(use_previous_last_frame),
            "_smartblog_transition_code": int(max(0, min(100, int(transition_code or 0)))),
        }
        if isinstance(audio, dict) and audio:
            entry["_smartblog_ltx_audio"] = dict(audio)
        direct_clip_url = str(direct_clip_video_url or "").strip()
        if bool(direct_clip) or direct_clip_url:
            entry["_smartblog_direct_clip"] = bool(direct_clip)
            entry["_smartblog_direct_clip_video_url"] = str(direct_clip_url)
        if "seed" in cfg:
            entry["_smartblog_ltx_seed"] = cfg.get("seed")
        if "conditioning_strength" in cfg or "conditioningStrength" in cfg:
            entry["_smartblog_ltx_conditioning_strength"] = cfg.get(
                "conditioning_strength",
                cfg.get("conditioningStrength"),
            )
        if bool(append):
            timeline.append(entry)
            next_index += 1
        return entry

    def add_audio_entry(
        item: Any,
        *,
        frame: dict[str, Any],
        frame_index: int,
        frame_pos: int,
        frame_audio_index: int,
        frame_audio_total: int,
        preserve_boundary: bool,
        frame_inserts: list[dict[str, Any]] | None = None,
        transition_code: int = 0,
    ) -> None:
        nonlocal next_index
        if isinstance(item, str):
            url = str(item or "").strip()
            entry: dict[str, Any] = {"url": url}
        elif isinstance(item, dict):
            entry = dict(item)
            url = ""
            for key in (
                "url",
                "audio_url",
                "audioUrl",
                "audio_file_url",
                "audioFileUrl",
                "signed_url",
                "signedUrl",
                "sound_url",
                "speech_url",
                "src",
            ):
                url = str(entry.get(key) or "").strip()
                if url:
                    break
            entry["url"] = url
        else:
            return
        if not str(entry.get("url") or "").strip():
            return
        if "source_audio_index" not in entry and "index" in entry:
            entry["source_audio_index"] = entry.get("index")
        entry["index"] = int(next_index)
        entry["_smartblog_audio_chunk"] = True
        entry["_smartblog_frame_api"] = True
        entry["_smartblog_timeline_kind"] = "avatar"
        entry["_smartblog_timeline_order"] = int(next_index)
        entry["_smartblog_frame_index"] = int(frame_index)
        entry["_smartblog_frame_pos"] = int(frame_pos)
        entry["_smartblog_frame_audio_index"] = int(frame_audio_index)
        entry["_smartblog_frame_audio_total"] = int(frame_audio_total)
        if bool(preserve_boundary):
            entry["_smartblog_timeline_no_merge"] = True
        if frame_inserts:
            entry["_smartblog_frame_inserts"] = [dict(item or {}) for item in list(frame_inserts or []) if isinstance(item, dict)]
        entry.setdefault("frame_index", int(frame_index))
        entry.setdefault("frameIndex", int(frame_index))
        avatar_url = frame_avatar_url(frame)
        if avatar_url and not _smartblog_render_audio_entry_avatar_url(entry):
            entry["avatar_url"] = str(avatar_url)
        if _smartblog_render_audio_entry_avatar_index(entry) is None:
            entry["avatar_index"] = int(frame_pos)
        if not _smartblog_audio_entry_text(entry):
            frame_text = str(
                (frame or {}).get("text")
                or (frame or {}).get("script_text")
                or (frame or {}).get("scriptText")
                or ""
            ).strip()
            if frame_text:
                entry["text"] = frame_text
        for align_key in ("alignment", "normalized_alignment", "normalizedAlignment"):
            if align_key not in entry and isinstance((frame or {}).get(align_key), dict):
                entry[align_key] = dict((frame or {}).get(align_key) or {})
        add_frame_overrides(entry, dict(frame or {}))
        entry["_smartblog_transition_code"] = int(max(0, min(100, int(transition_code or 0))))
        timeline.append(entry)
        next_index += 1

    for frame_pos, frame in enumerate(frames):
        frame = dict(frame or {})
        frame_index = frame_index_for(frame, fallback=int(frame_pos))
        frame_transition_code = int(_smartblog_transition_code(frame))
        frame_transition_consumed = False

        def consume_frame_transition(extra_code: int = 0) -> int:
            nonlocal frame_transition_consumed
            code = int(max(0, min(100, int(extra_code or 0))))
            if not bool(frame_transition_consumed):
                code = max(code, int(frame_transition_code))
                frame_transition_consumed = True
            return int(code)

        kind = _smartblog_frame_kind(frame)
        if kind == "ltx":
            frame_cfg = _smartblog_frame_ltx_config(frame)
            use_prev = _smartblog_optional_bool(
                frame.get("use_previous_last_frame", frame.get("usePreviousLastFrame", frame_cfg.get("use_previous_last_frame", frame_cfg.get("usePreviousLastFrame"))))
            )
            add_ltx_entry(
                frame=frame,
                frame_index=int(frame_index),
                frame_pos=int(frame_pos),
                cfg=frame_cfg,
                image_url=frame_avatar_url(frame),
                use_previous_last_frame=bool(use_prev),
                transition_code=consume_frame_transition(_smartblog_transition_code(_smartblog_frame_ltx_config(frame))),
            )
            continue

        raw_chunks: list[Any] = []
        for key in ("audio_chunks", "audioChunks", "audio_urls", "audioUrls", "audio_files", "audioFiles", "audio_segments", "audioSegments"):
            if isinstance(frame.get(key), (list, tuple)):
                raw_chunks = list(frame.get(key) or [])
                break
        if not raw_chunks:
            for key in ("audio_url", "audioUrl", "audio_file_url", "audioFileUrl", "signed_url", "signedUrl", "speech_url", "sound_url"):
                value = str(frame.get(key) or "").strip()
                if value:
                    raw_chunks = [dict(frame, url=value)]
                    break

        raw_inserts = frame.get("inserts") if isinstance(frame.get("inserts"), (list, tuple)) else []
        raw_inserts_by_after: dict[int, list[dict[str, Any]]] = {}
        for raw_insert in list(raw_inserts or []):
            if not isinstance(raw_insert, dict):
                continue
            try:
                after = int(raw_insert.get("after_chunk", raw_insert.get("afterChunk", -1)))
            except Exception:
                after = -1
            raw_inserts_by_after.setdefault(int(after), []).append(dict(raw_insert))

        frame_overlay_entries: list[dict[str, Any]] = []
        cut_insert_entries_by_after: dict[int, list[dict[str, Any]]] = {}

        def add_insert(raw_insert: dict[str, Any], *, after: int) -> None:
            cfg = dict(raw_insert.get("ltx") or {}) if isinstance(raw_insert.get("ltx"), dict) else {}
            for key in (
                "prompt",
                "negative_prompt",
                "negativePrompt",
                "duration_seconds",
                "durationSeconds",
                "duration_sec",
                "durationSec",
                "seed",
                "width",
                "height",
                "num_frames",
                "numFrames",
                "frame_rate",
                "frameRate",
                "fps",
                "conditioning_strength",
                "conditioningStrength",
            ):
                if key in raw_insert and key not in cfg:
                    cfg[key] = raw_insert.get(key)
            explicit_image_url = str(
                raw_insert.get("image_url")
                or raw_insert.get("imageUrl")
                or raw_insert.get("avatar_url")
                or raw_insert.get("avatarUrl")
                or ""
            ).strip()
            frame_image_url = str(frame_avatar_url(frame) or "").strip()
            image_url = str(explicit_image_url or frame_image_url or "").strip()
            mode = str(raw_insert.get("mode") or raw_insert.get("insert_mode") or raw_insert.get("insertMode") or "cut").strip().lower()
            if mode not in {"cut", "overlay"}:
                mode = "cut"
            try:
                t_offset_seconds = float(raw_insert.get("t_offset_seconds", raw_insert.get("tOffsetSeconds", 0.0)) or 0.0)
            except Exception:
                t_offset_seconds = 0.0
            use_prev_opt = _smartblog_optional_bool(
                raw_insert.get(
                    "use_previous_last_frame",
                    raw_insert.get("usePreviousLastFrame", cfg.get("use_previous_last_frame", cfg.get("usePreviousLastFrame"))),
                )
            )
            explicit_separate_image = bool(explicit_image_url) and not _smartblog_same_asset_url(
                explicit_image_url,
                frame_image_url,
            )
            use_prev = bool(use_prev_opt) if use_prev_opt is not None else not bool(explicit_separate_image)
            audio_block = raw_insert.get("audio") if isinstance(raw_insert.get("audio"), dict) else {}
            direct_clip_opt = _smartblog_optional_bool(
                raw_insert.get(
                    "direct_clip",
                    raw_insert.get("directClip", raw_insert.get("direct_clip_passthrough", raw_insert.get("directClipPassthrough"))),
                )
            )
            direct_clip_video_url = _smartblog_first_text(
                raw_insert.get("video_url"),
                raw_insert.get("videoUrl"),
                raw_insert.get("source_video_url"),
                raw_insert.get("sourceVideoUrl"),
                raw_insert.get("asset_video_url"),
                raw_insert.get("assetVideoUrl"),
                raw_insert.get("asset_url"),
                raw_insert.get("assetUrl"),
                raw_insert.get("media_url"),
                raw_insert.get("mediaUrl"),
                raw_insert.get("src"),
                raw_insert.get("url"),
            )
            direct_clip = bool(direct_clip_opt) if direct_clip_opt is not None else False
            insert_entry = add_ltx_entry(
                frame=frame,
                frame_index=int(frame_index),
                frame_pos=int(frame_pos),
                cfg=cfg,
                image_url=str(image_url),
                insert_after_chunk=int(after),
                mode=str(mode),
                t_offset_seconds=float(max(0.0, float(t_offset_seconds))),
                use_previous_last_frame=bool(use_prev),
                audio=dict(audio_block or {}),
                transition_code=_smartblog_transition_code(raw_insert, cfg),
                direct_clip=bool(direct_clip),
                direct_clip_video_url=str(direct_clip_video_url or ""),
                append=False,
            )
            insert_entry["_smartblog_insert_order"] = int(
                sum(len(items) for items in cut_insert_entries_by_after.values()) + len(frame_overlay_entries)
            )
            if str(mode) == "overlay":
                frame_overlay_entries.append(dict(insert_entry))
            else:
                cut_insert_entries_by_after.setdefault(int(after), []).append(dict(insert_entry))

        for after in sorted(raw_inserts_by_after):
            for raw_insert in raw_inserts_by_after.get(int(after), []):
                add_insert(raw_insert, after=int(after))
        total = int(len(raw_chunks))
        preserve = False

        def append_cut_insert_entries(after: int, *, as_first_frame_content: bool = False) -> None:
            nonlocal next_index
            entries = cut_insert_entries_by_after.pop(int(after), [])
            for insert_entry in list(entries or []):
                item = dict(insert_entry or {})
                if bool(as_first_frame_content) and not bool(frame_transition_consumed):
                    item["_smartblog_transition_code"] = int(
                        max(
                            int(item.get("_smartblog_transition_code") or 0),
                            int(consume_frame_transition()),
                        )
                    )
                item["index"] = int(next_index)
                item["_smartblog_timeline_order"] = int(next_index)
                timeline.append(item)
                next_index += 1

        append_cut_insert_entries(-1, as_first_frame_content=True)
        for chunk_pos, chunk in enumerate(raw_chunks):
            add_audio_entry(
                chunk,
                frame=frame,
                frame_index=int(frame_index),
                frame_pos=int(frame_pos),
                frame_audio_index=int(chunk_pos),
                frame_audio_total=int(total),
                preserve_boundary=bool(preserve),
                frame_inserts=list(frame_overlay_entries),
                transition_code=consume_frame_transition() if not bool(frame_transition_consumed) else 0,
            )
            chunk_ids: set[int] = {int(chunk_pos)}
            if isinstance(chunk, dict):
                for key in ("index", "source_audio_index", "sourceAudioIndex"):
                    try:
                        chunk_ids.add(int(chunk.get(key)))
                    except Exception:
                        pass
            for after in sorted(list(cut_insert_entries_by_after)):
                if int(after) in chunk_ids:
                    append_cut_insert_entries(int(after))
        for after in sorted(list(cut_insert_entries_by_after)):
            append_cut_insert_entries(int(after))

    for idx, entry in enumerate(timeline):
        entry["index"] = int(idx)
        entry["_smartblog_timeline_order"] = int(idx)
    return timeline


def _smartblog_render_audio_entries(claim: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    sources = (assets, payload_assets, payload, job, claim)
    list_keys = (
        "audio_chunks",
        "audioChunks",
        "audio_urls",
        "audioUrls",
        "audio_files",
        "audioFiles",
        "audio_segments",
        "audioSegments",
        "segments",
        "clips",
    )
    url_keys = (
        "url",
        "audio_url",
        "audioUrl",
        "audio_file_url",
        "audioFileUrl",
        "signed_url",
        "signedUrl",
        "sound_url",
        "speech_url",
        "src",
    )
    out: list[dict[str, Any]] = []

    def frame_avatar_url(frame: dict[str, Any]) -> str:
        return _smartblog_render_audio_entry_avatar_url(dict(frame or {}))

    def frame_avatar_index(frame: dict[str, Any], *, fallback: int) -> int:
        idx = _smartblog_render_audio_entry_avatar_index(dict(frame or {}))
        if idx is None:
            return int(fallback)
        return int(idx)

    def add_item(item: Any, *, index: int) -> None:
        if isinstance(item, str):
            url = str(item or "").strip()
            entry: dict[str, Any] = {"url": url}
        elif isinstance(item, dict):
            url = ""
            for key in url_keys:
                url = str(item.get(key) or "").strip()
                if url:
                    break
            entry = dict(item)
            entry["url"] = url
        else:
            return
        url_s = str(entry.get("url") or "").strip()
        if not url_s:
            return
        entry.setdefault("index", int(index))
        entry["_smartblog_audio_chunk"] = True
        out.append(entry)

    def add_frame_audio_item(
        item: Any,
        *,
        index: int,
        frame_index: int,
        frame_pos: int,
        frame_audio_index: int,
        frame_audio_total: int,
        frame: dict[str, Any],
    ) -> None:
        before = int(len(out))
        add_item(item, index=int(index))
        if int(len(out)) <= before:
            return
        entry = out[-1]
        if "index" in entry:
            entry.setdefault("source_audio_index", entry.get("index"))
        entry["index"] = int(index)
        entry["_smartblog_frame_api"] = True
        entry["_smartblog_frame_index"] = int(frame_index)
        entry["_smartblog_frame_audio_index"] = int(frame_audio_index)
        entry["_smartblog_frame_audio_total"] = int(frame_audio_total)
        entry.setdefault("frame_index", int(frame_index))
        entry.setdefault("frameIndex", int(frame_index))
        avatar_url = frame_avatar_url(dict(frame or {}))
        if avatar_url and not _smartblog_render_audio_entry_avatar_url(entry):
            entry["avatar_url"] = str(avatar_url)
        if _smartblog_render_audio_entry_avatar_index(entry) is None:
            entry["avatar_index"] = int(frame_avatar_index(dict(frame or {}), fallback=int(frame_pos)))
        if not _smartblog_audio_entry_text(entry):
            frame_text = str(
                (frame or {}).get("text")
                or (frame or {}).get("script_text")
                or (frame or {}).get("scriptText")
                or ""
            ).strip()
            if frame_text:
                entry["text"] = frame_text
        for align_key in ("alignment", "normalized_alignment", "normalizedAlignment"):
            if align_key not in entry and isinstance((frame or {}).get(align_key), dict):
                entry[align_key] = dict((frame or {}).get(align_key) or {})
        frame_video = (frame or {}).get("video") if isinstance((frame or {}).get("video"), dict) else {}
        frame_filters = (frame or {}).get("filters") if isinstance((frame or {}).get("filters"), dict) else {}
        present, value = _smartblog_first_present(
            frame,
            "video_prompt",
            "videoPrompt",
            "prompt",
            "render_prompt",
            "renderPrompt",
        )
        if not present:
            present, value = _smartblog_first_present(
                frame_video,
                "video_prompt",
                "videoPrompt",
                "prompt",
                "render_prompt",
                "renderPrompt",
            )
        if present:
            entry["_smartblog_frame_video_prompt"] = value
        present, value = _smartblog_first_present(
            frame,
            "video_negative_prompt",
            "videoNegativePrompt",
            "negative_prompt",
            "negativePrompt",
        )
        if not present:
            present, value = _smartblog_first_present(
                frame_video,
                "video_negative_prompt",
                "videoNegativePrompt",
                "negative_prompt",
                "negativePrompt",
            )
        if present:
            entry["_smartblog_frame_video_negative_prompt"] = value
        present, value = _smartblog_first_present(frame, "face_restore", "faceRestore")
        if not present:
            present, value = _smartblog_first_present(frame_filters, "face_restore", "faceRestore")
        if present:
            entry["_smartblog_frame_face_restore"] = value
        present, value = _smartblog_first_present(frame, "background_restore", "backgroundRestore")
        if not present:
            present, value = _smartblog_first_present(frame_filters, "background_restore", "backgroundRestore")
        if present:
            entry["_smartblog_frame_background_restore"] = value

    frame_entries: list[dict[str, Any]] = []
    for src in (assets, payload_assets, payload, job, claim):
        if not isinstance(src, dict):
            continue
        frames = _smartblog_render_frames(src)
        if not frames:
            continue
        next_index = 0
        for frame_pos, frame in enumerate(frames):
            frame_index = int(frame_pos)
            for idx_key in ("index", "frame_index", "frameIndex", "order", "position"):
                if idx_key not in frame:
                    continue
                try:
                    frame_index = int(frame.get(idx_key))
                    break
                except Exception:
                    pass
            raw_chunks = None
            for key in ("audio_chunks", "audioChunks", "audio_urls", "audioUrls", "audio_files", "audioFiles", "audio_segments", "audioSegments"):
                if isinstance(frame.get(key), (list, tuple)):
                    raw_chunks = list(frame.get(key) or [])
                    break
            if raw_chunks is None:
                for key in ("audio_url", "audioUrl", "audio_file_url", "audioFileUrl", "signed_url", "signedUrl", "speech_url", "sound_url"):
                    value = str(frame.get(key) or "").strip()
                    if value:
                        raw_chunks = [dict(frame, url=value)]
                        break
            if not raw_chunks:
                continue
            before_len = int(len(out))
            total = int(len(raw_chunks))
            for chunk_pos, chunk in enumerate(raw_chunks):
                add_frame_audio_item(
                    chunk,
                    index=int(next_index),
                    frame_index=int(frame_index),
                    frame_pos=int(frame_pos),
                    frame_audio_index=int(chunk_pos),
                    frame_audio_total=int(total),
                    frame=dict(frame),
                )
                if int(len(out)) > int(next_index):
                    next_index = int(len(out))
            if int(len(out)) > before_len:
                frame_entries.extend(out[before_len:])
        if frame_entries:
            return frame_entries

    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in list_keys:
            raw = src.get(key)
            if not isinstance(raw, (list, tuple)):
                continue
            before_len = int(len(out))
            for idx, item in enumerate(raw):
                add_item(item, index=int(idx))
            if out:
                chunk = out[before_len:]
                if len(chunk) > 1:
                    def sort_index(entry: dict[str, Any]) -> int:
                        try:
                            return int(entry.get("index"))
                        except Exception:
                            return 0

                    out[before_len:] = sorted(chunk, key=sort_index)
                return out

    single_url = _smartblog_render_asset_url(claim, "audio")
    if single_url:
        return [{"url": str(single_url), "index": 0}]
    return []


def _smartblog_render_audio_entry_local_path(entry: dict[str, Any]) -> str:
    for key in ("local_path", "localPath", "path", "file_path", "filePath"):
        text = str((entry or {}).get(key) or "").strip()
        if text and os.path.exists(text):
            return text
    url = str((entry or {}).get("url") or "").strip()
    if url.startswith("file://"):
        path = urllib.parse.unquote(url[len("file://") :])
        if path and os.path.exists(path):
            return path
    if url and "://" not in url and os.path.exists(url):
        return url
    return ""


def _smartblog_render_script_text(claim: dict[str, Any]) -> str:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    content_item = claim.get("content_item") if isinstance(claim.get("content_item"), dict) else {}
    content_metadata = content_item.get("metadata_json") if isinstance(content_item.get("metadata_json"), dict) else {}
    for src in (assets, payload_assets, payload, content_metadata, content_item, job, claim):
        if not isinstance(src, dict):
            continue
        for key in ("script_text", "scriptText", "text", "transcript"):
            text = str(src.get(key) or "").strip()
            if text:
                return text
    return ""


def _smartblog_split_nextframe_script(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [str(part or "").strip() for part in _SMARTBLOG_NEXTFRAME_RE.split(raw)]
    chunks = [re.sub(r"\s+", " ", part).strip() for part in parts if str(part or "").strip()]
    return chunks or [re.sub(r"\s+", " ", raw).strip()]


def _smartblog_render_tts_max_chars() -> int:
    return int(max(500, min(5000, _safe_int_env("SMARTBLOG_RENDER_TTS_MAX_CHARS", 4500))))


def _smartblog_render_tts_target_words() -> int:
    return int(max(30, min(500, _safe_int_env("SMARTBLOG_RENDER_TTS_TARGET_WORDS", 220))))


def _smartblog_render_tts_visual_merge_max_sec() -> float:
    return float(max(0.0, min(180.0, _safe_float_env("SMARTBLOG_RENDER_TTS_VISUAL_MERGE_MAX_SEC", 24.0))))


def _smartblog_render_audio_segment_max_sec() -> float:
    return float(
        max(
            0.0,
            min(
                180.0,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AUDIO_SEGMENT_MAX_SEC",
                    float(_smartblog_render_tts_visual_merge_max_sec() or 24.0),
                ),
            ),
        )
    )


def _smartblog_render_single_avatar_one_pass_enabled() -> bool:
    return bool(_env_flag("SMARTBLOG_RENDER_SINGLE_AVATAR_ONE_PASS", "1"))


def _smartblog_render_avatar_liveaudio_one_pass_enabled() -> bool:
    return bool(_env_flag("SMARTBLOG_RENDER_AVATAR_LIVEAUDIO_ONE_PASS", "1"))


def _smartblog_audio_entries_same_avatar_without_timeline_breaks(entries: list[dict[str, Any]]) -> bool:
    ordered = [dict(entry or {}) for entry in list(entries or []) if isinstance(entry, dict)]
    if not ordered:
        return False
    visual_key: tuple[str, str] | None = None
    for entry in ordered:
        kind = str(entry.get("_smartblog_timeline_kind") or "avatar").strip().lower()
        if kind and kind != "avatar":
            return False
        if bool(entry.get("_smartblog_timeline_no_merge")):
            return False
        key = _smartblog_worker_tts_visual_key(entry)
        if visual_key is None:
            visual_key = key
        elif key != visual_key:
            return False
    return visual_key is not None


def _smartblog_render_tts_concurrency() -> int:
    return int(max(1, min(12, _safe_int_env("SMARTBLOG_RENDER_TTS_CONCURRENCY", 6))))


def _smartblog_render_tts_timeout_sec(*, chars: int, words: int, fallback_sec: float) -> float:
    min_timeout = float(max(10.0, min(600.0, _safe_float_env("SMARTBLOG_RENDER_TTS_MIN_TIMEOUT_SEC", 120.0))))
    max_timeout = float(max(min_timeout, min(900.0, _safe_float_env("SMARTBLOG_RENDER_TTS_MAX_TIMEOUT_SEC", 240.0))))
    base = float(max(1.0, min(300.0, _safe_float_env("SMARTBLOG_RENDER_TTS_TIMEOUT_BASE_SEC", 30.0))))
    per_word = float(max(0.0, min(5.0, _safe_float_env("SMARTBLOG_RENDER_TTS_TIMEOUT_PER_WORD_SEC", 0.7))))
    per_char = float(max(0.0, min(1.0, _safe_float_env("SMARTBLOG_RENDER_TTS_TIMEOUT_PER_CHAR_SEC", 0.0))))
    estimated = float(base + float(max(0, int(words))) * per_word + float(max(0, int(chars))) * per_char)
    return float(min(max_timeout, max(float(fallback_sec or 0.0), min_timeout, estimated)))


def _smartblog_render_tts_word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+(?:['’`-][^\W_]+)?", str(text or ""), flags=re.UNICODE))


def _smartblog_split_text_for_eleven_tts(text: str, *, max_chars: int, target_words: int | None = None) -> list[str]:
    limit = int(max(500, min(5000, int(max_chars or 4500))))
    word_target = int(max(0, int(target_words or 0)))
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return []
    if len(raw) <= limit and (not word_target or _smartblog_render_tts_word_count(raw) <= int(word_target * 1.25)):
        return [raw]

    def split_piece(piece: str) -> list[str]:
        piece_s = re.sub(r"\s+", " ", str(piece or "").strip())
        if not piece_s:
            return []
        piece_words = int(_smartblog_render_tts_word_count(piece_s))
        if len(piece_s) <= limit and (not word_target or piece_words <= int(word_target * 1.35)):
            return [piece_s]
        for pattern in (
            r"(?<=[.!?…])\s+",
            r"(?<=[,;:])\s+",
            r"\s+[—-]\s+",
        ):
            parts = [part.strip() for part in re.split(pattern, piece_s) if part.strip()]
            if len(parts) > 1 and all(len(part) < len(piece_s) for part in parts):
                out: list[str] = []
                for part in parts:
                    out.extend(split_piece(part))
                return out
        words = piece_s.split()
        out: list[str] = []
        cur = ""
        cur_words = 0
        for word in words:
            word_s = str(word or "").strip()
            if not word_s:
                continue
            if len(word_s) > limit:
                if cur:
                    out.append(cur)
                    cur = ""
                    cur_words = 0
                for start in range(0, len(word_s), limit):
                    part = word_s[start : start + limit].strip()
                    if part:
                        out.append(part)
                continue
            cand = f"{cur} {word_s}".strip() if cur else word_s
            cand_words = int(cur_words + _smartblog_render_tts_word_count(word_s))
            if len(cand) <= limit and (not word_target or cand_words <= word_target or not cur):
                cur = cand
                cur_words = cand_words
            else:
                if cur:
                    out.append(cur)
                cur = word_s
                cur_words = int(_smartblog_render_tts_word_count(word_s))
        if cur:
            out.append(cur)
        return out

    units: list[str] = []
    for paragraph in re.split(r"\n+", str(text or "")):
        para = re.sub(r"\s+", " ", paragraph.strip())
        if not para:
            continue
        matches = re.findall(r".+?(?:[.!?…]+[\"'”’»)\]]*|$)", para)
        for match in matches:
            unit = re.sub(r"\s+", " ", str(match or "").strip())
            if unit:
                units.append(unit)
    if not units:
        units = [raw]

    pieces: list[str] = []
    current = ""
    current_words = 0
    for unit in units:
        for part in split_piece(unit):
            candidate = f"{current} {part}".strip() if current else part
            part_words = int(_smartblog_render_tts_word_count(part))
            candidate_words = int(current_words + part_words)
            too_long = bool(len(candidate) > limit)
            too_many_words = bool(word_target and current and candidate_words > word_target)
            if current and (too_long or too_many_words):
                pieces.append(current)
                current = part
                current_words = part_words
            else:
                current = candidate
                current_words = candidate_words
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def _smartblog_split_script_for_render_tts(
    text: str,
    *,
    max_chars: int | None = None,
    target_words: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    frame_chunks = _smartblog_split_nextframe_script(text)
    limit = _smartblog_render_tts_max_chars() if max_chars is None else int(max_chars)
    word_target = _smartblog_render_tts_target_words() if target_words is None else int(target_words)
    out: list[dict[str, Any]] = []
    for frame_idx, frame_text in enumerate(frame_chunks):
        subchunks = _smartblog_split_text_for_eleven_tts(
            frame_text,
            max_chars=int(limit),
            target_words=int(word_target),
        )
        for sub_idx, sub_text in enumerate(subchunks):
            out.append(
                {
                    "text": str(sub_text),
                    "frame_index": int(frame_idx),
                    "frame_subindex": int(sub_idx),
                    "frame_subtotal": int(len(subchunks)),
                }
            )
    return out, int(len(frame_chunks))


def _smartblog_render_voice_config(claim: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    payload = _smartblog_job_payload(claim)
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    payload_voice = payload.get("voice") if isinstance(payload.get("voice"), dict) else {}
    payload_voice_config = payload.get("voice_config") if isinstance(payload.get("voice_config"), dict) else {}
    claim_voice = claim.get("voice") if isinstance(claim.get("voice"), dict) else {}
    persona_voice = persona.get("voice_config") if isinstance(persona.get("voice_config"), dict) else {}
    persona_prompt_config = persona.get("prompt_config") if isinstance(persona.get("prompt_config"), dict) else {}
    persona_prompt_voice = (
        persona_prompt_config.get("voice_config")
        if isinstance(persona_prompt_config.get("voice_config"), dict)
        else {}
    )
    merged: dict[str, Any] = {}
    for src in (persona_prompt_voice, persona_voice, payload_voice_config, payload_voice, claim_voice):
        if isinstance(src, dict):
            merged.update(src)

    voice_id = str(merged.get("voice_id") or os.getenv("DEFAULT_ELEVEN_VOICE_ID", "") or "").strip()
    model_id = str(merged.get("model_id") or os.getenv("WORKER_ELEVEN_WS_MODEL_ID", "eleven_v3") or "eleven_v3").strip()
    stability_mode = str(merged.get("stability_mode") or "").strip().lower()
    if stability_mode not in {"creative", "natural", "robust"}:
        try:
            stability = float(merged.get("stability"))
            stability_mode = "creative" if stability <= 0.33 else "robust" if stability >= 0.67 else "natural"
        except Exception:
            stability_mode = "natural"

    mode_defaults = {
        "creative": {"stability": 0.35, "similarity_boost": 0.88, "style": 0.18},
        "natural": {"stability": 0.55, "similarity_boost": 0.92, "style": 0.0},
        "robust": {"stability": 0.85, "similarity_boost": 0.95, "style": 0.0},
    }
    settings: dict[str, Any] = dict(mode_defaults.get(stability_mode, mode_defaults["natural"]))
    for key in ("stability", "similarity_boost", "style", "speed"):
        if merged.get(key) is None:
            continue
        try:
            settings[key] = float(merged.get(key))
        except Exception:
            pass
    parsed_boost = _smartblog_optional_bool(merged.get("use_speaker_boost"))
    settings["use_speaker_boost"] = True if parsed_boost is None else bool(parsed_boost)
    return str(voice_id), str(model_id), dict(settings), str(stability_mode)


def _smartblog_render_audio_entry_avatar_index(entry: dict[str, Any]) -> int | None:
    base_raw = (
        entry.get("avatar_index_base")
        if "avatar_index_base" in entry
        else entry.get("avatarIndexBase")
        if "avatarIndexBase" in entry
        else entry.get("index_base")
        if "index_base" in entry
        else entry.get("indexBase")
    )
    one_based = str(base_raw or "").strip().lower() in {"1", "one", "one-based", "one_based", "human"}
    for key in ("avatar_index", "avatarIndex", "photo_index", "photoIndex", "image_index", "imageIndex", "frame_index", "frameIndex"):
        if key not in entry:
            continue
        try:
            raw = int(entry.get(key))
        except Exception:
            continue
        return max(0, int(raw) - 1) if bool(one_based) else int(raw)
    return None


def _smartblog_render_audio_entry_avatar_url(entry: dict[str, Any]) -> str:
    for key in (
        "avatar_url",
        "avatarUrl",
        "image_url",
        "imageUrl",
        "photo_url",
        "photoUrl",
        "reference_image_url",
        "referenceImageUrl",
    ):
        text = str(entry.get(key) or "").strip()
        if text:
            return text
    return ""


def _smartblog_transition_blur_expr(
    *,
    duration_sec: float,
    blur_in_sec: float,
    blur_out_sec: float,
    strength: float,
    has_start: bool,
    has_end: bool,
) -> str:
    duration = max(0.001, float(duration_sec))
    max_w = max(0.0, min(1.0, float(strength)))
    terms: list[str] = []
    if bool(has_start) and float(blur_out_sec) > 0.0:
        out_sec = max(0.001, min(duration, float(blur_out_sec)))
        terms.append(f"if(lt(T\\,{out_sec:.6f})\\,{max_w:.6f}*(1-T/{out_sec:.6f})\\,0)")
    if bool(has_end) and float(blur_in_sec) > 0.0:
        start = max(0.0, duration - float(blur_in_sec))
        denom = max(0.001, duration - start)
        terms.append(f"if(gt(T\\,{start:.6f})\\,{max_w:.6f}*((T-{start:.6f})/{denom:.6f})\\,0)")
    if not terms or max_w <= 0.0:
        return "0"
    if len(terms) == 1:
        body = terms[0]
    else:
        body = "max(" + "\\,".join(terms) + ")"
    return f"min({max_w:.6f}\\,{body})"


def _smartblog_transition_start_expr(*, blur_out_sec: float, strength: float, has_start: bool) -> str:
    if not bool(has_start):
        return "0"
    out_sec = max(0.001, float(blur_out_sec))
    max_w = max(0.0, min(1.0, float(strength)))
    if max_w <= 0.0:
        return "0"
    return f"if(lt(t\\,{out_sec:.6f})\\,{max_w:.6f}*(1-t/{out_sec:.6f})\\,0)"


def _smartblog_avatar_transition_style_has_punch(style: str) -> bool:
    return str(style or "").strip().lower() in {
        "punch",
        "punch_in",
        "cross_blur_punch",
        "blur_punch",
    }


def _smartblog_apply_segment_transition_blur(
    *,
    src_path: str,
    out_path: str,
    has_start: bool,
    has_end: bool,
    width: int = 0,
    height: int = 0,
) -> str:
    if not bool(_env_flag("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR", "1")):
        return str(src_path)
    if not bool(has_start or has_end):
        return str(src_path)
    duration = float(video_duration_sec(str(src_path)))
    if duration <= 0.0:
        return str(src_path)
    blur_in_sec = max(0.0, min(2.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR_IN_SEC", 0.25)))
    blur_out_sec = max(0.0, min(2.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR_OUT_SEC", 0.35)))
    strength = max(0.0, min(1.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR_STRENGTH", 0.60)))
    sigma = max(0.0, min(64.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR_SIGMA", 8.0)))
    style = str(os.getenv("SMARTBLOG_RENDER_AVATAR_TRANSITION_STYLE", "cross_blur_punch") or "cross_blur_punch").strip().lower()
    punch_zoom = max(1.0, min(1.25, _safe_float_env("SMARTBLOG_RENDER_AVATAR_TRANSITION_PUNCH_ZOOM", 1.055)))
    punch_enabled = (
        bool(_smartblog_avatar_transition_style_has_punch(style))
        and bool(has_start)
        and int(width) > 0
        and int(height) > 0
        and float(punch_zoom) > 1.0005
        and float(blur_out_sec) > 0.0
    )
    if strength <= 0.0 or (sigma <= 0.0 and not bool(punch_enabled)):
        return str(src_path)
    weight = _smartblog_transition_blur_expr(
        duration_sec=float(duration),
        blur_in_sec=float(blur_in_sec),
        blur_out_sec=float(blur_out_sec),
        strength=float(strength),
        has_start=bool(has_start),
        has_end=bool(has_end),
    )
    if weight == "0" and not bool(punch_enabled):
        return str(src_path)
    out = os.path.abspath(str(out_path or "").strip())
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    filters: list[str] = []
    video_label = "0:v"
    if bool(punch_enabled):
        zoom_weight = _smartblog_transition_start_expr(
            blur_out_sec=float(blur_out_sec),
            strength=1.0,
            has_start=bool(has_start),
        )
        zoom_expr = f"1+({float(punch_zoom) - 1.0:.6f})*({zoom_weight})"
        filters.append(
            f"[0:v]scale=w='trunc(iw*({zoom_expr})/2)*2':"
            f"h='trunc(ih*({zoom_expr})/2)*2':eval=frame,"
            f"crop=w={int(width)}:h={int(height)}:x=(iw-{int(width)})/2:y=(ih-{int(height)})/2[base]"
        )
        video_label = "base"
    if weight != "0" and sigma > 0.0:
        filters.append(
            f"[{video_label}]split[orig][tmp];"
            f"[tmp]gblur=sigma={float(sigma):.3f}[blur];"
            f"[orig][blur]blend=all_expr='A*(1-({weight}))+B*{weight}'[v]"
        )
    elif bool(punch_enabled):
        filters.append(f"[{video_label}]null[v]")
    filter_complex = ";".join(filters)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(src_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            str(os.getenv("SMARTBLOG_RENDER_TRANSITION_VIDEO_ENCODER", "libx264") or "libx264"),
            "-preset",
            str(os.getenv("SMARTBLOG_RENDER_TRANSITION_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("SMARTBLOG_RENDER_TRANSITION_CRF", "18") or "18"),
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
    logging.warning(
        "SmartBlog render avatar transition: src=%s out=%s style=%s duration=%.3fs start=%d end=%d in=%.2fs out=%.2fs strength=%.2f sigma=%.1f punch=%.3f",
        os.path.basename(str(src_path)),
        os.path.basename(str(out)),
        str(style or "blur"),
        float(duration),
        1 if bool(has_start) else 0,
        1 if bool(has_end) else 0,
        float(blur_in_sec),
        float(blur_out_sec),
        float(strength),
        float(sigma),
        float(punch_zoom),
    )
    return str(out)


def _smartblog_hunyuan_visual_match_filters(*, reference: bool = False) -> list[str]:
    if not bool(_env_flag("SMARTBLOG_HUNYUAN_VISUAL_MATCH", os.getenv("SMARTBLOG_LTX_VISUAL_MATCH", "1") or "1")):
        return []
    if bool(reference):
        override = str(
            os.getenv("SMARTBLOG_HUNYUAN_REFERENCE_VISUAL_MATCH_FILTER")
            or os.getenv("SMARTBLOG_LTX_REFERENCE_VISUAL_MATCH_FILTER")
            or ""
        ).strip()
    else:
        override = str(os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_FILTER") or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_FILTER") or "").strip()
    if override:
        if override.lower() in {"0", "false", "no", "off", "none"}:
            return []
        return [str(override)]
    filters: list[str] = []
    eq_default = (
        "eq=contrast=1.025:saturation=1.07:gamma=0.995"
        if bool(reference)
        else "eq=contrast=1.055:saturation=1.18:gamma=0.985"
    )
    if bool(reference):
        raw_eq = str(
            os.getenv("SMARTBLOG_HUNYUAN_REFERENCE_VISUAL_MATCH_EQ")
            or os.getenv("SMARTBLOG_LTX_REFERENCE_VISUAL_MATCH_EQ")
            or eq_default
        ).strip()
    else:
        raw_eq = str(os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_EQ") or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_EQ") or eq_default).strip()
    if raw_eq and raw_eq.lower() not in {"0", "false", "no", "off", "none"}:
        filters.append(raw_eq if raw_eq.startswith("eq=") else f"eq={raw_eq}")
    if bool(reference):
        raw_unsharp = str(
            os.getenv("SMARTBLOG_HUNYUAN_REFERENCE_VISUAL_MATCH_UNSHARP")
            or os.getenv("SMARTBLOG_LTX_REFERENCE_VISUAL_MATCH_UNSHARP")
            or "3:3:0.20:3:3:0.0"
            or ""
        ).strip()
    else:
        raw_unsharp = str(
            os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_UNSHARP")
            or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_UNSHARP")
            or os.getenv("REMOTE_EDGE_FILE_UNSHARP", "3:3:0.30:3:3:0.0")
            or ""
        ).strip()
    if raw_unsharp and raw_unsharp.lower() not in {"0", "false", "no", "off", "none"}:
        filters.append(raw_unsharp if raw_unsharp.startswith("unsharp=") else f"unsharp={raw_unsharp}")
    return filters


def _smartblog_apply_video_filter_chain(*, src_path: str, out_path: str, filters: list[str], log_label: str) -> str:
    src = os.path.abspath(str(src_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"{log_label} source missing: {src}")
    if not out:
        raise RuntimeError(f"{log_label} output path is required")
    clean_filters = [str(item).strip() for item in list(filters or []) if str(item or "").strip()]
    if not clean_filters:
        return str(src)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    filter_chain = ",".join(clean_filters)
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
            str(filter_chain),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            str(os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_VIDEO_ENCODER") or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_VIDEO_ENCODER", "libx264") or "libx264"),
            "-preset",
            str(os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_PRESET") or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("SMARTBLOG_HUNYUAN_VISUAL_MATCH_CRF") or os.getenv("SMARTBLOG_LTX_VISUAL_MATCH_CRF", "18") or "18"),
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
    logging.warning(
        "%s applied: src=%s out=%s filters=%s",
        str(log_label),
        os.path.basename(str(src)),
        os.path.basename(str(out)),
        str(filter_chain),
    )
    return str(out)


def _smartblog_apply_hunyuan_visual_match(*, src_path: str, out_path: str, reference_image_path: str = "") -> str:
    src = os.path.abspath(str(src_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"Hunyuan visual match source missing: {src}")
    if not out:
        raise RuntimeError("Hunyuan visual match output path is required")
    ref = os.path.abspath(str(reference_image_path or "").strip())
    has_ref = bool(ref and os.path.exists(ref))
    if has_ref:
        ref_mode = str(
            os.getenv("SMARTBLOG_HUNYUAN_REFERENCE_COLOR_MATCH_MODE")
            or os.getenv("SMARTBLOG_LTX_REFERENCE_COLOR_MATCH_MODE", "match")
            or "match"
        ).strip().lower()
        if ref_mode in {"", "0", "false", "no", "off", "none", "preserve", "copy", "passthrough"}:
            logging.warning(
                "SmartBlog Hunyuan visual match skipped for referenced continuation: src=%s ref=%s mode=%s",
                os.path.basename(str(src)),
                os.path.basename(str(ref)),
                str(ref_mode or "preserve"),
            )
            return str(src)
    filters = _smartblog_hunyuan_visual_match_filters(reference=bool(has_ref))
    if not filters:
        return str(src_path)
    return _smartblog_apply_video_filter_chain(
        src_path=str(src),
        out_path=str(out),
        filters=list(filters),
        log_label="SmartBlog Hunyuan visual match",
    )


def _smartblog_concat_mp4_copy(segment_paths: list[str], out_path: str) -> str:
    paths = [os.path.abspath(str(p)) for p in list(segment_paths or []) if str(p or "").strip()]
    if not paths:
        raise RuntimeError("no MP4 segments to concatenate")
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f"MP4 segment missing: {path}")
    out = os.path.abspath(str(out_path or "").strip())
    if not out:
        raise RuntimeError("concat output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    list_path = f"{out}.concat.txt"

    def quote_concat_path(path: str) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"

    with open(list_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(f"file {quote_concat_path(path)}\n")
    subprocess.run(
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
            list_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out,
        ],
        check=True,
    )
    return out


def _smartblog_concat_mp4_reencode(
    segment_paths: list[str],
    out_path: str,
    *,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
) -> str:
    paths = [os.path.abspath(str(p)) for p in list(segment_paths or []) if str(p or "").strip()]
    if not paths:
        raise RuntimeError("no MP4 segments to concatenate")
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f"MP4 segment missing: {path}")
    out = os.path.abspath(str(out_path or "").strip())
    if not out:
        raise RuntimeError("concat output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    list_path = f"{out}.concat.txt"

    def quote_concat_path(path: str) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"

    with open(list_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(f"file {quote_concat_path(path)}\n")

    vf_parts: list[str] = []
    if float(fps or 0.0) > 0.0:
        vf_parts.append(f"fps={float(fps):.6f}")
    if int(width) > 0 and int(height) > 0:
        vf_parts.append(f"scale={int(width)}:{int(height)}:flags=bicubic")
    vf_parts.append("setsar=1")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(vf_parts),
    ]
    if float(fps or 0.0) > 0.0:
        cmd += ["-r", f"{float(fps):.6f}", "-fps_mode", "cfr"]
    cmd += [
        "-c:v",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_VIDEO_ENCODER", "libx264") or "libx264"),
        "-preset",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_PRESET", "veryfast") or "veryfast"),
        "-crf",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_CRF", "18") or "18"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        str(os.getenv("REMOTE_EDGE_FILE_AUDIO_BITRATE", "160k") or "160k"),
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def _smartblog_expected_timeline_duration(segment_infos: list[dict[str, Any]]) -> float:
    total = 0.0
    for info in list(segment_infos or []):
        if not isinstance(info, dict):
            continue
        raw = (
            info.get("timeline_duration_sec")
            if info.get("timeline_duration_sec") is not None
            else info.get("target_duration_sec")
        )
        if raw is None:
            raw = info.get("duration_sec")
        try:
            duration = float(raw or 0.0)
        except Exception:
            duration = 0.0
        if duration > 0.0:
            total += float(duration)
    return float(total)


def _smartblog_concat_final_timeline(
    segment_paths: list[str],
    out_path: str,
    *,
    segment_infos: list[dict[str, Any]],
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
) -> str:
    expected_duration = float(_smartblog_expected_timeline_duration(list(segment_infos or [])) or 0.0)
    path_duration = 0.0
    probe_parts: list[str] = []
    for path in list(segment_paths or []):
        try:
            probe = probe_video_metadata(str(path))
            path_duration += float(probe.duration_sec or 0.0)
            probe_parts.append(
                f"{os.path.basename(str(path))}:{float(probe.fps or 0.0):.3f}fps/{int(probe.frames or 0)}f/{float(probe.duration_sec or 0.0):.3f}s"
            )
        except Exception:
            try:
                path_duration += float(video_duration_sec(str(path)) or 0.0)
            except Exception:
                pass

    reference_duration = float(expected_duration or path_duration or 0.0)
    tolerance = max(0.35, float(reference_duration) * 0.01) if reference_duration > 0.0 else 0.35
    force_cfr = bool(float(fps or 0.0) > 0.0) and not bool(
        _env_flag("SMARTBLOG_RENDER_FINAL_TIMELINE_COPY_IF_DURATION_OK", "0")
    )

    if bool(force_cfr):
        _smartblog_concat_mp4_reencode(
            list(segment_paths or []),
            str(out_path),
            width=int(width or 0),
            height=int(height or 0),
            fps=float(fps or 0.0),
        )
        actual_duration = float(video_duration_sec(str(out_path)) or 0.0)
        logging.warning(
            "SmartBlog final timeline concat: mode=cfr_reencode segments=%d expected=%.3fs paths=%.3fs actual=%.3fs fps=%.3f tolerance=%.3fs inputs=%s",
            int(len(segment_paths or [])),
            float(expected_duration),
            float(path_duration),
            float(actual_duration),
            float(fps or 0.0),
            float(tolerance),
            "; ".join(probe_parts[:12]),
        )
        if reference_duration > 0.0 and actual_duration > 0.0 and abs(actual_duration - reference_duration) > tolerance:
            raise RuntimeError(
                "render final timeline duration mismatch after CFR reencode: "
                f"expected={expected_duration:.3f}s paths={path_duration:.3f}s actual={actual_duration:.3f}s"
            )
        return str(out_path)

    _smartblog_concat_mp4_copy(list(segment_paths or []), str(out_path))
    actual_duration = float(video_duration_sec(str(out_path)) or 0.0)
    logging.warning(
        "SmartBlog final timeline concat: mode=copy segments=%d expected=%.3fs paths=%.3fs actual=%.3fs tolerance=%.3fs inputs=%s",
        int(len(segment_paths or [])),
        float(expected_duration),
        float(path_duration),
        float(actual_duration),
        float(tolerance),
        "; ".join(probe_parts[:12]),
    )
    if reference_duration > 0.0 and actual_duration > 0.0 and abs(actual_duration - reference_duration) <= tolerance:
        return str(out_path)

    logging.warning(
        "SmartBlog final timeline concat duration mismatch: expected=%.3fs paths=%.3fs actual=%.3fs; retrying with reencode",
        float(expected_duration),
        float(path_duration),
        float(actual_duration),
    )
    _smartblog_concat_mp4_reencode(
        list(segment_paths or []),
        str(out_path),
        width=int(width or 0),
        height=int(height or 0),
        fps=float(fps or 0.0),
    )
    actual_duration = float(video_duration_sec(str(out_path)) or 0.0)
    logging.warning(
        "SmartBlog final timeline reencoded: expected=%.3fs paths=%.3fs actual=%.3fs",
        float(expected_duration),
        float(path_duration),
        float(actual_duration),
    )
    if reference_duration > 0.0 and actual_duration > 0.0 and abs(actual_duration - reference_duration) > tolerance:
        raise RuntimeError(
            "render final timeline duration mismatch after reencode: "
            f"expected={expected_duration:.3f}s paths={path_duration:.3f}s actual={actual_duration:.3f}s"
        )
    return str(out_path)


def _smartblog_extract_last_video_frame(video_path: str, out_path: str) -> str:
    src = os.path.abspath(str(video_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"video source missing for last-frame extraction: {video_path}")
    if not out:
        raise RuntimeError("last-frame output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed for last-frame extraction: {src}")
    frame = None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            for idx in range(max(0, frame_count - 8), frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, candidate = cap.read()
                if not ok or candidate is None:
                    break
                frame = candidate
    finally:
        cap.release()
    if frame is None:
        raise RuntimeError(f"could not read last frame from video: {src}")
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    tmp_out = f"{out}.tmp.png"
    ok, encoded = cv2.imencode(".png", frame)
    if not ok or encoded is None:
        raise RuntimeError(f"cv2.imencode failed for last-frame extraction: {out}")
    with open(tmp_out, "wb") as f:
        f.write(encoded.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_out, out)
    if cv2.imread(out, cv2.IMREAD_COLOR) is None:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-sseof",
                    "-0.5",
                    "-i",
                    src,
                    "-frames:v",
                    "1",
                    out,
                ],
                check=True,
            )
        except Exception as e:
            raise RuntimeError(f"last-frame PNG readback failed and ffmpeg fallback failed: {out}") from e
        if cv2.imread(out, cv2.IMREAD_COLOR) is None:
            raise RuntimeError(f"last-frame extraction produced unreadable image: {out}")
    logging.warning(
        "SmartBlog render extracted continuation frame: src=%s out=%s size=%dx%d",
        os.path.basename(str(src)),
        os.path.basename(str(out)),
        int(src_w),
        int(src_h),
    )
    return str(out)


def _smartblog_parse_render_size_hw(size: Any) -> tuple[int, int] | None:
    raw = str(size or "").strip().lower()
    match = re.match(r"^\s*(\d{2,5})\s*[*x]\s*(\d{2,5})\s*$", raw)
    if not match:
        return None
    height = int(match.group(1))
    width = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return int(height), int(width)


def _smartblog_compensate_continuation_ref_framing(
    src_path: str,
    out_path: str,
    *,
    target_width: int,
    target_height: int,
    job_id: str = "",
    segment_index: int = 0,
    reason: str = "",
) -> str:
    src = os.path.abspath(str(src_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    target_w = int(target_width or 0)
    target_h = int(target_height or 0)
    if not src or not os.path.exists(src) or not out or target_w <= 0 or target_h <= 0:
        return str(src)
    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        return str(src)
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if src_w <= 0 or src_h <= 0:
        return str(src)
    src_aspect = float(src_w) / float(src_h)
    target_aspect = float(target_w) / float(target_h)
    tolerance = float(max(0.0, min(0.005, _safe_float_env("SMARTBLOG_RENDER_CONTINUATION_ASPECT_TOLERANCE", 0.0005))))
    if target_aspect <= 0.0 or abs(src_aspect - target_aspect) / max(target_aspect, 1e-6) < tolerance:
        return str(src)

    if src_aspect > target_aspect:
        canvas_w = int(src_w)
        canvas_h = int(math.ceil(float(src_w) / float(target_aspect)))
    else:
        canvas_h = int(src_h)
        canvas_w = int(math.ceil(float(src_h) * float(target_aspect)))
    canvas_w = max(int(src_w), int(canvas_w))
    canvas_h = max(int(src_h), int(canvas_h))
    if canvas_w % 2:
        canvas_w += 1
    if canvas_h % 2:
        canvas_h += 1

    cover_scale = max(float(canvas_w) / float(src_w), float(canvas_h) / float(src_h))
    cover_w = max(1, int(round(float(src_w) * float(cover_scale))))
    cover_h = max(1, int(round(float(src_h) * float(cover_scale))))
    cover = cv2.resize(image, (cover_w, cover_h), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (cover_w - canvas_w) // 2)
    y0 = max(0, (cover_h - canvas_h) // 2)
    bg = cover[y0 : y0 + canvas_h, x0 : x0 + canvas_w].copy()
    if int(bg.shape[1]) != canvas_w or int(bg.shape[0]) != canvas_h:
        bg = cv2.resize(bg, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    sigma = max(12.0, min(float(min(canvas_w, canvas_h)) * 0.045, 36.0))
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    paste_x = max(0, (canvas_w - src_w) // 2)
    paste_y = max(0, (canvas_h - src_h) // 2)
    bg[paste_y : paste_y + src_h, paste_x : paste_x + src_w] = image
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not cv2.imwrite(out, bg):
        raise RuntimeError(f"cv2.imwrite failed for continuation framing: {out}")
    logging.warning(
        "SmartBlog render continuation image framed: job=%s segment=%d reason=%s src=%s out=%s src=%dx%d target=%dx%d canvas=%dx%d pad_x=%d pad_y=%d",
        str(job_id or "-"),
        int(segment_index),
        str(reason or "-"),
        os.path.basename(str(src)),
        os.path.basename(str(out)),
        int(src_w),
        int(src_h),
        int(target_w),
        int(target_h),
        int(canvas_w),
        int(canvas_h),
        int(canvas_w - src_w),
        int(canvas_h - src_h),
    )
    return str(out)


def _smartblog_mux_silent_audio(
    *,
    video_path: str,
    out_path: str,
    duration_sec: float,
    sample_rate: int = 48000,
) -> str:
    src = os.path.abspath(str(video_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"silent mux source missing: {src}")
    if not out:
        raise RuntimeError("silent mux output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    duration = max(0.001, float(duration_sec or 0.0))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            src,
            "-f",
            "lavfi",
            "-t",
            f"{float(duration):.6f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={int(max(1, int(sample_rate)))}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
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
            out,
        ],
        check=True,
    )
    return str(out)


def _smartblog_mux_external_audio(
    *,
    video_path: str,
    audio_path: str,
    out_path: str,
    duration_sec: float,
    sample_rate: int = 48000,
) -> str:
    src = os.path.abspath(str(video_path or "").strip())
    audio = os.path.abspath(str(audio_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"audio mux video source missing: {src}")
    if not audio or not os.path.exists(audio):
        raise RuntimeError(f"audio mux audio source missing: {audio}")
    if not out:
        raise RuntimeError("audio mux output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    duration = max(0.001, float(duration_sec or video_duration_sec(src) or 0.0))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(src),
            "-i",
            str(audio),
            "-filter_complex",
            (
                f"[1:a:0]aformat=channel_layouts=stereo,aresample={int(max(1, int(sample_rate)))},"
                f"apad,atrim=0:{float(duration):.6f}[a]"
            ),
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
        ],
        check=True,
    )
    logging.warning(
        "SmartBlog video external audio muxed: video=%s audio=%s out=%s duration=%.3fs",
        os.path.basename(str(src)),
        os.path.basename(str(audio)),
        os.path.basename(str(out)),
        float(duration),
    )
    return str(out)


def _smartblog_audio_duration_sec(path: str) -> float:
    src = os.path.abspath(str(path or "").strip())
    if not src or not os.path.exists(src):
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if int(proc.returncode or 0) != 0:
        return 0.0
    try:
        value = float(str(proc.stdout or "").strip() or 0.0)
    except Exception:
        return 0.0
    return float(value if math.isfinite(value) and value > 0.0 else 0.0)


def _smartblog_video_has_audio_stream(path: str) -> bool:
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


def _smartblog_ffmpeg_volume_db(value: float) -> str:
    try:
        gain = float(value)
    except Exception:
        gain = 0.0
    if not math.isfinite(gain):
        gain = 0.0
    gain = float(max(-60.0, min(24.0, gain)))
    return f"{gain:.3f}dB"


def _smartblog_mux_mixed_audio(
    *,
    video_path: str,
    voice_audio_path: str = "",
    background_audio_path: str = "",
    out_path: str,
    duration_sec: float,
    sample_rate: int = 48000,
    voice_gain_db: float = 0.0,
    background_gain_db: float = 0.0,
) -> str:
    src = os.path.abspath(str(video_path or "").strip())
    voice = os.path.abspath(str(voice_audio_path or "").strip()) if str(voice_audio_path or "").strip() else ""
    bg = os.path.abspath(str(background_audio_path or "").strip()) if str(background_audio_path or "").strip() else ""
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"audio mix video source missing: {src}")
    if voice and not os.path.exists(voice):
        raise RuntimeError(f"audio mix voice source missing: {voice}")
    if bg and not os.path.exists(bg):
        raise RuntimeError(f"audio mix background source missing: {bg}")
    if not voice and not bg:
        return _smartblog_mux_silent_audio(
            video_path=str(src),
            out_path=str(out),
            duration_sec=float(duration_sec),
            sample_rate=int(sample_rate),
        )
    if voice and not bg and abs(float(voice_gain_db or 0.0)) <= 0.001:
        return _smartblog_mux_external_audio(
            video_path=str(src),
            audio_path=str(voice),
            out_path=str(out),
            duration_sec=float(duration_sec),
            sample_rate=int(sample_rate),
        )
    if bg and not voice and abs(float(background_gain_db or 0.0)) <= 0.001:
        return _smartblog_mux_external_audio(
            video_path=str(src),
            audio_path=str(bg),
            out_path=str(out),
            duration_sec=float(duration_sec),
            sample_rate=int(sample_rate),
        )
    if not out:
        raise RuntimeError("audio mix output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    duration = max(0.001, float(duration_sec or video_duration_sec(src) or 0.0))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src)]
    filter_parts: list[str] = []
    mix_inputs: list[str] = []
    audio_input_index = 1
    if voice:
        cmd.extend(["-i", str(voice)])
        filter_parts.append(
            f"[{audio_input_index}:a:0]aformat=channel_layouts=stereo,"
            f"aresample={int(max(1, int(sample_rate)))},apad,atrim=0:{float(duration):.6f},"
            f"volume={_smartblog_ffmpeg_volume_db(float(voice_gain_db or 0.0))}[voice]"
        )
        mix_inputs.append("[voice]")
        audio_input_index += 1
    if bg:
        cmd.extend(["-i", str(bg)])
        filter_parts.append(
            f"[{audio_input_index}:a:0]aformat=channel_layouts=stereo,"
            f"aresample={int(max(1, int(sample_rate)))},apad,atrim=0:{float(duration):.6f},"
            f"volume={_smartblog_ffmpeg_volume_db(float(background_gain_db or 0.0))}[bg]"
        )
        mix_inputs.append("[bg]")
        audio_input_index += 1
    if len(mix_inputs) == 1:
        filter_parts.append(f"{mix_inputs[0]}atrim=0:{float(duration):.6f}[a]")
    else:
        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={int(len(mix_inputs))}:duration=longest:dropout_transition=0:normalize=0,"
            f"atrim=0:{float(duration):.6f}[a]"
        )
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
    subprocess.run(cmd, check=True)
    logging.warning(
        "SmartBlog video audio mixed: video=%s voice=%s bg=%s out=%s duration=%.3fs voice_gain=%.1fdB bg_gain=%.1fdB",
        os.path.basename(str(src)),
        os.path.basename(str(voice)) if voice else "-",
        os.path.basename(str(bg)) if bg else "-",
        os.path.basename(str(out)),
        float(duration),
        float(voice_gain_db or 0.0),
        float(background_gain_db or 0.0),
    )
    return str(out)


def _smartblog_mix_background_music(
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
    if not out:
        raise RuntimeError("background music output path is required")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    duration = max(0.001, float(duration_sec or video_duration_sec(src) or 0.0))
    has_main_audio = bool(_smartblog_video_has_audio_stream(src))
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
            f"volume={_smartblog_ffmpeg_volume_db(float(gain_db or 0.0))}"
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and duck_db < -0.001:
        logging.warning(
            "SmartBlog background music ducking failed, retrying plain mix: video=%s music=%s duration=%.3fs rc=%s stderr=%s",
            os.path.basename(str(src)),
            os.path.basename(str(music)),
            float(duration),
            result.returncode,
            (result.stderr or "").strip()[-1000:],
        )
        cmd = build_cmd(use_ducking=False)
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "background music mix failed: "
            f"rc={result.returncode} stderr={(result.stderr or '').strip()[-2000:]}"
        )
    logging.warning(
        "SmartBlog background music mixed: video=%s music=%s out=%s duration=%.3fs gain=%.1fdB loop=%d duck=%.1fdB fade_in=%.2fs fade_out=%.2fs offset=%.2fs main_audio=%d",
        os.path.basename(str(src)),
        os.path.basename(str(music)),
        os.path.basename(str(out)),
        float(duration),
        float(gain_db or 0.0),
        1 if bool(loop) else 0,
        float(duck_db),
        float(fade_in),
        float(fade_out),
        float(start_offset),
        1 if has_main_audio else 0,
    )
    return str(out)


def _smartblog_int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _smartblog_entry_source_audio_index(entry: dict[str, Any], default: int = 0) -> int:
    return _smartblog_int_value(
        (entry or {}).get("source_audio_index", (entry or {}).get("sourceAudioIndex", (entry or {}).get("index"))),
        int(default),
    )


def _smartblog_frame_insert_entries(entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw = entry.get("_smartblog_frame_inserts") if isinstance(entry, dict) else []
    inserts = [dict(item or {}) for item in list(raw or []) if isinstance(item, dict)]
    inserts.sort(
        key=lambda item: (
            int(item.get("_smartblog_insert_order") or 0),
            int(item.get("_smartblog_ltx_insert_after_chunk") or -1),
        )
    )
    return inserts


def _smartblog_avatar_frame_audio_chunks(info: dict[str, Any]) -> list[dict[str, Any]]:
    entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
    raw = entry.get("_smartblog_frame_audio_chunks") if isinstance(entry, dict) else []
    chunks = [dict(item or {}) for item in list(raw or []) if isinstance(item, dict)]
    duration = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
    if not chunks:
        chunks = [
            {
                "chunk_pos": 0,
                "index": int(_smartblog_entry_source_audio_index(entry, 0)),
                "source_audio_index": int(_smartblog_entry_source_audio_index(entry, 0)),
                "offset_sec": 0.0,
                "duration_sec": float(max(0.0, duration)),
                "text": _smartblog_audio_entry_text(entry),
                "alignment": _smartblog_audio_entry_alignment(entry, normalized=False),
                "normalized_alignment": _smartblog_audio_entry_alignment(entry, normalized=True),
            }
        ]
    if len(chunks) == 1 and float(chunks[0].get("duration_sec") or 0.0) <= 0.0 and duration > 0.0:
        chunks[0]["duration_sec"] = float(duration)
    chunks.sort(key=lambda item: float(item.get("offset_sec") or 0.0))
    return chunks


def _smartblog_insert_mode(insert: dict[str, Any]) -> str:
    mode = str((insert or {}).get("_smartblog_ltx_mode") or (insert or {}).get("mode") or "cut").strip().lower()
    return "overlay" if mode == "overlay" else "cut"


def _smartblog_insert_duration_sec(insert: dict[str, Any]) -> float:
    try:
        return float(max(0.0, float((insert or {}).get("_smartblog_ltx_duration_sec") or 0.0)))
    except Exception:
        return 0.0


def _smartblog_direct_clip_requested(insert: dict[str, Any]) -> bool:
    if not isinstance(insert, dict):
        return False
    opt = _smartblog_optional_bool(
        insert.get(
            "_smartblog_direct_clip",
            insert.get(
                "direct_clip",
                insert.get("directClip", insert.get("direct_clip_passthrough", insert.get("directClipPassthrough"))),
            ),
        )
    )
    return bool(opt) if opt is not None else False


def _smartblog_direct_clip_video_url(insert: dict[str, Any]) -> str:
    if not isinstance(insert, dict):
        return ""
    return str(
        _smartblog_first_text(
            insert.get("_smartblog_direct_clip_video_url"),
            insert.get("video_url"),
            insert.get("videoUrl"),
            insert.get("source_video_url"),
            insert.get("sourceVideoUrl"),
            insert.get("asset_video_url"),
            insert.get("assetVideoUrl"),
            insert.get("asset_url"),
            insert.get("assetUrl"),
            insert.get("media_url"),
            insert.get("mediaUrl"),
            insert.get("src"),
            insert.get("url"),
        )
        or ""
    ).strip()


def _smartblog_insert_start_sec(
    insert: dict[str, Any],
    *,
    chunks: list[dict[str, Any]],
    segment_duration_sec: float,
) -> float:
    mode = _smartblog_insert_mode(dict(insert or {}))
    if mode == "overlay":
        try:
            return float(max(0.0, min(float(segment_duration_sec), float(insert.get("_smartblog_ltx_t_offset_seconds") or 0.0))))
        except Exception:
            return 0.0
    try:
        after = int(insert.get("_smartblog_ltx_insert_after_chunk"))
    except Exception:
        after = -1
    if after < 0:
        return 0.0
    for chunk in list(chunks or []):
        ids: set[int] = set()
        for key in ("chunk_pos", "index", "source_audio_index"):
            try:
                ids.add(int(chunk.get(key)))
            except Exception:
                pass
        if int(after) in ids:
            return float(
                max(
                    0.0,
                    min(
                        float(segment_duration_sec),
                        float(chunk.get("offset_sec") or 0.0) + float(chunk.get("duration_sec") or 0.0),
                    ),
                )
            )
    if 0 <= int(after) < len(chunks):
        chunk = chunks[int(after)]
        return float(
            max(
                0.0,
                min(
                    float(segment_duration_sec),
                    float(chunk.get("offset_sec") or 0.0) + float(chunk.get("duration_sec") or 0.0),
                ),
            )
        )
    return float(max(0.0, min(float(segment_duration_sec), float(segment_duration_sec))))


def _smartblog_cut_inserts_for_segment(info: dict[str, Any]) -> list[dict[str, Any]]:
    entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
    chunks = _smartblog_avatar_frame_audio_chunks(dict(info or {}))
    duration = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
    out: list[dict[str, Any]] = []
    for insert in _smartblog_frame_insert_entries(dict(entry or {})):
        if _smartblog_insert_mode(insert) != "cut":
            continue
        item = dict(insert)
        item["_smartblog_insert_start_sec"] = float(
            _smartblog_insert_start_sec(item, chunks=list(chunks), segment_duration_sec=float(duration))
        )
        out.append(item)
    out.sort(key=lambda item: (float(item.get("_smartblog_insert_start_sec") or 0.0), int(item.get("_smartblog_insert_order") or 0)))
    return out


def _smartblog_cut_insert_shift_before_time(info: dict[str, Any], *, time_sec: float) -> float:
    shift = 0.0
    for insert in _smartblog_cut_inserts_for_segment(dict(info or {})):
        start = float(insert.get("_smartblog_insert_start_sec") or 0.0)
        if float(start) <= float(time_sec) + 1e-6:
            shift += float(_smartblog_insert_duration_sec(insert))
    return float(max(0.0, shift))


def _smartblog_trim_mp4_interval(
    *,
    src_path: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    fps: float = 0.0,
) -> str:
    src = os.path.abspath(str(src_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"trim source missing: {src}")
    duration = max(0.0, float(end_sec) - float(start_sec))
    if duration <= 0.005:
        return ""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vf_parts: list[str] = []
    if float(fps or 0.0) > 0.0:
        vf_parts.append(f"fps={float(fps):.6f}")
    vf_parts.append(f"scale={int(width)}:{int(height)}:flags=bicubic")
    vf_parts.append("setsar=1")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{float(start_sec):.6f}",
        "-t",
        f"{float(duration):.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(vf_parts),
    ]
    if float(fps or 0.0) > 0.0:
        cmd += ["-r", f"{float(fps):.6f}", "-fps_mode", "cfr"]
    cmd += [
        "-c:v",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_VIDEO_ENCODER", "libx264") or "libx264"),
        "-preset",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_PRESET", "veryfast") or "veryfast"),
        "-crf",
        str(os.getenv("SMARTBLOG_RENDER_SPLICE_CRF", "18") or "18"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        str(os.getenv("REMOTE_EDGE_FILE_AUDIO_BITRATE", "160k") or "160k"),
        "-ar",
        "48000",
        "-ac",
        "2",
        "-reset_timestamps",
        "1",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return str(out)


def _smartblog_normalize_direct_clip_video(
    *,
    src_path: str,
    out_path: str,
    duration_sec: float,
    width: int,
    height: int,
    fps: float,
) -> str:
    src = os.path.abspath(str(src_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"direct clip source missing: {src}")
    if not out:
        raise RuntimeError("direct clip output path is required")
    duration = float(max(0.1, float(duration_sec or 0.0)))
    out_fps = float(max(1.0, float(fps or 0.0) or float(_smartblog_render_source_fps())))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vf = (
        f"fps={float(out_fps):.6f},"
        f"scale={int(width)}:{int(height)}:flags=bicubic,"
        "setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={float(duration):.6f},"
        f"trim=0:{float(duration):.6f},"
        "setpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-r",
        f"{float(out_fps):.6f}",
        "-fps_mode",
        "cfr",
        "-an",
        "-c:v",
        str(os.getenv("SMARTBLOG_RENDER_DIRECT_CLIP_VIDEO_ENCODER", os.getenv("SMARTBLOG_RENDER_SPLICE_VIDEO_ENCODER", "libx264")) or "libx264"),
        "-preset",
        str(os.getenv("SMARTBLOG_RENDER_DIRECT_CLIP_PRESET", os.getenv("SMARTBLOG_RENDER_SPLICE_PRESET", "veryfast")) or "veryfast"),
        "-crf",
        str(os.getenv("SMARTBLOG_RENDER_DIRECT_CLIP_CRF", os.getenv("SMARTBLOG_RENDER_SPLICE_CRF", "18")) or "18"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(out):
        raise RuntimeError(f"direct clip normalization produced no output: {out}")
    return str(out)


def _smartblog_overlay_timeline_clip(
    *,
    base_path: str,
    overlay_path: str,
    out_path: str,
    start_sec: float,
    duration_sec: float,
    width: int,
    height: int,
) -> str:
    base = os.path.abspath(str(base_path or "").strip())
    overlay = os.path.abspath(str(overlay_path or "").strip())
    out = os.path.abspath(str(out_path or "").strip())
    if not base or not os.path.exists(base):
        raise RuntimeError(f"overlay base missing: {base}")
    if not overlay or not os.path.exists(overlay):
        raise RuntimeError(f"overlay clip missing: {overlay}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    start = float(max(0.0, float(start_sec or 0.0)))
    duration = float(max(0.001, float(duration_sec or video_duration_sec(overlay) or 0.0)))
    start_ms = int(round(start * 1000.0))
    filter_complex = (
        f"[0:v]scale={int(width)}:{int(height)}:flags=bicubic,setsar=1[basev];"
        f"[1:v]scale={int(width)}:{int(height)}:flags=bicubic,setsar=1,"
        f"trim=0:{duration:.6f},setpts=PTS-STARTPTS+{start:.6f}/TB[ov];"
        f"[basev][ov]overlay=0:0:enable='between(t,{start:.6f},{start + duration:.6f})'[v];"
        f"[0:a]aformat=channel_layouts=stereo,aresample=48000[ba];"
        f"[1:a]aformat=channel_layouts=stereo,aresample=48000,volume=1.0,"
        f"adelay={start_ms}|{start_ms},apad[a1];"
        f"[ba][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(base),
            "-i",
            str(overlay),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            str(os.getenv("SMARTBLOG_RENDER_OVERLAY_VIDEO_ENCODER", "libx264") or "libx264"),
            "-preset",
            str(os.getenv("SMARTBLOG_RENDER_OVERLAY_PRESET", "veryfast") or "veryfast"),
            "-crf",
            str(os.getenv("SMARTBLOG_RENDER_OVERLAY_CRF", "18") or "18"),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            str(os.getenv("REMOTE_EDGE_FILE_AUDIO_BITRATE", "160k") or "160k"),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return str(out)


def _smartblog_sample_video_frames_for_prompt(
    *,
    video_path: str,
    out_dir: str,
    count: int = 10,
    max_dim: int = 512,
) -> list[str]:
    src = os.path.abspath(str(video_path or "").strip())
    if not src or not os.path.exists(src):
        raise RuntimeError(f"sample video source missing: {src}")
    out_root = os.path.abspath(str(out_dir or "").strip())
    os.makedirs(out_root, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed for prompt sampling: {src}")
    paths: list[str] = []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_count = int(max(1, min(24, int(count or 10))))
        if frame_count <= 0:
            indices = [0]
        else:
            indices = [
                int(max(0, min(frame_count - 1, round(((i + 0.5) / float(target_count)) * float(frame_count - 1)))))
                for i in range(target_count)
            ]
        seen: set[int] = set()
        for sample_pos, idx in enumerate(indices):
            if int(idx) in seen:
                continue
            seen.add(int(idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = int(frame.shape[0]), int(frame.shape[1])
            longest = max(h, w)
            limit = int(max(128, min(1024, int(max_dim or 512))))
            if longest > limit:
                scale = float(limit) / float(longest)
                frame = cv2.resize(
                    frame,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            out_path = os.path.join(out_root, f"sample_{int(sample_pos):02d}_f{int(idx):06d}.jpg")
            if cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86]):
                paths.append(out_path)
    finally:
        cap.release()
    if not paths:
        raise RuntimeError(f"no prompt sample frames extracted from: {src}")
    return paths


def _smartblog_audio_entry_alignment(entry: dict[str, Any], *, normalized: bool) -> dict[str, Any] | None:
    keys = (
        ("normalized_alignment", "normalizedAlignment", "elevenlabs_normalized_alignment", "elevenlabsNormalizedAlignment")
        if bool(normalized)
        else ("alignment", "elevenlabs_alignment", "elevenlabsAlignment")
    )
    for key in keys:
        value = entry.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def _smartblog_audio_entry_text(entry: dict[str, Any]) -> str:
    for key in ("text", "script_text", "scriptText", "transcript", "subtitle_text", "subtitleText"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return ""


def _smartblog_alignment_chars_starts_ends(alignment: dict[str, Any] | None) -> tuple[list[str], list[float], list[float]]:
    if not isinstance(alignment, dict) or not alignment:
        return [], [], []
    chars_raw = alignment.get("characters") or alignment.get("chars")
    chars = [str(ch) for ch in chars_raw] if isinstance(chars_raw, (list, tuple)) else list(str(chars_raw or ""))

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

    starts = _float_list(
        alignment.get("character_start_times_seconds")
        or alignment.get("characterStartTimesSeconds")
        or alignment.get("character_start_times")
        or alignment.get("start_times")
        or alignment.get("starts")
    )
    ends = _float_list(
        alignment.get("character_end_times_seconds")
        or alignment.get("characterEndTimesSeconds")
        or alignment.get("character_end_times")
        or alignment.get("end_times")
        or alignment.get("ends")
    )
    n = min(len(chars), len(starts), len(ends))
    return chars[:n], starts[:n], ends[:n]


def _smartblog_audio_entry_alignment_duration_sec(entry: dict[str, Any]) -> float:
    for normalized in (True, False):
        alignment = _smartblog_audio_entry_alignment(dict(entry or {}), normalized=bool(normalized))
        _chars, _starts, ends = _smartblog_alignment_chars_starts_ends(alignment)
        finite_ends: list[float] = []
        for value in list(ends or []):
            try:
                end = float(value)
            except Exception:
                continue
            if math.isfinite(end) and end > 0.0:
                finite_ends.append(float(end))
        if finite_ends:
            return float(max(finite_ends))
    return 0.0


def _smartblog_audio_entry_alignment_samples(entry: dict[str, Any], *, sample_rate: int) -> int:
    duration = float(_smartblog_audio_entry_alignment_duration_sec(dict(entry or {})))
    if duration <= 0.0:
        return 0
    return int(max(1, int(round(float(duration) * float(max(1, int(sample_rate)))))))


def _smartblog_merge_audio_entry_alignment(
    entries: list[dict[str, Any]],
    *,
    offsets_sec: list[float],
    normalized: bool,
) -> dict[str, Any] | None:
    out_chars: list[str] = []
    out_starts: list[float] = []
    out_ends: list[float] = []
    for entry, offset in zip(list(entries or []), list(offsets_sec or []), strict=False):
        alignment = _smartblog_audio_entry_alignment(dict(entry or {}), normalized=bool(normalized))
        chars, starts, ends = _smartblog_alignment_chars_starts_ends(alignment)
        n = min(len(chars), len(starts), len(ends))
        if n <= 0:
            continue
        for ch, start, end in zip(chars[:n], starts[:n], ends[:n], strict=False):
            try:
                s = float(start) + float(offset)
                e = float(end) + float(offset)
            except Exception:
                continue
            if not math.isfinite(s) or not math.isfinite(e):
                continue
            if e < s:
                e = s
            out_chars.append(str(ch))
            out_starts.append(float(s))
            out_ends.append(float(e))
    if not out_chars:
        return None
    return {
        "characters": list(out_chars),
        "character_start_times_seconds": list(out_starts),
        "character_end_times_seconds": list(out_ends),
    }


def _smartblog_slice_alignment(
    alignment: dict[str, Any] | None,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any] | None:
    chars, starts, ends = _smartblog_alignment_chars_starts_ends(alignment)
    if not chars:
        return None
    start_f = float(max(0.0, float(start_sec)))
    end_f = float(max(start_f, float(end_sec)))
    out_chars: list[str] = []
    out_starts: list[float] = []
    out_ends: list[float] = []
    for ch, raw_start, raw_end in zip(chars, starts, ends, strict=False):
        try:
            s = float(raw_start)
            e = float(raw_end)
        except Exception:
            continue
        if not math.isfinite(s) or not math.isfinite(e):
            continue
        if e < s:
            e = s
        if e <= start_f or s >= end_f:
            continue
        out_chars.append(str(ch))
        out_starts.append(float(max(0.0, s - start_f)))
        out_ends.append(float(max(0.0, min(e, end_f) - start_f)))
    if not out_chars:
        return None
    return {
        "characters": list(out_chars),
        "character_start_times_seconds": list(out_starts),
        "character_end_times_seconds": list(out_ends),
    }


def _smartblog_slice_audio_entry_for_segment(entry: dict[str, Any], *, start_sec: float, end_sec: float) -> dict[str, Any]:
    out = dict(entry or {})
    text_chars: list[str] = []
    for normalized, keys in (
        (False, ("alignment", "elevenlabs_alignment", "elevenlabsAlignment")),
        (
            True,
            (
                "normalized_alignment",
                "normalizedAlignment",
                "elevenlabs_normalized_alignment",
                "elevenlabsNormalizedAlignment",
            ),
        ),
    ):
        sliced = _smartblog_slice_alignment(
            _smartblog_audio_entry_alignment(dict(entry or {}), normalized=bool(normalized)),
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        if not sliced:
            continue
        target_key = "normalized_alignment" if bool(normalized) else "alignment"
        out[target_key] = dict(sliced)
        for alias in keys:
            if alias != target_key and alias in out:
                out.pop(alias, None)
        if not text_chars:
            text_chars = [str(ch) for ch in list(sliced.get("characters") or [])]
    if text_chars:
        sliced_text = re.sub(r"\s+", " ", "".join(text_chars)).strip()
        if sliced_text:
            out["text"] = sliced_text
    return out


def _smartblog_worker_tts_visual_key(entry: dict[str, Any]) -> tuple[str, str]:
    avatar_url = _smartblog_canonical_media_url(
        str(_smartblog_render_audio_entry_avatar_url(dict(entry or {})) or "").strip()
    )
    if avatar_url:
        return ("url", avatar_url)
    avatar_idx = _smartblog_render_audio_entry_avatar_index(dict(entry or {}))
    if avatar_idx is None:
        avatar_idx = 0
    return ("index", str(int(avatar_idx)))


def _smartblog_merge_worker_tts_entries_for_visual_segments(
    entries: list[dict[str, Any]],
    *,
    run_dir: str,
    job_id: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(entry or {}) for entry in list(entries or []) if isinstance(entry, dict)],
        key=lambda item: int(item.get("index") or 0),
    )
    if len(ordered) <= 1:
        return ordered
    if not all(bool(entry.get("_smartblog_worker_tts")) for entry in ordered):
        return ordered

    single_avatar_one_pass = bool(
        _smartblog_render_single_avatar_one_pass_enabled()
        and _smartblog_audio_entries_same_avatar_without_timeline_breaks(ordered)
    )
    max_merge_sec = 0.0 if bool(single_avatar_one_pass) else float(_smartblog_render_tts_visual_merge_max_sec())
    if bool(single_avatar_one_pass):
        logging.warning(
            "SmartBlog render worker TTS one-pass visual merge: job=%s chunks=%d key=%s",
            str(job_id or "-"),
            int(len(ordered)),
            "/".join(_smartblog_worker_tts_visual_key(ordered[0])),
        )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    current_sec = 0.0
    for entry in ordered:
        key = _smartblog_worker_tts_visual_key(entry)
        entry_sec = 0.0
        try:
            path = _smartblog_render_audio_entry_local_path(entry)
            if path:
                entry_sec = float(wav_duration_seconds(str(path)))
        except Exception:
            entry_sec = 0.0
        would_exceed = (
            bool(current)
            and float(max_merge_sec) > 0.0
            and float(current_sec) > 0.0
            and float(current_sec + max(0.0, float(entry_sec))) > float(max_merge_sec)
        )
        if current and (key != current_key or bool(would_exceed)):
            groups.append(list(current))
            current = []
            current_sec = 0.0
        current.append(dict(entry))
        current_sec += max(0.0, float(entry_sec))
        current_key = key
    if current:
        groups.append(list(current))
    if len(groups) == len(ordered):
        return ordered

    merged_entries: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups):
        pcm_parts: list[np.ndarray] = []
        offsets_sec: list[float] = []
        sample_rate: int | None = None
        total_samples = 0
        for entry in group:
            path = _smartblog_render_audio_entry_local_path(entry)
            if not path:
                logging.warning(
                    "SmartBlog render worker TTS visual merge skipped: job=%s reason=missing_local_path entry=%s",
                    str(job_id or "-"),
                    int(entry.get("index") or 0),
                )
                return ordered
            pcm, rate = _smartblog_wav_pcm16_mono(str(path))
            if pcm.size <= 0:
                logging.warning(
                    "SmartBlog render worker TTS visual merge skipped: job=%s reason=empty_wav entry=%s",
                    str(job_id or "-"),
                    int(entry.get("index") or 0),
                )
                return ordered
            if sample_rate is None:
                sample_rate = int(rate)
            if int(rate) != int(sample_rate):
                logging.warning(
                    "SmartBlog render worker TTS visual merge skipped: job=%s reason=sample_rate_mismatch entry=%s rate=%d expected=%d",
                    str(job_id or "-"),
                    int(entry.get("index") or 0),
                    int(rate),
                    int(sample_rate),
                )
                return ordered
            offsets_sec.append(float(total_samples) / float(max(1, int(sample_rate))))
            pcm_parts.append(np.asarray(pcm, dtype=np.int16))
            total_samples += int(pcm.size)

        merged_pcm = np.concatenate(pcm_parts).astype(np.int16, copy=False) if pcm_parts else np.zeros(0, dtype=np.int16)
        if merged_pcm.size <= 0 or sample_rate is None:
            return ordered
        out_path = os.path.join(str(run_dir), f"audio_visual_segment_{int(group_idx):03d}_worker_tts_16k.wav")
        _smartblog_write_wav_pcm16_mono(str(out_path), merged_pcm, sample_rate=int(sample_rate))
        first = dict(group[0])
        merged: dict[str, Any] = {
            **first,
            "url": f"file://{urllib.parse.quote(str(out_path))}",
            "local_path": str(out_path),
            "index": int(group_idx),
            "text": " ".join(_smartblog_audio_entry_text(entry) for entry in group if _smartblog_audio_entry_text(entry)).strip(),
            "tts_subchunk_index": 0,
            "tts_subchunk_total": int(len(group)),
            "worker_tts_merged_subchunks": int(len(group)),
            "worker_tts_source_indices": [int(entry.get("index") or 0) for entry in group],
            "_smartblog_audio_chunk": True,
            "_smartblog_worker_tts": True,
            "_smartblog_worker_tts_visual_merged": True,
        }
        alignment = _smartblog_merge_audio_entry_alignment(group, offsets_sec=list(offsets_sec), normalized=False)
        normalized_alignment = _smartblog_merge_audio_entry_alignment(group, offsets_sec=list(offsets_sec), normalized=True)
        if isinstance(alignment, dict) and alignment:
            merged["alignment"] = dict(alignment)
        if isinstance(normalized_alignment, dict) and normalized_alignment:
            merged["normalized_alignment"] = dict(normalized_alignment)
        merged_entries.append(merged)
        logging.warning(
            "SmartBlog render worker TTS visual merge: job=%s visual_segment=%d chunks=%d audio_sec=%.3f key=%s",
            str(job_id or "-"),
            int(group_idx + 1),
            int(len(group)),
            float(merged_pcm.size) / float(max(1, int(sample_rate))),
            "/".join(_smartblog_worker_tts_visual_key(first)),
        )
    return merged_entries


def _smartblog_subtitle_chunks_from_segments(segment_infos: list[dict[str, Any]]) -> list[RenderSubtitleChunk]:
    chunks: list[RenderSubtitleChunk] = []
    offset_sec = 0.0
    for pos, info in enumerate(list(segment_infos or [])):
        kind = str((info or {}).get("kind") or "avatar").strip().lower()
        file_index = _smartblog_segment_file_index(dict(info or {}), fallback=int(pos))
        entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
        duration = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
        sample_rate = int(info.get("sample_rate") or 16000)
        before_sec = float(int(info.get("boundary_before_samples") or 0)) / float(max(1, int(sample_rate)))
        core_start_sec = float(int(info.get("boundary_core_start_samples") or 0)) / float(max(1, int(sample_rate)))
        timeline_duration_raw = info.get("timeline_duration_sec") if info.get("timeline_duration_sec") is not None else duration
        timeline_duration = float(timeline_duration_raw or 0.0)
        if kind == "ltx":
            offset_sec += max(0.0, float(timeline_duration))
            continue
        source_chunks = _smartblog_avatar_frame_audio_chunks(dict(info or {}))
        cut_inserts = _smartblog_cut_inserts_for_segment(dict(info or {}))
        if source_chunks and (len(source_chunks) > 1 or cut_inserts):
            for chunk_pos, chunk in enumerate(source_chunks):
                chunk_offset = float(chunk.get("offset_sec") or 0.0)
                chunk_duration = float(chunk.get("duration_sec") or 0.0)
                if chunk_duration <= 0.0:
                    alignment = chunk.get("normalized_alignment") if isinstance(chunk.get("normalized_alignment"), dict) else chunk.get("alignment")
                    if isinstance(alignment, dict):
                        ends = alignment.get("character_end_times_seconds")
                        if isinstance(ends, (list, tuple)) and ends:
                            try:
                                chunk_duration = float(max(float(x) for x in ends if x is not None))
                            except Exception:
                                chunk_duration = 0.0
                if chunk_duration <= 0.0 and int(chunk_pos) + 1 < len(source_chunks):
                    chunk_duration = max(0.0, float(source_chunks[int(chunk_pos) + 1].get("offset_sec") or 0.0) - chunk_offset)
                if chunk_duration <= 0.0:
                    chunk_duration = max(0.0, duration - chunk_offset)
                shift = _smartblog_cut_insert_shift_before_time(dict(info or {}), time_sec=float(chunk_offset))
                try:
                    chunk_index = int(chunk.get("source_audio_index", chunk.get("index", file_index)))
                except Exception:
                    chunk_index = int(file_index)
                chunk_start = float(offset_sec + chunk_offset + shift)
                chunks.append(
                    RenderSubtitleChunk(
                        index=int(chunk_index),
                        text=str(chunk.get("text") or ""),
                        start_sec=float(chunk_start),
                        end_sec=float(chunk_start + max(0.0, chunk_duration)),
                        alignment_offset_sec=float(chunk_start + before_sec - core_start_sec),
                        alignment=chunk.get("alignment") if isinstance(chunk.get("alignment"), dict) else None,
                        normalized_alignment=(
                            chunk.get("normalized_alignment") if isinstance(chunk.get("normalized_alignment"), dict) else None
                        ),
                    )
                )
            offset_sec += max(0.0, float(timeline_duration))
            continue
        try:
            chunk_index = int(entry.get("index"))
        except Exception:
            chunk_index = int(info.get("index") or pos)
        chunks.append(
            RenderSubtitleChunk(
                index=int(chunk_index),
                text=_smartblog_audio_entry_text(entry),
                start_sec=float(offset_sec),
                end_sec=float(offset_sec + max(0.0, duration)),
                alignment_offset_sec=float(offset_sec + before_sec - core_start_sec),
                alignment=_smartblog_audio_entry_alignment(entry, normalized=False),
                normalized_alignment=_smartblog_audio_entry_alignment(entry, normalized=True),
            )
        )
        offset_sec += max(0.0, float(timeline_duration))
    return chunks


def _smartblog_render_subtitle_chunks_json(segment_infos: list[dict[str, Any]]) -> str:
    chunks = _smartblog_subtitle_chunks_from_segments(list(segment_infos or []))
    if not chunks:
        return ""
    payload: list[dict[str, Any]] = []
    for chunk in chunks:
        payload.append(
            {
                "index": int(chunk.index),
                "text": str(chunk.text or ""),
                "start_sec": float(chunk.start_sec),
                "end_sec": float(chunk.end_sec),
                "alignment_offset_sec": float(chunk.alignment_offset_sec),
                "alignment": dict(chunk.alignment) if isinstance(chunk.alignment, dict) else None,
                "normalized_alignment": (
                    dict(chunk.normalized_alignment) if isinstance(chunk.normalized_alignment, dict) else None
                ),
            }
        )
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        logging.warning("SmartBlog render subtitles finalizer payload skipped: chunks=%d err=%s", len(chunks), e)
        return ""


def _smartblog_render_subtitle_chunks_json_from_audio_entries(
    audio_entries: list[dict[str, Any]],
    *,
    total_duration_sec: float = 0.0,
) -> str:
    entries = [dict(entry or {}) for entry in list(audio_entries or []) if isinstance(entry, dict)]
    if not entries:
        return ""
    segment_infos: list[dict[str, Any]] = []
    remaining_duration = float(max(0.0, float(total_duration_sec or 0.0)))
    for idx, entry in enumerate(entries):
        duration = float(_smartblog_audio_entry_alignment_duration_sec(entry))
        if duration <= 0.0:
            for key in (
                "duration_sec",
                "duration_seconds",
                "duration",
                "audio_duration_sec",
                "audioDurationSec",
                "audio_duration_seconds",
                "audioDurationSeconds",
            ):
                try:
                    duration = float(entry.get(key) or 0.0)
                except Exception:
                    duration = 0.0
                if duration > 0.0:
                    break
        if duration <= 0.0 and len(entries) == 1 and remaining_duration > 0.0:
            duration = float(remaining_duration)
        if duration <= 0.0:
            continue
        remaining_duration = max(0.0, float(remaining_duration) - float(duration))
        segment_infos.append(
            {
                "kind": "avatar",
                "index": int(entry.get("index") if entry.get("index") is not None else idx),
                "audio_entry": dict(entry),
                "target_duration_sec": float(duration),
                "duration_sec": float(duration),
                "timeline_duration_sec": float(duration),
                "sample_rate": 16000,
                "boundary_before_samples": 0,
                "boundary_core_start_samples": 0,
            }
        )
    if not segment_infos:
        return ""
    return _smartblog_render_subtitle_chunks_json(segment_infos)


def _smartblog_subtitle_chunk_from_segment(info: dict[str, Any], *, fallback_index: int) -> RenderSubtitleChunk:
    entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
    duration = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
    sample_rate = int(info.get("sample_rate") or 16000)
    before_sec = float(int(info.get("boundary_before_samples") or 0)) / float(max(1, int(sample_rate)))
    core_start_sec = float(int(info.get("boundary_core_start_samples") or 0)) / float(max(1, int(sample_rate)))
    try:
        chunk_index = int(entry.get("index"))
    except Exception:
        chunk_index = int(info.get("index") or fallback_index)
    return RenderSubtitleChunk(
        index=int(chunk_index),
        text=_smartblog_audio_entry_text(entry),
        start_sec=0.0,
        end_sec=max(0.0, float(duration)),
        alignment_offset_sec=float(before_sec - core_start_sec),
        alignment=_smartblog_audio_entry_alignment(entry, normalized=False),
        normalized_alignment=_smartblog_audio_entry_alignment(entry, normalized=True),
    )


def _smartblog_maybe_burn_render_subtitles(
    *,
    video_path: str,
    out_path: str,
    ass_path: str,
    segment_infos: list[dict[str, Any]],
    width: int,
    height: int,
    enabled: bool = True,
) -> str:
    if not bool(enabled):
        return str(video_path)
    chunks = _smartblog_subtitle_chunks_from_segments(list(segment_infos or []))
    if not chunks:
        return str(video_path)
    subtitle_start_sec = min(float(chunk.start_sec) for chunk in chunks)
    subtitle_end_sec = max(float(chunk.end_sec) for chunk in chunks)
    video_duration = 0.0
    try:
        video_duration = float(video_duration_sec(str(video_path)) or 0.0)
    except Exception:
        video_duration = 0.0
    block_count = int(
        write_render_subtitles_ass(
            chunks,
            out_path=str(ass_path),
            width=int(width),
            height=int(height),
        )
    )
    if block_count <= 0:
        logging.warning("SmartBlog render subtitles skipped: reason=no_subtitle_blocks")
        return str(video_path)
    burned = burn_ass_subtitles(input_path=str(video_path), ass_path=str(ass_path), output_path=str(out_path))
    logging.warning(
        "SmartBlog render subtitles burned: blocks=%d chunks=%d timeline=%.3f..%.3fs video=%.3fs delta=%.3fs src=%s out=%s ass=%s",
        int(block_count),
        int(len(chunks)),
        float(subtitle_start_sec),
        float(subtitle_end_sec),
        float(video_duration),
        float(video_duration - subtitle_end_sec) if video_duration > 0.0 else 0.0,
        os.path.basename(str(video_path)),
        os.path.basename(str(burned)),
        os.path.basename(str(ass_path)),
    )
    return str(burned)


def _smartblog_render_subtitle_stage() -> str:
    value = str(os.getenv("SMARTBLOG_RENDER_SUBTITLE_STAGE", "segment") or "segment").strip().lower()
    if value in {"0", "off", "false", "none", "disabled"}:
        return "off"
    if value in {"final", "post", "postprocess"}:
        return "final"
    return "segment"


def _smartblog_maybe_burn_render_segment_subtitles(
    *,
    video_path: str,
    out_path: str,
    ass_path: str,
    segment_info: dict[str, Any],
    segment_index: int,
    width: int,
    height: int,
    enabled: bool = True,
) -> str:
    if not bool(enabled):
        return str(video_path)
    if _smartblog_render_subtitle_stage() != "segment":
        return str(video_path)
    chunk = _smartblog_subtitle_chunk_from_segment(dict(segment_info or {}), fallback_index=int(segment_index))
    block_count = int(
        write_render_subtitles_ass(
            [chunk],
            out_path=str(ass_path),
            width=int(width),
            height=int(height),
        )
    )
    if block_count <= 0:
        logging.warning(
            "SmartBlog render segment subtitles skipped: segment=%d reason=no_subtitle_blocks",
            int(segment_index),
        )
        return str(video_path)
    burned = burn_ass_subtitles(input_path=str(video_path), ass_path=str(ass_path), output_path=str(out_path))
    logging.warning(
        "SmartBlog render segment subtitles burned: segment=%d blocks=%d src=%s out=%s",
        int(segment_index),
        int(block_count),
        os.path.basename(str(video_path)),
        os.path.basename(str(burned)),
    )
    return str(burned)


def _smartblog_maybe_burn_render_watermark(
    *,
    video_path: str,
    out_path: str,
    watermark_text: str,
    run_dir: str,
    width: int,
    height: int,
) -> str:
    text = str(watermark_text or "").strip()
    if not text:
        return str(video_path)
    burned = burn_watermark_video(
        input_path=str(video_path),
        output_path=str(out_path),
        text=str(text),
        width=int(width),
        height=int(height),
        work_dir=str(run_dir),
        env_prefixes=("SMARTBLOG",),
    )
    logging.warning(
        "SmartBlog render watermark burned: chars=%d src=%s out=%s",
        int(len(str(text))),
        os.path.basename(str(video_path)),
        os.path.basename(str(burned)),
    )
    return str(burned)


def _smartblog_postprocess_render_segment_for_concat(
    *,
    src_path: str,
    run_dir: str,
    segment_info: dict[str, Any],
    segment_index: int,
    segment_count: int,
    width: int,
    height: int,
    burn_subtitles: bool = True,
) -> str:
    current = str(src_path)
    has_transition_start = bool((segment_info or {}).get("avatar_transition_in"))
    has_transition_end = bool((segment_info or {}).get("avatar_transition_out"))
    if bool(has_transition_start or has_transition_end) and bool(_env_flag("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR", "1")):
        current = _smartblog_apply_segment_transition_blur(
            src_path=str(current),
            out_path=os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_transition.mp4"),
            has_start=bool(has_transition_start),
            has_end=bool(has_transition_end),
            width=int(width),
            height=int(height),
        )
    current = _smartblog_maybe_burn_render_segment_subtitles(
        video_path=str(current),
        out_path=os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_subtitled.mp4"),
        ass_path=os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}.ass"),
        segment_info=dict(segment_info or {}),
        segment_index=int(segment_index),
        width=int(width),
        height=int(height),
        enabled=bool(burn_subtitles),
    )
    if not os.path.exists(str(current)):
        raise RuntimeError(f"render_video segment postprocess produced no output: {segment_index}")
    return str(current)


def _smartblog_wav_pcm16_mono(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = max(1, int(wf.getnchannels() or 1))
        sample_width = int(wf.getsampwidth() or 0)
        sample_rate = max(1, int(wf.getframerate() or 16000))
        frames = int(wf.getnframes() or 0)
        raw = wf.readframes(max(0, frames))
    if int(sample_width) != 2:
        raise RuntimeError(f"expected PCM16 WAV: {wav_path}")
    arr = np.frombuffer(raw, dtype="<i2")
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int16), int(sample_rate)
    if int(channels) > 1:
        usable = int(arr.size // int(channels)) * int(channels)
        if usable <= 0:
            return np.zeros(0, dtype=np.int16), int(sample_rate)
        arr_i32 = arr[:usable].reshape((-1, int(channels))).astype(np.int32)
        arr = np.clip(np.rint(arr_i32.mean(axis=1)), -32768, 32767).astype(np.int16)
    else:
        arr = arr.astype(np.int16, copy=False)
    return arr, int(sample_rate)


def _smartblog_write_wav_pcm16_mono(wav_path: str, pcm: np.ndarray, *, sample_rate: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(str(wav_path))) or ".", exist_ok=True)
    arr = np.asarray(pcm, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(max(1, int(sample_rate))))
        wf.writeframes(arr.astype("<i2", copy=False).tobytes())
    return str(wav_path)


def _smartblog_apply_lipsync_tail_close(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    duration_ms: float,
    floor: float,
) -> np.ndarray:
    arr = np.asarray(pcm, dtype=np.int16)
    if arr.size <= 0:
        return arr
    try:
        samples = int(round(float(sample_rate) * float(duration_ms) / 1000.0))
    except Exception:
        samples = 0
    samples = int(max(0, min(int(samples), int(arr.size))))
    if samples <= 1:
        return arr
    floor_f = float(max(0.0, min(1.0, float(floor))))
    out = arr.astype(np.float32, copy=True)
    env = np.linspace(1.0, floor_f, int(samples), dtype=np.float32)
    out[-int(samples) :] *= env
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def _smartblog_liveaudio_queue_chunk_samples(sample_rate: int) -> int:
    runtime_sample_rate = int(max(1, _safe_int_env("WORKER_AUDIO_SAMPLE_RATE", 16000)))
    locked_samples = int(max(1, int(LOCKED_MICRO_CHUNK_SAMPLES)))
    if int(sample_rate or 0) == int(runtime_sample_rate):
        return int(locked_samples)
    return int(max(1, round(float(locked_samples) * float(max(1, int(sample_rate))) / float(runtime_sample_rate))))


def _smartblog_render_onepass_block_frames(infer_frames: int) -> int:
    clip_frames = int(max(1, int(infer_frames or 1)))
    try:
        wan_blocks = int(_safe_int_env("SMARTBLOG_WAN_NUM_FRAMES_PER_BLOCK", 0))
    except Exception:
        wan_blocks = 0
    block_frames = int(wan_blocks) * 4 if int(wan_blocks) > 0 else 0
    if int(block_frames) <= 0:
        block_frames = int(clip_frames)
    return int(max(1, min(int(clip_frames), int(block_frames))))


def _smartblog_render_onepass_max_conditioning_frames(infer_frames: int) -> int:
    clip_frames = int(max(1, int(infer_frames or 1)))
    block_frames = int(_smartblog_render_onepass_block_frames(int(clip_frames)))
    configured = int(
        max(
            clip_frames,
            _safe_int_env("SMARTBLOG_RENDER_ONEPASS_MAX_CONDITIONING_FRAMES", clip_frames * 2),
        )
    )
    aligned = int(math.ceil(float(configured) / float(block_frames)) * int(block_frames))
    return int(max(clip_frames, min(512, int(aligned))))


def _smartblog_render_onepass_audio_ranges(
    *,
    total_samples: int,
    sample_rate: int,
    fps: int,
    infer_frames: int,
    boundary_preroll_frames: int = 0,
) -> list[tuple[int, int, int, int, int, int]]:
    total_samples_i = int(max(0, int(total_samples or 0)))
    if total_samples_i <= 0:
        return []
    sample_rate_i = int(max(1, int(sample_rate or 16000)))
    fps_i = int(max(1, int(fps or 16)))
    infer_frames_i = int(max(1, int(infer_frames or 1)))
    block_frames_i = int(_smartblog_render_onepass_block_frames(int(infer_frames_i)))
    preroll_frames_i = int(max(0, min(int(block_frames_i), int(boundary_preroll_frames or 0))))
    total_frames_i = int(
        max(
            1,
            round(float(total_samples_i) * float(fps_i) / float(sample_rate_i)),
        )
    )
    max_conditioning_frames = int(_smartblog_render_onepass_max_conditioning_frames(int(infer_frames_i)))
    min_tail_frames_i = int(
        max(
            0,
            _safe_int_env("SMARTBLOG_RENDER_ONEPASS_MIN_TAIL_FRAMES", int(infer_frames_i)),
        )
    )
    tail_visible_start_frames = int(preroll_frames_i if int(total_frames_i) > int(max_conditioning_frames) else 0)
    max_tail_source_frames = int(max(1, int(max_conditioning_frames) - int(tail_visible_start_frames)))
    min_tail_frames_i = int(min(int(max_tail_source_frames), int(min_tail_frames_i)))
    frame_ranges: list[tuple[int, int]] = []

    prev_frame = 0
    while int(prev_frame) < int(total_frames_i):
        visible_start_frames = int(preroll_frames_i if int(prev_frame) > 0 else 0)
        max_source_frames = int(max(1, int(max_conditioning_frames) - int(visible_start_frames)))
        remaining_frames = int(total_frames_i) - int(prev_frame)
        if int(remaining_frames) <= int(max_source_frames):
            frame_end = int(total_frames_i)
        else:
            frame_end = int(prev_frame) + int(max_source_frames)
        frame_ranges.append((int(prev_frame), int(frame_end)))
        prev_frame = int(frame_end)

    if len(frame_ranges) > 2 and int(min_tail_frames_i) > 0:
        prev_start, prev_end = frame_ranges[-2]
        tail_start, tail_end = frame_ranges[-1]
        tail_frames = int(tail_end) - int(tail_start)
        if 0 < int(tail_frames) < int(min_tail_frames_i):
            need_frames = int(min_tail_frames_i) - int(tail_frames)
            transfer_frames = int(math.ceil(float(need_frames) / float(block_frames_i)) * int(block_frames_i))
            prev_frames = int(prev_end) - int(prev_start)
            # Do not "fix" a short final tail by making the preceding visible
            # range shorter than a full inference window. With fixed 64-frame
            # streaming windows, stealing one 32-frame WAN block creates a
            # visible half-window right before the end and the avatar can jump
            # there. Leave the previous full window intact and let only the
            # tiny final remainder be handled as tail.
            min_prev_visible_frames = int(max(int(infer_frames_i), int(block_frames_i)))
            max_transfer_frames = int(max(0, int(prev_frames) - int(min_prev_visible_frames)))
            transfer_frames = int(min(int(transfer_frames), int(max_transfer_frames)))
            transfer_frames = int(int(transfer_frames) // int(block_frames_i) * int(block_frames_i))
            if int(transfer_frames) > 0:
                boundary = int(tail_start) - int(transfer_frames)
                frame_ranges[-2] = (int(prev_start), int(boundary))
                frame_ranges[-1] = (int(boundary), int(tail_end))

    ranges: list[tuple[int, int, int, int, int, int]] = []
    prev_sample = 0
    for range_idx, (frame_start, frame_end) in enumerate(frame_ranges):
        if int(frame_end) <= int(frame_start):
            continue
        if int(range_idx) == len(frame_ranges) - 1:
            sample_end = int(total_samples_i)
        else:
            sample_end = int(round(float(frame_end) * float(sample_rate_i) / float(fps_i)))
            sample_end = int(max(int(prev_sample) + 1, min(int(total_samples_i), int(sample_end))))
        visible_start_frames = int(preroll_frames_i if int(frame_start) > 0 else 0)
        source_frames = int(max(1, int(frame_end) - int(frame_start)))
        conditioning_frames = int(
            max(
                int(infer_frames_i),
                int(
                    math.ceil(float(source_frames + int(visible_start_frames)) / float(block_frames_i))
                    * int(block_frames_i)
                ),
            )
        )
        conditioning_frames = int(max(int(source_frames) + int(visible_start_frames), int(conditioning_frames)))
        if int(sample_end) > int(prev_sample):
            ranges.append(
                (
                    int(prev_sample),
                    int(sample_end),
                    int(frame_start),
                    int(frame_end),
                    int(conditioning_frames),
                    int(visible_start_frames),
                )
            )
        prev_sample = int(sample_end)
    if not ranges:
        conditioning_frames = int(
            max(
                int(infer_frames_i),
                int(math.ceil(float(total_frames_i) / float(block_frames_i)) * int(block_frames_i)),
            )
        )
        return [(0, int(total_samples_i), 0, int(total_frames_i), int(conditioning_frames), 0)]
    return ranges


def _smartblog_audible_bounds_pcm16(pcm: np.ndarray, *, silence_db: float) -> tuple[int, int]:
    arr = np.asarray(pcm, dtype=np.int16)
    total = int(arr.size)
    if total <= 0:
        return 0, 0
    try:
        threshold = int(max(1.0, min(32767.0, 32767.0 * (10.0 ** (float(silence_db) / 20.0)))))
    except Exception:
        threshold = 104
    idx = np.flatnonzero(np.abs(arr.astype(np.int32)) > int(threshold))
    if idx.size <= 0:
        return 0, int(total)
    return int(idx[0]), int(idx[-1]) + 1


def _smartblog_audio_split_meta(info: dict[str, Any] | None) -> tuple[int, int, str] | None:
    entry = (info or {}).get("audio_entry") if isinstance((info or {}).get("audio_entry"), dict) else {}
    try:
        split_total = int(entry.get("_smartblog_audio_split_total") or 0)
        split_index = int(entry.get("_smartblog_audio_split_index") or 0)
    except Exception:
        return None
    if int(split_total) <= 1:
        return None
    source = str(
        entry.get("_smartblog_audio_split_source_index")
        or entry.get("_smartblog_audio_split_source_key")
        or entry.get("source_audio_index")
        or entry.get("index")
        or ""
    ).strip()
    if not source:
        source = str((info or {}).get("index") or "")
    return int(split_index), int(split_total), str(source)


def _smartblog_is_seamless_audio_split_boundary(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_meta = _smartblog_audio_split_meta(left)
    right_meta = _smartblog_audio_split_meta(right)
    if left_meta is None or right_meta is None:
        return False
    left_index, left_total, left_source = left_meta
    right_index, right_total, right_source = right_meta
    if int(left_total) != int(right_total) or str(left_source) != str(right_source):
        return False
    if int(right_index) != int(left_index) + 1:
        return False
    left_avatar = str((left or {}).get("avatar_path") or "").strip()
    right_avatar = str((right or {}).get("avatar_path") or "").strip()
    if left_avatar and right_avatar and left_avatar != right_avatar:
        return False
    return True


def _smartblog_prepare_audio_chunk_boundaries(
    *,
    segment_infos: list[dict[str, Any]],
    run_dir: str,
) -> None:
    if len(segment_infos) <= 1:
        return
    if not _env_flag("SMARTBLOG_RENDER_AUDIO_BOUNDARY_SPLIT", "1"):
        return
    silence_db = float(_safe_float_env("SMARTBLOG_RENDER_BOUNDARY_SILENCE_DB", -50.0))
    keep_sec = float(max(0.0, min(1.0, _safe_float_env("SMARTBLOG_RENDER_BOUNDARY_KEEP_SEC", 0.08))))
    initial_lead_sec = float(max(0.0, min(1.0, _safe_float_env("SMARTBLOG_RENDER_AUDIO_BOUNDARY_LEAD_SEC", 0.0))))
    extra_gap_sec = float(max(0.0, min(3.0, _safe_float_env("SMARTBLOG_RENDER_AUDIO_BOUNDARY_GAP_SEC", 0.0))))
    final_tail_sec = float(max(0.0, min(3.0, _safe_float_env("SMARTBLOG_RENDER_FINAL_TAIL_SEC", 0.04))))
    trim_audible_audio = _env_flag("SMARTBLOG_RENDER_AUDIO_BOUNDARY_TRIM_AUDIBLE", "0")

    loaded: list[dict[str, Any]] = []
    for pos, info in enumerate(segment_infos):
        pcm, sample_rate = _smartblog_wav_pcm16_mono(str(info.get("audio_wav_path") or ""))
        raw_total = int(pcm.size)
        if raw_total <= 0:
            raise RuntimeError(f"render_video audio segment {int(pos) + 1} is empty after decode")
        total = int(pcm.size)
        if total <= 0:
            raise RuntimeError(f"render_video audio segment {int(pos) + 1} is empty after decode")
        first, last = _smartblog_audible_bounds_pcm16(pcm, silence_db=float(silence_db))
        keep_samples = int(round(float(keep_sec) * float(sample_rate)))
        core_start = int(max(0, int(first) - int(keep_samples)))
        core_end = int(min(int(total), int(last) + int(keep_samples)))
        split_meta = _smartblog_audio_split_meta(info)
        if split_meta is not None:
            split_index, split_total, _source = split_meta
            # Technical splits are only an inference/memory boundary. Keep the
            # original PCM continuous at their internal edges; otherwise a
            # long single TTS file gets an audible pause close to the end when
            # the remainder segment is short.
            if int(split_index) > 0:
                core_start = 0
            if int(split_index) < int(split_total) - 1:
                core_end = int(total)
        if core_end <= core_start:
            core_start = 0
            core_end = int(total)
        loaded.append(
            {
                "pcm": pcm,
                "sample_rate": int(sample_rate),
                "core_start": int(core_start),
                "core_end": int(core_end),
                "lead_silence": int(max(0, int(core_start))),
                "tail_silence": int(max(0, int(total) - int(core_end))),
                "before": 0,
                "after": 0,
                "orig_samples": int(raw_total),
                "target_samples": int(total),
            }
        )

    loaded[0]["before"] = int(
        min(
            int(loaded[0]["lead_silence"]),
            int(round(float(initial_lead_sec) * float(int(loaded[0]["sample_rate"])))),
        )
    )
    seamless_boundaries = 0
    for pos in range(0, len(loaded) - 1):
        if _smartblog_is_seamless_audio_split_boundary(segment_infos[pos], segment_infos[pos + 1]):
            loaded[pos]["after"] = 0
            loaded[pos + 1]["before"] = 0
            seamless_boundaries += 1
            continue
        rate = int(loaded[pos]["sample_rate"])
        next_rate = int(loaded[pos + 1]["sample_rate"])
        if int(rate) != int(next_rate):
            logging.warning(
                "SmartBlog render audio boundary rate mismatch: segment=%d rate=%d next_rate=%d",
                int(pos + 1),
                int(rate),
                int(next_rate),
            )
        extra_samples = int(round(float(extra_gap_sec) * float(rate)))
        # Do not feed detected TTS silence into LiveAvatar. The model tends to
        # animate residual mouth/face motion on silence, which creates visible
        # jumps right before the next avatar/photo boundary. Keep only an
        # explicit configured gap; by default cuts happen at the audible
        # boundary and visual transitions are handled by the video layer.
        total_gap = int(extra_samples)
        loaded[pos]["after"] = int(total_gap // 2)
        loaded[pos + 1]["before"] = int(total_gap - int(loaded[pos]["after"]))

    loaded[-1]["after"] = int(
        min(
            int(loaded[-1]["tail_silence"]),
            int(round(float(final_tail_sec) * float(int(loaded[-1]["sample_rate"])))),
        )
    )

    total_samples = 0
    for pos, info in enumerate(segment_infos):
        file_index = _smartblog_segment_file_index(dict(info or {}), fallback=int(pos))
        item = loaded[pos]
        sample_rate = int(item["sample_rate"])
        before = int(max(0, int(item["before"])))
        after = int(max(0, int(item["after"])))
        core = np.asarray(item["pcm"][int(item["core_start"]) : int(item["core_end"])], dtype=np.int16)
        parts: list[np.ndarray] = []
        if before > 0:
            parts.append(np.zeros(before, dtype=np.int16))
        parts.append(core)
        if after > 0:
            parts.append(np.zeros(after, dtype=np.int16))
        processed = np.concatenate(parts).astype(np.int16, copy=False) if parts else np.zeros(0, dtype=np.int16)
        if processed.size <= 0:
            raise RuntimeError(f"render_video audio segment {int(pos) + 1} became empty after boundary split")
        original_path = str(info.get("audio_wav_path") or "")
        original_pcm = np.asarray(item["pcm"], dtype=np.int16)
        # The audible timeline must remain exact. Boundary trimming is only a
        # lipsync input treatment; trimming the final mixed audio can silently
        # remove soft words at chunk edges while the remaining A/V stays in sync.
        lipsync_pcm = np.zeros_like(original_pcm)
        if int(item["core_end"]) > int(item["core_start"]):
            lipsync_pcm[int(item["core_start"]) : int(item["core_end"])] = original_pcm[
                int(item["core_start"]) : int(item["core_end"])
            ]
        lipsync_path = os.path.join(str(run_dir), f"audio_segment_{int(file_index):03d}_boundary_lipsync_16k.wav")
        _smartblog_write_wav_pcm16_mono(str(lipsync_path), lipsync_pcm, sample_rate=int(sample_rate))
        info["audio_wav_path_original"] = str(original_path)
        info["lipsync_audio_wav_path"] = str(lipsync_path)
        info["lipsync_alignment_offset_sec"] = 0.0
        if bool(trim_audible_audio):
            processed_path = os.path.join(str(run_dir), f"audio_segment_{int(file_index):03d}_boundary_16k.wav")
            _smartblog_write_wav_pcm16_mono(processed_path, processed, sample_rate=int(sample_rate))
            info["audio_wav_path"] = str(processed_path)
            info["target_samples"] = int(processed.size)
            info["target_duration_sec"] = float(processed.size) / float(max(1, int(sample_rate)))
            info["duration_sec"] = float(processed.size) / float(max(1, int(sample_rate)))
        else:
            info["audio_wav_path"] = str(original_path)
            info["target_samples"] = int(item["orig_samples"])
            info["target_duration_sec"] = float(item["orig_samples"]) / float(max(1, int(sample_rate)))
            info["duration_sec"] = float(item["orig_samples"]) / float(max(1, int(sample_rate)))
        info["sample_rate"] = int(sample_rate)
        info["boundary_before_samples"] = int(before)
        info["boundary_after_samples"] = int(after)
        info["boundary_orig_samples"] = int(item["orig_samples"])
        info["boundary_core_start_samples"] = int(item["core_start"])
        info["boundary_core_end_samples"] = int(item["core_end"])
        info["boundary_lipsync_path"] = str(lipsync_path)
        total_samples += int(info["target_samples"])
        logging.warning(
            "SmartBlog render audio boundary segment: index=%d orig=%.3fs out=%.3fs lipsync=%.3fs before=%.3fs after=%.3fs core=%.3fs audible_trim=%d avatar=%s",
            int(file_index + 1),
            float(item["orig_samples"]) / float(max(1, int(sample_rate))),
            float(int(info["target_samples"])) / float(max(1, int(sample_rate))),
            float(lipsync_pcm.size) / float(max(1, int(sample_rate))),
            float(before) / float(max(1, int(sample_rate))),
            float(after) / float(max(1, int(sample_rate))),
            float(max(0, int(item["core_end"]) - int(item["core_start"]))) / float(max(1, int(sample_rate))),
            1 if bool(trim_audible_audio) else 0,
            os.path.basename(str(info.get("avatar_path") or "")) or "-",
        )

    logging.warning(
        "SmartBlog render audio boundary split: segments=%d total=%.3fs extra_gap=%.3fs keep=%.3fs lead=%.3fs final_tail=%.3fs silence_db=%.1f",
        int(len(segment_infos)),
        float(total_samples) / float(max(1, int(loaded[0]["sample_rate"]))),
        float(extra_gap_sec),
        float(keep_sec),
        float(initial_lead_sec),
        float(final_tail_sec),
        float(silence_db),
    )


def _smartblog_render_avatar_audio_preprocess_enabled() -> bool:
    return bool(_env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_PREPROCESS", "0"))


def _smartblog_render_avatar_audio_preprocess_preset() -> str:
    preset = str(os.getenv("SMARTBLOG_RENDER_AVATAR_AUDIO_PRESET", "clear") or "clear").strip().lower()
    if preset not in {"safe", "clear", "less_jitter", "speech", "vocal", "auto"}:
        preset = "clear"
    return preset


def _smartblog_detect_avatar_audio_preset(wav_path: str, *, has_alignment: bool = False) -> str:
    path_s = str(wav_path or "").strip()
    try:
        pcm, sample_rate = _smartblog_wav_pcm16_mono(path_s)
    except Exception as e:
        logging.warning("SmartBlog render avatar audio auto preset failed: file=%s err=%s", os.path.basename(path_s), str(e))
        return "speech"
    if int(pcm.size) <= 0 or int(sample_rate) <= 0:
        return "speech"

    max_sec = float(max(5.0, min(180.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_AUTO_ANALYSIS_SEC", 60.0))))
    max_samples = int(round(float(max_sec) * float(sample_rate)))
    arr = np.asarray(pcm[:max_samples], dtype=np.float32) / 32768.0
    if int(arr.size) < int(sample_rate):
        return "speech"

    frame_n = int(max(256, round(float(sample_rate) * 0.050)))
    hop_n = int(max(128, round(float(sample_rate) * 0.025)))
    if int(arr.size) < frame_n:
        return "speech"
    frame_count = 1 + int((int(arr.size) - int(frame_n)) // int(hop_n))
    if frame_count <= 2:
        return "speech"

    starts = np.arange(0, int(frame_count) * int(hop_n), int(hop_n), dtype=np.int64)
    frames = np.stack([arr[int(start) : int(start) + int(frame_n)] for start in starts], axis=0)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    p95 = float(np.percentile(rms, 95))
    active_threshold = float(max(0.004, p95 * 0.09))
    active = rms >= active_threshold
    silence_ratio = float(1.0 - (float(np.mean(active)) if active.size else 0.0))
    active_rms = rms[active] if np.any(active) else rms
    energy_cv = float(np.std(active_rms) / (float(np.mean(active_rms)) + 1e-6))

    spectral_flatness = 0.0
    centroid_hz = 0.0
    high_ratio = 0.0
    try:
        max_fft_frames = int(max(1, min(1200, int(frames.shape[0]))))
        fft_frames = frames[:max_fft_frames] * np.hanning(int(frame_n)).astype(np.float32)
        power = np.square(np.abs(np.fft.rfft(fft_frames, axis=1))).astype(np.float32) + 1e-12
        freqs = np.fft.rfftfreq(int(frame_n), d=1.0 / float(sample_rate)).astype(np.float32)
        band = (freqs >= 80.0) & (freqs <= 7800.0)
        high = (freqs >= 2800.0) & (freqs <= 7800.0)
        band_power = power[:, band]
        band_sum = np.sum(band_power, axis=1) + 1e-12
        spectral_flatness = float(np.mean(np.exp(np.mean(np.log(band_power), axis=1)) / band_sum * float(band_power.shape[1])))
        centroid_hz = float(np.mean(np.sum(power[:, band] * freqs[band], axis=1) / band_sum))
        high_ratio = float(np.mean(np.sum(power[:, high], axis=1) / band_sum))
    except Exception:
        spectral_flatness = 0.0
        centroid_hz = 0.0
        high_ratio = 0.0

    duration_sec = float(arr.size) / float(max(1, int(sample_rate)))
    score = 0.0
    if silence_ratio < 0.08:
        score += 1.0
    elif silence_ratio < 0.16:
        score += 0.45
    if energy_cv < 0.60:
        score += 0.75
    elif energy_cv < 0.85:
        score += 0.35
    if spectral_flatness > 0.18:
        score += 1.0
    elif spectral_flatness > 0.11:
        score += 0.45
    if high_ratio > 0.30:
        score += 0.55
    if centroid_hz > 2300.0:
        score += 0.35
    if duration_sec >= 8.0 and silence_ratio < 0.14 and spectral_flatness > 0.10:
        score += 0.45

    music_score = 0.0
    if duration_sec >= 12.0:
        music_score += 0.45
    if silence_ratio < 0.45:
        music_score += 0.65
    elif silence_ratio < 0.56:
        music_score += 0.25
    if energy_cv < 0.75:
        music_score += 0.65
    elif energy_cv < 0.95:
        music_score += 0.25
    if spectral_flatness > 0.035:
        music_score += 0.45
    elif spectral_flatness > 0.020:
        music_score += 0.20
    if high_ratio > 0.08:
        music_score += 0.35
    if centroid_hz > 800.0:
        music_score += 0.25
    if duration_sec >= 20.0 and silence_ratio < 0.45 and energy_cv < 0.80:
        music_score += 0.30
    if not bool(has_alignment):
        music_score += 0.55

    threshold = float(max(0.5, min(5.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_AUTO_SONG_THRESHOLD", 2.25))))
    music_threshold = float(
        max(
            0.5,
            min(
                5.0,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_AUTO_MUSIC_THRESHOLD",
                    2.05 if not bool(has_alignment) else 3.75,
                ),
            ),
        )
    )
    resolved = "vocal" if score >= threshold or music_score >= music_threshold else "speech"
    logging.warning(
        "SmartBlog render avatar audio auto preset: file=%s resolved=%s score=%.2f threshold=%.2f music_score=%.2f music_threshold=%.2f has_alignment=%d silence=%.3f energy_cv=%.3f flatness=%.3f centroid=%.0f high_ratio=%.3f duration=%.2f",
        os.path.basename(path_s),
        str(resolved),
        float(score),
        float(threshold),
        float(music_score),
        float(music_threshold),
        int(bool(has_alignment)),
        float(silence_ratio),
        float(energy_cv),
        float(spectral_flatness),
        float(centroid_hz),
        float(high_ratio),
        float(duration_sec),
    )
    return str(resolved)


def _smartblog_detect_avatar_audio_voice_shape(wav_path: str) -> str:
    forced = str(os.getenv("SMARTBLOG_RENDER_AVATAR_AUDIO_VOICE_SHAPE", "auto") or "auto").strip().lower()
    if forced in {"male", "female", "neutral"}:
        return forced
    path_s = str(wav_path or "").strip()
    try:
        pcm, sample_rate = _smartblog_wav_pcm16_mono(path_s)
    except Exception as e:
        logging.warning("SmartBlog render avatar audio voice-shape failed: file=%s err=%s", os.path.basename(path_s), str(e))
        return "neutral"
    if int(pcm.size) <= 0 or int(sample_rate) <= 0:
        return "neutral"

    max_sec = float(max(3.0, min(60.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_VOICE_SHAPE_SEC", 30.0))))
    arr = np.asarray(pcm[: int(round(float(max_sec) * float(sample_rate)))], dtype=np.float32) / 32768.0
    if int(arr.size) < int(sample_rate):
        return "neutral"

    frame_n = int(max(320, round(float(sample_rate) * 0.040)))
    hop_n = int(max(160, round(float(sample_rate) * 0.030)))
    frame_count = 1 + int((int(arr.size) - int(frame_n)) // int(hop_n))
    if frame_count <= 4:
        return "neutral"
    starts = np.arange(0, int(frame_count) * int(hop_n), int(hop_n), dtype=np.int64)
    frames = np.stack([arr[int(start) : int(start) + int(frame_n)] for start in starts], axis=0)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    active_threshold = float(max(0.004, float(np.percentile(rms, 90)) * 0.12))
    active_idx = np.where(rms >= active_threshold)[0]
    if int(active_idx.size) <= 4:
        return "neutral"

    f0_values: list[float] = []
    max_frames = int(max(12, min(240, _safe_int_env("SMARTBLOG_RENDER_AVATAR_AUDIO_VOICE_SHAPE_MAX_FRAMES", 120))))
    selected = active_idx[:: max(1, int(math.ceil(float(active_idx.size) / float(max_frames))))]
    lag_min = int(max(1, round(float(sample_rate) / 350.0)))
    lag_max = int(max(lag_min + 1, round(float(sample_rate) / 85.0)))
    window = np.hanning(int(frame_n)).astype(np.float32)
    for frame_i in selected:
        frame = np.asarray(frames[int(frame_i)], dtype=np.float32)
        frame = (frame - float(np.mean(frame))) * window
        energy = float(np.dot(frame, frame))
        if energy <= 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[int(frame_n) - 1 :]
        if int(corr.size) <= lag_max:
            continue
        segment = corr[lag_min:lag_max]
        if int(segment.size) <= 0:
            continue
        peak_rel = int(np.argmax(segment))
        peak = float(segment[peak_rel])
        confidence = float(peak / (float(corr[0]) + 1e-9))
        if confidence < 0.22:
            continue
        lag = int(lag_min + peak_rel)
        if lag <= 0:
            continue
        f0_values.append(float(sample_rate) / float(lag))

    f0 = float(np.median(f0_values)) if f0_values else 0.0
    shape = "neutral"
    if f0 >= 178.0:
        shape = "female"
    elif 85.0 <= f0 <= 148.0:
        shape = "male"
    logging.warning(
        "SmartBlog render avatar audio voice-shape: file=%s shape=%s f0=%.1f frames=%d active=%d",
        os.path.basename(path_s),
        str(shape),
        float(f0),
        int(len(f0_values)),
        int(active_idx.size),
    )
    return str(shape)


def _smartblog_avatar_audio_shape_float_env(name: str, default: float, *, voice_shape: str) -> float:
    shape_s = str(voice_shape or "neutral").strip().lower()
    name_s = str(name or "").strip()
    if name_s.startswith("SMARTBLOG_RENDER_AVATAR_AUDIO_") and shape_s in {"male", "female"}:
        shape_name = name_s.replace(
            "SMARTBLOG_RENDER_AVATAR_AUDIO_",
            f"SMARTBLOG_RENDER_AVATAR_AUDIO_{shape_s.upper()}_",
            1,
        )
        if str(os.getenv(shape_name, "") or "").strip():
            return _safe_float_env(shape_name, default)
    return _safe_float_env(name_s, default)


def _smartblog_render_avatar_audio_filter(preset: str, *, voice_shape: str = "neutral") -> str:
    preset_s = str(preset or "clear").strip().lower()
    shape_s = str(voice_shape or "neutral").strip().lower()
    if preset_s == "auto":
        preset_s = "speech"
    if preset_s == "speech":
        if shape_s == "female":
            return ",".join(
                [
                    "highpass=f=115",
                    "lowpass=f=7400",
                    "equalizer=f=220:t=q:w=1:g=-4.5",
                    "equalizer=f=750:t=q:w=1:g=-0.3",
                    "equalizer=f=1150:t=q:w=1:g=-0.6",
                    "equalizer=f=1900:t=q:w=1:g=1.2",
                    "equalizer=f=3300:t=q:w=1:g=3.4",
                    "equalizer=f=4800:t=q:w=1:g=2.0",
                    "speechnorm=e=2.4:c=2:t=0.025:r=0.0007:f=0.0009:p=0.90",
                    "deesser=i=0.18",
                    "acompressor=threshold=-22dB:ratio=2.4:attack=2:release=70:makeup=1.2",
                    "alimiter=limit=0.95",
                ]
            )
        if shape_s == "male":
            male_pitch = float(max(1.0, min(1.18, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_MALE_PITCH", 1.095))))
            male_vowel_db = float(
                max(0.0, min(7.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_MALE_VOWEL_DB", 3.0)))
            )
            male_presence_db = float(
                max(0.0, min(10.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_MALE_PRESENCE_DB", 6.4)))
            )
            male_upper_db = float(
                max(0.0, min(8.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_MALE_UPPER_DB", 4.6)))
            )
            return ",".join(
                [
                    f"rubberband=pitch={male_pitch:.3f}:transients=crisp:detector=compound:formant=shifted:pitchq=speed",
                    "highpass=f=125",
                    "lowpass=f=7900",
                    "equalizer=f=190:t=q:w=1:g=-5.5",
                    "equalizer=f=340:t=q:w=1:g=-2.4",
                    f"equalizer=f=950:t=q:w=1:g={male_vowel_db:.2f}",
                    "equalizer=f=1550:t=q:w=1:g=2.8",
                    f"equalizer=f=3250:t=q:w=1:g={male_presence_db:.2f}",
                    f"equalizer=f=5000:t=q:w=1:g={male_upper_db:.2f}",
                    "speechnorm=e=5.2:c=2:t=0.018:r=0.00045:f=0.00055:p=0.96",
                    "deesser=i=0.14",
                    "acompressor=threshold=-26dB:ratio=3.8:attack=1:release=45:makeup=2.8",
                    "volume=1.08",
                    "alimiter=limit=0.97",
                ]
            )
        return ",".join(
            [
                "highpass=f=95",
                "lowpass=f=7800",
                "equalizer=f=220:t=q:w=1:g=-4",
                "equalizer=f=750:t=q:w=1:g=0.8",
                "equalizer=f=1800:t=q:w=1:g=1.6",
                "equalizer=f=3200:t=q:w=1:g=4.2",
                "equalizer=f=4700:t=q:w=1:g=2.4",
                "speechnorm=e=3.2:c=2:t=0.025:r=0.0006:f=0.0008:p=0.92",
                "deesser=i=0.12",
                "acompressor=threshold=-23dB:ratio=2.8:attack=2:release=65:makeup=1.8",
                "alimiter=limit=0.96",
            ]
        )
    if preset_s == "vocal":
        return ",".join(
            [
                "highpass=f=145",
                "lowpass=f=6800",
                "equalizer=f=180:t=q:w=1:g=-7",
                "equalizer=f=320:t=q:w=1:g=-4",
                "equalizer=f=850:t=q:w=1:g=1.5",
                "equalizer=f=1450:t=q:w=1:g=2.5",
                "equalizer=f=2600:t=q:w=1:g=4.5",
                "equalizer=f=3900:t=q:w=1:g=3.0",
                "speechnorm=e=5.5:c=2:t=0.018:r=0.0007:f=0.0007:p=0.96",
                "deesser=i=0.12",
                "acompressor=threshold=-26dB:ratio=3.0:attack=2:release=85:makeup=3",
                "volume=1.12",
                "alimiter=limit=0.96",
            ]
        )
    if preset_s == "safe":
        return ",".join(
            [
                "highpass=f=90",
                "lowpass=f=10000",
                "equalizer=f=250:t=q:w=1:g=-2",
                "equalizer=f=3200:t=q:w=1:g=2",
                "deesser=i=0.15",
                "acompressor=threshold=-22dB:ratio=2.4:attack=8:release=130:makeup=2",
                "volume=1.15",
                "alimiter=limit=0.95",
            ]
        )
    if preset_s == "less_jitter":
        return ",".join(
            [
                "highpass=f=90",
                "equalizer=f=250:t=q:w=1:g=-3",
                "equalizer=f=3000:t=q:w=1:g=1.5",
                "lowpass=f=8500",
                "deesser=i=0.28",
                "acompressor=threshold=-24dB:ratio=2:attack=12:release=180:makeup=1",
                "volume=1.10",
                "alimiter=limit=0.95",
            ]
        )
    return ",".join(
        [
            "highpass=f=110",
            "lowpass=f=9500",
            "equalizer=f=250:t=q:w=1:g=-4",
            "equalizer=f=900:t=q:w=1:g=1.5",
            "equalizer=f=2600:t=q:w=1:g=2.5",
            "equalizer=f=3900:t=q:w=1:g=4.5",
            "deesser=i=0.18",
            "acompressor=threshold=-21dB:ratio=3.2:attack=4:release=90:makeup=3",
            "volume=1.25",
            "alimiter=limit=0.95",
        ]
    )


def _smartblog_is_avatar_audio_vowel(ch: str) -> bool:
    return str(ch or "").lower() in set("aeiouyаеёиоуыэюя")


def _smartblog_is_avatar_audio_letter_or_number(ch: str) -> bool:
    return bool(re.match(r"^[^\W_]$", str(ch or ""), flags=re.UNICODE))


def _smartblog_apply_avatar_audio_letter_dynamics(
    arr: np.ndarray,
    *,
    sample_rate: int,
    alignment: dict[str, Any] | None,
    alignment_offset_sec: float,
    voice_shape: str,
    preset: str,
) -> tuple[np.ndarray, int, int]:
    if not _env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_DYNAMICS", "1"):
        return arr, 0, 0
    chars, starts, ends = _smartblog_alignment_chars_starts_ends(alignment)
    if not chars:
        return arr, 0, 0

    shape_s = str(voice_shape or "neutral").strip().lower()
    preset_s = str(preset or "").strip().lower()
    consonant_gain = float(
        max(
            1.0,
            min(
                2.5,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CONSONANT_GAIN",
                    1.12
                    if preset_s == "vocal"
                    else 1.42
                    if shape_s == "male"
                    else 1.20
                    if shape_s == "female"
                    else 1.32,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CONSONANT_GAIN",
                    1.42 if shape_s == "male" else 1.20,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    labial_gain = float(
        max(
            consonant_gain,
            min(
                2.8,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_LABIAL_GAIN",
                    1.18
                    if preset_s == "vocal"
                    else 1.58
                    if shape_s == "male"
                    else 1.30
                    if shape_s == "female"
                    else 1.48,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_LABIAL_GAIN",
                    1.58 if shape_s == "male" else 1.30,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    vowel_gain = float(
        max(
            0.65,
            min(
                1.35,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_VOWEL_GAIN",
                    1.04 if preset_s == "vocal" else 1.04 if shape_s == "male" else 0.90 if shape_s == "female" else 1.0,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_VOWEL_GAIN",
                    1.04 if shape_s == "male" else 0.90,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    pre_ms = float(max(0.0, min(100.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_PRE_MS", 6.0))))
    consonant_post_ms = float(
        max(
            5.0,
            min(
                180.0,
                _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CONSONANT_POST_MS",
                    42.0,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    closure_enabled = bool(_env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CLOSURE", "1"))
    closure_ms = float(
        max(
            0.0,
            min(
                80.0,
                _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CLOSURE_MS",
                    24.0,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    closure_gain = float(
        max(
            0.50,
            min(
                0.98,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CLOSURE_GAIN",
                    0.86
                    if preset_s == "vocal"
                    else 0.66
                    if shape_s == "male"
                    else 0.76
                    if shape_s == "female"
                    else 0.70,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CLOSURE_GAIN",
                    0.66 if shape_s == "male" else 0.76,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    vowel_min_ms = float(max(20.0, min(300.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_VOWEL_MIN_MS", 70.0))))
    center_focus_enabled = bool(_env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_FOCUS", "1"))
    center_edge_gain = float(
        max(
            0.55,
            min(
                0.99,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_EDGE_GAIN",
                    0.94
                    if preset_s == "vocal"
                    else 0.82
                    if shape_s == "male"
                    else 0.88
                    if shape_s == "female"
                    else 0.85,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_EDGE_GAIN",
                    0.82 if shape_s == "male" else 0.88,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    center_peak_consonant_gain = float(
        max(
            1.0,
            min(
                1.8,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_CONSONANT_GAIN",
                    1.08
                    if preset_s == "vocal"
                    else 1.22
                    if shape_s == "male"
                    else 1.10
                    if shape_s == "female"
                    else 1.16,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_CONSONANT_GAIN",
                    1.22 if shape_s == "male" else 1.10,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    center_peak_vowel_gain = float(
        max(
            0.75,
            min(
                1.35,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_VOWEL_GAIN",
                    1.04
                    if preset_s == "vocal"
                    else 1.08
                    if shape_s == "male"
                    else 0.98
                    if shape_s == "female"
                    else 1.03,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_VOWEL_GAIN",
                    1.08 if shape_s == "male" else 0.98,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    center_min_ms = float(max(8.0, min(120.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_LETTER_CENTER_MIN_MS", 18.0))))
    gain = np.ones((int(arr.size),), dtype=np.float32)
    consonants = 0
    vowels = 0
    closures = 0
    center_focus = 0
    labials = set("pbmfvпбмфв")
    closure_labials = set("pbmпбм")
    for ch, start, end in zip(chars, starts, ends, strict=False):
        ch_s = str(ch or "")
        if not _smartblog_is_avatar_audio_letter_or_number(ch_s):
            continue
        try:
            s = float(start) + float(alignment_offset_sec)
            e = float(end) + float(alignment_offset_sec)
        except Exception:
            continue
        if not math.isfinite(s) or not math.isfinite(e):
            continue
        if e < s:
            e = s
        center = float((float(s) + float(e)) * 0.5)
        duration_ms = float(max(0.0, e - s) * 1000.0)
        if bool(center_focus_enabled) and duration_ms >= center_min_ms:
            focus_start_n = int(max(0, min(int(arr.size), round(float(s) * float(sample_rate)))))
            focus_end_n = int(max(focus_start_n + 1, min(int(arr.size), round(float(e) * float(sample_rate)))))
            if focus_end_n > focus_start_n:
                n = int(focus_end_n - focus_start_n)
                x = np.linspace(-1.0, 1.0, n, endpoint=True, dtype=np.float32)
                triangle = 1.0 - np.abs(x)
                peak = float(
                    center_peak_vowel_gain
                    if _smartblog_is_avatar_audio_vowel(ch_s)
                    else center_peak_consonant_gain
                )
                env = float(center_edge_gain) + (float(peak) - float(center_edge_gain)) * triangle
                gain[focus_start_n:focus_end_n] = gain[focus_start_n:focus_end_n] * env.astype(np.float32, copy=False)
                center_focus += 1
        if _smartblog_is_avatar_audio_vowel(ch_s):
            if abs(vowel_gain - 1.0) <= 0.001:
                continue
            if duration_ms < vowel_min_ms:
                continue
            start_n = int(max(0, min(int(arr.size), round(float(s) * float(sample_rate)))))
            end_n = int(max(start_n + 1, min(int(arr.size), round(float(e) * float(sample_rate)))))
            gain[start_n:end_n] = np.minimum(gain[start_n:end_n], float(vowel_gain))
            vowels += 1
            continue

        g = float(labial_gain if ch_s.lower() in labials else consonant_gain)
        is_closure_labial = bool(ch_s.lower() in closure_labials)
        if bool(closure_enabled) and is_closure_labial and closure_ms > 0.0:
            closure_start_n = int(
                max(0, min(int(arr.size), round((float(center) - closure_ms / 1000.0) * float(sample_rate))))
            )
            closure_end_n = int(max(closure_start_n + 1, min(int(arr.size), round(float(center) * float(sample_rate)))))
            gain[closure_start_n:closure_end_n] = np.minimum(gain[closure_start_n:closure_end_n], float(closure_gain))
            closures += 1
        start_n = int(max(0, min(int(arr.size), round((float(center) - pre_ms / 1000.0) * float(sample_rate)))))
        end_n = int(
            max(
                start_n + 1,
                min(int(arr.size), round((float(center) + consonant_post_ms / 1000.0) * float(sample_rate))),
            )
        )
        center_n = int(max(start_n, min(end_n - 1, round(float(center) * float(sample_rate)))))
        idxs = np.arange(start_n, end_n, dtype=np.float32)
        dist = np.abs(idxs - float(center_n))
        max_dist = float(max(1.0, float(center_n - start_n), float((end_n - 1) - center_n)))
        triangle = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)
        burst_env = 1.0 + (float(g) - 1.0) * triangle
        gain[start_n:end_n] = gain[start_n:end_n] * burst_env.astype(np.float32, copy=False)
        consonants += 1

    if consonants <= 0 and vowels <= 0 and center_focus <= 0:
        return arr, 0, 0
    out = np.asarray(arr, dtype=np.float32) * gain
    logging.warning(
        "SmartBlog render avatar audio letter dynamics: preset=%s voice_shape=%s consonants=%d vowels=%d closures=%d center_focus=%d consonant_gain=%.2f labial_gain=%.2f vowel_gain=%.2f closure_gain=%.2f closure_ms=%.1f center_edge=%.2f center_consonant=%.2f center_vowel=%.2f",
        str(preset),
        str(voice_shape),
        int(consonants),
        int(vowels),
        int(closures),
        int(center_focus),
        float(consonant_gain),
        float(labial_gain),
        float(vowel_gain),
        float(closure_gain),
        float(closure_ms),
        float(center_edge_gain),
        float(center_peak_consonant_gain),
        float(center_peak_vowel_gain),
    )
    return out.astype(np.float32, copy=False), int(consonants), int(vowels)


def _smartblog_apply_avatar_audio_transient_boost(
    arr: np.ndarray,
    *,
    sample_rate: int,
    voice_shape: str,
    preset: str,
) -> tuple[np.ndarray, int]:
    if not _env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_TRANSIENT_BOOST", "1"):
        return arr, 0
    if int(arr.size) < int(sample_rate // 2):
        return arr, 0
    shape_s = str(voice_shape or "neutral").strip().lower()
    preset_s = str(preset or "").strip().lower()
    default_gain = 1.30 if shape_s == "male" else 1.10 if shape_s == "female" else 1.20
    env_name = "SMARTBLOG_RENDER_AVATAR_AUDIO_TRANSIENT_GAIN"
    if preset_s == "vocal":
        default_gain = 1.14
        env_name = "SMARTBLOG_RENDER_AVATAR_AUDIO_VOCAL_TRANSIENT_GAIN"
    max_gain = float(
        max(
            1.0,
            min(
                2.5,
                _smartblog_avatar_audio_shape_float_env(env_name, default_gain, voice_shape=shape_s),
            ),
        )
    )
    if max_gain <= 1.001:
        return arr, 0
    frame_ms = float(max(4.0, min(40.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_TRANSIENT_FRAME_MS", 12.0))))
    release_ms = float(max(10.0, min(180.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_TRANSIENT_RELEASE_MS", 48.0))))
    threshold_db = float(
        max(
            0.1,
            min(
                20.0,
                _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_TRANSIENT_THRESHOLD_DB",
                    2.0,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    frame_n = int(max(32, round(float(sample_rate) * frame_ms / 1000.0)))
    hop_n = int(max(16, int(frame_n // 2)))
    frame_count = 1 + int((int(arr.size) - int(frame_n)) // int(hop_n))
    if frame_count <= 2:
        return arr, 0
    starts = np.arange(0, int(frame_count) * int(hop_n), int(hop_n), dtype=np.int64)
    rms = np.empty((int(frame_count),), dtype=np.float32)
    for idx, start in enumerate(starts):
        frame = arr[int(start) : int(start) + int(frame_n)]
        rms[int(idx)] = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + 1e-12))
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    active_threshold = float(np.percentile(db, 60) - 12.0)
    gain = np.ones((int(arr.size),), dtype=np.float32)
    applied = 0
    release_n = int(max(frame_n, round(float(sample_rate) * release_ms / 1000.0)))
    for idx in range(1, int(frame_count)):
        diff = float(db[int(idx)] - db[int(idx - 1)])
        if diff < threshold_db or float(db[int(idx)]) < active_threshold:
            continue
        local_gain = float(1.0 + (float(max_gain) - 1.0) * min(1.0, diff / 10.0))
        start_n = int(starts[int(idx)])
        end_n = int(min(int(arr.size), start_n + release_n))
        if end_n <= start_n:
            continue
        gain[start_n:end_n] = np.maximum(gain[start_n:end_n], float(local_gain))
        applied += 1
    if applied <= 0:
        return arr, 0
    out = np.asarray(arr, dtype=np.float32) * gain
    logging.warning(
        "SmartBlog render avatar audio transient boost: preset=%s voice_shape=%s events=%d gain=%.2f threshold_db=%.2f",
        str(preset),
        str(voice_shape),
        int(applied),
        float(max_gain),
        float(threshold_db),
    )
    return out.astype(np.float32, copy=False), int(applied)


def _smartblog_apply_avatar_audio_start_boost(
    arr: np.ndarray,
    *,
    sample_rate: int,
    alignment: dict[str, Any] | None,
    alignment_offset_sec: float,
    voice_shape: str,
    preset: str,
) -> tuple[np.ndarray, float, float, float, int]:
    if not _env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_START_BOOST", "1"):
        return arr, 1.0, 0.0, 0.0, 0
    if int(arr.size) <= 0 or int(sample_rate) <= 0:
        return arr, 1.0, 0.0, 0.0, 0

    shape_s = str(voice_shape or "neutral").strip().lower()
    preset_s = str(preset or "").strip().lower()
    base_gain = float(
        max(
            1.0,
            min(
                2.2,
                _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_START_BOOST_GAIN",
                    1.22,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    if shape_s == "male":
        base_gain = float(min(2.2, base_gain * 1.10))
    elif shape_s == "female":
        base_gain = float(max(1.0, base_gain * 0.88))
    if preset_s == "vocal":
        base_gain = float(max(1.0, base_gain * 0.82))

    start_sec = float(
        max(
            0.0,
            min(
                2.0,
                _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_START_BOOST_SEC",
                    0.55,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    if start_sec <= 0.005 or base_gain <= 1.0001:
        return arr, 1.0, 0.0, 0.0, 0

    target_peak = float(
        max(
            0.20,
            min(
                0.95,
                _safe_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_START_TARGET_PEAK",
                    0.80 if shape_s == "male" else 0.70 if shape_s == "female" or preset_s == "vocal" else 0.76,
                )
                if shape_s not in {"male", "female"}
                else _smartblog_avatar_audio_shape_float_env(
                    "SMARTBLOG_RENDER_AVATAR_AUDIO_START_TARGET_PEAK",
                    0.80 if shape_s == "male" else 0.70,
                    voice_shape=shape_s,
                ),
            ),
        )
    )
    boost_n = int(min(int(arr.size), max(1, round(float(start_sec) * float(sample_rate)))))
    head = np.asarray(arr[:boost_n], dtype=np.float32)
    peak = float(np.max(np.abs(head))) if int(head.size) else 0.0
    effective_gain = float(base_gain)
    if peak > 1e-5:
        effective_gain = float(min(effective_gain, max(1.0, target_peak / peak)))
    if peak >= target_peak * 0.98:
        effective_gain = float(1.0 + (effective_gain - 1.0) * 0.25)
    if effective_gain <= 1.0001:
        effective_gain = 1.0

    gain = np.ones((int(arr.size),), dtype=np.float32)
    if effective_gain > 1.0001:
        attack_n = int(min(boost_n, max(1, round(float(sample_rate) * 0.020))))
        hold_n = int(min(boost_n, max(attack_n, round(float(sample_rate) * 0.160))))
        if attack_n > 0:
            gain[:attack_n] = np.linspace(1.0, effective_gain, attack_n, endpoint=True, dtype=np.float32)
        if hold_n > attack_n:
            gain[attack_n:hold_n] = float(effective_gain)
        if boost_n > hold_n:
            decay = np.linspace(0.0, 1.0, boost_n - hold_n, endpoint=True, dtype=np.float32)
            smooth = 0.5 - 0.5 * np.cos(np.pi * decay)
            gain[hold_n:boost_n] = float(effective_gain) + (1.0 - float(effective_gain)) * smooth

    vowel_hits = 0
    if _env_flag("SMARTBLOG_RENDER_AVATAR_AUDIO_START_VOWEL_BOOST", "1"):
        vowel_gain = float(
            max(
                1.0,
                min(
                    1.6,
                    _safe_float_env(
                        "SMARTBLOG_RENDER_AVATAR_AUDIO_START_VOWEL_GAIN",
                        1.16 if shape_s == "male" else 1.04 if shape_s == "female" or preset_s == "vocal" else 1.10,
                    )
                    if shape_s not in {"male", "female"}
                    else _smartblog_avatar_audio_shape_float_env(
                        "SMARTBLOG_RENDER_AVATAR_AUDIO_START_VOWEL_GAIN",
                        1.16 if shape_s == "male" else 1.04,
                        voice_shape=shape_s,
                    ),
                ),
            )
        )
        vowel_window_sec = float(
            max(
                0.0,
                min(
                    2.0,
                    _smartblog_avatar_audio_shape_float_env(
                        "SMARTBLOG_RENDER_AVATAR_AUDIO_START_VOWEL_WINDOW_SEC",
                        start_sec + 0.25,
                        voice_shape=shape_s,
                    ),
                ),
            )
        )
        if vowel_gain > 1.0001 and vowel_window_sec > 0.0:
            chars, starts, ends = _smartblog_alignment_chars_starts_ends(alignment)
            for ch, start, end in zip(chars, starts, ends, strict=False):
                ch_s = str(ch or "")
                if not _smartblog_is_avatar_audio_vowel(ch_s):
                    continue
                try:
                    s = float(start) + float(alignment_offset_sec)
                    e = float(end) + float(alignment_offset_sec)
                except Exception:
                    continue
                if not math.isfinite(s) or not math.isfinite(e):
                    continue
                if e < s:
                    e = s
                center = float((s + e) * 0.5)
                if center < 0.0 or center > vowel_window_sec:
                    continue
                half_sec = float(max(0.030, min(0.120, max(0.0, e - s) * 0.65)))
                start_n = int(max(0, min(int(arr.size), round((center - half_sec) * float(sample_rate)))))
                end_n = int(max(start_n + 1, min(int(arr.size), round((center + half_sec) * float(sample_rate)))))
                if end_n <= start_n:
                    continue
                n = int(end_n - start_n)
                x = np.linspace(-1.0, 1.0, n, endpoint=True, dtype=np.float32)
                triangle = 1.0 - np.abs(x)
                gain[start_n:end_n] = gain[start_n:end_n] * (1.0 + (float(vowel_gain) - 1.0) * triangle)
                vowel_hits += 1

    if effective_gain <= 1.0001 and vowel_hits <= 0:
        return arr, 1.0, start_sec, peak, 0

    out = np.asarray(arr, dtype=np.float32) * gain
    logging.warning(
        "SmartBlog render avatar audio start boost: preset=%s voice_shape=%s gain=%.2f target_peak=%.2f peak=%.3f sec=%.2f vowel_hits=%d",
        str(preset),
        str(voice_shape),
        float(effective_gain),
        float(target_peak),
        float(peak),
        float(start_sec),
        int(vowel_hits),
    )
    return out.astype(np.float32, copy=False), float(effective_gain), float(start_sec), float(peak), int(vowel_hits)


def _smartblog_prepare_avatar_lipsync_audio_file(
    *,
    source_wav: str,
    output_wav: str,
    sample_rate: int,
    preset: str,
    alignment: dict[str, Any] | None = None,
    alignment_offset_sec: float = 0.0,
) -> str:
    source_s = str(source_wav or "").strip()
    output_s = str(output_wav or "").strip()
    if not source_s or not os.path.exists(source_s):
        raise RuntimeError(f"avatar audio preprocess source is missing: {source_s}")
    if not output_s:
        raise RuntimeError("avatar audio preprocess output is empty")
    os.makedirs(os.path.dirname(os.path.abspath(output_s)) or ".", exist_ok=True)
    requested_preset = str(preset or "clear").strip().lower()
    alignment_chars, _, _ = _smartblog_alignment_chars_starts_ends(alignment)
    has_alignment = bool(alignment_chars)
    effective_preset = (
        _smartblog_detect_avatar_audio_preset(str(source_s), has_alignment=bool(has_alignment))
        if requested_preset == "auto"
        else str(requested_preset)
    )
    voice_shape = (
        _smartblog_detect_avatar_audio_voice_shape(str(source_s))
        if str(effective_preset).strip().lower() in {"speech", "clear", "safe", "less_jitter"}
        else "vocal"
    )
    filter_s = _smartblog_render_avatar_audio_filter(str(effective_preset), voice_shape=str(voice_shape))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        source_s,
        "-ac",
        "1",
        "-ar",
        str(int(max(1, int(sample_rate or 16000)))),
        "-af",
        filter_s,
        "-c:a",
        "pcm_s16le",
        output_s,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if int(proc.returncode) != 0:
        stderr = str(proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"avatar audio preprocess ffmpeg failed: {stderr[-1200:]}")
    if not os.path.exists(output_s) or os.path.getsize(output_s) <= 44:
        raise RuntimeError(f"avatar audio preprocess produced no output: {output_s}")
    _smartblog_postprocess_avatar_lipsync_audio_wav(
        output_s,
        preset=str(effective_preset),
        voice_shape=str(voice_shape),
        alignment=alignment,
        alignment_offset_sec=float(alignment_offset_sec),
    )
    logging.warning(
        "SmartBlog render avatar audio preprocess: preset=%s effective=%s voice_shape=%s src=%s out=%s dt=%.3fs",
        str(requested_preset),
        str(effective_preset),
        str(voice_shape),
        os.path.basename(source_s),
        os.path.basename(output_s),
        float(time.perf_counter() - t0),
    )
    return output_s


def _smartblog_postprocess_avatar_lipsync_audio_wav(
    wav_path: str,
    *,
    preset: str,
    voice_shape: str = "neutral",
    alignment: dict[str, Any] | None = None,
    alignment_offset_sec: float = 0.0,
) -> None:
    path_s = str(wav_path or "").strip()
    if not path_s or not os.path.exists(path_s):
        return
    pcm, sample_rate = _smartblog_wav_pcm16_mono(path_s)
    if int(pcm.size) <= 0 or int(sample_rate) <= 0:
        return
    advance_ms = float(max(-500.0, min(500.0, _safe_float_env("SMARTBLOG_RENDER_AVATAR_AUDIO_ADVANCE_MS", 60.0))))

    arr = pcm.astype(np.float32) / 32768.0
    if abs(float(advance_ms)) >= 0.5:
        shift = int(round(float(sample_rate) * float(advance_ms) / 1000.0))
        if shift > 0 and shift < int(arr.size):
            arr = np.concatenate((arr[shift:], np.zeros((shift,), dtype=np.float32))).astype(np.float32, copy=False)
        elif shift < 0 and -shift < int(arr.size):
            n = int(-shift)
            arr = np.concatenate((np.zeros((n,), dtype=np.float32), arr[:-n])).astype(np.float32, copy=False)

    arr, start_gain, start_sec, start_peak, start_vowels = _smartblog_apply_avatar_audio_start_boost(
        arr,
        sample_rate=int(sample_rate),
        alignment=alignment,
        alignment_offset_sec=float(alignment_offset_sec),
        voice_shape=str(voice_shape),
        preset=str(preset),
    )

    arr, letter_consonants, letter_vowels = _smartblog_apply_avatar_audio_letter_dynamics(
        arr,
        sample_rate=int(sample_rate),
        alignment=alignment,
        alignment_offset_sec=float(alignment_offset_sec),
        voice_shape=str(voice_shape),
        preset=str(preset),
    )
    arr, transient_events = _smartblog_apply_avatar_audio_transient_boost(
        arr,
        sample_rate=int(sample_rate),
        voice_shape=str(voice_shape),
        preset=str(preset),
    )
    if (
        start_gain <= 1.0001
        and int(start_vowels) <= 0
        and abs(float(advance_ms)) < 0.5
        and int(letter_consonants) <= 0
        and int(letter_vowels) <= 0
        and int(transient_events) <= 0
    ):
        return

    arr = np.tanh(arr * 1.15) / np.tanh(1.15)
    out = np.clip(np.rint(np.clip(arr, -1.0, 1.0) * 32767.0), -32768, 32767).astype(np.int16)
    _smartblog_write_wav_pcm16_mono(path_s, out, sample_rate=int(sample_rate))
    logging.warning(
        "SmartBlog render avatar audio postprocess: preset=%s voice_shape=%s file=%s start_gain=%.2f start_sec=%.2f start_peak=%.3f start_vowels=%d advance_ms=%.1f letter_consonants=%d letter_vowels=%d transient_events=%d",
        str(preset),
        str(voice_shape),
        os.path.basename(path_s),
        float(start_gain),
        float(start_sec),
        float(start_peak),
        int(start_vowels),
        float(advance_ms),
        int(letter_consonants),
        int(letter_vowels),
        int(transient_events),
    )


def _smartblog_prepare_avatar_lipsync_audio_segments(
    *,
    segment_infos: list[dict[str, Any]],
    run_dir: str,
    job_id: str,
) -> int:
    if not _smartblog_render_avatar_audio_preprocess_enabled():
        return 0
    preset = _smartblog_render_avatar_audio_preprocess_preset()
    applied = 0
    for pos, info in enumerate(list(segment_infos or [])):
        file_index = _smartblog_segment_file_index(dict(info or {}), fallback=int(pos))
        source_wav = str(info.get("lipsync_audio_wav_path") or info.get("audio_wav_path") or "").strip()
        if not source_wav:
            logging.warning(
                "SmartBlog render avatar audio preprocess skipped: job=%s segment=%d reason=missing_audio",
                str(job_id or "-"),
                int(file_index),
            )
            continue
        sample_rate = int(info.get("sample_rate") or 16000)
        entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
        alignment = _smartblog_audio_entry_alignment(entry, normalized=True) or _smartblog_audio_entry_alignment(
            entry,
            normalized=False,
        )
        before_sec = float(int(info.get("boundary_before_samples") or 0)) / float(max(1, int(sample_rate)))
        core_start_sec = float(int(info.get("boundary_core_start_samples") or 0)) / float(max(1, int(sample_rate)))
        if "lipsync_alignment_offset_sec" in info:
            try:
                alignment_offset_sec = float(info.get("lipsync_alignment_offset_sec") or 0.0)
            except Exception:
                alignment_offset_sec = 0.0
        else:
            alignment_offset_sec = float(before_sec - core_start_sec)
        out_path = os.path.join(str(run_dir), f"audio_segment_{int(file_index):03d}_avatar_{preset}_16k.wav")
        try:
            prepared = _smartblog_prepare_avatar_lipsync_audio_file(
                source_wav=str(source_wav),
                output_wav=str(out_path),
                sample_rate=int(sample_rate),
                preset=str(preset),
                alignment=alignment if isinstance(alignment, dict) else None,
                alignment_offset_sec=float(alignment_offset_sec),
            )
        except Exception as e:
            logging.warning(
                "SmartBlog render avatar audio preprocess failed: job=%s segment=%d preset=%s err=%s",
                str(job_id or "-"),
                int(file_index),
                str(preset),
                str(e),
            )
            continue
        info["lipsync_audio_wav_path"] = str(prepared)
        info["avatar_audio_preprocess_preset"] = str(preset)
        applied += 1
    if int(applied) > 0:
        logging.warning(
            "SmartBlog render avatar audio preprocess ready: job=%s preset=%s applied=%d/%d",
            str(job_id or "-"),
            str(preset),
            int(applied),
            int(len(segment_infos or [])),
        )
    return int(applied)


def _smartblog_video_config(claim: dict[str, Any]) -> dict[str, Any]:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    persona_payload = payload.get("persona") if isinstance(payload.get("persona"), dict) else {}
    merged: dict[str, Any] = {}

    def merge_from(src: Any) -> None:
        if isinstance(src, dict):
            merged.update(src)

    for src in (payload, persona_payload, job, persona, claim):
        if not isinstance(src, dict):
            continue
        merge_from(src.get("prompt_config"))
        merge_from(src.get("video"))
    return merged


def _smartblog_render_mode(claim: dict[str, Any]) -> str:
    payload = _smartblog_job_payload(claim)
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    payload_assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    video = _smartblog_video_config(claim)
    for src in (assets, payload_assets, video, payload, claim):
        if not isinstance(src, dict):
            continue
        value = _smartblog_first_text(src.get("render_mode"), src.get("renderMode"), src.get("mode"))
        text = str(value or "").strip().lower().replace("-", "_")
        if text in {"t2v", "text_to_video", "text2video", "hunyuan_t2v"}:
            return "t2v"
        if text in {"i2v", "image_to_video", "image2video", "avatar", "avatar_video"}:
            return "i2v"
    return "i2v"


def _smartblog_normalize_video_audio_config(
    raw: dict[str, Any] | None,
    *,
    default_gain_db: float = -12.0,
) -> dict[str, Any]:
    raw = dict(raw or {}) if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode") or raw.get("audio_mode") or raw.get("audioMode") or "").strip().lower()
    if mode not in {"auto", "prompt", "asset", "off"}:
        if raw.get("audio_url") or raw.get("audioUrl") or raw.get("url"):
            mode = "asset"
        elif raw.get("prompt") or raw.get("negative_prompt") or raw.get("negativePrompt"):
            mode = "prompt"
        elif raw:
            mode = "auto"
        else:
            mode = "auto"
    try:
        gain_db = float(raw.get("gain_db", raw.get("gainDb", default_gain_db)))
    except Exception:
        gain_db = float(default_gain_db)
    if not math.isfinite(gain_db):
        gain_db = float(default_gain_db)
    gain_db = float(max(-60.0, min(12.0, gain_db)))
    return {
        "mode": str(mode),
        "prompt": str(raw.get("prompt") or "").strip(),
        "negative_prompt": str(raw.get("negative_prompt") or raw.get("negativePrompt") or "").strip(),
        "audio_url": str(raw.get("audio_url") or raw.get("audioUrl") or raw.get("url") or "").strip(),
        "gain_db": float(gain_db),
    }


def _smartblog_video_audio_config(claim: dict[str, Any]) -> dict[str, Any]:
    video = _smartblog_video_config(claim)
    raw = video.get("audio") if isinstance(video.get("audio"), dict) else {}
    return _smartblog_normalize_video_audio_config(
        raw,
        default_gain_db=_safe_float_env("SMARTBLOG_VIDEO_AUDIO_GAIN_DB", -12.0),
    )


def _smartblog_background_music_config(claim: dict[str, Any]) -> dict[str, Any]:
    video = _smartblog_video_config(claim)
    raw = video.get("background_music") if isinstance(video.get("background_music"), dict) else {}
    if not raw and isinstance(video.get("backgroundMusic"), dict):
        raw = video.get("backgroundMusic") or {}
    if not isinstance(raw, dict):
        raw = {}
    audio_url = str(raw.get("audio_url") or raw.get("audioUrl") or raw.get("url") or "").strip()
    if not audio_url:
        return {"enabled": False}

    def clamp_float(value: Any, *, default: float, min_value: float, max_value: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if not math.isfinite(out):
            out = float(default)
        return float(max(float(min_value), min(float(max_value), out)))

    loop_raw = raw.get("loop")
    if loop_raw is None:
        loop = True
    elif isinstance(loop_raw, str):
        loop = str(loop_raw).strip().lower() not in {"0", "false", "no", "off"}
    else:
        loop = bool(loop_raw)
    return {
        "enabled": True,
        "audio_url": str(audio_url),
        "loop": bool(loop),
        "gain_db": clamp_float(raw.get("gain_db", raw.get("gainDb", 0.0)), default=0.0, min_value=-24.0, max_value=12.0),
        "duck_voice_db": clamp_float(
            raw.get("duck_voice_db", raw.get("duckVoiceDb", 0.0)),
            default=0.0,
            min_value=-24.0,
            max_value=0.0,
        ),
        "fade_in_seconds": clamp_float(
            raw.get("fade_in_seconds", raw.get("fadeInSeconds", 0.0)),
            default=0.0,
            min_value=0.0,
            max_value=60.0,
        ),
        "fade_out_seconds": clamp_float(
            raw.get("fade_out_seconds", raw.get("fadeOutSeconds", 0.0)),
            default=0.0,
            min_value=0.0,
            max_value=60.0,
        ),
        "start_offset_seconds": clamp_float(
            raw.get("start_offset_seconds", raw.get("startOffsetSeconds", 0.0)),
            default=0.0,
            min_value=0.0,
            max_value=86400.0,
        ),
    }


def _smartblog_animation_config(claim: dict[str, Any]) -> dict[str, Any]:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    persona_payload = payload.get("persona") if isinstance(payload.get("persona"), dict) else {}
    merged: dict[str, Any] = {}

    def merge_from(src: Any) -> None:
        if not isinstance(src, dict):
            return
        for key in ("speaking_prompt", "idle_prompt"):
            if key in src:
                merged[key] = src.get(key)
        if "speaking_animation_mode" in src:
            merged["speaking_prompt"] = src.get("speaking_animation_mode")
        if "idle_animation_mode" in src:
            merged["idle_prompt"] = src.get("idle_animation_mode")

    for src in (payload, persona_payload, job, persona, claim):
        if not isinstance(src, dict):
            continue
        merge_from(src.get("prompt_config"))
        merge_from(src.get("animation"))
        merge_from(src)
    return merged


def _smartblog_remote_edge_render_enabled() -> bool:
    return bool(_env_flag("REMOTE_EDGE_ENABLED", "0") and _env_flag("REMOTE_EDGE_SKIP_LOCAL_DECODE", "0"))


def _smartblog_stream_file_render_enabled() -> bool:
    return bool(_env_flag("SMARTBLOG_RENDER_STREAM_FILE", "0") and not _smartblog_remote_edge_render_enabled())


def _smartblog_claim_workspace_id(claim: dict[str, Any]) -> str:
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    payload = _smartblog_job_payload(claim)
    return str(job.get("workspace_id") or payload.get("workspace_id") or claim.get("workspace_id") or "").strip()


def _smartblog_claim_persona_id(claim: dict[str, Any]) -> str:
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    payload = _smartblog_job_payload(claim)
    persona = claim.get("persona") if isinstance(claim.get("persona"), dict) else {}
    return str(
        job.get("persona_id")
        or payload.get("persona_id")
        or persona.get("id")
        or claim.get("persona_id")
        or ""
    ).strip()


def _smartblog_write_render_remote_edge_manifest(
    *,
    claim: dict[str, Any],
    live_raw_dir: str,
    job_id: str,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    target_audio_samples: int,
    target_duration_sec: float,
    upload: dict[str, Any],
    public_url: str,
    watermark_text: str | None = None,
    remote_finalizer: bool | None = None,
    file_output_fps: int | None = None,
) -> None:
    host = str(os.getenv("REMOTE_EDGE_HOST", "") or "").strip()
    if not host:
        raise RuntimeError("REMOTE_EDGE_HOST is required for render_video remote edge output")
    signed_url = str(upload.get("signed_url") or "").strip()
    upload_path = str(upload.get("path") or "").strip()
    if not signed_url:
        raise RuntimeError("render_video remote edge output requires upload.signed_url")
    if not upload_path:
        raise RuntimeError("render_video remote edge output requires upload.path")
    watermark = _smartblog_watermark_text(claim) if watermark_text is None else normalize_watermark_text(watermark_text)
    output_fps = int(file_output_fps or fps or 0)
    if output_fps <= 0:
        output_fps = int(max(1, int(fps)))
    manifest = {
        "version": 1,
        "enabled": True,
        "host": host,
        "port": int(_required_positive_int_env("REMOTE_EDGE_PORT")),
        "mode": str(_required_str_env("REMOTE_EDGE_STREAM_MODE")).strip().lower(),
        "auth_token": str(os.getenv("REMOTE_EDGE_SHARED_SECRET", "") or ""),
        "connect_timeout_sec": float(_required_positive_float_env("REMOTE_EDGE_CONNECT_TIMEOUT_SEC")),
        "write_timeout_sec": float(_required_positive_float_env("REMOTE_EDGE_WRITE_TIMEOUT_SEC")),
        "output": "file",
        "session_id": str(job_id or f"render-{int(time.time() * 1000)}"),
        "job_id": str(job_id or ""),
        "workspace_id": _smartblog_claim_workspace_id(claim),
        "persona_id": _smartblog_claim_persona_id(claim),
        "width": int(width),
        "height": int(height),
        "fps": int(max(1, int(fps))),
        "sample_rate": int(max(1, int(sample_rate))),
        "file_target_audio_samples": int(max(0, int(target_audio_samples or 0))),
        "file_target_duration_sec": float(max(0.0, float(target_duration_sec or 0.0))),
        "file_upload_url": signed_url,
        "file_upload_path": upload_path,
        "file_public_url": str(public_url or ""),
        "file_content_type": "video/mp4",
        "file_progress_path": os.path.join(str(live_raw_dir), "remote_edge_file_progress.json"),
        "file_output_fps": int(max(1, int(output_fps))),
        "watermark_text": str(watermark or ""),
        "created_at_ms": int(time.time() * 1000.0),
    }
    if remote_finalizer is not None:
        manifest["file_remote_finalizer"] = bool(remote_finalizer)
    os.makedirs(str(live_raw_dir), exist_ok=True)
    path = os.path.join(str(live_raw_dir), "remote_edge.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _smartblog_read_remote_edge_file_progress(progress_path: str) -> dict[str, Any]:
    path = str(progress_path or "").strip()
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return dict(obj) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _smartblog_read_render_model_raw_progress(live_raw_dir: str) -> dict[str, Any]:
    root = os.path.abspath(str(live_raw_dir or "").strip())
    if not root:
        return {}
    progress_path = os.path.join(root, "progress.json")
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj:
            obj.setdefault("source", "model_raw_json")
            return obj
    except Exception:
        pass

    meta_path = os.path.join(root, "stream.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    if not isinstance(meta, dict) or str(meta.get("transport") or "").strip() != "shm_ring":
        return {}
    shm_name = str(meta.get("shm_name") or "").strip()
    if not shm_name:
        return {}
    shm = None
    try:
        shm = attach_shared_memory_no_tracker(str(shm_name))
        header = live_raw_shm_read_header(shm.buf)
        if isinstance(header, dict) and header:
            header.setdefault("source", "model_raw_shm")
            return header
    except Exception:
        return {}
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
    return {}


def _smartblog_render_model_raw_progress_fraction(
    raw_progress: dict[str, Any],
    *,
    target_frames: int,
) -> float | None:
    if not isinstance(raw_progress, dict) or not raw_progress:
        return None
    try:
        written_frames = int(raw_progress.get("written_frames") or 0)
    except Exception:
        written_frames = 0
    try:
        enqueued_frames = int(raw_progress.get("enqueued_frames") or 0)
    except Exception:
        enqueued_frames = 0
    done = bool(raw_progress.get("done"))
    target = int(max(1, int(target_frames or 0)))
    if bool(done):
        return 1.0
    frames = int(max(0, int(written_frames), int(enqueued_frames)))
    if frames <= 0:
        return None
    return float(max(0.0, min(0.995, float(frames) / float(target))))


def _smartblog_estimated_render_inference_stage_frac(
    *,
    num_clip: int,
    started_mono: float,
) -> float:
    elapsed_sec = max(0.0, float(time.monotonic()) - float(started_mono or time.monotonic()))
    # Long one-pass avatar renders can legitimately spend tens of minutes
    # inside a single model request. Keep the synthetic progress moving even
    # after the normal estimator reaches its expected duration, because the API
    # sweeper checks numeric progress movement, not just heartbeat freshness.
    sec_per_clip = max(1.0, min(60.0, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_SEC_PER_CLIP", 14.0)))
    min_expected_sec = max(1.0, min(300.0, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_MIN_SEC", 12.0)))
    expected_sec = max(float(min_expected_sec), float(max(1, int(num_clip))) * float(sec_per_clip))
    start_frac = max(0.0, min(0.5, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_START", 0.05)))
    final_cap_frac = max(start_frac, min(0.99, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_CAP", 0.99)))
    linear_cap_default = min(0.90, float(final_cap_frac))
    linear_cap_frac = max(
        start_frac,
        min(float(final_cap_frac), _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_LINEAR_CAP", linear_cap_default)),
    )
    frac = float(start_frac + (linear_cap_frac - start_frac) * min(1.0, float(elapsed_sec) / float(expected_sec)))
    if elapsed_sec > expected_sec and final_cap_frac > linear_cap_frac:
        tail_step_sec = max(
            30.0,
            min(600.0, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_TAIL_STEP_SEC", 180.0)),
        )
        # Render jobs use a 65-point inference stage. Advancing the stage by
        # 1/65 therefore advances the overall job progress by about one point.
        inference_weight_points = max(
            1.0,
            min(100.0, _safe_float_env("SMARTBLOG_RENDER_INFERENCE_PROGRESS_STAGE_POINTS", 65.0)),
        )
        tail_frac = float(linear_cap_frac) + (
            max(0.0, float(elapsed_sec) - float(expected_sec)) / (float(tail_step_sec) * float(inference_weight_points))
        )
        frac = max(float(frac), min(float(final_cap_frac), float(tail_frac)))
    return float(max(start_frac, min(final_cap_frac, frac)))


def _smartblog_hunyuan_progress_expected_sec(
    *,
    num_frames: int,
    frame_rate: int | float,
    duration_sec: float = 0.0,
    num_inference_steps: int | None = None,
) -> float:
    frames_i = int(max(1, int(num_frames or 1)))
    fps_f = float(max(1.0, float(frame_rate or 1.0)))
    duration_f = float(duration_sec or 0.0)
    if duration_f <= 0.0:
        duration_f = float(frames_i) / float(fps_f)
    if num_inference_steps is None:
        num_inference_steps = int(
            max(
                1,
                int(
                    os.getenv("SMARTBLOG_HUNYUAN_NUM_INFERENCE_STEPS")
                    or "8"
                ),
            )
        )
    steps_i = int(max(1, int(num_inference_steps or 1)))

    floor_sec = max(
        10.0,
        _safe_float_env(
            "SMARTBLOG_HUNYUAN_PROGRESS_EXPECTED_SEC",
            60.0,
        ),
    )
    sec_per_frame_step = max(
        0.001,
        _safe_float_env(
            "SMARTBLOG_HUNYUAN_PROGRESS_SEC_PER_FRAME_STEP",
            0.12,
        ),
    )
    sec_per_video_sec = max(
        1.0,
        _safe_float_env(
            "SMARTBLOG_HUNYUAN_PROGRESS_SEC_PER_VIDEO_SEC",
            18.0,
        ),
    )
    slack = max(
        1.0,
        _safe_float_env(
            "SMARTBLOG_HUNYUAN_PROGRESS_EXPECTED_SLACK",
            1.0,
        ),
    )
    cap_sec = max(
        floor_sec,
        _safe_float_env(
            "SMARTBLOG_HUNYUAN_PROGRESS_EXPECTED_MAX_SEC",
            _safe_float_env("SMARTBLOG_LTX_PROGRESS_EXPECTED_MAX_SEC", 7200.0),
        ),
    )

    frame_step_estimate = float(frames_i) * float(steps_i) * float(sec_per_frame_step)
    duration_estimate = float(duration_f) * float(sec_per_video_sec)
    estimate = max(float(floor_sec), float(frame_step_estimate), float(duration_estimate)) * float(slack)
    return float(max(10.0, min(float(cap_sec), float(estimate))))


def _smartblog_hunyuan_progress_local_frac(*, elapsed_sec: float, expected_sec: float) -> float:
    expected = max(1.0, float(expected_sec or 1.0))
    elapsed = max(0.0, float(elapsed_sec or 0.0))
    linear_cap = max(
        0.10,
        min(0.98, _safe_float_env("SMARTBLOG_HUNYUAN_PROGRESS_LINEAR_CAP", 0.94)),
    )
    final_cap = max(
        linear_cap,
        min(0.995, _safe_float_env("SMARTBLOG_HUNYUAN_PROGRESS_FINAL_CAP", 0.995)),
    )
    frac = float(linear_cap) * min(1.0, float(elapsed) / float(expected))
    if elapsed > expected and final_cap > linear_cap:
        tail_step_sec = max(
            15.0,
            min(300.0, _safe_float_env("SMARTBLOG_HUNYUAN_PROGRESS_TAIL_STEP_SEC", 45.0)),
        )
        tail_frac = float(linear_cap) + max(0.0, float(elapsed) - float(expected)) / float(tail_step_sec) * 0.01
        frac = max(float(frac), min(float(final_cap), float(tail_frac)))
    return float(max(0.0, min(float(final_cap), float(frac))))


def _smartblog_render_prompt(claim: dict[str, Any]) -> str:
    video = _smartblog_video_config(claim)
    payload = _smartblog_job_payload(claim)
    effective_text = _smartblog_first_text(
        video.get("effective_prompt"),
        video.get("effectivePrompt"),
        payload.get("video_effective_prompt"),
        payload.get("video_effective_prompt_override"),
        payload.get("effective_prompt"),
        payload.get("effectivePrompt"),
        payload.get("effective_prompt_override"),
    )
    video_text = _smartblog_first_text(
        video.get("prompt"),
        video.get("video_prompt"),
        payload.get("video_prompt"),
        payload.get("prompt"),
        payload.get("render_prompt"),
    )
    if effective_text:
        return effective_text
    text = video_text
    if text:
        return text
    fallback = _smartblog_render_default_speaking_prompt()
    if fallback:
        return fallback
    raise RuntimeError("SmartBlog render default speaking prompt is empty")


def _smartblog_render_idle_prompt(claim: dict[str, Any]) -> str:
    animation = _smartblog_animation_config(claim)
    payload = _smartblog_job_payload(claim)
    text = _smartblog_first_text(
        animation.get("idle_prompt"),
        payload.get("idle_prompt"),
        payload.get("idle_animation_mode"),
    )
    if text:
        return text
    return _smartblog_render_default_idle_prompt()


def _smartblog_render_video_prompt(claim: dict[str, Any]) -> str:
    video = _smartblog_video_config(claim)
    payload = _smartblog_job_payload(claim)
    for cand in (
        video.get("prompt"),
        video.get("video_prompt"),
        payload.get("video_prompt"),
        payload.get("prompt"),
        payload.get("render_prompt"),
    ):
        text = str(cand or "").strip()
        if text:
            return text
    return ""


def _smartblog_render_negative_prompt(claim: dict[str, Any]) -> str:
    video = _smartblog_video_config(claim)
    payload = _smartblog_job_payload(claim)
    selected = ""
    for cand in (
        video.get("effective_negative_prompt"),
        video.get("effectiveNegativePrompt"),
        payload.get("effective_negative_prompt"),
        payload.get("effectiveNegativePrompt"),
        video.get("negative_prompt"),
        video.get("video_negative_prompt"),
        payload.get("negative_prompt"),
        payload.get("video_negative_prompt"),
        payload.get("negativePrompt"),
    ):
        text = str(cand or "").strip()
        if text:
            selected = text
            break
    if not selected:
        selected = str(os.getenv("SMARTBLOG_RENDER_DEFAULT_NEGATIVE_PROMPT", "") or "").strip()
    return str(selected or "").strip()


def _smartblog_ltx_source_dicts(claim: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    content = claim.get("content_item") if isinstance(claim.get("content_item"), dict) else {}
    content_metadata = content.get("metadata_json") if isinstance(content.get("metadata_json"), dict) else {}
    video = _smartblog_video_config(claim)
    sources: list[dict[str, Any]] = []

    def add(src: Any, *, nested: bool = False) -> None:
        if isinstance(src, dict):
            item = dict(src)
            if bool(nested):
                item["_smartblog_ltx_nested"] = True
            sources.append(item)

    for src in (video, payload, assets, content_metadata, content, job, claim):
        add(src)
        if isinstance(src, dict):
            for key in ("ltx", "ltx_video", "ltxVideo", "video2", "video_model", "videoModel", "render"):
                add(src.get(key), nested=True)
    return sources


def _smartblog_ltx_render_requested(claim: dict[str, Any]) -> bool:
    for src in _smartblog_ltx_source_dicts(claim):
        for key in (
            "render_engine",
            "renderEngine",
            "video_engine",
            "videoEngine",
            "engine",
            "backend",
            "model",
            "model_id",
            "modelId",
            "generator",
            "pipeline",
            "provider",
        ):
            if key not in src:
                continue
            value = src.get(key)
            if isinstance(value, dict):
                value = _smartblog_first_text(value.get("id"), value.get("name"), value.get("model"), value.get("backend"))
            text = str(value or "").strip().lower()
            if text and any(token in text for token in ("ltx", "ltxv", "ltx-video", "ltx_video")):
                return True
    return False


def _smartblog_ltx_prompt(claim: dict[str, Any]) -> str:
    for src in _smartblog_ltx_source_dicts(claim):
        text = _smartblog_first_text(
            src.get("ltx_prompt"),
            src.get("ltxPrompt"),
            src.get("effective_prompt"),
            src.get("effectivePrompt"),
            src.get("video_prompt"),
            src.get("videoPrompt"),
            src.get("prompt"),
            src.get("render_prompt"),
            src.get("renderPrompt"),
        )
        if text:
            return text
    return _smartblog_render_default_hunyuan_prompt()


def _smartblog_ltx_parse_size_text(text: Any) -> tuple[int, int] | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None
    match = re.search(r"(\d{2,5})\s*[x*]\s*(\d{2,5})", raw)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width > 0 and height > 0:
        return width, height
    return None


def _smartblog_ltx_dimensions(claim: dict[str, Any]) -> tuple[int, int]:
    ltx_width_keys = ("ltx_width", "ltxWidth")
    ltx_height_keys = ("ltx_height", "ltxHeight")
    nested_width_keys = ("width", "output_width", "outputWidth")
    nested_height_keys = ("height", "output_height", "outputHeight")
    for src in _smartblog_ltx_source_dicts(claim):
        nested = bool(src.get("_smartblog_ltx_nested"))
        width_keys = ltx_width_keys + (nested_width_keys if nested else ())
        height_keys = ltx_height_keys + (nested_height_keys if nested else ())
        width: int | None = None
        height: int | None = None
        for key in width_keys:
            if key in src:
                width = _smartblog_optional_int(src.get(key))
                break
        for key in height_keys:
            if key in src:
                height = _smartblog_optional_int(src.get(key))
                break
        if width and height:
            return int(width), int(height)
        size_keys = ("ltx_size", "ltxSize", "ltx_resolution", "ltxResolution")
        if nested:
            size_keys = size_keys + ("resolution", "output_resolution", "outputResolution")
        for key in size_keys:
            if key in src:
                parsed = _smartblog_ltx_parse_size_text(src.get(key))
                if parsed is not None:
                    return parsed

    orientation = _smartblog_orientation_hint(claim)
    orientation = "landscape" if str(orientation or "").strip().lower() == "landscape" else "portrait"
    profile = str(os.getenv("SMARTBLOG_HUNYUAN_SIZE_PROFILE") or os.getenv("SMARTBLOG_LTX_SIZE_PROFILE", "liveavatar") or "liveavatar").strip().lower()
    parsed_profile = _smartblog_ltx_parse_size_text(profile)
    if parsed_profile is not None:
        return parsed_profile
    if profile in {"social-640", "social_640", "short640", "short-640", "short_640"}:
        return (1152, 640) if orientation == "landscape" else (640, 1152)
    if profile in {"hunyuan-832", "hunyuan_832", "social-832", "social_832", "832"}:
        return (832, 480) if orientation == "landscape" else (480, 832)
    if profile in {"hunyuan-480", "hunyuan_480", "social-848", "social_848", "native-480", "native_480", "848"}:
        return (848, 480) if orientation == "landscape" else (480, 848)
    if profile in {"social-864", "social_864", "864", "near720", "near-720", "near_720"}:
        return (864, 480) if orientation == "landscape" else (480, 864)
    if profile in {"social", "social-720", "social_720", "9:16", "16:9"}:
        return (800, 448) if orientation == "landscape" else (448, 800)
    render_size, _out_w, _out_h = _smartblog_render_profile(orientation=orientation)
    try:
        target_h_s, target_w_s = str(render_size).split("*", 1)
        return int(target_w_s), int(target_h_s)
    except Exception:
        return (832, 448) if orientation == "landscape" else (448, 832)


def _smartblog_ltx_output_dimensions(claim: dict[str, Any], *, width: int, height: int) -> tuple[int, int]:
    orientation = _smartblog_orientation_hint(claim)
    if orientation not in {"portrait", "landscape"}:
        orientation = "landscape" if int(width) >= int(height) else "portrait"
    _render_size, out_w, out_h = _smartblog_render_profile(orientation=str(orientation))
    return int(out_w), int(out_h)


def _smartblog_ltx_media_background_restore(*, face_restore: float, background_restore: float) -> float:
    return float(max(0.0, min(1.0, float(background_restore))))


def _smartblog_ltx_num_frames(claim: dict[str, Any], *, duration_sec: float | None = None) -> int:
    explicit_frames: int | None = None
    for src in _smartblog_ltx_source_dicts(claim):
        for key in _SMARTBLOG_LTX_FRAME_COUNT_KEYS:
            if key in src:
                value = _smartblog_optional_int(src.get(key))
                if value is not None:
                    explicit_frames = int(max(9, value))
                    break
    if explicit_frames is not None:
        default = int(explicit_frames)
    else:
        duration = 0.0
        try:
            duration = float(duration_sec or 0.0)
        except Exception:
            duration = 0.0
        if not math.isfinite(duration) or duration <= 0.0:
            duration = float(_smartblog_ltx_claim_duration_seconds(claim, default=0.0))
        if duration > 0.0:
            frame_rate = int(_smartblog_ltx_frame_rate(claim))
            default = int(max(9, math.ceil(float(duration) * float(frame_rate)) + 1))
        else:
            default = int(max(9, _safe_int_env("SMARTBLOG_HUNYUAN_NUM_FRAMES", _safe_int_env("SMARTBLOG_LTX_NUM_FRAMES", 121))))
    if _env_flag("SMARTBLOG_HUNYUAN_SNAP_NUM_FRAMES", os.getenv("SMARTBLOG_LTX_SNAP_NUM_FRAMES", "1") or "1"):
        return int(((int(default) - 2) // 8 + 1) * 8 + 1)
    return int(default)


def _smartblog_ltx_frame_rate(claim: dict[str, Any]) -> int:
    forced = str(os.getenv("SMARTBLOG_HUNYUAN_FORCE_FRAME_RATE") or os.getenv("SMARTBLOG_LTX_FORCE_FRAME_RATE", "") or "").strip()
    if forced:
        try:
            return int(max(1, min(60, int(float(forced)))))
        except Exception:
            pass
    value = int(max(1, min(60, _safe_int_env("SMARTBLOG_HUNYUAN_FRAME_RATE", _safe_int_env("SMARTBLOG_LTX_FRAME_RATE", 16)))))
    if _env_flag("SMARTBLOG_HUNYUAN_ALLOW_PAYLOAD_FPS", os.getenv("SMARTBLOG_LTX_ALLOW_PAYLOAD_FPS", "0") or "0"):
        for src in _smartblog_ltx_source_dicts(claim):
            for key in ("ltx_frame_rate", "ltxFrameRate", "frame_rate", "frameRate", "fps"):
                if key in src:
                    parsed = _smartblog_optional_int(src.get(key))
                    if parsed is not None:
                        value = int(max(1, min(60, parsed)))
                        break
    return int(value)


def _smartblog_ltx_seed(claim: dict[str, Any]) -> int:
    value = int(os.getenv("BASE_SEED", "420") or 420)
    for src in _smartblog_ltx_source_dicts(claim):
        for key in ("ltx_seed", "ltxSeed", "seed", "base_seed", "baseSeed"):
            if key in src:
                parsed = _smartblog_optional_int(src.get(key))
                if parsed is not None:
                    return int(parsed)
    return int(value)


def _smartblog_ltx_conditioning_strength(claim: dict[str, Any]) -> float:
    value = float(max(0.0, min(1.0, _safe_float_env("SMARTBLOG_HUNYUAN_CONDITIONING_STRENGTH", _safe_float_env("SMARTBLOG_LTX_CONDITIONING_STRENGTH", 1.0)))))
    for src in _smartblog_ltx_source_dicts(claim):
        for key in ("ltx_conditioning_strength", "ltxConditioningStrength", "conditioning_strength", "conditioningStrength"):
            if key in src:
                try:
                    return float(max(0.0, min(1.0, float(src.get(key)))))
                except Exception:
                    pass
    return float(value)


def _smartblog_hunyuan_latest_mp4(output_dir: str) -> str:
    root = os.path.abspath(str(output_dir or "").strip())
    candidates: list[tuple[float, str]] = []
    if not root or not os.path.isdir(root):
        return ""
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not str(filename).lower().endswith(".mp4"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                candidates.append((float(os.path.getmtime(path)), path))
            except Exception:
                pass
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])


def _smartblog_render_entry_prompt(claim: dict[str, Any], entry: dict[str, Any] | None) -> str:
    present, value = _smartblog_first_present(
        entry if isinstance(entry, dict) else {},
        "_smartblog_frame_video_prompt",
        "video_prompt",
        "videoPrompt",
        "prompt",
        "render_prompt",
        "renderPrompt",
    )
    if present:
        text = str(value or "").strip()
        if text:
            return text
        fallback = _smartblog_render_default_speaking_prompt()
        if fallback:
            return fallback
        raise RuntimeError("SmartBlog render frame video_prompt is empty and default speaking prompt is empty")
    return _smartblog_render_prompt(claim)


def _smartblog_render_entry_video_prompt(claim: dict[str, Any], entry: dict[str, Any] | None) -> str:
    present, value = _smartblog_first_present(
        entry if isinstance(entry, dict) else {},
        "_smartblog_frame_video_prompt",
        "video_prompt",
        "videoPrompt",
        "prompt",
        "render_prompt",
        "renderPrompt",
    )
    if present:
        return str(value or "").strip()
    return _smartblog_render_video_prompt(claim)


def _smartblog_render_entry_negative_prompt(claim: dict[str, Any], entry: dict[str, Any] | None) -> str:
    present, value = _smartblog_first_present(
        entry if isinstance(entry, dict) else {},
        "_smartblog_frame_video_negative_prompt",
        "video_negative_prompt",
        "videoNegativePrompt",
        "negative_prompt",
        "negativePrompt",
    )
    if present:
        return str(value or "").strip()
    return _smartblog_render_negative_prompt(claim)


def _smartblog_render_entry_filters(claim: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    filters = dict(_smartblog_filters(claim))
    src = entry if isinstance(entry, dict) else {}
    present, value = _smartblog_first_present(src, "_smartblog_frame_face_restore", "face_restore", "faceRestore")
    if present:
        filters["face_restore"] = value
    present, value = _smartblog_first_present(
        src,
        "_smartblog_frame_background_restore",
        "background_restore",
        "backgroundRestore",
    )
    if present:
        filters["background_restore"] = value
    return filters


def _smartblog_sanitize_static_negative_prompt(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    conflict_needles = (
        "静态",
        "静止",
        "不动",
        "static",
        "still image",
        "still frame",
        "motionless",
        "frozen",
    )
    # Negative prompts are normally comma-separated. Drop only conflicting
    # clauses so low-motion/static positive prompts are not contradicted by the
    # model's generic anti-frozen-frame defaults.
    raw_parts = re.split(r"([,，])", src)
    kept: list[str] = []
    removed = 0
    i = 0
    while i < len(raw_parts):
        part = str(raw_parts[i] or "")
        sep = str(raw_parts[i + 1] or "") if i + 1 < len(raw_parts) else ""
        body = part.strip()
        body_l = body.lower()
        drop = bool(body) and any(str(needle).lower() in body_l for needle in conflict_needles)
        if drop:
            removed += 1
        elif body:
            kept.append(body)
        i += 2
        _ = sep
    out = ", ".join(kept).strip()
    if removed > 0:
        logging.info(
            "SmartBlog render negative prompt sanitized: removed_static_conflicts=%d chars=%d->%d",
            int(removed),
            int(len(src)),
            int(len(out)),
        )
    return out


def _smartblog_orientation_hint(claim: dict[str, Any]) -> str:
    return smartblog_orientation_from_claim(claim)


def _smartblog_render_profile(*, orientation: str) -> tuple[str, int, int]:
    profile_name = str(os.getenv("SMARTBLOG_RENDER_VIDEO_PROFILE", "") or "").strip().lower()
    orientation_s = "landscape" if str(orientation or "").strip().lower() == "landscape" else "portrait"
    if _env_flag("SMARTBLOG_RENDER_VIDEO_HIGH_RES_2X", "0") or profile_name in {
        "832",
        "832p",
        "highres-2x",
        "highres_2x",
        "high-res-2x",
        "high_res_2x",
        "2x",
        "next",
    }:
        return _SMARTBLOG_RENDER_832P_PROFILES[orientation_s]
    if profile_name in {
        "compact",
        "compact704",
        "compact_704",
        "compact-704",
        "704",
        "704p",
        "b200",
        "b200_safe",
        "b200-safe",
    }:
        return _SMARTBLOG_RENDER_COMPACT_704_PROFILES[orientation_s]
    if profile_name in {
        "native720p",
        "native_720p",
        "native-720p",
        "native720",
        "native_720",
        "native-720",
        "social-native",
        "social_native",
        "full",
        "full720p",
        "full_720p",
    }:
        return _SMARTBLOG_RENDER_NATIVE_720P_PROFILES[orientation_s]
    if _env_flag("SMARTBLOG_RENDER_VIDEO_HIGH_RES", "0") or profile_name in {"720", "720p", "hd", "high", "high-res", "high_res"}:
        return _SMARTBLOG_RENDER_720P_PROFILES[orientation_s]
    profile = smartblog_live_profile_for_orientation(str(orientation))
    return (str(profile.render_size), int(profile.output_width), int(profile.output_height))


def _smartblog_render_output_fps() -> float:
    raw = str(os.getenv("SMARTBLOG_RENDER_VIDEO_OUTPUT_FPS", "30") or "30").strip()
    try:
        fps = float(raw)
    except Exception:
        fps = 30.0
    return float(max(1.0, min(120.0, fps)))


def _smartblog_render_source_fps() -> float:
    return float(max(1, int(WORKER_FPS)))


def _smartblog_render_delivery_fps() -> float:
    raw = str(
        os.getenv("SMARTBLOG_RENDER_VIDEO_DELIVERY_FPS")
        or os.getenv("SMARTBLOG_RENDER_VIDEO_FINAL_FPS")
        or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_FPS")
        or "30"
    ).strip()
    try:
        fps = float(raw)
    except Exception:
        fps = 30.0
    return float(max(1.0, min(120.0, fps)))


def _smartblog_remote_finalizer_upscale_enabled(claim: dict[str, Any] | None = None) -> bool:
    claim = claim if isinstance(claim, dict) else {}
    payload = _smartblog_job_payload(claim)
    job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
    video = _smartblog_video_config(claim)
    assets = claim.get("assets") if isinstance(claim.get("assets"), dict) else {}
    sources: list[dict[str, Any]] = []
    for src in (video, assets, payload, job, claim):
        if not isinstance(src, dict):
            continue
        sources.append(src)
        for nested_key in (
            "video",
            "render",
            "render_options",
            "output",
            "delivery",
            "postprocess",
            "postprocessing",
            "post_processing",
            "processing",
            "spec",
        ):
            nested = src.get(nested_key)
            if isinstance(nested, dict):
                sources.append(nested)
    for src in sources:
        for key in ("upscale", "upscale_enabled", "upscaleEnabled", "super_resolution", "superResolution"):
            if key not in src:
                continue
            parsed = _smartblog_optional_bool(src.get(key))
            if parsed is not None:
                return bool(parsed)
    env_value = _smartblog_optional_bool(
        os.getenv("SMARTBLOG_RENDER_DELIVERY_UPSCALE_ENABLED")
        if os.getenv("SMARTBLOG_RENDER_DELIVERY_UPSCALE_ENABLED") is not None
        else os.getenv("SMARTBLOG_RENDER_UPSCALE_ENABLED")
    )
    if env_value is not None:
        return bool(env_value)
    return False


def _smartblog_remote_finalizer_quality_pass_enabled() -> bool:
    """Whether the remote media worker should run the VSR/deblur quality pass.

    This is intentionally separate from delivery upscale. SmartBlog render jobs
    normally stay at 720p delivery, but still need one VSR quality pass before
    the final resize/encode step.
    """
    env_value = _smartblog_optional_bool(
        os.getenv("SMARTBLOG_RENDER_QUALITY_PASS_ENABLED")
        if os.getenv("SMARTBLOG_RENDER_QUALITY_PASS_ENABLED") is not None
        else os.getenv("REMOTE_EDGE_FILE_UPSCALE_ENABLED")
    )
    if env_value is not None:
        return bool(env_value)
    return False


def _smartblog_render_delivery_dimensions(
    claim: dict[str, Any] | None,
    *,
    output_width: int,
    output_height: int,
) -> tuple[int, int]:
    width_env = _smartblog_optional_int(
        os.getenv("SMARTBLOG_RENDER_DELIVERY_WIDTH")
        or os.getenv("SMARTBLOG_RENDER_VIDEO_DELIVERY_WIDTH")
        or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_WIDTH")
    )
    height_env = _smartblog_optional_int(
        os.getenv("SMARTBLOG_RENDER_DELIVERY_HEIGHT")
        or os.getenv("SMARTBLOG_RENDER_VIDEO_DELIVERY_HEIGHT")
        or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_HEIGHT")
    )
    if width_env and height_env:
        return int(max(2, width_env - (width_env % 2))), int(max(2, height_env - (height_env % 2)))
    if not _smartblog_remote_finalizer_upscale_enabled(claim):
        return int(output_width), int(output_height)
    long_edge = _smartblog_optional_int(
        os.getenv("SMARTBLOG_RENDER_UPSCALE_LONG_EDGE")
        or os.getenv("REMOTE_EDGE_FILE_UPSCALE_LONG_EDGE")
        or "1920"
    ) or 1920
    short_edge = _smartblog_optional_int(
        os.getenv("SMARTBLOG_RENDER_UPSCALE_SHORT_EDGE")
        or os.getenv("REMOTE_EDGE_FILE_UPSCALE_SHORT_EDGE")
        or "1080"
    ) or 1080
    if int(output_width) >= int(output_height):
        return int(long_edge), int(short_edge)
    return int(short_edge), int(long_edge)


def _validate_smartblog_render_size(size: str) -> str:
    size_s = str(size or "").strip()
    if size_s not in _SMARTBLOG_VALID_RENDER_SIZES:
        raise RuntimeError(f"unsupported render size for current model runtime: {size_s!r}")
    return size_s


def _smartblog_progress_stage_fields(
    *,
    job_type: str,
    stage: str,
    stage_label: str | None = None,
) -> dict[str, Any]:
    stage_s = str(stage or "").strip()
    if not stage_s:
        raise RuntimeError("SmartBlog progress stage is required")
    order = tuple(_SMARTBLOG_RENDER_PROGRESS_STAGES.get(str(job_type or "").strip().lower(), ()))
    stage_index: int | None = None
    stage_total: int | None = None
    if order:
        stage_total = int(len(order))
        if stage_s in order:
            stage_index = int(order.index(stage_s) + 1)
    label_s = str(stage_label or "").strip() or _SMARTBLOG_STAGE_LABELS.get(stage_s, stage_s.replace("_", " ").title())
    return {
        "stage": stage_s,
        "stage_label": label_s,
        "stage_index": stage_index,
        "stage_total": stage_total,
    }


def _smartblog_stage_progress_total(
    *,
    job_type: str,
    stage: str,
    stage_progress: float = 1.0,
) -> int:
    job_type_s = str(job_type or "").strip().lower()
    stage_s = str(stage or "").strip()
    progress_f = float(max(0.0, min(1.0, float(stage_progress))))
    order = tuple(_SMARTBLOG_RENDER_PROGRESS_STAGES.get(job_type_s, ()))
    weights = _SMARTBLOG_RENDER_PROGRESS_WEIGHTS.get(job_type_s) or {}
    if order and stage_s in order and all(str(name or "").strip() in weights for name in order):
        done = int(sum(int(weights[str(name).strip()]) for name in order[: order.index(stage_s)]))
        total = done + int(weights[stage_s]) * progress_f
        return int(min(100, max(0, int(total))))
    if order and stage_s in order:
        done = float(order.index(stage_s))
        total = ((done + progress_f) / float(len(order))) * 100.0
        return int(min(100, max(0, int(total))))
    return int(min(100, max(0, int(progress_f * 100.0))))


class SmartBlogRenderJobsMixin:
    def _smartblog_job_run_dir(self, job_id: str) -> str:
        base = os.path.abspath("./worker_runs")
        path = os.path.join(base, sanitize_job_id(str(job_id or "") or f"smartblog_{int(time.time() * 1000)}"))
        os.makedirs(path, exist_ok=True)
        return path

    def _smartblog_storage_url_cache(self) -> dict[str, str]:
        cache = getattr(self, "_smartblog_storage_download_urls", None)
        if not isinstance(cache, dict):
            cache = {}
            self._smartblog_storage_download_urls = cache
        return cache

    def _smartblog_storage_cache_keys(self, path: str) -> list[str]:
        raw = str(path or "").strip().lstrip("/")
        if not raw:
            return []
        keys = [raw]
        if raw.startswith("generated-assets/"):
            keys.append(raw[len("generated-assets/") :].lstrip("/"))
        parts = raw.split("/", 1)
        if len(parts) == 2 and parts[0] in {
            "builder-renders",
            "generated-assets",
            "worker-uploads",
        }:
            keys.append(parts[1].lstrip("/"))
        return list(dict.fromkeys(k for k in keys if k))

    def _smartblog_remember_storage_urls(self, path: str, *urls: str) -> None:
        url = ""
        for candidate in urls:
            candidate_s = str(candidate or "").strip()
            if candidate_s and _smartblog_http_url(candidate_s):
                url = candidate_s
                break
        if not url:
            return
        cache = self._smartblog_storage_url_cache()
        for key in self._smartblog_storage_cache_keys(path):
            cache[key] = url

    def _smartblog_cached_storage_download_url(self, path: str) -> str:
        cache = self._smartblog_storage_url_cache()
        for key in self._smartblog_storage_cache_keys(path):
            url = str(cache.get(key) or "").strip()
            if url:
                return url
        return ""

    async def _smartblog_progress_checked(
        self,
        *,
        job_id: str,
        progress: int,
        stage: str,
        stage_label: str | None = None,
        stage_index: int | None = None,
        stage_total: int | None = None,
    ) -> None:
        progress_payload = {
            "job_id": str(job_id),
            "progress": int(min(100, max(0, int(progress)))),
            "stage": str(stage),
            "stage_label": stage_label,
            "stage_index": stage_index,
            "stage_total": stage_total,
        }
        resp = await self._smartblog_api.progress(
            job_id=str(job_id),
            progress=int(progress_payload["progress"]),
            stage=str(stage),
            stage_label=stage_label,
            stage_index=stage_index,
            stage_total=stage_total,
        )
        self._last_progress_ok_mono = float(time.monotonic())
        self._smartblog_last_progress_payload = dict(progress_payload)
        if isinstance(resp, dict) and (resp.get("success") is False or resp.get("ok") is False):
            self._session_stop_sent_by_server = True
            raise SmartBlogJobStoppedByServer(
                smartblog_api_rejection_reason(resp, default="SmartBlog progress rejected by API")
            )
        if bool(resp.get("stop")):
            self._session_stop_sent_by_server = True
            raise SmartBlogJobStoppedByServer(
                str(resp.get("reason") or resp.get("status") or "job stopped by server").strip()
            )

    async def _smartblog_progress_keepalive_loop(
        self,
        *,
        job_id: str,
        job_type: str,
        stop_event: asyncio.Event,
    ) -> None:
        heartbeat_sec = max(5.0, min(45.0, _safe_float_env("SMARTBLOG_RENDER_JOB_HEARTBEAT_SEC", 20.0)))
        stale_sec = max(heartbeat_sec, min(55.0, _safe_float_env("SMARTBLOG_RENDER_JOB_HEARTBEAT_STALE_SEC", 35.0)))
        default_payload = {
            "job_id": str(job_id),
            "progress": max(
                1,
                _smartblog_stage_progress_total(
                    job_type=str(job_type),
                    stage="prepare",
                    stage_progress=0.1,
                ),
            ),
            **_smartblog_progress_stage_fields(
                job_type=str(job_type),
                stage="prepare",
                stage_label="Worker heartbeat",
            ),
        }
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=float(heartbeat_sec))
                break
            except asyncio.TimeoutError:
                pass
            now = float(time.monotonic())
            last_ok = float(getattr(self, "_last_progress_ok_mono", 0.0) or 0.0)
            if last_ok > 0.0 and (now - last_ok) < float(stale_sec):
                continue
            payload = dict(getattr(self, "_smartblog_last_progress_payload", {}) or {})
            if str(payload.get("job_id") or "") != str(job_id):
                payload = dict(default_payload)
            try:
                await self._smartblog_progress_checked(
                    job_id=str(job_id),
                    progress=int(payload.get("progress") or default_payload["progress"]),
                    stage=str(payload.get("stage") or default_payload["stage"]),
                    stage_label=payload.get("stage_label") or default_payload.get("stage_label"),
                    stage_index=payload.get("stage_index") or default_payload.get("stage_index"),
                    stage_total=payload.get("stage_total") or default_payload.get("stage_total"),
                )
            except SmartBlogJobStoppedByServer as e:
                logging.warning("SmartBlog render keepalive stopped by server: id=%s err=%s", str(job_id), e)
                self._session_stop_sent_by_server = True
                try:
                    self._render_cancel.set()
                except Exception:
                    pass
                task = getattr(self, "_active_smartblog_job_task", None)
                if isinstance(task, asyncio.Task) and (not task.done()):
                    task.cancel()
                return
            except Exception as e:
                logging.warning("SmartBlog render keepalive failed open: id=%s err=%s", str(job_id), e)

    async def _smartblog_wait_with_progress(
        self,
        *,
        task: asyncio.Task,
        job_id: str,
        progress: int,
        stage: str,
        stage_label: str | None = None,
        stage_index: int | None = None,
        stage_total: int | None = None,
        heartbeat_sec: float | None = None,
        cancel_model_infer_on_stop: bool = False,
        progress_provider: Any | None = None,
    ):
        heartbeat = float(heartbeat_sec) if heartbeat_sec is not None else float(
            _safe_float_env("SMARTBLOG_RENDER_PROGRESS_SEC", 1.0)
        )
        while True:
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=float(max(0.5, heartbeat)))
            except asyncio.TimeoutError:
                progress_kwargs = {
                    "progress": int(progress),
                    "stage": str(stage),
                    "stage_label": stage_label,
                    "stage_index": stage_index,
                    "stage_total": stage_total,
                }
                if progress_provider is not None:
                    try:
                        provided = progress_provider()
                        if isinstance(provided, dict):
                            progress_kwargs.update(provided)
                    except Exception as e:
                        logging.warning("SmartBlog progress provider failed: job=%s err=%s", job_id, e)
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=int(progress_kwargs.get("progress") or 0),
                    stage=str(progress_kwargs.get("stage") or stage),
                    stage_label=progress_kwargs.get("stage_label"),
                    stage_index=progress_kwargs.get("stage_index"),
                    stage_total=progress_kwargs.get("stage_total"),
                )
            except SmartBlogJobStoppedByServer:
                if bool(cancel_model_infer_on_stop):
                    try:
                        await self._model_client.cancel_active_infer(reason="smartblog_render_stopped")
                    except Exception as e:
                        logging.warning("SmartBlog render stop cancel_active_infer failed: job=%s err=%s", job_id, e)
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise

    async def _smartblog_download_file(self, *, url: str, out_path: str) -> str:
        src = await self._smartblog_resolve_download_url(str(url or "").strip())
        dst = os.path.abspath(str(out_path or "").strip())
        if not src:
            raise RuntimeError("download URL is required")
        if not dst:
            raise RuntimeError("download destination path is required")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp_path = f"{dst}.tmp"
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=600.0, write=60.0, pool=60.0), follow_redirects=True) as client:
            async with client.stream("GET", src) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            f.write(chunk)
        os.replace(tmp_path, dst)
        return dst

    async def _smartblog_resolve_download_url(self, src: str) -> str:
        raw = str(src or "").strip()
        if not raw:
            return ""
        cached = self._smartblog_cached_storage_download_url(raw)
        if cached:
            return cached
        storage_path = _smartblog_generated_assets_path(raw)
        if not storage_path:
            return raw
        cached = self._smartblog_cached_storage_download_url(storage_path)
        if cached:
            return cached

        try:
            resp = await self._smartblog_api.get_download_url(path=str(storage_path))
            smartblog_validate_action_response(resp, action="get_download_url")
            signed = str(
                resp.get("download_url")
                or resp.get("signed_url")
                or resp.get("signedURL")
                or resp.get("signedUrl")
                or resp.get("public_url")
                or resp.get("url")
                or ""
            ).strip()
            resp_path = str(resp.get("path") or resp.get("storage_path") or storage_path).strip()
            if signed:
                self._smartblog_remember_storage_urls(resp_path, signed)
                self._smartblog_remember_storage_urls(storage_path, signed)
                return signed
        except Exception as e:
            logging.debug("SmartBlog get_download_url fallback: path=%s err=%s", storage_path, e)

        service_key = smartblog_supabase_service_role_key()
        path = urllib.parse.quote(storage_path.lstrip("/"), safe="/")
        endpoint = f"{smartblog_supabase_url().rstrip('/')}/storage/v1/object/sign/generated-assets/{path}"
        headers = {
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=60.0, write=20.0, pool=60.0)) as client:
            resp = await client.post(endpoint, headers=headers, json={"expiresIn": 3600})
            resp.raise_for_status()
            payload = resp.json()
        signed = str(payload.get("signedURL") or payload.get("signedUrl") or payload.get("signed_url") or "").strip()
        if not signed:
            raise RuntimeError(f"Supabase signed download URL response missing signedURL for {storage_path}")
        if _smartblog_http_url(signed):
            return signed
        if signed.startswith("/"):
            return f"{smartblog_supabase_url().rstrip('/')}/storage/v1{signed}"
        return f"{smartblog_supabase_url().rstrip('/')}/storage/v1/{signed.lstrip('/')}"

    async def _smartblog_upload_file(
        self,
        *,
        signed_url: str,
        file_path: str,
        content_type: str,
        progress_state: dict[str, Any] | None = None,
    ) -> None:
        url = str(signed_url or "").strip()
        src = os.path.abspath(str(file_path or "").strip())
        if not url:
            raise RuntimeError("upload.signed_url is required")
        if not src or not os.path.exists(src):
            raise RuntimeError(f"upload source file missing: {src}")

        def _upload_sync() -> None:
            put_file_to_signed_url(
                signed_url=str(url),
                path=str(src),
                content_type=str(content_type or "application/octet-stream"),
                connect_timeout=20.0,
                read_timeout=1800.0,
                progress_state=progress_state,
                env_prefix="SMARTBLOG_SIGNED_UPLOAD",
                log_prefix="smartblog-worker-upload",
            )

        await asyncio.to_thread(_upload_sync)

    @staticmethod
    def _smartblog_remote_service_enabled(service_url: str, env_name: str) -> bool:
        if _env_flag(str(env_name), "0"):
            return True
        return bool(str(service_url or "").strip() and not _smartblog_service_url_is_local(str(service_url)))

    @staticmethod
    def _smartblog_local_media_services_forbidden() -> bool:
        profile = str(os.getenv("WORKER_PROFILE_NAME", "") or "").strip().lower()
        default = "1" if profile == "b200-avatar-commander" else "0"
        return _env_flag("SMARTBLOG_FORBID_LOCAL_MEDIA_SERVICES", default)

    @staticmethod
    def _smartblog_file_base64_payload(path: str, *, content_type: str | None = None) -> dict[str, Any]:
        src = os.path.abspath(str(path or "").strip())
        if not src or not os.path.exists(src):
            raise RuntimeError(f"remote service input file missing: {src}")
        with open(src, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return {
            "filename": os.path.basename(src),
            "content_type": str(content_type or _smartblog_guess_content_type(src)),
            "data": data,
        }

    async def _smartblog_prepare_remote_service_output_upload(
        self,
        *,
        job_id_hint: str,
        folder: str,
        filename: str,
        content_type: str,
    ) -> tuple[str, str, str]:
        clean_folder = str(folder or "remote-media").replace("\\", "/").strip("/")
        clean_filename = os.path.basename(str(filename or "").strip()) or "output.bin"
        upload_plan = SmartBlogRenderFinalizePlan(
            job_id=str(job_id_hint or "remote-media"),
            job_type=SMARTBLOG_JOB_TYPE_RENDER_VIDEO,
            signed_url="",
            upload_path=f"{clean_folder}/{clean_filename}",
            file_path="",
            content_type=str(content_type or "application/octet-stream"),
            complete_kwargs={},
            run_dir="",
        )
        signed_url, upload_path = await self._smartblog_resolve_upload_target(upload_plan)
        return str(signed_url), str(upload_path), self._smartblog_public_storage_url(str(upload_path))

    async def _smartblog_download_remote_service_output(
        self,
        *,
        data: dict[str, Any],
        local_path: str,
        log_prefix: str,
        label: str,
    ) -> str:
        out_local = os.path.abspath(str(local_path or "").strip())
        if not out_local:
            raise RuntimeError(f"{label} remote service local output path is empty")
        os.makedirs(os.path.dirname(out_local) or ".", exist_ok=True)
        output_url = str(
            data.get("output_storage_path")
            or data.get("storage_path")
            or data.get("signed_url")
            or data.get("output_url")
            or data.get("public_url")
            or data.get("download_url")
            or ""
        ).strip()
        if output_url:
            await self._smartblog_download_file(url=str(output_url), out_path=str(out_local))
            logging.warning(
                "%s %s remote output downloaded: %s bytes=%d",
                str(log_prefix),
                str(label),
                os.path.basename(str(out_local)),
                int(os.path.getsize(out_local)) if os.path.exists(out_local) else 0,
            )
            return str(out_local)
        output_b64 = str(data.get("output_base64") or data.get("base64") or "").strip()
        if output_b64:
            if "," in output_b64[:128]:
                output_b64 = output_b64.split(",", 1)[1]
            with open(out_local, "wb") as f:
                f.write(base64.b64decode(output_b64))
            return str(out_local)
        output_path = str(data.get("output_path") or "").strip()
        if output_path and os.path.exists(output_path):
            return str(output_path)
        raise RuntimeError(f"{log_prefix} {label} remote service response has no downloadable output: {data}")

    async def _smartblog_upload_remote_service_source(
        self,
        *,
        source_path: str,
        job_id_hint: str,
        folder: str,
        filename: str,
        content_type: str,
    ) -> str:
        signed_url, upload_path, public_url = await self._smartblog_prepare_remote_service_output_upload(
            job_id_hint=str(job_id_hint),
            folder=str(folder),
            filename=str(filename),
            content_type=str(content_type),
        )
        await self._smartblog_upload_file(
            signed_url=str(signed_url),
            file_path=str(source_path),
            content_type=str(content_type),
        )
        try:
            download_url = await self._smartblog_resolve_download_url(str(upload_path))
            if download_url:
                return str(download_url)
        except Exception:
            logging.exception(
                "SmartBlog remote service source signed download URL failed; falling back to upload public URL: path=%s",
                str(upload_path or "-"),
            )
        return str(public_url)

    async def _smartblog_run_remote_musetalk(
        self,
        *,
        source_url: str,
        signed_url: str,
        content_type: str,
        source_fps: int = 0,
    ) -> dict[str, Any]:
        service_url = str(_smartblog_file_musetalk_service_url() or "").strip()
        if not service_url:
            raise RuntimeError("SMARTBLOG_MUSETALK_SERVICE_URL/REMOTE_EDGE_FILE_MUSETALK_SERVICE_URL is required")
        endpoint = str(service_url).rstrip("/")
        if not endpoint.endswith("/lipsync"):
            endpoint = f"{endpoint}/lipsync"
        src = str(source_url or "").strip()
        upload_url = str(signed_url or "").strip()
        if not src:
            raise RuntimeError("remote MuseTalk source_url is required")
        if not upload_url:
            raise RuntimeError("remote MuseTalk signed upload URL is required")

        def _run_sync() -> dict[str, Any]:
            import requests

            secret = str(
                os.getenv("SMARTBLOG_MUSETALK_SHARED_SECRET")
                or os.getenv("REMOTE_EDGE_FILE_MUSETALK_SHARED_SECRET")
                or ""
            ).strip()
            headers = {"Authorization": f"Bearer {secret}"} if secret else {}
            files: dict[str, tuple[None, str]] = {
                "source_url": (None, str(src)),
                "upload_url": (None, str(upload_url)),
                "upload_content_type": (None, str(content_type or "video/mp4")),
                "backend": (None, str(os.getenv("SMARTBLOG_RENDER_MUSETALK_BACKEND", "resident") or "resident")),
                "version": (None, str(os.getenv("SMARTBLOG_RENDER_MUSETALK_VERSION", "v15") or "v15")),
                "use_float16": (
                    None,
                    "1" if _env_flag("SMARTBLOG_RENDER_MUSETALK_USE_FLOAT16", "1") else "0",
                ),
                "batch_size": (
                    None,
                    str(max(1, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_BATCH_SIZE", 12))),
                ),
                "bbox_shift": (None, str(_safe_int_env("SMARTBLOG_RENDER_MUSETALK_BBOX_SHIFT", 0))),
                "parsing_mode": (None, str(os.getenv("SMARTBLOG_RENDER_MUSETALK_PARSING_MODE", "jaw") or "jaw")),
                "extra_margin": (None, str(max(0, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_EXTRA_MARGIN", 10)))),
                "left_cheek_width": (
                    None,
                    str(max(0, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_LEFT_CHEEK_WIDTH", 90))),
                ),
                "right_cheek_width": (
                    None,
                    str(max(0, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_RIGHT_CHEEK_WIDTH", 90))),
                ),
                "fixed_bbox": (
                    None,
                    "1" if _env_flag("SMARTBLOG_RENDER_MUSETALK_FIXED_BBOX", "1") else "0",
                ),
                "bbox_sample_frames": (
                    None,
                    str(max(1, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_BBOX_SAMPLE_FRAMES", 5))),
                ),
                "mask_stride": (
                    None,
                    str(max(1, _safe_int_env("SMARTBLOG_RENDER_MUSETALK_MASK_STRIDE", 999999))),
                ),
            }
            force_fps_env = _safe_int_env("SMARTBLOG_RENDER_MUSETALK_FORCE_PROCESSING_FPS", 0)
            force_fps = int(force_fps_env or source_fps or 0)
            if int(force_fps) > 0:
                files["force_processing_fps"] = (None, str(int(force_fps)))
            resp = requests.post(
                endpoint,
                files=files,
                headers=headers,
                timeout=(
                    max(3.0, _safe_float_env("SMARTBLOG_MUSETALK_CONNECT_TIMEOUT_SEC", 20.0)),
                    max(30.0, _safe_float_env("SMARTBLOG_MUSETALK_READ_TIMEOUT_SEC", 3600.0)),
                ),
            )
            if int(resp.status_code) != 200:
                body = str(getattr(resp, "text", "") or "")[-4000:]
                raise RuntimeError(f"remote MuseTalk failed HTTP {resp.status_code}: {body}")
            try:
                payload = dict(resp.json() or {})
            except Exception as e:
                raise RuntimeError("remote MuseTalk returned non-JSON response") from e
            if not bool(payload.get("uploaded")):
                raise RuntimeError(f"remote MuseTalk did not confirm upload: keys={sorted(payload.keys())}")
            return payload

        started = float(time.perf_counter())
        payload = await asyncio.to_thread(_run_sync)
        logging.warning(
            "SmartBlog remote MuseTalk complete: backend=%s source_bytes=%s bytes=%s frames=%s fps=%s elapsed=%.3fs timings=%s",
            str(payload.get("backend", "")),
            str(payload.get("source_bytes", "")),
            str(payload.get("bytes", "")),
            str(payload.get("frames", "")),
            str(payload.get("fps", payload.get("source_fps", ""))),
            float(time.perf_counter() - float(started)),
            json.dumps(payload.get("timings_sec") or {}, ensure_ascii=True, sort_keys=True),
        )
        return payload

    async def _smartblog_run_remote_file_finalizer(
        self,
        *,
        source_url: str,
        signed_url: str,
        content_type: str,
        source_fps: int = 0,
        target_width: int = 0,
        target_height: int = 0,
        target_fps: int = 0,
        upscale_enabled: bool = False,
        background_music_url: str = "",
        background_music_gain_db: float = 0.0,
        background_music_loop: bool = True,
        background_music_duck_voice_db: float = 0.0,
        background_music_fade_in_seconds: float = 0.0,
        background_music_fade_out_seconds: float = 0.0,
        background_music_start_offset_seconds: float = 0.0,
        subtitle_chunks_json: str = "",
        watermark_text: str = "",
        poster_signed_url: str = "",
        poster_upload_path: str = "",
        poster_content_type: str = "image/jpeg",
        progress_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service_url, media_worker_lease = await _smartblog_acquire_media_worker_lease(
            "finalizer",
            _smartblog_file_upscale_fallback_service_url(),
            log_prefix="remote-finalizer",
        )
        if not service_url:
            raise RuntimeError("REMOTE_EDGE_FILE_UPSCALE_SERVICE_URL/SMARTBLOG_FILE_UPSCALE_SERVICE_URL is required")
        src = str(source_url or "").strip()
        upload_url = str(signed_url or "").strip()
        if not src:
            raise RuntimeError("remote finalizer source_url is required")
        if not upload_url:
            raise RuntimeError("remote finalizer signed upload URL is required")

        def _run_sync() -> dict[str, Any]:
            import requests

            secret = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_SHARED_SECRET", "") or "").strip()
            headers = {"Authorization": f"Bearer {secret}"} if secret else {}
            connect_timeout = max(3.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_CONNECT_TIMEOUT_SEC", 20.0))
            read_timeout = max(30.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_READ_TIMEOUT_SEC", 3600.0))
            service_retry_sec = max(
                0.0,
                _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_SERVICE_UNAVAILABLE_RETRY_SEC", 600.0),
            )
            service_retry_poll_sec = max(
                1.0,
                _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_SERVICE_UNAVAILABLE_RETRY_POLL_SEC", 10.0),
            )
            service_retry_deadline = float(time.monotonic()) + float(service_retry_sec)
            service_transient_codes = {404, 429, 500, 502, 503, 504}

            def _retry_service_unavailable(*, where: str, status_code: int, body: str = "") -> bool:
                if int(status_code) not in service_transient_codes:
                    return False
                if service_retry_sec <= 0.0 or float(time.monotonic()) >= service_retry_deadline:
                    return False
                wait_sec = min(float(service_retry_poll_sec), max(0.0, service_retry_deadline - float(time.monotonic())))
                logging.warning(
                    "SmartBlog remote finalizer service unavailable: where=%s HTTP %s; retrying in %.1fs for up to %.1fs body=%s",
                    str(where),
                    int(status_code),
                    float(wait_sec),
                    max(0.0, float(service_retry_deadline) - float(time.monotonic())),
                    str(body or "")[-500:],
                )
                time.sleep(float(wait_sec))
                return True

            params: dict[str, str] = {
                "quality": str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_QUALITY", "DEBLUR_HIGH") or "DEBLUR_HIGH"),
                "scale": str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_SCALE", "1") or "1"),
                "upscale": "1" if bool(upscale_enabled) else "0",
            }
            remote_rife = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE", "") or "").strip()
            if remote_rife:
                params["rife"] = str(remote_rife)
            fps_s = str(target_fps or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_FPS", "") or "").strip()
            if fps_s:
                params["target_fps"] = str(fps_s)
            source_fps_s = str(source_fps or os.getenv("REMOTE_EDGE_FILE_UPSCALE_SOURCE_FPS", "") or "").strip()
            if source_fps_s:
                params["source_fps"] = str(source_fps_s)
            width_s = str(target_width or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_WIDTH", "") or "").strip()
            height_s = str(target_height or os.getenv("REMOTE_EDGE_FILE_UPSCALE_TARGET_HEIGHT", "") or "").strip()
            if width_s:
                params["target_width"] = str(width_s)
            if height_s:
                params["target_height"] = str(height_s)
            remote_rife_batch = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE_BATCH_SOURCE_FRAMES", "") or "").strip()
            if remote_rife_batch:
                params["rife_batch_source_frames"] = str(remote_rife_batch)
            remote_rife_stage = str(os.getenv("REMOTE_EDGE_FILE_UPSCALE_RIFE_STAGE", "") or "").strip()
            if remote_rife_stage:
                params["rife_stage"] = str(remote_rife_stage)
            files = {
                "source_url": (None, str(src)),
                "upload_url": (None, str(upload_url)),
                "upload_content_type": (None, str(content_type or "video/mp4")),
            }
            poster_url = str(poster_signed_url or "").strip()
            if poster_url:
                files["poster_upload_url"] = (None, poster_url)
                files["poster_upload_content_type"] = (None, str(poster_content_type or "image/jpeg"))
                if str(poster_upload_path or "").strip():
                    files["poster_storage_path"] = (None, str(poster_upload_path).strip())
            subtitles_payload = str(subtitle_chunks_json or "").strip()
            if subtitles_payload:
                files["subtitle_chunks_json"] = (None, subtitles_payload)
            watermark_payload = normalize_watermark_text(str(watermark_text or ""))
            if watermark_payload:
                files["watermark_text"] = (None, str(watermark_payload))
            music_url = str(background_music_url or "").strip()
            if music_url:
                files.update(
                    {
                        "background_music_url": (None, str(music_url)),
                        "background_music_gain_db": (None, f"{float(background_music_gain_db or 0.0):.6f}"),
                        "background_music_loop": (None, "1" if bool(background_music_loop) else "0"),
                        "background_music_duck_voice_db": (
                            None,
                            f"{float(background_music_duck_voice_db or 0.0):.6f}",
                        ),
                        "background_music_fade_in_seconds": (
                            None,
                            f"{float(background_music_fade_in_seconds or 0.0):.6f}",
                        ),
                        "background_music_fade_out_seconds": (
                            None,
                            f"{float(background_music_fade_out_seconds or 0.0):.6f}",
                        ),
                        "background_music_start_offset_seconds": (
                            None,
                            f"{float(background_music_start_offset_seconds or 0.0):.6f}",
                        ),
                    }
                )
            if _env_flag("REMOTE_EDGE_FILE_UPSCALE_ASYNC", "1"):
                base_url = str(service_url).rstrip("/")
                jobs_url = f"{base_url}/jobs" if base_url.endswith("/upscale") else f"{base_url}/upscale/jobs"
                async_started = False
                try:
                    while True:
                        start_resp = requests.post(
                            jobs_url,
                            params=params,
                            files=files,
                            headers=headers,
                            timeout=(
                                connect_timeout,
                                max(10.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_ASYNC_START_TIMEOUT_SEC", 90.0)),
                            ),
                        )
                        if int(start_resp.status_code) in {429, 500, 502, 503, 504}:
                            body = str(getattr(start_resp, "text", "") or "")[-4000:]
                            if _retry_service_unavailable(
                                where="async_start",
                                status_code=int(start_resp.status_code),
                                body=body,
                            ):
                                continue
                        break
                    if int(start_resp.status_code) not in {404, 405}:
                        if int(start_resp.status_code) < 200 or int(start_resp.status_code) >= 300:
                            body = str(getattr(start_resp, "text", "") or "")[-4000:]
                            raise RuntimeError(f"remote finalizer async start failed HTTP {start_resp.status_code}: {body}")
                        start_payload = dict(start_resp.json() or {})
                        async_job_id = str(start_payload.get("job_id") or start_payload.get("id") or "").strip()
                        if not async_job_id:
                            raise RuntimeError(
                                "remote finalizer async start returned no job_id: "
                                f"{json.dumps(start_payload, ensure_ascii=False)[:1000]}"
                            )
                        async_started = True
                        status_url = f"{jobs_url.rstrip('/')}/{async_job_id}"
                        poll_sec = max(1.0, _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_ASYNC_POLL_SEC", 5.0))
                        deadline = time.monotonic() + float(read_timeout)
                        transient_status_deadline = 0.0
                        last_status = ""
                        while True:
                            if time.monotonic() >= deadline:
                                raise RuntimeError(
                                    f"remote finalizer async timed out after {read_timeout:.1f}s job_id={async_job_id}"
                                )
                            time.sleep(float(poll_sec))
                            status_resp = requests.get(
                                status_url,
                                headers=headers,
                                timeout=(connect_timeout, max(10.0, min(60.0, float(read_timeout)))),
                            )
                            if int(status_resp.status_code) < 200 or int(status_resp.status_code) >= 300:
                                body = str(getattr(status_resp, "text", "") or "")[-2000:]
                                if int(status_resp.status_code) in {429, 500, 502, 503, 504}:
                                    now_mono = float(time.monotonic())
                                    if transient_status_deadline <= 0.0:
                                        transient_status_deadline = now_mono + max(
                                            30.0,
                                            _safe_float_env(
                                                "REMOTE_EDGE_FILE_UPSCALE_ASYNC_STATUS_TRANSIENT_SEC",
                                                240.0,
                                            ),
                                        )
                                        logging.warning(
                                            "SmartBlog remote finalizer async transient status error: "
                                            "job=%s HTTP %s; retrying status for %.1fs",
                                            async_job_id,
                                            int(status_resp.status_code),
                                            max(0.0, float(transient_status_deadline) - now_mono),
                                        )
                                    if now_mono < transient_status_deadline:
                                        continue
                                raise RuntimeError(
                                    f"remote finalizer async status failed HTTP {status_resp.status_code}: {body}"
                                )
                            transient_status_deadline = 0.0
                            status_payload = dict(status_resp.json() or {})
                            status = str(status_payload.get("status") or "").strip().lower()
                            stage = str(status_payload.get("stage") or "").strip()
                            progress_value = status_payload.get("progress")
                            frames_value = status_payload.get("frames")
                            total_frames_value = status_payload.get("total_frames")
                            if isinstance(progress_state, dict):
                                progress_state["status"] = status
                                progress_state["stage"] = stage
                                progress_state["progress"] = progress_value
                                progress_state["updated_at_mono"] = float(time.monotonic())
                            status_key = (
                                f"{status}:{stage}:{progress_value}:"
                                f"{frames_value}/{total_frames_value}"
                            )
                            if status_key != last_status:
                                logging.warning(
                                    "SmartBlog remote finalizer async status: job=%s status=%s stage=%s progress=%s frames=%s/%s",
                                    async_job_id,
                                    status,
                                    stage,
                                    str(progress_value if progress_value is not None else ""),
                                    str(frames_value if frames_value is not None else ""),
                                    str(total_frames_value if total_frames_value is not None else ""),
                                )
                                last_status = status_key
                            if status == "running" and stage == "processing":
                                remote_updated_at = status_payload.get("updated_at")
                                try:
                                    remote_stale_sec = time.time() - float(remote_updated_at)
                                except Exception:
                                    remote_stale_sec = 0.0
                                remote_stale_limit = max(
                                    0.0,
                                    _safe_float_env("REMOTE_EDGE_FILE_UPSCALE_ASYNC_PROGRESS_STALE_TIMEOUT_SEC", 420.0),
                                )
                                if remote_stale_limit > 0.0 and remote_stale_sec > remote_stale_limit:
                                    raise RuntimeError(
                                        "remote finalizer async progress stalled "
                                        f"for {remote_stale_sec:.1f}s job_id={async_job_id} "
                                        f"stage={stage} progress={progress_value} "
                                        f"frames={frames_value}/{total_frames_value}"
                                    )
                            if status == "completed":
                                result = dict(status_payload.get("result") or {})
                                if not result:
                                    result = status_payload
                                if not bool(result.get("uploaded")):
                                    raise RuntimeError(
                                        "remote finalizer async completed without upload confirmation: "
                                        f"keys={sorted(result.keys())}"
                                    )
                                return result
                            if status in {"failed", "error", "cancelled", "canceled"}:
                                raise RuntimeError(
                                    "remote finalizer async failed: "
                                    f"{str(status_payload.get('error') or status_payload)[:4000]}"
                                )
                        # unreachable
                    logging.warning(
                        "SmartBlog remote finalizer async endpoint unavailable status=%s; falling back to sync /upscale",
                        int(start_resp.status_code),
                    )
                except Exception:
                    if async_started or not _env_flag("REMOTE_EDGE_FILE_UPSCALE_ASYNC_FALLBACK_SYNC", "1"):
                        raise
                    logging.exception("SmartBlog remote finalizer async failed; falling back to sync /upscale")
            while True:
                resp = requests.post(
                    str(service_url),
                    params=params,
                    files=files,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),
                )
                if int(resp.status_code) == 200:
                    break
                body = str(getattr(resp, "text", "") or "")[-4000:]
                if _retry_service_unavailable(where="sync", status_code=int(resp.status_code), body=body):
                    continue
                raise RuntimeError(f"remote finalizer failed HTTP {resp.status_code}: {body}")
            try:
                payload = dict(resp.json() or {})
            except Exception as e:
                raise RuntimeError("remote finalizer returned non-JSON response") from e
            if not bool(payload.get("uploaded")):
                raise RuntimeError(f"remote finalizer did not confirm upload: keys={sorted(payload.keys())}")
            return payload

        started = float(time.perf_counter())
        try:
            payload = await asyncio.to_thread(_run_sync)
            logging.warning(
                "SmartBlog remote finalizer complete: source_bytes=%s bytes=%s frames=%s fps=%s->%s size=%sx%s upscale=%d subtitles=%s watermark=%s elapsed=%.3fs",
                str(payload.get("source_bytes", "")),
                str(payload.get("bytes", "")),
                str(payload.get("frames", "")),
                str(payload.get("source_fps", "")),
                str(payload.get("output_fps", "")),
                str(payload.get("output_width", "")),
                str(payload.get("output_height", "")),
                1 if bool(upscale_enabled) else 0,
                str(payload.get("subtitles", "")),
                str(payload.get("watermark", "")),
                float(time.perf_counter() - float(started)),
            )
            return payload
        finally:
            await _smartblog_release_media_worker_lease(media_worker_lease, log_prefix="remote-finalizer")

    async def _smartblog_resolve_upload_target(
        self,
        plan: SmartBlogRenderFinalizePlan,
    ) -> tuple[str, str]:
        signed_url = str(plan.signed_url or "").strip()
        upload_path = str(plan.upload_path or "").strip()
        if signed_url and not _env_flag("SMARTBLOG_REFRESH_SIGNED_UPLOAD_URL", "1"):
            return signed_url, upload_path

        filename = os.path.basename(upload_path) or os.path.basename(str(plan.file_path or "").strip())
        if not filename:
            filename = f"{sanitize_job_id(str(plan.job_id or 'job'))}.bin"
        folder = os.path.dirname(upload_path).replace("\\", "/").strip("/")
        if not folder:
            folder = f"{sanitize_job_id(str(plan.job_type or 'render'))}"
        try:
            resp = await self._smartblog_api.get_upload_url(filename=filename, folder=_smartblog_upload_url_folder(folder))
            smartblog_validate_action_response(resp, action="get_upload_url")
            refreshed_signed_url = str(resp.get("signed_url") or resp.get("upload_url") or "").strip()
            refreshed_upload_path = str(resp.get("path") or resp.get("storage_path") or upload_path).strip()
            self._smartblog_remember_storage_urls(
                refreshed_upload_path,
                str(resp.get("download_url") or ""),
                str(resp.get("signed_download_url") or ""),
                str(resp.get("public_url") or ""),
            )
            if not refreshed_signed_url:
                raise RuntimeError("get_upload_url returned no signed_url")
            if not refreshed_upload_path:
                raise RuntimeError("get_upload_url returned no path")
            return refreshed_signed_url, refreshed_upload_path
        except Exception:
            if signed_url:
                logging.exception(
                    "SmartBlog signed upload refresh failed open: job=%s path=%s",
                    str(plan.job_id or "-"),
                    str(upload_path or "-"),
                )
                return signed_url, upload_path
            raise

    async def _smartblog_resolve_poster_upload_target(
        self,
        plan: SmartBlogRenderFinalizePlan,
        upload_path: str,
    ) -> tuple[str, str]:
        if not _env_flag("SMARTBLOG_FINAL_POSTER_ENABLED", "1"):
            return "", ""
        base_upload_path = str(upload_path or plan.upload_path or "").replace("\\", "/").strip("/")
        folder = os.path.dirname(base_upload_path).replace("\\", "/").strip("/")
        if not folder:
            folder = f"{sanitize_job_id(str(plan.job_type or 'render'))}"
        source_name = os.path.basename(base_upload_path) or os.path.basename(str(plan.file_path or "").strip())
        stem = os.path.splitext(source_name)[0].strip() or sanitize_job_id(str(plan.job_id or "render"))
        filename = f"{stem}_poster.jpg"
        resp = await self._smartblog_api.get_upload_url(
            filename=filename,
            folder=_smartblog_upload_url_folder(folder),
            content_type="image/jpeg",
        )
        smartblog_validate_action_response(resp, action="get_upload_url")
        signed_url = str(resp.get("signed_url") or resp.get("upload_url") or "").strip()
        poster_path = str(resp.get("path") or resp.get("storage_path") or "").strip()
        self._smartblog_remember_storage_urls(
            poster_path,
            str(resp.get("download_url") or ""),
            str(resp.get("signed_download_url") or ""),
            str(resp.get("public_url") or ""),
        )
        if not signed_url:
            raise RuntimeError("poster get_upload_url returned no signed_url")
        if not poster_path:
            raise RuntimeError("poster get_upload_url returned no path")
        return signed_url, poster_path

    @staticmethod
    def _smartblog_extract_video_poster(video_path: str, poster_path: str) -> dict[str, Any]:
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
            raw_scan = str(os.getenv("SMARTBLOG_FINAL_POSTER_SCAN_SECONDS", "0.4,0.8,1.2,1.8,2.5") or "")
            scan_seconds: list[float] = []
            for part in raw_scan.split(","):
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
            min_mean = float(_safe_float_env("SMARTBLOG_FINAL_POSTER_MIN_MEAN", 10.0))
            min_std = float(_safe_float_env("SMARTBLOG_FINAL_POSTER_MIN_STD", 3.0))
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
            max_width = int(max(0, _safe_int_env("SMARTBLOG_FINAL_POSTER_MAX_WIDTH", 720)))
            height, width = int(best_frame.shape[0]), int(best_frame.shape[1])
            if max_width > 0 and width > max_width:
                scale = float(max_width) / float(width)
                best_frame = cv2.resize(
                    best_frame,
                    (int(max_width), max(2, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                height, width = int(best_frame.shape[0]), int(best_frame.shape[1])
            quality = int(max(40, min(100, _safe_int_env("SMARTBLOG_FINAL_POSTER_JPEG_QUALITY", 90))))
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

    async def _smartblog_generate_and_upload_final_poster(
        self,
        *,
        job_id: str,
        source_video_path: str,
        poster_signed_url: str,
        poster_upload_path: str,
        run_dir: str,
    ) -> dict[str, Any]:
        if not poster_signed_url or not poster_upload_path:
            return {}
        poster_file = os.path.join(
            str(run_dir or tempfile.gettempdir()),
            f"{sanitize_job_id(str(job_id or 'render'))}_final_poster.jpg",
        )
        result = await asyncio.to_thread(
            self._smartblog_extract_video_poster,
            str(source_video_path),
            str(poster_file),
        )
        await self._smartblog_upload_file(
            signed_url=str(poster_signed_url),
            file_path=str(poster_file),
            content_type="image/jpeg",
        )
        result["poster_uploaded"] = True
        result["poster_storage_path"] = str(poster_upload_path)
        return result

    def _smartblog_complete_kwargs_for_upload_path(
        self,
        plan: SmartBlogRenderFinalizePlan,
        upload_path: str,
    ) -> dict[str, Any]:
        out = dict(plan.complete_kwargs or {})
        new_path = str(upload_path or "").strip()
        old_path = str(plan.upload_path or "").strip()
        effective_path = new_path or old_path
        if effective_path:
            out.setdefault("storage_path", effective_path)
        if not new_path or new_path == old_path:
            return out
        if "storage_path" in out:
            out["storage_path"] = new_path
        if "video_url" in out:
            out["video_url"] = self._smartblog_public_storage_url(new_path)
        return out

    @staticmethod
    def _smartblog_validate_complete_response(resp: dict[str, Any] | None) -> None:
        smartblog_validate_complete_response(resp)

    async def _smartblog_finalize_render_job(self, plan: SmartBlogRenderFinalizePlan) -> None:
        job_id = str(plan.job_id or "").strip()
        job_type = str(plan.job_type or "").strip().lower()
        run_dir = os.path.abspath(str(plan.run_dir or "").strip()) if str(plan.run_dir or "").strip() else ""
        t0 = float(time.perf_counter())
        previous_finalizing_job_id = str(getattr(self, "_smartblog_finalizing_job_id", "") or "")
        self._smartblog_finalizing_job_id = job_id
        try:
            skip_upload = bool(getattr(plan, "skip_upload", False))
            remote_finalizer_source = str(
                getattr(plan, "remote_finalizer_source_url", "") or getattr(plan, "remote_finalizer_source_path", "") or ""
            ).strip()
            remote_finalizer = bool(remote_finalizer_source)
            if bool(skip_upload):
                signed_url = str(plan.signed_url or "").strip()
                upload_path = str(plan.upload_path or "").strip()
                if not upload_path:
                    raise RuntimeError("skip_upload finalize requires upload_path")
            else:
                signed_url, upload_path = await self._smartblog_resolve_upload_target(plan)
            poster_signed_url = ""
            poster_upload_path = ""
            complete_poster_path = ""
            if not bool(skip_upload):
                try:
                    poster_signed_url, poster_upload_path = await self._smartblog_resolve_poster_upload_target(
                        plan,
                        upload_path,
                    )
                except Exception:
                    logging.exception(
                        "SmartBlog final poster upload target unavailable; continuing without poster: id=%s path=%s",
                        job_id or "-",
                        str(upload_path or "-"),
                    )
            upload_progress_start = 97
            upload_progress_wait = 98
            upload_progress_end = 99
            if job_type in set(smartblog_render_job_types()):
                upload_progress_start = _smartblog_stage_progress_total(
                    job_type=job_type,
                    stage="upload",
                    stage_progress=0.0,
                )
                upload_progress_wait = _smartblog_stage_progress_total(
                    job_type=job_type,
                    stage="upload",
                    stage_progress=0.5,
                )
                upload_progress_end = min(
                    99,
                    _smartblog_stage_progress_total(
                        job_type=job_type,
                        stage="upload",
                        stage_progress=0.9,
                    ),
                )
            if bool(remote_finalizer):
                finalizer_progress_start = _smartblog_stage_progress_total(
                    job_type=job_type,
                    stage="encode",
                    stage_progress=0.0,
                )
                finalizer_progress_wait = _smartblog_stage_progress_total(
                    job_type=job_type,
                    stage="encode",
                    stage_progress=0.05,
                )
                finalizer_progress_end = _smartblog_stage_progress_total(
                    job_type=job_type,
                    stage="encode",
                    stage_progress=1.0,
                )
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=finalizer_progress_start,
                    **_smartblog_progress_stage_fields(
                        job_type=job_type,
                        stage="encode",
                        stage_label="Waiting for media worker finalizer",
                    ),
                )
                source_url = str(getattr(plan, "remote_finalizer_source_url", "") or "").strip()
                if not source_url:
                    source_url = await self._smartblog_resolve_download_url(
                        str(getattr(plan, "remote_finalizer_source_path", "") or "")
                    )
                finalizer_started = float(time.monotonic())
                finalizer_expected_sec = float(
                    max(30.0, _safe_float_env("SMARTBLOG_REMOTE_FINALIZER_PROGRESS_EXPECTED_SEC", 90.0))
                )
                finalizer_progress_state: dict[str, Any] = {}
                remote_finalizer_result: dict[str, Any] = {}
                finalizer_task = asyncio.create_task(
                    self._smartblog_run_remote_file_finalizer(
                        source_url=str(source_url),
                        signed_url=str(signed_url),
                        content_type=str(plan.content_type or "video/mp4"),
                        source_fps=int(getattr(plan, "remote_finalizer_source_fps", 0) or 0),
                        target_width=int(getattr(plan, "remote_finalizer_target_width", 0) or 0),
                        target_height=int(getattr(plan, "remote_finalizer_target_height", 0) or 0),
                        target_fps=int(getattr(plan, "remote_finalizer_target_fps", 0) or 0),
                        upscale_enabled=bool(getattr(plan, "remote_finalizer_upscale_enabled", False)),
                        background_music_url=str(getattr(plan, "remote_finalizer_background_music_url", "") or ""),
                        background_music_gain_db=float(
                            getattr(plan, "remote_finalizer_background_music_gain_db", 0.0) or 0.0
                        ),
                        background_music_loop=bool(getattr(plan, "remote_finalizer_background_music_loop", True)),
                        background_music_duck_voice_db=float(
                            getattr(plan, "remote_finalizer_background_music_duck_voice_db", 0.0) or 0.0
                        ),
                        background_music_fade_in_seconds=float(
                            getattr(plan, "remote_finalizer_background_music_fade_in_seconds", 0.0) or 0.0
                        ),
                        background_music_fade_out_seconds=float(
                            getattr(plan, "remote_finalizer_background_music_fade_out_seconds", 0.0) or 0.0
                        ),
                        background_music_start_offset_seconds=float(
                            getattr(plan, "remote_finalizer_background_music_start_offset_seconds", 0.0) or 0.0
                        ),
                        subtitle_chunks_json=str(getattr(plan, "remote_finalizer_subtitle_chunks_json", "") or ""),
                        watermark_text=str(getattr(plan, "remote_finalizer_watermark_text", "") or ""),
                        poster_signed_url=str(poster_signed_url),
                        poster_upload_path=str(poster_upload_path),
                        poster_content_type="image/jpeg",
                        progress_state=finalizer_progress_state,
                    ),
                    name=f"smartblog-remote-finalizer-{sanitize_job_id(job_id or job_type or 'job')}",
                )

                def _remote_finalizer_progress_provider() -> dict[str, Any]:
                    elapsed = float(max(0.0, time.monotonic() - finalizer_started))
                    progress_cap = int(max(finalizer_progress_start, min(finalizer_progress_end - 1, 99)))
                    span = int(max(0, progress_cap - int(finalizer_progress_start)))
                    raw_progress = finalizer_progress_state.get("progress")
                    frac: float | None = None
                    if raw_progress is not None:
                        try:
                            raw_f = float(raw_progress)
                            frac = raw_f / 100.0 if raw_f > 1.0 else raw_f
                            frac = float(max(0.0, min(1.0, float(frac))))
                        except Exception:
                            frac = None
                    if frac is None:
                        frac = float(max(0.0, min(1.0, elapsed / float(finalizer_expected_sec))))
                    if span > 0:
                        bump = int(math.ceil(float(span) * float(frac))) if elapsed > 0.0 or frac > 0.0 else 0
                        progress_value = int(min(progress_cap, int(finalizer_progress_start) + max(0, bump)))
                    else:
                        progress_value = int(finalizer_progress_wait)
                    if elapsed >= finalizer_expected_sec and raw_progress is None:
                        progress_value = int(progress_cap)
                    elapsed_label = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s"
                    rtx_stage = str(finalizer_progress_state.get("stage") or "").strip()
                    return {
                        "progress": progress_value,
                        **_smartblog_progress_stage_fields(
                            job_type=job_type,
                            stage="encode",
                            stage_label=(
                                f"Media worker {rtx_stage} ({elapsed_label})"
                                if rtx_stage
                                else f"Waiting/finalizing on media worker ({elapsed_label})"
                            ),
                        ),
                    }

                await self._smartblog_wait_with_progress(
                    task=finalizer_task,
                    job_id=job_id,
                    progress=finalizer_progress_wait,
                    progress_provider=_remote_finalizer_progress_provider,
                    **_smartblog_progress_stage_fields(
                        job_type=job_type,
                        stage="encode",
                        stage_label="Finalizing on media worker",
                    ),
                    heartbeat_sec=float(_safe_float_env("SMARTBLOG_REMOTE_FINALIZER_PROGRESS_SEC", 3.0)),
                )
                try:
                    remote_finalizer_result = dict(finalizer_task.result() or {})
                    if bool(remote_finalizer_result.get("poster_uploaded")) and remote_finalizer_result.get("poster_storage_path"):
                        poster_upload_path = str(remote_finalizer_result.get("poster_storage_path") or poster_upload_path)
                    if bool(remote_finalizer_result.get("poster_uploaded")) and poster_upload_path:
                        complete_poster_path = str(poster_upload_path)
                except Exception:
                    remote_finalizer_result = {}
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=finalizer_progress_end,
                    **_smartblog_progress_stage_fields(job_type=job_type, stage="encode"),
                )
            elif bool(skip_upload):
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=upload_progress_end,
                    **_smartblog_progress_stage_fields(job_type=job_type, stage="upload"),
                )
            else:
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=upload_progress_start,
                    **_smartblog_progress_stage_fields(job_type=job_type, stage="upload"),
                )
                upload_progress_state: dict[str, Any] = {}
                upload_task = asyncio.create_task(
                    self._smartblog_upload_file(
                        signed_url=signed_url,
                        file_path=str(plan.file_path or ""),
                        content_type=str(plan.content_type or "application/octet-stream"),
                        progress_state=upload_progress_state,
                    ),
                    name=f"smartblog-upload-{sanitize_job_id(job_id or job_type or 'job')}",
                )

                def _upload_progress_provider() -> dict[str, Any]:
                    try:
                        size = int(upload_progress_state.get("size") or os.path.getsize(str(plan.file_path or "")) or 0)
                    except Exception:
                        size = 0
                    uploaded = int(upload_progress_state.get("uploaded") or 0)
                    frac = 0.0
                    if size > 0:
                        frac = max(0.0, min(0.9, float(uploaded) / float(size)))
                    return {
                        "progress": _smartblog_stage_progress_total(
                            job_type=job_type,
                            stage="upload",
                            stage_progress=float(frac),
                        ),
                        **_smartblog_progress_stage_fields(
                            job_type=job_type,
                            stage="upload",
                            stage_label=f"Uploading result ({int((uploaded / size) * 100) if size > 0 else 0}%)",
                        ),
                    }

                await self._smartblog_wait_with_progress(
                    task=upload_task,
                    job_id=job_id,
                    progress=upload_progress_wait,
                    progress_provider=_upload_progress_provider,
                    **_smartblog_progress_stage_fields(job_type=job_type, stage="upload"),
                    heartbeat_sec=float(_safe_float_env("SMARTBLOG_UPLOAD_PROGRESS_SEC", 2.0)),
                )
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=upload_progress_end,
                    **_smartblog_progress_stage_fields(job_type=job_type, stage="upload"),
                )
                if poster_signed_url and poster_upload_path:
                    try:
                        poster_result = await self._smartblog_generate_and_upload_final_poster(
                            job_id=job_id,
                            source_video_path=str(plan.file_path or ""),
                            poster_signed_url=str(poster_signed_url),
                            poster_upload_path=str(poster_upload_path),
                            run_dir=run_dir,
                        )
                        if bool(poster_result.get("poster_uploaded")) and poster_upload_path:
                            complete_poster_path = str(poster_upload_path)
                    except Exception:
                        logging.exception(
                            "SmartBlog local final poster generation failed; continuing without poster: id=%s",
                            job_id or "-",
                        )
            complete_kwargs = self._smartblog_complete_kwargs_for_upload_path(plan, upload_path)
            if complete_poster_path:
                complete_kwargs.setdefault("poster_storage_path", str(complete_poster_path))
            complete_resp = await self._smartblog_api.complete(job_id=job_id, **complete_kwargs)
            self._smartblog_validate_complete_response(complete_resp)
            self._smartblog_finalize_completed_count = int(
                getattr(self, "_smartblog_finalize_completed_count", 0) or 0
            ) + 1
            self._smartblog_finalize_last_complete_mono = float(time.monotonic())
            logging.info(
                "SmartBlog finalize complete: id=%s type=%s elapsed_ms=%.1f",
                job_id or "-",
                job_type or "-",
                float((time.perf_counter() - t0) * 1000.0),
            )
        except asyncio.CancelledError:
            logging.warning("SmartBlog finalize cancelled: id=%s type=%s", job_id or "-", job_type or "-")
            raise
        except SmartBlogJobStoppedByServer as e:
            logging.warning(
                "SmartBlog finalize stopped by server: id=%s type=%s reason=%s",
                job_id or "-",
                job_type or "-",
                str(e or "-"),
            )
        except Exception as e:
            if smartblog_is_transient_api_error(e):
                self._smartblog_finalize_last_error = str(e or "SmartBlog transient finalize failure")[:1500]
                self._smartblog_finalize_last_error_mono = float(time.monotonic())
                logging.warning(
                    "SmartBlog finalize transient failure; leaving job for server requeue: id=%s type=%s err=%s",
                    job_id or "-",
                    job_type or "-",
                    e,
                )
                return
            err = str(e or "SmartBlog finalize failed").strip() or "SmartBlog finalize failed"
            self._smartblog_finalize_failed_count = int(getattr(self, "_smartblog_finalize_failed_count", 0) or 0) + 1
            self._smartblog_finalize_last_error = err[:1500]
            self._smartblog_finalize_last_error_mono = float(time.monotonic())
            logging.exception("SmartBlog finalize failed: id=%s type=%s err=%s", job_id or "-", job_type or "-", e)
            try:
                await self._smartblog_api.fail(job_id=job_id, error_text=err[:1500])
            except Exception as fail_err:
                logging.warning(
                    "SmartBlog finalize fail-report failed: id=%s type=%s err=%s",
                    job_id or "-",
                    job_type or "-",
                    fail_err,
                )
        finally:
            self._smartblog_finalizing_job_id = previous_finalizing_job_id
            if run_dir:
                try:
                    await asyncio.to_thread(shutil.rmtree, run_dir, True)
                except Exception:
                    pass

    def _smartblog_track_finalize_task(
        self,
        task: asyncio.Task,
        *,
        job_id: str,
        job_type: str,
    ) -> None:
        tasks = getattr(self, "_smartblog_finalize_tasks", None)
        if not isinstance(tasks, set):
            tasks = set()
            setattr(self, "_smartblog_finalize_tasks", tasks)
        tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            try:
                tasks.discard(done_task)
            except Exception:
                pass
            try:
                _ = done_task.exception()
            except asyncio.CancelledError:
                logging.warning(
                    "SmartBlog finalize task cancelled: id=%s type=%s",
                    str(job_id or "-"),
                    str(job_type or "-"),
                )
            except Exception:
                pass

        task.add_done_callback(_done)

    def _smartblog_schedule_finalize(self, plan: SmartBlogRenderFinalizePlan) -> None:
        job_id = str(plan.job_id or "").strip()
        job_type = str(plan.job_type or "").strip().lower()
        task = asyncio.create_task(
            self._smartblog_finalize_render_job(plan),
            name=f"smartblog-finalize-{sanitize_job_id(job_id or job_type or 'job')}",
        )
        self._smartblog_track_finalize_task(task, job_id=job_id, job_type=job_type)
        logging.info(
            "SmartBlog finalize scheduled: id=%s type=%s pending=%d",
            job_id or "-",
            job_type or "-",
            len(getattr(self, "_smartblog_finalize_tasks", set()) or set()),
        )

    async def _smartblog_wait_for_finalize_tasks(self, *, timeout_sec: float = 120.0) -> None:
        tasks = [
            task
            for task in list(getattr(self, "_smartblog_finalize_tasks", set()) or set())
            if isinstance(task, asyncio.Task) and (not task.done())
        ]
        if not tasks:
            return
        timeout = max(1.0, float(timeout_sec or 120.0))
        logging.info("Waiting for SmartBlog finalize tasks: pending=%d timeout_sec=%.1f", len(tasks), timeout)
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logging.warning("Cancelling stuck SmartBlog finalize tasks: pending=%d", len(pending))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        elif done:
            await asyncio.gather(*done, return_exceptions=True)

    def _smartblog_public_storage_url(self, upload_path: str) -> str:
        path = str(upload_path or "").strip().lstrip("/")
        if not path:
            raise RuntimeError("upload.path is required to build public URL")
        cached = self._smartblog_cached_storage_download_url(path)
        if cached:
            return cached
        return f"{smartblog_supabase_url()}/storage/v1/object/public/generated-assets/{path}"

    async def _smartblog_process_media_phase(
        self,
        *,
        job_id: str,
        req: MediaProcessRequest,
        pending_progress: int | None,
        pending_stage: str | None = None,
        pending_stage_label: str | None = None,
        pending_stage_index: int | None = None,
        pending_stage_total: int | None = None,
    ) -> str:
        task = asyncio.create_task(self._model_client.media_process(req=req))
        if pending_progress is None:
            resp = await task
        else:
            resp = await self._smartblog_wait_with_progress(
                task=task,
                job_id=job_id,
                progress=int(pending_progress),
                stage=str(pending_stage or "inference"),
                stage_label=pending_stage_label,
                stage_index=pending_stage_index,
                stage_total=pending_stage_total,
            )
        if not bool(resp.ok):
            raise RuntimeError(str(resp.error or "media_process failed"))
        out_path = str(resp.output_path or "").strip()
        if not out_path or not os.path.exists(out_path):
            raise RuntimeError("media_process produced no output file")
        return out_path

    async def _smartblog_prepare_render_continuation_image(
        self,
        *,
        job_id: str,
        source_video_path: str,
        raw_frame_path: str,
        output_path: str,
        face_restore: float,
        segment_index: int,
        reason: str,
        background_restore: float = 0.0,
        target_width: int = 0,
        target_height: int = 0,
    ) -> str:
        raw_path = await asyncio.to_thread(
            _smartblog_extract_last_video_frame,
            str(source_video_path),
            str(raw_frame_path),
        )
        ref_face_restore = float(max(0.0, min(1.0, float(face_restore))))
        ref_background_restore = float(max(0.0, min(1.0, float(background_restore))))
        if bool(_env_flag("SMARTBLOG_RENDER_CONTINUATION_FORCE_BACKGROUND_RESTORE", "0")):
            ref_background_restore = 1.0
        should_reprocess = bool(_env_flag("SMARTBLOG_RENDER_CONTINUATION_REPROCESS", "1")) and (
            ref_face_restore > 0.0 or ref_background_restore > 0.0
        )
        if should_reprocess:
            req = MediaProcessRequest(
                source_path=str(raw_path),
                source_kind="image",
                output_path=str(output_path),
                output_width=0,
                output_height=0,
                output_fps=0.0,
                preserve_audio=False,
                upscale=False,
                face_restore=float(ref_face_restore),
                background_restore=float(ref_background_restore),
                jpeg_quality=95,
                trim_duration_sec=0.0,
            )
            enhanced_path = await self._smartblog_process_media_phase(
                job_id=str(job_id),
                req=req,
                pending_progress=None,
            )
        else:
            enhanced_path = str(raw_path)
        final_path = str(enhanced_path)
        final_w = 0
        final_h = 0
        if int(target_width or 0) > 0 and int(target_height or 0) > 0:
            framed_path = os.path.splitext(str(output_path))[0] + "_framed.png"
            final_path = await asyncio.to_thread(
                _smartblog_compensate_continuation_ref_framing,
                str(enhanced_path),
                str(framed_path),
                target_width=int(target_width),
                target_height=int(target_height),
                job_id=str(job_id),
                segment_index=int(segment_index),
                reason=str(reason or ""),
            )
        try:
            final_img = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
            if final_img is not None:
                final_h = int(final_img.shape[0])
                final_w = int(final_img.shape[1])
        except Exception:
            final_w = 0
            final_h = 0
        logging.warning(
            "SmartBlog render continuation image prepared: job=%s segment=%d reason=%s source=%s raw=%s out=%s final=%s final_size=%dx%d reprocess=%d face_restore=%.2f background_restore=%.2f framed=%d target=%dx%d",
            str(job_id or "-"),
            int(segment_index),
            str(reason or "-"),
            os.path.basename(str(source_video_path)),
            os.path.basename(str(raw_path)),
            os.path.basename(str(enhanced_path)),
            os.path.basename(str(final_path)),
            int(final_w),
            int(final_h),
            1 if should_reprocess else 0,
            float(ref_face_restore),
            float(ref_background_restore),
            1 if str(final_path) != str(enhanced_path) else 0,
            int(target_width or 0),
            int(target_height or 0),
        )
        return str(final_path)

    async def _smartblog_run_subprocess_checked(
        self,
        *,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        log_prefix: str,
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            *[str(part) for part in cmd],
            cwd=str(cwd or None) if str(cwd or "").strip() else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    lines.append(text)
                    if len(lines) > 80:
                        lines = lines[-80:]
                    logging.info("%s %s", str(log_prefix), text)
            return_code = await proc.wait()
        except asyncio.CancelledError:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise
        if int(return_code or 0) != 0:
            tail = "\n".join(lines[-30:])
            raise RuntimeError(f"{log_prefix} failed with exit={return_code}: {tail}")

    def _smartblog_hunyuan_service_on_demand_enabled(self) -> bool:
        if self._smartblog_local_media_services_forbidden():
            return False
        return _env_flag("SMARTBLOG_HUNYUAN_SERVICE_ON_DEMAND", "0") or _env_flag(
            "SMARTBLOG_LTX_SERVICE_ON_DEMAND", "0"
        )

    def _smartblog_hunyuan_swap_modeld_enabled(self) -> bool:
        if _env_flag("SMARTBLOG_HUNYUAN_SWAP_MODELD", "0") or _env_flag("SMARTBLOG_LTX_SWAP_MODELD", "0"):
            return True
        if not _env_flag("SMARTBLOG_HUNYUAN_AUTO_SWAP_MODELD", "1"):
            return False

        def _visible_gpus(raw: str) -> list[str]:
            values: list[str] = []
            for part in str(raw or "").split(","):
                item = str(part or "").strip()
                if item:
                    values.append(item)
            return values

        visible = _visible_gpus(os.getenv("CUDA_VISIBLE_DEVICES") or "")
        hunyuan_visible = _visible_gpus(
            os.getenv("SMARTBLOG_HUNYUAN_CUDA_VISIBLE_DEVICES")
            or os.getenv("SMARTBLOG_LTX_CUDA_VISIBLE_DEVICES")
            or ""
        )
        if len(visible) == 1:
            return True
        if len(hunyuan_visible) == 1 and len(visible) <= 1:
            return True
        return False

    async def _smartblog_hunyuan_service_ctl(
        self,
        action: str,
        *,
        timeout_sec: float | None = None,
        log_prefix: str = "SmartBlog",
    ) -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script = os.path.join(root_dir, "scripts", "hunyuan_service.sh")
        if not os.path.exists(script):
            raise RuntimeError(f"Hunyuan service script is missing: {script}")
        proc = await asyncio.create_subprocess_exec(
            "bash",
            str(script),
            str(action),
            cwd=str(root_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout_sec or _safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_CTL_TIMEOUT_SEC", 1800.0)),
            )
        except asyncio.TimeoutError as e:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise RuntimeError(f"{log_prefix} Hunyuan service {action} timed out") from e
        out = (out_bytes or b"").decode("utf-8", errors="replace").strip()
        if int(proc.returncode or 0) != 0:
            raise RuntimeError(f"{log_prefix} Hunyuan service {action} failed exit={proc.returncode}: {out[-2000:]}")
        if out:
            logging.warning("%s Hunyuan service %s: %s", str(log_prefix), str(action), out[-1000:])

    async def _smartblog_modeld_ctl(
        self,
        action: str,
        *,
        timeout_sec: float | None = None,
        log_prefix: str = "SmartBlog",
    ) -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script = os.path.join(root_dir, "scripts", "control.sh")
        if not os.path.exists(script):
            raise RuntimeError(f"control script is missing: {script}")
        proc = await asyncio.create_subprocess_exec(
            "bash",
            str(script),
            str(action),
            cwd=str(root_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout_sec or _safe_float_env("SMARTBLOG_MODELD_CTL_TIMEOUT_SEC", 1800.0)),
            )
        except asyncio.TimeoutError as e:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise RuntimeError(f"{log_prefix} modeld control {action} timed out") from e
        out = (out_bytes or b"").decode("utf-8", errors="replace").strip()
        if int(proc.returncode or 0) != 0:
            raise RuntimeError(f"{log_prefix} modeld control {action} failed exit={proc.returncode}: {out[-2000:]}")
        if out:
            logging.warning("%s modeld control %s: %s", str(log_prefix), str(action), out[-1000:])

    async def _smartblog_hunyuan_service_generate(
        self,
        *,
        payload: dict[str, Any],
        log_prefix: str,
    ) -> dict[str, Any]:
        fallback_url = str(
            os.getenv("SMARTBLOG_HUNYUAN_SERVICE_URL")
            or os.getenv("SMARTBLOG_LTX_SERVICE_URL")
            or ""
        ).strip()
        service_url, media_worker_lease = await _smartblog_acquire_media_worker_lease(
            "hunyuan",
            fallback_url,
            log_prefix=str(log_prefix),
        )
        service_url = str(service_url or "").rstrip("/")
        if not service_url:
            raise RuntimeError("SMARTBLOG_HUNYUAN_SERVICE_URL is empty")
        remote_service = self._smartblog_remote_service_enabled(
            service_url,
            "SMARTBLOG_HUNYUAN_SERVICE_REMOTE",
        )
        if self._smartblog_local_media_services_forbidden() and (
            not bool(remote_service) or _smartblog_service_url_is_local(str(service_url))
        ):
            raise RuntimeError(
                "local Hunyuan service is forbidden for this worker profile; "
                "configure/lease a remote media worker instead"
            )
        on_demand = bool((not remote_service) and self._smartblog_hunyuan_service_on_demand_enabled())
        swap_modeld = bool(on_demand and self._smartblog_hunyuan_swap_modeld_enabled())
        swapped_modeld = False
        try:
            if bool(on_demand):
                if bool(swap_modeld):
                    await self._smartblog_hunyuan_service_ctl(
                        "stop",
                        timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_STOP_TIMEOUT_SEC", 120.0),
                        log_prefix=str(log_prefix),
                    )
                    await self._smartblog_modeld_ctl(
                        "stop-modeld",
                        timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SWAP_MODELD_STOP_TIMEOUT_SEC", 240.0),
                        log_prefix=str(log_prefix),
                    )
                    swapped_modeld = True
                await self._smartblog_hunyuan_service_ctl(
                    "start-wait",
                    timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_READY_TIMEOUT_SEC", 1800.0),
                    log_prefix=str(log_prefix),
                )
            request_payload = dict(payload or {})
            remote_local_output_path = ""
            if bool(remote_service):
                local_output_dir = os.path.abspath(
                    str(
                        request_payload.get("output_path")
                        or request_payload.get("output_dir")
                        or os.path.join("outputs", "hunyuan_service")
                    )
                )
                os.makedirs(local_output_dir, exist_ok=True)
                output_basename = f"remote_hunyuan_{sanitize_job_id(str(log_prefix or 'job'))}_{int(time.time() * 1000)}.mp4"
                remote_local_output_path = os.path.join(str(local_output_dir), output_basename)
                input_items: list[dict[str, Any]] = []
                media_paths = request_payload.get("conditioning_media_paths")
                if media_paths is not None and not isinstance(media_paths, list):
                    media_paths = [media_paths]
                for path in list(media_paths or []):
                    path_s = str(path or "").strip()
                    if path_s and os.path.exists(path_s):
                        input_items.append(self._smartblog_file_base64_payload(path_s, content_type="image/png"))
                image_path = str(request_payload.get("image_path") or request_payload.get("input_media_path") or "").strip()
                if image_path and os.path.exists(image_path):
                    item = self._smartblog_file_base64_payload(image_path, content_type="image/png")
                    if item.get("data") not in {existing.get("data") for existing in input_items}:
                        input_items.append(item)
                if input_items:
                    request_payload["conditioning_media_base64"] = input_items
                    request_payload.pop("conditioning_media_paths", None)
                    request_payload.pop("image_path", None)
                    request_payload.pop("input_media_path", None)
                signed_url, upload_path, public_url = await self._smartblog_prepare_remote_service_output_upload(
                    job_id_hint=str(log_prefix or "hunyuan"),
                    folder=f"worker-uploads/remote-hunyuan/{sanitize_job_id(str(log_prefix or 'job'))}",
                    filename=str(output_basename),
                    content_type="video/mp4",
                )
                request_payload["output_upload_url"] = str(signed_url)
                request_payload["output_public_url"] = str(public_url)
                request_payload["output_storage_path"] = str(upload_path)
                request_payload.pop("output_path", None)
                request_payload.pop("output_dir", None)
                logging.warning(
                    "%s Hunyuan remote service request: url=%s inputs_base64=%d upload_path=%s",
                    str(log_prefix),
                    str(service_url),
                    int(len(input_items)),
                    str(upload_path),
                )
            timeout_sec = float(max(30.0, _safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_TIMEOUT_SEC", _safe_float_env("SMARTBLOG_LTX_SERVICE_TIMEOUT_SEC", 1800.0))))
            started = float(time.monotonic())
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_sec))) as session:
                async with session.post(f"{service_url}/generate", json=dict(request_payload or {})) as resp:
                    text = await resp.text()
                    if int(resp.status) >= 400:
                        raise RuntimeError(f"{log_prefix} Hunyuan service HTTP {int(resp.status)}: {text[-1000:]}")
                    try:
                        data = json.loads(text)
                    except Exception as e:
                        raise RuntimeError(f"{log_prefix} Hunyuan service returned invalid JSON: {text[-1000:]}") from e
            if not isinstance(data, dict) or not bool(data.get("ok")):
                raise RuntimeError(f"{log_prefix} Hunyuan service failed: {data}")
            output_path = str(data.get("output_path") or "").strip()
            if output_path and os.path.exists(output_path):
                pass
            elif bool(remote_service):
                output_path = await self._smartblog_download_remote_service_output(
                    data=dict(data),
                    local_path=str(remote_local_output_path),
                    log_prefix=str(log_prefix),
                    label="Hunyuan",
                )
                data["output_path"] = str(output_path)
                data["output_paths"] = [str(output_path)]
            else:
                raise RuntimeError(f"{log_prefix} Hunyuan service produced missing output: {output_path or data}")
            logging.warning(
                "%s Hunyuan service done in %.2fs output=%s",
                str(log_prefix),
                float(time.monotonic()) - float(started),
                str(output_path),
            )
            return dict(data)
        finally:
            await _smartblog_release_media_worker_lease(media_worker_lease, log_prefix=str(log_prefix))
            if bool(swapped_modeld):
                try:
                    await self._smartblog_hunyuan_service_ctl(
                        "stop",
                        timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_STOP_TIMEOUT_SEC", 120.0),
                        log_prefix=str(log_prefix),
                    )
                except Exception as e:
                    logging.warning("%s Hunyuan service stop after swap failed: %s", str(log_prefix), e)
                await self._smartblog_modeld_ctl(
                    "start-modeld",
                    timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SWAP_MODELD_START_TIMEOUT_SEC", 1800.0),
                    log_prefix=str(log_prefix),
                )

    async def _smartblog_gemini_audio_prompt_for_video(
        self,
        *,
        video_path: str,
        run_dir: str,
        base_prompt: str,
        audio_direction: str = "",
        duration_sec: float,
        log_prefix: str,
    ) -> str:
        fallback_prompt = _smartblog_first_text(str(audio_direction or ""), _smartblog_render_default_mmaudio_prompt())
        if not _env_flag("SMARTBLOG_MMAUDIO_GEMINI_PROMPT", "1"):
            return str(fallback_prompt or "").strip()
        api_key = str(
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            logging.warning("%s Gemini audio prompt skipped: no API key", str(log_prefix))
            return str(fallback_prompt or "").strip()
        model = str(os.getenv("SMARTBLOG_MMAUDIO_GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash").strip()
        if model.startswith("models/"):
            model = model[len("models/") :]
        sample_count = int(max(1, min(24, _safe_int_env("SMARTBLOG_MMAUDIO_GEMINI_FRAME_COUNT", 10))))
        max_dim = int(max(128, min(1024, _safe_int_env("SMARTBLOG_MMAUDIO_GEMINI_FRAME_MAX_DIM", 512))))
        try:
            frame_paths = await asyncio.to_thread(
                _smartblog_sample_video_frames_for_prompt,
                video_path=str(video_path),
                out_dir=os.path.join(str(run_dir), "mmaudio_gemini_frames"),
                count=int(sample_count),
                max_dim=int(max_dim),
            )
        except Exception as e:
            logging.warning("%s Gemini audio prompt frame sampling failed: %s", str(log_prefix), str(e))
            return str(fallback_prompt or "").strip()

        audio_direction_text = str(audio_direction or "").strip()
        direction_sentence = (
            " User requested this audio direction; incorporate it only if it fits the visible video: "
            f"{audio_direction_text[:800]}"
            if audio_direction_text
            else ""
        )
        instruction = (
            "You are writing a concise sound-design prompt for MMAudio large_44k_v2. "
            "Analyze these evenly sampled frames from the whole video clip in chronological order. "
            "Return only one English prompt, max 360 characters. "
            "Describe synchronized ambience and concrete sound effects that match visible motion, materials, location, and camera movement. "
            "Do not add speech, singing, music, narration, audience noise, or branded sounds unless clearly visible. "
            f"Clip duration: {float(duration_sec):.2f}s."
            f"{direction_sentence}"
        )
        parts: list[dict[str, Any]] = [{"text": instruction}]
        for path in frame_paths:
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": data}})
            except Exception:
                continue
        if len(parts) <= 1:
            return str(base_prompt or "").strip()
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(max(0.0, min(1.5, _safe_float_env("SMARTBLOG_MMAUDIO_GEMINI_TEMPERATURE", 0.35)))),
                "topP": 0.9,
                "maxOutputTokens": int(max(32, min(256, _safe_int_env("SMARTBLOG_MMAUDIO_GEMINI_MAX_TOKENS", 96)))),
            },
        }
        timeout_sec = float(max(5.0, min(120.0, _safe_float_env("SMARTBLOG_MMAUDIO_GEMINI_TIMEOUT_SEC", 30.0))))
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(str(model), safe='')}:generateContent"
            f"?key={urllib.parse.quote(str(api_key), safe='')}"
        )
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_sec))) as session:
                async with session.post(url, json=payload) as resp:
                    text = await resp.text()
                    if int(resp.status) >= 400:
                        logging.warning("%s Gemini audio prompt HTTP %d: %s", str(log_prefix), int(resp.status), text[-800:])
                        return str(fallback_prompt or "").strip()
                    data = json.loads(text)
            chunks: list[str] = []
            for candidate in data.get("candidates") or []:
                content = candidate.get("content") if isinstance(candidate, dict) else {}
                for part in (content or {}).get("parts") or []:
                    value = str((part or {}).get("text") or "").strip()
                    if value:
                        chunks.append(value)
                if chunks:
                    break
            prompt = re.sub(r"\s+", " ", " ".join(chunks)).strip().strip('"')
            if prompt:
                prompt = prompt[:500].strip()
                logging.warning(
                    "%s Gemini audio prompt ready: frames=%d chars=%d prompt=%s",
                    str(log_prefix),
                    int(len(frame_paths)),
                    int(len(prompt)),
                    prompt[:240],
                )
                return str(prompt)
        except Exception as e:
            logging.warning("%s Gemini audio prompt failed: %s", str(log_prefix), str(e))
        return str(fallback_prompt or "").strip()

    async def _smartblog_mmaudio_service_generate(
        self,
        *,
        video_path: str,
        prompt: str,
        negative_prompt: str,
        duration_sec: float,
        output_path: str,
        log_prefix: str,
    ) -> str:
        service_url, media_worker_lease = await _smartblog_acquire_media_worker_lease(
            "mmaudio",
            str(os.getenv("SMARTBLOG_MMAUDIO_SERVICE_URL", "") or "").strip(),
            log_prefix=str(log_prefix),
        )
        service_url = str(service_url or "").rstrip("/")
        timeout_sec = float(max(30.0, _safe_float_env("SMARTBLOG_MMAUDIO_SERVICE_TIMEOUT_SEC", 900.0)))
        payload = {
            "video_path": str(video_path),
            "prompt": str(prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "duration": float(max(0.1, float(duration_sec or 0.0))),
            "output_path": str(output_path),
            "variant": str(os.getenv("SMARTBLOG_MMAUDIO_VARIANT", "large_44k_v2") or "large_44k_v2"),
            "seed": int(_safe_int_env("SMARTBLOG_MMAUDIO_SEED", 42)),
            "num_steps": int(_safe_int_env("SMARTBLOG_MMAUDIO_NUM_STEPS", 25)),
            "cfg_strength": float(_safe_float_env("SMARTBLOG_MMAUDIO_CFG_STRENGTH", 4.5)),
        }
        if service_url:
            remote_service = self._smartblog_remote_service_enabled(
                service_url,
                "SMARTBLOG_MMAUDIO_SERVICE_REMOTE",
            )
            if self._smartblog_local_media_services_forbidden() and (
                not bool(remote_service) or _smartblog_service_url_is_local(str(service_url))
            ):
                raise RuntimeError(
                    "local MMAudio service is forbidden for this worker profile; "
                    "configure/lease a remote media worker instead"
                )
            if bool(remote_service):
                source_name = f"mmaudio_source_{sanitize_job_id(str(log_prefix or 'job'))}_{int(time.time() * 1000)}.mp4"
                source_url = await self._smartblog_upload_remote_service_source(
                    source_path=str(video_path),
                    job_id_hint=str(log_prefix or "mmaudio"),
                    folder=f"worker-uploads/remote-mmaudio/{sanitize_job_id(str(log_prefix or 'job'))}/source",
                    filename=str(source_name),
                    content_type="video/mp4",
                )
                output_name = os.path.basename(str(output_path or "").strip()) or (
                    f"remote_mmaudio_{sanitize_job_id(str(log_prefix or 'job'))}_{int(time.time() * 1000)}.wav"
                )
                signed_url, upload_path, public_url = await self._smartblog_prepare_remote_service_output_upload(
                    job_id_hint=str(log_prefix or "mmaudio"),
                    folder=f"worker-uploads/remote-mmaudio/{sanitize_job_id(str(log_prefix or 'job'))}/output",
                    filename=str(output_name),
                    content_type="audio/wav",
                )
                payload["video_url"] = str(source_url)
                payload["output_path"] = (
                    f"/tmp/smartblog-mmaudio-service-output/"
                    f"{sanitize_job_id(str(log_prefix or 'job'))}_{int(time.time() * 1000)}_{output_name}"
                )
                payload["output_upload_url"] = str(signed_url)
                payload["output_public_url"] = str(public_url)
                payload["output_storage_path"] = str(upload_path)
                payload.pop("video_path", None)
                logging.warning(
                    "%s MMAudio remote service request: url=%s source=%s remote_output=%s upload_path=%s",
                    str(log_prefix),
                    str(service_url),
                    str(source_url),
                    str(payload.get("output_path") or ""),
                    str(upload_path),
                )
            started = float(time.monotonic())
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_sec))) as session:
                    async with session.post(f"{service_url}/generate", json=payload) as resp:
                        text = await resp.text()
                        if int(resp.status) >= 400:
                            raise RuntimeError(f"{log_prefix} MMAudio service HTTP {int(resp.status)}: {text[-1000:]}")
                        data = json.loads(text)
                if not isinstance(data, dict) or not bool(data.get("ok")):
                    raise RuntimeError(f"{log_prefix} MMAudio service failed: {data}")
                out = str(data.get("output_path") or "").strip()
                if out and os.path.exists(out):
                    pass
                elif bool(remote_service):
                    out = await self._smartblog_download_remote_service_output(
                        data=dict(data),
                        local_path=str(output_path),
                        log_prefix=str(log_prefix),
                        label="MMAudio",
                    )
                else:
                    raise RuntimeError(f"{log_prefix} MMAudio service produced missing output: {out or data}")
                logging.warning(
                    "%s MMAudio service done in %.2fs output=%s",
                    str(log_prefix),
                    float(time.monotonic()) - float(started),
                    os.path.basename(str(out)),
                )
                return str(out)
            finally:
                await _smartblog_release_media_worker_lease(media_worker_lease, log_prefix=str(log_prefix))

        if self._smartblog_local_media_services_forbidden():
            raise RuntimeError(
                "local MMAudio service is forbidden for this worker profile; "
                "configure/lease a remote media worker instead"
            )

        root = os.path.abspath(str(os.getenv("SMARTBLOG_MMAUDIO_ROOT", "/opt/MMAudio") or "/opt/MMAudio"))
        python = os.path.abspath(str(os.getenv("SMARTBLOG_MMAUDIO_PYTHON", "/opt/MMAudio/venv/bin/python") or "/opt/MMAudio/venv/bin/python"))
        demo = os.path.join(root, "demo.py")
        if not os.path.exists(python):
            raise RuntimeError(f"MMAudio python not found: {python}")
        if not os.path.exists(demo):
            raise RuntimeError(f"MMAudio demo.py not found under {root}")
        out_dir = os.path.dirname(os.path.abspath(str(output_path)))
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            str(python),
            str(demo),
            "--variant",
            str(os.getenv("SMARTBLOG_MMAUDIO_VARIANT", "large_44k_v2") or "large_44k_v2"),
            "--video",
            str(video_path),
            "--prompt",
            str(prompt or ""),
            "--negative_prompt",
            str(negative_prompt or ""),
            "--duration",
            f"{float(duration_sec):.6f}",
            "--cfg_strength",
            str(float(_safe_float_env("SMARTBLOG_MMAUDIO_CFG_STRENGTH", 4.5))),
            "--num_steps",
            str(int(_safe_int_env("SMARTBLOG_MMAUDIO_NUM_STEPS", 25))),
            "--seed",
            str(int(_safe_int_env("SMARTBLOG_MMAUDIO_SEED", 42))),
            "--output",
            str(out_dir),
            "--skip_video_composite",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(os.getenv("SMARTBLOG_MMAUDIO_CUDA_VISIBLE_DEVICES", "0") or "0")
        env.setdefault("HF_HOME", str(os.getenv("SMARTBLOG_MMAUDIO_HF_HOME", "/root/smartblog-assets/mmaudio/hf") or ""))
        await self._smartblog_run_subprocess_checked(
            cmd=cmd,
            cwd=str(root),
            env=env,
            log_prefix=str(log_prefix),
        )
        expected = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(str(video_path)))[0]}.flac")
        if os.path.exists(expected):
            return str(expected)
        candidates = sorted(
            [
                os.path.join(out_dir, name)
                for name in os.listdir(out_dir)
                if name.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".aac"))
            ],
            key=lambda path: os.path.getmtime(path),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
        raise RuntimeError(f"{log_prefix} MMAudio command produced no audio in {out_dir}")

    async def _smartblog_prepare_hunyuan_voice_audio(
        self,
        *,
        audio_entries: list[dict[str, Any]],
        run_dir: str,
        job_id: str,
    ) -> tuple[str, float]:
        entries = [dict(entry or {}) for entry in list(audio_entries or []) if isinstance(entry, dict)]
        if not entries:
            return "", 0.0
        local_paths: list[str] = []
        alignment_duration = 0.0
        for idx, entry in enumerate(entries):
            local = _smartblog_render_audio_entry_local_path(entry)
            if not local:
                url_s = str(entry.get("url") or "").strip()
                if not url_s:
                    continue
                ext = os.path.splitext(urllib.parse.urlparse(str(url_s)).path)[1].strip() or ".audio"
                if len(ext) > 12:
                    ext = ".audio"
                local = os.path.join(str(run_dir), f"hunyuan_voice_{int(idx):03d}{ext}")
                await self._smartblog_download_file(url=str(url_s), out_path=str(local))
            if local and os.path.exists(str(local)):
                local_paths.append(str(local))
                align_sec = float(_smartblog_audio_entry_alignment_duration_sec(entry))
                if align_sec > 0.0:
                    alignment_duration += float(align_sec)
        if not local_paths:
            return "", 0.0
        if len(local_paths) == 1:
            duration = float(alignment_duration or _smartblog_audio_duration_sec(local_paths[0]) or 0.0)
            logging.warning(
                "SmartBlog Hunyuan voice audio ready: job=%s chunks=1 duration=%.3fs file=%s",
                str(job_id or "-"),
                float(duration),
                os.path.basename(str(local_paths[0])),
            )
            return str(local_paths[0]), float(duration)

        out_path = os.path.join(str(run_dir), "hunyuan_voice_concat.wav")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        filter_parts: list[str] = []
        concat_inputs: list[str] = []
        for idx, path in enumerate(local_paths):
            cmd.extend(["-i", str(path)])
            filter_parts.append(
                f"[{int(idx)}:a:0]aformat=channel_layouts=mono,aresample={int(self.sample_rate)}[a{int(idx)}]"
            )
            concat_inputs.append(f"[a{int(idx)}]")
        filter_parts.append(f"{''.join(concat_inputs)}concat=n={int(len(local_paths))}:v=0:a=1[a]")
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[a]",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(int(self.sample_rate)),
                "-ac",
                "1",
                str(out_path),
            ]
        )
        await asyncio.to_thread(subprocess.run, cmd, check=True)
        duration = float(alignment_duration or _smartblog_audio_duration_sec(out_path) or wav_duration_seconds(out_path) or 0.0)
        logging.warning(
            "SmartBlog Hunyuan voice audio concatenated: job=%s chunks=%d duration=%.3fs file=%s",
            str(job_id or "-"),
            int(len(local_paths)),
            float(duration),
            os.path.basename(str(out_path)),
        )
        return str(out_path), float(duration)

    async def _smartblog_apply_background_music_if_needed(
        self,
        *,
        claim: dict[str, Any],
        video_path: str,
        run_dir: str,
        out_path: str,
        job_id: str,
        pending_progress: int | None = None,
        pending_stage_fields: dict[str, Any] | None = None,
    ) -> str:
        cfg = _smartblog_background_music_config(claim)
        if not bool(cfg.get("enabled")):
            return str(video_path)
        audio_url = str(cfg.get("audio_url") or "").strip()
        if not audio_url:
            return str(video_path)
        ext = os.path.splitext(urllib.parse.urlparse(str(audio_url)).path)[1].strip() or ".audio"
        if len(ext) > 12:
            ext = ".audio"
        music_path = os.path.join(str(run_dir), f"background_music{ext}")
        await self._smartblog_download_file(url=str(audio_url), out_path=str(music_path))
        duration = float(video_duration_sec(str(video_path)) or 0.0)
        mix_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_mix_background_music,
                video_path=str(video_path),
                music_path=str(music_path),
                out_path=str(out_path),
                duration_sec=float(duration),
                sample_rate=48000,
                gain_db=float(cfg.get("gain_db") or 0.0),
                loop=bool(cfg.get("loop", True)),
                duck_voice_db=float(cfg.get("duck_voice_db") or 0.0),
                fade_in_seconds=float(cfg.get("fade_in_seconds") or 0.0),
                fade_out_seconds=float(cfg.get("fade_out_seconds") or 0.0),
                start_offset_seconds=float(cfg.get("start_offset_seconds") or 0.0),
            ),
            name=f"smartblog-background-music-{sanitize_job_id(str(job_id or 'job'))}",
        )
        if pending_progress is not None:
            mixed = await self._smartblog_wait_with_progress(
                task=mix_task,
                job_id=str(job_id),
                progress=int(pending_progress),
                **dict(pending_stage_fields or {}),
            )
        else:
            mixed = await mix_task
        logging.warning(
            "SmartBlog render background_music applied: job=%s src=%s out=%s gain=%.2fdB duck=%.2fdB",
            str(job_id or "-"),
            os.path.basename(str(video_path)),
            os.path.basename(str(mixed)),
            float(cfg.get("gain_db") or 0.0),
            float(cfg.get("duck_voice_db") or 0.0),
        )
        return str(mixed)

    async def _smartblog_generate_mmaudio_for_hunyuan_clip(
        self,
        *,
        claim: dict[str, Any],
        video_path: str,
        run_dir: str,
        segment_index: int,
        prompt: str,
        negative_prompt: str,
        duration_sec: float,
        log_prefix: str,
        audio_config: dict[str, Any] | None = None,
    ) -> str:
        cfg = dict(audio_config or _smartblog_video_audio_config(claim))
        mode = str(cfg.get("mode") or "auto").strip().lower()
        if mode in {"off", "mute", "muted", "silent", "none", "no_audio", "no-audio"}:
            logging.warning("%s MMAudio skipped by video.audio.mode=%s", str(log_prefix), str(mode or "off"))
            return ""
        if mode == "asset":
            audio_url = str(cfg.get("audio_url") or "").strip()
            if not audio_url:
                logging.warning("%s video.audio.mode=asset has no audio_url; falling back to silent background", str(log_prefix))
                return ""
            ext = os.path.splitext(urllib.parse.urlparse(str(audio_url)).path)[1].strip() or ".audio"
            if len(ext) > 12:
                ext = ".audio"
            out_path = os.path.join(str(run_dir), f"video_audio_asset_{int(segment_index):03d}{ext}")
            await self._smartblog_download_file(url=str(audio_url), out_path=str(out_path))
            logging.warning("%s video.audio asset ready: %s", str(log_prefix), os.path.basename(str(out_path)))
            return str(out_path)
        if not _env_flag("SMARTBLOG_MMAUDIO_ENABLED", "0"):
            return ""
        try:
            duration = float(max(0.1, float(duration_sec or video_duration_sec(video_path) or 0.0)))
            audio_direction = ""
            if mode == "prompt":
                audio_direction = str(cfg.get("prompt") or "").strip()
                logging.warning(
                    "%s MMAudio using video.audio prompt mode via Gemini: direction_chars=%d",
                    str(log_prefix),
                    int(len(audio_direction)),
                )
            audio_prompt = await self._smartblog_gemini_audio_prompt_for_video(
                video_path=str(video_path),
                run_dir=str(run_dir),
                base_prompt=str(prompt or ""),
                audio_direction=str(audio_direction),
                duration_sec=float(duration),
                log_prefix=str(log_prefix),
            )
            if not audio_prompt:
                audio_prompt = _smartblog_render_default_mmaudio_prompt()
            default_negative = str(
                os.getenv(
                    "SMARTBLOG_MMAUDIO_NEGATIVE_PROMPT",
                    "speech, talking, narration, vocals, singing, music, melody, distorted, clipping, harsh noise",
                )
                or ""
            ).strip()
            if mode == "prompt":
                mmaudio_negative = _smartblog_first_text(str(cfg.get("negative_prompt") or ""), default_negative)
            else:
                mmaudio_negative = _smartblog_first_text(str(negative_prompt or ""), default_negative)
            out_path = os.path.join(str(run_dir), f"mmaudio_hunyuan_segment_{int(segment_index):03d}.wav")
            audio_path = await self._smartblog_mmaudio_service_generate(
                video_path=str(video_path),
                prompt=str(audio_prompt),
                negative_prompt=str(mmaudio_negative),
                duration_sec=float(duration),
                output_path=str(out_path),
                log_prefix=str(log_prefix),
            )
            logging.warning(
                "%s MMAudio ready: duration=%.3fs prompt_chars=%d audio=%s",
                str(log_prefix),
                float(duration),
                int(len(str(audio_prompt))),
                os.path.basename(str(audio_path)),
            )
            return str(audio_path)
        except Exception as e:
            if _env_flag("SMARTBLOG_MMAUDIO_FAIL_OPEN", "1"):
                logging.exception("%s MMAudio failed; falling back to silent Hunyuan audio: %s", str(log_prefix), str(e))
                return ""
            raise

    async def _smartblog_prepare_direct_clip_timeline_clip(
        self,
        claim: dict[str, Any],
        *,
        entry: dict[str, Any],
        run_dir: str,
        segment_index: int,
        segment_count: int,
        output_path: str,
        output_width: int,
        output_height: int,
        output_fps: float,
        duration_sec: float,
        render_job_type: str,
        render_progress: Any,
        audio_config: dict[str, Any] | None = None,
        report_progress: bool = True,
    ) -> str:
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        video_url = _smartblog_direct_clip_video_url(dict(entry or {}))
        if not video_url:
            raise RuntimeError(f"render_video direct insert {int(segment_index) + 1} requires video_url")
        duration = float(max(0.1, float(duration_sec or _smartblog_insert_duration_sec(dict(entry or {})) or 0.0)))
        direct_dir = os.path.join(str(run_dir), f"direct_clip_{int(segment_index):03d}")
        os.makedirs(direct_dir, exist_ok=True)
        ext = os.path.splitext(urllib.parse.urlparse(str(video_url)).path)[1].strip() or ".mp4"
        if len(ext) > 12:
            ext = ".mp4"
        source_path = os.path.join(str(direct_dir), f"source{ext}")
        if bool(report_progress):
            await self._smartblog_progress_checked(
                job_id=str(job_id),
                progress=render_progress("inference", float(segment_index) / float(max(1, int(segment_count)))),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="inference",
                    stage_label="Downloading direct insert",
                ),
            )
        await self._smartblog_download_file(url=str(video_url), out_path=str(source_path))
        processed_path = os.path.join(str(direct_dir), "processed_no_audio.mp4")
        media_req = MediaProcessRequest(
            source_path=str(source_path),
            source_kind="video",
            output_path=str(processed_path),
            output_width=int(output_width),
            output_height=int(output_height),
            output_fps=float(output_fps),
            preserve_audio=False,
            upscale=False,
            face_restore=0.0,
            background_restore=0.0,
            trim_duration_sec=float(duration),
        )
        processed_path = await self._smartblog_process_media_phase(
            job_id=str(job_id),
            req=media_req,
            pending_progress=(
                render_progress("inference", min(1.0, (float(segment_index) + 0.45) / float(max(1, int(segment_count)))))
                if bool(report_progress)
                else None
            ),
            **{
                f"pending_{key}": value
                for key, value in _smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="inference",
                    stage_label="Preparing direct insert",
                ).items()
            },
        )
        normalized_video = await asyncio.to_thread(
            _smartblog_normalize_direct_clip_video,
            src_path=str(processed_path),
            out_path=os.path.join(str(direct_dir), "normalized_no_audio.mp4"),
            duration_sec=float(duration),
            width=int(output_width),
            height=int(output_height),
            fps=float(output_fps),
        )
        video_audio_cfg = dict(audio_config) if isinstance(audio_config, dict) else None
        direct_audio_path = ""
        if video_audio_cfg:
            log_prefix = f"SmartBlog timeline direct insert job={job_id or '-'} segment={int(segment_index)}"
            direct_audio_path = await self._smartblog_generate_mmaudio_for_hunyuan_clip(
                claim=claim,
                video_path=str(normalized_video),
                run_dir=str(direct_dir),
                segment_index=int(segment_index),
                prompt=str(entry.get("_smartblog_ltx_prompt") or ""),
                negative_prompt=str(entry.get("_smartblog_ltx_negative_prompt") or ""),
                duration_sec=float(duration),
                log_prefix=str(log_prefix),
                audio_config=dict(video_audio_cfg),
            )
        muxed = await asyncio.to_thread(
            _smartblog_mux_mixed_audio,
            video_path=str(normalized_video),
            background_audio_path=str(direct_audio_path or ""),
            out_path=str(output_path),
            duration_sec=float(duration),
            sample_rate=48000,
            background_gain_db=(
                float(video_audio_cfg.get("gain_db") or 0.0)
                if isinstance(video_audio_cfg, dict) and str(video_audio_cfg.get("mode") or "") == "asset"
                else 0.0
            ),
        )
        if not os.path.exists(str(muxed)):
            raise RuntimeError(f"render_video direct insert {int(segment_index) + 1} produced no muxed output")
        logging.warning(
            "SmartBlog render direct insert ready: job=%s segment=%d/%d url=%s duration=%.3fs out=%dx%d fps=%.2f audio=%d",
            str(job_id or "-"),
            int(segment_index + 1),
            int(segment_count),
            os.path.basename(urllib.parse.urlparse(str(video_url)).path) or "-",
            float(duration),
            int(output_width),
            int(output_height),
            float(output_fps),
            1 if bool(direct_audio_path) else 0,
        )
        return str(muxed)

    async def _smartblog_render_hunyuan_timeline_clip(
        self,
        claim: dict[str, Any],
        *,
        entry: dict[str, Any],
        run_dir: str,
        segment_index: int,
        segment_count: int,
        output_path: str,
        output_width: int,
        output_height: int,
        output_fps: float,
        duration_sec: float,
        render_job_type: str,
        render_progress: Any,
        conditioning_image_path: str = "",
        audio_config: dict[str, Any] | None = None,
        report_progress: bool = True,
    ) -> str:
        if not _env_flag("SMARTBLOG_HUNYUAN_RENDER_ENABLED", os.getenv("SMARTBLOG_LTX_RENDER_ENABLED", "1") or "1"):
            raise RuntimeError("render_video timeline requested Hunyuan, but SMARTBLOG_HUNYUAN_RENDER_ENABLED/SMARTBLOG_LTX_RENDER_ENABLED=0")
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        cfg = dict(entry.get("_smartblog_ltx_config") or {})
        prompt = str(entry.get("_smartblog_ltx_prompt") or "").strip()
        if not prompt:
            prompt = _smartblog_first_text(cfg.get("prompt"), cfg.get("ltx_prompt"), cfg.get("ltxPrompt"))
        if not prompt:
            prompt = _smartblog_render_default_hunyuan_prompt()
        if not prompt:
            raise RuntimeError(f"render_video Hunyuan timeline segment {int(segment_index) + 1} requires prompt")
        negative_prompt = str(entry.get("_smartblog_ltx_negative_prompt") or "").strip()
        if not negative_prompt:
            negative_prompt = _smartblog_first_text(
                cfg.get("negative_prompt"),
                cfg.get("negativePrompt"),
                _smartblog_render_negative_prompt(claim),
            )
        duration = float(max(0.1, float(duration_sec or _smartblog_ltx_duration_seconds(cfg))))

        fake_claim = dict(claim)
        fake_video = dict(claim.get("video") or {}) if isinstance(claim.get("video"), dict) else {}
        fake_video["prompt"] = str(prompt)
        fake_video["effective_prompt"] = str(prompt)
        fake_video["negative_prompt"] = str(negative_prompt)
        fake_claim["video"] = fake_video
        ltx_cfg = dict(cfg)
        ltx_cfg["prompt"] = str(prompt)
        ltx_cfg["negative_prompt"] = str(negative_prompt)
        ltx_cfg["duration_seconds"] = float(duration)
        frame_rate = int(max(1, min(60, _smartblog_ltx_frame_rate({"ltx": ltx_cfg}))))
        if not _smartblog_ltx_has_frame_count(ltx_cfg):
            ltx_cfg["num_frames"] = int(max(9, math.ceil(float(duration) * float(frame_rate)) + 1))
        fake_claim["ltx"] = ltx_cfg
        width, height = _smartblog_ltx_dimensions(fake_claim)
        width = int(max(32, width))
        height = int(max(32, height))
        num_frames = int(_smartblog_ltx_num_frames(fake_claim, duration_sec=float(duration)))
        seed = int(_smartblog_ltx_seed(fake_claim))

        ltx_dir = os.path.join(str(run_dir), f"hunyuan_segment_{int(segment_index):03d}")
        output_dir = os.path.join(str(ltx_dir), "out")
        os.makedirs(output_dir, exist_ok=True)
        ltx_filters = _smartblog_render_entry_filters(claim, entry)
        ltx_face_restore = _smartblog_float_filter(ltx_filters, "face_restore", 0.5)
        ltx_background_restore = _smartblog_render_background_restore_filter(
            ltx_filters,
            job_id=f"{job_id or '-'}_hunyuan_{int(segment_index):03d}",
        )
        ltx_media_background_restore = _smartblog_ltx_media_background_restore(
            face_restore=float(ltx_face_restore),
            background_restore=float(ltx_background_restore),
        )
        image_url = str(entry.get("_smartblog_ltx_image_url") or "").strip()
        conditioning_path = ""
        override_conditioning_path = str(conditioning_image_path or "").strip()
        if override_conditioning_path and os.path.exists(override_conditioning_path):
            conditioning_path = str(override_conditioning_path)
        elif image_url:
            ext = os.path.splitext(urllib.parse.urlparse(str(image_url)).path)[1].strip() or ".png"
            if len(ext) > 8:
                ext = ".png"
            conditioning_path = os.path.join(str(ltx_dir), f"conditioning{ext}")
            await self._smartblog_download_file(url=str(image_url), out_path=str(conditioning_path))

        strength = entry.get("_smartblog_ltx_conditioning_strength")
        if strength is None:
            strength = _smartblog_ltx_conditioning_strength(fake_claim)
        if conditioning_path:
            strength = float(max(0.0, min(1.0, float(strength))))
        ltx_payload: dict[str, Any] = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt),
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "frame_rate": int(frame_rate),
            "seed": int(seed),
            "output_path": str(output_dir),
        }
        if conditioning_path:
            ltx_payload.update(
                {
                    "conditioning_media_paths": [str(conditioning_path)],
                    "conditioning_strengths": [float(strength)],
                    "conditioning_start_frames": [0],
                }
            )
        logging.warning(
            "SmartBlog render timeline Hunyuan insert config: job=%s segment=%d/%d size=%dx%d frames=%d fps=%d out=%dx%d out_fps=%.2f duration=%.3fs seed=%d condition=%d face_restore=%.2f background_restore=%.3f media_background_restore=%.3f prompt_chars=%d",
            str(job_id or "-"),
            int(segment_index + 1),
            int(segment_count),
            int(width),
            int(height),
            int(num_frames),
            int(frame_rate),
            int(output_width),
            int(output_height),
            float(output_fps),
            float(duration),
            int(seed),
            1 if conditioning_path else 0,
            float(ltx_face_restore),
            float(ltx_background_restore),
            float(ltx_media_background_restore),
            int(len(str(prompt))),
        )
        started_mono = float(time.monotonic())
        log_prefix = f"SmartBlog timeline Hunyuan insert job={job_id or '-'} segment={int(segment_index)}"
        progress_expected_sec = _smartblog_hunyuan_progress_expected_sec(
            num_frames=int(num_frames),
            frame_rate=int(frame_rate),
            duration_sec=float(duration),
        )
        logging.warning(
            "%s progress estimate: frames=%d fps=%d duration=%.3fs expected=%.1fs segments=%d",
            str(log_prefix),
            int(num_frames),
            int(frame_rate),
            float(duration),
            float(progress_expected_sec),
            int(segment_count),
        )
        ltx_task = asyncio.create_task(
            self._smartblog_hunyuan_service_generate(
                payload=dict(ltx_payload),
                log_prefix=str(log_prefix),
            )
        )

        def _ltx_progress_provider() -> dict[str, Any]:
            elapsed = max(0.0, float(time.monotonic()) - float(started_mono))
            expected = float(progress_expected_sec)
            local_frac = _smartblog_hunyuan_progress_local_frac(elapsed_sec=float(elapsed), expected_sec=float(expected))
            frac = (float(segment_index) + float(local_frac)) / float(max(1, int(segment_count)))
            return {
                "progress": render_progress("inference", frac),
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan insert"),
            }

        if bool(report_progress):
            ltx_result = await self._smartblog_wait_with_progress(
                task=ltx_task,
                job_id=str(job_id),
                progress=render_progress("inference", float(segment_index) / float(max(1, int(segment_count)))),
                progress_provider=_ltx_progress_provider,
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan insert"),
            )
        else:
            ltx_result = await ltx_task
        raw_video = str((ltx_result or {}).get("output_path") or "").strip() if isinstance(ltx_result, dict) else ""
        if not raw_video:
            raw_video = _smartblog_hunyuan_latest_mp4(output_dir)
        if not raw_video or not os.path.exists(raw_video):
            raise RuntimeError(f"render_video Hunyuan timeline segment {int(segment_index) + 1} produced no mp4 output")
        processed_path = os.path.join(str(ltx_dir), "processed.mp4")
        media_req = MediaProcessRequest(
            source_path=str(raw_video),
            source_kind="video",
            output_path=str(processed_path),
            output_width=int(output_width),
            output_height=int(output_height),
            output_fps=float(output_fps),
            preserve_audio=False,
            upscale=bool(int(output_width) > int(width) or int(output_height) > int(height)),
            face_restore=float(ltx_face_restore),
            background_restore=float(ltx_media_background_restore),
            trim_duration_sec=float(duration),
        )
        processed_path = await self._smartblog_process_media_phase(
            job_id=str(job_id),
            req=media_req,
            pending_progress=(
                render_progress("inference", min(1.0, (float(segment_index) + 0.95) / float(max(1, int(segment_count)))))
                if bool(report_progress)
                else None
            ),
            **{
                f"pending_{key}": value
                for key, value in _smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="inference",
                    stage_label="Encoding Hunyuan insert",
                ).items()
            },
        )
        processed_path = await asyncio.to_thread(
            _smartblog_apply_hunyuan_visual_match,
            src_path=str(processed_path),
            out_path=os.path.join(str(ltx_dir), "matched.mp4"),
            reference_image_path=str(conditioning_path),
        )
        video_audio_cfg = dict(audio_config) if isinstance(audio_config, dict) else _smartblog_video_audio_config(claim)
        mmaudio_started = float(time.monotonic())
        mmaudio_task = asyncio.create_task(
            self._smartblog_generate_mmaudio_for_hunyuan_clip(
                claim=claim,
                video_path=str(processed_path),
                run_dir=str(ltx_dir),
                segment_index=int(segment_index),
                prompt=str(prompt),
                negative_prompt=str(negative_prompt),
                duration_sec=float(duration),
                log_prefix=str(log_prefix),
                audio_config=dict(video_audio_cfg),
            )
        )

        def _mmaudio_progress_provider() -> dict[str, Any]:
            elapsed = max(0.0, float(time.monotonic()) - float(mmaudio_started))
            expected = max(5.0, _safe_float_env("SMARTBLOG_MMAUDIO_PROGRESS_EXPECTED_SEC", 12.0))
            frac = (
                float(segment_index)
                + min(0.99, 0.95 + 0.04 * min(1.0, elapsed / expected))
            ) / float(max(1, int(segment_count)))
            return {
                "progress": render_progress("inference", frac),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="inference",
                    stage_label="Generating Hunyuan audio",
                ),
            }

        if bool(report_progress):
            mmaudio_path = await self._smartblog_wait_with_progress(
                task=mmaudio_task,
                job_id=str(job_id),
                progress=render_progress(
                    "inference",
                    (float(segment_index) + 0.95) / float(max(1, int(segment_count))),
                ),
                progress_provider=_mmaudio_progress_provider,
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="inference",
                    stage_label="Generating Hunyuan audio",
                ),
            )
        else:
            mmaudio_path = await mmaudio_task
        muxed = await asyncio.to_thread(
            _smartblog_mux_mixed_audio,
            video_path=str(processed_path),
            background_audio_path=str(mmaudio_path or ""),
            out_path=str(output_path),
            duration_sec=float(duration),
            sample_rate=48000,
            background_gain_db=float(video_audio_cfg.get("gain_db") or 0.0) if str(video_audio_cfg.get("mode") or "") == "asset" else 0.0,
        )
        if not os.path.exists(str(muxed)):
            raise RuntimeError(f"render_video Hunyuan timeline segment {int(segment_index) + 1} produced no muxed output")
        return str(muxed)

    async def _smartblog_compose_avatar_segment_inserts(
        self,
        *,
        claim: dict[str, Any],
        base_path: str,
        info: dict[str, Any],
        run_dir: str,
        segment_index: int,
        segment_count: int,
        output_width: int,
        output_height: int,
        output_fps: float,
        render_job_type: str,
        render_progress: Any,
        previous_timeline_path: str = "",
    ) -> tuple[str, str]:
        entry = dict(info.get("audio_entry") or {})
        inserts = _smartblog_frame_insert_entries(entry)
        if not inserts:
            info["timeline_duration_sec"] = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
            return str(base_path), str(base_path)

        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        segment_duration = float(info.get("target_duration_sec") or info.get("duration_sec") or video_duration_sec(str(base_path)) or 0.0)
        chunks = _smartblog_avatar_frame_audio_chunks(dict(info or {}))
        composed_base = str(base_path)
        cut_inserts: list[dict[str, Any]] = []
        overlay_inserts: list[dict[str, Any]] = []
        for raw in inserts:
            item = dict(raw)
            item["_smartblog_insert_start_sec"] = float(
                _smartblog_insert_start_sec(item, chunks=list(chunks), segment_duration_sec=float(segment_duration))
            )
            if _smartblog_insert_mode(item) == "overlay":
                overlay_inserts.append(item)
            else:
                cut_inserts.append(item)
        overlay_inserts.sort(key=lambda item: (float(item.get("_smartblog_insert_start_sec") or 0.0), int(item.get("_smartblog_insert_order") or 0)))
        cut_inserts.sort(key=lambda item: (float(item.get("_smartblog_insert_start_sec") or 0.0), int(item.get("_smartblog_insert_order") or 0)))

        async def render_insert_clip(insert: dict[str, Any], *, previous_visual_path: str, insert_index: int) -> str:
            duration = float(max(0.1, _smartblog_insert_duration_sec(insert)))
            transition_code = int(max(0, min(100, int(insert.get("_smartblog_transition_code") or 0))))
            audio_block = insert.get("_smartblog_ltx_audio") if isinstance(insert.get("_smartblog_ltx_audio"), dict) else {}
            insert_audio_cfg = (
                _smartblog_normalize_video_audio_config(dict(audio_block), default_gain_db=0.0)
                if audio_block
                else None
            )
            conditioning_image_path = ""
            if (
                not _smartblog_direct_clip_requested(dict(insert or {}))
                and bool(insert.get("_smartblog_ltx_use_previous_last_frame"))
                and previous_visual_path
                and os.path.exists(str(previous_visual_path))
            ):
                filters = _smartblog_render_entry_filters(claim, insert)
                face_restore = _smartblog_float_filter(filters, "face_restore", 0.5)
                background_restore = _smartblog_render_background_restore_filter(
                    filters,
                    job_id=f"{job_id or '-'}_insert_continuation_{int(segment_index):03d}_{int(insert_index):03d}",
                )
                conditioning_image_path = await self._smartblog_prepare_render_continuation_image(
                    job_id=str(job_id),
                    source_video_path=str(previous_visual_path),
                    raw_frame_path=os.path.join(
                        str(run_dir),
                        f"insert_{int(segment_index):03d}_{int(insert_index):03d}_continuation_raw.png",
                    ),
                    output_path=os.path.join(
                        str(run_dir),
                        f"insert_{int(segment_index):03d}_{int(insert_index):03d}_continuation_enhanced.png",
                    ),
                    face_restore=float(face_restore),
                    segment_index=int(segment_index),
                    reason="insert",
                    background_restore=float(background_restore),
                    target_width=int(output_width),
                    target_height=int(output_height),
                )
            output_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_insert_{int(insert_index):03d}.mp4")
            if _smartblog_direct_clip_requested(dict(insert or {})):
                hunyuan_path = await self._smartblog_prepare_direct_clip_timeline_clip(
                    claim=claim,
                    entry=dict(insert),
                    run_dir=str(run_dir),
                    segment_index=int(segment_index),
                    segment_count=int(max(1, int(segment_count))),
                    output_path=str(output_path),
                    output_width=int(output_width),
                    output_height=int(output_height),
                    output_fps=float(output_fps),
                    duration_sec=float(duration),
                    render_job_type=str(render_job_type),
                    render_progress=render_progress,
                    audio_config=insert_audio_cfg,
                )
            else:
                hunyuan_path = await self._smartblog_render_hunyuan_timeline_clip(
                    claim=claim,
                    entry=dict(insert),
                    run_dir=str(run_dir),
                    segment_index=int(segment_index),
                    segment_count=int(max(1, int(segment_count))),
                    output_path=str(output_path),
                    output_width=int(output_width),
                    output_height=int(output_height),
                    output_fps=float(output_fps),
                    duration_sec=float(duration),
                    render_job_type=str(render_job_type),
                    render_progress=render_progress,
                    conditioning_image_path=str(conditioning_image_path),
                    audio_config=insert_audio_cfg,
                )
            if transition_code > 0 and bool(_env_flag("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR", "1")):
                hunyuan_path = await asyncio.to_thread(
                    _smartblog_apply_segment_transition_blur,
                    src_path=str(hunyuan_path),
                    out_path=os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_insert_{int(insert_index):03d}_transition.mp4"),
                    has_start=True,
                    has_end=False,
                    width=int(output_width),
                    height=int(output_height),
                )
            return str(hunyuan_path)

        for overlay_idx, insert in enumerate(overlay_inserts):
            start = float(max(0.0, min(float(segment_duration), float(insert.get("_smartblog_insert_start_sec") or 0.0))))
            duration = float(max(0.1, min(_smartblog_insert_duration_sec(insert), max(0.1, float(segment_duration) - float(start)))))
            previous_for_overlay = str(previous_timeline_path or "")
            if bool(insert.get("_smartblog_ltx_use_previous_last_frame")) and start > 0.05:
                prefix_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_overlay_{int(overlay_idx):03d}_prefix.mp4")
                trimmed = await asyncio.to_thread(
                    _smartblog_trim_mp4_interval,
                    src_path=str(composed_base),
                    out_path=str(prefix_path),
                    start_sec=0.0,
                    end_sec=float(start),
                    width=int(output_width),
                    height=int(output_height),
                    fps=float(output_fps),
                )
                if trimmed:
                    previous_for_overlay = str(trimmed)
            hunyuan_path = await render_insert_clip(insert, previous_visual_path=str(previous_for_overlay), insert_index=1000 + int(overlay_idx))
            composed_base = await asyncio.to_thread(
                _smartblog_overlay_timeline_clip,
                base_path=str(composed_base),
                overlay_path=str(hunyuan_path),
                out_path=os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_overlay_{int(overlay_idx):03d}.mp4"),
                start_sec=float(start),
                duration_sec=float(duration),
                width=int(output_width),
                height=int(output_height),
            )

        overlay_duration = float(video_duration_sec(str(composed_base)) or 0.0)
        if overlay_duration > 0.0 and overlay_duration > float(segment_duration) + 0.08:
            normalized_overlay_path = os.path.join(
                str(run_dir),
                f"render_segment_{int(segment_index):03d}_overlay_timeline_normalized.mp4",
            )
            normalized = await asyncio.to_thread(
                _smartblog_trim_mp4_interval,
                src_path=str(composed_base),
                out_path=str(normalized_overlay_path),
                start_sec=0.0,
                end_sec=float(segment_duration),
                width=int(output_width),
                height=int(output_height),
                fps=float(output_fps),
            )
            if normalized:
                logging.warning(
                    "SmartBlog render avatar overlay duration normalized: job=%s segment=%d expected=%.3fs actual=%.3fs",
                    str(job_id or "-"),
                    int(segment_index),
                    float(segment_duration),
                    float(overlay_duration),
                )
                composed_base = str(normalized)

        if not cut_inserts:
            info["timeline_duration_sec"] = float(segment_duration)
            logging.warning(
                "SmartBlog render avatar inserts composed: job=%s segment=%d overlays=%d cuts=0 timeline=%.3fs",
                str(job_id or "-"),
                int(segment_index),
                int(len(overlay_inserts)),
                float(segment_duration),
            )
            return str(composed_base), str(composed_base)

        pieces: list[str] = []
        cursor = 0.0
        total_cut_duration = 0.0
        for cut_idx, insert in enumerate(cut_inserts):
            start = float(max(cursor, min(float(segment_duration), float(insert.get("_smartblog_insert_start_sec") or 0.0))))
            if start > cursor + 0.005:
                part_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_part_{int(cut_idx):03d}.mp4")
                trimmed = await asyncio.to_thread(
                    _smartblog_trim_mp4_interval,
                    src_path=str(composed_base),
                    out_path=str(part_path),
                    start_sec=float(cursor),
                    end_sec=float(start),
                    width=int(output_width),
                    height=int(output_height),
                    fps=float(output_fps),
                )
                if trimmed:
                    pieces.append(str(trimmed))
            previous_visual = str(pieces[-1]) if pieces else str(previous_timeline_path or "")
            hunyuan_path = await render_insert_clip(insert, previous_visual_path=str(previous_visual), insert_index=int(cut_idx))
            pieces.append(str(hunyuan_path))
            total_cut_duration += float(_smartblog_insert_duration_sec(insert))
            cursor = float(start)
        if cursor < segment_duration - 0.005:
            part_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_part_tail.mp4")
            trimmed = await asyncio.to_thread(
                _smartblog_trim_mp4_interval,
                src_path=str(composed_base),
                out_path=str(part_path),
                start_sec=float(cursor),
                end_sec=float(segment_duration),
                width=int(output_width),
                height=int(output_height),
                fps=float(output_fps),
            )
            if trimmed:
                pieces.append(str(trimmed))
        if not pieces:
            info["timeline_duration_sec"] = float(segment_duration)
            return str(composed_base), str(composed_base)
        composed_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_composed_inserts.mp4")
        expected_timeline_duration = float(segment_duration + total_cut_duration)
        await asyncio.to_thread(
            _smartblog_concat_mp4_reencode,
            list(pieces),
            str(composed_path),
            width=int(output_width),
            height=int(output_height),
            fps=float(output_fps),
        )
        actual_timeline_duration = float(video_duration_sec(str(composed_path)) or 0.0)
        if actual_timeline_duration > 0.0 and abs(float(actual_timeline_duration) - float(expected_timeline_duration)) > 0.35:
            if actual_timeline_duration > float(expected_timeline_duration) + 0.35:
                normalized_path = os.path.join(str(run_dir), f"render_segment_{int(segment_index):03d}_composed_inserts_normalized.mp4")
                normalized = await asyncio.to_thread(
                    _smartblog_trim_mp4_interval,
                    src_path=str(composed_path),
                    out_path=str(normalized_path),
                    start_sec=0.0,
                    end_sec=float(expected_timeline_duration),
                    width=int(output_width),
                    height=int(output_height),
                    fps=float(output_fps),
                )
                if normalized:
                    logging.warning(
                        "SmartBlog render insert compose duration normalized: job=%s segment=%d expected=%.3fs actual=%.3fs pieces=%d cuts=%d overlays=%d",
                        str(job_id or "-"),
                        int(segment_index),
                        float(expected_timeline_duration),
                        float(actual_timeline_duration),
                        int(len(pieces)),
                        int(len(cut_inserts)),
                        int(len(overlay_inserts)),
                    )
                    composed_path = str(normalized)
                    actual_timeline_duration = float(video_duration_sec(str(composed_path)) or expected_timeline_duration)
            if actual_timeline_duration > 0.0 and abs(float(actual_timeline_duration) - float(expected_timeline_duration)) > 0.35:
                raise RuntimeError(
                    "render insert compose duration mismatch: "
                    f"job={job_id or '-'} segment={int(segment_index)} "
                    f"expected={expected_timeline_duration:.3f}s actual={actual_timeline_duration:.3f}s "
                    f"pieces={len(pieces)} cuts={len(cut_inserts)} overlays={len(overlay_inserts)}"
                )
        if actual_timeline_duration > 0.0 and actual_timeline_duration < float(expected_timeline_duration) - 0.35:
            raise RuntimeError(
                "render insert compose duration mismatch: "
                f"job={job_id or '-'} segment={int(segment_index)} "
                f"expected={expected_timeline_duration:.3f}s actual={actual_timeline_duration:.3f}s "
                f"pieces={len(pieces)} cuts={len(cut_inserts)} overlays={len(overlay_inserts)}"
            )
        info["timeline_duration_sec"] = float(actual_timeline_duration or expected_timeline_duration)
        logging.warning(
            "SmartBlog render avatar inserts composed: job=%s segment=%d overlays=%d cuts=%d avatar=%.3fs inserted=%.3fs timeline=%.3fs actual=%.3fs",
            str(job_id or "-"),
            int(segment_index),
            int(len(overlay_inserts)),
            int(len(cut_inserts)),
            float(segment_duration),
            float(total_cut_duration),
            float(info.get("timeline_duration_sec") or 0.0),
            float(actual_timeline_duration or 0.0),
        )
        return str(composed_path), str(composed_path)

    async def _smartblog_render_hunyuan_video_job(
        self,
        claim: dict[str, Any],
        *,
        run_dir: str,
        render_job_type: str,
        render_progress: Any,
        audio_entries: list[dict[str, Any]] | None = None,
        render_mode: str | None = None,
    ) -> SmartBlogRenderFinalizePlan:
        if not _env_flag("SMARTBLOG_HUNYUAN_RENDER_ENABLED", os.getenv("SMARTBLOG_LTX_RENDER_ENABLED", "1") or "1"):
            raise RuntimeError("render_video requested Hunyuan backend, but SMARTBLOG_HUNYUAN_RENDER_ENABLED/SMARTBLOG_LTX_RENDER_ENABLED=0")
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        upload = claim.get("upload") if isinstance(claim.get("upload"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        render_mode_s = str(render_mode or _smartblog_render_mode(claim) or "i2v").strip().lower()
        if render_mode_s not in {"i2v", "t2v"}:
            render_mode_s = "i2v"
        prompt = _smartblog_ltx_prompt(claim)
        if not prompt:
            raise RuntimeError("render_video Hunyuan backend requires video.prompt/effective_prompt/prompt")
        negative_prompt = _smartblog_render_negative_prompt(claim)
        voice_audio_path, voice_duration_sec = await self._smartblog_prepare_hunyuan_voice_audio(
            audio_entries=list(audio_entries or []),
            run_dir=str(run_dir),
            job_id=str(job_id),
        )
        target_duration_sec = float(voice_duration_sec or _smartblog_ltx_claim_duration_seconds(claim, default=0.0) or 0.0)
        width, height = _smartblog_ltx_dimensions(claim)
        width = int(max(32, width))
        height = int(max(32, height))
        out_w, out_h = _smartblog_ltx_output_dimensions(claim, width=int(width), height=int(height))
        frame_rate = int(_smartblog_ltx_frame_rate(claim))
        num_frames = int(_smartblog_ltx_num_frames(claim, duration_sec=float(target_duration_sec) if target_duration_sec > 0.0 else None))
        output_fps = float(
            max(
                1.0,
                min(
                    60.0,
                    _safe_float_env("SMARTBLOG_HUNYUAN_OUTPUT_FPS", _safe_float_env("SMARTBLOG_LTX_OUTPUT_FPS", float(frame_rate))),
                ),
            )
        )
        seed = int(_smartblog_ltx_seed(claim))
        watermark_text = _smartblog_watermark_text(claim)
        remote_finalizer_enabled = bool(
            _smartblog_render_edge_finalizer_background_enabled()
            and str(_smartblog_file_upscale_service_url() or "").strip()
        )
        filters = _smartblog_render_entry_filters(claim, None)
        face_restore = _smartblog_float_filter(filters, "face_restore", 0.5)
        background_restore = _smartblog_render_background_restore_filter(filters, job_id=job_id)
        media_background_restore = _smartblog_ltx_media_background_restore(
            face_restore=float(face_restore),
            background_restore=float(background_restore),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=max(1, render_progress("prepare", 0.2)),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="prepare", stage_label="Preparing Hunyuan request"),
        )

        avatar_urls = _smartblog_render_asset_urls(claim, "avatar")
        conditioning_path = ""
        if render_mode_s != "t2v" and avatar_urls:
            ext = os.path.splitext(urllib.parse.urlparse(str(avatar_urls[0])).path)[1].strip() or ".png"
            if len(ext) > 8:
                ext = ".png"
            conditioning_path = os.path.join(run_dir, f"hunyuan_conditioning{ext}")
            await self._smartblog_download_file(url=str(avatar_urls[0]), out_path=str(conditioning_path))

        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("prepare", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="prepare", stage_label="Prepared Hunyuan assets"),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("tts", 1.0),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="tts",
                stage_label="Hunyuan voice audio ready" if voice_audio_path else "Hunyuan does not require TTS",
            ),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("face_detect", 1.0),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="face_detect",
                stage_label="Hunyuan T2V ready" if render_mode_s == "t2v" else "Hunyuan conditioning ready",
            ),
        )

        output_dir = os.path.join(run_dir, "hunyuan_output")
        os.makedirs(output_dir, exist_ok=True)
        ltx_payload: dict[str, Any] = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt),
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "frame_rate": int(frame_rate),
            "seed": int(seed),
            "output_path": str(output_dir),
            "task": str(render_mode_s),
            "render_mode": str(render_mode_s),
        }
        if conditioning_path and render_mode_s != "t2v":
            ltx_payload.update(
                {
                    "conditioning_media_paths": [str(conditioning_path)],
                    "conditioning_strengths": [float(_smartblog_ltx_conditioning_strength(claim))],
                    "conditioning_start_frames": [0],
                }
            )
        logging.warning(
            "SmartBlog Hunyuan render config: job=%s mode=%s size=%dx%d frames=%d fps=%d output_fps=%.2f final=%dx%d seed=%d condition=%d voice=%.3fs face_restore=%.2f background_restore=%.3f media_background_restore=%.3f prompt_chars=%d negative_chars=%d",
            str(job_id),
            str(render_mode_s),
            int(width),
            int(height),
            int(num_frames),
            int(frame_rate),
            float(output_fps),
            int(out_w),
            int(out_h),
            int(seed),
            1 if conditioning_path and render_mode_s != "t2v" else 0,
            float(voice_duration_sec or 0.0),
            float(face_restore),
            float(background_restore),
            float(media_background_restore),
            int(len(str(prompt))),
            int(len(str(negative_prompt))),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("inference", 0.02),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan"),
        )
        started_mono = float(time.monotonic())
        log_prefix = f"SmartBlog Hunyuan job={job_id or '-'}"
        progress_expected_sec = _smartblog_hunyuan_progress_expected_sec(
            num_frames=int(num_frames),
            frame_rate=int(frame_rate),
            duration_sec=float(num_frames) / float(max(1, int(frame_rate))),
        )
        logging.warning(
            "%s progress estimate: frames=%d fps=%d expected=%.1fs",
            str(log_prefix),
            int(num_frames),
            int(frame_rate),
            float(progress_expected_sec),
        )
        ltx_task = asyncio.create_task(
            self._smartblog_hunyuan_service_generate(
                payload=dict(ltx_payload),
                log_prefix=str(log_prefix),
            )
        )

        def _ltx_progress_provider() -> dict[str, Any]:
            elapsed = max(0.0, float(time.monotonic()) - float(started_mono))
            expected = float(progress_expected_sec)
            local_frac = _smartblog_hunyuan_progress_local_frac(elapsed_sec=float(elapsed), expected_sec=float(expected))
            frac = 0.02 + 0.90 * float(local_frac)
            return {
                "progress": render_progress("inference", frac),
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan"),
            }

        ltx_result = await self._smartblog_wait_with_progress(
            task=ltx_task,
            job_id=job_id,
            progress=render_progress("inference", 0.02),
            progress_provider=_ltx_progress_provider,
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan"),
        )
        raw_video = str((ltx_result or {}).get("output_path") or "").strip() if isinstance(ltx_result, dict) else ""
        if not raw_video:
            raw_video = _smartblog_hunyuan_latest_mp4(output_dir)
        if not raw_video or not os.path.exists(raw_video):
            raise RuntimeError("render_video Hunyuan produced no mp4 output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("inference", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Hunyuan complete"),
        )

        final_video_path = os.path.join(run_dir, "render_hunyuan_final.mp4")
        media_output_w = int(width) if bool(remote_finalizer_enabled) else int(out_w)
        media_output_h = int(height) if bool(remote_finalizer_enabled) else int(out_h)
        media_req = MediaProcessRequest(
            source_path=str(raw_video),
            source_kind="video",
            output_path=str(final_video_path),
            output_width=int(media_output_w),
            output_height=int(media_output_h),
            output_fps=float(output_fps),
            preserve_audio=False,
            upscale=bool(int(media_output_w) > int(width) or int(media_output_h) > int(height)),
            face_restore=float(face_restore),
            background_restore=float(media_background_restore),
            trim_duration_sec=float(target_duration_sec) if target_duration_sec > 0.0 else 0.0,
        )
        out_path = await self._smartblog_process_media_phase(
            job_id=job_id,
            req=media_req,
            pending_progress=render_progress("encode", 0.25),
            **{
                f"pending_{key}": value
                for key, value in _smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Encoding Hunyuan output",
                ).items()
            },
        )
        visual_match_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_apply_hunyuan_visual_match,
                src_path=str(out_path),
                out_path=os.path.join(run_dir, "render_hunyuan_matched.mp4"),
            ),
            name=f"smartblog-hunyuan-visual-match-{sanitize_job_id(str(job_id or 'job'))}",
        )
        out_path = await self._smartblog_wait_with_progress(
            task=visual_match_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.35),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Matching Hunyuan visual style",
            ),
        )
        duration = float(max(0.1, float(target_duration_sec or 0.0) or (float(num_frames) / float(max(1, int(frame_rate))))))
        video_audio_cfg = _smartblog_video_audio_config(claim)
        mmaudio_started = float(time.monotonic())
        mmaudio_task = asyncio.create_task(
            self._smartblog_generate_mmaudio_for_hunyuan_clip(
                claim=claim,
                video_path=str(out_path),
                run_dir=str(run_dir),
                segment_index=0,
                prompt=str(prompt),
                negative_prompt=str(negative_prompt),
                duration_sec=float(duration),
                log_prefix=f"SmartBlog Hunyuan job={job_id or '-'}",
                audio_config=dict(video_audio_cfg),
            )
        )

        def _mmaudio_progress_provider() -> dict[str, Any]:
            elapsed = max(0.0, float(time.monotonic()) - float(mmaudio_started))
            expected = max(5.0, _safe_float_env("SMARTBLOG_MMAUDIO_PROGRESS_EXPECTED_SEC", 12.0))
            frac = 0.45 + 0.25 * min(1.0, elapsed / expected)
            return {
                "progress": render_progress("encode", frac),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Generating Hunyuan audio",
                ),
            }

        mmaudio_path = await self._smartblog_wait_with_progress(
            task=mmaudio_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.45),
            progress_provider=_mmaudio_progress_provider,
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Generating Hunyuan audio",
            ),
        )
        if voice_audio_path or mmaudio_path:
            audio_mux_task = asyncio.create_task(
                asyncio.to_thread(
                    _smartblog_mux_mixed_audio,
                    video_path=str(out_path),
                    voice_audio_path=str(voice_audio_path or ""),
                    background_audio_path=str(mmaudio_path or ""),
                    out_path=os.path.join(run_dir, "render_hunyuan_audio.mp4"),
                    duration_sec=float(duration),
                    sample_rate=48000,
                    voice_gain_db=0.0,
                    background_gain_db=(
                        float(video_audio_cfg.get("gain_db") or 0.0)
                        if str(video_audio_cfg.get("mode") or "") == "asset"
                        else 0.0
                    ),
                ),
                name=f"smartblog-hunyuan-audio-mux-{sanitize_job_id(str(job_id or 'job'))}",
            )
            out_path = await self._smartblog_wait_with_progress(
                task=audio_mux_task,
                job_id=str(job_id),
                progress=render_progress("encode", 0.72),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Mixing Hunyuan audio",
                ),
            )
        if bool(remote_finalizer_enabled):
            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=render_progress("encode", 0.80),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Uploading Hunyuan for media finalizer",
                ),
            )
            source_upload_path = (
                f"worker-uploads/render_segments/{sanitize_job_id(job_id or 'render')}/"
                f"{sanitize_job_id(job_id or 'render')}_hunyuan_pre_finalizer.mp4"
            )
            source_upload_plan = SmartBlogRenderFinalizePlan(
                job_id=f"{job_id}_hunyuan_pre_finalizer",
                job_type=str(render_job_type),
                signed_url="",
                upload_path=str(source_upload_path),
                file_path=str(out_path),
                content_type="video/mp4",
                complete_kwargs={},
                run_dir=str(run_dir),
            )
            source_signed_url, source_uploaded_path = await self._smartblog_resolve_upload_target(source_upload_plan)
            await self._smartblog_upload_file(
                signed_url=str(source_signed_url),
                file_path=str(out_path),
                content_type="video/mp4",
            )
            burn_subtitles = bool(_smartblog_render_burn_in_subtitles_enabled(claim))
            remote_finalizer_subtitle_chunks_json = (
                _smartblog_render_subtitle_chunks_json_from_audio_entries(
                    list(audio_entries or []),
                    total_duration_sec=float(duration),
                )
                if bool(burn_subtitles)
                else ""
            )
            background_music_cfg = _smartblog_background_music_config(claim)
            background_music_enabled = bool(background_music_cfg.get("enabled"))
            remote_source_fps = int(round(float(output_fps)))
            remote_target_fps = int(round(float(_smartblog_render_delivery_fps())))
            remote_upscale_enabled = bool(_smartblog_remote_finalizer_quality_pass_enabled())
            remote_target_w, remote_target_h = _smartblog_render_delivery_dimensions(
                claim,
                output_width=int(out_w),
                output_height=int(out_h),
            )
            upload_path_claim = str(upload.get("path") or "").strip()
            upload_public_url = self._smartblog_public_storage_url(upload_path_claim) if upload_path_claim else ""
            logging.warning(
                "SmartBlog Hunyuan queued remote finalizer: job=%s source=%s target=%s subtitles_finalizer=%d watermark_chars=%d background_music=%d background_music_gain=%.2fdB upscale=%d target=%dx%d fps=%d->%d",
                str(job_id or "-"),
                str(source_uploaded_path),
                str(upload_path_claim),
                1 if bool(remote_finalizer_subtitle_chunks_json) else 0,
                int(len(str(watermark_text or "").strip())),
                1 if bool(background_music_enabled) else 0,
                float(background_music_cfg.get("gain_db") or 0.0),
                1 if bool(remote_upscale_enabled) else 0,
                int(remote_target_w),
                int(remote_target_h),
                int(remote_source_fps),
                int(remote_target_fps),
            )
            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=render_progress("encode", 0.02),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Queued media finalizer",
                ),
            )
            return SmartBlogRenderFinalizePlan(
                job_id=job_id,
                job_type=str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
                signed_url=str(upload.get("signed_url") or ""),
                upload_path=str(upload_path_claim),
                file_path="",
                content_type="video/mp4",
                complete_kwargs={"video_url": str(upload_public_url)} if upload_public_url else {},
                run_dir=str(run_dir),
                remote_finalizer_source_path=str(source_uploaded_path),
                remote_finalizer_source_fps=int(remote_source_fps),
                remote_finalizer_target_width=int(remote_target_w),
                remote_finalizer_target_height=int(remote_target_h),
                remote_finalizer_target_fps=int(remote_target_fps),
                remote_finalizer_upscale_enabled=bool(remote_upscale_enabled),
                remote_finalizer_background_music_url=(
                    str(background_music_cfg.get("audio_url") or "") if bool(background_music_enabled) else ""
                ),
                remote_finalizer_background_music_gain_db=float(background_music_cfg.get("gain_db") or 0.0),
                remote_finalizer_background_music_loop=bool(background_music_cfg.get("loop", True)),
                remote_finalizer_background_music_duck_voice_db=float(
                    background_music_cfg.get("duck_voice_db") or 0.0
                ),
                remote_finalizer_background_music_fade_in_seconds=float(
                    background_music_cfg.get("fade_in_seconds") or 0.0
                ),
                remote_finalizer_background_music_fade_out_seconds=float(
                    background_music_cfg.get("fade_out_seconds") or 0.0
                ),
                remote_finalizer_background_music_start_offset_seconds=float(
                    background_music_cfg.get("start_offset_seconds") or 0.0
                ),
                remote_finalizer_subtitle_chunks_json=str(remote_finalizer_subtitle_chunks_json),
                remote_finalizer_watermark_text=str(watermark_text or ""),
            )
        out_path = await self._smartblog_apply_background_music_if_needed(
            claim=claim,
            video_path=str(out_path),
            run_dir=str(run_dir),
            out_path=os.path.join(run_dir, "render_hunyuan_background_music.mp4"),
            job_id=str(job_id),
            pending_progress=render_progress("encode", 0.82),
            pending_stage_fields=_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Mixing background music",
            ),
        )
        watermark_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_maybe_burn_render_watermark,
                video_path=str(out_path),
                out_path=os.path.join(run_dir, "render_hunyuan_watermarked.mp4"),
                watermark_text=str(watermark_text or ""),
                run_dir=str(run_dir),
                width=int(out_w),
                height=int(out_h),
            ),
            name=f"smartblog-hunyuan-watermark-{sanitize_job_id(str(job_id or 'job'))}",
        )
        out_path = await self._smartblog_wait_with_progress(
            task=watermark_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.92),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Burning watermark",
            ),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="encode", stage_label="Encoded Hunyuan output"),
        )
        upload_path_claim = str(upload.get("path") or "").strip()
        upload_public_url = self._smartblog_public_storage_url(upload_path_claim) if upload_path_claim else ""
        return SmartBlogRenderFinalizePlan(
            job_id=job_id,
            job_type=str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
            signed_url=str(upload.get("signed_url") or ""),
            upload_path=str(upload.get("path") or ""),
            file_path=str(out_path),
            content_type="video/mp4",
            complete_kwargs={
                "video_url": str(upload_public_url or self._smartblog_public_storage_url(str(upload.get("path") or ""))),
            },
            run_dir=str(run_dir),
        )

    async def _smartblog_synthesize_render_tts_audio_entries(
        self,
        *,
        claim: dict[str, Any],
        run_dir: str,
        job_id: str,
        render_job_type: str,
        render_progress: Any,
    ) -> list[dict[str, Any]]:
        script_text = _smartblog_render_script_text(claim)
        chunks, nextframe_block_count = _smartblog_split_script_for_render_tts(script_text)
        if not chunks:
            raise RuntimeError(
                "render_video has no audio_url/audio_chunks and no assets.script_text for worker-side TTS"
            )

        voice_id, model_id, voice_settings, stability_mode = _smartblog_render_voice_config(claim)
        api_key = str(
            self._runtime_eleven_api_key()
            or os.getenv("ELEVENLABS_API_KEY", "")
            or os.getenv("ELEVENLABS_KEY", "")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError("render_video worker-side TTS requires ELEVENLABS_API_KEY")
        if not voice_id:
            raise RuntimeError("render_video worker-side TTS requires claim.voice.voice_id")
        if not model_id:
            model_id = "eleven_v3"

        session = await self._get_render_eleven_http_session()
        seed = self._render_eleven_v3_seed()
        total = int(len(chunks))
        avatar_count = int(len(_smartblog_render_asset_urls(claim, "avatar")))
        concurrency = int(_smartblog_render_tts_concurrency())
        semaphore = asyncio.Semaphore(int(concurrency))
        progress_lock = asyncio.Lock()
        completed_count = 0
        logging.warning(
            "SmartBlog render worker TTS start: job=%s chunks=%d nextframe_blocks=%d max_chars=%d target_words=%d concurrency=%d voice=%s model=%s stability_mode=%s settings=%s",
            str(job_id or "-"),
            int(total),
            int(nextframe_block_count),
            int(_smartblog_render_tts_max_chars()),
            int(_smartblog_render_tts_target_words()),
            int(concurrency),
            str(voice_id),
            str(model_id),
            str(stability_mode),
            json.dumps(dict(voice_settings or {}), ensure_ascii=True, sort_keys=True),
        )

        await self._smartblog_progress_checked(
            job_id=str(job_id),
            progress=int(render_progress("tts", 0.0)),
            stage="tts",
            stage_label="Generating voiced audio",
            stage_index=0,
            stage_total=int(total),
        )

        async def _mark_tts_completed() -> None:
            nonlocal completed_count
            async with progress_lock:
                completed_count = int(completed_count + 1)
                await self._smartblog_progress_checked(
                    job_id=str(job_id),
                    progress=int(render_progress("tts", float(completed_count) / float(max(1, total)))),
                    stage="tts",
                    stage_label="Generating voiced audio",
                    stage_index=int(completed_count),
                    stage_total=int(total),
                )

        async def _synthesize_one_tts_chunk(idx: int, chunk: dict[str, Any]) -> dict[str, Any] | None:
            chunk_text = str((chunk or {}).get("text") or "").strip()
            frame_index = int((chunk or {}).get("frame_index") or 0)
            frame_subindex = int((chunk or {}).get("frame_subindex") or 0)
            frame_subtotal = int((chunk or {}).get("frame_subtotal") or 1)
            tts_text = str(self._sanitize_render_tts_text(chunk_text) or "").strip()
            if not tts_text:
                await _mark_tts_completed()
                return None
            if not self._render_has_terminal_punct(tts_text):
                tts_text = str(self._ensure_render_terminal_punct(tts_text) or "").strip()
            if not tts_text:
                await _mark_tts_completed()
                return None
            subtitle_text = str(self._sanitize_render_description_text(tts_text) or tts_text).strip()
            words = int(self._render_tts_word_count(tts_text))
            base_timeout_sec = float(
                self._render_eleven_http_segment_timeout_sec(
                    chars=int(len(tts_text)),
                    words=int(words),
                )
            )
            timeout_sec = float(
                _smartblog_render_tts_timeout_sec(
                    chars=int(len(tts_text)),
                    words=int(words),
                    fallback_sec=float(base_timeout_sec),
                )
            )
            request_timeout = aiohttp.ClientTimeout(total=float(timeout_sec))
            pcm = bytearray()
            alignment: dict[str, Any] | None = None
            normalized_alignment: dict[str, Any] | None = None
            t0 = time.perf_counter()
            logging.warning(
                "SmartBlog render worker TTS segment start: job=%s segment=%d/%d frame=%d sub=%d/%d chars=%d words=%d timeout=%.1fs",
                str(job_id or "-"),
                int(idx + 1),
                int(total),
                int(frame_index + 1),
                int(frame_subindex + 1),
                int(frame_subtotal),
                int(len(tts_text)),
                int(words),
                float(timeout_sec),
            )

            async def _request_tts_once(*, expect_timestamps: bool) -> None:
                nonlocal alignment, normalized_alignment
                req_url, req_headers, req_payload, output_format = self._build_eleven_http_tts_request(
                    text=str(tts_text),
                    api_key=str(api_key),
                    voice_id=str(voice_id),
                    voice_settings=dict(voice_settings),
                    model_id=str(model_id),
                    seed=seed,
                    with_timestamps=bool(expect_timestamps),
                )
                async with session.post(
                    req_url,
                    headers=req_headers,
                    json=req_payload,
                    timeout=request_timeout,
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text(errors="replace")
                        raise RuntimeError(f"Eleven render TTS error: HTTP {resp.status}: {body[:300]}")
                    if bool(expect_timestamps):
                        data = await resp.json(content_type=None)
                        if not isinstance(data, dict):
                            raise RuntimeError("Eleven render TTS with timestamps returned non-object JSON")
                        audio_b64 = str(data.get("audio_base64") or data.get("audio") or "").strip()
                        if not audio_b64:
                            raise RuntimeError("Eleven render TTS with timestamps returned no audio_base64")
                        part = base64.b64decode(audio_b64)
                        if len(part) % 2:
                            part = part[: len(part) - 1]
                        pcm.extend(bytes(part))
                        if isinstance(data.get("alignment"), dict):
                            alignment = dict(data.get("alignment") or {})
                        if isinstance(data.get("normalized_alignment"), dict):
                            normalized_alignment = dict(data.get("normalized_alignment") or {})
                        logging.info(
                            "SmartBlog render worker TTS timestamps ok: job=%s segment=%d output=%s bytes=%d",
                            str(job_id or "-"),
                            int(idx + 1),
                            str(output_format),
                            int(len(pcm)),
                        )
                        return
                    async for part in resp.content.iter_chunked(65536):
                        if part:
                            pcm.extend(bytes(part))
                    logging.info(
                        "SmartBlog render worker TTS raw stream ok: job=%s segment=%d output=%s bytes=%d",
                        str(job_id or "-"),
                        int(idx + 1),
                        str(output_format),
                        int(len(pcm)),
                    )

            async with semaphore:
                try:
                    try:
                        await _request_tts_once(expect_timestamps=True)
                    except Exception as e:
                        logging.warning(
                            "SmartBlog render worker TTS timestamps failed; retrying raw stream: job=%s segment=%d err=%r",
                            str(job_id or "-"),
                            int(idx + 1),
                            e,
                        )
                        pcm.clear()
                        alignment = None
                        normalized_alignment = None
                        await _request_tts_once(expect_timestamps=False)
                    if len(pcm) % 2:
                        del pcm[-1:]
                    if not pcm:
                        raise RuntimeError(f"Eleven render TTS produced no audio for segment {int(idx + 1)}")

                    if bool(_env_flag("SMARTBLOG_RENDER_TTS_FORCED_ALIGNMENT", "1")) and subtitle_text:
                        try:
                            clean_alignment = await self._render_eleven_forced_align_subtitles(
                                session=session,
                                api_key=str(api_key),
                                pcm=bytes(pcm),
                                sample_rate=int(self.sample_rate),
                                text=str(subtitle_text),
                                trace=f"render.{job_id}",
                                segment_index=int(idx),
                            )
                            if isinstance(clean_alignment, dict) and clean_alignment:
                                alignment = dict(clean_alignment)
                                normalized_alignment = dict(clean_alignment)
                        except Exception as e:
                            logging.warning(
                                "SmartBlog render worker TTS forced-alignment failed; using TTS timestamps: job=%s segment=%d err=%s",
                                str(job_id or "-"),
                                int(idx + 1),
                                str(e),
                            )

                    wav_path = os.path.join(str(run_dir), f"audio_segment_{int(idx):03d}_worker_tts_16k.wav")
                    self._write_pcm16le_wav(str(wav_path), bytes(pcm), sample_rate=int(self.sample_rate))
                    duration_sec = float(len(pcm) // 2) / float(max(1, int(self.sample_rate)))
                    entry: dict[str, Any] = {
                        "url": f"file://{urllib.parse.quote(str(wav_path))}",
                        "local_path": str(wav_path),
                        "index": int(idx),
                        "text": str(subtitle_text or tts_text),
                        "nextframe_index": int(frame_index),
                        "tts_subchunk_index": int(frame_subindex),
                        "tts_subchunk_total": int(frame_subtotal),
                        "_smartblog_audio_chunk": True,
                        "_smartblog_worker_tts": True,
                    }
                    if int(nextframe_block_count) > 1 and int(avatar_count) > 1:
                        entry["avatar_index"] = int(frame_index) % int(avatar_count)
                    if isinstance(alignment, dict) and alignment:
                        entry["alignment"] = dict(alignment)
                    if isinstance(normalized_alignment, dict) and normalized_alignment:
                        entry["normalized_alignment"] = dict(normalized_alignment)
                    logging.warning(
                        "SmartBlog render worker TTS segment ready: job=%s segment=%d/%d audio_sec=%.3f align=%d dt=%.3fs",
                        str(job_id or "-"),
                        int(idx + 1),
                        int(total),
                        float(duration_sec),
                        1 if isinstance(entry.get("normalized_alignment"), dict) else 0,
                        float(time.perf_counter() - t0),
                    )
                    return entry
                finally:
                    await _mark_tts_completed()

        tts_tasks = [
            asyncio.create_task(
                _synthesize_one_tts_chunk(int(idx), dict(chunk or {})),
                name=f"smartblog-render-tts-{sanitize_job_id(str(job_id or 'job'))}-{int(idx)}",
            )
            for idx, chunk in enumerate(chunks)
        ]
        raw_entries = await asyncio.gather(*tts_tasks)
        entries = sorted(
            [entry for entry in raw_entries if isinstance(entry, dict) and entry.get("local_path")],
            key=lambda item: int(item.get("index") or 0),
        )
        entries = _smartblog_merge_worker_tts_entries_for_visual_segments(
            list(entries),
            run_dir=str(run_dir),
            job_id=str(job_id),
        )

        if not entries:
            raise RuntimeError("render_video worker-side TTS produced no usable audio chunks")
        await self._smartblog_progress_checked(
            job_id=str(job_id),
            progress=int(render_progress("tts", 1.0)),
            stage="tts",
            stage_label="Generating voiced audio",
            stage_index=int(total),
            stage_total=int(total),
        )
        logging.warning(
            "SmartBlog render worker TTS done: job=%s chunks=%d",
            str(job_id or "-"),
            int(len(entries)),
        )
        return entries

    async def _smartblog_merge_frame_audio_entries_for_visual_segments(
        self,
        entries: list[dict[str, Any]],
        *,
        run_dir: str,
        job_id: str,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            [dict(entry or {}) for entry in list(entries or []) if isinstance(entry, dict)],
            key=lambda item: int(item.get("index") or 0),
        )
        if len(ordered) <= 1:
            if ordered and bool(ordered[0].get("_smartblog_frame_api")):
                item = dict(ordered[0])
                item["_smartblog_frame_visual_segment"] = True
                item.setdefault(
                    "_smartblog_frame_audio_chunks",
                    [
                        {
                            "chunk_pos": int(item.get("_smartblog_frame_audio_index") or 0),
                            "index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "source_audio_index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "offset_sec": 0.0,
                            "duration_sec": 0.0,
                            "text": _smartblog_audio_entry_text(item),
                            "alignment": _smartblog_audio_entry_alignment(item, normalized=False),
                            "normalized_alignment": _smartblog_audio_entry_alignment(item, normalized=True),
                        }
                    ],
                )
                return [item]
            return ordered
        if not any(bool(entry.get("_smartblog_frame_api")) for entry in ordered):
            return ordered

        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_key: tuple[int, tuple[str, str]] | None = None
        for entry in ordered:
            try:
                frame_idx = int(entry.get("_smartblog_frame_index"))
            except Exception:
                frame_idx = int(entry.get("index") or 0)
            key = (int(frame_idx), _smartblog_worker_tts_visual_key(entry))
            if current and key != current_key:
                groups.append(list(current))
                current = []
            current.append(dict(entry))
            current_key = key
        if current:
            groups.append(list(current))
        if len(groups) == len(ordered):
            normalized: list[dict[str, Any]] = []
            for group_idx, group in enumerate(groups):
                item = dict(group[0])
                item["index"] = int(group_idx)
                item["_smartblog_frame_visual_segment"] = True
                item.setdefault(
                    "_smartblog_frame_audio_chunks",
                    [
                        {
                            "chunk_pos": int(item.get("_smartblog_frame_audio_index") or 0),
                            "index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "source_audio_index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "offset_sec": 0.0,
                            "duration_sec": 0.0,
                            "text": _smartblog_audio_entry_text(item),
                            "alignment": _smartblog_audio_entry_alignment(item, normalized=False),
                            "normalized_alignment": _smartblog_audio_entry_alignment(item, normalized=True),
                        }
                    ],
                )
                normalized.append(item)
            logging.warning(
                "SmartBlog render frames normalized: job=%s frames=%d chunks=%d merged=0",
                str(job_id or "-"),
                int(len(normalized)),
                int(len(ordered)),
            )
            return normalized

        merged_entries: list[dict[str, Any]] = []
        for group_idx, group in enumerate(groups):
            if len(group) <= 1:
                item = dict(group[0])
                item["index"] = int(group_idx)
                item["_smartblog_frame_visual_segment"] = True
                item.setdefault(
                    "_smartblog_frame_audio_chunks",
                    [
                        {
                            "chunk_pos": int(item.get("_smartblog_frame_audio_index") or 0),
                            "index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "source_audio_index": int(_smartblog_entry_source_audio_index(item, int(item.get("index") or 0))),
                            "offset_sec": 0.0,
                            "duration_sec": 0.0,
                            "text": _smartblog_audio_entry_text(item),
                            "alignment": _smartblog_audio_entry_alignment(item, normalized=False),
                            "normalized_alignment": _smartblog_audio_entry_alignment(item, normalized=True),
                        }
                    ],
                )
                merged_entries.append(item)
                continue

            pcm_parts: list[np.ndarray] = []
            offsets_sec: list[float] = []
            source_chunks: list[dict[str, Any]] = []
            frame_inserts: list[dict[str, Any]] = []
            seen_insert_keys: set[str] = set()
            sample_rate: int | None = None
            total_samples = 0
            for chunk_pos, entry in enumerate(group):
                local_audio_path = _smartblog_render_audio_entry_local_path(entry)
                wav_path = os.path.join(
                    str(run_dir),
                    f"audio_frame_{int(group_idx):03d}_chunk_{int(chunk_pos):03d}_16k.wav",
                )
                if local_audio_path:
                    await asyncio.to_thread(to_wav_16k_mono, local_audio_path, wav_path)
                else:
                    url_s = str((entry or {}).get("url") or "").strip()
                    if not url_s:
                        logging.warning(
                            "SmartBlog render frame audio merge skipped: job=%s frame=%d reason=missing_url",
                            str(job_id or "-"),
                            int(group_idx),
                        )
                        return ordered
                    audio_in_path = os.path.join(
                        str(run_dir),
                        f"audio_frame_{int(group_idx):03d}_chunk_{int(chunk_pos):03d}.input",
                    )
                    await self._smartblog_download_file(url=url_s, out_path=audio_in_path)
                    await asyncio.to_thread(to_wav_16k_mono, audio_in_path, wav_path)
                pcm, rate = _smartblog_wav_pcm16_mono(str(wav_path))
                if pcm.size <= 0:
                    logging.warning(
                        "SmartBlog render frame audio merge skipped: job=%s frame=%d chunk=%d reason=empty_wav",
                        str(job_id or "-"),
                        int(group_idx),
                        int(chunk_pos),
                    )
                    return ordered
                if sample_rate is None:
                    sample_rate = int(rate)
                if int(rate) != int(sample_rate):
                    logging.warning(
                        "SmartBlog render frame audio merge skipped: job=%s frame=%d chunk=%d reason=sample_rate_mismatch rate=%d expected=%d",
                        str(job_id or "-"),
                        int(group_idx),
                        int(chunk_pos),
                        int(rate),
                        int(sample_rate),
                    )
                    return ordered
                alignment_samples = int(
                    _smartblog_audio_entry_alignment_samples(dict(entry or {}), sample_rate=int(sample_rate))
                )
                if alignment_samples > 0 and int(alignment_samples) != int(pcm.size):
                    raw_size = int(pcm.size)
                    if int(pcm.size) < int(alignment_samples):
                        pcm = np.concatenate(
                            [
                                np.asarray(pcm, dtype=np.int16),
                                np.zeros(int(alignment_samples) - int(pcm.size), dtype=np.int16),
                            ]
                        ).astype(np.int16, copy=False)
                    else:
                        pcm = np.asarray(pcm[: int(alignment_samples)], dtype=np.int16)
                    logging.warning(
                        "SmartBlog render frame audio alignment duration applied: job=%s frame=%d chunk=%d wav=%.3fs alignment=%.3fs",
                        str(job_id or "-"),
                        int(group_idx),
                        int(chunk_pos),
                        float(raw_size) / float(max(1, int(sample_rate))),
                        float(alignment_samples) / float(max(1, int(sample_rate))),
                    )
                offset_sec = float(total_samples) / float(max(1, int(sample_rate)))
                duration_sec = float(pcm.size) / float(max(1, int(sample_rate)))
                offsets_sec.append(float(offset_sec))
                pcm_parts.append(np.asarray(pcm, dtype=np.int16))
                try:
                    source_index = int(entry.get("source_audio_index", entry.get("index") or chunk_pos))
                except Exception:
                    source_index = int(chunk_pos)
                try:
                    frame_audio_index = int(entry.get("_smartblog_frame_audio_index", chunk_pos))
                except Exception:
                    frame_audio_index = int(chunk_pos)
                source_chunks.append(
                    {
                        "chunk_pos": int(frame_audio_index),
                        "index": int(source_index),
                        "source_audio_index": int(source_index),
                        "offset_sec": float(offset_sec),
                        "duration_sec": float(duration_sec),
                        "text": _smartblog_audio_entry_text(entry),
                        "alignment": _smartblog_audio_entry_alignment(entry, normalized=False),
                        "normalized_alignment": _smartblog_audio_entry_alignment(entry, normalized=True),
                    }
                )
                for raw_insert in list(entry.get("_smartblog_frame_inserts") or []):
                    if not isinstance(raw_insert, dict):
                        continue
                    key = json.dumps(
                        {
                            "order": raw_insert.get("_smartblog_insert_order"),
                            "after": raw_insert.get("_smartblog_ltx_insert_after_chunk"),
                            "mode": raw_insert.get("_smartblog_ltx_mode"),
                            "prompt": raw_insert.get("_smartblog_ltx_prompt"),
                            "duration": raw_insert.get("_smartblog_ltx_duration_sec"),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    if key in seen_insert_keys:
                        continue
                    seen_insert_keys.add(key)
                    frame_inserts.append(dict(raw_insert))
                total_samples += int(pcm.size)

            if not pcm_parts or sample_rate is None or int(total_samples) <= 0:
                return ordered
            merged_pcm = np.concatenate(pcm_parts).astype(np.int16, copy=False)
            out_path = os.path.join(str(run_dir), f"audio_frame_{int(group_idx):03d}_merged_16k.wav")
            _smartblog_write_wav_pcm16_mono(str(out_path), merged_pcm, sample_rate=int(sample_rate))
            first = dict(group[0])
            merged: dict[str, Any] = {
                **first,
                "url": f"file://{urllib.parse.quote(str(out_path))}",
                "local_path": str(out_path),
                "index": int(group_idx),
                "text": " ".join(_smartblog_audio_entry_text(entry) for entry in group if _smartblog_audio_entry_text(entry)).strip(),
                "frame_audio_chunks_merged": int(len(group)),
                "frame_audio_source_indices": [int(entry.get("index") or 0) for entry in group],
                "_smartblog_frame_audio_chunks": list(source_chunks),
                "_smartblog_audio_chunk": True,
                "_smartblog_frame_api": True,
                "_smartblog_frame_visual_segment": True,
            }
            if frame_inserts:
                merged["_smartblog_frame_inserts"] = list(frame_inserts)
            alignment = _smartblog_merge_audio_entry_alignment(group, offsets_sec=list(offsets_sec), normalized=False)
            normalized_alignment = _smartblog_merge_audio_entry_alignment(group, offsets_sec=list(offsets_sec), normalized=True)
            if isinstance(alignment, dict) and alignment:
                merged["alignment"] = dict(alignment)
            if isinstance(normalized_alignment, dict) and normalized_alignment:
                merged["normalized_alignment"] = dict(normalized_alignment)
            merged_entries.append(merged)
            logging.warning(
                "SmartBlog render frame audio merged: job=%s frame_segment=%d chunks=%d audio_sec=%.3f key=%s",
                str(job_id or "-"),
                int(group_idx + 1),
                int(len(group)),
                float(merged_pcm.size) / float(max(1, int(sample_rate))),
                "/".join(_smartblog_worker_tts_visual_key(first)),
            )

        logging.warning(
            "SmartBlog render frames normalized: job=%s frames=%d chunks=%d merged=%d",
            str(job_id or "-"),
            int(len(merged_entries)),
            int(len(ordered)),
            int(sum(max(0, len(group) - 1) for group in groups)),
        )
        return merged_entries

    async def _smartblog_merge_timeline_entries_for_visual_segments(
        self,
        entries: list[dict[str, Any]],
        *,
        run_dir: str,
        job_id: str,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            [dict(entry or {}) for entry in list(entries or []) if isinstance(entry, dict)],
            key=lambda item: int(item.get("_smartblog_timeline_order", item.get("index") or 0)),
        )
        if not ordered:
            return []
        if not any(str(entry.get("_smartblog_timeline_kind") or "") == "ltx" for entry in ordered):
            return await self._smartblog_merge_frame_audio_entries_for_visual_segments(
                list(ordered),
                run_dir=str(run_dir),
                job_id=str(job_id),
            )

        normalized: list[dict[str, Any]] = []
        pending_audio: list[dict[str, Any]] = []

        async def flush_audio() -> None:
            nonlocal pending_audio, normalized
            if not pending_audio:
                return
            if any(bool(entry.get("_smartblog_timeline_no_merge")) for entry in pending_audio):
                normalized.extend(dict(entry) for entry in pending_audio)
            else:
                merged = await self._smartblog_merge_frame_audio_entries_for_visual_segments(
                    list(pending_audio),
                    run_dir=str(run_dir),
                    job_id=str(job_id),
                )
                normalized.extend(dict(entry) for entry in merged)
            pending_audio = []

        for entry in ordered:
            kind = str(entry.get("_smartblog_timeline_kind") or "avatar").strip().lower()
            if kind == "ltx":
                await flush_audio()
                normalized.append(dict(entry))
                continue
            pending_audio.append(dict(entry))
        await flush_audio()

        for idx, entry in enumerate(normalized):
            entry["index"] = int(idx)
            entry["_smartblog_timeline_order"] = int(idx)
        logging.warning(
            "SmartBlog render timeline normalized: job=%s input=%d output=%d hunyuan=%d",
            str(job_id or "-"),
            int(len(ordered)),
            int(len(normalized)),
            int(sum(1 for entry in normalized if str(entry.get("_smartblog_timeline_kind") or "") == "ltx")),
        )
        return normalized

    async def _smartblog_render_avatar_liveaudio_one_pass(
        self,
        *,
        claim: dict[str, Any],
        segment_infos: list[dict[str, Any]],
        run_dir: str,
        job_id: str,
        render_job_type: str,
        render_progress: Any,
        render_size: str,
        out_w: int,
        out_h: int,
        audio_sample_rate: int,
        total_target_samples: int,
        total_duration_sec: float,
        sample_steps: int,
        face_restore: float,
        background_restore: float,
        remote_edge_enabled: bool,
        stream_file_enabled: bool,
        upload: dict[str, Any],
        upload_public_url: str,
        watermark_text: str,
        burn_subtitles: bool,
        artifact_job_id: str = "",
    ) -> SmartBlogRenderFinalizePlan:
        artifact_id = str(artifact_job_id or job_id or "avatar_onepass").strip()
        avatar_segments = [
            dict(info or {})
            for info in list(segment_infos or [])
            if str((info or {}).get("kind") or "avatar").strip().lower() == "avatar"
        ]
        if not avatar_segments or len(avatar_segments) != len(segment_infos or []):
            raise RuntimeError("avatar one-pass render requires avatar-only segment infos")
        if not bool(remote_edge_enabled or stream_file_enabled):
            raise RuntimeError("avatar one-pass render requires remote-edge or stream-file output")

        queue_dir = self._prepare_liveaudio_queue_dir(str(run_dir), f"{artifact_id}_render_avatar_onepass")
        background_music_cfg = _smartblog_background_music_config(claim)
        background_music_enabled = bool(background_music_cfg.get("enabled"))
        finalizer_service_url = str(_smartblog_file_upscale_service_url() or "").strip()
        edge_background_finalizer_enabled = bool(
            remote_edge_enabled
            and _smartblog_render_edge_finalizer_background_enabled()
            and str((upload or {}).get("signed_url") or "").strip()
            and str((upload or {}).get("path") or "").strip()
        )
        logging.warning(
            "SmartBlog render avatar one-pass finalizer route: job=%s artifact=%s edge_background=%d remote_edge=%d service_url=%d background_music=%d upload_signed=%d upload_path=%d burn_subtitles=%d watermark_chars=%d",
            str(job_id or "-"),
            str(artifact_id or "-"),
            1 if bool(edge_background_finalizer_enabled) else 0,
            1 if bool(remote_edge_enabled) else 0,
            1 if bool(finalizer_service_url) else 0,
            1 if bool(background_music_enabled) else 0,
            1 if bool(str((upload or {}).get("signed_url") or "").strip()) else 0,
            1 if bool(str((upload or {}).get("path") or "").strip()) else 0,
            1 if bool(burn_subtitles) else 0,
            int(len(str(watermark_text or "").strip())),
        )
        remote_finalizer_subtitle_chunks_json = (
            _smartblog_render_subtitle_chunks_json(list(segment_infos or []))
            if bool(edge_background_finalizer_enabled) and bool(burn_subtitles)
            else ""
        )
        edge_subtitles_enabled = bool(
            _env_flag("SMARTBLOG_RENDER_ONEPASS_LIVEAUDIO_SUBTITLES", "0")
            and not bool(remote_finalizer_subtitle_chunks_json)
        )
        first_avatar_path = str(avatar_segments[0].get("avatar_path") or "").strip()
        if not first_avatar_path:
            raise RuntimeError("avatar one-pass render has empty first avatar path")

        queue_items: list[dict[str, Any]] = []
        legacy_liveaudio_chunk_samples = int(_smartblog_liveaudio_queue_chunk_samples(int(audio_sample_rate or 16000)))
        infer_frames_for_queue = int(max(1, int(LOCKED_INFER_FRAMES)))
        onepass_block_frames = int(_smartblog_render_onepass_block_frames(int(infer_frames_for_queue)))
        # Keep render-video avatar joins on the exact audio/subtitle timeline by
        # default. A hidden ref-start lipsync preroll is still available for
        # experiments through SMARTBLOG_RENDER_AVATAR_REF_START_PREROLL_FRAMES,
        # but using it as the default makes normal avatar-to-avatar joins trim
        # visible video while subtitles keep the correct duration.
        ref_start_preroll_frames_default = 0
        ref_start_preroll_frames = int(
            max(
                0,
                min(
                    int(onepass_block_frames),
                    _safe_int_env(
                        "SMARTBLOG_RENDER_AVATAR_REF_START_PREROLL_FRAMES",
                        int(ref_start_preroll_frames_default),
                    ),
                ),
            )
        )

        for idx, info in enumerate(avatar_segments, start=1):
            audio_wav = str(info.get("audio_wav_path") or "").strip()
            if not audio_wav or not os.path.exists(audio_wav):
                raise RuntimeError(f"avatar one-pass render missing audio chunk {idx}: {audio_wav}")
            pcm, pcm_rate = _smartblog_wav_pcm16_mono(str(audio_wav))
            sample_rate_i = int(pcm_rate or info.get("sample_rate") or audio_sample_rate or 16000)
            target_samples = int(info.get("target_samples") or int(pcm.size) or 0)
            if target_samples <= 0:
                target_samples = int(pcm.size)
            if int(pcm.size) < int(target_samples):
                pcm = np.concatenate(
                    [np.asarray(pcm, dtype=np.int16), np.zeros(int(target_samples) - int(pcm.size), dtype=np.int16)]
                ).astype(np.int16, copy=False)
            elif int(pcm.size) > int(target_samples):
                pcm = np.asarray(pcm[: int(target_samples)], dtype=np.int16)
            else:
                pcm = np.asarray(pcm, dtype=np.int16)

            lipsync_pcm: np.ndarray | None = None
            lipsync_rate = int(sample_rate_i)
            lipsync_audio_wav = str(info.get("lipsync_audio_wav_path") or "").strip()
            if lipsync_audio_wav and os.path.exists(lipsync_audio_wav):
                try:
                    lipsync_pcm_i, lipsync_rate_i = _smartblog_wav_pcm16_mono(str(lipsync_audio_wav))
                    if int(lipsync_rate_i or 0) == int(sample_rate_i):
                        lipsync_pcm = np.asarray(lipsync_pcm_i, dtype=np.int16)
                        lipsync_rate = int(lipsync_rate_i)
                        if int(lipsync_pcm.size) < int(target_samples):
                            lipsync_pcm = np.concatenate(
                                [
                                    lipsync_pcm,
                                    np.zeros(int(target_samples) - int(lipsync_pcm.size), dtype=np.int16),
                                ]
                            ).astype(np.int16, copy=False)
                        elif int(lipsync_pcm.size) > int(target_samples):
                            lipsync_pcm = np.asarray(lipsync_pcm[: int(target_samples)], dtype=np.int16)
                    else:
                        logging.warning(
                            "SmartBlog render avatar one-pass lipsync rate mismatch: job=%s segment=%d rate=%d lipsync_rate=%d",
                            str(job_id or "-"),
                            int(idx),
                            int(sample_rate_i),
                            int(lipsync_rate_i or 0),
                        )
                except Exception as e:
                    logging.warning(
                        "SmartBlog render avatar one-pass lipsync split skipped: job=%s segment=%d path=%s err=%s",
                        str(job_id or "-"),
                        int(idx),
                        os.path.basename(str(lipsync_audio_wav)),
                        str(e),
                    )
            cur_avatar_path = str(info.get("avatar_path") or "").strip()
            prev_avatar_path = ""
            if int(idx) > 1:
                try:
                    prev_avatar_path = str(avatar_segments[int(idx) - 2].get("avatar_path") or "").strip()
                except Exception:
                    prev_avatar_path = ""
            next_avatar_path = ""
            if int(idx) < int(len(avatar_segments)):
                try:
                    next_avatar_path = str(avatar_segments[int(idx)].get("avatar_path") or "").strip()
                except Exception:
                    next_avatar_path = ""
            avatar_ref_changes_prev = bool(prev_avatar_path and cur_avatar_path and prev_avatar_path != cur_avatar_path)
            avatar_ref_changes_next = bool(cur_avatar_path and next_avatar_path and cur_avatar_path != next_avatar_path)
            if bool(avatar_ref_changes_next):
                close_ms = float(_safe_float_env("SMARTBLOG_RENDER_AVATAR_BOUNDARY_LIPSYNC_CLOSE_MS", 100.0))
                close_ms = float(max(0.0, min(500.0, float(close_ms))))
                close_floor = float(_safe_float_env("SMARTBLOG_RENDER_AVATAR_BOUNDARY_LIPSYNC_CLOSE_FLOOR", 0.04))
                close_floor = float(max(0.0, min(1.0, float(close_floor))))
                if float(close_ms) > 0.0:
                    if lipsync_pcm is None:
                        lipsync_pcm = np.asarray(pcm, dtype=np.int16).copy()
                        lipsync_rate = int(sample_rate_i)
                    else:
                        lipsync_pcm = np.asarray(lipsync_pcm, dtype=np.int16).copy()
                    lipsync_pcm = _smartblog_apply_lipsync_tail_close(
                        lipsync_pcm,
                        sample_rate=int(lipsync_rate),
                        duration_ms=float(close_ms),
                        floor=float(close_floor),
                    )
                    logging.info(
                        "SmartBlog render avatar one-pass lipsync boundary close applied: job=%s segment=%d close_ms=%.1f floor=%.3f",
                        str(job_id or "-"),
                        int(idx),
                        float(close_ms),
                        float(close_floor),
                    )
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            visual_prompt = _smartblog_render_entry_prompt(claim, entry)
            negative_prompt = _smartblog_render_entry_negative_prompt(claim, entry)
            boundary_preroll_frames = int(
                max(
                    0,
                    min(
                        int(_smartblog_render_onepass_block_frames(int(infer_frames_for_queue))),
                        _safe_int_env("SMARTBLOG_RENDER_ONEPASS_BOUNDARY_PREROLL_FRAMES", 0),
                    ),
                )
            )
            chunk_ranges = _smartblog_render_onepass_audio_ranges(
                total_samples=int(target_samples),
                sample_rate=int(sample_rate_i),
                fps=int(WORKER_FPS),
                infer_frames=int(infer_frames_for_queue),
                boundary_preroll_frames=int(boundary_preroll_frames),
            )
            for (
                start_sample,
                end_sample,
                source_frame_start,
                source_frame_end,
                conditioning_frames,
                visible_start_frames,
            ) in chunk_ranges:
                if int(end_sample) <= int(start_sample):
                    continue
                source_samples_i = int(end_sample) - int(start_sample)
                source_frames_i = int(max(1, int(source_frame_end) - int(source_frame_start)))
                queue_visible_start_frames = int(visible_start_frames)
                queue_conditioning_frames = int(conditioning_frames)
                lipsync_preroll_frames = 0
                lipsync_preroll_samples = 0
                if (
                    bool(avatar_ref_changes_prev)
                    and int(source_frame_start) == 0
                    and int(ref_start_preroll_frames) > 0
                ):
                    lipsync_preroll_frames = int(min(int(ref_start_preroll_frames), int(source_frames_i)))
                    lipsync_preroll_samples = int(
                        max(
                            1,
                            round(
                                float(lipsync_preroll_frames)
                                * float(max(1, int(sample_rate_i)))
                                / float(max(1, int(WORKER_FPS)))
                            ),
                        )
                    )
                    queue_visible_start_frames = int(queue_visible_start_frames) + int(lipsync_preroll_frames)
                    queue_conditioning_frames = int(
                        max(
                            int(queue_conditioning_frames),
                            int(
                                math.ceil(
                                    float(source_frames_i + int(queue_visible_start_frames))
                                    / float(max(1, int(onepass_block_frames)))
                                )
                                * int(max(1, int(onepass_block_frames)))
                            ),
                        )
                    )
                queue_idx = int(len(queue_items) + 1)
                chunk_wav = os.path.join(str(queue_dir), f"{int(queue_idx):06d}.wav")
                _smartblog_write_wav_pcm16_mono(
                    str(chunk_wav),
                    np.asarray(pcm[int(start_sample) : int(end_sample)], dtype=np.int16),
                    sample_rate=int(sample_rate_i),
                )
                lipsync_chunk_path = ""
                lipsync_source_pcm = lipsync_pcm if lipsync_pcm is not None else (pcm if int(lipsync_preroll_frames) > 0 else None)
                if lipsync_source_pcm is not None:
                    lipsync_chunk_path = os.path.join(str(queue_dir), f"{int(queue_idx):06d}.lipsync.wav")
                    lipsync_slice = np.asarray(
                        lipsync_source_pcm[int(start_sample) : int(end_sample)],
                        dtype=np.int16,
                    )
                    if int(lipsync_preroll_frames) > 0:
                        lead = np.asarray(
                            lipsync_source_pcm[
                                int(start_sample) : int(min(int(end_sample), int(start_sample) + int(lipsync_preroll_samples)))
                            ],
                            dtype=np.int16,
                        )
                        if int(lead.size) < int(lipsync_preroll_samples):
                            pad_value = int(lead[-1]) if int(lead.size) > 0 else 0
                            lead = np.concatenate(
                                [
                                    lead,
                                    np.full(
                                        int(lipsync_preroll_samples) - int(lead.size),
                                        int(pad_value),
                                        dtype=np.int16,
                                    ),
                                ]
                            ).astype(np.int16, copy=False)
                        lipsync_slice = np.concatenate([lead, lipsync_slice]).astype(np.int16, copy=False)
                        logging.info(
                            "SmartBlog render avatar one-pass ref-start lipsync preroll applied: job=%s segment=%d chunk=%d preroll_frames=%d preroll_samples=%d visible_frames=%d visible_start=%d conditioning_frames=%d prev_ref=%s cur_ref=%s",
                            str(job_id or "-"),
                            int(idx),
                            int(queue_idx),
                            int(lipsync_preroll_frames),
                            int(lipsync_preroll_samples),
                            int(source_frames_i),
                            int(queue_visible_start_frames),
                            int(queue_conditioning_frames),
                            os.path.basename(str(prev_avatar_path or "")),
                            os.path.basename(str(cur_avatar_path or "")),
                        )
                    _smartblog_write_wav_pcm16_mono(
                        str(lipsync_chunk_path),
                        np.asarray(lipsync_slice, dtype=np.int16),
                        sample_rate=int(lipsync_rate),
                    )
                subtitle_entry = entry
                if bool(edge_subtitles_enabled):
                    subtitle_entry = _smartblog_slice_audio_entry_for_segment(
                        dict(entry or {}),
                        start_sec=float(start_sample) / float(max(1, int(sample_rate_i))),
                        end_sec=float(end_sample) / float(max(1, int(sample_rate_i))),
                    )
                subtitle_text = _smartblog_audio_entry_text(subtitle_entry) if bool(edge_subtitles_enabled) else None
                alignment = (
                    _smartblog_audio_entry_alignment(subtitle_entry, normalized=False)
                    if bool(edge_subtitles_enabled)
                    else None
                )
                normalized_alignment = (
                    _smartblog_audio_entry_alignment(subtitle_entry, normalized=True)
                    if bool(edge_subtitles_enabled)
                    else None
                )
                queue_items.append(
                    {
                        "queue_idx": int(queue_idx),
                        "source_samples": int(source_samples_i),
                        "source_frames": int(source_frames_i),
                        "conditioning_frames": int(queue_conditioning_frames),
                        "visible_start_frames": int(queue_visible_start_frames),
                        "subtitle_text": subtitle_text,
                        "subtitle_alignment": alignment if isinstance(alignment, dict) else None,
                        "subtitle_normalized_alignment": (
                            normalized_alignment if isinstance(normalized_alignment, dict) else None
                        ),
                        "lipsync_audio_path": str(lipsync_chunk_path or ""),
                        "embedded_visible_start_frames": bool(int(lipsync_preroll_frames) > 0),
                        "avatar_ref_path": str(info.get("avatar_path") or ""),
                        "visual_prompt": str(visual_prompt or ""),
                        "negative_prompt": str(negative_prompt or ""),
                    }
                )

        if not queue_items:
            raise RuntimeError("avatar one-pass render produced no liveaudio queue chunks")

        queue_total_samples = int(sum(int(item.get("source_samples") or 0) for item in queue_items))
        queue_total_frames = int(sum(int(item.get("source_frames") or 0) for item in queue_items))
        queue_segment_preview = ",".join(
            (
                f"{int(item.get('queue_idx') or 0)}:"
                f"{int(item.get('source_samples') or 0)}s/"
                f"{int(item.get('source_frames') or 0)}f/"
                f"{int(item.get('visible_start_frames') or 0)}v/"
                f"{int(item.get('conditioning_frames') or 0)}c"
            )
            for item in queue_items[:16]
        )
        if len(queue_items) > 16:
            queue_segment_preview = f"{queue_segment_preview},..."
        queue_sample_delta = int(queue_total_samples) - int(total_target_samples)
        logging.warning(
            "SmartBlog render avatar one-pass queue coverage: job=%s artifact=%s chunks=%d samples=%d target_samples=%d delta=%d frames=%d preview=%s",
            str(job_id or "-"),
            str(artifact_id or "-"),
            int(len(queue_items)),
            int(queue_total_samples),
            int(total_target_samples),
            int(queue_sample_delta),
            int(queue_total_frames),
            str(queue_segment_preview or "-"),
        )
        if abs(int(queue_sample_delta)) > max(1, int(audio_sample_rate or 16000) // 100):
            raise RuntimeError(
                "avatar one-pass liveaudio queue sample coverage mismatch: "
                f"queue={int(queue_total_samples)} target={int(total_target_samples)} delta={int(queue_sample_delta)}"
            )

        for item in queue_items:
            queue_idx = int(item.get("queue_idx") or 0)
            source_samples_i = int(item.get("source_samples") or 0)
            self._write_liveaudio_chunk_meta(
                str(queue_dir),
                chunk_idx=int(queue_idx),
                kind="speech",
                audible=True,
                source_samples=int(source_samples_i),
                source_frames=int(item.get("source_frames") or 0),
                conditioning_frames=int(item.get("conditioning_frames") or 0),
                visible_start_frames=int(item.get("visible_start_frames") or 0),
                visible_frames=int(item.get("source_frames") or 0),
                turn_done=bool(int(queue_idx) == int(len(queue_items))),
                subtitle_text=item.get("subtitle_text") if bool(edge_subtitles_enabled) else None,
                subtitle_start_samples=0 if bool(edge_subtitles_enabled) else None,
                subtitle_end_samples=int(source_samples_i) if bool(edge_subtitles_enabled) else None,
                subtitle_total_samples=int(source_samples_i) if bool(edge_subtitles_enabled) else None,
                subtitle_alignment=item.get("subtitle_alignment") if bool(edge_subtitles_enabled) else None,
                subtitle_normalized_alignment=(
                    item.get("subtitle_normalized_alignment") if bool(edge_subtitles_enabled) else None
                ),
                subtitle_alignment_base_samples=0 if bool(edge_subtitles_enabled) else None,
                lipsync_audio_path=str(item.get("lipsync_audio_path") or ""),
                embedded_visible_start_frames=bool(item.get("embedded_visible_start_frames")),
                avatar_ref_path=str(item.get("avatar_ref_path") or ""),
                visual_prompt=str(item.get("visual_prompt") or ""),
                negative_prompt=str(item.get("negative_prompt") or ""),
            )

        self._write_liveaudio_done_marker(
            str(queue_dir),
            chunks_total=int(len(queue_items)),
            total_samples=int(total_target_samples),
            speech_end_samples=int(total_target_samples),
            speech_end_sec=float(total_duration_sec),
            sample_rate=int(audio_sample_rate or 16000),
        )

        raw_video_path = os.path.join(str(run_dir), "render_avatar_onepass_raw.mp4")
        remote_live_raw_dir: str | None = None
        remote_progress_path = ""
        if bool(remote_edge_enabled):
            raw_upload_plan = SmartBlogRenderFinalizePlan(
                job_id=f"{artifact_id}_avatar_onepass",
                job_type=str(render_job_type),
                signed_url="",
                upload_path=(
                    f"worker-uploads/render_segments/{sanitize_job_id(job_id or 'render')}/"
                    f"{sanitize_job_id(artifact_id)}_avatar_onepass_raw.mp4"
                ),
                file_path=str(raw_video_path),
                content_type="video/mp4",
                complete_kwargs={},
                run_dir=str(run_dir),
            )
            raw_signed_url, raw_upload_path = await self._smartblog_resolve_upload_target(raw_upload_plan)
            remote_live_raw_dir = os.path.abspath(prepare_live_raw_dir(str(run_dir), f"{artifact_id}_avatar_onepass_remote"))
            remote_progress_path = os.path.join(str(remote_live_raw_dir), "remote_edge_file_progress.json")
            _smartblog_write_render_remote_edge_manifest(
                claim=claim,
                live_raw_dir=str(remote_live_raw_dir),
                job_id=f"{artifact_id}_avatar_onepass",
                width=int(out_w),
                height=int(out_h),
                fps=int(WORKER_FPS),
                sample_rate=int(audio_sample_rate or 16000),
                target_audio_samples=int(total_target_samples),
                target_duration_sec=float(total_duration_sec),
                upload={"signed_url": str(raw_signed_url), "path": str(raw_upload_path)},
                public_url=self._smartblog_public_storage_url(str(raw_upload_path)),
                watermark_text="",
                remote_finalizer=False if bool(edge_background_finalizer_enabled or finalizer_service_url) else None,
                file_output_fps=int(round(float(_smartblog_render_source_fps()))),
            )

        infer_frames = int(LOCKED_INFER_FRAMES)
        queue_source_frames = [int(item.get("source_frames") or 0) for item in queue_items]
        queue_conditioning_frames = [int(item.get("conditioning_frames") or 0) for item in queue_items]
        short_queue_frame_threshold = max(1, int(infer_frames) // 2)
        short_queue_chunks = sum(
            1
            for frames_i in queue_source_frames
            if int(frames_i) > 0 and int(frames_i) < int(short_queue_frame_threshold)
        )
        min_queue_frames = min(queue_source_frames) if queue_source_frames else 0
        max_queue_frames = max(queue_source_frames) if queue_source_frames else 0
        min_conditioning_frames = min(queue_conditioning_frames) if queue_conditioning_frames else 0
        max_conditioning_frames = max(queue_conditioning_frames) if queue_conditioning_frames else 0
        queue_frame_ranges = ",".join(
            (
                f"{int(item.get('source_frames') or 0)}+"
                f"{int(item.get('visible_start_frames') or 0)}/"
                f"{int(item.get('conditioning_frames') or 0)}"
            )
            for item in queue_items[:12]
        )
        if len(queue_items) > 12:
            queue_frame_ranges = f"{queue_frame_ranges},..."
        avatar_ref_changes = 0
        for item_idx in range(1, len(queue_items)):
            prev_ref = str(queue_items[item_idx - 1].get("avatar_ref_path") or "")
            cur_ref = str(queue_items[item_idx].get("avatar_ref_path") or "")
            if prev_ref != cur_ref:
                avatar_ref_changes += 1
        # Liveaudio one-pass emits one visible model block per clip. With
        # SMARTBLOG_WAN_NUM_FRAMES_PER_BLOCK=8 and INFER_FRAMES=64 that is
        # 32 visible frames, not 64. Budget clips against the visible block
        # size or long one-pass renders stop early and the edge pads the tail.
        onepass_visible_block_frames = int(_smartblog_render_onepass_block_frames(int(infer_frames)))
        try:
            clip_pad_sec = float(os.getenv("WORKER_AUDIO_NUM_CLIP_PAD_SEC", "0.25") or 0.25)
        except Exception:
            clip_pad_sec = 0.25
        clip_pad_sec = max(0.0, min(5.0, float(clip_pad_sec)))
        onepass_frames_needed = int(
            math.ceil((float(total_duration_sec) + float(clip_pad_sec)) * float(max(1, int(WORKER_FPS))))
        )
        num_clip = max(
            1,
            int(math.ceil(float(onepass_frames_needed) / float(max(1, int(onepass_visible_block_frames)))))
            + int(max(0, _safe_int_env("SMARTBLOG_RENDER_ONEPASS_NUM_CLIP_MARGIN", 1))),
        )
        raw_target_frames = int(max(1, int(num_clip) * int(onepass_visible_block_frames)))
        logging.warning(
            "SmartBlog render avatar one-pass clip budget: job=%s artifact=%s duration=%.3fs fps=%d infer_frames=%d visible_block_frames=%d frames_needed=%d num_clip=%d raw_target_frames=%d",
            str(job_id or "-"),
            str(artifact_id or "-"),
            float(total_duration_sec),
            int(WORKER_FPS),
            int(infer_frames),
            int(onepass_visible_block_frames),
            int(onepass_frames_needed),
            int(num_clip),
            int(raw_target_frames),
        )
        infer_started_mono = float(time.monotonic())

        def _onepass_progress_provider() -> dict[str, Any]:
            estimated_frac = _smartblog_estimated_render_inference_stage_frac(
                num_clip=int(num_clip),
                started_mono=float(infer_started_mono),
            )
            remote_frac: float | None = None
            model_raw_frac: float | None = None
            try:
                progress_path = str(remote_progress_path or "")
                if progress_path and os.path.exists(progress_path):
                    with open(progress_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        phase = str(obj.get("phase") or "").strip().lower()
                        error_text = str(obj.get("error") or obj.get("message") or "").strip()
                        if phase == "error" or error_text:
                            return {
                                "_fatal_error": "render_video avatar one-pass remote edge failed"
                                + (f": {error_text}" if error_text else ""),
                                "progress": render_progress("inference", 0.0),
                                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
                            }
                        raw = obj.get("progress")
                        if raw is None:
                            raw = obj.get("stage_progress")
                        if raw is not None:
                            remote_frac = float(raw)
            except Exception:
                pass
            if remote_live_raw_dir:
                model_raw_frac = _smartblog_render_model_raw_progress_fraction(
                    _smartblog_read_render_model_raw_progress(str(remote_live_raw_dir)),
                    target_frames=int(raw_target_frames),
                )
            frac = max(
                float(estimated_frac),
                float(remote_frac) if remote_frac is not None else 0.0,
                float(model_raw_frac) if model_raw_frac is not None else 0.0,
            )
            return {
                "progress": render_progress("inference", max(0.0, min(1.0, float(frac)))),
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
            }

        first_entry = avatar_segments[0].get("audio_entry") if isinstance(avatar_segments[0].get("audio_entry"), dict) else {}
        face_restore_values: list[float] = []
        background_restore_values: list[float] = []
        for info in avatar_segments:
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            filters_i = _smartblog_render_entry_filters(claim, entry)
            face_restore_values.append(float(_smartblog_float_filter(filters_i, "face_restore", float(face_restore))))
            background_restore_values.append(
                float(
                    _smartblog_render_background_restore_filter(
                        filters_i,
                        job_id=f"{job_id or '-'}_avatar_onepass_filters",
                    )
                )
            )
        onepass_face_restore = float(max(face_restore_values) if face_restore_values else float(face_restore))
        onepass_background_restore = float(
            max(background_restore_values) if background_restore_values else float(background_restore)
        )
        base_prompt = _smartblog_render_entry_prompt(claim, first_entry)
        base_video_prompt = _smartblog_render_entry_video_prompt(claim, first_entry)
        base_negative_prompt = _smartblog_render_entry_negative_prompt(claim, first_entry)
        logging.warning(
            "SmartBlog render avatar one-pass start: job=%s artifact=%s segments=%d queue_chunks=%d short_queue_chunks=%d short_threshold_frames=%d queue_frames=%d..%d conditioning_frames=%d..%d queue_frame_ranges=%s avatar_ref_changes=%d chunk_strategy=frame_aligned max_chunk_sec=%.3f duration=%.3fs queue=%s size=%s output=%dx%d fps=%.2f sample_steps=%d face_restore=%.2f background_restore=%.2f remote_edge=%d stream_file=%d",
            str(job_id or "-"),
            str(artifact_id or "-"),
            int(len(avatar_segments)),
            int(len(queue_items)),
            int(short_queue_chunks),
            int(short_queue_frame_threshold),
            int(min_queue_frames),
            int(max_queue_frames),
            int(min_conditioning_frames),
            int(max_conditioning_frames),
            str(queue_frame_ranges or "-"),
            int(avatar_ref_changes),
            float(legacy_liveaudio_chunk_samples) / float(max(1, int(audio_sample_rate or 16000))),
            float(total_duration_sec),
            os.path.basename(str(queue_dir)),
            str(render_size),
            int(out_w),
            int(out_h),
            float(_smartblog_render_output_fps()),
            int(sample_steps),
            float(onepass_face_restore),
            float(onepass_background_restore),
            1 if bool(remote_edge_enabled) else 0,
            1 if bool(stream_file_enabled) else 0,
        )

        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("inference", 0.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
        )
        infer_task = asyncio.create_task(
            self._model_client.infer(
                req=InferRequest(
                    prompt=str(base_prompt),
                    video_prompt=str(base_video_prompt),
                    negative_prompt=str(base_negative_prompt),
                    idle_prompt=_smartblog_render_idle_prompt(claim),
                    image_path=str(first_avatar_path),
                    audio_path=str(self._liveaudio_uri_from_dir(str(queue_dir))),
                    lipsync_audio_path="",
                    num_clip=int(num_clip),
                    sample_steps=int(sample_steps),
                    sample_guide_scale=float(getattr(self.args, "sample_guide_scale", 0.0) or 0.0),
                    infer_frames=int(infer_frames),
                    size=str(render_size),
                    base_seed=int(os.getenv("BASE_SEED", "420") or 420),
                    sample_solver=str(getattr(self.args, "sample_solver", "euler") or "euler"),
                    face_restore=float(onepass_face_restore),
                    background_restore=float(onepass_background_restore),
                    job_id=f"{artifact_id}_avatar_onepass",
                    enable_live_hls=False,
                    live_raw_dir=remote_live_raw_dir if bool(remote_edge_enabled) else None,
                    save_live_raw_mp4=False,
                    stream_file_output_path=str(raw_video_path) if bool(stream_file_enabled) else "",
                    stream_file_output_width=int(out_w) if bool(stream_file_enabled) else 0,
                    stream_file_output_height=int(out_h) if bool(stream_file_enabled) else 0,
                    stream_file_output_fps=float(_smartblog_render_source_fps()) if bool(stream_file_enabled) else 0.0,
                    stream_file_trim_duration_sec=float(total_duration_sec) if bool(stream_file_enabled) else 0.0,
                    stream_file_interpolation=(
                        str(os.getenv("SMARTBLOG_RENDER_STREAM_INTERPOLATION", "") or "")
                        if bool(stream_file_enabled)
                        else ""
                    ),
                    tpp_cfg_mode=_smartblog_render_tpp_cfg_mode(),
                )
            )
        )
        infer_resp = await self._smartblog_wait_with_progress(
            task=infer_task,
            job_id=job_id,
            progress=render_progress("inference", 0.0),
            cancel_model_infer_on_stop=True,
            progress_provider=_onepass_progress_provider,
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
        )
        if not bool(infer_resp.ok):
            raise RuntimeError(str(infer_resp.error or "render_video avatar one-pass infer failed"))
        out_path = str(infer_resp.video_path or raw_video_path).strip() or str(raw_video_path)
        if out_path.startswith(SMARTBLOG_REMOTE_EDGE_UPLOADED_PREFIX):
            uploaded_path = out_path[len(SMARTBLOG_REMOTE_EDGE_UPLOADED_PREFIX) :].strip()
            if not uploaded_path:
                raise RuntimeError("render_video avatar one-pass remote edge returned empty upload path")
            if bool(edge_background_finalizer_enabled):
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=render_progress("inference", 1.0),
                    **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
                )
                upload_path = str((upload or {}).get("path") or "").strip()
                video_url = str(upload_public_url or "").strip()
                if not video_url and upload_path:
                    video_url = self._smartblog_public_storage_url(str(upload_path))
                remote_source_fps = int(round(float(_smartblog_render_output_fps())))
                remote_target_fps = int(round(float(_smartblog_render_delivery_fps())))
                remote_upscale_enabled = bool(_smartblog_remote_finalizer_quality_pass_enabled())
                remote_target_w, remote_target_h = _smartblog_render_delivery_dimensions(
                    claim,
                    output_width=int(out_w),
                    output_height=int(out_h),
                )
                remote_finalizer_source_path = str(uploaded_path)
                if bool(_smartblog_render_avatar_musetalk_enabled(claim)):
                    musetalk_url = str(_smartblog_file_musetalk_service_url() or "").strip()
                    if musetalk_url:
                        await self._smartblog_progress_checked(
                            job_id=job_id,
                            progress=render_progress("encode", 0.70),
                            **_smartblog_progress_stage_fields(
                                job_type=render_job_type,
                                stage="encode",
                                stage_label="MuseTalk lip-sync",
                            ),
                        )
                        musetalk_upload_plan = SmartBlogRenderFinalizePlan(
                            job_id=f"{artifact_id}_avatar_musetalk",
                            job_type=str(render_job_type),
                            signed_url="",
                            upload_path=(
                                f"worker-uploads/render_segments/{sanitize_job_id(job_id or 'render')}/"
                                f"{sanitize_job_id(artifact_id)}_avatar_musetalk_raw.mp4"
                            ),
                            file_path="",
                            content_type="video/mp4",
                            complete_kwargs={},
                            run_dir=str(run_dir),
                        )
                        musetalk_signed_url, musetalk_uploaded_path = await self._smartblog_resolve_upload_target(
                            musetalk_upload_plan
                        )
                        musetalk_source_url = await self._smartblog_resolve_download_url(str(remote_finalizer_source_path))
                        musetalk_task = asyncio.create_task(
                            self._smartblog_run_remote_musetalk(
                                source_url=str(musetalk_source_url),
                                signed_url=str(musetalk_signed_url),
                                content_type="video/mp4",
                                source_fps=int(remote_source_fps),
                            ),
                            name=f"smartblog-musetalk-{sanitize_job_id(job_id or artifact_id or 'job')}",
                        )

                        def _musetalk_progress_provider() -> dict[str, Any]:
                            return {
                                "progress": render_progress("encode", 0.78),
                                **_smartblog_progress_stage_fields(
                                    job_type=render_job_type,
                                    stage="encode",
                                    stage_label="MuseTalk lip-sync",
                                ),
                            }

                        await self._smartblog_wait_with_progress(
                            task=musetalk_task,
                            job_id=job_id,
                            progress=render_progress("encode", 0.78),
                            progress_provider=_musetalk_progress_provider,
                            **_smartblog_progress_stage_fields(
                                job_type=render_job_type,
                                stage="encode",
                                stage_label="MuseTalk lip-sync",
                            ),
                            heartbeat_sec=float(_safe_float_env("SMARTBLOG_MUSETALK_PROGRESS_SEC", 3.0)),
                        )
                        remote_finalizer_source_path = str(musetalk_uploaded_path)
                    else:
                        logging.warning(
                            "SmartBlog render avatar MuseTalk requested but service URL is empty: job=%s",
                            str(job_id or "-"),
                        )
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=render_progress("encode", 0.02),
                    **_smartblog_progress_stage_fields(
                        job_type=render_job_type,
                        stage="encode",
                        stage_label="Queued media finalizer",
                    ),
                )
                logging.warning(
                    "SmartBlog render avatar one-pass queued remote finalizer: job=%s source=%s musetalk=%d target=%s subtitles_edge=%d subtitles_finalizer=%d watermark_chars=%d background_music=%d background_music_gain=%.2fdB upscale=%d target=%dx%d fps=%d->%d",
                    str(job_id or "-"),
                    str(remote_finalizer_source_path),
                    1 if str(remote_finalizer_source_path) != str(uploaded_path) else 0,
                    str(upload_path),
                    1 if bool(edge_subtitles_enabled) else 0,
                    1 if bool(remote_finalizer_subtitle_chunks_json) else 0,
                    int(len(str(watermark_text or "").strip())),
                    1 if bool(background_music_enabled) else 0,
                    float(background_music_cfg.get("gain_db") or 0.0),
                    1 if bool(remote_upscale_enabled) else 0,
                    int(remote_target_w),
                    int(remote_target_h),
                    int(remote_source_fps),
                    int(remote_target_fps),
                )
                return SmartBlogRenderFinalizePlan(
                    job_id=job_id,
                    job_type=str((claim.get("job") if isinstance(claim.get("job"), dict) else {}).get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
                    signed_url=str((upload or {}).get("signed_url") or ""),
                    upload_path=str(upload_path),
                    file_path="",
                    content_type="video/mp4",
                    complete_kwargs={
                        "storage_path": str(upload_path),
                        **({"video_url": str(video_url)} if video_url else {}),
                    },
                    run_dir=str(run_dir),
                    remote_finalizer_source_path=str(remote_finalizer_source_path),
                    remote_finalizer_source_fps=int(remote_source_fps),
                    remote_finalizer_target_width=int(remote_target_w),
                    remote_finalizer_target_height=int(remote_target_h),
                    remote_finalizer_target_fps=int(remote_target_fps),
                    remote_finalizer_upscale_enabled=bool(remote_upscale_enabled),
                    remote_finalizer_background_music_url=(
                        str(background_music_cfg.get("audio_url") or "") if bool(background_music_enabled) else ""
                    ),
                    remote_finalizer_background_music_gain_db=float(background_music_cfg.get("gain_db") or 0.0),
                    remote_finalizer_background_music_loop=bool(background_music_cfg.get("loop", True)),
                    remote_finalizer_background_music_duck_voice_db=float(
                        background_music_cfg.get("duck_voice_db") or 0.0
                    ),
                    remote_finalizer_background_music_fade_in_seconds=float(
                        background_music_cfg.get("fade_in_seconds") or 0.0
                    ),
                    remote_finalizer_background_music_fade_out_seconds=float(
                        background_music_cfg.get("fade_out_seconds") or 0.0
                    ),
                    remote_finalizer_background_music_start_offset_seconds=float(
                        background_music_cfg.get("start_offset_seconds") or 0.0
                    ),
                    remote_finalizer_subtitle_chunks_json=str(remote_finalizer_subtitle_chunks_json),
                    remote_finalizer_watermark_text=str(watermark_text or ""),
                )
            await self._smartblog_download_file(url=str(uploaded_path), out_path=str(raw_video_path))
            out_path = str(raw_video_path)
        if not out_path or not os.path.exists(out_path):
            raise RuntimeError("render_video avatar one-pass produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("inference", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
        )

        final_video_path = str(out_path)
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.05),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Preparing final video",
            ),
        )
        subtitle_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_maybe_burn_render_subtitles,
                video_path=str(final_video_path),
                out_path=os.path.join(str(run_dir), "render_final_subtitled.mp4"),
                ass_path=os.path.join(str(run_dir), "render_subtitles.ass"),
                segment_infos=list(segment_infos),
                width=int(out_w),
                height=int(out_h),
                enabled=bool(burn_subtitles),
            ),
            name=f"smartblog-onepass-subtitles-{sanitize_job_id(str(job_id or 'job'))}",
        )
        final_video_path = await self._smartblog_wait_with_progress(
            task=subtitle_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.25),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Burning subtitles",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video avatar one-pass subtitle burn produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.45),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Subtitles ready",
            ),
        )
        final_video_path = await self._smartblog_apply_background_music_if_needed(
            claim=claim,
            video_path=str(final_video_path),
            run_dir=str(run_dir),
            out_path=os.path.join(str(run_dir), "render_final_background_music.mp4"),
            job_id=str(job_id),
            pending_progress=render_progress("encode", 0.62),
            pending_stage_fields=_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Mixing background music",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video avatar one-pass background music mux produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.75),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Final audio ready",
            ),
        )
        watermark_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_maybe_burn_render_watermark,
                video_path=str(final_video_path),
                out_path=os.path.join(str(run_dir), "render_final_watermarked.mp4"),
                watermark_text=str(watermark_text or ""),
                run_dir=str(run_dir),
                width=int(out_w),
                height=int(out_h),
            ),
            name=f"smartblog-onepass-watermark-{sanitize_job_id(str(job_id or 'job'))}",
        )
        final_video_path = await self._smartblog_wait_with_progress(
            task=watermark_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.88),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Burning watermark",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video avatar one-pass watermark burn produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="encode"),
        )
        upload_path = str((upload or {}).get("path") or "").strip()
        video_url = str(upload_public_url or "").strip()
        if not video_url and upload_path:
            video_url = self._smartblog_public_storage_url(str(upload_path))
        complete_kwargs = {"video_url": str(video_url)} if video_url else {}
        return SmartBlogRenderFinalizePlan(
            job_id=job_id,
            job_type=str((claim.get("job") if isinstance(claim.get("job"), dict) else {}).get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
            signed_url=str((upload or {}).get("signed_url") or ""),
            upload_path=str(upload_path),
            file_path=str(final_video_path),
            content_type="video/mp4",
            complete_kwargs=complete_kwargs,
            run_dir=str(run_dir),
        )

    async def _smartblog_render_segmented_video_job(
        self,
        claim: dict[str, Any],
        *,
        audio_entries: list[dict[str, Any]],
        avatar_urls: list[str],
    ) -> SmartBlogRenderFinalizePlan:
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        upload = claim.get("upload") if isinstance(claim.get("upload"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        run_dir = self._smartblog_job_run_dir(job_id)
        final_video_path = os.path.join(run_dir, "render_final.mp4")
        watermark_text = _smartblog_watermark_text(claim)
        render_job_type = str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO).strip().lower()
        if render_job_type not in set(smartblog_render_job_types()):
            render_job_type = SMARTBLOG_JOB_TYPE_RENDER_VIDEO
        render_progress = lambda stage, frac=1.0: _smartblog_stage_progress_total(
            job_type=render_job_type,
            stage=stage,
            stage_progress=frac,
        )
        remote_edge_enabled = bool(_smartblog_remote_edge_render_enabled())
        stream_file_enabled = bool(_smartblog_stream_file_render_enabled())
        if not remote_edge_enabled and not stream_file_enabled:
            raise RuntimeError("segmented render_video requires remote-edge file output or local stream-file rendering")
        if not audio_entries:
            raise RuntimeError("render_video segmented mode requires audio segments")
        if not avatar_urls:
            raise RuntimeError("render_video segmented mode requires avatar images")

        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=max(1, render_progress("prepare", 0.2)),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="prepare"),
        )

        if any("_smartblog_timeline_kind" in entry for entry in list(audio_entries or []) if isinstance(entry, dict)):
            audio_entries = await self._smartblog_merge_timeline_entries_for_visual_segments(
                list(audio_entries),
                run_dir=str(run_dir),
                job_id=str(job_id),
            )
        else:
            audio_entries = await self._smartblog_merge_frame_audio_entries_for_visual_segments(
                list(audio_entries),
                run_dir=str(run_dir),
                job_id=str(job_id),
            )
        if not audio_entries:
            raise RuntimeError("render_video segmented mode requires audio segments")
        single_avatar_one_pass = bool(
            _smartblog_render_single_avatar_one_pass_enabled()
            and _smartblog_audio_entries_same_avatar_without_timeline_breaks(list(audio_entries))
        )
        if bool(single_avatar_one_pass):
            logging.warning(
                "SmartBlog render single-avatar one-pass enabled: job=%s entries=%d key=%s",
                str(job_id or "-"),
                int(len(audio_entries)),
                "/".join(_smartblog_worker_tts_visual_key(dict(audio_entries[0] or {}))),
            )

        avatar_cache: dict[str, str] = {}
        avatar_hash_cache: dict[str, str] = {}
        avatar_paths: list[str] = []
        for idx, avatar_url in enumerate(list(avatar_urls)):
            url_s = str(avatar_url or "").strip()
            cache_key = _smartblog_canonical_media_url(url_s)
            if not url_s or cache_key in avatar_cache:
                if url_s and url_s not in avatar_cache and cache_key in avatar_cache:
                    avatar_cache[url_s] = str(avatar_cache[cache_key])
                continue
            path = os.path.join(run_dir, f"avatar_{int(idx):03d}.png")
            await self._smartblog_download_file(url=url_s, out_path=path)
            try:
                digest = _smartblog_file_sha256(str(path))
                canonical_path = avatar_hash_cache.get(str(digest))
                if canonical_path:
                    avatar_cache[cache_key] = str(canonical_path)
                    avatar_cache[url_s] = str(canonical_path)
                    try:
                        os.remove(str(path))
                    except Exception:
                        pass
                    continue
                avatar_hash_cache[str(digest)] = str(path)
            except Exception:
                pass
            avatar_cache[cache_key] = str(path)
            avatar_cache[url_s] = str(path)
            avatar_paths.append(str(path))
        if not avatar_paths:
            raise RuntimeError("render_video segmented mode downloaded no avatar images")

        first_img = cv2.imread(str(avatar_paths[0]), cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f"cv2.imread failed for avatar: {avatar_paths[0]}")
        src_h, src_w = int(first_img.shape[0]), int(first_img.shape[1])
        orientation = _smartblog_orientation_hint(claim)
        if orientation not in {"portrait", "landscape"}:
            orientation = "landscape" if int(src_w) >= int(src_h) else "portrait"
        render_size, out_w, out_h = _smartblog_render_profile(orientation=orientation)
        render_size = _validate_smartblog_render_size(render_size)
        try:
            target_h_s, target_w_s = str(render_size).split("*", 1)
            target_h = int(target_h_s)
            target_w = int(target_w_s)
        except Exception as e:
            raise RuntimeError(f"invalid render size {render_size!r}") from e
        remote_finalizer_service_url = str(_smartblog_file_upscale_service_url() or "").strip()
        remote_finalizer_enabled = bool(
            _smartblog_render_edge_finalizer_background_enabled()
            and bool(remote_finalizer_service_url)
        )
        timeline_w = int(target_w) if bool(remote_finalizer_enabled) else int(out_w)
        timeline_h = int(target_h) if bool(remote_finalizer_enabled) else int(out_h)
        timeline_w = max(2, int(timeline_w) - (int(timeline_w) % 2))
        timeline_h = max(2, int(timeline_h) - (int(timeline_h) % 2))

        segment_infos: list[dict[str, Any]] = []
        total_target_samples = 0
        audio_sample_rate = 16000
        for idx, entry in enumerate(list(audio_entries)):
            if str((entry or {}).get("_smartblog_timeline_kind") or "avatar").strip().lower() == "ltx":
                duration_sec = float(
                    max(
                        0.1,
                        float(
                            (entry or {}).get("_smartblog_ltx_duration_sec")
                            or _smartblog_ltx_duration_seconds(dict((entry or {}).get("_smartblog_ltx_config") or {}))
                        ),
                    )
                )
                transition_code = int(max(0, min(100, int((entry or {}).get("_smartblog_transition_code") or 0))))
                segment_infos.append(
                    {
                        "index": int(idx),
                        "kind": "ltx",
                        "target_samples": int(round(float(duration_sec) * float(audio_sample_rate))),
                        "target_duration_sec": float(duration_sec),
                        "timeline_duration_sec": float(duration_sec),
                        "duration_sec": float(duration_sec),
                        "sample_rate": int(audio_sample_rate),
                        "audio_entry": dict(entry or {}),
                        "avatar_transition_code": int(transition_code),
                        "avatar_transition_in": bool(transition_code > 0),
                        "avatar_transition_out": False,
                    }
                )
                total_target_samples += int(round(float(duration_sec) * float(audio_sample_rate)))
                continue
            url_s = str((entry or {}).get("url") or "").strip()
            local_audio_path = _smartblog_render_audio_entry_local_path(entry or {})
            if not url_s and not local_audio_path:
                continue
            audio_in_path = os.path.join(run_dir, f"audio_segment_{int(idx):03d}.input")
            audio_wav_path = os.path.join(run_dir, f"audio_segment_{int(idx):03d}_16k.wav")
            if local_audio_path:
                await asyncio.to_thread(to_wav_16k_mono, local_audio_path, audio_wav_path)
            else:
                await self._smartblog_download_file(url=url_s, out_path=audio_in_path)
                await asyncio.to_thread(to_wav_16k_mono, audio_in_path, audio_wav_path)
            segment_samples, segment_rate = wav_sample_count(audio_wav_path)
            if int(segment_rate) <= 0:
                segment_rate = 16000
            audio_sample_rate = int(segment_rate)
            segment_duration = (
                float(segment_samples) / float(segment_rate)
                if int(segment_samples) > 0
                else float(wav_duration_seconds(audio_wav_path))
            )
            if segment_duration <= 0.0:
                raise RuntimeError(f"render_video audio segment {idx + 1} is empty")
            target_samples = int(max(1, int(segment_samples or round(segment_duration * segment_rate))))
            alignment_target_samples = int(
                _smartblog_audio_entry_alignment_samples(dict(entry or {}), sample_rate=int(segment_rate))
            )
            if alignment_target_samples > 0 and int(alignment_target_samples) != int(target_samples):
                logging.warning(
                    "SmartBlog render audio alignment duration applied: job=%s segment=%d wav=%.3fs alignment=%.3fs",
                    str(job_id or "-"),
                    int(idx + 1),
                    float(target_samples) / float(max(1, int(segment_rate))),
                    float(alignment_target_samples) / float(max(1, int(segment_rate))),
                )
                target_samples = int(alignment_target_samples)
            if _env_flag("SMARTBLOG_RENDER_TRIM_TRAILING_SILENCE", "1") and int(alignment_target_samples) <= 0:
                audible_samples = int(
                    wav_audible_sample_count(
                        audio_wav_path,
                        silence_db=float(_safe_float_env("SMARTBLOG_RENDER_TAIL_SILENCE_DB", -50.0)),
                        tail_keep_sec=float(_safe_float_env("SMARTBLOG_RENDER_TAIL_KEEP_SEC", 0.35)),
                    )
                )
                if 0 < int(audible_samples) < int(target_samples):
                    trimmed_samples = int(target_samples) - int(audible_samples)
                    if int(trimmed_samples) > int(float(segment_rate) * 0.05):
                        logging.warning(
                            "SmartBlog render audio segment tail trimmed: job=%s segment=%d removed=%.3fs target=%.3fs kept=%.3fs",
                            str(job_id or "-"),
                            int(idx + 1),
                            float(trimmed_samples) / float(max(1, int(segment_rate))),
                            float(target_samples) / float(max(1, int(segment_rate))),
                            float(audible_samples) / float(max(1, int(segment_rate))),
                        )
                    target_samples = int(audible_samples)

            avatar_url_override = _smartblog_render_audio_entry_avatar_url(entry or {})
            if avatar_url_override:
                avatar_cache_key = _smartblog_canonical_media_url(str(avatar_url_override or ""))
                avatar_path = avatar_cache.get(avatar_cache_key) or avatar_cache.get(avatar_url_override)
                if not avatar_path:
                    avatar_path = os.path.join(run_dir, f"avatar_segment_{int(idx):03d}.png")
                    await self._smartblog_download_file(url=avatar_url_override, out_path=avatar_path)
                    try:
                        digest = _smartblog_file_sha256(str(avatar_path))
                        canonical_path = avatar_hash_cache.get(str(digest))
                        if canonical_path:
                            try:
                                os.remove(str(avatar_path))
                            except Exception:
                                pass
                            avatar_path = str(canonical_path)
                        else:
                            avatar_hash_cache[str(digest)] = str(avatar_path)
                    except Exception:
                        pass
                    avatar_cache[avatar_cache_key] = str(avatar_path)
                    avatar_cache[avatar_url_override] = str(avatar_path)
            else:
                avatar_idx = _smartblog_render_audio_entry_avatar_index(entry or {})
                if avatar_idx is None:
                    avatar_idx = 0
                avatar_path = avatar_paths[int(avatar_idx) % int(len(avatar_paths))]

            entry_one_pass = bool(
                _smartblog_render_single_avatar_one_pass_enabled()
                and bool((entry or {}).get("_smartblog_frame_visual_segment"))
                and not bool((entry or {}).get("_smartblog_timeline_no_merge"))
            )
            segment_max_sec = 0.0 if bool(single_avatar_one_pass or entry_one_pass) else float(_smartblog_render_audio_segment_max_sec())
            max_segment_samples = (
                int(max(1, round(float(segment_max_sec) * float(max(1, int(segment_rate))))))
                if float(segment_max_sec) > 0.0
                else 0
            )
            source_parts: list[tuple[str, int, dict[str, Any]]] = []
            if int(max_segment_samples) > 0 and int(target_samples) > int(max_segment_samples):
                pcm_full, pcm_rate = _smartblog_wav_pcm16_mono(str(audio_wav_path))
                if int(pcm_rate) != int(segment_rate):
                    segment_rate = int(pcm_rate)
                    audio_sample_rate = int(segment_rate)
                    max_segment_samples = int(max(1, round(float(segment_max_sec) * float(max(1, int(segment_rate))))))
                if int(pcm_full.size) < int(target_samples):
                    pcm_trimmed = np.concatenate(
                        [
                            np.asarray(pcm_full, dtype=np.int16),
                            np.zeros(int(target_samples) - int(pcm_full.size), dtype=np.int16),
                        ]
                    ).astype(np.int16, copy=False)
                else:
                    pcm_trimmed = np.asarray(pcm_full[: int(target_samples)], dtype=np.int16)
                part_count = int(math.ceil(float(max(1, int(pcm_trimmed.size))) / float(max(1, int(max_segment_samples)))))
                for part_idx, start_sample in enumerate(range(0, int(pcm_trimmed.size), int(max_segment_samples))):
                    end_sample = int(min(int(pcm_trimmed.size), int(start_sample) + int(max_segment_samples)))
                    part_pcm = np.asarray(pcm_trimmed[int(start_sample) : int(end_sample)], dtype=np.int16)
                    if part_pcm.size <= 0:
                        continue
                    part_path = os.path.join(run_dir, f"audio_segment_{int(idx):03d}_part_{int(part_idx):03d}_16k.wav")
                    _smartblog_write_wav_pcm16_mono(str(part_path), part_pcm, sample_rate=int(segment_rate))
                    part_entry = _smartblog_slice_audio_entry_for_segment(
                        dict(entry or {}),
                        start_sec=float(start_sample) / float(max(1, int(segment_rate))),
                        end_sec=float(end_sample) / float(max(1, int(segment_rate))),
                    )
                    part_entry["_smartblog_audio_split_index"] = int(part_idx)
                    part_entry["_smartblog_audio_split_total"] = int(part_count)
                    part_entry["_smartblog_audio_split_source_index"] = int((entry or {}).get("index") or idx)
                    source_parts.append((str(part_path), int(part_pcm.size), part_entry))
                logging.warning(
                    "SmartBlog render audio entry split: job=%s entry=%d duration=%.3fs parts=%d max_sec=%.3f",
                    str(job_id),
                    int(idx),
                    float(target_samples) / float(max(1, int(segment_rate))),
                    int(len(source_parts)),
                    float(segment_max_sec),
                )
            if not source_parts:
                source_parts.append((str(audio_wav_path), int(target_samples), dict(entry or {})))

            base_transition_code = int(max(0, min(100, int((entry or {}).get("_smartblog_transition_code") or 0))))
            for part_idx, (part_wav_path, part_samples, part_entry) in enumerate(source_parts):
                segment_infos.append(
                    {
                        "index": int(len(segment_infos)),
                        "audio_url": str(url_s),
                        "kind": "avatar",
                        "audio_wav_path": str(part_wav_path),
                        "avatar_path": str(avatar_path),
                        "target_samples": int(part_samples),
                        "target_duration_sec": float(part_samples) / float(max(1, int(segment_rate))),
                        "duration_sec": float(part_samples) / float(max(1, int(segment_rate))),
                        "sample_rate": int(segment_rate),
                        "audio_entry": dict(part_entry or {}),
                        "avatar_transition_code": int(base_transition_code if int(part_idx) == 0 else 0),
                    }
                )
                total_target_samples += int(part_samples)

        if not segment_infos:
            raise RuntimeError("render_video segmented mode produced no usable audio segments")
        for pos, info in enumerate(segment_infos):
            transition_code = int(max(0, min(100, int(info.get("avatar_transition_code") or 0))))
            info["avatar_transition_code"] = int(transition_code)
            info["avatar_transition_in"] = bool(transition_code > 0)
            info["avatar_transition_out"] = False
        avatar_runs: list[list[dict[str, Any]]] = []
        current_run: list[dict[str, Any]] = []
        for info in segment_infos:
            if str(info.get("kind") or "avatar").strip().lower() == "ltx":
                if current_run:
                    avatar_runs.append(list(current_run))
                    current_run = []
                continue
            current_run.append(info)
        if current_run:
            avatar_runs.append(list(current_run))
        for avatar_run in avatar_runs:
            _smartblog_prepare_audio_chunk_boundaries(segment_infos=avatar_run, run_dir=str(run_dir))
            _smartblog_prepare_avatar_lipsync_audio_segments(
                segment_infos=avatar_run,
                run_dir=str(run_dir),
                job_id=str(job_id),
            )
        total_target_samples = int(sum(int(info.get("target_samples") or 0) for info in segment_infos))
        audio_sample_rate = int(segment_infos[0].get("sample_rate") or audio_sample_rate or 16000)
        total_duration_sec = float(total_target_samples) / float(max(1, int(audio_sample_rate)))

        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("prepare", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="prepare"),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("tts", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="tts"),
        )
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("face_detect", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="face_detect"),
        )

        infer_frames = int(LOCKED_INFER_FRAMES)
        sample_steps = int(_smartblog_render_sample_steps(claim))
        filters = _smartblog_filters(claim)
        face_restore = _smartblog_float_filter(filters, "face_restore", 0.5)
        background_restore = _smartblog_render_background_restore_filter(filters, job_id=job_id)
        upload_path_claim = str(upload.get("path") or "").strip()
        upload_public_url = self._smartblog_public_storage_url(upload_path_claim) if upload_path_claim else ""
        burn_subtitles = bool(_smartblog_render_burn_in_subtitles_enabled(claim))
        logging.warning(
            "SmartBlog render segmented config: job=%s segments=%d avatars=%d orientation=%s render_size=%s timeline_work=%dx%d delivery=%dx%d output_fps=%.2f sample_steps=%d face_restore=%.2f background_restore=%.2f subtitles=%d watermark_chars=%d duration=%.3fs remote_edge=%d stream_file=%d remote_finalizer=%d",
            str(job_id),
            int(len(segment_infos)),
            int(len(avatar_paths)),
            str(orientation),
            str(render_size),
            int(timeline_w),
            int(timeline_h),
            int(out_w),
            int(out_h),
            float(_smartblog_render_output_fps()),
            int(sample_steps),
            float(face_restore),
            float(background_restore),
            1 if bool(burn_subtitles) else 0,
            int(len(str(watermark_text or "").strip())),
            float(total_duration_sec),
            1 if bool(remote_edge_enabled) else 0,
            1 if bool(stream_file_enabled) else 0,
            1 if bool(remote_finalizer_enabled) else 0,
        )

        segment_video_paths: list[str] = []
        segment_continuity_paths: list[str] = []
        segment_postprocess_tasks: list[asyncio.Future] = []
        has_hunyuan_segments = any(str(info.get("kind") or "avatar").strip().lower() == "ltx" for info in segment_infos)
        has_frame_inserts = any(
            str(info.get("kind") or "avatar").strip().lower() != "ltx"
            and bool(_smartblog_frame_insert_entries(dict((info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}) or {})))
            for info in segment_infos
        )
        force_final_subtitles = bool(burn_subtitles and (has_frame_inserts or has_hunyuan_segments))
        if (
            not bool(has_hunyuan_segments)
            and not bool(has_frame_inserts)
            and bool(_smartblog_render_avatar_liveaudio_one_pass_enabled())
        ):
            return await self._smartblog_render_avatar_liveaudio_one_pass(
                claim=claim,
                segment_infos=list(segment_infos),
                run_dir=str(run_dir),
                job_id=str(job_id),
                render_job_type=str(render_job_type),
                render_progress=render_progress,
                render_size=str(render_size),
                out_w=int(out_w),
                out_h=int(out_h),
                audio_sample_rate=int(audio_sample_rate),
                total_target_samples=int(total_target_samples),
                total_duration_sec=float(total_duration_sec),
                sample_steps=int(sample_steps),
                face_restore=float(face_restore),
                background_restore=float(background_restore),
                remote_edge_enabled=bool(remote_edge_enabled),
                stream_file_enabled=bool(stream_file_enabled),
                upload=dict(upload or {}),
                upload_public_url=str(upload_public_url or ""),
                watermark_text=str(watermark_text or ""),
                burn_subtitles=bool(burn_subtitles),
            )
        segment_postprocess_background = (
            bool(_env_flag("SMARTBLOG_RENDER_SEGMENT_POSTPROCESS_BACKGROUND", "1"))
            and not bool(has_hunyuan_segments)
            and not bool(has_frame_inserts)
        )
        segment_count = int(len(segment_infos))

        def _segment_frame_index(info: dict[str, Any] | None) -> int | None:
            if not isinstance(info, dict):
                return None
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            for key in ("_smartblog_frame_index", "frame_index", "frameIndex"):
                try:
                    value = int((entry or {}).get(key))
                    if value >= 0:
                        return int(value)
                except Exception:
                    pass
            return None

        def _segment_has_frame_inserts(info: dict[str, Any] | None) -> bool:
            if not isinstance(info, dict):
                return False
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            return bool(_smartblog_frame_insert_entries(dict(entry or {})))

        def _ltx_entry_is_timeline_insert(info: dict[str, Any] | None) -> bool:
            if not isinstance(info, dict):
                return False
            if str(info.get("kind") or "avatar").strip().lower() != "ltx":
                return False
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            return isinstance(entry, dict) and entry.get("_smartblog_ltx_insert_after_chunk") is not None

        def _avatar_segment_audio_ids(info: dict[str, Any] | None) -> set[int]:
            ids: set[int] = set()
            if not isinstance(info, dict):
                return ids
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            for key in (
                "_smartblog_frame_audio_index",
                "frame_audio_index",
                "frameAudioIndex",
                "source_audio_index",
                "sourceAudioIndex",
                "index",
            ):
                try:
                    ids.add(int((entry or {}).get(key)))
                except Exception:
                    pass
            for key in ("frame_audio_source_indices", "worker_tts_source_indices"):
                raw = (entry or {}).get(key)
                if isinstance(raw, (list, tuple)):
                    for item in raw:
                        try:
                            ids.add(int(item))
                        except Exception:
                            pass
            for chunk in _smartblog_avatar_frame_audio_chunks(dict(info or {})):
                for key in ("chunk_pos", "index", "source_audio_index"):
                    try:
                        ids.add(int((chunk or {}).get(key)))
                    except Exception:
                        pass
            return ids

        attached_ltx_segment_indices: set[int] = set()
        avatar_indices_by_frame: dict[int, list[int]] = {}
        for info_idx, info in enumerate(segment_infos):
            if str((info or {}).get("kind") or "avatar").strip().lower() != "avatar":
                continue
            frame_idx = _segment_frame_index(info)
            if frame_idx is None:
                continue
            avatar_indices_by_frame.setdefault(int(frame_idx), []).append(int(info_idx))

        for info_idx, info in enumerate(segment_infos):
            if not _ltx_entry_is_timeline_insert(info):
                continue
            entry = dict(info.get("audio_entry") or {})
            frame_idx = _segment_frame_index(info)
            candidates = list(avatar_indices_by_frame.get(int(frame_idx), [])) if frame_idx is not None else []
            if not candidates:
                candidates = [
                    int(pos)
                    for pos, candidate in enumerate(segment_infos)
                    if int(pos) < int(info_idx)
                    and str((candidate or {}).get("kind") or "avatar").strip().lower() == "avatar"
                ]
            if not candidates:
                continue
            try:
                after_chunk = int(entry.get("_smartblog_ltx_insert_after_chunk"))
            except Exception:
                after_chunk = -1
            host_idx = int(candidates[0])
            if int(after_chunk) >= 0:
                exact = [
                    int(pos)
                    for pos in candidates
                    if int(after_chunk) in _avatar_segment_audio_ids(segment_infos[int(pos)])
                ]
                if exact:
                    host_idx = int(exact[-1])
                else:
                    before = [int(pos) for pos in candidates if int(pos) < int(info_idx)]
                    if before:
                        host_idx = int(before[-1])
            host_info = segment_infos[int(host_idx)]
            host_entry = host_info.get("audio_entry") if isinstance(host_info.get("audio_entry"), dict) else {}
            host_inserts = [
                dict(item or {})
                for item in list((host_entry or {}).get("_smartblog_frame_inserts") or [])
                if isinstance(item, dict)
            ]
            insert_entry = dict(entry or {})
            insert_entry["_smartblog_ltx_mode"] = _smartblog_insert_mode(insert_entry)
            insert_entry.setdefault("_smartblog_insert_order", int(len(host_inserts)))
            host_inserts.append(dict(insert_entry))
            host_entry["_smartblog_frame_inserts"] = list(host_inserts)
            host_info["audio_entry"] = dict(host_entry)
            segment_infos[int(host_idx)] = dict(host_info)
            info["timeline_duration_sec"] = 0.0
            info["_smartblog_attached_to_avatar_segment"] = int(host_idx)
            segment_infos[int(info_idx)] = dict(info)
            attached_ltx_segment_indices.add(int(info_idx))
            logging.warning(
                "SmartBlog render layered timeline insert attached: job=%s ltx_segment=%d host_segment=%d frame=%s after_chunk=%d mode=%s duration=%.3fs",
                str(job_id),
                int(info_idx),
                int(host_idx),
                str(frame_idx if frame_idx is not None else "-"),
                int(after_chunk),
                str(insert_entry.get("_smartblog_ltx_mode") or "cut"),
                float(_smartblog_insert_duration_sec(insert_entry)),
            )

        def _can_group_avatar_segment(info: dict[str, Any] | None) -> bool:
            if not isinstance(info, dict):
                return False
            if str(info.get("kind") or "avatar").strip().lower() != "avatar":
                return False
            return True

        render_units: list[dict[str, Any]] = []
        mixed_avatar_onepass = bool(
            _env_flag("SMARTBLOG_RENDER_MIXED_AVATAR_RUN_ONE_PASS", "1")
            and bool(_smartblog_render_avatar_liveaudio_one_pass_enabled())
            and bool(remote_edge_enabled or stream_file_enabled)
        )
        unit_i = 0
        while unit_i < int(len(segment_infos)):
            if int(unit_i) in attached_ltx_segment_indices:
                unit_i += 1
                continue
            cur_info = dict(segment_infos[int(unit_i)] or {})
            if bool(mixed_avatar_onepass) and _can_group_avatar_segment(cur_info):
                run_start = int(unit_i)
                run_infos: list[dict[str, Any]] = []
                run_end = int(unit_i)
                while unit_i < int(len(segment_infos)):
                    if int(unit_i) in attached_ltx_segment_indices:
                        unit_i += 1
                        continue
                    if not _can_group_avatar_segment(segment_infos[int(unit_i)]):
                        break
                    run_infos.append(dict(segment_infos[int(unit_i)] or {}))
                    run_end = int(unit_i)
                    unit_i += 1
                if len(run_infos) > 1:
                    render_units.append(
                        {
                            "kind": "avatar_run",
                            "start": int(run_start),
                            "end": int(run_end),
                            "infos": list(run_infos),
                        }
                    )
                    continue
                cur_info = dict(run_infos[0] if run_infos else cur_info)
                render_units.append({"kind": "single", "index": int(run_start), "info": cur_info})
                continue
            render_units.append({"kind": "single", "index": int(unit_i), "info": cur_info})
            unit_i += 1

        avatar_run_units = sum(1 for unit in render_units if str(unit.get("kind") or "") == "avatar_run")
        if int(avatar_run_units) > 0:
            logging.warning(
                "SmartBlog render mixed avatar units: job=%s segments=%d units=%d avatar_runs=%d",
                str(job_id),
                int(len(segment_infos)),
                int(len(render_units)),
                int(avatar_run_units),
            )

        render_unit_count = int(max(1, len(render_units)))

        def _unit_inference_render_progress(
            *,
            unit_pos: int,
            phase_start: float = 0.0,
            phase_end: float = 1.0,
            input_start: float = 0.0,
            input_end: float = 1.0,
        ):
            unit_start = float(max(0.0, min(1.0, float(unit_pos) / float(render_unit_count))))
            unit_end = float(max(unit_start, min(1.0, float(unit_pos + 1) / float(render_unit_count))))
            phase_start_f = float(max(0.0, min(1.0, float(phase_start))))
            phase_end_f = float(max(phase_start_f, min(1.0, float(phase_end))))
            input_start_f = float(max(0.0, min(1.0, float(input_start))))
            input_end_f = float(max(input_start_f + 1e-6, min(1.0, float(input_end))))

            def mapped_progress(stage: str, frac: float = 1.0) -> int:
                stage_s = str(stage or "").strip().lower()
                frac_f = float(max(0.0, min(1.0, float(frac))))
                frac_f = float(max(0.0, min(1.0, (frac_f - input_start_f) / (input_end_f - input_start_f))))
                if stage_s in {"prepare", "tts", "face_detect"}:
                    nested = 0.02 * frac_f
                elif stage_s == "inference":
                    nested = 0.02 + 0.78 * frac_f
                elif stage_s == "encode":
                    nested = 0.82 + 0.16 * frac_f
                elif stage_s == "upload":
                    nested = 0.98 + 0.02 * frac_f
                else:
                    nested = frac_f
                phase_frac = phase_start_f + (phase_end_f - phase_start_f) * float(max(0.0, min(1.0, nested)))
                global_frac = unit_start + (unit_end - unit_start) * float(max(0.0, min(1.0, phase_frac)))
                return render_progress("inference", float(global_frac))

            return mapped_progress

        def _unit_is_standalone_hunyuan(unit: dict[str, Any] | None) -> bool:
            if not isinstance(unit, dict) or str(unit.get("kind") or "") != "single":
                return False
            info = unit.get("info") if isinstance(unit.get("info"), dict) else {}
            return str((info or {}).get("kind") or "avatar").strip().lower() == "ltx"

        def _hunyuan_info_can_prefetch(info: dict[str, Any] | None) -> bool:
            if not isinstance(info, dict):
                return False
            if str(info.get("kind") or "avatar").strip().lower() != "ltx":
                return False
            entry = info.get("audio_entry") if isinstance(info.get("audio_entry"), dict) else {}
            if _smartblog_direct_clip_requested(dict(entry or {})):
                return False
            if bool((entry or {}).get("_smartblog_ltx_use_previous_last_frame")):
                return False
            return True

        hunyuan_prefetch_tasks: dict[int, asyncio.Task] = {}
        hunyuan_prefetch_enabled = bool(
            _env_flag("SMARTBLOG_RENDER_PREFETCH_INDEPENDENT_HUNYUAN", "1")
            and _env_flag("SMARTBLOG_HUNYUAN_SERVICE_REMOTE", "0")
        )
        hunyuan_prefetch_concurrency = int(
            max(1, min(4, _safe_int_env("SMARTBLOG_RENDER_PREFETCH_HUNYUAN_CONCURRENCY", 1)))
        )
        hunyuan_prefetch_sem = asyncio.Semaphore(int(hunyuan_prefetch_concurrency))

        async def _prefetch_hunyuan_segment(pos: int, info: dict[str, Any]) -> str:
            async with hunyuan_prefetch_sem:
                seg_video_path = os.path.join(run_dir, f"render_segment_{int(pos):03d}.mp4")
                seg_duration = float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0)
                if float(seg_duration) <= 0.0:
                    seg_duration = float(_smartblog_ltx_duration_seconds(dict(info.get("audio_entry") or {})) or 0.0)
                logging.warning(
                    "SmartBlog render Hunyuan prefetch start: job=%s segment=%d duration=%.3fs concurrency=%d",
                    str(job_id),
                    int(pos),
                    float(seg_duration),
                    int(hunyuan_prefetch_concurrency),
                )
                started = float(time.monotonic())
                path = await self._smartblog_render_hunyuan_timeline_clip(
                    claim=claim,
                    entry=dict(info.get("audio_entry") or {}),
                    run_dir=str(run_dir),
                    segment_index=int(pos),
                    segment_count=int(segment_count),
                    output_path=str(seg_video_path),
                    output_width=int(timeline_w),
                    output_height=int(timeline_h),
                    output_fps=float(_smartblog_render_source_fps()),
                    duration_sec=float(seg_duration),
                    render_job_type=str(render_job_type),
                    render_progress=render_progress,
                    conditioning_image_path="",
                    report_progress=False,
                )
                logging.warning(
                    "SmartBlog render Hunyuan prefetch ready: job=%s segment=%d elapsed=%.3fs file=%s",
                    str(job_id),
                    int(pos),
                    float(time.monotonic()) - float(started),
                    os.path.basename(str(path)),
                )
                return str(path)

        if bool(hunyuan_prefetch_enabled):
            for unit in render_units:
                if not _unit_is_standalone_hunyuan(unit):
                    continue
                pos = int(unit.get("index") or 0)
                info = dict(unit.get("info") or segment_infos[int(pos)] or {})
                if not _hunyuan_info_can_prefetch(info):
                    continue
                hunyuan_prefetch_tasks[int(pos)] = asyncio.create_task(
                    _prefetch_hunyuan_segment(int(pos), dict(info)),
                    name=f"smartblog-hunyuan-prefetch-{sanitize_job_id(job_id or 'job')}-{int(pos):03d}",
                )
            if hunyuan_prefetch_tasks:
                logging.warning(
                    "SmartBlog render Hunyuan prefetch scheduled: job=%s clips=%d concurrency=%d",
                    str(job_id),
                    int(len(hunyuan_prefetch_tasks)),
                    int(hunyuan_prefetch_concurrency),
                )

        for unit_pos, unit in enumerate(render_units):
            if str(unit.get("kind") or "") == "avatar_run":
                run_infos = [dict(info or {}) for info in list(unit.get("infos") or [])]
                if not run_infos:
                    continue
                run_start = int(unit.get("start") or 0)
                run_end = int(unit.get("end") or run_start)
                first_run_info = dict(run_infos[0] or {})
                prev_info = segment_infos[int(run_start) - 1] if int(run_start) > 0 else {}
                if (
                    bool(segment_continuity_paths)
                    and _segment_frame_index(prev_info) is not None
                    and _segment_frame_index(prev_info) == _segment_frame_index(first_run_info)
                    and not bool(first_run_info.get("avatar_transition_in"))
                ):
                    prev_segment_path = str(segment_continuity_paths[-1])
                    try:
                        run_filters = _smartblog_render_entry_filters(
                            claim,
                            first_run_info.get("audio_entry") if isinstance(first_run_info.get("audio_entry"), dict) else {},
                        )
                        run_face_restore = _smartblog_float_filter(run_filters, "face_restore", float(face_restore))
                        run_background_restore = _smartblog_render_background_restore_filter(
                            run_filters,
                            job_id=f"{job_id or '-'}_avatar_run_{int(run_start):03d}_continuation",
                        )
                        render_hw = _smartblog_parse_render_size_hw(str(render_size))
                        avatar_target_h, avatar_target_w = render_hw if render_hw is not None else (0, 0)
                        continuation_path = await self._smartblog_prepare_render_continuation_image(
                            job_id=str(job_id),
                            source_video_path=str(prev_segment_path),
                            raw_frame_path=os.path.join(str(run_dir), f"avatar_run_{int(run_start):03d}_continuation_raw.png"),
                            output_path=os.path.join(str(run_dir), f"avatar_run_{int(run_start):03d}_continuation_enhanced.png"),
                            face_restore=float(run_face_restore),
                            segment_index=int(run_start),
                            reason="avatar_run",
                            background_restore=float(run_background_restore),
                            target_width=int(avatar_target_w or 0),
                            target_height=int(avatar_target_h or 0),
                        )
                        run_infos[0]["avatar_path"] = str(continuation_path)
                        logging.warning(
                            "SmartBlog render avatar-run continuation source: job=%s run=%d..%d frame=%s source=%s face_restore=%.2f background_restore=%.2f",
                            str(job_id),
                            int(run_start),
                            int(run_end),
                            str(_segment_frame_index(first_run_info)),
                            os.path.basename(str(prev_segment_path)),
                            float(run_face_restore),
                            float(run_background_restore),
                        )
                    except Exception as e:
                        logging.exception(
                            "SmartBlog render avatar-run continuation frame failed: job=%s run=%d..%d frame=%s err=%s",
                            str(job_id),
                            int(run_start),
                            int(run_end),
                            str(_segment_frame_index(first_run_info)),
                            str(e),
                        )
                        raise
                run_total_samples = int(sum(int(info.get("target_samples") or 0) for info in run_infos))
                run_sample_rate = int(run_infos[0].get("sample_rate") or audio_sample_rate or 16000)
                run_duration = float(sum(float(info.get("target_duration_sec") or info.get("duration_sec") or 0.0) for info in run_infos))
                if float(run_duration) <= 0.0:
                    run_duration = float(run_total_samples) / float(max(1, int(run_sample_rate)))
                group_run_dir = os.path.join(str(run_dir), f"avatar_run_{int(run_start):03d}_{int(run_end):03d}")
                os.makedirs(str(group_run_dir), exist_ok=True)
                logging.warning(
                    "SmartBlog render mixed avatar-run start: job=%s run=%d..%d segments=%d duration=%.3fs",
                    str(job_id),
                    int(run_start),
                    int(run_end),
                    int(len(run_infos)),
                    float(run_duration),
                )
                run_has_inserts = any(_segment_has_frame_inserts(info) for info in run_infos)
                run_has_segment_postprocess = any(
                    bool((info or {}).get("avatar_transition_in"))
                    or bool((info or {}).get("avatar_transition_out"))
                    or bool(burn_subtitles and not force_final_subtitles)
                    for info in run_infos
                )
                run_avatar_progress_end = 0.65 if bool(run_has_inserts or run_has_segment_postprocess) else 1.0
                run_avatar_render_progress = _unit_inference_render_progress(
                    unit_pos=int(unit_pos),
                    phase_start=0.0,
                    phase_end=float(run_avatar_progress_end),
                )
                run_post_render_progress = _unit_inference_render_progress(
                    unit_pos=int(unit_pos),
                    phase_start=float(run_avatar_progress_end),
                    phase_end=1.0,
                )
                group_plan = await self._smartblog_render_avatar_liveaudio_one_pass(
                    claim=claim,
                    segment_infos=list(run_infos),
                    run_dir=str(group_run_dir),
                    job_id=str(job_id),
                    artifact_job_id=f"{job_id}_avatar_run_{int(run_start):03d}_{int(run_end):03d}",
                    render_job_type=str(render_job_type),
                    render_progress=run_avatar_render_progress,
                    render_size=str(render_size),
                    out_w=int(out_w),
                    out_h=int(out_h),
                    audio_sample_rate=int(run_sample_rate),
                    total_target_samples=int(run_total_samples),
                    total_duration_sec=float(run_duration),
                    sample_steps=int(sample_steps),
                    face_restore=float(face_restore),
                    background_restore=float(background_restore),
                    remote_edge_enabled=bool(remote_edge_enabled),
                    stream_file_enabled=bool(stream_file_enabled),
                    upload={},
                    upload_public_url="",
                    watermark_text="",
                    burn_subtitles=False,
                )
                group_path = str(group_plan.file_path or "").strip()
                if not group_path or not os.path.exists(group_path):
                    raise RuntimeError(f"render_video avatar run {run_start}..{run_end} produced no output")
                if not bool(run_has_inserts or run_has_segment_postprocess):
                    segment_video_paths.append(str(group_path))
                    segment_continuity_paths.append(str(group_path))
                    for run_info in run_infos:
                        original_idx = int(run_info.get("index") or 0)
                        timeline_duration = float(
                            run_info.get("target_duration_sec") or run_info.get("duration_sec") or 0.0
                        )
                        run_info["timeline_duration_sec"] = float(timeline_duration)
                        if 0 <= int(original_idx) < len(segment_infos):
                            segment_infos[int(original_idx)]["timeline_duration_sec"] = float(timeline_duration)
                    continue

                logging.warning(
                    "SmartBlog render mixed avatar-run layer split: job=%s run=%d..%d segments=%d inserts=%d postprocess=%d",
                    str(job_id),
                    int(run_start),
                    int(run_end),
                    int(len(run_infos)),
                    1 if bool(run_has_inserts) else 0,
                    1 if bool(run_has_segment_postprocess) else 0,
                )
                run_offset = 0.0
                for local_pos, run_info_raw in enumerate(run_infos):
                    run_info = dict(run_info_raw or {})
                    original_idx = int(run_info.get("index") or (int(run_start) + int(local_pos)))
                    segment_duration = float(run_info.get("target_duration_sec") or run_info.get("duration_sec") or 0.0)
                    if float(segment_duration) <= 0.0:
                        segment_duration = float(video_duration_sec(str(group_path)) or 0.0)
                    split_base_path = os.path.join(
                        str(group_run_dir),
                        f"avatar_run_{int(run_start):03d}_{int(run_end):03d}_segment_{int(original_idx):03d}.mp4",
                    )
                    trimmed = await asyncio.to_thread(
                        _smartblog_trim_mp4_interval,
                        src_path=str(group_path),
                        out_path=str(split_base_path),
                        start_sec=float(run_offset),
                        end_sec=float(run_offset + max(0.0, float(segment_duration))),
                        width=int(timeline_w),
                        height=int(timeline_h),
                        fps=float(_smartblog_render_source_fps()),
                    )
                    run_offset += max(0.0, float(segment_duration))
                    if not trimmed:
                        continue
                    processed_segment_path = await asyncio.to_thread(
                        _smartblog_postprocess_render_segment_for_concat,
                        src_path=str(trimmed),
                        run_dir=str(group_run_dir),
                        segment_info=dict(run_info or {}),
                        segment_index=int(original_idx),
                        segment_count=int(segment_count),
                        width=int(timeline_w),
                        height=int(timeline_h),
                        burn_subtitles=bool(burn_subtitles and not force_final_subtitles),
                    )
                    avatar_continuity_path = str(processed_segment_path)
                    if bool(_segment_has_frame_inserts(run_info)):
                        run_insert_render_progress = _unit_inference_render_progress(
                            unit_pos=int(unit_pos),
                            phase_start=float(run_avatar_progress_end),
                            phase_end=1.0,
                        )
                        processed_segment_path, avatar_continuity_path = await self._smartblog_compose_avatar_segment_inserts(
                            claim=claim,
                            base_path=str(processed_segment_path),
                            info=run_info,
                            run_dir=str(group_run_dir),
                            segment_index=int(original_idx),
                            segment_count=int(segment_count),
                            output_width=int(timeline_w),
                            output_height=int(timeline_h),
                            output_fps=float(_smartblog_render_source_fps()),
                            render_job_type=str(render_job_type),
                            render_progress=run_insert_render_progress,
                            previous_timeline_path=str(segment_continuity_paths[-1]) if segment_continuity_paths else "",
                        )
                    else:
                        run_info["timeline_duration_sec"] = float(segment_duration)
                    if 0 <= int(original_idx) < len(segment_infos):
                        timeline_duration_raw = (
                            run_info.get("timeline_duration_sec")
                            if run_info.get("timeline_duration_sec") is not None
                            else segment_duration
                        )
                        segment_infos[int(original_idx)]["timeline_duration_sec"] = float(
                            timeline_duration_raw or 0.0
                        )
                    segment_video_paths.append(str(processed_segment_path))
                    segment_continuity_paths.append(str(avatar_continuity_path))
                continue

            pos = int(unit.get("index") or 0)
            info = dict(unit.get("info") or segment_infos[int(pos)] or {})
            seg_frac_base = float(pos) / float(max(1, segment_count))
            seg_video_path = os.path.join(run_dir, f"render_segment_{int(pos):03d}.mp4")
            seg_progress_path = f"{seg_video_path}.progress.json"
            seg_duration = float(info["target_duration_sec"])
            seg_job_id = f"{job_id}_seg_{int(pos):03d}"
            if str(info.get("kind") or "avatar").strip().lower() == "ltx":
                await self._smartblog_progress_checked(
                    job_id=job_id,
                    progress=render_progress("inference", seg_frac_base),
                    **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference", stage_label="Running Hunyuan insert"),
                )
                hunyuan_entry = dict(info.get("audio_entry") or {})
                hunyuan_conditioning_image_path = ""
                insert_after: int | None = None
                if hunyuan_entry.get("_smartblog_ltx_insert_after_chunk") is not None:
                    try:
                        insert_after = int(hunyuan_entry.get("_smartblog_ltx_insert_after_chunk"))
                    except Exception:
                        insert_after = None
                prev_info = segment_infos[int(pos) - 1] if int(pos) > 0 else {}
                same_frame_as_prev = (
                    _segment_frame_index(prev_info) is not None
                    and _segment_frame_index(prev_info) == _segment_frame_index(info)
                )
                should_continue_from_prev = bool(segment_continuity_paths) and bool(
                    hunyuan_entry.get("_smartblog_ltx_use_previous_last_frame")
                    and not _smartblog_direct_clip_requested(dict(hunyuan_entry or {}))
                )
                if should_continue_from_prev:
                    prev_segment_path = str(segment_continuity_paths[-1])
                    try:
                        hunyuan_continuation_filters = _smartblog_render_entry_filters(claim, hunyuan_entry)
                        hunyuan_continuation_face_restore = _smartblog_float_filter(
                            hunyuan_continuation_filters,
                            "face_restore",
                            0.5,
                        )
                        hunyuan_continuation_background_restore = _smartblog_render_background_restore_filter(
                            hunyuan_continuation_filters,
                            job_id=f"{job_id or '-'}_hunyuan_continuation_{int(pos):03d}",
                        )
                        hunyuan_conditioning_image_path = await self._smartblog_prepare_render_continuation_image(
                            job_id=str(job_id),
                            source_video_path=str(prev_segment_path),
                            raw_frame_path=os.path.join(str(run_dir), f"hunyuan_segment_{int(pos):03d}_continuation_raw.png"),
                            output_path=os.path.join(str(run_dir), f"hunyuan_segment_{int(pos):03d}_continuation_enhanced.png"),
                            face_restore=float(hunyuan_continuation_face_restore),
                            segment_index=int(pos),
                            reason="hunyuan",
                            background_restore=float(hunyuan_continuation_background_restore),
                            target_width=int(timeline_w),
                            target_height=int(timeline_h),
                        )
                        logging.warning(
                            "SmartBlog render Hunyuan continuation source: job=%s segment=%d after_chunk=%d source=%s face_restore=%.2f background_restore=%.2f resize=0 exact_frame=1",
                            str(job_id),
                            int(pos),
                            int(insert_after) if insert_after is not None else -999,
                            os.path.basename(str(prev_segment_path)),
                            float(hunyuan_continuation_face_restore),
                            float(hunyuan_continuation_background_restore),
                        )
                    except Exception as e:
                        logging.exception(
                            "SmartBlog render Hunyuan continuation frame failed: job=%s segment=%d after_chunk=%s err=%s",
                            str(job_id),
                            int(pos),
                            str(insert_after),
                            str(e),
                        )
                        raise
                prefetch_task = hunyuan_prefetch_tasks.pop(int(pos), None)
                if prefetch_task is not None:
                    if prefetch_task.done():
                        hunyuan_path = await prefetch_task
                    else:
                        hunyuan_path = await self._smartblog_wait_with_progress(
                            task=prefetch_task,
                            job_id=str(job_id),
                            progress=render_progress("inference", seg_frac_base),
                            **_smartblog_progress_stage_fields(
                                job_type=render_job_type,
                                stage="inference",
                                stage_label="Waiting Hunyuan prefetch",
                            ),
                        )
                    logging.warning(
                        "SmartBlog render Hunyuan prefetch consumed: job=%s segment=%d done=%d",
                        str(job_id),
                        int(pos),
                        1 if bool(prefetch_task.done()) else 0,
                    )
                else:
                    if _smartblog_direct_clip_requested(dict(hunyuan_entry or {})):
                        audio_block = (
                            hunyuan_entry.get("_smartblog_ltx_audio")
                            if isinstance(hunyuan_entry.get("_smartblog_ltx_audio"), dict)
                            else {}
                        )
                        direct_audio_cfg = (
                            _smartblog_normalize_video_audio_config(dict(audio_block), default_gain_db=0.0)
                            if audio_block
                            else None
                        )
                        hunyuan_path = await self._smartblog_prepare_direct_clip_timeline_clip(
                            claim=claim,
                            entry=dict(hunyuan_entry),
                            run_dir=str(run_dir),
                            segment_index=int(pos),
                            segment_count=int(segment_count),
                            output_path=str(seg_video_path),
                            output_width=int(timeline_w),
                            output_height=int(timeline_h),
                            output_fps=float(_smartblog_render_source_fps()),
                            duration_sec=float(seg_duration),
                            render_job_type=str(render_job_type),
                            render_progress=render_progress,
                            audio_config=direct_audio_cfg,
                        )
                    else:
                        hunyuan_path = await self._smartblog_render_hunyuan_timeline_clip(
                            claim=claim,
                            entry=dict(hunyuan_entry),
                            run_dir=str(run_dir),
                            segment_index=int(pos),
                            segment_count=int(segment_count),
                            output_path=str(seg_video_path),
                            output_width=int(timeline_w),
                            output_height=int(timeline_h),
                            output_fps=float(_smartblog_render_source_fps()),
                            duration_sec=float(seg_duration),
                            render_job_type=str(render_job_type),
                            render_progress=render_progress,
                            conditioning_image_path=str(hunyuan_conditioning_image_path),
                        )
                hunyuan_continuity_path = str(hunyuan_path)
                if bool(info.get("avatar_transition_in")) and bool(_env_flag("SMARTBLOG_RENDER_AVATAR_TRANSITION_BLUR", "1")):
                    hunyuan_path = await asyncio.to_thread(
                        _smartblog_apply_segment_transition_blur,
                        src_path=str(hunyuan_path),
                        out_path=os.path.join(str(run_dir), f"render_segment_{int(pos):03d}_transition.mp4"),
                        has_start=True,
                        has_end=False,
                        width=int(timeline_w),
                        height=int(timeline_h),
                    )
                segment_video_paths.append(str(hunyuan_path))
                segment_continuity_paths.append(str(hunyuan_continuity_path))
                continue
            seg_remote_live_raw_dir: str | None = None
            seg_remote_progress_path = ""
            seg_entry = dict(info.get("audio_entry") or {})
            seg_filters = _smartblog_render_entry_filters(claim, seg_entry)
            seg_face_restore = _smartblog_float_filter(seg_filters, "face_restore", float(face_restore))
            seg_background_restore = _smartblog_render_background_restore_filter(seg_filters, job_id=seg_job_id)
            seg_prompt = _smartblog_render_entry_prompt(claim, seg_entry)
            seg_video_prompt = _smartblog_render_entry_video_prompt(claim, seg_entry)
            seg_negative_prompt = _smartblog_render_entry_negative_prompt(claim, seg_entry)
            if any(
                key in seg_entry
                for key in (
                    "_smartblog_frame_video_prompt",
                    "_smartblog_frame_video_negative_prompt",
                    "_smartblog_frame_face_restore",
                    "_smartblog_frame_background_restore",
                )
            ):
                logging.warning(
                    "SmartBlog render frame overrides: job=%s segment=%d frame=%s video_prompt=%d negative_prompt=%d face_restore=%.2f background_restore=%.2f",
                    str(job_id),
                    int(pos),
                    str(seg_entry.get("_smartblog_frame_index", "-")),
                    int(len(str(seg_video_prompt or ""))),
                    int(len(str(seg_negative_prompt or ""))),
                    float(seg_face_restore),
                    float(seg_background_restore),
                )
            seg_num_clip = max(
                1,
                int(
                    auto_num_clip_for_duration(
                        seg_duration,
                        fps=int(WORKER_FPS),
                        infer_frames=int(infer_frames),
                    )
                ),
            )
            if bool(remote_edge_enabled):
                seg_upload_plan = SmartBlogRenderFinalizePlan(
                    job_id=str(seg_job_id),
                    job_type=str(render_job_type),
                    signed_url="",
                    upload_path=(
                        f"worker-uploads/render_segments/{sanitize_job_id(job_id or 'render')}/"
                        f"segment_{int(pos):03d}.mp4"
                    ),
                    file_path=str(seg_video_path),
                    content_type="video/mp4",
                    complete_kwargs={},
                    run_dir=run_dir,
                )
                seg_signed_url, seg_upload_path = await self._smartblog_resolve_upload_target(seg_upload_plan)
                seg_remote_live_raw_dir = os.path.abspath(prepare_live_raw_dir(str(run_dir), f"{seg_job_id}_remote"))
                seg_remote_progress_path = os.path.join(str(seg_remote_live_raw_dir), "remote_edge_file_progress.json")
                _smartblog_write_render_remote_edge_manifest(
                    claim=claim,
                    live_raw_dir=str(seg_remote_live_raw_dir),
                    job_id=str(seg_job_id),
                    width=int(timeline_w),
                    height=int(timeline_h),
                    fps=int(WORKER_FPS),
                    sample_rate=int(info.get("sample_rate") or audio_sample_rate or 16000),
                    target_audio_samples=int(info.get("target_samples") or 0),
                    target_duration_sec=float(seg_duration),
                    upload={"signed_url": str(seg_signed_url), "path": str(seg_upload_path)},
                    public_url=self._smartblog_public_storage_url(str(seg_upload_path)),
                    watermark_text="",
                    remote_finalizer=False if bool(remote_finalizer_service_url) else None,
                    file_output_fps=int(round(float(_smartblog_render_source_fps()))),
                )

            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=render_progress("inference", seg_frac_base),
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
            )

            seg_infer_progress_started_mono = float(time.monotonic())
            avatar_model_path = str(info["avatar_path"])
            prev_info = segment_infos[int(pos) - 1] if int(pos) > 0 else {}
            if (
                bool(segment_continuity_paths)
                and _segment_frame_index(prev_info) is not None
                and _segment_frame_index(prev_info) == _segment_frame_index(info)
                and not bool(info.get("avatar_transition_in"))
            ):
                prev_segment_path = str(segment_continuity_paths[-1])
                try:
                    render_hw = _smartblog_parse_render_size_hw(str(render_size))
                    avatar_target_h, avatar_target_w = render_hw if render_hw is not None else (0, 0)
                    avatar_model_path = await self._smartblog_prepare_render_continuation_image(
                        job_id=str(job_id),
                        source_video_path=str(prev_segment_path),
                        raw_frame_path=os.path.join(str(run_dir), f"avatar_segment_{int(pos):03d}_continuation_raw.png"),
                        output_path=os.path.join(str(run_dir), f"avatar_segment_{int(pos):03d}_continuation_enhanced.png"),
                        face_restore=float(seg_face_restore),
                        segment_index=int(pos),
                        reason="avatar",
                        background_restore=float(seg_background_restore),
                        target_width=int(avatar_target_w or 0),
                        target_height=int(avatar_target_h or 0),
                    )
                    logging.warning(
                        "SmartBlog render avatar continuation source: job=%s segment=%d frame=%s source=%s face_restore=%.2f background_restore=%.2f exact_frame=1 anti_crop=1 model_target=%dx%d",
                        str(job_id),
                        int(pos),
                        str(_segment_frame_index(info)),
                        os.path.basename(str(prev_segment_path)),
                        float(seg_face_restore),
                        float(seg_background_restore),
                        int(avatar_target_w or 0),
                        int(avatar_target_h or 0),
                    )
                except Exception as e:
                    logging.exception(
                        "SmartBlog render avatar continuation frame failed: job=%s segment=%d frame=%s err=%s",
                        str(job_id),
                        int(pos),
                        str(_segment_frame_index(info)),
                        str(e),
                    )
                    raise

            def _segment_progress_provider(
                *,
                progress_path: str = str(seg_remote_progress_path or seg_progress_path),
                model_raw_dir: str = str(seg_remote_live_raw_dir or ""),
                segment_pos: int = int(pos),
                segment_num_clip: int = int(seg_num_clip),
                segment_target_frames: int = int(max(1, int(seg_num_clip) * int(infer_frames))),
                started_mono: float = float(seg_infer_progress_started_mono),
            ) -> dict[str, Any]:
                estimated_frac = _smartblog_estimated_render_inference_stage_frac(
                    num_clip=int(segment_num_clip),
                    started_mono=float(started_mono),
                )
                remote_frac: float | None = None
                model_raw_frac: float | None = None
                try:
                    with open(progress_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        raw = obj.get("progress")
                        if raw is None:
                            raw = obj.get("stage_progress")
                        if raw is not None:
                            remote_frac = float(raw)
                except Exception:
                    pass
                if model_raw_dir:
                    model_raw_frac = _smartblog_render_model_raw_progress_fraction(
                        _smartblog_read_render_model_raw_progress(str(model_raw_dir)),
                        target_frames=int(segment_target_frames),
                    )
                frac = max(
                    float(estimated_frac),
                    float(remote_frac) if remote_frac is not None else 0.0,
                    float(model_raw_frac) if model_raw_frac is not None else 0.0,
                )
                total_frac = (float(segment_pos) + max(0.0, min(1.0, float(frac)))) / float(max(1, segment_count))
                return {
                    "progress": render_progress("inference", total_frac),
                    **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
                }

            infer_task = asyncio.create_task(
                self._model_client.infer(
                    req=InferRequest(
                        prompt=str(seg_prompt),
                        video_prompt=str(seg_video_prompt),
                        negative_prompt=str(seg_negative_prompt),
                        idle_prompt=_smartblog_render_idle_prompt(claim),
                        image_path=str(avatar_model_path),
                        audio_path=str(info["audio_wav_path"]),
                        lipsync_audio_path=str(info.get("lipsync_audio_wav_path") or ""),
                        num_clip=int(seg_num_clip),
                        sample_steps=int(sample_steps),
                        sample_guide_scale=float(getattr(self.args, "sample_guide_scale", 0.0) or 0.0),
                        infer_frames=int(infer_frames),
                        size=str(render_size),
                        base_seed=int(os.getenv("BASE_SEED", "420") or 420),
                        sample_solver=str(getattr(self.args, "sample_solver", "euler") or "euler"),
                        face_restore=float(seg_face_restore),
                        background_restore=float(seg_background_restore),
                        job_id=f"{job_id}_seg_{int(pos):03d}",
                        enable_live_hls=False,
                        live_raw_dir=seg_remote_live_raw_dir if bool(remote_edge_enabled) else None,
                        save_live_raw_mp4=False,
                        stream_file_output_path=str(seg_video_path) if bool(stream_file_enabled) else "",
                        stream_file_output_width=int(timeline_w) if bool(stream_file_enabled) else 0,
                        stream_file_output_height=int(timeline_h) if bool(stream_file_enabled) else 0,
                        stream_file_output_fps=float(_smartblog_render_source_fps()) if bool(stream_file_enabled) else 0.0,
                        stream_file_trim_duration_sec=float(seg_duration) if bool(stream_file_enabled) else 0.0,
                        stream_file_interpolation=(
                            str(os.getenv("SMARTBLOG_RENDER_STREAM_INTERPOLATION", "") or "")
                            if bool(stream_file_enabled)
                            else ""
                        ),
                        tpp_cfg_mode=_smartblog_render_tpp_cfg_mode(),
                    )
                )
            )
            infer_resp = await self._smartblog_wait_with_progress(
                task=infer_task,
                job_id=job_id,
                progress=render_progress("inference", seg_frac_base),
                cancel_model_infer_on_stop=True,
                progress_provider=_segment_progress_provider,
                **_smartblog_progress_stage_fields(job_type=render_job_type, stage="inference"),
            )
            if not bool(infer_resp.ok):
                raise RuntimeError(str(infer_resp.error or f"render_video segment {pos + 1} infer failed"))
            out_path = str(infer_resp.video_path or seg_video_path).strip() or str(seg_video_path)
            if out_path.startswith(SMARTBLOG_REMOTE_EDGE_UPLOADED_PREFIX):
                uploaded_path = out_path[len(SMARTBLOG_REMOTE_EDGE_UPLOADED_PREFIX) :].strip()
                if not uploaded_path:
                    raise RuntimeError(f"render_video segment {pos + 1} remote edge returned empty upload path")
                await self._smartblog_download_file(url=str(uploaded_path), out_path=str(seg_video_path))
                out_path = str(seg_video_path)
            if not out_path or not os.path.exists(out_path):
                raise RuntimeError(f"render_video segment {pos + 1} produced no output")
            avatar_continuity_path = str(out_path)
            postprocess_kwargs = {
                "src_path": str(out_path),
                "run_dir": str(run_dir),
                "segment_info": dict(info or {}),
                "segment_index": int(pos),
                "segment_count": int(segment_count),
                "width": int(timeline_w),
                "height": int(timeline_h),
                "burn_subtitles": bool(burn_subtitles and not force_final_subtitles),
            }
            if bool(segment_postprocess_background):
                segment_postprocess_tasks.append(
                    asyncio.create_task(
                        asyncio.to_thread(_smartblog_postprocess_render_segment_for_concat, **postprocess_kwargs),
                        name=f"smartblog-render-segment-postprocess-{sanitize_job_id(job_id or 'job')}-{int(pos):03d}",
                    )
                )
                segment_continuity_paths.append(str(avatar_continuity_path))
            else:
                processed_segment_path = await asyncio.to_thread(_smartblog_postprocess_render_segment_for_concat, **postprocess_kwargs)
                segment_has_inserts = bool(_smartblog_frame_insert_entries(dict(seg_entry or {})))
                if bool(segment_has_inserts):
                    processed_segment_path, avatar_continuity_path = await self._smartblog_compose_avatar_segment_inserts(
                        claim=claim,
                        base_path=str(processed_segment_path),
                        info=info,
                        run_dir=str(run_dir),
                        segment_index=int(pos),
                        segment_count=int(segment_count),
                        output_width=int(timeline_w),
                        output_height=int(timeline_h),
                        output_fps=float(_smartblog_render_source_fps()),
                        render_job_type=str(render_job_type),
                        render_progress=render_progress,
                        previous_timeline_path=str(segment_continuity_paths[-1]) if segment_continuity_paths else "",
                    )
                    if 0 <= int(pos) < len(segment_infos):
                        timeline_duration_raw = (
                            info.get("timeline_duration_sec")
                            if info.get("timeline_duration_sec") is not None
                            else video_duration_sec(str(processed_segment_path))
                        )
                        try:
                            segment_infos[int(pos)]["timeline_duration_sec"] = float(
                                timeline_duration_raw or seg_duration or 0.0
                            )
                        except Exception:
                            segment_infos[int(pos)]["timeline_duration_sec"] = float(seg_duration or 0.0)
                segment_video_paths.append(str(processed_segment_path))
                segment_continuity_paths.append(str(avatar_continuity_path))

        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.10),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Preparing timeline compose",
            ),
        )
        if segment_postprocess_tasks:
            postprocess_future = asyncio.gather(*segment_postprocess_tasks)
            concat_video_paths = list(
                await self._smartblog_wait_with_progress(
                    task=postprocess_future,
                    job_id=job_id,
                    progress=render_progress("encode", 0.25),
                    **_smartblog_progress_stage_fields(
                        job_type=render_job_type,
                        stage="encode",
                        stage_label="Post-processing segments",
                    ),
                )
            )
        else:
            concat_video_paths = list(segment_video_paths)
        concat_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_concat_final_timeline,
                list(concat_video_paths),
                str(final_video_path),
                segment_infos=list(segment_infos),
                width=int(timeline_w),
                height=int(timeline_h),
                fps=float(_smartblog_render_source_fps()),
            ),
            name=f"smartblog-render-concat-{sanitize_job_id(str(job_id or 'job'))}",
        )
        await self._smartblog_wait_with_progress(
            task=concat_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.45),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Composing final timeline",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video segmented concat produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.55),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Timeline composed",
            ),
        )
        if bool(remote_finalizer_enabled):
            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=render_progress("encode", 0.62),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Uploading timeline for media finalizer",
                ),
            )
            source_upload_path = (
                f"worker-uploads/render_segments/{sanitize_job_id(job_id or 'render')}/"
                f"{sanitize_job_id(job_id or 'render')}_timeline_pre_finalizer.mp4"
            )
            source_upload_plan = SmartBlogRenderFinalizePlan(
                job_id=f"{job_id}_timeline_pre_finalizer",
                job_type=str(render_job_type),
                signed_url="",
                upload_path=str(source_upload_path),
                file_path=str(final_video_path),
                content_type="video/mp4",
                complete_kwargs={},
                run_dir=str(run_dir),
            )
            source_signed_url, source_uploaded_path = await self._smartblog_resolve_upload_target(source_upload_plan)
            await self._smartblog_upload_file(
                signed_url=str(source_signed_url),
                file_path=str(final_video_path),
                content_type="video/mp4",
            )
            background_music_cfg = _smartblog_background_music_config(claim)
            background_music_enabled = bool(background_music_cfg.get("enabled"))
            remote_finalizer_subtitle_chunks_json = (
                _smartblog_render_subtitle_chunks_json(list(segment_infos or [])) if bool(burn_subtitles) else ""
            )
            remote_source_fps = int(round(float(_smartblog_render_source_fps())))
            remote_target_fps = int(round(float(_smartblog_render_delivery_fps())))
            remote_upscale_enabled = bool(_smartblog_remote_finalizer_quality_pass_enabled())
            remote_target_w, remote_target_h = _smartblog_render_delivery_dimensions(
                claim,
                output_width=int(out_w),
                output_height=int(out_h),
            )
            upload_path_final = str(upload.get("path") or "").strip()
            upload_public_url_final = str(upload_public_url or "").strip()
            if not upload_public_url_final and upload_path_final:
                upload_public_url_final = self._smartblog_public_storage_url(str(upload_path_final))
            logging.warning(
                "SmartBlog render segmented queued remote finalizer: job=%s source=%s target=%s subtitles_finalizer=%d watermark_chars=%d background_music=%d background_music_gain=%.2fdB upscale=%d target=%dx%d fps=%d->%d",
                str(job_id or "-"),
                str(source_uploaded_path),
                str(upload_path_final),
                1 if bool(remote_finalizer_subtitle_chunks_json) else 0,
                int(len(str(watermark_text or "").strip())),
                1 if bool(background_music_enabled) else 0,
                float(background_music_cfg.get("gain_db") or 0.0),
                1 if bool(remote_upscale_enabled) else 0,
                int(remote_target_w),
                int(remote_target_h),
                int(remote_source_fps),
                int(remote_target_fps),
            )
            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=render_progress("encode", 0.02),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Queued media finalizer",
                ),
            )
            return SmartBlogRenderFinalizePlan(
                job_id=job_id,
                job_type=str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
                signed_url=str(upload.get("signed_url") or ""),
                upload_path=str(upload_path_final),
                file_path="",
                content_type="video/mp4",
                complete_kwargs={
                    "storage_path": str(upload_path_final),
                    **({"video_url": str(upload_public_url_final)} if upload_public_url_final else {}),
                },
                run_dir=str(run_dir),
                remote_finalizer_source_path=str(source_uploaded_path),
                remote_finalizer_source_fps=int(remote_source_fps),
                remote_finalizer_target_width=int(remote_target_w),
                remote_finalizer_target_height=int(remote_target_h),
                remote_finalizer_target_fps=int(remote_target_fps),
                remote_finalizer_upscale_enabled=bool(remote_upscale_enabled),
                remote_finalizer_background_music_url=(
                    str(background_music_cfg.get("audio_url") or "") if bool(background_music_enabled) else ""
                ),
                remote_finalizer_background_music_gain_db=float(background_music_cfg.get("gain_db") or 0.0),
                remote_finalizer_background_music_loop=bool(background_music_cfg.get("loop", True)),
                remote_finalizer_background_music_duck_voice_db=float(
                    background_music_cfg.get("duck_voice_db") or 0.0
                ),
                remote_finalizer_background_music_fade_in_seconds=float(
                    background_music_cfg.get("fade_in_seconds") or 0.0
                ),
                remote_finalizer_background_music_fade_out_seconds=float(
                    background_music_cfg.get("fade_out_seconds") or 0.0
                ),
                remote_finalizer_background_music_start_offset_seconds=float(
                    background_music_cfg.get("start_offset_seconds") or 0.0
                ),
                remote_finalizer_subtitle_chunks_json=str(remote_finalizer_subtitle_chunks_json),
                remote_finalizer_watermark_text=str(watermark_text or ""),
            )
        if _smartblog_render_subtitle_stage() == "final" or bool(force_final_subtitles):
            subtitle_task = asyncio.create_task(
                asyncio.to_thread(
                    _smartblog_maybe_burn_render_subtitles,
                    video_path=str(final_video_path),
                    out_path=os.path.join(run_dir, "render_final_subtitled.mp4"),
                    ass_path=os.path.join(run_dir, "render_subtitles.ass"),
                    segment_infos=list(segment_infos),
                    width=int(out_w),
                    height=int(out_h),
                    enabled=bool(burn_subtitles),
                ),
                name=f"smartblog-render-subtitles-{sanitize_job_id(str(job_id or 'job'))}",
            )
            final_video_path = await self._smartblog_wait_with_progress(
                task=subtitle_task,
                job_id=str(job_id),
                progress=render_progress("encode", 0.65),
                **_smartblog_progress_stage_fields(
                    job_type=render_job_type,
                    stage="encode",
                    stage_label="Burning subtitles",
                ),
            )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video subtitle burn produced no output")
        final_video_path = await self._smartblog_apply_background_music_if_needed(
            claim=claim,
            video_path=str(final_video_path),
            run_dir=str(run_dir),
            out_path=os.path.join(str(run_dir), "render_final_background_music.mp4"),
            job_id=str(job_id),
            pending_progress=render_progress("encode", 0.78),
            pending_stage_fields=_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Mixing background music",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video background music mux produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 0.84),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Final audio ready",
            ),
        )
        watermark_task = asyncio.create_task(
            asyncio.to_thread(
                _smartblog_maybe_burn_render_watermark,
                video_path=str(final_video_path),
                out_path=os.path.join(run_dir, "render_final_watermarked.mp4"),
                watermark_text=str(watermark_text or ""),
                run_dir=str(run_dir),
                width=int(out_w),
                height=int(out_h),
            ),
            name=f"smartblog-render-watermark-{sanitize_job_id(str(job_id or 'job'))}",
        )
        final_video_path = await self._smartblog_wait_with_progress(
            task=watermark_task,
            job_id=str(job_id),
            progress=render_progress("encode", 0.92),
            **_smartblog_progress_stage_fields(
                job_type=render_job_type,
                stage="encode",
                stage_label="Burning watermark",
            ),
        )
        if not os.path.exists(final_video_path):
            raise RuntimeError("render_video watermark burn produced no output")
        await self._smartblog_progress_checked(
            job_id=job_id,
            progress=render_progress("encode", 1.0),
            **_smartblog_progress_stage_fields(job_type=render_job_type, stage="encode"),
        )
        return SmartBlogRenderFinalizePlan(
            job_id=job_id,
            job_type=str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO),
            signed_url=str(upload.get("signed_url") or ""),
            upload_path=str(upload.get("path") or ""),
            file_path=str(final_video_path),
            content_type="video/mp4",
            complete_kwargs={
                "video_url": str(upload_public_url or self._smartblog_public_storage_url(str(upload.get("path") or ""))),
            },
            run_dir=run_dir,
        )

    async def _smartblog_render_video_job(self, claim: dict[str, Any]) -> SmartBlogRenderFinalizePlan:
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        upload = claim.get("upload") if isinstance(claim.get("upload"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        avatar_urls = _smartblog_render_asset_urls(claim, "avatar")
        avatar_url = str(avatar_urls[0] if avatar_urls else "").strip()
        timeline_entries = _smartblog_render_timeline_entries(claim)
        audio_entries = list(timeline_entries) if timeline_entries else _smartblog_render_audio_entries(claim)
        audio_url = str((audio_entries[0] or {}).get("url") if audio_entries else "").strip()
        run_dir = self._smartblog_job_run_dir(job_id)
        render_mode = _smartblog_render_mode(claim)
        render_job_type = str(job.get("job_type") or SMARTBLOG_JOB_TYPE_RENDER_VIDEO).strip().lower()
        if render_job_type not in set(smartblog_render_job_types()):
            render_job_type = SMARTBLOG_JOB_TYPE_RENDER_VIDEO
        render_progress = lambda stage, frac=1.0: _smartblog_stage_progress_total(
            job_type=render_job_type,
            stage=stage,
            stage_progress=frac,
        )
        if render_mode == "t2v":
            if not audio_url and not timeline_entries and _smartblog_render_script_text(claim):
                audio_entries = await self._smartblog_synthesize_render_tts_audio_entries(
                    claim=claim,
                    run_dir=str(run_dir),
                    job_id=str(job_id),
                    render_job_type=str(render_job_type),
                    render_progress=render_progress,
                )
                audio_url = str((audio_entries[0] or {}).get("url") if audio_entries else "").strip()
            return await self._smartblog_render_hunyuan_video_job(
                claim,
                run_dir=str(run_dir),
                render_job_type=str(render_job_type),
                render_progress=render_progress,
                audio_entries=list(audio_entries),
                render_mode=str(render_mode),
            )
        if not timeline_entries and _smartblog_ltx_render_requested(claim):
            return await self._smartblog_render_hunyuan_video_job(
                claim,
                run_dir=str(run_dir),
                render_job_type=str(render_job_type),
                render_progress=render_progress,
                audio_entries=list(audio_entries),
                render_mode=str(render_mode),
            )
        if not avatar_url:
            raise RuntimeError("render_video requires assets.avatar_url or payload_json.avatar_url/image_url/photo_url")
        if not audio_url and not timeline_entries:
            audio_entries = await self._smartblog_synthesize_render_tts_audio_entries(
                claim=claim,
                run_dir=str(run_dir),
                job_id=str(job_id),
                render_job_type=str(render_job_type),
                render_progress=render_progress,
            )
            audio_url = str((audio_entries[0] or {}).get("url") if audio_entries else "").strip()
        if not audio_url and not timeline_entries:
            raise RuntimeError(
                "render_video requires assets.audio_url/audio_chunks or assets.script_text for worker-side TTS"
            )
        return await self._smartblog_render_segmented_video_job(
            claim,
            audio_entries=list(audio_entries),
            avatar_urls=list(avatar_urls),
        )

    async def _run_smartblog_render_job(self, claim: dict[str, Any]) -> None:
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_id = str(job.get("id") or "").strip()
        job_type = str(job.get("job_type") or "").strip().lower()
        self._active_claim = dict(claim or {})
        self._active_job = dict(job or {})
        self.current_publish_task_id = job_id or None
        self.current_session_id = job_id or None
        self._active_source = "smartblog"
        self._active_session_started_mono = float(time.monotonic())
        self._smartblog_session_transport = "render"
        self._last_progress_ok_mono = float(time.monotonic())
        hunyuan_on_demand = self._smartblog_hunyuan_service_on_demand_enabled()
        render_job_type = job_type if job_type in set(smartblog_render_job_types()) else SMARTBLOG_JOB_TYPE_RENDER_VIDEO
        keepalive_stop = asyncio.Event()
        keepalive_task: asyncio.Task | None = None
        try:
            await self._smartblog_progress_checked(
                job_id=job_id,
                progress=max(
                    1,
                    _smartblog_stage_progress_total(
                        job_type=str(render_job_type),
                        stage="prepare",
                        stage_progress=0.1,
                    ),
                ),
                **_smartblog_progress_stage_fields(
                    job_type=str(render_job_type),
                    stage="prepare",
                    stage_label="Worker claimed render job",
                ),
            )
            keepalive_task = asyncio.create_task(
                self._smartblog_progress_keepalive_loop(
                    job_id=str(job_id),
                    job_type=str(render_job_type),
                    stop_event=keepalive_stop,
                ),
                name=f"smartblog-render-keepalive-{sanitize_job_id(job_id or job_type or 'job')}",
            )
            if bool(hunyuan_on_demand):
                # The single-GPU B200 render profile cannot keep LiveAvatar and
                # Hunyuan resident at the same time during long avatar passes.
                # Stop any leftover Hunyuan service before the job; it will be
                # started lazily when a Hunyuan clip is actually requested.
                await self._smartblog_hunyuan_service_ctl(
                    "stop",
                    timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_STOP_TIMEOUT_SEC", 120.0),
                    log_prefix=f"SmartBlog render job {job_id or '-'}",
                )
            finalize_plan: SmartBlogRenderFinalizePlan | None = None
            if job_type in set(smartblog_render_job_types()):
                finalize_plan = await self._smartblog_render_video_job(claim)
            else:
                raise RuntimeError(f"unsupported SmartBlog render job_type={job_type!r}")
            if finalize_plan is not None:
                if bool(smartblog_render_finalize_background_enabled()):
                    self._smartblog_schedule_finalize(finalize_plan)
                else:
                    await self._smartblog_finalize_render_job(finalize_plan)
            keepalive_stop.set()
        except SmartBlogJobStoppedByServer:
            keepalive_stop.set()
            logging.warning("SmartBlog render job stopped by server: id=%s type=%s", job_id or "-", job_type or "-")
        except Exception as e:
            keepalive_stop.set()
            if smartblog_is_transient_api_error(e):
                logging.warning(
                    "SmartBlog render job transient API/network failure; leaving job for server requeue: id=%s type=%s err=%s",
                    job_id or "-",
                    job_type or "-",
                    e,
                )
                return
            err = str(e or "SmartBlog render job failed").strip() or "SmartBlog render job failed"
            logging.exception("SmartBlog render job failed: id=%s type=%s err=%s", job_id or "-", job_type or "-", e)
            if _smartblog_render_is_local_runtime_failure(e):
                marker = getattr(self, "_smartblog_mark_claim_quarantine", None)
                if callable(marker):
                    try:
                        marker(
                            reason="local_runtime_failure",
                            error=e,
                            job_id=job_id,
                            job_type=job_type,
                        )
                    except Exception as marker_err:
                        logging.warning(
                            "SmartBlog render claim quarantine marker failed: id=%s err=%s",
                            job_id or "-",
                            marker_err,
                        )
            await self._smartblog_api.fail(job_id=job_id, error_text=err[:1500])
        finally:
            if bool(hunyuan_on_demand):
                try:
                    await self._smartblog_hunyuan_service_ctl(
                        "stop",
                        timeout_sec=_safe_float_env("SMARTBLOG_HUNYUAN_SERVICE_STOP_TIMEOUT_SEC", 120.0),
                        log_prefix=f"SmartBlog render job {job_id or '-'}",
                    )
                except Exception as e:
                    logging.warning(
                        "SmartBlog render job Hunyuan on-demand stop failed: id=%s err=%s",
                        job_id or "-",
                        e,
                    )
            self.current_publish_task_id = None
            self.current_session_id = None
            self._active_claim = None
            self._active_job = None
            self._smartblog_last_progress_payload = {}
            self._active_source = ""
            self._active_session_started_mono = 0.0
            self._smartblog_session_transport = "channel"
            keepalive_stop.set()
            if isinstance(keepalive_task, asyncio.Task) and (not keepalive_task.done()):
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)

    async def _run_smartblog_job(self, claim: dict[str, Any]) -> None:
        if smartblog_is_render_job(claim):
            await self._run_smartblog_render_job(claim)
            return
        job_type = str(smartblog_job_type_value(claim) or "").strip().lower()
        raise RuntimeError(f"render-only worker cannot run SmartBlog job_type={job_type!r}")
