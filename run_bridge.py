from __future__ import annotations

import base64
import json
import mimetypes
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path as SysPath
from typing import Any

from cog import BasePredictor, Input, Path, Secret

try:
    from cog import CancelationException
except Exception:
    class CancelationException(BaseException):
        pass


DEFAULT_VLOGME_API_URL = "https://vlogme.ai/api/public/v1"
DEFAULT_WATERMARK_TEXT = "Created by VlogMe.AI"
TERMINAL_SUCCESS = {"completed", "complete", "succeeded", "success", "done"}
TERMINAL_FAILURE = {"failed", "failure", "error", "errored", "cancelled", "canceled"}


class _ReplicatePredictionCancelled(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[replicate-avatar-bridge] {message}", flush=True)


def _secret_value(secret: Secret | None) -> str:
    if secret is None:
        return ""
    return (secret.get_secret_value() or "").strip()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _guess_mime(path: SysPath, fallback: str) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or fallback


def _data_uri(path: Path, fallback_mime: str) -> str:
    src = SysPath(str(path))
    mime = _guess_mime(src, fallback_mime)
    encoded = base64.b64encode(src.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _json_request(
    method: str,
    url: str,
    *,
    token: str,
    body: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "vlogme-replicate-avatar-bridge/1.0",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VlogMe API {method} {url} failed: HTTP {exc.code}: {detail}") from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"VlogMe API returned non-object JSON from {url}")
    return parsed


def _download_file(url: str, out_path: SysPath, *, timeout: int = 600) -> SysPath:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "vlogme-replicate-avatar-bridge/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    if not out_path.exists() or out_path.stat().st_size <= 0:
        raise RuntimeError("Downloaded VlogMe output is empty")
    return out_path


def _try_cancel_vlogme_job(api_root: str, token: str, video_id: str, reason: str) -> None:
    if not video_id:
        return
    try:
        _json_request("DELETE", f"{api_root}/videos/{video_id}", token=token, timeout=60)
        _log(f"requested VlogMe cancellation: id={video_id} reason={reason}")
    except Exception as exc:
        _log(f"VlogMe cancellation skipped/failed: id={video_id} reason={reason} error={exc}")


class _VlogMeCancelOnSignal:
    def __init__(self, api_root: str, token: str, video_id: str) -> None:
        self.api_root = api_root
        self.token = token
        self.video_id = video_id
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_VlogMeCancelOnSignal":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for sig, handler in self.previous.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        return False

    def _handle(self, signum: int, _frame: Any) -> None:
        _try_cancel_vlogme_job(
            self.api_root,
            self.token,
            self.video_id,
            f"replicate_signal_{int(signum)}",
        )
        raise _ReplicatePredictionCancelled(f"Replicate prediction cancelled by signal {int(signum)}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        _log("bridge setup complete; no model weights are loaded")

    def predict(
        self,
        avatar_image: Path = Input(description="Face/avatar reference image"),
        audio: Path = Input(description="Speech audio to animate"),
        vlogme_api_token: Secret | None = Input(
            description="VlogMe API token. Use a Secret input or VLOGME_API_TOKEN env var.",
            default=None,
        ),
        vlogme_api_url: str = Input(
            description="VlogMe public API root",
            default=DEFAULT_VLOGME_API_URL,
        ),
        title: str = Input(
            description="Title for the VlogMe render job",
            default="Replicate avatar render",
        ),
        aspect_ratio: str = Input(
            description="Output aspect ratio",
            default="9:16",
            choices=["9:16", "16:9", "1:1"],
        ),
        live_subtitles: bool = Input(
            description="Burn/live subtitle preference for the VlogMe render",
            default=True,
        ),
        face_restore: float = Input(
            description="Optional face restore strength 0.0..1.0. -1 keeps VlogMe defaults.",
            default=-1.0,
        ),
        video_prompt: str = Input(
            description="Optional render prompt override",
            default="",
        ),
        video_negative_prompt: str = Input(
            description="Optional negative prompt override",
            default="",
        ),
        watermark_text: str = Input(
            description="Optional watermark text. Empty uses the VlogMe brand watermark when watermark_enabled is true.",
            default="",
        ),
        watermark_enabled: bool = Input(
            description="Burn a watermark into the final video. Keep true for free generations; paid callers can set false.",
            default=True,
        ),
        timeout_sec: int = Input(
            description="Maximum time to wait for VlogMe to finish",
            default=1740,
        ),
        poll_interval_sec: int = Input(
            description="Polling interval while VlogMe worker renders",
            default=10,
        ),
    ) -> Path:
        token = _secret_value(vlogme_api_token) or os.environ.get("VLOGME_API_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Missing VlogMe API token: pass vlogme_api_token as a Secret input")

        api_root = str(vlogme_api_url or DEFAULT_VLOGME_API_URL).strip().rstrip("/")
        if not api_root:
            raise RuntimeError("vlogme_api_url is empty")

        _log("submitting VlogMe render job")
        create_body: dict[str, Any] = {
            "title": str(title or "Replicate avatar render").strip() or "Replicate avatar render",
            "portrait_base64": _data_uri(avatar_image, "image/jpeg"),
            "audio_base64": _data_uri(audio, "audio/wav"),
            "aspect_ratio": aspect_ratio,
            "live_subtitles": bool(live_subtitles),
        }
        if 0.0 <= float(face_restore) <= 1.0:
            create_body["face_restore"] = float(face_restore)
        if str(video_prompt or "").strip():
            create_body["video_prompt"] = str(video_prompt).strip()
        if str(video_negative_prompt or "").strip():
            create_body["video_negative_prompt"] = str(video_negative_prompt).strip()
        watermark_disable_allowed = _env_flag("VLOGME_BRIDGE_ALLOW_WATERMARK_DISABLE", "0")
        should_apply_watermark = bool(watermark_enabled) or not watermark_disable_allowed
        if not bool(watermark_enabled) and not watermark_disable_allowed:
            _log("watermark disable requested but not allowed for this bridge deployment")
        if should_apply_watermark:
            create_body["watermark_text"] = (
                str(watermark_text or "").strip()
                or os.environ.get("VLOGME_BRIDGE_DEFAULT_WATERMARK_TEXT", "").strip()
                or DEFAULT_WATERMARK_TEXT
            )

        created = _json_request("POST", f"{api_root}/videos", token=token, body=create_body)
        video_id = str(created.get("id") or "").strip()
        if not video_id:
            raise RuntimeError(f"VlogMe create response is missing id: {created!r}")

        _log(
            "VlogMe job accepted: "
            f"id={video_id} status={created.get('status', '')} "
            f"estimated_seconds={created.get('estimated_seconds', '')}"
        )

        deadline = time.time() + max(30, int(timeout_sec or 0))
        poll_interval = max(2, min(60, int(poll_interval_sec or 10)))
        last_status = ""
        last_progress = -1

        try:
            with _VlogMeCancelOnSignal(api_root, token, video_id):
                while time.time() < deadline:
                    status_doc = _json_request("GET", f"{api_root}/videos/{video_id}", token=token)
                    status = str(status_doc.get("status") or "").strip().lower()
                    progress = int(float(status_doc.get("progress") or 0))
                    stage = str(status_doc.get("stage") or "").strip()
                    if status != last_status or progress != last_progress:
                        _log(f"VlogMe status: {status or 'unknown'} progress={progress} stage={stage}")
                        last_status = status
                        last_progress = progress

                    if status in TERMINAL_SUCCESS:
                        video_url = str(status_doc.get("video_url") or "").strip()
                        if not video_url:
                            raise RuntimeError(f"VlogMe completed without video_url: {status_doc!r}")
                        out_path = SysPath("/tmp/vlogme-avatar-bridge/avatar.mp4")
                        _log("downloading completed VlogMe output")
                        return Path(str(_download_file(video_url, out_path)))

                    if status in TERMINAL_FAILURE:
                        message = str(status_doc.get("error_message") or "VlogMe render failed").strip()
                        raise RuntimeError(message or "VlogMe render failed")

                    time.sleep(poll_interval)
        except (KeyboardInterrupt, CancelationException, _ReplicatePredictionCancelled):
            _try_cancel_vlogme_job(api_root, token, video_id, "replicate_cancelled")
            raise

        _try_cancel_vlogme_job(api_root, token, video_id, "replicate_bridge_timeout")
        raise RuntimeError(f"Timed out waiting for VlogMe render {video_id}")
