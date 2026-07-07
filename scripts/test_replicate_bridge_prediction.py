from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


API_ROOT = "https://api.replicate.com/v1"


def _request(method: str, url: str, *, token: str, body: dict | None = None) -> dict:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "vlogme-replicate-avatar-bridge-test/1.0",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate API {method} {url} failed: HTTP {exc.code}: {detail}") from exc


def _try_cancel(cancel_url: str, *, token: str, prediction_id: str) -> None:
    if not cancel_url:
        return
    try:
        _request("POST", cancel_url, token=token)
        print(f"Cancel requested for prediction {prediction_id}")
    except Exception as exc:
        print(f"Warning: failed to cancel prediction {prediction_id}: {exc}", file=sys.stderr)


def _data_uri(path: Path, fallback_mime: str) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or fallback_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _make_smoke_audio(path: Path, *, seconds: float = 8.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            envelope = 0.45 + 0.15 * math.sin(2 * math.pi * 2.2 * t)
            carrier = (
                0.50 * math.sin(2 * math.pi * 155.0 * t)
                + 0.35 * math.sin(2 * math.pi * 232.0 * t)
                + 0.15 * math.sin(2 * math.pi * 310.0 * t)
            )
            sample = int(max(-1.0, min(1.0, envelope * carrier)) * 32767)
            wav.writeframesraw(sample.to_bytes(2, "little", signed=True))


def _latest_version_id(model_name: str, *, token: str) -> str:
    versions = _request("GET", f"{API_ROOT}/models/{model_name}/versions", token=token)
    results = versions.get("results", []) if isinstance(versions, dict) else versions
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"No versions found for model {model_name}")
    version_id = str(results[0].get("id") or "").strip()
    if not version_id:
        raise RuntimeError(f"Latest version has no id: {results[0]!r}")
    return version_id


def _tail_logs(prediction: dict, max_chars: int = 8000) -> str:
    logs = str(prediction.get("logs") or "")
    return logs if len(logs) <= max_chars else logs[-max_chars:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a smoke prediction on the Replicate VlogMe bridge.")
    parser.add_argument("--model", default="lex2029/vlogme-avatar-bridge")
    parser.add_argument("--deployment", default="")
    parser.add_argument("--image", default="runtime/SmartBlog-Live/assets/ref_user_photo.jpg")
    parser.add_argument("--audio", default="")
    parser.add_argument("--audio-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--vlogme-api-url", default="https://vlogme.ai/api/public/v1")
    parser.add_argument("--title", default="Replicate bridge smoke test")
    parser.add_argument("--aspect-ratio", default="9:16", choices=["9:16", "16:9", "1:1"])
    parser.add_argument("--face-restore", type=float, default=-1.0)
    parser.add_argument("--live-subtitles", type=int, default=1)
    args = parser.parse_args()

    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not replicate_token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN")
    vlogme_token = os.environ.get("VLOGME_API_TOKEN", "").strip()
    if not vlogme_token:
        raise RuntimeError("Missing VLOGME_API_TOKEN")

    root = Path(__file__).resolve().parents[1]
    image_path = (root / args.image).resolve()
    audio_path = (root / args.audio).resolve() if args.audio.strip() else root / "tmp" / "replicate-bridge-smoke.wav"
    if not image_path.exists():
        raise RuntimeError(f"Missing smoke image: {image_path}")
    if args.audio.strip():
        if not audio_path.exists():
            raise RuntimeError(f"Missing smoke audio: {audio_path}")
    else:
        _make_smoke_audio(audio_path, seconds=float(args.audio_seconds))

    payload = {
        "input": {
            "avatar_image": _data_uri(image_path, "image/jpeg"),
            "audio": _data_uri(audio_path, "audio/wav"),
            "vlogme_api_token": vlogme_token,
            "vlogme_api_url": str(args.vlogme_api_url).strip(),
            "title": str(args.title).strip() or "Replicate bridge smoke test",
            "aspect_ratio": args.aspect_ratio,
            "live_subtitles": bool(args.live_subtitles),
            "face_restore": float(args.face_restore),
        },
    }
    if args.deployment:
        create_url = f"{API_ROOT}/deployments/{args.deployment}/predictions"
        print(f"Using deployment {args.deployment}")
    else:
        print(f"Resolving latest version for {args.model}")
        version_id = _latest_version_id(args.model, token=replicate_token)
        print(f"Using version {version_id[:12]}...")
        payload["version"] = version_id
        create_url = f"{API_ROOT}/predictions"

    prediction = _request("POST", create_url, token=replicate_token, body=payload)
    prediction_id = prediction.get("id")
    get_url = (prediction.get("urls") or {}).get("get")
    cancel_url = (prediction.get("urls") or {}).get("cancel")
    if not prediction_id or not get_url:
        raise RuntimeError(f"Prediction create response is missing id/get URL: {prediction!r}")
    print(f"Prediction started: {prediction_id}")

    started_at = time.time()
    deadline = started_at + int(args.timeout_sec)
    last_status = ""
    try:
        while time.time() < deadline:
            prediction = _request("GET", get_url, token=replicate_token)
            status = str(prediction.get("status") or "")
            elapsed = time.time() - started_at
            if status != last_status:
                print(f"Status: {status} after {elapsed:.0f}s")
                last_status = status
            if status in {"succeeded", "failed", "canceled"}:
                print(
                    json.dumps(
                        {
                            "id": prediction.get("id"),
                            "status": status,
                            "output": prediction.get("output"),
                            "error": prediction.get("error"),
                            "metrics": prediction.get("metrics"),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                logs = _tail_logs(prediction)
                if logs:
                    print("Log tail:")
                    print(logs)
                return 0 if status == "succeeded" else 1
            time.sleep(max(2, int(args.poll_sec)))
    except KeyboardInterrupt:
        _try_cancel(str(cancel_url or ""), token=replicate_token, prediction_id=str(prediction_id))
        raise

    print(f"Timed out waiting for prediction {prediction_id}", file=sys.stderr)
    _try_cancel(str(cancel_url or ""), token=replicate_token, prediction_id=str(prediction_id))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
