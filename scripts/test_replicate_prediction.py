from __future__ import annotations

import argparse
import base64
import json
import math
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
        "User-Agent": "vlogme-replicate-avatar-smoke-test/1.0",
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


def _data_uri(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _make_smoke_audio(path: Path, *, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            envelope = 0.35 + 0.25 * math.sin(2 * math.pi * 3.0 * t)
            carrier = (
                0.55 * math.sin(2 * math.pi * 170.0 * t)
                + 0.30 * math.sin(2 * math.pi * 255.0 * t)
                + 0.15 * math.sin(2 * math.pi * 380.0 * t)
            )
            sample = int(max(-1.0, min(1.0, envelope * carrier)) * 32767)
            wav.writeframesraw(sample.to_bytes(2, "little", signed=True))


def _latest_version_id(model_name: str, *, token: str) -> str:
    versions = _request("GET", f"{API_ROOT}/models/{model_name}/versions", token=token)
    if isinstance(versions, dict) and isinstance(versions.get("results"), list):
        results = versions["results"]
    elif isinstance(versions, list):
        results = versions
    else:
        raise RuntimeError(f"Unexpected versions response: {versions!r}")
    if not results:
        raise RuntimeError(f"No versions found for model {model_name}")
    version_id = str(results[0].get("id") or "").strip()
    if not version_id:
        raise RuntimeError(f"Latest version has no id: {results[0]!r}")
    return version_id


def _tail_logs(prediction: dict, max_chars: int = 4000) -> str:
    logs = str(prediction.get("logs") or "")
    if len(logs) <= max_chars:
        return logs
    return logs[-max_chars:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a smoke prediction on the Replicate avatar model.")
    parser.add_argument("--model", default="lex2029/vlogme-avatar")
    parser.add_argument("--deployment", default="")
    parser.add_argument("--image", default="runtime/SmartBlog-Live/assets/ref_user_photo.jpg")
    parser.add_argument("--timeout-sec", type=int, default=5400)
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN GitHub secret")

    root = Path(__file__).resolve().parents[1]
    image_path = (root / args.image).resolve()
    audio_path = root / "tmp" / "replicate-smoke-speech.wav"
    if not image_path.exists():
        raise RuntimeError(f"Missing smoke image: {image_path}")
    _make_smoke_audio(audio_path)

    payload = {
        "input": {
            "avatar_image": _data_uri(image_path, "image/jpeg"),
            "audio": _data_uri(audio_path, "audio/wav"),
        },
    }
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        payload["input"]["hf_token"] = hf_token
    if args.deployment:
        create_url = f"{API_ROOT}/deployments/{args.deployment}/predictions"
        print(f"Using deployment {args.deployment}")
    else:
        print(f"Resolving latest version for {args.model}")
        version_id = _latest_version_id(args.model, token=token)
        print(f"Using version {version_id[:12]}...")
        payload["version"] = version_id
        create_url = f"{API_ROOT}/predictions"

    prediction = _request(
        "POST",
        create_url,
        token=token,
        body=payload,
    )
    prediction_id = prediction.get("id")
    get_url = (prediction.get("urls") or {}).get("get")
    if not prediction_id or not get_url:
        raise RuntimeError(f"Prediction create response is missing id/get URL: {prediction!r}")
    print(f"Prediction started: {prediction_id}")

    deadline = time.time() + args.timeout_sec
    last_status = ""
    while time.time() < deadline:
        prediction = _request("GET", get_url, token=token)
        status = str(prediction.get("status") or "")
        if status != last_status:
            print(f"Status: {status}")
            last_status = status
        if status in {"succeeded", "failed", "canceled"}:
            print(json.dumps(
                {
                    "id": prediction.get("id"),
                    "status": status,
                    "output": prediction.get("output"),
                    "error": prediction.get("error"),
                    "metrics": prediction.get("metrics"),
                },
                indent=2,
                ensure_ascii=False,
            ))
            logs = _tail_logs(prediction)
            if logs:
                print("Log tail:")
                print(logs)
            return 0 if status == "succeeded" else 1
        time.sleep(args.poll_sec)

    print(f"Timed out waiting for prediction {prediction_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
