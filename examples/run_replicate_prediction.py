from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://api.replicate.com/v1"
DEFAULT_MODEL = "lex2029/vlogme-avatar-bridge"
TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


def request_json(
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
        "User-Agent": "vlogme-replicate-example/1.0",
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
        raise RuntimeError(f"Replicate API {method} {url} failed: HTTP {exc.code}: {detail}") from exc

    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Replicate API returned non-object JSON: {parsed!r}")
    return parsed


def data_uri(path: Path, fallback_mime: str) -> str:
    mime = mimetypes.guess_type(str(path))[0] or fallback_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def latest_version_id(model: str, *, token: str) -> str:
    versions = request_json("GET", f"{API_ROOT}/models/{model}/versions", token=token)
    results = versions.get("results", [])
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"No versions found for {model}")
    version = str(results[0].get("id") or "").strip()
    if not version:
        raise RuntimeError(f"Latest version has no id: {results[0]!r}")
    return version


def resolve_file(path_s: str) -> Path:
    path = Path(path_s).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public VlogMe Avatar Replicate model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default="test_assets/friendly_ai_presenter.jpg")
    parser.add_argument("--audio", default="test_assets/presenter_8s.wav")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-sec", type=int, default=10)
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set REPLICATE_API_TOKEN before running this example")

    image_path = resolve_file(args.image)
    audio_path = resolve_file(args.audio)

    print(f"Resolving latest version for {args.model}...")
    version = latest_version_id(str(args.model), token=token)

    payload = {
        "version": version,
        "input": {
            "avatar_image": data_uri(image_path, "image/jpeg"),
            "audio": data_uri(audio_path, "audio/wav"),
            "live_subtitles": not bool(args.no_subtitles),
        },
    }

    prediction = request_json("POST", f"{API_ROOT}/predictions", token=token, body=payload)
    prediction_id = str(prediction.get("id") or "").strip()
    get_url = str((prediction.get("urls") or {}).get("get") or "").strip()
    if not prediction_id or not get_url:
        raise RuntimeError(f"Prediction response is missing id/get URL: {prediction!r}")

    print(f"Prediction: https://replicate.com/p/{prediction_id}")

    deadline = time.time() + max(60, int(args.timeout_sec))
    last_status = ""
    while time.time() < deadline:
        prediction = request_json("GET", get_url, token=token)
        status = str(prediction.get("status") or "").strip()
        if status != last_status:
            print(f"Status: {status or 'unknown'}")
            last_status = status

        if status in TERMINAL_STATUSES:
            if status == "succeeded":
                print(f"Output: {prediction.get('output')}")
                return 0
            logs = str(prediction.get("logs") or "")
            if logs:
                print("\nLog tail:\n" + logs[-4000:], file=sys.stderr)
            print(f"Prediction ended with status={status}: {prediction.get('error')}", file=sys.stderr)
            return 1

        time.sleep(max(2, int(args.poll_sec)))

    raise RuntimeError(f"Timed out waiting for prediction {prediction_id}")


if __name__ == "__main__":
    raise SystemExit(main())
