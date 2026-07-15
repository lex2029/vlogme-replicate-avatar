# Creating VlogMe Jobs From the Replicate Bridge

This is the canonical contract for creating a VlogMe render from the public
Replicate bridge or another service that must behave exactly like that bridge.
The working implementation is `run_bridge.py`.

## Why Jobs Can Wait for the Wrong Fleet

VlogMe classifies this channel from the top-level request field:

```json
"replicate_free": true
```

The value must be the JSON boolean `true`. If it is missing, `false`, the string
`"true"`, nested under another object, or replaced with
`"source": "replicate_free"`, VlogMe treats the request as a normal API render
with source `api_v1`. Such a job can wait on the standard private render fleet,
including the private B300 path. Callers must not set `source` themselves.

## Create Request

Send:

```text
POST https://vlogme.ai/api/public/v1/videos
```

Required headers:

```text
Authorization: Bearer <VLOGME_API_TOKEN>
Accept: application/json
Content-Type: application/json
Idempotency-Key: <one UUID per logical generation>
```

Use the same `Idempotency-Key` when retrying the same generation after a timeout
or lost response. VlogMe then returns the original render instead of creating or
charging another one. Generate a new key only for a genuinely new generation.

The known-good bridge body is:

```json
{
  "title": "Replicate avatar render",
  "portrait_base64": "data:image/jpeg;base64,<BASE64_IMAGE>",
  "audio_base64": "data:audio/wav;base64,<BASE64_AUDIO>",
  "aspect_ratio": "9:16",
  "live_subtitles": true,
  "watermark_text": "Created by VlogMe.AI",
  "replicate_free": true
}
```

Important details:

- Supply exactly one portrait source: `portrait_base64` or `portrait_url`.
- Supply prerecorded speech as `audio_base64` or `audio_url`. The Replicate
  bridge trims it to at most 10 seconds before this request.
- Base64 values should be complete data URIs, including the MIME prefix.
- `aspect_ratio` is `9:16` for the public bridge.
- `live_subtitles` and `replicate_free` are booleans, not strings or integers.
- Do not send `source`, worker IDs, hardware names, B300 settings, provider
  names, or internal queue fields. VlogMe owns downstream routing.

## Python Reference

```python
import base64
import mimetypes
import time
import uuid
from pathlib import Path

import requests


API_ROOT = "https://vlogme.ai/api/public/v1"
TOKEN = "<read VLOGME_API_TOKEN from the secret store>"


def data_uri(path: Path, fallback_mime: str) -> str:
    mime = mimetypes.guess_type(path.name)[0] or fallback_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


idempotency_key = str(uuid.uuid4())
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Idempotency-Key": idempotency_key,
}
body = {
    "title": "Replicate avatar render",
    "portrait_base64": data_uri(Path("avatar.jpg"), "image/jpeg"),
    "audio_base64": data_uri(Path("speech.wav"), "audio/wav"),
    "aspect_ratio": "9:16",
    "live_subtitles": True,
    "watermark_text": "Created by VlogMe.AI",
    "replicate_free": True,
}

# If this call times out before a response is received, retry it with the same
# idempotency_key and the same logical payload.
response = requests.post(
    f"{API_ROOT}/videos",
    headers=headers,
    json=body,
    timeout=120,
)
response.raise_for_status()
video_id = response.json()["id"]

deadline = time.time() + 1740
while time.time() < deadline:
    status_response = requests.get(
        f"{API_ROOT}/videos/{video_id}",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        },
        timeout=120,
    )
    status_response.raise_for_status()
    video = status_response.json()
    status = str(video.get("status") or "").lower()

    if status in {"completed", "complete", "succeeded", "success", "done"}:
        video_url = video.get("video_url")
        if not video_url:
            raise RuntimeError("VlogMe completed without video_url")
        print(video_url)
        break

    if status in {"failed", "failure", "error", "errored", "cancelled", "canceled"}:
        raise RuntimeError(video.get("error_message") or "VlogMe render failed")

    time.sleep(10)
else:
    requests.delete(
        f"{API_ROOT}/videos/{video_id}",
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=60,
    )
    raise TimeoutError(f"Timed out waiting for VlogMe render {video_id}")
```

Do not hardcode or commit the token. In this repository, the deployed bridge
reads `VLOGME_API_TOKEN` from the environment or from the private runtime token
file created during the authenticated GitHub publishing workflow.

## Expected Responses

Successful creation returns HTTP `202` with an object containing at least:

```json
{
  "id": "<VLOGME_VIDEO_UUID>",
  "status": "preparing",
  "estimated_seconds": 10
}
```

Poll `GET /videos/<id>` with the same bearer token every 10 seconds. Expected
progression is similar to:

```text
preparing / transcribing_audio
preparing / generating_avatar_clip
running / finalize_start
completed
```

On `completed`, download `video_url` immediately. It is a signed URL and can
expire; call `GET /videos/<id>` again to receive a fresh URL.

## Capacity and Error Handling

- HTTP `429` with code `rate_limited` means the public Replicate channel is
  busy or still inside its one-hour cooldown. Do not create a normal API job as
  a fallback. Return the overload error to Replicate.
- HTTP `400` mentioning `Idempotency-Key` means the required header is absent.
- HTTP `400 invalid_input` means the body shape, MIME data URI, or field type is
  wrong. Log the server error body without logging tokens or base64 payloads.
- HTTP `402` means the VlogMe account behind the API token has insufficient
  credits.
- If the caller cancels after receiving a VlogMe ID, send
  `DELETE /videos/<id>` with the same bearer token.
- Never start a replacement render merely because polling timed out. First
  retry or query the existing ID; retries of the create POST must retain the
  original `Idempotency-Key`.

## Diagnosis Checklist

When another service says that its job is waiting for a personal B300, inspect
the actual outbound JSON before anything else:

1. Confirm `replicate_free` exists at the top level and is the boolean `true`.
2. Confirm the service calls `/api/public/v1/videos`, not an older or internal
   endpoint.
3. Confirm `Idempotency-Key` is present and stable for retries.
4. Confirm the response VlogMe ID is polled, instead of creating another job on
   every poll or timeout.
5. Confirm the service does not add `source`, provider, hardware, worker, or
   B300 routing fields.
6. Compare its payload with `run_bridge.py` and run the hosted E2E workflow.

## Last Known-Good Verification

Verified end to end on 2026-07-15:

- GitHub Actions run: `29438152513`
- Replicate prediction: `8j5cc216rxrmw0czcvkv2pde80`
- VlogMe video: `49d86857-638b-469f-b098-f8eb3169425d`
- Result: `vlogme_ai.mp4`, valid 8.000-second MP4

The hosted test workflow is
`.github/workflows/test-replicate-bridge-prediction.yml`.
