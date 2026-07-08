from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request


REPLICATE_API_ROOT = "https://api.replicate.com/v1"


def _request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> tuple[int, str]:
    req_headers = {
        "Accept": "application/json",
        "User-Agent": "vlogme-replicate-webhook-auth-test/1.0",
        **(headers or {}),
    }
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = body.encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _replicate_webhook_secret(token: str) -> str:
    status, raw = _request(
        "GET",
        f"{REPLICATE_API_ROOT}/webhooks/default/secret",
        token=token,
    )
    if status != 200:
        raise RuntimeError(f"Could not fetch Replicate webhook secret: HTTP {status}: {raw}")
    parsed = json.loads(raw)
    secret = str(parsed.get("key") or "").strip()
    if not secret:
        raise RuntimeError("Replicate webhook secret response did not contain key")
    return secret


def _signed_headers(secret: str, body: str) -> dict[str, str]:
    webhook_id = f"msg_codex_{int(time.time() * 1000)}"
    timestamp = str(int(time.time()))
    key_part = secret.removeprefix("whsec_")
    digest = hmac.new(
        base64.b64decode(key_part),
        f"{webhook_id}.{timestamp}.{body}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = base64.b64encode(digest).decode("ascii")
    return {
        "webhook-id": webhook_id,
        "webhook-timestamp": timestamp,
        "webhook-signature": f"v1,{signature}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify VlogMe's Replicate webhook endpoint auth.")
    parser.add_argument(
        "--endpoint",
        default="https://vlogme.ai/api/public/v1/replicate/webhook",
    )
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN")

    body = json.dumps(
        {
            "id": "codex-webhook-auth-smoke",
            "status": "starting",
            "model": "lex2029/vlogme-avatar-bridge",
            "deployment": "lex2029/vlogme-avatar-bridge-cpu",
            "logs": "",
        },
        separators=(",", ":"),
    )

    secret = _replicate_webhook_secret(token)
    good_status, good_body = _request(
        "POST",
        args.endpoint,
        body=body,
        headers=_signed_headers(secret, body),
    )
    print(f"Signed webhook POST returned HTTP {good_status}: {good_body[:500]}")
    if good_status != 200:
        return 1

    bad_headers = _signed_headers(secret, body)
    bad_headers["webhook-signature"] = "v1,invalid"
    bad_status, bad_body = _request("POST", args.endpoint, body=body, headers=bad_headers)
    print(f"Invalid-signature webhook POST returned HTTP {bad_status}: {bad_body[:500]}")
    if bad_status != 401:
        print("Expected invalid signature to be rejected with HTTP 401", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
