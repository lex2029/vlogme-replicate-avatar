from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


API_ROOT = "https://api.replicate.com/v1"
ACTIVE_STATUSES = {"starting", "processing"}


def _request(method: str, url: str, *, token: str, body: dict | None = None) -> dict:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "vlogme-replicate-active-cancel/1.0",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate API {method} {url} failed: HTTP {exc.code}: {detail}") from exc


def _matches(prediction: dict, *, model: str, deployment: str) -> bool:
    if model and str(prediction.get("model") or "").strip() != model:
        return False
    if deployment and str(prediction.get("deployment") or "").strip() != deployment:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Cancel active Replicate predictions for a model/deployment.")
    parser.add_argument("--model", default="")
    parser.add_argument("--deployment", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN")
    if not args.model and not args.deployment:
        raise RuntimeError("Pass --model, --deployment, or both")

    url = f"{API_ROOT}/predictions"
    seen = 0
    canceled = []
    matched_active = []
    while url and seen < max(1, int(args.limit)):
        page = _request("GET", url, token=token)
        for prediction in page.get("results", []) or []:
            seen += 1
            if seen > int(args.limit):
                break
            status = str(prediction.get("status") or "").strip().lower()
            if status not in ACTIVE_STATUSES:
                continue
            if not _matches(prediction, model=args.model, deployment=args.deployment):
                continue
            prediction_id = str(prediction.get("id") or "")
            matched_active.append(prediction_id)
            cancel_url = str((prediction.get("urls") or {}).get("cancel") or "")
            if args.dry_run:
                continue
            if cancel_url:
                _request("POST", cancel_url, token=token)
                canceled.append(prediction_id)
        next_url = str(page.get("next") or "")
        url = next_url if next_url and seen < int(args.limit) else ""

    print(
        json.dumps(
            {
                "scanned": seen,
                "matched_active": matched_active,
                "canceled": canceled,
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
