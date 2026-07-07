from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://api.replicate.com/v1"


def _request(method: str, url: str, *, token: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "vlogme-replicate-model-metadata/1.0",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate API {method} {url} failed: HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Replicate API returned non-object JSON: {parsed!r}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Replicate model metadata.")
    parser.add_argument("--model", default="lex2029/vlogme-avatar-bridge")
    parser.add_argument(
        "--description",
        default="Create a vertical talking-avatar video from any centered photo and speech audio.",
    )
    parser.add_argument("--readme", default="docs/replicate-model-readme.md")
    parser.add_argument("--github-url", default="https://github.com/lex2029/vlogme-replicate-avatar")
    parser.add_argument(
        "--visibility",
        choices=["", "public", "private"],
        default="",
        help="Optional model visibility. Replicate may require changing this in the web UI.",
    )
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN")

    owner, _, name = str(args.model).partition("/")
    if not owner or not name:
        raise RuntimeError("--model must be in owner/name format")

    readme_path = Path(args.readme)
    readme = readme_path.read_text(encoding="utf-8")
    body = {
        "description": str(args.description).strip(),
        "readme": readme,
        "github_url": str(args.github_url).strip(),
    }
    if str(args.visibility or "").strip():
        body["visibility"] = str(args.visibility).strip()

    updated = _request(
        "PATCH",
        f"{API_ROOT}/models/{owner}/{name}",
        token=token,
        body=body,
    )
    print(
        json.dumps(
            {
                "url": updated.get("url"),
                "owner": updated.get("owner"),
                "name": updated.get("name"),
                "visibility": updated.get("visibility"),
                "description": updated.get("description"),
                "github_url": updated.get("github_url"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
