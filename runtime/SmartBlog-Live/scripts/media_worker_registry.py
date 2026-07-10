#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_env_files() -> None:
    root = Path(os.getenv("SMARTBLOG_APP_DIR") or Path(__file__).resolve().parents[1]).resolve()
    for path in (
        root / "config" / "worker_secrets.conf",
        root / "config" / "worker_secrets.local.conf",
    ):
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = value.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[media-registry] env file ignored: {path} err={e}", file=sys.stderr, flush=True)


_load_env_files()


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _root_dir() -> Path:
    return Path(_env("SMARTBLOG_APP_DIR") or Path(__file__).resolve().parents[1]).resolve()


def _worker_api_url() -> str:
    return (
        _env("VLOGME_WORKER_API_URL")
        or _env("SMARTBLOG_WORKER_API_URL")
        or "https://vlogme.ai/api/public/v1/worker-api"
    ).rstrip("/")


def _worker_api_key() -> str:
    return _env("WORKER_API_KEY") or _env("VLOGME_WORKER_API_KEY") or _env("POSTPROCESSING_WORKER_TOKEN")


def _json_call(payload: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    key = _worker_api_key()
    if not key:
        raise RuntimeError("WORKER_API_KEY/VLOGME_WORKER_API_KEY/POSTPROCESSING_WORKER_TOKEN is required")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _worker_api_url(),
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; VlogMeMediaWorker/1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return json.loads(data or "{}")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker-api HTTP {e.code}: {text[-1000:]}") from e


def _runpod_proxy_url(port: int, suffix: str = "") -> str:
    explicit_base = _env("VLOGME_MEDIA_PUBLIC_BASE") or _env("SMARTBLOG_MEDIA_PUBLIC_BASE") or _env("RUNPOD_PUBLIC_BASE_URL")
    if explicit_base:
        base = explicit_base.rstrip("/")
        if "{port}" in base:
            base = base.format(port=int(port))
        return f"{base}{suffix}"
    pod_id = _env("RUNPOD_POD_ID") or _env("RUNPOD_ENDPOINT_ID") or _env("SMARTBLOG_RUNPOD_POD_ID")
    if pod_id:
        return f"https://{pod_id}-{int(port)}.proxy.runpod.net{suffix}"
    return ""


def _default_worker_id() -> str:
    return (
        _env("VLOGME_MEDIA_WORKER_ID")
        or _env("SMARTBLOG_MEDIA_WORKER_ID")
        or _env("RUNPOD_POD_ID")
        or _env("RUNPOD_ENDPOINT_ID")
        or _env("HOSTNAME")
        or socket.gethostname()
        or "media-worker"
    )


def _media_urls() -> dict[str, str]:
    hunyuan = (
        _env("VLOGME_PUBLIC_HUNYUAN_URL")
        or _env("SMARTBLOG_PUBLIC_HUNYUAN_URL")
        or _env("SMARTBLOG_HUNYUAN_PUBLIC_URL")
        or _runpod_proxy_url(8798)
    )
    mmaudio = (
        _env("VLOGME_PUBLIC_MMAUDIO_URL")
        or _env("SMARTBLOG_PUBLIC_MMAUDIO_URL")
        or _env("SMARTBLOG_MMAUDIO_PUBLIC_URL")
        or _runpod_proxy_url(8799)
    )
    finalizer = (
        _env("VLOGME_PUBLIC_FINALIZER_URL")
        or _env("VLOGME_PUBLIC_UPSCALE_URL")
        or _env("SMARTBLOG_PUBLIC_FINALIZER_URL")
        or _env("SMARTBLOG_PUBLIC_UPSCALE_URL")
        or _env("SMARTBLOG_FILE_UPSCALE_PUBLIC_URL")
        or _runpod_proxy_url(8888, "/upscale")
    )
    videoedit = (
        _env("VLOGME_PUBLIC_VIDEOEDIT_URL")
        or _env("SMARTBLOG_PUBLIC_VIDEOEDIT_URL")
        or finalizer
    )
    videoupscale = (
        _env("VLOGME_PUBLIC_VIDEOUPSCALE_URL")
        or _env("SMARTBLOG_PUBLIC_VIDEOUPSCALE_URL")
        or finalizer
    )
    videointerpolate = (
        _env("VLOGME_PUBLIC_VIDEOINTERPOLATE_URL")
        or _env("SMARTBLOG_PUBLIC_VIDEOINTERPOLATE_URL")
        or finalizer
    )
    videomatte = (
        _env("VLOGME_PUBLIC_VIDEOMATTE_URL")
        or _env("SMARTBLOG_PUBLIC_VIDEOMATTE_URL")
        or videoedit
    )
    musetalk = (
        _env("VLOGME_PUBLIC_MUSETALK_URL")
        or _env("SMARTBLOG_PUBLIC_MUSETALK_URL")
        or _env("SMARTBLOG_MUSETALK_PUBLIC_URL")
        or _runpod_proxy_url(8800)
    )
    urls = {
        "hunyuan": hunyuan,
        "ltx": hunyuan,
        "mmaudio": mmaudio,
        "finalizer": finalizer,
        "upscale": finalizer,
        "videoedit": videoedit,
        "videoupscale": videoupscale,
        "videointerpolate": videointerpolate,
        "videomatte": videomatte,
    }
    if musetalk and _truthy(_env("SMARTBLOG_MUSETALK_SERVICE_ENABLED"), default=False):
        urls["musetalk"] = musetalk
        urls["lipsync"] = musetalk
    return {k: v.rstrip("/") if not v.endswith("/upscale") else v for k, v in urls.items() if v}


def register_once(args: argparse.Namespace) -> dict[str, Any]:
    urls = _media_urls()
    if not urls:
        raise RuntimeError(
            "No public media URLs. Set RUNPOD_POD_ID or VLOGME_PUBLIC_HUNYUAN_URL/"
            "VLOGME_PUBLIC_FINALIZER_URL explicitly."
        )
    payload = {
        "action": "register_media_worker",
        "worker_id": args.worker_id or _default_worker_id(),
        "role": args.role,
        "status": args.status,
        "priority": int(args.priority),
        "ttl_seconds": int(args.ttl_seconds),
        "urls": urls,
        "capabilities": {
            "hunyuan": bool(urls.get("hunyuan")),
            "mmaudio": bool(urls.get("mmaudio")),
            "finalizer": bool(urls.get("finalizer") or urls.get("upscale")),
            "videoedit": bool(urls.get("videoedit")),
            "videoupscale": bool(urls.get("videoupscale")),
            "videointerpolate": bool(urls.get("videointerpolate")),
            "videomatte": bool(urls.get("videomatte")),
            "musetalk": bool(urls.get("musetalk")),
        },
        "metadata": {
            "hostname": socket.gethostname(),
            "runpod_pod_id": _env("RUNPOD_POD_ID") or None,
            "git_sha": _git_sha(),
        },
    }
    return _json_call(payload, timeout=float(args.timeout))


def _git_sha() -> str | None:
    head = _root_dir() / ".git" / "HEAD"
    try:
        if not head.exists():
            return None
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref: "):
            ref_path = _root_dir() / ".git" / text[5:].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
        return text[:12]
    except Exception:
        return None


def _pool_file() -> Path:
    configured = (
        _env("VLOGME_MEDIA_WORKER_POOL_FILE")
        or _env("SMARTBLOG_MEDIA_WORKER_POOL_FILE")
        or _env("SMARTBLOG_HUNYUAN_WORKER_POOL_FILE")
    )
    if configured:
        return Path(configured).resolve()
    return _root_dir() / "runtime" / "media_worker_pool.json"


def refresh_once(args: argparse.Namespace) -> dict[str, Any]:
    data = _json_call(
        {
            "action": "media_worker_pool",
            "role": args.role if args.role_filter else None,
            "ttl_seconds": int(args.ttl_seconds),
            "include_stale": bool(args.include_stale),
        },
        timeout=float(args.timeout),
    )
    workers = data.get("workers")
    if not isinstance(workers, list):
        pool = data.get("pool") if isinstance(data.get("pool"), dict) else {}
        workers = pool.get("workers") if isinstance(pool.get("workers"), list) else []
    out = {
        "schema": "vlogme.media_worker_pool.v1",
        "generated_at": data.get("generated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _worker_api_url(),
        "workers": workers,
    }
    path = _pool_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as f:
        json.dump(out, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
        tmp_name = f.name
    os.replace(tmp_name, path)
    return out


def loop(kind: str, args: argparse.Namespace) -> None:
    interval = float(args.interval)
    failure_sleep = float(args.failure_sleep)
    while True:
        try:
            if kind == "register":
                result = register_once(args)
                worker = result.get("worker") if isinstance(result, dict) else {}
                print(
                    f"[media-registry] registered worker={worker.get('worker_id') or args.worker_id or _default_worker_id()} "
                    f"urls={','.join(sorted((_media_urls()).keys()))}",
                    flush=True,
                )
            else:
                result = refresh_once(args)
                print(
                    f"[media-registry] refreshed pool={_pool_file()} workers={len(result.get('workers') or [])}",
                    flush=True,
                )
            time.sleep(interval)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[media-registry] {kind} failed: {e}", file=sys.stderr, flush=True)
            time.sleep(failure_sleep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register/refresh VlogMe RTX media workers through VlogMe worker-api.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--role", default=_env("VLOGME_MEDIA_WORKER_ROLE") or _env("SMARTBLOG_MEDIA_WORKER_ROLE", "rtxpro6000-media"))
        p.add_argument("--ttl-seconds", type=int, default=int((_env("VLOGME_MEDIA_REGISTRY_TTL_SEC") or _env("SMARTBLOG_MEDIA_REGISTRY_TTL_SEC", "120")) or 120))
        p.add_argument("--timeout", type=float, default=float((_env("VLOGME_MEDIA_REGISTRY_TIMEOUT_SEC") or _env("SMARTBLOG_MEDIA_REGISTRY_TIMEOUT_SEC", "20")) or 20))
        p.add_argument("--interval", type=float, default=float((_env("VLOGME_MEDIA_REGISTRY_INTERVAL_SEC") or _env("SMARTBLOG_MEDIA_REGISTRY_INTERVAL_SEC", "30")) or 30))
        p.add_argument("--failure-sleep", type=float, default=float((_env("VLOGME_MEDIA_REGISTRY_FAILURE_SLEEP_SEC") or _env("SMARTBLOG_MEDIA_REGISTRY_FAILURE_SLEEP_SEC", "10")) or 10))

    p = sub.add_parser("register-once")
    add_common(p)
    p.add_argument("--worker-id", default=_default_worker_id())
    p.add_argument("--status", default=_env("VLOGME_MEDIA_WORKER_STATUS") or _env("SMARTBLOG_MEDIA_WORKER_STATUS", "ready"))
    p.add_argument("--priority", type=int, default=int((_env("VLOGME_MEDIA_WORKER_PRIORITY") or _env("SMARTBLOG_MEDIA_WORKER_PRIORITY", "100")) or 100))

    p = sub.add_parser("register-loop")
    add_common(p)
    p.add_argument("--worker-id", default=_default_worker_id())
    p.add_argument("--status", default=_env("VLOGME_MEDIA_WORKER_STATUS") or _env("SMARTBLOG_MEDIA_WORKER_STATUS", "ready"))
    p.add_argument("--priority", type=int, default=int((_env("VLOGME_MEDIA_WORKER_PRIORITY") or _env("SMARTBLOG_MEDIA_WORKER_PRIORITY", "100")) or 100))

    p = sub.add_parser("refresh-once")
    add_common(p)
    p.add_argument("--include-stale", action="store_true", default=_truthy(_env("VLOGME_MEDIA_REGISTRY_INCLUDE_STALE") or _env("SMARTBLOG_MEDIA_REGISTRY_INCLUDE_STALE"), False))
    p.add_argument("--role-filter", action="store_true", default=_truthy(_env("VLOGME_MEDIA_REGISTRY_ROLE_FILTER") or _env("SMARTBLOG_MEDIA_REGISTRY_ROLE_FILTER"), False))

    p = sub.add_parser("refresh-loop")
    add_common(p)
    p.add_argument("--include-stale", action="store_true", default=_truthy(_env("VLOGME_MEDIA_REGISTRY_INCLUDE_STALE") or _env("SMARTBLOG_MEDIA_REGISTRY_INCLUDE_STALE"), False))
    p.add_argument("--role-filter", action="store_true", default=_truthy(_env("VLOGME_MEDIA_REGISTRY_ROLE_FILTER") or _env("SMARTBLOG_MEDIA_REGISTRY_ROLE_FILTER"), False))

    args = parser.parse_args(argv)
    if args.cmd == "register-once":
        print(json.dumps(register_once(args), ensure_ascii=True, sort_keys=True))
    elif args.cmd == "register-loop":
        loop("register", args)
    elif args.cmd == "refresh-once":
        print(json.dumps(refresh_once(args), ensure_ascii=True, sort_keys=True))
    elif args.cmd == "refresh-loop":
        loop("refresh", args)
    else:
        parser.error("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
