from __future__ import annotations

from .common import *


def smartblog_mock_claim_file() -> str:
    return str(os.getenv("SMARTBLOG_MOCK_CLAIM_FILE", "") or "").strip()


def smartblog_mock_state_dir() -> str:
    raw = str(os.getenv("SMARTBLOG_MOCK_STATE_DIR", "") or "").strip()
    if raw:
        return os.path.abspath(raw)
    return os.path.abspath("./runtime/smartblog_mock")


def smartblog_supabase_url() -> str:
    raw = str(os.getenv("SUPABASE_URL") or "").strip()
    if not raw:
        raise RuntimeError("SUPABASE_URL is not set.")
    raw = raw.rstrip("/")
    suffix = "/functions/v1/worker-api"
    if raw.endswith(suffix):
        raw = raw[: -len(suffix)]
    return raw


def smartblog_supabase_service_role_key() -> str:
    token = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if token:
        return token
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not set.")


def smartblog_worker_api_key() -> str:
    token = str(
        os.getenv("WORKER_API_KEY")
        or os.getenv("VLOGME_WORKER_API_KEY")
        or os.getenv("VLOGME_RENDER_WORKER_TOKEN")
        or os.getenv("POSTPROCESSING_WORKER_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError("WORKER_API_KEY/VLOGME_WORKER_API_KEY is not set.")
    return token


def smartblog_worker_id() -> str:
    raw = str(
        os.getenv("VLOGME_RENDER_WORKER_ID")
        or os.getenv("SMARTBLOG_WORKER_ID")
        or os.getenv("RUNPOD_POD_ID")
        or os.getenv("HOSTNAME")
        or ""
    ).strip()
    if raw:
        return raw[:120]
    try:
        return str(os.uname().nodename or "render-worker")[:120]
    except Exception:
        return "render-worker"


def smartblog_worker_api_url() -> str:
    raw = str(
        os.getenv("VLOGME_WORKER_API_URL")
        or os.getenv("VLOGME_RENDER_WORKER_API_URL")
        or os.getenv("SMARTBLOG_WORKER_API_URL")
        or ""
    ).strip()
    if raw:
        return raw.rstrip("/")
    return f"{smartblog_supabase_url()}/functions/v1/worker-api"


def smartblog_worker_api_single_endpoint(url: str | None = None) -> bool:
    explicit = str(os.getenv("SMARTBLOG_WORKER_API_SINGLE_ENDPOINT") or "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    base = str(url or smartblog_worker_api_url()).rstrip("/")
    return "/api/public/v1/worker-api" in base


def smartblog_api_base_url() -> str:
    return smartblog_worker_api_url()


def smartblog_complete_ended_by(reason: str | None = None) -> str:
    reason_s = str(reason or "").strip().lower()
    if reason_s in {"client", "user", "viewer"}:
        return "client"
    client_markers = (
        "client_canceled",
        "client_cancelled",
        "canceled_by_client",
        "cancelled_by_client",
        "user_canceled",
        "user_cancelled",
    )
    if any(marker in reason_s for marker in client_markers):
        return "client"
    return "worker"


_SMARTBLOG_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class SmartBlogAPIRejected(RuntimeError):
    pass


def _smartblog_response_status_code(exc: Exception) -> int:
    try:
        response = getattr(exc, "response", None)
        return int(getattr(response, "status_code", 0) or 0)
    except Exception:
        return 0


def smartblog_is_transient_api_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status = _smartblog_response_status_code(exc)
        return bool(status in _SMARTBLOG_TRANSIENT_STATUS_CODES or status >= 500)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    status = _smartblog_response_status_code(exc)
    if status:
        return bool(status in _SMARTBLOG_TRANSIENT_STATUS_CODES or status >= 500)

    # Uploads use requests in a worker thread. Avoid importing requests here so
    # this helper stays cheap for environments that only exercise the API client.
    cls_name = exc.__class__.__name__.lower()
    module_name = exc.__class__.__module__.lower()
    if "requests" in module_name and any(
        token in cls_name
        for token in (
            "connectionerror",
            "connecttimeout",
            "readtimeout",
            "timeout",
            "proxyerror",
            "sslerror",
        )
    ):
        return True
    return False


def smartblog_api_rejection_reason(resp: dict[str, Any] | None, *, default: str) -> str:
    if not isinstance(resp, dict):
        return str(default or "SmartBlog API rejected response")
    return (
        str(resp.get("reason") or "").strip()
        or str(resp.get("error") or "").strip()
        or str(resp.get("message") or "").strip()
        or str(default or "SmartBlog API rejected response")
    )


def smartblog_validate_action_response(resp: dict[str, Any] | None, *, action: str) -> None:
    if not isinstance(resp, dict):
        return
    if resp.get("success") is not False and resp.get("ok") is not False:
        return
    action_s = str(action or "action").strip() or "action"
    raise SmartBlogAPIRejected(
        smartblog_api_rejection_reason(resp, default=f"SmartBlog {action_s} rejected by API")
    )


def smartblog_validate_complete_response(resp: dict[str, Any] | None) -> None:
    smartblog_validate_action_response(resp, action="complete")


def _smartblog_retry_after_seconds(resp: httpx.Response | None, *, fallback: float) -> float:
    if resp is None:
        return float(fallback)
    raw = str(resp.headers.get("retry-after", "") or "").strip()
    if raw:
        try:
            return max(0.25, min(30.0, float(raw)))
        except Exception:
            pass
    return float(fallback)


class SmartBlogClient:
    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        token = str(api_key or "").strip()
        if not token:
            token = smartblog_worker_api_key()
        self._base_url = str(base_url or smartblog_api_base_url()).rstrip("/")
        self._single_endpoint = bool(smartblog_worker_api_single_endpoint(self._base_url))
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def call_action(self, body: dict[str, Any], *, max_retries: int = 3) -> dict[str, Any]:
        payload = dict(body or {})
        if not str(payload.get("action") or "").strip():
            raise RuntimeError("SmartBlog API action is required")
        last_err: Exception | None = None
        attempts = max(1, int(max_retries))
        for attempt in range(attempts):
            try:
                resp = await self._client.post(self._base_url, json=payload)
                if resp.status_code in _SMARTBLOG_TRANSIENT_STATUS_CODES or resp.status_code >= 500:
                    last_err = httpx.HTTPStatusError(
                        f"SmartBlog API transient HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    if attempt >= attempts - 1:
                        raise last_err
                    await asyncio.sleep(
                        _smartblog_retry_after_seconds(resp, fallback=min(8.0, float(2**attempt)))
                    )
                    continue
                resp.raise_for_status()
                obj = resp.json()
                if not isinstance(obj, dict):
                    raise RuntimeError("SmartBlog API returned non-object response")
                return dict(obj)
            except Exception as e:
                if not smartblog_is_transient_api_error(e):
                    raise
                last_err = e
                if attempt >= attempts - 1:
                    break
                await asyncio.sleep(min(8.0, float(2**attempt)))
                continue
        if last_err is not None:
            raise last_err
        raise RuntimeError("SmartBlog API request failed after retries")

    async def poll(self, *, job_type: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        body: dict[str, Any] = {"action": "poll", "worker_id": smartblog_worker_id()}
        job_type_s = str(job_type or "").strip()
        if job_type_s:
            params["job_type"] = job_type_s
            body["job_type"] = job_type_s
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                if self._single_endpoint:
                    resp = await self._client.post(self._base_url, json=body)
                else:
                    resp = await self._client.post(f"{self._base_url}/poll", params=params, json=body)
                if resp.status_code == 429:
                    await asyncio.sleep(_smartblog_retry_after_seconds(resp, fallback=min(8.0, float(2**attempt))))
                    continue
                resp.raise_for_status()
                obj = resp.json()
                if isinstance(obj, list):
                    return [dict(x) for x in obj if isinstance(x, dict)]
                if not isinstance(obj, dict):
                    raise RuntimeError("SmartBlog API poll returned non-object response")
                jobs = obj.get("jobs")
                if not isinstance(jobs, list):
                    return []
                return [dict(x) for x in jobs if isinstance(x, dict)]
            except Exception as e:
                if not smartblog_is_transient_api_error(e):
                    raise
                last_err = e
                if attempt >= 2:
                    break
                await asyncio.sleep(min(8.0, float(2**attempt)))
        if last_err is not None:
            raise last_err
        return []

    async def claim(self, *, job_id: str) -> dict[str, Any]:
        return await self.call_action(
            {
                "action": "claim",
                "job_id": str(job_id),
                "worker_id": smartblog_worker_id(),
            }
        )

    async def progress(
        self,
        *,
        job_id: str,
        progress: int,
        stage: str,
        stage_label: str | None = None,
        stage_index: int | None = None,
        stage_total: int | None = None,
    ) -> dict[str, Any]:
        stage_s = str(stage or "").strip()
        if not stage_s:
            raise RuntimeError("SmartBlog progress stage is required")
        body: dict[str, Any] = {
            "action": "progress",
            "job_id": str(job_id),
            "progress": int(min(100, max(0, int(progress)))),
            "stage": stage_s,
        }
        if stage_label is not None and str(stage_label or "").strip():
            body["stage_label"] = str(stage_label).strip()
        if stage_index is not None:
            body["stage_index"] = int(max(1, int(stage_index)))
        if stage_total is not None:
            body["stage_total"] = int(max(1, int(stage_total)))
        return await self.call_action(body)

    async def complete(
        self,
        *,
        job_id: str,
        summary: str | None = None,
        messages_count: int | None = None,
        replies_count: int | None = None,
        video_url: str | None = None,
        storage_path: str | None = None,
        poster_url: str | None = None,
        poster_storage_path: str | None = None,
        reason: str | None = None,
        ended_by: str | None = None,
        duration_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"action": "complete", "job_id": str(job_id)}
        if video_url:
            body["video_url"] = str(video_url)
        if storage_path:
            body["storage_path"] = str(storage_path)
        if poster_url:
            body["poster_url"] = str(poster_url)
        if poster_storage_path:
            body["poster_storage_path"] = str(poster_storage_path)
        ended_by_s = str(ended_by or "").strip()
        if not ended_by_s and reason:
            ended_by_s = smartblog_complete_ended_by(reason)
        if ended_by_s:
            body["ended_by"] = smartblog_complete_ended_by(ended_by_s)
        if summary:
            body["summary"] = str(summary)
        if messages_count is not None:
            body["messages_count"] = int(max(0, int(messages_count)))
        if replies_count is not None:
            body["replies_count"] = int(max(0, int(replies_count)))
        if duration_seconds is not None:
            body["duration_seconds"] = int(max(0, round(float(duration_seconds))))
        return await self.call_action(body)

    async def fail(self, *, job_id: str, error_text: str) -> dict[str, Any]:
        body = {
            "action": "fail",
            "job_id": str(job_id),
            "error_text": str(error_text or "").strip() or "Worker failed",
        }
        return await self.call_action(body)

    async def release(self, *, job_id: str) -> dict[str, Any]:
        return await self.call_action({"action": "release", "job_id": str(job_id)})

    async def get_upload_url(
        self,
        *,
        filename: str,
        folder: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "action": "get_upload_url",
            "filename": str(filename or "").strip(),
            "folder": str(folder or "").strip(),
        }
        if content_type:
            body["content_type"] = str(content_type).strip()
        return await self.call_action(body)

    async def get_download_url(self, *, path: str) -> dict[str, Any]:
        return await self.call_action(
            {
                "action": "get_download_url",
                "path": str(path or "").strip(),
            }
        )

    async def log_event(
        self,
        *,
        severity: str,
        message: str,
        source: str | None = None,
        category: str | None = None,
        stack: str | None = None,
        context_json: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        job_id: str | None = None,
        fingerprint: str | None = None,
    ) -> dict[str, Any]:
        severity_s = str(severity or "").strip().lower()
        if severity_s not in {"error", "warn", "info"}:
            raise RuntimeError(f"unsupported SmartBlog log_event severity: {severity!r}")
        message_s = str(message or "").strip()
        if not message_s:
            raise RuntimeError("SmartBlog log_event message is required")
        body: dict[str, Any] = {
            "action": "log_event",
            "severity": severity_s,
            "message": message_s,
        }
        if str(source or "").strip():
            body["source"] = str(source).strip()
        if str(category or "").strip():
            body["category"] = str(category).strip()
        if str(stack or "").strip():
            body["stack"] = str(stack).strip()
        if isinstance(context_json, dict) and context_json:
            body["context_json"] = dict(context_json)
        if str(workspace_id or "").strip():
            body["workspace_id"] = str(workspace_id).strip()
        if str(job_id or "").strip():
            body["job_id"] = str(job_id).strip()
        if str(fingerprint or "").strip():
            body["fingerprint"] = str(fingerprint).strip()
        return await self.call_action(body)


class LocalSmartBlogMockClient:
    def __init__(self, claim_file: str, *, state_dir: str | None = None) -> None:
        claim_path = os.path.abspath(str(claim_file or "").strip())
        if not claim_path:
            raise RuntimeError("SMARTBLOG_MOCK_CLAIM_FILE is not set.")
        self._claim_file = claim_path
        self._state_dir = os.path.abspath(str(state_dir or smartblog_mock_state_dir()).strip())
        self._active_claim: dict[str, Any] | None = None

    async def aclose(self) -> None:
        return None

    def _normalize_claim(self, obj: dict[str, Any]) -> dict[str, Any]:
        claim = dict((obj.get("claim") if isinstance(obj.get("claim"), dict) else obj) or {})
        job = dict((claim.get("job") if isinstance(claim.get("job"), dict) else {}) or {})
        live_session = dict((claim.get("live_session") if isinstance(claim.get("live_session"), dict) else {}) or {})
        claim["job"] = job
        claim["live_session"] = live_session
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            job_id = f"mock_job_{int(time.time() * 1000)}"
            job["id"] = job_id
        job.setdefault("job_type", "render_video")
        if not str(live_session.get("id") or "").strip():
            live_session["id"] = f"mock_live_{job_id}"
        return claim

    def _load_claim(self) -> dict[str, Any]:
        with open(self._claim_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise RuntimeError("SMARTBLOG_MOCK_CLAIM_FILE must contain a JSON object")
        return self._normalize_claim(obj)

    def _write_state_json(self, name: str, payload: dict[str, Any]) -> None:
        os.makedirs(self._state_dir, exist_ok=True)
        out_path = os.path.join(self._state_dir, str(name))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    async def poll(self, *, job_type: str | None = None) -> list[dict[str, Any]]:
        if not os.path.exists(self._claim_file):
            return []
        claim = self._load_claim()
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        job_type_s = str(job_type or "").strip()
        if job_type_s and str(job.get("job_type") or "").strip() != job_type_s:
            return []
        return [dict(job)]

    async def claim(self, *, job_id: str) -> dict[str, Any]:
        if not os.path.exists(self._claim_file):
            return {"claimed": False}
        claim = self._load_claim()
        job = claim.get("job") if isinstance(claim.get("job"), dict) else {}
        actual_job_id = str(job.get("id") or "").strip()
        if str(job_id or "").strip() and str(job_id or "").strip() != actual_job_id:
            return {"claimed": False}
        os.makedirs(self._state_dir, exist_ok=True)
        consumed_path = os.path.join(self._state_dir, f"{actual_job_id}.claim.json")
        shutil.move(self._claim_file, consumed_path)
        claim["claimed"] = True
        claim["_mock_claim_path"] = consumed_path
        self._active_claim = dict(claim)
        self._write_state_json(
            f"{actual_job_id}.claim_ack.json",
            {"claimed": True, "job_id": actual_job_id, "claimed_at_unix_ms": int(time.time() * 1000)},
        )
        return claim

    async def progress(
        self,
        *,
        job_id: str,
        progress: int,
        stage: str,
        stage_label: str | None = None,
        stage_index: int | None = None,
        stage_total: int | None = None,
    ) -> dict[str, Any]:
        stage_s = str(stage or "").strip()
        if not stage_s:
            raise RuntimeError("SmartBlog progress stage is required")
        payload = {
            "ok": True,
            "job_id": str(job_id),
            "progress": int(min(100, max(0, int(progress)))),
            "stage": stage_s,
            "stage_label": str(stage_label or "").strip(),
            "stage_index": None if stage_index is None else int(max(1, int(stage_index))),
            "stage_total": None if stage_total is None else int(max(1, int(stage_total))),
            "updated_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"{job_id}.progress.json", payload)
        return payload

    async def complete(
        self,
        *,
        job_id: str,
        summary: str | None = None,
        messages_count: int | None = None,
        replies_count: int | None = None,
        video_url: str | None = None,
        storage_path: str | None = None,
        poster_url: str | None = None,
        poster_storage_path: str | None = None,
        reason: str | None = None,
        ended_by: str | None = None,
        duration_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        ended_by_s = str(ended_by or "").strip()
        if not ended_by_s and reason:
            ended_by_s = smartblog_complete_ended_by(reason)
        payload: dict[str, Any] = {
            "ok": True,
            "job_id": str(job_id),
            "summary": str(summary or ""),
            "messages_count": None if messages_count is None else int(max(0, int(messages_count))),
            "replies_count": None if replies_count is None else int(max(0, int(replies_count))),
            "duration_seconds": None
            if duration_seconds is None
            else int(max(0, round(float(duration_seconds)))),
            "video_url": str(video_url or ""),
            "storage_path": str(storage_path or ""),
            "poster_url": str(poster_url or ""),
            "poster_storage_path": str(poster_storage_path or ""),
            "ended_by": smartblog_complete_ended_by(ended_by_s) if ended_by_s else "",
            "completed_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"{job_id}.complete.json", payload)
        return payload

    async def fail(self, *, job_id: str, error_text: str) -> dict[str, Any]:
        payload = {
            "ok": True,
            "job_id": str(job_id),
            "error_text": str(error_text or "unknown error"),
            "failed_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"{job_id}.fail.json", payload)
        return payload

    async def release(self, *, job_id: str) -> dict[str, Any]:
        payload = {
            "ok": True,
            "job_id": str(job_id),
            "released_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"{job_id}.release.json", payload)
        return payload

    async def get_upload_url(
        self,
        *,
        filename: str,
        folder: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        filename_s = os.path.basename(str(filename or "").strip()) or f"upload_{int(time.time() * 1000)}.bin"
        folder_parts = [part.strip() for part in str(folder or "").replace("\\", "/").split("/") if part.strip()]
        folder_s = "/".join(folder_parts)
        path = f"worker-uploads/{folder_s}/{filename_s}" if folder_s else f"worker-uploads/{filename_s}"
        payload = {
            "ok": True,
            "signed_url": f"https://mock.smartblog.local/storage/v1/object/upload/sign/generated-assets/{path}?token=mock",
            "public_url": f"https://mock.smartblog.local/storage/v1/object/sign/generated-assets/{path}?token=mock",
            "path": path,
            "created_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"upload_url_{int(time.time() * 1000)}.json", payload)
        return payload

    async def get_download_url(self, *, path: str) -> dict[str, Any]:
        path_s = str(path or "").replace("\\", "/").strip("/")
        payload = {
            "ok": True,
            "success": True,
            "signed_url": f"https://mock.smartblog.local/storage/v1/object/sign/{path_s}?token=mock",
            "download_url": f"https://mock.smartblog.local/storage/v1/object/sign/{path_s}?token=mock",
            "url": f"https://mock.smartblog.local/storage/v1/object/sign/{path_s}?token=mock",
            "path": path_s,
            "storage_path": path_s,
            "created_at_unix_ms": int(time.time() * 1000),
        }
        self._write_state_json(f"download_url_{int(time.time() * 1000)}.json", payload)
        return payload

    async def log_event(
        self,
        *,
        severity: str,
        message: str,
        source: str | None = None,
        category: str | None = None,
        stack: str | None = None,
        context_json: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        job_id: str | None = None,
        fingerprint: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "ok": True,
            "severity": str(severity or "").strip().lower(),
            "message": str(message or "").strip(),
            "source": str(source or "").strip() or "worker:external",
            "category": str(category or "").strip(),
            "stack": str(stack or "").strip(),
            "context_json": dict(context_json or {}),
            "workspace_id": str(workspace_id or "").strip(),
            "job_id": str(job_id or "").strip(),
            "fingerprint": str(fingerprint or "").strip(),
            "logged_at_unix_ms": int(time.time() * 1000),
        }
        target = str(job_id or workspace_id or "worker").strip() or "worker"
        self._write_state_json(f"{target}.log_event.json", payload)
        return {"ok": True}
